#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""scriptconfig-based CLI for vllm-stack.

Each subcommand is a ``scfg.DataConfig`` subclass; ``ManageCLI`` composes
them into a single ``scfg.ModalCLI`` exposed as the ``vllm-stack`` entry
point. Because every subcommand is a ``DataConfig``, the same class can
be invoked from the shell (``vllm-stack render --profile X``) or from
Python (``RenderCLI.main(argv=False, profile='X')``).

Layout:
* Path / config helpers (config_path, generated_dir, runtime_*).
* Override resolution (apply_config_overrides, has_runtime_overrides,
  effective_allow_unsupported, effective_inventory, config_for_runtime).
* Planning helpers (build_plan, save_plan, ensure_renderable,
  render_is_stale).
* Backend-specific helpers (_compose_base_cmd, _kubeai_stub,
  _compose_up_with_router_recreate, _maybe_rerender).
* DataConfig mixins for override flags shared across subcommands.
* Per-command DataConfig classes, grouped by topic.
* ``ManageCLI`` ModalCLI and ``main`` entry point.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import requests
import scriptconfig as scfg

from .catalog import PROFILE_NAME_ALIASES, profile_summary
from .benchmark import run_benchmark
from .config import (
    CONFIG_FILE,
    MODELS_FILE,
    default_output_config,
    generated_dir_for_config,
    initial_config,
    kubeai_generated_dir_for_config,
    kubeai_local_values_path,
    load_kubeai_resource_profiles,
    load_yaml,
    normalized_catalogs,
    normalized_state,
    plan_path_for_config,
    save_kubeai_resource_profiles,
    save_yaml,
)
from .contracts import load_profile_contract
from .docker_utils import (
    PortInUseError,
    check_ports_available,
    compose_down,
    compose_recreate_router,
    compose_up,
    docker_rm_dirs,
    our_published_ports,
)
from .env_utils import parse_env_file
from .exporters import export_benchmark_bundle
from .hardware import detect_inventory, simulate_inventory
from .kubeai_ops import CommandError, deploy_rendered_artifacts, print_status as kubeai_print_status
from .paths import (
    CONFIG_DIR_ENV,
    DATA_DIR_ENV,
    config_root,
    data_root,
    set_config_root,
    set_data_root,
)
from .profile_runtime import default_base_url
from .renderer import render_from_lock
from .resolver import resolve
from .validator import validate_resolved
from .verification import verify_profile


# ---------------------------------------------------------------------------
# Path / config helpers
# ---------------------------------------------------------------------------


def config_path() -> Path:
    return config_root() / CONFIG_FILE


def models_path() -> Path:
    return config_root() / MODELS_FILE


def generated_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg if cfg is not None else _safe_load_config()
    return generated_dir_for_config(cfg)


def kubeai_generated_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg if cfg is not None else _safe_load_config()
    return kubeai_generated_dir_for_config(cfg)


def plan_path(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg if cfg is not None else _safe_load_config()
    return plan_path_for_config(cfg)


def _safe_load_config() -> dict[str, Any]:
    """Load config.yaml if present; otherwise return defaults."""
    path = config_path()
    if path.exists():
        return load_yaml(path)
    return initial_config()


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        raise SystemExit(
            f"No config.yaml found at {path}. Run "
            "`vllm-stack setup --backend compose --profile qwen2-5-7b-instruct-turbo-default` first, "
            f"or point ${CONFIG_DIR_ENV} / --config-dir at an existing config."
        )
    return load_yaml(path)


def runtime_dir_for_config(cfg: dict[str, Any]) -> Path:
    state = cfg.get("state", {})
    runtime = state.get("runtime")
    if not runtime:
        return data_root() / "runtime"
    p = Path(runtime)
    if p.is_absolute():
        return p
    return data_root() / p


def runtime_env_path(cfg: dict[str, Any]) -> Path:
    return generated_dir(cfg) / ".env"


def runtime_litellm_config_path(cfg: dict[str, Any]) -> Path:
    return runtime_dir_for_config(cfg) / "litellm_config.yaml"


def backend_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("backend", "compose")).lower()


# ---------------------------------------------------------------------------
# Env-var / override resolution
# ---------------------------------------------------------------------------


def _env_text(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    text = value.strip()
    return text or None


def _env_bool(name: str) -> bool | None:
    value = _env_text(name)
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on", "enabled"}:
        return True
    if lowered in {"0", "false", "no", "off", "disabled"}:
        return False
    raise SystemExit(f"Invalid boolean value for {name}: {value!r}")


def _env_int(name: str) -> int | None:
    value = _env_text(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as ex:
        raise SystemExit(f"Invalid integer value for {name}: {value!r}") from ex


def _as_mapping(args: Any) -> dict[str, Any]:
    """Coerce a CLI args object into a plain dict.

    Works for ``None``, ``argparse.Namespace``, and ``scfg.DataConfig``
    instances. Used to side-step name clashes between user-declared fields
    and ``DataConfig`` builtins (e.g. ``namespace`` is a property on the
    base class; ``getattr(cfg, 'namespace')`` returns the property, not the
    field value, while ``asdict()['namespace']`` returns the field value).
    """
    if args is None:
        return {}
    if hasattr(args, "asdict"):
        return dict(args.asdict())
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    return dict(args)


def _arg_or_env(args_dict: dict[str, Any], attr: str, env_name: str, *, caster=None):
    """Look up ``attr`` in the args dict, falling back to env var ``env_name``."""
    value = args_dict.get(attr)
    if value is not None:
        return value
    env_value = _env_text(env_name)
    if env_value is None:
        return None
    if caster is None:
        return env_value
    try:
        return caster(env_value)
    except ValueError as ex:
        raise SystemExit(f"Invalid value for {env_name}: {env_value!r}") from ex


def _configured_state_paths(state_root: str) -> dict[str, str]:
    base = Path(state_root)
    return {
        "hf_cache": str(base / "hf-cache"),
        "vllm_cache": str(base / "vllm-cache"),
        "open_webui": str(base / "open-webui"),
        "postgres_open_webui": str(base / "postgres-open-webui"),
        "postgres_litellm": str(base / "postgres-litellm"),
        "runtime": str(base / "runtime"),
    }


def apply_config_overrides(cfg: dict[str, Any], args: Any | None) -> dict[str, Any]:
    """Merge runtime overrides (CLI args + env vars) on top of ``cfg``.

    ``args`` may be an ``argparse.Namespace``, a ``scfg.DataConfig`` instance,
    or any mapping; it is coerced to a plain dict via ``_as_mapping``.
    """
    if args is None:
        return deepcopy(cfg)
    overrides = _as_mapping(args)
    out = deepcopy(cfg)
    out.setdefault("runtime", {})
    out.setdefault("ports", {})
    out.setdefault("state", {})
    out.setdefault("output", {})
    out.setdefault("cluster", {})
    out["cluster"].setdefault("ingress", {})

    backend = _arg_or_env(overrides, "backend", "VLLM_SERVICE_BACKEND")
    if backend:
        out["backend"] = backend

    profile = _arg_or_env(overrides, "profile", "VLLM_SERVICE_PROFILE")
    if profile:
        out["active_profile"] = profile

    compose_cmd = _arg_or_env(overrides, "compose_cmd", "VLLM_SERVICE_COMPOSE_CMD")
    if compose_cmd:
        out["runtime"]["compose_cmd"] = compose_cmd

    litellm_port = overrides.get("litellm_port")
    if litellm_port is None:
        litellm_port = _env_int("VLLM_SERVICE_LITELLM_PORT")
    if litellm_port is not None:
        out["ports"]["litellm"] = litellm_port

    open_webui_port = overrides.get("open_webui_port")
    if open_webui_port is None:
        open_webui_port = _env_int("VLLM_SERVICE_OPEN_WEBUI_PORT")
    if open_webui_port is not None:
        out["ports"]["open_webui"] = open_webui_port

    postgres_port = overrides.get("postgres_port")
    if postgres_port is None:
        postgres_port = _env_int("VLLM_SERVICE_POSTGRES_PORT")
    if postgres_port is not None:
        out["ports"]["postgres"] = postgres_port

    state_root = _arg_or_env(overrides, "state_root", "VLLM_SERVICE_STATE_ROOT")
    if state_root:
        out["state"].update(_configured_state_paths(state_root))

    runtime_dir = _arg_or_env(overrides, "runtime_dir", "VLLM_SERVICE_RUNTIME_DIR")
    if runtime_dir:
        out["state"]["runtime"] = runtime_dir

    generated_dir_override = _arg_or_env(overrides, "generated_dir", "VLLM_SERVICE_GENERATED_DIR")
    if generated_dir_override:
        out["output"]["generated_dir"] = generated_dir_override
    elif not out["output"].get("generated_dir"):
        out["output"]["generated_dir"] = default_output_config()["generated_dir"]

    namespace = _arg_or_env(overrides, "namespace", "VLLM_SERVICE_NAMESPACE")
    if namespace:
        out["cluster"]["namespace"] = namespace

    ingress_host = _arg_or_env(overrides, "ingress_host", "VLLM_SERVICE_INGRESS_HOST")
    if ingress_host:
        out["cluster"]["ingress"]["host"] = ingress_host

    ingress_enabled = overrides.get("ingress_enabled")
    if ingress_enabled is None:
        ingress_enabled = _env_bool("VLLM_SERVICE_INGRESS_ENABLED")
    if ingress_enabled is not None:
        out["cluster"]["ingress"]["enabled"] = bool(ingress_enabled)

    return out


_OVERRIDE_ATTRS = (
    "profile",
    "backend",
    "compose_cmd",
    "litellm_port",
    "open_webui_port",
    "postgres_port",
    "namespace",
    "ingress_host",
    "ingress_enabled",
    "simulate_hardware",
    "allowed_gpus",
    "generated_dir",
)

_OVERRIDE_ENVS = (
    "VLLM_SERVICE_BACKEND",
    "VLLM_SERVICE_PROFILE",
    "VLLM_SERVICE_COMPOSE_CMD",
    "VLLM_SERVICE_LITELLM_PORT",
    "VLLM_SERVICE_OPEN_WEBUI_PORT",
    "VLLM_SERVICE_POSTGRES_PORT",
    "VLLM_SERVICE_NAMESPACE",
    "VLLM_SERVICE_INGRESS_HOST",
    "VLLM_SERVICE_INGRESS_ENABLED",
    "VLLM_SERVICE_GENERATED_DIR",
    "VLLM_SERVICE_ALLOWED_GPUS",
)


def has_runtime_overrides(args: Any | None) -> bool:
    if args is None:
        return False
    overrides = _as_mapping(args)
    if any(overrides.get(attr) is not None for attr in _OVERRIDE_ATTRS):
        return True
    return any(_env_text(name) is not None for name in _OVERRIDE_ENVS)


def effective_allow_unsupported(args: Any | None, cfg: dict[str, Any]) -> bool:
    overrides = _as_mapping(args)
    arg_value = bool(overrides.get("allow_unsupported"))
    policy_value = bool(cfg.get("policy", {}).get("allow_unsupported_render", False))
    return arg_value or policy_value


def _parse_allowed_gpus(raw: Any) -> list[int] | None:
    """Parse a comma-separated list of GPU indices, or ``None`` if unset.

    Accepts ints (when the value comes from ``data=`` kwargs in the
    programmatic API), as well as strings of the form ``"1"`` or ``"1,3"``.
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [x.strip() for x in str(raw).split(",") if x.strip()]
    try:
        return [int(x) for x in items]
    except (TypeError, ValueError) as ex:
        raise SystemExit(
            f"Invalid --allowed-gpus value {raw!r}: expected a comma-separated "
            f"list of integer GPU indices (e.g. '1' or '1,3'). {ex}"
        )


def _filter_inventory_to_allowed(inventory: dict[str, Any], allowed: list[int] | None) -> dict[str, Any]:
    """Return a new inventory containing only the GPUs whose ``index`` is in ``allowed``.

    Real indices are preserved — there is no renumbering — so a profile
    that says ``placement.gpu_indices: [1, 3]`` still pins to physical
    GPUs 1 and 3 after filtering.
    """
    if not allowed:
        return inventory
    allowed_set = set(allowed)
    filtered = [g for g in inventory.get("gpus", []) if g.get("index") in allowed_set]
    return {"gpu_count": len(filtered), "gpus": filtered}


def effective_inventory(args: Any | None) -> dict[str, Any] | None:
    """Build the inventory the resolver should see, honoring CLI / env overrides.

    Returns ``None`` when nothing is constraining the inventory, so the
    resolver falls back to ``detect_inventory()`` at plan time.
    """
    overrides = _as_mapping(args)
    spec = overrides.get("simulate_hardware")
    allowed = _parse_allowed_gpus(
        overrides.get("allowed_gpus") or _env_text("VLLM_SERVICE_ALLOWED_GPUS")
    )
    if not spec and allowed is None:
        return None
    base = simulate_inventory(spec) if spec else detect_inventory()
    return _filter_inventory_to_allowed(base, allowed)


def config_for_runtime(args: Any | None, *, allow_missing: bool = False) -> dict[str, Any]:
    if config_path().exists():
        cfg = load_yaml(config_path())
    elif allow_missing:
        cfg = initial_config()
    else:
        raise SystemExit(
            f"No config.yaml found at {config_path()}. Run "
            "`vllm-stack setup --backend compose --profile qwen2-5-7b-instruct-turbo-default` first, "
            f"or point ${CONFIG_DIR_ENV} / --config-dir at an existing config."
        )
    return apply_config_overrides(cfg, args)


# ---------------------------------------------------------------------------
# Plan helpers
# ---------------------------------------------------------------------------


def build_plan(
    cfg: dict[str, Any],
    *,
    profile_name: str | None = None,
    allow_unsupported: bool = False,
    inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = resolve(cfg, inventory=inventory, profile_name=profile_name)
    report = validate_resolved(resolved)
    return {
        "schema_version": 1,
        "allow_unsupported": bool(allow_unsupported),
        "validated": report,
        "deployment": resolved,
    }


def save_plan(plan: dict[str, Any], cfg: dict[str, Any] | None = None) -> Path:
    path = plan_path(cfg)
    save_yaml(path, plan)
    return path


def ensure_renderable(plan: dict[str, Any]) -> None:
    validated = plan.get("validated", {}) or {}
    if validated.get("errors") and not plan.get("allow_unsupported", False):
        raise SystemExit(
            "Refusing to render because the resolved plan contains validation errors. "
            "Use `--allow-unsupported` to override."
        )


def render_is_stale(cfg: dict[str, Any] | None = None) -> bool:
    cfg = load_config() if cfg is None else cfg
    cfg_path = config_path()
    current_plan = plan_path(cfg)
    backend = backend_name(cfg)

    if backend == "kubeai":
        kubeai_root = kubeai_generated_dir(cfg)
        required_outputs = [
            current_plan,
            kubeai_root / "namespace.yaml",
            kubeai_root / "kubeai-values.yaml",
            kubeai_root / "models.yaml",
        ]
    else:
        required_outputs = [
            current_plan,
            generated_dir(cfg) / "docker-compose.yml",
            runtime_env_path(cfg),
            runtime_litellm_config_path(cfg),
        ]

    if any(not p.exists() for p in required_outputs):
        return True

    if cfg_path.exists():
        oldest_generated = min(p.stat().st_mtime for p in required_outputs)
        if cfg_path.stat().st_mtime > oldest_generated:
            return True
        if backend == "kubeai":
            local_values_path = kubeai_local_values_path()
            if local_values_path.exists() and local_values_path.stat().st_mtime > oldest_generated:
                return True

    if any(current_plan.stat().st_mtime > p.stat().st_mtime for p in required_outputs if p != current_plan):
        return True
    return False


# ---------------------------------------------------------------------------
# Backend-specific helpers
# ---------------------------------------------------------------------------


def _compose_base_cmd(cfg: dict[str, Any]) -> list[str]:
    """Build the shared ``docker compose -f ... --env-file ...`` prefix.

    Used by every compose-wrapper subcommand so the user doesn't have to
    cd into the rendered-artifacts directory just to run a one-shot
    ``ps`` / ``restart`` / ``pull`` / ``logs``.
    """
    compose_file = generated_dir(cfg) / "docker-compose.yml"
    env_file = generated_dir(cfg) / ".env"
    return cfg["runtime"]["compose_cmd"].split() + [
        "-f", str(compose_file),
        "--env-file", str(env_file),
    ]


def _kubeai_stub(cmd_name: str) -> None:
    """Raise for a day-2-ops subcommand that has no kubeai implementation yet.

    These wrappers (logs/ps/restart/pull/start/stop) compose docker-compose
    invocations and have no kubectl equivalent in this CLI. Until somebody
    writes one, surface the gap as ``NotImplementedError`` so callers can
    distinguish "kubeai doesn't do this yet" from a real failure.
    """
    raise NotImplementedError(
        f"`{cmd_name}` is not implemented for the kubeai backend yet. "
        f"Use the equivalent kubectl command in the meantime "
        f"(e.g. `kubectl -n <namespace> ...`)."
    )


def _compose_up_with_router_recreate(
    cfg: dict[str, Any],
    *,
    detach: bool,
) -> None:
    """Run ``compose up`` and refresh LiteLLM's model list to match the new render.

    A compose ``up`` against the rendered stack only restarts vLLM services
    whose specs changed. LiteLLM and Open WebUI keep their existing
    containers and would therefore serve stale model lists until something
    else refreshed them. Two ways to do that:

    1. **Live refresh** (preferred): talk to LiteLLM's admin API
       (``/model/new`` / ``/model/delete``) to diff and apply alias changes
       in-process. LiteLLM and Open WebUI stay up — users hitting unchanged
       models see no disruption. Skipped automatically on cold start or if
       the admin API is unreachable.
    2. **Container recreate** (fallback): force-recreate the litellm and
       open-webui containers so they reload the rendered YAML on startup.
       Brief user-visible downtime; used only when live refresh fails.
    """
    compose_file = generated_dir(cfg) / "docker-compose.yml"
    env_file = generated_dir(cfg) / ".env"

    _preflight_check_ports(cfg)

    compose_up(
        cfg["runtime"]["compose_cmd"],
        compose_file,
        env_file,
        detach=detach,
        remove_orphans=True,
    )

    # If `up` failed it would have already raised; only do the router refresh
    # in detached mode (foreground `up` keeps the user attached to logs and
    # leaves cycling decisions to compose).
    if not detach:
        return

    try:
        _litellm_refresh_router_live(cfg)
        return
    except RouterRefreshError as ex:
        print(
            f"Live router refresh skipped ({ex}); "
            "recreating litellm container to reload config from YAML "
            "(open-webui stays up)..."
        )

    compose_recreate_router(
        cfg["runtime"]["compose_cmd"],
        compose_file,
        env_file,
        detach=True,
    )


class RouterRefreshError(RuntimeError):
    """Live LiteLLM router refresh did not complete; caller should fall back."""


def _resolve_env_refs(obj: Any, env: dict[str, str]) -> Any:
    """Recursively replace ``os.environ/VAR`` strings with the value from ``env``.

    Mirrors LiteLLM's YAML-load substitution so we can feed the admin API
    literal credentials. If a referenced variable isn't in ``env`` the original
    string is left alone — caller will get the upstream error to debug.
    """
    if isinstance(obj, str):
        if obj.startswith("os.environ/"):
            var = obj.removeprefix("os.environ/")
            return env.get(var, obj)
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_env_refs(v, env) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_refs(v, env) for v in obj]
    return obj


def _litellm_refresh_router_live(cfg: dict[str, Any]) -> None:
    """Sync the running LiteLLM router's model list to match the rendered YAML.

    Diffs ``GET /model/info`` (current state in the running container) against
    the rendered ``litellm_config.yaml`` (desired state), then applies
    ``POST /model/delete`` and ``POST /model/new`` for the differences. Aliases
    that didn't change keep serving traffic without interruption.

    Raises ``RouterRefreshError`` on any failure (admin API unreachable,
    auth missing, individual call fails); the caller falls back to the
    full container-recreate path.
    """
    import yaml as _yaml

    litellm_port = cfg.get("ports", {}).get("litellm")
    if not litellm_port:
        raise RouterRefreshError("litellm port not configured in cfg")
    base = f"http://127.0.0.1:{litellm_port}"

    env = parse_env_file(runtime_env_path(cfg))
    master_key = env.get("LITELLM_MASTER_KEY", "").strip()
    if not master_key:
        raise RouterRefreshError("LITELLM_MASTER_KEY missing from rendered .env")
    headers = {"Authorization": f"Bearer {master_key}", "Content-Type": "application/json"}

    config_path_ = runtime_litellm_config_path(cfg)
    if not config_path_.exists():
        raise RouterRefreshError(f"rendered litellm config not found at {config_path_}")
    try:
        desired_doc = _yaml.safe_load(config_path_.read_text(encoding="utf-8")) or {}
    except _yaml.YAMLError as ex:
        raise RouterRefreshError(f"could not parse {config_path_}: {ex}") from ex
    desired_models = desired_doc.get("model_list") or []

    # The rendered YAML keeps secrets as `os.environ/VAR` references so the
    # file itself isn't sensitive. LiteLLM resolves these only at YAML-load
    # time on container startup — the admin API takes literal values. Inline
    # the actual env values now so /model/new gets a usable upstream.
    desired_models = [_resolve_env_refs(m, env) for m in desired_models]

    try:
        resp = requests.get(f"{base}/model/info", headers=headers, timeout=5)
        resp.raise_for_status()
    except requests.exceptions.RequestException as ex:
        raise RouterRefreshError(f"GET /model/info failed: {ex}") from ex
    current_models = resp.json().get("data") or []

    # Key by alias (model_name). Within an alias, the "upstream" identity is
    # litellm_params.model (e.g. "openai/qwen3.5-9b"). If that changes, the
    # alias points to a different service and must be re-added; if it matches,
    # the alias is untouched and continues serving.
    def upstream_of(entry):
        return (entry.get("litellm_params") or {}).get("model")

    current_by_alias = {m["model_name"]: m for m in current_models}
    desired_by_alias = {m["model_name"]: m for m in desired_models}

    to_delete_ids: list[str] = []
    to_add: list[dict] = []

    for alias, current in current_by_alias.items():
        desired = desired_by_alias.get(alias)
        if desired is None:
            to_delete_ids.append(current["model_info"]["id"])
        elif upstream_of(current) != upstream_of(desired):
            to_delete_ids.append(current["model_info"]["id"])
            to_add.append(desired)

    for alias, desired in desired_by_alias.items():
        if alias not in current_by_alias:
            to_add.append(desired)

    if not to_delete_ids and not to_add:
        return

    # Delete-before-add so the same alias can transition to a new upstream
    # without LiteLLM rejecting a duplicate model_name.
    for model_id in to_delete_ids:
        try:
            resp = requests.post(
                f"{base}/model/delete",
                headers=headers,
                json={"id": model_id},
                timeout=5,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as ex:
            raise RouterRefreshError(f"DELETE id={model_id} failed: {ex}") from ex

    for model in to_add:
        try:
            resp = requests.post(
                f"{base}/model/new",
                headers=headers,
                json=model,
                timeout=5,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as ex:
            alias = model.get("model_name", "<unknown>")
            raise RouterRefreshError(f"POST /model/new alias={alias} failed: {ex}") from ex

    summary_parts = []
    if to_delete_ids:
        summary_parts.append(f"removed {len(to_delete_ids)} alias(es)")
    if to_add:
        summary_parts.append(f"added {len(to_add)} alias(es)")
    print(f"Live LiteLLM router refresh: {', '.join(summary_parts)}.")


def _preflight_check_ports(cfg: dict[str, Any]) -> None:
    """Verify the host ports the rendered stack will publish are free.

    Skips ports that are already published by our own compose project (e.g.
    the user is running ``up`` or ``switch --apply`` against a stack that's
    already up — compose will detect the config change and recreate those
    containers). Only flags ports owned by something we don't control, so the
    user sees a real conflict, not a self-collision.
    """
    ports = cfg.get("ports", {})
    candidates: list[tuple[str, int, str]] = []
    litellm_port = ports.get("litellm")
    if litellm_port:
        candidates.append(("litellm", int(litellm_port), "0.0.0.0"))
    open_webui_port = ports.get("open_webui")
    if open_webui_port:
        candidates.append(("open-webui", int(open_webui_port), "0.0.0.0"))

    owned = our_published_ports(
        cfg["runtime"]["compose_cmd"],
        generated_dir(cfg) / "docker-compose.yml",
        runtime_env_path(cfg),
    )
    to_check = [(svc, port, host) for svc, port, host in candidates if port not in owned]

    try:
        check_ports_available(to_check)
    except PortInUseError as ex:
        raise SystemExit(str(ex)) from ex


def _apply_path_overrides(config: Any) -> None:
    """Honour ``--config-dir`` / ``--data-dir`` from a parsed subcommand config."""
    overrides = _as_mapping(config)
    if overrides.get("config_dir"):
        set_config_root(overrides["config_dir"])
    if overrides.get("data_dir"):
        set_data_root(overrides["data_dir"])


def _maybe_rerender(config: Any, cfg: dict[str, Any]) -> None:
    """Re-run RenderCLI if runtime overrides changed or rendered outputs are stale.

    Both ``up`` and ``deploy`` need this so the rendered artifacts always
    match the current config + overrides before any container action.
    """
    if has_runtime_overrides(config) or render_is_stale(cfg):
        overrides = _as_mapping(config)
        RenderCLI.main(
            argv=False,
            profile=overrides.get("profile"),
            backend=overrides.get("backend"),
            compose_cmd=overrides.get("compose_cmd"),
            litellm_port=overrides.get("litellm_port"),
            open_webui_port=overrides.get("open_webui_port"),
            postgres_port=overrides.get("postgres_port"),
            namespace=overrides.get("namespace"),
            ingress_host=overrides.get("ingress_host"),
            ingress_enabled=overrides.get("ingress_enabled"),
            generated_dir=overrides.get("generated_dir"),
            allow_unsupported=bool(overrides.get("allow_unsupported")),
            simulate_hardware=overrides.get("simulate_hardware"),
            allowed_gpus=overrides.get("allowed_gpus"),
            yes=bool(overrides.get("yes")),
        )


# ---------------------------------------------------------------------------
# DataConfig mixins for common override flags
# ---------------------------------------------------------------------------


class _PathOverridesMixin(scfg.DataConfig):
    """Adds global ``--config-dir`` / ``--data-dir`` to a subcommand."""

    config_dir = scfg.Value(
        None,
        type=str,
        help=(
            f"Directory containing config.yaml / models.yaml. Defaults to "
            f"~/.config/vllm_service (XDG_CONFIG_HOME) or ${CONFIG_DIR_ENV} when set."
        ),
    )
    data_dir = scfg.Value(
        None,
        type=str,
        help=(
            f"Directory for rendered artifacts and bind-mount state. Defaults to "
            f"~/.cache/vllm_service (XDG_CACHE_HOME) or ${DATA_DIR_ENV} when set."
        ),
    )


class _BackendOverrideMixin(scfg.DataConfig):
    backend = scfg.Value(None, choices=["compose", "kubeai"], help="Active backend override.")


class _ComposeOverrideMixin(scfg.DataConfig):
    compose_cmd = scfg.Value(None, type=str, help="Docker compose command override (e.g. 'podman compose').")


class _ProfileOverrideMixin(scfg.DataConfig):
    profile = scfg.Value(None, type=str, help="Active profile override (sets config.active_profile).")


class _PortOverridesMixin(scfg.DataConfig):
    litellm_port = scfg.Value(None, type=int)
    open_webui_port = scfg.Value(None, type=int)
    postgres_port = scfg.Value(None, type=int)


class _ClusterOverridesMixin(scfg.DataConfig):
    namespace = scfg.Value(None, type=str, help="Kubernetes namespace for kubeai deployments.")
    ingress_host = scfg.Value(None, type=str, help="Ingress host (kubeai only).")
    ingress_enabled = scfg.Value(
        None,
        isflag=True,
        alias=["ingress"],
        help="Enable cluster ingress (kubeai only); use --no-ingress to disable.",
    )


class _GeneratedDirOverrideMixin(scfg.DataConfig):
    generated_dir = scfg.Value(
        None,
        type=str,
        help=(
            "Directory to write rendered artifacts (docker-compose.yml, .env, "
            "plan.yaml, kubeai/*) into. Defaults to <data-dir>/generated. May "
            "also be set via VLLM_SERVICE_GENERATED_DIR or output.generated_dir."
        ),
    )


class _AllowUnsupportedMixin(scfg.DataConfig):
    allow_unsupported = scfg.Value(False, isflag=True, help="Allow validation errors when planning/rendering.")


class _SimulateHardwareMixin(scfg.DataConfig):
    simulate_hardware = scfg.Value(
        None,
        type=str,
        help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80). Useful for planning on smaller machines.",
    )


class _AllowedGpusMixin(scfg.DataConfig):
    allowed_gpus = scfg.Value(
        None,
        type=str,
        help=(
            "Restrict placement to a comma-separated list of GPU indices "
            "(e.g. '1' or '1,3'). Real indices are preserved — the rendered "
            "compose stack pins device_ids to exactly those GPUs. May also "
            "be set via VLLM_SERVICE_ALLOWED_GPUS. Useful for integration "
            "tests on machines where some GPUs are tied up."
        ),
    )


class _PlanOverridesCLI(
    _PathOverridesMixin,
    _ProfileOverrideMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
    _GeneratedDirOverrideMixin,
    _AllowUnsupportedMixin,
    _SimulateHardwareMixin,
    _AllowedGpusMixin,
):
    """Standard set of overrides for any command that builds a plan."""

    pass


class _SwitchPathOverridesCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
    _AllowUnsupportedMixin,
    _SimulateHardwareMixin,
    _AllowedGpusMixin,
):
    """Overrides for commands that take a positional ``profile`` (no --profile)."""

    pass


# ---------------------------------------------------------------------------
# Profile / config management commands
# ---------------------------------------------------------------------------


class InitCLI(_PathOverridesMixin):
    """Write a fresh config.yaml + empty models.yaml under config_root()."""

    force = scfg.Value(False, isflag=True, help="Overwrite an existing config.yaml.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg_path = config_path()
        if cfg_path.exists() and not config.force:
            raise SystemExit("config.yaml already exists. Use --force to overwrite.")
        save_yaml(cfg_path, initial_config())
        if not models_path().exists():
            save_yaml(models_path(), {"models": {}, "profiles": {}})
        print(f"Wrote {cfg_path}")
        return 0


class SetupCLI(
    _PathOverridesMixin,
    _ProfileOverrideMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
    _GeneratedDirOverrideMixin,
):
    """Create / update config.yaml with the requested overrides applied."""

    reset = scfg.Value(False, isflag=True, help="Start from default config values before applying overrides.")
    state_root = scfg.Value(None, type=str, help="Base directory for state.* paths.")
    runtime_dir = scfg.Value(None, type=str, help="Override state.runtime specifically.")
    resource_profiles_file = scfg.Value(
        None,
        type=str,
        help="For kubeai setups, sync a local Helm values file with resourceProfiles into kubeai-values.local.yaml.",
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg_path = config_path()
        if cfg_path.exists() and not config.reset:
            cfg = load_yaml(cfg_path)
        else:
            cfg = initial_config()
        cfg = apply_config_overrides(cfg, config)
        save_yaml(cfg_path, cfg)
        if not models_path().exists():
            save_yaml(models_path(), {"models": {}, "profiles": {}})
        if config.resource_profiles_file:
            # User-supplied path: anchor on CWD so a typed relative path
            # behaves as the user expects.
            source = Path(config.resource_profiles_file)
            if not source.is_absolute():
                source = Path.cwd() / source
            values_doc = load_yaml(source)
            if "resourceProfiles" not in values_doc:
                raise SystemExit(f"{source} is missing a top-level resourceProfiles map")
            target = save_kubeai_resource_profiles(values_doc)
            if plan_path(cfg).exists():
                plan_path(cfg).unlink()
            print(f"Wrote {target}")
        print(f"Wrote {cfg_path}")
        print(
            f"Configured backend={cfg.get('backend', 'compose')} "
            f"active_profile={cfg.get('active_profile', '') or '<unset>'}"
        )
        return 0


class ResolveCLI(_PlanOverridesCLI):
    """Resolve the active profile into a deployment dict and print it."""

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        save_plan(plan, cfg)
        print(json.dumps(plan["deployment"], indent=2))
        return 0


class ValidateCLI(_PlanOverridesCLI):
    """Resolve + validate the active profile; exit non-zero on validation errors."""

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        save_plan(plan, cfg)
        print(json.dumps(plan["validated"], indent=2))
        return 0 if plan["validated"]["ok"] else 2


class LockCLI(_PlanOverridesCLI):
    """Write plan.yaml after resolving + validating the active profile."""

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        if not plan["validated"]["ok"] and not plan["allow_unsupported"]:
            raise SystemExit(
                "Refusing to write plan.yaml because validation failed. Use --allow-unsupported to override."
            )
        save_plan(plan, cfg)
        print(json.dumps(plan, indent=2))
        return 0


class RenderCLI(_PlanOverridesCLI):
    """Render the active profile's deployment into compose/kubeai artifacts."""

    yes = scfg.Value(
        False,
        isflag=True,
        short_alias=["y"],
        help="Apply rendered changes without prompting. Without this, render shows a per-file diff and asks for confirmation.",
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        ensure_renderable(plan)
        save_plan(plan, cfg)
        render_from_lock(plan, assume_yes=bool(config.yes))
        print(f"Wrote {plan_path(cfg)}")
        if backend_name(cfg) == "kubeai":
            print(f"Rendered KubeAI artifacts into {kubeai_generated_dir(cfg)}")
        else:
            print(f"Rendered Compose into {generated_dir(cfg)}")
            print(f"Rendered mounted runtime files into {runtime_dir_for_config(cfg)}")
        return 0


class SwitchCLI(_SwitchPathOverridesCLI):
    """Persist a new active_profile and re-render (optionally re-applying)."""

    profile = scfg.Value(None, type=str, position=1, help="Profile name to switch to.")
    apply = scfg.Value(False, isflag=True, help="Also apply the new profile to the running stack.")
    yes = scfg.Value(False, isflag=True, short_alias=["y"], help="Apply rendered changes without prompting.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if not config.profile:
            raise SystemExit("switch: missing required profile name")
        persisted_cfg = load_config()
        persisted_cfg["active_profile"] = config.profile
        save_yaml(config_path(), persisted_cfg)
        cfg = apply_config_overrides(persisted_cfg, config)

        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        ensure_renderable(plan)
        save_plan(plan, cfg)
        render_from_lock(plan, assume_yes=bool(config.yes))
        if config.apply:
            if backend_name(cfg) == "compose":
                _compose_up_with_router_recreate(cfg, detach=True)
            else:
                deploy_rendered_artifacts(plan["deployment"])
        print(f"Switched active_profile to {config.profile}")
        return 0


class ListModelsCLI(_PathOverridesMixin):
    """Print every model in the merged catalog."""

    __command__ = "list-models"

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = load_config() if config_path().exists() else initial_config()
        cats = normalized_catalogs(cfg)
        for name, model in cats.get("models", {}).items():
            ref = model.get("hf_model_id") or model.get("url", "")
            print(f"{name}: {ref}")
        return 0


class ListProfilesCLI(_PathOverridesMixin):
    """Print every serving profile in the merged catalog."""

    __command__ = "list-profiles"

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = load_config() if config_path().exists() else initial_config()
        cats = normalized_catalogs(cfg)
        profiles = cats.get("profiles", {})
        hidden_legacy = set(PROFILE_NAME_ALIASES)
        for name, profile in profiles.items():
            if name in hidden_legacy:
                continue
            if profile.get("kind") == "invalid-profile":
                continue
            summary = profile_summary(profile)
            print(
                f"{name}: public={summary['public_name']} logical={summary['logical_model_name']} "
                f"protocol={summary['protocol_mode']} base_model={summary['base_model']}"
            )
        return 0


class ExplainCLI(_PathOverridesMixin):
    """Pretty-print a YAML file (defaults to the current plan.yaml)."""

    __command__ = "explain"

    file = scfg.Value(None, type=str, help="Path to read (default: current plan.yaml).")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if config.file:
            target = Path(config.file)
            if not target.is_absolute():
                target = Path.cwd() / target
        else:
            target = plan_path()
        if not target.exists():
            raise SystemExit(f"Missing file: {target}")
        print(json.dumps(load_yaml(target), indent=2))
        return 0


class DescribeProfileCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _AllowUnsupportedMixin,
    _SimulateHardwareMixin,
):
    """Print the profile contract for a given profile name."""

    __command__ = "describe-profile"

    profile = scfg.Value(None, type=str, position=1, help="Profile name to describe.")
    format = scfg.Value("yaml", choices=["json", "yaml"])
    output = scfg.Value(None, type=str, help="Write to this file instead of stdout.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        overrides = _as_mapping(config)
        if not overrides.get("profile"):
            raise SystemExit("describe-profile: missing required profile name")
        contract = load_profile_contract(
            overrides["profile"],
            backend=_arg_or_env(overrides, "backend", "VLLM_SERVICE_BACKEND"),
            simulate_hardware_spec=overrides.get("simulate_hardware"),
        )
        return _print_structured(contract, overrides["format"], overrides.get("output"))


def _print_structured(data: dict[str, Any], fmt: str, output: str | None) -> int:
    if fmt == "yaml":
        import yaml

        text = yaml.safe_dump(data, sort_keys=False)
    else:
        text = json.dumps(data, indent=2)
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
        print(f"Wrote {target}")
        return 0
    print(text)
    return 0


class VerifyProfileCLI(_SwitchPathOverridesCLI):
    """Sanity-check a resolved profile (post-render expectations)."""

    __command__ = "verify-profile"

    profile = scfg.Value(None, type=str, position=1, help="Profile name to verify.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if not config.profile:
            raise SystemExit("verify-profile: missing required profile name")
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        result = verify_profile(plan["deployment"])
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 2


class KubeaiSyncResourceProfilesCLI(_PathOverridesMixin):
    """Pull a Helm ``resourceProfiles`` values file into kubeai-values.local.yaml."""

    __command__ = "kubeai-sync-resource-profiles"

    from_file = scfg.Value(None, type=str, required=True, help="Helm values file with a top-level resourceProfiles map.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config, allow_missing=True)
        # User-supplied path: anchor on CWD.
        source = Path(config.from_file)
        if not source.is_absolute():
            source = Path.cwd() / source
        values_doc = load_yaml(source)
        if "resourceProfiles" not in values_doc:
            raise SystemExit(f"{source} is missing a top-level resourceProfiles map")
        target = save_kubeai_resource_profiles(values_doc)
        if plan_path(cfg).exists():
            plan_path(cfg).unlink()
        profiles, _, _ = load_kubeai_resource_profiles()
        print(f"Wrote {target}")
        print(f"Synced {len(profiles)} KubeAI resource profile(s)")
        return 0


# ---------------------------------------------------------------------------
# Export / bundle commands (transitional; helm_audit owns the canonical path)
# ---------------------------------------------------------------------------


class _ExportBundleCLI(_SwitchPathOverridesCLI):
    """Shared body for ``export-benchmark-bundle`` / ``export-helm-bundle``."""

    profile = scfg.Value(None, type=str, position=1, help="Profile name to export.")
    base_url = scfg.Value(None, type=str)
    output_dir = scfg.Value(None, type=str)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if not config.profile:
            raise SystemExit(f"{cls.__command__}: missing required profile name")
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        ensure_renderable(plan)
        print(
            "Benchmark bundle export here is transitional; prefer the helm_audit "
            "integration layer for CRFM HELM bundle generation."
        )
        output_dir = None
        if config.output_dir:
            output_dir = Path(config.output_dir)
            if not output_dir.is_absolute():
                output_dir = Path.cwd() / output_dir
        result = export_benchmark_bundle(
            plan["deployment"],
            base_url=config.base_url,
            output_dir=output_dir,
        )
        print(f"Wrote {result['bundle_path']}")
        print(f"Wrote {result['model_deployments_path']}")
        return 0


class ExportBenchmarkBundleCLI(_ExportBundleCLI):
    """Export a benchmark bundle (transitional; helm_audit owns the canonical path)."""

    __command__ = "export-benchmark-bundle"


class ExportHelmBundleCLI(_ExportBundleCLI):
    """Alias of export-benchmark-bundle that emits the legacy helm/ layout."""

    __command__ = "export-helm-bundle"


# ---------------------------------------------------------------------------
# Runtime commands
# ---------------------------------------------------------------------------


class UpCLI(_PlanOverridesCLI):
    """Bring the rendered compose stack up. Re-renders first if anything changed."""

    detach = scfg.Value(False, isflag=True, short_alias=["d"], help="Run in background instead of attaching to logs.")
    yes = scfg.Value(False, isflag=True, short_alias=["y"], help="If `up` triggers a re-render, apply changes without prompting.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            raise SystemExit("`up` only supports the compose backend. Use `deploy` for kubeai.")
        _maybe_rerender(config, cfg)
        _compose_up_with_router_recreate(cfg, detach=bool(config.detach))
        return 0


class DownCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
):
    """Bring the rendered compose stack down (does not touch volumes)."""

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            raise SystemExit("`down` only supports the compose backend.")
        compose_down(
            cfg["runtime"]["compose_cmd"],
            generated_dir(cfg) / "docker-compose.yml",
            runtime_env_path(cfg),
        )
        return 0


class PurgeCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
):
    """Stop the stack and delete all Docker-written state directories.

    Uses a temporary Alpine container to remove directories that Docker wrote
    as root, avoiding ``Permission denied`` errors from plain ``rm -rf``.
    """

    yes = scfg.Value(False, isflag=True, short_alias=["y"], help="Skip confirmation prompt.")
    delete_cache = scfg.Value(
        False, isflag=True,
        help="Also delete hf-cache and vllm-cache (model weights). By default those are preserved.",
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config, allow_missing=True)

        state = normalized_state(cfg.get("state", {}))
        always_delete = ["postgres_litellm", "postgres_open_webui", "open_webui", "runtime"]
        model_dirs = ["hf_cache", "vllm_cache"]
        keys = always_delete + model_dirs if config.delete_cache else always_delete
        dirs_to_delete = [Path(state[k]) for k in keys if Path(state[k]).exists()]

        if not dirs_to_delete:
            print("Nothing to purge — state directories do not exist.")
            return 0

        if not config.yes:
            print("The following directories will be deleted (via Docker to handle root-owned files):")
            for d in dirs_to_delete:
                print(f"  {d}")
            answer = input("Proceed? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                print("Aborted.")
                return 1

        compose_file = generated_dir(cfg) / "docker-compose.yml"
        if compose_file.exists() and backend_name(cfg) == "compose":
            try:
                compose_down(
                    cfg["runtime"]["compose_cmd"],
                    compose_file,
                    runtime_env_path(cfg),
                )
            except Exception as ex:
                print(f"Warning: compose down failed (containers may already be stopped): {ex}")

        compose_cmd = cfg.get("runtime", {}).get("compose_cmd", "docker compose")
        docker_cmd = compose_cmd.split()[0]
        docker_rm_dirs(dirs_to_delete, docker_cmd=docker_cmd)
        print("Purge complete.")
        return 0


class DeployCLI(_PlanOverridesCLI):
    """Apply the rendered stack to its backend (kubeai apply / compose up)."""

    detach = scfg.Value(False, isflag=True, short_alias=["d"])
    yes = scfg.Value(False, isflag=True, short_alias=["y"], help="If `deploy` triggers a re-render, apply changes without prompting.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        _maybe_rerender(config, cfg)
        if backend_name(cfg) == "kubeai":
            plan = load_yaml(plan_path(cfg))
            try:
                deploy_rendered_artifacts(plan["deployment"])
            except CommandError as ex:
                namespace = cfg.get("cluster", {}).get("namespace", "kubeai")
                raise SystemExit(
                    f"Failed to deploy to namespace {namespace!r}. Confirm "
                    f"`vllm-stack setup --backend kubeai --namespace {namespace}` "
                    "matches the namespace where the KubeAI Helm release is installed.\n"
                    f"Original error: {ex}"
                ) from ex
            return 0
        compose_up(
            cfg["runtime"]["compose_cmd"],
            generated_dir(cfg) / "docker-compose.yml",
            generated_dir(cfg) / ".env",
            detach=bool(config.detach),
            remove_orphans=True,
        )
        return 0


class EnvCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
):
    """Inspect the rendered .env file (path, single value, or eval-friendly export)."""

    key = scfg.Value(None, type=str, position=1, help="Print only this variable's value. Empty = all.")
    export = scfg.Value(False, isflag=True, help="Print `export KEY=value` lines suitable for `eval`.")
    path = scfg.Value(False, isflag=True, help="Print only the absolute path to .env (default if no flags).")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            raise SystemExit("`env` only applies to the compose backend.")
        env_file = runtime_env_path(cfg)
        if not env_file.exists():
            raise SystemExit(
                f"No .env at {env_file}. Run `vllm-stack render` first."
            )
        # Default with no flags and no key: print the path so users can do
        #   `source $(vllm-stack env)` (with `set -a` if they want export semantics).
        if config.key is None and not config.export:
            print(env_file)
            return 0
        env = parse_env_file(env_file)
        if config.key:
            if config.key not in env:
                raise SystemExit(f"{config.key!r} not found in {env_file}")
            print(env[config.key])
            return 0
        # --export: emit eval-friendly export lines.
        for k, v in env.items():
            print(f"export {k}={shlex.quote(v)}")
        return 0


class StatusCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
):
    """Show the runtime status of the rendered stack."""

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) == "kubeai":
            namespace = cfg.get("cluster", {}).get("namespace", "kubeai")
            try:
                kubeai_print_status(namespace)
            except CommandError as ex:
                raise SystemExit(
                    f"Failed to query KubeAI resources in namespace {namespace!r}. Confirm "
                    f"`vllm-stack setup --backend kubeai --namespace {namespace}` "
                    "matches the namespace where the KubeAI Helm release is installed.\n"
                    f"Original error: {ex}"
                ) from ex
            return 0
        proc = subprocess.run(_compose_base_cmd(cfg) + ["ps"])
        return int(proc.returncode)


# ---------------------------------------------------------------------------
# Compose day-2-ops wrappers (raise NotImplementedError on kubeai)
# ---------------------------------------------------------------------------


class _ComposeWrapperBase(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
):
    """Common fields for ``docker compose <subcmd>`` wrappers."""

    services = scfg.Value(None, nargs="*", position=1, help="Optional service names to filter (empty = all).")


class LogsCLI(_ComposeWrapperBase):
    """Tail rendered Compose service logs without typing the full docker compose path."""

    follow = scfg.Value(False, isflag=True, short_alias=["f"], help="Stream logs (docker compose logs -f).")
    tail = scfg.Value(None, type=str, help="Tail the last N lines (default: all). Pass a number or 'all'.")
    timestamps = scfg.Value(False, isflag=True)
    no_color = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            _kubeai_stub("logs")
        cmd = _compose_base_cmd(cfg) + ["logs"]
        if config.follow:
            cmd.append("--follow")
        if config.tail is not None:
            cmd.extend(["--tail", str(config.tail)])
        if config.no_color:
            cmd.append("--no-color")
        if config.timestamps:
            cmd.append("--timestamps")
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class PsCLI(_ComposeWrapperBase):
    """``docker compose ps`` for the rendered stack."""

    all = scfg.Value(False, isflag=True, short_alias=["a"], help="Include stopped containers.")
    services_only = scfg.Value(False, isflag=True, help="Print only service names (passes --services to docker compose).")
    quiet = scfg.Value(False, isflag=True, short_alias=["q"], help="Print only container IDs.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            _kubeai_stub("ps")
        cmd = _compose_base_cmd(cfg) + ["ps"]
        if config.all:
            cmd.append("--all")
        if config.services_only:
            cmd.append("--services")
        if config.quiet:
            cmd.append("--quiet")
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class RestartCLI(_ComposeWrapperBase):
    """``docker compose restart [services...]``."""

    timeout = scfg.Value(None, type=int, help="Stop timeout in seconds.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            _kubeai_stub("restart")
        cmd = _compose_base_cmd(cfg) + ["restart"]
        if config.timeout is not None:
            cmd.extend(["--timeout", str(config.timeout)])
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class PullCLI(_ComposeWrapperBase):
    """``docker compose pull [services...]``."""

    quiet = scfg.Value(False, isflag=True, short_alias=["q"])
    ignore_pull_failures = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            _kubeai_stub("pull")
        cmd = _compose_base_cmd(cfg) + ["pull"]
        if config.quiet:
            cmd.append("--quiet")
        if config.ignore_pull_failures:
            cmd.append("--ignore-pull-failures")
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class StartCLI(_ComposeWrapperBase):
    """``docker compose start [services...]``."""

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            _kubeai_stub("start")
        cmd = _compose_base_cmd(cfg) + ["start"]
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class StopCLI(_ComposeWrapperBase):
    """``docker compose stop [services...]``."""

    timeout = scfg.Value(None, type=int)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != "compose":
            _kubeai_stub("stop")
        cmd = _compose_base_cmd(cfg) + ["stop"]
        if config.timeout is not None:
            cmd.extend(["--timeout", str(config.timeout)])
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


# ---------------------------------------------------------------------------
# Smoke-test / benchmark commands
# ---------------------------------------------------------------------------


def _infer_default_base_url(cfg: dict[str, Any], config: Any) -> str:
    deployment = {
        "backend": backend_name(cfg),
        "cluster": cfg.get("cluster", {}),
        "ports": cfg.get("ports", {}),
    }
    return default_base_url(deployment, explicit=_as_mapping(config).get("base_url"))


def _smoke_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: Any = None,
    timeout: float = 30,
) -> requests.Response:
    """Wrapper around ``requests.{get,post}`` that emits actionable errors.

    The smoke test runs against a stack that may be (a) not listening yet,
    (b) listening but with an unhealthy upstream that resets connections, or
    (c) returning HTTP errors during model load. Each of those produces a
    different requests exception with a giant traceback by default; map them
    onto one-liner ``SystemExit`` messages that tell the user what to check.
    """
    try:
        if method.upper() == "GET":
            resp = requests.get(url, headers=headers, timeout=timeout)
        else:
            resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)
    except requests.exceptions.Timeout as ex:
        raise SystemExit(
            f"Request to {url} timed out after {timeout}s.\n"
            "The model may still be loading, or the server is overloaded.\n"
            "  vllm-stack logs vllm-*"
        ) from ex
    except requests.exceptions.ConnectionError as ex:
        # Two distinct sub-cases inside ConnectionError that warrant different
        # remediation: (a) nothing listening on the port, (b) something is
        # listening but it closed the connection without responding (typical
        # of LiteLLM up but a depended-on vLLM container still loading the
        # model and failing the dependency health-check chain).
        cause = ex.args[0] if ex.args else ex
        cause_str = str(cause)
        if "RemoteDisconnected" in cause_str or "Connection aborted" in cause_str:
            raise SystemExit(
                f"Connection to {url} was closed before a response arrived.\n"
                "The router is listening but an upstream service is not ready yet.\n"
                "Check container status and logs:\n"
                "  vllm-stack ps\n"
                "  vllm-stack logs vllm-*"
            ) from ex
        if "Connection refused" in cause_str or "Failed to establish a new connection" in cause_str:
            raise SystemExit(
                f"Could not connect to {url}: nothing is listening yet.\n"
                "If you just ran `vllm-stack up`, give the router a few seconds.\n"
                "  vllm-stack ps                # confirm the litellm container is running\n"
                "  vllm-stack logs litellm      # check for startup errors"
            ) from ex
        raise SystemExit(f"Connection error reaching {url}: {cause_str}") from ex
    status = getattr(resp, "status_code", 200)
    if status >= 400:
        body = getattr(resp, "text", "") or ""
        body = body.strip()
        if len(body) > 500:
            body = body[:500] + "... [truncated]"
        reason = getattr(resp, "reason", "") or ""
        if status in (401, 403):
            raise SystemExit(
                f"{status} {reason} from {url}.\n"
                "The auth key didn't match what the running container expects.\n"
                "If you re-rendered after the container started, the key in .env "
                "may have changed. Restart with:\n"
                "  vllm-stack down && vllm-stack up -d\n"
                f"Response: {body}"
            )
        if status == 503:
            raise SystemExit(
                f"{status} {reason} from {url}.\n"
                "An upstream service is unavailable (commonly the vLLM engine is still loading).\n"
                "  vllm-stack logs vllm-*\n"
                f"Response: {body}"
            )
        raise SystemExit(
            f"HTTP {status} {reason} from {url}.\nResponse: {body}"
        )
    return resp


def _resolve_smoke_test_protocol(
    cfg: dict[str, Any],
    config: Any,
    model_name: str,
) -> str:
    """Pick the OpenAI route for smoke-test based on protocol resolution order.

    1. ``--protocol`` CLI override (``chat`` or ``completions``).
    2. Resolved deployment: if the requested model maps to a service whose
       protocol_mode is known, use that.
    3. Active profile's primary service protocol_mode.
    4. Fallback: ``chat``.
    """
    overrides = _as_mapping(config)
    explicit = overrides.get("protocol")
    if explicit:
        return str(explicit)
    try:
        plan = build_plan(
            cfg,
            profile_name=overrides.get("profile"),
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
    except Exception:
        return "chat"
    services = plan.get("deployment", {}).get("services", [])
    for svc in services:
        if model_name in svc.get("served_aliases", []) or model_name == svc.get("served_model_name"):
            return str(svc.get("protocol_mode") or "chat")
    if services:
        return str(services[0].get("protocol_mode") or "chat")
    return "chat"


class SmokeTestCLI(
    _PathOverridesMixin,
    _ProfileOverrideMixin,
    _BackendOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
    _AllowUnsupportedMixin,
    _SimulateHardwareMixin,
):
    """Probe the running router with a single chat/completions request."""

    __command__ = "smoke-test"

    base_url = scfg.Value(None, type=str)
    api_key = scfg.Value(None, type=str)
    model = scfg.Value(None, type=str)
    prompt = scfg.Value("Say hello in one sentence.", type=str)
    max_tokens = scfg.Value(128, type=int)
    skip_chat = scfg.Value(False, isflag=True)
    protocol = scfg.Value(
        None,
        choices=["chat", "completions"],
        help="Force the smoke-test endpoint. Defaults to the resolved profile's protocol_mode.",
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        env = parse_env_file(runtime_env_path(cfg)) if backend_name(cfg) == "compose" else {}
        base_url = _infer_default_base_url(cfg, config)
        headers = {"Content-Type": "application/json"}
        api_key = config.api_key or env.get("LITELLM_MASTER_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        models_resp = _smoke_request("GET", f"{base_url}/models", headers=headers, timeout=30)
        models = models_resp.json().get("data", [])
        print(json.dumps(models_resp.json(), indent=2))
        if config.skip_chat:
            return 0
        if not models:
            raise SystemExit("No models returned from /models")
        model_name = config.model or models[0]["id"]
        protocol = _resolve_smoke_test_protocol(cfg, config, model_name)
        if protocol == "completions":
            payload = {
                "model": model_name,
                "prompt": config.prompt,
                "max_tokens": config.max_tokens,
            }
            endpoint = f"{base_url}/completions"
        else:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": config.prompt}],
                "max_tokens": config.max_tokens,
            }
            endpoint = f"{base_url}/chat/completions"
        resp = _smoke_request("POST", endpoint, headers=headers, json_body=payload, timeout=120)
        print(json.dumps(resp.json(), indent=2))
        return 0


class BenchmarkCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
):
    """Run benchmark_prompts.json against the router."""

    model = scfg.Value(None, type=str, required=True)
    base_url = scfg.Value(None, type=str)
    api_key = scfg.Value(None, type=str)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        # benchmark_prompts.json is a user-supplied fixture. Look for it
        # first in the config dir, then fall back to CWD so an ad-hoc
        # invocation from a checkout still picks up a sibling file.
        prompts_path = config_root() / "benchmark_prompts.json"
        if not prompts_path.exists():
            prompts_path = Path.cwd() / "benchmark_prompts.json"
        if not prompts_path.exists():
            raise SystemExit(
                f"benchmark_prompts.json not found at {config_root() / 'benchmark_prompts.json'} "
                f"or {Path.cwd() / 'benchmark_prompts.json'}"
            )
        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        cfg = config_for_runtime(config)
        env = parse_env_file(runtime_env_path(cfg))
        base_url = config.base_url or f"http://127.0.0.1:{cfg['ports']['litellm']}/v1"
        api_key = config.api_key or env.get("LITELLM_MASTER_KEY", "")
        data = run_benchmark(base_url, api_key, config.model, prompts)
        print(json.dumps(data, indent=2))
        return 0


# ---------------------------------------------------------------------------
# Modal CLI + entry point
# ---------------------------------------------------------------------------


class ManageCLI(scfg.ModalCLI):
    description = (
        "Render and run vLLM serving profiles through the Compose or KubeAI "
        "backends. Primary workflow: setup -> render -> up (or deploy)."
    )

    # Config / profile management
    setup = SetupCLI
    init = InitCLI
    resolve = ResolveCLI
    validate = ValidateCLI
    lock = LockCLI
    render = RenderCLI
    switch = SwitchCLI
    list_models = ListModelsCLI
    list_profiles = ListProfilesCLI
    explain = ExplainCLI
    describe_profile = DescribeProfileCLI
    verify_profile = VerifyProfileCLI
    kubeai_sync_resource_profiles = KubeaiSyncResourceProfilesCLI
    export_benchmark_bundle = ExportBenchmarkBundleCLI
    export_helm_bundle = ExportHelmBundleCLI

    # Runtime
    up = UpCLI
    down = DownCLI
    purge = PurgeCLI
    deploy = DeployCLI
    status = StatusCLI
    env = EnvCLI
    smoke_test = SmokeTestCLI
    benchmark = BenchmarkCLI

    # Compose day-2-ops wrappers
    logs = LogsCLI
    ps = PsCLI
    restart = RestartCLI
    pull = PullCLI
    start = StartCLI
    stop = StopCLI


def main(argv=None) -> int:
    rv = ManageCLI.main(argv=argv)
    return int(rv) if rv is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
