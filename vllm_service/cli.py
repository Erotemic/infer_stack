from __future__ import annotations

import argparse
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import requests

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
    plan_path_for_config,
    save_kubeai_resource_profiles,
    save_yaml,
)
from .contracts import load_profile_contract
from .docker_utils import compose_down, compose_recreate_router, compose_up
from .env_utils import parse_env_file
from .exporters import export_benchmark_bundle
from .hardware import simulate_inventory
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
    """Load config.yaml if present; otherwise return defaults.

    Used by helpers that may run before a config exists (e.g. ``cmd_init``
    or ``cmd_explain --file <path>``). Callers that already have a cfg in
    hand should pass it explicitly to avoid redundant disk reads.
    """
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


def _arg_or_env(args: argparse.Namespace, attr: str, env_name: str, *, caster=None):
    if hasattr(args, attr):
        value = getattr(args, attr)
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
        "open_webui": str(base / "open-webui"),
        "postgres_open_webui": str(base / "postgres-open-webui"),
        "postgres_litellm": str(base / "postgres-litellm"),
        "runtime": str(base / "runtime"),
    }


def apply_config_overrides(cfg: dict[str, Any], args: argparse.Namespace | None) -> dict[str, Any]:
    if args is None:
        return deepcopy(cfg)
    out = deepcopy(cfg)
    out.setdefault("runtime", {})
    out.setdefault("ports", {})
    out.setdefault("state", {})
    out.setdefault("output", {})
    out.setdefault("cluster", {})
    out["cluster"].setdefault("ingress", {})

    backend = _arg_or_env(args, "backend", "VLLM_SERVICE_BACKEND")
    if backend:
        out["backend"] = backend

    profile = _arg_or_env(args, "profile", "VLLM_SERVICE_PROFILE")
    if profile:
        out["active_profile"] = profile

    compose_cmd = _arg_or_env(args, "compose_cmd", "VLLM_SERVICE_COMPOSE_CMD")
    if compose_cmd:
        out["runtime"]["compose_cmd"] = compose_cmd

    litellm_port = getattr(args, "litellm_port", None)
    if litellm_port is None:
        litellm_port = _env_int("VLLM_SERVICE_LITELLM_PORT")
    if litellm_port is not None:
        out["ports"]["litellm"] = litellm_port

    open_webui_port = getattr(args, "open_webui_port", None)
    if open_webui_port is None:
        open_webui_port = _env_int("VLLM_SERVICE_OPEN_WEBUI_PORT")
    if open_webui_port is not None:
        out["ports"]["open_webui"] = open_webui_port

    postgres_port = getattr(args, "postgres_port", None)
    if postgres_port is None:
        postgres_port = _env_int("VLLM_SERVICE_POSTGRES_PORT")
    if postgres_port is not None:
        out["ports"]["postgres"] = postgres_port

    state_root = _arg_or_env(args, "state_root", "VLLM_SERVICE_STATE_ROOT")
    if state_root:
        out["state"].update(_configured_state_paths(state_root))

    runtime_dir = _arg_or_env(args, "runtime_dir", "VLLM_SERVICE_RUNTIME_DIR")
    if runtime_dir:
        out["state"]["runtime"] = runtime_dir

    generated_dir_override = _arg_or_env(args, "generated_dir", "VLLM_SERVICE_GENERATED_DIR")
    if generated_dir_override:
        out["output"]["generated_dir"] = generated_dir_override
    elif not out["output"].get("generated_dir"):
        # Older configs predate the ``output`` section; populate it so the
        # resolved path is visible/editable in config.yaml instead of being
        # silently picked up from defaults each run.
        out["output"]["generated_dir"] = default_output_config()["generated_dir"]

    namespace = _arg_or_env(args, "namespace", "VLLM_SERVICE_NAMESPACE")
    if namespace:
        out["cluster"]["namespace"] = namespace

    ingress_host = _arg_or_env(args, "ingress_host", "VLLM_SERVICE_INGRESS_HOST")
    if ingress_host:
        out["cluster"]["ingress"]["host"] = ingress_host

    ingress_enabled = getattr(args, "ingress_enabled", None)
    if ingress_enabled is None:
        ingress_enabled = _env_bool("VLLM_SERVICE_INGRESS_ENABLED")
    if ingress_enabled is not None:
        out["cluster"]["ingress"]["enabled"] = bool(ingress_enabled)

    return out


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


def effective_allow_unsupported(args: argparse.Namespace | None, cfg: dict[str, Any]) -> bool:
    arg_value = bool(getattr(args, "allow_unsupported", False)) if args is not None else False
    policy_value = bool(cfg.get("policy", {}).get("allow_unsupported_render", False))
    return arg_value or policy_value


def effective_inventory(args: argparse.Namespace | None) -> dict[str, Any] | None:
    spec = getattr(args, "simulate_hardware", None) if args is not None else None
    if not spec:
        return None
    return simulate_inventory(spec)


def backend_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("backend", "compose")).lower()


def config_for_runtime(args: argparse.Namespace | None, *, allow_missing: bool = False) -> dict[str, Any]:
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


def has_runtime_overrides(args: argparse.Namespace | None) -> bool:
    if args is None:
        return False
    attrs = [
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
        "generated_dir",
    ]
    if any(hasattr(args, attr) and getattr(args, attr) is not None for attr in attrs):
        return True
    env_names = [
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
    ]
    return any(_env_text(name) is not None for name in env_names)


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


def cmd_init(args: argparse.Namespace) -> int:
    cfg_path = config_path()
    if cfg_path.exists() and not args.force:
        raise SystemExit("config.yaml already exists. Use --force to overwrite.")
    save_yaml(cfg_path, initial_config())
    if not models_path().exists():
        save_yaml(models_path(), {"models": {}, "profiles": {}})
    print(f"Wrote {cfg_path}")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    cfg_path = config_path()
    if cfg_path.exists() and not args.reset:
        cfg = load_yaml(cfg_path)
    else:
        cfg = initial_config()
    cfg = apply_config_overrides(cfg, args)
    save_yaml(cfg_path, cfg)
    if not models_path().exists():
        save_yaml(models_path(), {"models": {}, "profiles": {}})
    if getattr(args, "resource_profiles_file", None):
        # User-supplied path: anchor on the user's current working directory
        # so ``--resource-profiles-file ./values.yaml`` behaves as typed.
        source = Path(args.resource_profiles_file)
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


def cmd_resolve(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    plan = build_plan(
        cfg,
        profile_name=args.profile,
        allow_unsupported=effective_allow_unsupported(args, cfg),
        inventory=effective_inventory(args),
    )
    save_plan(plan, cfg)
    print(json.dumps(plan["deployment"], indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    plan = build_plan(
        cfg,
        profile_name=getattr(args, "profile", None),
        allow_unsupported=effective_allow_unsupported(args, cfg),
        inventory=effective_inventory(args),
    )
    save_plan(plan, cfg)
    print(json.dumps(plan["validated"], indent=2))
    return 0 if plan["validated"]["ok"] else 2


def cmd_lock(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    plan = build_plan(
        cfg,
        profile_name=getattr(args, "profile", None),
        allow_unsupported=effective_allow_unsupported(args, cfg),
        inventory=effective_inventory(args),
    )
    if not plan["validated"]["ok"] and not plan["allow_unsupported"]:
        raise SystemExit(
            "Refusing to write plan.yaml because validation failed. Use --allow-unsupported to override."
        )
    save_plan(plan, cfg)
    print(json.dumps(plan, indent=2))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    plan = build_plan(
        cfg,
        profile_name=getattr(args, "profile", None),
        allow_unsupported=effective_allow_unsupported(args, cfg),
        inventory=effective_inventory(args),
    )
    ensure_renderable(plan)
    save_plan(plan, cfg)
    render_from_lock(plan, assume_yes=bool(getattr(args, "yes", False)))
    print(f"Wrote {plan_path(cfg)}")
    if backend_name(cfg) == "kubeai":
        print(f"Rendered KubeAI artifacts into {kubeai_generated_dir(cfg)}")
    else:
        print(f"Rendered Compose into {generated_dir(cfg)}")
        print(f"Rendered mounted runtime files into {runtime_dir_for_config(cfg)}")
    return 0


def cmd_up(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    if backend_name(cfg) != "compose":
        raise SystemExit("`up` only supports the compose backend. Use `deploy` for kubeai.")
    if has_runtime_overrides(args) or render_is_stale(cfg):
        render_args = argparse.Namespace(
            profile=getattr(args, "profile", None),
            backend=getattr(args, "backend", None),
            compose_cmd=getattr(args, "compose_cmd", None),
            litellm_port=getattr(args, "litellm_port", None),
            namespace=getattr(args, "namespace", None),
            ingress_host=getattr(args, "ingress_host", None),
            ingress_enabled=getattr(args, "ingress_enabled", None),
            generated_dir=getattr(args, "generated_dir", None),
            allow_unsupported=effective_allow_unsupported(args, cfg),
            simulate_hardware=getattr(args, "simulate_hardware", None),
            yes=bool(getattr(args, "yes", False)),
        )
        cmd_render(render_args)
    compose_file = generated_dir(cfg) / "docker-compose.yml"
    env_file = generated_dir(cfg) / ".env"
    # TODO: today we recreate litellm + open-webui wholesale on any apply.
    # That's correct but coarse — every active chat sees a brief router
    # blip even when only one of several vLLM services actually changed.
    # A finer design would diff the new router config against the running
    # one and use LiteLLM's admin API (/model/new, /model/delete) to
    # add/drop just the affected aliases, leaving unaffected models
    # serving traffic uninterrupted. Open WebUI's dropdown would still
    # need a re-fetch, but that can be done without a container recreate.
    # Worth doing once we have a multi-model deployment where users would
    # notice the blip.
    try:
        compose_up(
            cfg["runtime"]["compose_cmd"],
            compose_file,
            env_file,
            detach=args.detach,
            remove_orphans=True,
        )
    finally:
        if args.detach:
            # Always force-recreate the router and UI so they re-read the
            # rendered alias list. In `finally` so it still runs when
            # compose_up raises (e.g. a vLLM healthcheck times out before
            # litellm's `depends_on: service_healthy` gate releases) —
            # that was the failure mode that left a stale model list
            # advertised on /v1/models after a profile switch.
            compose_recreate_router(
                cfg["runtime"]["compose_cmd"],
                compose_file,
                env_file,
                detach=True,
            )
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    if backend_name(cfg) != "compose":
        raise SystemExit("`down` only supports the compose backend.")
    compose_down(cfg["runtime"]["compose_cmd"], generated_dir(cfg) / "docker-compose.yml", runtime_env_path(cfg))
    return 0


def cmd_switch(args: argparse.Namespace) -> int:
    persisted_cfg = load_config()
    persisted_cfg["active_profile"] = args.profile
    save_yaml(config_path(), persisted_cfg)
    cfg = apply_config_overrides(persisted_cfg, args)

    plan = build_plan(
        cfg,
        profile_name=args.profile,
        allow_unsupported=effective_allow_unsupported(args, cfg),
        inventory=effective_inventory(args),
    )
    ensure_renderable(plan)
    save_plan(plan, cfg)
    render_from_lock(plan, assume_yes=bool(getattr(args, "yes", False)))
    if args.apply:
        if backend_name(cfg) == "compose":
            compose_file = generated_dir(cfg) / "docker-compose.yml"
            env_file = generated_dir(cfg) / ".env"
            # TODO: this recreates litellm + open-webui wholesale, which
            # interrupts every model — even ones that didn't change between
            # profiles. A nicer future design would diff the new router
            # config against the running one and use LiteLLM's admin API
            # (/model/new, /model/delete) to add/drop just the affected
            # aliases, leaving unaffected vLLM services serving traffic the
            # whole time. Open WebUI's dropdown would still need a refresh,
            # but that can be done without a container restart. Worth doing
            # once we routinely run multi-model profiles where the blip is
            # user-visible.
            try:
                # Bring the stack up convergently. --remove-orphans drops
                # vLLM services no longer in the rendered compose file. We
                # never run `down -v` or anything that would touch the
                # Postgres/Open WebUI volumes.
                compose_up(
                    cfg["runtime"]["compose_cmd"],
                    compose_file,
                    env_file,
                    detach=True,
                    remove_orphans=True,
                )
            finally:
                # Always recreate litellm + open-webui so they re-read the
                # new alias list. We used to gate this on a byte-diff of
                # the rendered litellm_config.yaml, but that left a stale
                # model list when compose_up raised before the diff ran
                # (e.g. litellm's `depends_on: service_healthy` on a slow
                # vLLM service tripping the up timeout). `finally` makes
                # this fire regardless. Persistent volumes are untouched.
                compose_recreate_router(
                    cfg["runtime"]["compose_cmd"],
                    compose_file,
                    env_file,
                    detach=True,
                )
        else:
            deploy_rendered_artifacts(plan["deployment"])
    print(f"Switched active_profile to {args.profile}")
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    cfg = load_config() if config_path().exists() else initial_config()
    cats = normalized_catalogs(cfg)
    for name, model in cats.get("models", {}).items():
        ref = model.get("hf_model_id") or model.get("url", "")
        print(f"{name}: {ref}")
    return 0


def cmd_list_profiles(args: argparse.Namespace) -> int:
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


def cmd_explain(args: argparse.Namespace) -> int:
    if args.file:
        # User-supplied path: anchor on the user's current working directory.
        target = Path(args.file)
        if not target.is_absolute():
            target = Path.cwd() / target
    else:
        target = plan_path()
    if not target.exists():
        raise SystemExit(f"Missing file: {target}")
    print(json.dumps(load_yaml(target), indent=2))
    return 0


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


def cmd_describe_profile(args: argparse.Namespace) -> int:
    contract = load_profile_contract(
        args.profile,
        backend=_arg_or_env(args, "backend", "VLLM_SERVICE_BACKEND"),
        simulate_hardware_spec=getattr(args, "simulate_hardware", None),
    )
    return _print_structured(contract, args.format, args.output)


def _cmd_export_bundle(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    plan = build_plan(
        cfg,
        profile_name=args.profile,
        allow_unsupported=effective_allow_unsupported(args, cfg),
        inventory=effective_inventory(args),
    )
    ensure_renderable(plan)
    print(
        "Benchmark bundle export here is transitional; prefer the helm_audit "
        "integration layer for CRFM HELM bundle generation."
    )
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
    result = export_benchmark_bundle(
        plan["deployment"],
        base_url=args.base_url,
        output_dir=output_dir,
    )
    print(f"Wrote {result['bundle_path']}")
    print(f"Wrote {result['model_deployments_path']}")
    return 0


def cmd_export_benchmark_bundle(args: argparse.Namespace) -> int:
    return _cmd_export_bundle(args)


def cmd_export_helm_bundle(args: argparse.Namespace) -> int:
    return _cmd_export_bundle(args)


def cmd_verify_profile(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    plan = build_plan(
        cfg,
        profile_name=args.profile,
        allow_unsupported=effective_allow_unsupported(args, cfg),
        inventory=effective_inventory(args),
    )
    result = verify_profile(plan["deployment"])
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


def cmd_benchmark(args: argparse.Namespace) -> int:
    # benchmark_prompts.json is a user-supplied fixture. Look for it first in
    # the config dir, then fall back to the CWD so an ad-hoc invocation
    # ``vllm-stack benchmark`` from a checkout still picks up a sibling file.
    prompts_path = config_root() / "benchmark_prompts.json"
    if not prompts_path.exists():
        prompts_path = Path.cwd() / "benchmark_prompts.json"
    if not prompts_path.exists():
        raise SystemExit(
            f"benchmark_prompts.json not found at {config_root() / 'benchmark_prompts.json'} "
            f"or {Path.cwd() / 'benchmark_prompts.json'}"
        )
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    cfg = config_for_runtime(args)
    env = parse_env_file(runtime_env_path(cfg))
    base_url = args.base_url or f"http://127.0.0.1:{cfg['ports']['litellm']}/v1"
    api_key = args.api_key or env.get("LITELLM_MASTER_KEY", "")
    data = run_benchmark(base_url, api_key, args.model, prompts)
    print(json.dumps(data, indent=2))
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    if has_runtime_overrides(args) or render_is_stale(cfg):
        render_args = argparse.Namespace(
            profile=getattr(args, "profile", None),
            backend=getattr(args, "backend", None),
            compose_cmd=getattr(args, "compose_cmd", None),
            litellm_port=getattr(args, "litellm_port", None),
            namespace=getattr(args, "namespace", None),
            ingress_host=getattr(args, "ingress_host", None),
            ingress_enabled=getattr(args, "ingress_enabled", None),
            generated_dir=getattr(args, "generated_dir", None),
            allow_unsupported=effective_allow_unsupported(args, cfg),
            simulate_hardware=getattr(args, "simulate_hardware", None),
            yes=bool(getattr(args, "yes", False)),
        )
        cmd_render(render_args)
    if backend_name(cfg) == "kubeai":
        plan = load_yaml(plan_path(cfg))
        try:
            deploy_rendered_artifacts(plan["deployment"])
        except CommandError as ex:
            namespace = cfg.get("cluster", {}).get("namespace", "kubeai")
            raise SystemExit(
                f"Failed to deploy to namespace {namespace!r}. Confirm `python manage.py setup --backend kubeai --namespace {namespace}` "
                "matches the namespace where the KubeAI Helm release is installed.\n"
                f"Original error: {ex}"
            ) from ex
        return 0
    compose_up(
        cfg["runtime"]["compose_cmd"],
        generated_dir(cfg) / "docker-compose.yml",
        generated_dir(cfg) / ".env",
        detach=args.detach,
        remove_orphans=True,
    )
    return 0


def _compose_base_cmd(cfg: dict[str, Any]) -> list[str]:
    """Build the shared ``docker compose -f ... --env-file ...`` prefix.

    Every compose-wrapper subcommand uses this so that the user doesn't
    need to cd into the rendered-artifacts directory just to run a
    one-shot ``ps`` / ``restart`` / ``exec`` / ``pull`` / ``logs``.
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


def cmd_logs(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    if backend_name(cfg) != "compose":
        _kubeai_stub("logs")
    cmd = _compose_base_cmd(cfg) + ["logs"]
    if getattr(args, "follow", False):
        cmd.append("--follow")
    tail = getattr(args, "tail", None)
    if tail is not None:
        cmd.extend(["--tail", str(tail)])
    if getattr(args, "no_color", False):
        cmd.append("--no-color")
    if getattr(args, "timestamps", False):
        cmd.append("--timestamps")
    cmd.extend(getattr(args, "services", None) or [])
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def cmd_ps(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    if backend_name(cfg) != "compose":
        _kubeai_stub("ps")
    cmd = _compose_base_cmd(cfg) + ["ps"]
    if getattr(args, "all", False):
        cmd.append("--all")
    if getattr(args, "services_only", False):
        cmd.append("--services")
    if getattr(args, "quiet", False):
        cmd.append("--quiet")
    cmd.extend(getattr(args, "services", None) or [])
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def cmd_restart(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    if backend_name(cfg) != "compose":
        _kubeai_stub("restart")
    cmd = _compose_base_cmd(cfg) + ["restart"]
    timeout = getattr(args, "timeout", None)
    if timeout is not None:
        cmd.extend(["--timeout", str(timeout)])
    cmd.extend(getattr(args, "services", None) or [])
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def cmd_pull(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    if backend_name(cfg) != "compose":
        _kubeai_stub("pull")
    cmd = _compose_base_cmd(cfg) + ["pull"]
    if getattr(args, "quiet", False):
        cmd.append("--quiet")
    if getattr(args, "ignore_pull_failures", False):
        cmd.append("--ignore-pull-failures")
    cmd.extend(getattr(args, "services", None) or [])
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def cmd_start(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    if backend_name(cfg) != "compose":
        _kubeai_stub("start")
    cmd = _compose_base_cmd(cfg) + ["start"]
    cmd.extend(getattr(args, "services", None) or [])
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def cmd_stop(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    if backend_name(cfg) != "compose":
        _kubeai_stub("stop")
    cmd = _compose_base_cmd(cfg) + ["stop"]
    timeout = getattr(args, "timeout", None)
    if timeout is not None:
        cmd.extend(["--timeout", str(timeout)])
    cmd.extend(getattr(args, "services", None) or [])
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def cmd_status(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    if backend_name(cfg) == "kubeai":
        namespace = cfg.get("cluster", {}).get("namespace", "kubeai")
        try:
            kubeai_print_status(namespace)
        except CommandError as ex:
            raise SystemExit(
                f"Failed to query KubeAI resources in namespace {namespace!r}. Confirm `python manage.py setup --backend kubeai --namespace {namespace}` "
                "matches the namespace where the KubeAI Helm release is installed.\n"
                f"Original error: {ex}"
            ) from ex
        return 0
    proc = subprocess.run(_compose_base_cmd(cfg) + ["ps"])
    return int(proc.returncode)


def _infer_default_base_url(cfg: dict[str, Any], args: argparse.Namespace) -> str:
    deployment = {"backend": backend_name(cfg), "cluster": cfg.get("cluster", {}), "ports": cfg.get("ports", {})}
    return default_base_url(deployment, explicit=args.base_url)


def _resolve_smoke_test_protocol(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    model_name: str,
) -> str:
    """Pick the OpenAI route to exercise in smoke-test.

    Order of precedence:
      1. ``--protocol`` CLI override (``chat`` or ``completions``).
      2. Resolved deployment: if the requested model maps to a service whose
         protocol_mode is known, use that.
      3. Active profile's primary service protocol_mode.
      4. Fallback: ``chat``.
    """
    explicit = getattr(args, "protocol", None)
    if explicit:
        return str(explicit)
    try:
        plan = build_plan(
            cfg,
            profile_name=getattr(args, "profile", None),
            allow_unsupported=effective_allow_unsupported(args, cfg),
            inventory=effective_inventory(args),
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


def cmd_smoke_test(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    env = parse_env_file(runtime_env_path(cfg)) if backend_name(cfg) == "compose" else {}
    base_url = _infer_default_base_url(cfg, args)
    headers = {"Content-Type": "application/json"}
    api_key = args.api_key or env.get("LITELLM_MASTER_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    models_resp = requests.get(f"{base_url}/models", headers=headers, timeout=30)
    models_resp.raise_for_status()
    models = models_resp.json().get("data", [])
    print(json.dumps(models_resp.json(), indent=2))
    if args.skip_chat:
        return 0
    if not models:
        raise SystemExit("No models returned from /models")
    model_name = args.model or models[0]["id"]
    protocol = _resolve_smoke_test_protocol(cfg, args, model_name)
    if protocol == "completions":
        payload = {
            "model": model_name,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
        }
        endpoint = f"{base_url}/completions"
    else:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
        }
        endpoint = f"{base_url}/chat/completions"
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))
    return 0


def cmd_kubeai_sync_resource_profiles(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args, allow_missing=True)
    # User-supplied path: anchor on the user's current working directory.
    source = Path(args.from_file)
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


def add_override_args(
    parser: argparse.ArgumentParser,
    *,
    include_profile: bool = False,
    include_backend: bool = True,
    include_compose: bool = False,
    include_ports: bool = False,
    include_cluster: bool = False,
    include_state: bool = False,
    include_output: bool = True,
) -> None:
    if include_profile:
        parser.add_argument("--profile", default=None)
    if include_backend:
        parser.add_argument("--backend", choices=["compose", "kubeai"], default=None)
    if include_compose:
        parser.add_argument("--compose-cmd", default=None)
    if include_ports:
        parser.add_argument("--litellm-port", type=int, default=None)
        parser.add_argument("--open-webui-port", type=int, default=None)
        parser.add_argument("--postgres-port", type=int, default=None)
    if include_state:
        parser.add_argument("--state-root", default=None)
        parser.add_argument("--runtime-dir", default=None)
    if include_output:
        parser.add_argument(
            "--generated-dir",
            default=None,
            help=(
                "Directory to write rendered artifacts (docker-compose.yml, "
                ".env, plan.yaml, kubeai/*) into. Defaults to "
                "<data-dir>/generated (data-dir is ~/.cache/vllm_service by "
                "default). May also be set via the VLLM_SERVICE_GENERATED_DIR "
                "env var or output.generated_dir in config.yaml."
            ),
        )
    if include_cluster:
        parser.add_argument("--namespace", default=None)
        parser.add_argument("--ingress-host", default=None)
        parser.add_argument("--ingress", dest="ingress_enabled", action="store_true")
        parser.add_argument("--no-ingress", dest="ingress_enabled", action="store_false")
        parser.set_defaults(ingress_enabled=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Manage named serving profiles for local Compose and Kubernetes-backed KubeAI serving."
    )
    p.add_argument(
        "--config-dir",
        default=None,
        help=(
            f"Directory containing config.yaml / models.yaml. Defaults to "
            f"~/.config/vllm_service (XDG_CONFIG_HOME) or ${CONFIG_DIR_ENV} "
            f"when set."
        ),
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help=(
            f"Directory for rendered artifacts (generated/) and bind-mount "
            f"state (hf-cache, postgres volumes, etc.). Defaults to "
            f"~/.cache/vllm_service (XDG_CACHE_HOME) or ${DATA_DIR_ENV} "
            f"when set. Per-knob env vars (VLLM_SERVICE_GENERATED_DIR, "
            f"VLLM_SERVICE_STATE_ROOT) and config.yaml entries still take "
            f"precedence."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup")
    add_override_args(
        s,
        include_profile=True,
        include_backend=True,
        include_compose=True,
        include_ports=True,
        include_cluster=True,
        include_state=True,
    )
    s.add_argument("--reset", action="store_true", help="Start from default config values before applying overrides.")
    s.add_argument(
        "--resource-profiles-file",
        default=None,
        help="For kubeai setups, sync a local Helm values file with resourceProfiles into kubeai-values.local.yaml.",
    )
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("init")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_init)

    for name, func in [("resolve", cmd_resolve), ("validate", cmd_validate), ("lock", cmd_lock), ("render", cmd_render)]:
        s = sub.add_parser(name)
        add_override_args(s, include_profile=True, include_backend=True, include_compose=True, include_ports=True, include_cluster=True)
        s.add_argument("--allow-unsupported", action="store_true")
        s.add_argument("--simulate-hardware", default=None, metavar="NxM", help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80).")
        if name == "render":
            s.add_argument(
                "-y", "--yes",
                action="store_true",
                help="Apply rendered changes without prompting. Without this flag, render shows a per-file diff and asks for confirmation.",
            )
        s.set_defaults(func=func)

    s = sub.add_parser("up")
    add_override_args(s, include_profile=True, include_backend=True, include_compose=True, include_ports=True, include_cluster=True)
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM", help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80).")
    s.add_argument("-d", "--detach", action="store_true", help="Run in background instead of attaching to logs")
    s.add_argument(
        "-y", "--yes",
        action="store_true",
        help="If `up` triggers a re-render, apply the rendered changes without prompting.",
    )
    s.set_defaults(func=cmd_up)

    s = sub.add_parser("down")
    add_override_args(s, include_backend=True, include_compose=True, include_ports=True)
    s.set_defaults(func=cmd_down)

    s = sub.add_parser("switch")
    s.add_argument("profile")
    add_override_args(s, include_backend=True, include_compose=True, include_ports=True, include_cluster=True)
    s.add_argument("--apply", action="store_true")
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM", help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80).")
    s.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Apply rendered changes without prompting.",
    )
    s.set_defaults(func=cmd_switch)

    s = sub.add_parser("list-models")
    s.set_defaults(func=cmd_list_models)

    s = sub.add_parser("list-profiles")
    s.set_defaults(func=cmd_list_profiles)

    s = sub.add_parser("kubeai-sync-resource-profiles")
    s.add_argument("--from-file", required=True, help="Helm values file containing a top-level resourceProfiles map.")
    s.set_defaults(func=cmd_kubeai_sync_resource_profiles)

    s = sub.add_parser("explain")
    s.add_argument("--file", default=None)
    s.set_defaults(func=cmd_explain)

    s = sub.add_parser("describe-profile")
    s.add_argument("profile")
    add_override_args(s, include_backend=True)
    s.add_argument("--format", choices=["json", "yaml"], default="yaml")
    s.add_argument("--output", default=None)
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM", help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80).")
    s.set_defaults(func=cmd_describe_profile)

    s = sub.add_parser("export-benchmark-bundle")
    s.add_argument("profile")
    add_override_args(s, include_backend=True, include_compose=True, include_ports=True, include_cluster=True)
    s.add_argument("--base-url", default=None)
    s.add_argument("--output-dir", default=None)
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM", help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80).")
    s.set_defaults(func=cmd_export_benchmark_bundle)

    s = sub.add_parser("export-helm-bundle")
    s.add_argument("profile")
    add_override_args(s, include_backend=True, include_compose=True, include_ports=True, include_cluster=True)
    s.add_argument("--base-url", default=None)
    s.add_argument("--output-dir", default=None)
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM", help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80).")
    s.set_defaults(func=cmd_export_helm_bundle)

    s = sub.add_parser("verify-profile")
    s.add_argument("profile")
    add_override_args(s, include_backend=True, include_compose=True, include_ports=True, include_cluster=True)
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM", help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80).")
    s.set_defaults(func=cmd_verify_profile)

    s = sub.add_parser("benchmark")
    s.add_argument("--model", required=True)
    add_override_args(s, include_backend=True, include_compose=True, include_ports=True)
    s.add_argument("--base-url", default=None)
    s.add_argument("--api-key", default=None)
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("deploy")
    add_override_args(s, include_profile=True, include_backend=True, include_compose=True, include_ports=True, include_cluster=True)
    s.add_argument("-d", "--detach", action="store_true")
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM", help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80).")
    s.add_argument(
        "-y", "--yes",
        action="store_true",
        help="If `deploy` triggers a re-render, apply the rendered changes without prompting.",
    )
    s.set_defaults(func=cmd_deploy)

    s = sub.add_parser("status")
    add_override_args(s, include_backend=True, include_compose=True, include_ports=True, include_cluster=True)
    s.set_defaults(func=cmd_status)

    s = sub.add_parser(
        "logs",
        help="Tail rendered Compose service logs without typing the full docker compose path.",
    )
    add_override_args(s, include_backend=True, include_compose=True)
    s.add_argument(
        "services",
        nargs="*",
        help="Optional service names to filter (e.g. open-webui litellm). Empty = all services.",
    )
    s.add_argument(
        "-f", "--follow",
        action="store_true",
        help="Stream logs (docker compose logs -f).",
    )
    s.add_argument(
        "--tail",
        default=None,
        help="Tail the last N lines (default: all). Pass a number or 'all'.",
    )
    s.add_argument("--timestamps", action="store_true")
    s.add_argument("--no-color", action="store_true")
    s.set_defaults(func=cmd_logs)

    s = sub.add_parser("ps", help="`docker compose ps` for the rendered stack.")
    add_override_args(s, include_backend=True, include_compose=True)
    s.add_argument("services", nargs="*", help="Optional service filter.")
    s.add_argument("-a", "--all", action="store_true", help="Include stopped containers.")
    s.add_argument(
        "--services-only",
        action="store_true",
        help="Print only service names (passes --services to docker compose).",
    )
    s.add_argument("-q", "--quiet", action="store_true", help="Print only container IDs.")
    s.set_defaults(func=cmd_ps)

    s = sub.add_parser("restart", help="`docker compose restart [services...]`.")
    add_override_args(s, include_backend=True, include_compose=True)
    s.add_argument("services", nargs="*", help="Optional service filter (empty = all).")
    s.add_argument("--timeout", type=int, default=None, help="Stop timeout in seconds.")
    s.set_defaults(func=cmd_restart)

    s = sub.add_parser("pull", help="`docker compose pull [services...]`.")
    add_override_args(s, include_backend=True, include_compose=True)
    s.add_argument("services", nargs="*", help="Optional service filter (empty = all).")
    s.add_argument("-q", "--quiet", action="store_true")
    s.add_argument("--ignore-pull-failures", action="store_true")
    s.set_defaults(func=cmd_pull)

    s = sub.add_parser("start", help="`docker compose start [services...]` (re-starts stopped containers).")
    add_override_args(s, include_backend=True, include_compose=True)
    s.add_argument("services", nargs="*", help="Optional service filter (empty = all).")
    s.set_defaults(func=cmd_start)

    s = sub.add_parser("stop", help="`docker compose stop [services...]` (stops without removing).")
    add_override_args(s, include_backend=True, include_compose=True)
    s.add_argument("services", nargs="*", help="Optional service filter (empty = all).")
    s.add_argument("--timeout", type=int, default=None)
    s.set_defaults(func=cmd_stop)

    s = sub.add_parser("smoke-test")
    add_override_args(
        s,
        include_profile=True,
        include_backend=True,
        include_ports=True,
        include_cluster=True,
    )
    s.add_argument("--base-url", default=None)
    s.add_argument("--api-key", default=None)
    s.add_argument("--model", default=None)
    s.add_argument("--prompt", default="Say hello in one sentence.")
    s.add_argument("--max-tokens", type=int, default=128)
    s.add_argument("--skip-chat", action="store_true")
    s.add_argument(
        "--protocol",
        choices=["chat", "completions"],
        default=None,
        help="Force the smoke-test endpoint to /v1/chat/completions or /v1/completions. "
             "Defaults to the resolved profile's protocol_mode.",
    )
    s.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Allow protocol resolution to use a profile that has validation errors.",
    )
    s.add_argument("--simulate-hardware", default=None, metavar="NxM")
    s.set_defaults(func=cmd_smoke_test)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    # Apply --config-dir / --data-dir before any subcommand runs so all
    # downstream calls into config_root() / data_root() see the override.
    if getattr(args, "config_dir", None):
        set_config_root(args.config_dir)
    if getattr(args, "data_dir", None):
        set_data_root(args.data_dir)
    return int(args.func(args))


if __name__ == '__main__':
    main()
