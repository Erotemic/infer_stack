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
    GENERATED_DIR,
    KUBEAI_GENERATED_DIR,
    MODELS_FILE,
    PLAN_FILE,
    initial_config,
    kubeai_local_values_path,
    load_kubeai_resource_profiles,
    load_yaml,
    normalized_catalogs,
    save_kubeai_resource_profiles,
    save_yaml,
)
from .contracts import load_profile_contract
from .docker_utils import compose_down, compose_recreate_router, compose_up
from .env_utils import parse_env_file
from .exporters import export_benchmark_bundle
from .hardware import simulate_inventory
from .kubeai_ops import CommandError, deploy_rendered_artifacts, print_status as kubeai_print_status
from .profile_runtime import default_base_url
from .renderer import render_from_lock
from .resolver import resolve
from .validator import validate_resolved
from .verification import verify_profile


def root_dir() -> Path:
    return Path.cwd()


def config_path() -> Path:
    return root_dir() / CONFIG_FILE


def models_path() -> Path:
    return root_dir() / MODELS_FILE


def generated_dir() -> Path:
    return root_dir() / GENERATED_DIR


def kubeai_generated_dir() -> Path:
    return root_dir() / KUBEAI_GENERATED_DIR


def plan_path() -> Path:
    return root_dir() / PLAN_FILE


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        raise SystemExit(
            "No config.yaml found. Run `python manage.py setup --backend compose --profile qwen2-5-7b-instruct-turbo-default` first."
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
        "postgres": str(base / "postgres"),
        "runtime": str(base / "runtime"),
    }


def apply_config_overrides(cfg: dict[str, Any], args: argparse.Namespace | None) -> dict[str, Any]:
    if args is None:
        return deepcopy(cfg)
    out = deepcopy(cfg)
    out.setdefault("runtime", {})
    out.setdefault("ports", {})
    out.setdefault("state", {})
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
        return root_dir() / "state" / "runtime"
    p = Path(runtime)
    if p.is_absolute():
        return p
    return root_dir() / p


def runtime_env_path(cfg: dict[str, Any]) -> Path:
    return generated_dir() / ".env"


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
            "No config.yaml found. Run `python manage.py setup --backend compose --profile qwen2-5-7b-instruct-turbo-default` first."
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
    ]
    return any(_env_text(name) is not None for name in env_names)


def build_plan(
    cfg: dict[str, Any],
    *,
    profile_name: str | None = None,
    allow_unsupported: bool = False,
    inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = resolve(root_dir(), cfg, inventory=inventory, profile_name=profile_name)
    report = validate_resolved(resolved)
    return {
        "schema_version": 1,
        "allow_unsupported": bool(allow_unsupported),
        "validated": report,
        "deployment": resolved,
    }


def save_plan(plan: dict[str, Any]) -> Path:
    path = plan_path()
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
    current_plan = plan_path()
    backend = backend_name(cfg)

    if backend == "kubeai":
        required_outputs = [
            current_plan,
            kubeai_generated_dir() / "namespace.yaml",
            kubeai_generated_dir() / "kubeai-values.yaml",
            kubeai_generated_dir() / "models.yaml",
        ]
    else:
        required_outputs = [
            current_plan,
            generated_dir() / "docker-compose.yml",
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
            local_values_path = kubeai_local_values_path(root_dir())
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
        source = Path(args.resource_profiles_file)
        if not source.is_absolute():
            source = root_dir() / source
        values_doc = load_yaml(source)
        if "resourceProfiles" not in values_doc:
            raise SystemExit(f"{source} is missing a top-level resourceProfiles map")
        target = save_kubeai_resource_profiles(root_dir(), values_doc)
        if plan_path().exists():
            plan_path().unlink()
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
    save_plan(plan)
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
    save_plan(plan)
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
    save_plan(plan)
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
    save_plan(plan)
    render_from_lock(root_dir(), plan)
    print(f"Wrote {plan_path()}")
    if backend_name(cfg) == "kubeai":
        print(f"Rendered KubeAI artifacts into {kubeai_generated_dir()}")
    else:
        print(f"Rendered Compose into {generated_dir()}")
        print(f"Rendered mounted runtime files into {runtime_dir_for_config(cfg)}")
    return 0


def cmd_up(args: argparse.Namespace) -> int:
    cfg = config_for_runtime(args)
    if backend_name(cfg) != "compose":
        raise SystemExit("`up` only supports the compose backend. Use `deploy` for kubeai.")
    prior_litellm_cfg: str | None = None
    prior_path = runtime_litellm_config_path(cfg)
    if prior_path.exists():
        prior_litellm_cfg = prior_path.read_text(encoding="utf-8")
    if has_runtime_overrides(args) or render_is_stale(cfg):
        render_args = argparse.Namespace(
            profile=getattr(args, "profile", None),
            backend=getattr(args, "backend", None),
            compose_cmd=getattr(args, "compose_cmd", None),
            litellm_port=getattr(args, "litellm_port", None),
            namespace=getattr(args, "namespace", None),
            ingress_host=getattr(args, "ingress_host", None),
            ingress_enabled=getattr(args, "ingress_enabled", None),
            allow_unsupported=effective_allow_unsupported(args, cfg),
            simulate_hardware=getattr(args, "simulate_hardware", None),
        )
        cmd_render(render_args)
    compose_file = generated_dir() / "docker-compose.yml"
    env_file = generated_dir() / ".env"
    compose_up(
        cfg["runtime"]["compose_cmd"],
        compose_file,
        env_file,
        detach=args.detach,
        remove_orphans=True,
    )
    new_path = runtime_litellm_config_path(cfg)
    new_litellm_cfg = new_path.read_text(encoding="utf-8") if new_path.exists() else None
    if (
        args.detach
        and prior_litellm_cfg is not None
        and new_litellm_cfg is not None
        and prior_litellm_cfg != new_litellm_cfg
    ):
        # Router config changed since the last render. Force-recreate the
        # router and UI in place so they re-read the new alias list.
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
    compose_down(cfg["runtime"]["compose_cmd"], generated_dir() / "docker-compose.yml", runtime_env_path(cfg))
    return 0


def cmd_switch(args: argparse.Namespace) -> int:
    persisted_cfg = load_config()
    persisted_cfg["active_profile"] = args.profile
    save_yaml(config_path(), persisted_cfg)
    cfg = apply_config_overrides(persisted_cfg, args)

    prior_litellm_cfg: str | None = None
    if backend_name(cfg) == "compose":
        prior_path = runtime_litellm_config_path(cfg)
        if prior_path.exists():
            prior_litellm_cfg = prior_path.read_text(encoding="utf-8")

    plan = build_plan(
        cfg,
        profile_name=args.profile,
        allow_unsupported=effective_allow_unsupported(args, cfg),
        inventory=effective_inventory(args),
    )
    ensure_renderable(plan)
    save_plan(plan)
    render_from_lock(root_dir(), plan)
    if args.apply:
        if backend_name(cfg) == "compose":
            compose_file = generated_dir() / "docker-compose.yml"
            env_file = generated_dir() / ".env"
            # Bring the stack up convergently. --remove-orphans drops vLLM
            # services no longer in the rendered compose file. We never run
            # `down -v` or anything that would touch the Postgres/Open WebUI
            # volumes.
            compose_up(
                cfg["runtime"]["compose_cmd"],
                compose_file,
                env_file,
                detach=True,
                remove_orphans=True,
            )
            new_litellm_cfg = runtime_litellm_config_path(cfg).read_text(encoding="utf-8")
            if prior_litellm_cfg != new_litellm_cfg:
                # The router config (model aliases, upstream URLs) changed.
                # Recreate litellm + open-webui in place so they pick up the
                # new alias list. Persistent volumes are not touched.
                compose_recreate_router(
                    cfg["runtime"]["compose_cmd"],
                    compose_file,
                    env_file,
                    detach=True,
                )
        else:
            deploy_rendered_artifacts(root_dir(), plan["deployment"])
    print(f"Switched active_profile to {args.profile}")
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    cfg = load_config() if config_path().exists() else initial_config()
    cats = normalized_catalogs(root_dir(), cfg)
    for name, model in cats.get("models", {}).items():
        ref = model.get("hf_model_id") or model.get("url", "")
        print(f"{name}: {ref}")
    return 0


def cmd_list_profiles(args: argparse.Namespace) -> int:
    cfg = load_config() if config_path().exists() else initial_config()
    cats = normalized_catalogs(root_dir(), cfg)
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
    target = root_dir() / (args.file or PLAN_FILE)
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
        root=root_dir(),
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
    result = export_benchmark_bundle(
        root_dir(),
        plan["deployment"],
        base_url=args.base_url,
        output_dir=Path(args.output_dir) if args.output_dir else None,
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
    result = verify_profile(root_dir(), plan["deployment"])
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


def cmd_benchmark(args: argparse.Namespace) -> int:
    prompts = json.loads((root_dir() / "benchmark_prompts.json").read_text(encoding="utf-8"))
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
            allow_unsupported=effective_allow_unsupported(args, cfg),
            simulate_hardware=getattr(args, "simulate_hardware", None),
        )
        cmd_render(render_args)
    if backend_name(cfg) == "kubeai":
        plan = load_yaml(plan_path())
        try:
            deploy_rendered_artifacts(root_dir(), plan["deployment"])
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
        generated_dir() / "docker-compose.yml",
        generated_dir() / ".env",
        detach=args.detach,
        remove_orphans=True,
    )
    return 0


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
    proc = subprocess.run(cfg["runtime"]["compose_cmd"].split() + ["-f", str(generated_dir() / "docker-compose.yml"), "ps"])
    return int(proc.returncode)


def _infer_default_base_url(cfg: dict[str, Any], args: argparse.Namespace) -> str:
    deployment = {"backend": backend_name(cfg), "cluster": cfg.get("cluster", {}), "ports": cfg.get("ports", {})}
    return default_base_url(deployment, explicit=args.base_url)


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
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
    }
    resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))
    return 0


def cmd_kubeai_sync_resource_profiles(args: argparse.Namespace) -> int:
    source = Path(args.from_file)
    if not source.is_absolute():
        source = root_dir() / source
    values_doc = load_yaml(source)
    if "resourceProfiles" not in values_doc:
        raise SystemExit(f"{source} is missing a top-level resourceProfiles map")
    target = save_kubeai_resource_profiles(root_dir(), values_doc)
    if plan_path().exists():
        plan_path().unlink()
    profiles, _, _ = load_kubeai_resource_profiles(root_dir())
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
        s.set_defaults(func=func)

    s = sub.add_parser("up")
    add_override_args(s, include_profile=True, include_backend=True, include_compose=True, include_ports=True, include_cluster=True)
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM", help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80).")
    s.add_argument("-d", "--detach", action="store_true", help="Run in background instead of attaching to logs")
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
    s.set_defaults(func=cmd_deploy)

    s = sub.add_parser("status")
    add_override_args(s, include_backend=True, include_compose=True, include_ports=True, include_cluster=True)
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("smoke-test")
    add_override_args(s, include_backend=True, include_ports=True, include_cluster=True)
    s.add_argument("--base-url", default=None)
    s.add_argument("--api-key", default=None)
    s.add_argument("--model", default=None)
    s.add_argument("--prompt", default="Say hello in one sentence.")
    s.add_argument("--max-tokens", type=int, default=128)
    s.add_argument("--skip-chat", action="store_true")
    s.set_defaults(func=cmd_smoke_test)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == '__main__':
    main()
