# Compare how scriptconfig looks
from __future__ import annotations

import scriptconfig as scfg
import json
from pathlib import Path
from typing import Any

from .benchmark import run_benchmark
from .config import (
    CONFIG_FILE,
    MODELS_FILE,
    generated_dir_for_config,
    initial_config,
    normalized_state,
    load_yaml,
    plan_path_for_config,
    save_yaml,
)
from .docker_utils import compose_down, compose_up
from .env_utils import parse_env_file
from .paths import config_root
from .renderer import render_from_lock
from .resolver import resolve
from .validator import validate_resolved


def config_path() -> Path:
    return config_root() / CONFIG_FILE


def models_path() -> Path:
    return config_root() / MODELS_FILE


def _safe_cfg() -> dict[str, Any]:
    path = config_path()
    if path.exists():
        return load_yaml(path)
    return initial_config()


def generated_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg if cfg is not None else _safe_cfg()
    return generated_dir_for_config(cfg)


def plan_path(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg if cfg is not None else _safe_cfg()
    return plan_path_for_config(cfg)


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        raise SystemExit("No config.yaml found. Run `manage init` first.")
    return load_yaml(path)


def runtime_dir_for_config(cfg: dict) -> Path:
    state = normalized_state(cfg.get("state", {}))
    return Path(state["runtime"])


def runtime_env_path(cfg: dict) -> Path:
    return generated_dir(cfg) / ".env"


def runtime_litellm_config_path(cfg: dict) -> Path:
    return runtime_dir_for_config(cfg) / "litellm_config.yaml"


def effective_allow_unsupported(config) -> bool:
    arg_value = bool(config.get("allow_unsupported", False))
    policy_value = bool(config.get("policy", {}).get("allow_unsupported_render", False))
    return arg_value or policy_value


def build_plan(
    cfg: dict[str, Any],
    *,
    profile_name: str | None = None,
    allow_unsupported: bool = False,
) -> dict[str, Any]:
    resolved = resolve(cfg, profile_name=profile_name)
    report = validate_resolved(resolved)
    return {
        "schema_version": 1,
        "allow_unsupported": bool(allow_unsupported),
        "validated": report,
        "deployment": resolved,
    }


def save_plan(plan, cfg=None):
    path = plan_path(cfg)
    save_yaml(path, plan)
    return path


def ensure_renderable(plan):
    validated = plan.get("validated", {}) or {}
    if validated.get("errors") and not plan.get("allow_unsupported", False):
        raise SystemExit(
            "Refusing to render because the resolved plan contains validation errors. "
            "Use `--allow-unsupported` to override."
        )


def render_is_stale(cfg=None):
    cfg = load_config() if cfg is None else cfg
    cfg_path = config_path()
    current_plan = plan_path(cfg)
    compose_file = generated_dir(cfg) / "docker-compose.yml"
    runtime_env = runtime_env_path(cfg)
    runtime_litellm_cfg = runtime_litellm_config_path(cfg)

    required_outputs = [
        current_plan,
        compose_file,
        runtime_env,
        runtime_litellm_cfg,
    ]
    if any(not p.exists() for p in required_outputs):
        return True

    if cfg_path.exists():
        oldest_generated = min(p.stat().st_mtime for p in required_outputs)
        if cfg_path.stat().st_mtime > oldest_generated:
            return True

    if current_plan.stat().st_mtime > compose_file.stat().st_mtime:
        return True
    if current_plan.stat().st_mtime > runtime_env.stat().st_mtime:
        return True
    if current_plan.stat().st_mtime > runtime_litellm_cfg.stat().st_mtime:
        return True

    return False


class InitCLI(scfg.DataConfig):
    force = scfg.Value(False, isflag=True, help='overwrite existing config')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cfg_path = config_path()
        if cfg_path.exists() and not config.force:
            raise SystemExit("config.yaml already exists. Use --force to overwrite.")
        save_yaml(cfg_path, initial_config())
        if not models_path().exists():
            save_yaml(models_path(), {"models": {}, "profiles": {}})
        print(f"Wrote {cfg_path}")
        return 0


class ResolveCLI(scfg.DataConfig):
    profile = scfg.Value(None, type=str, help='profile name')
    allow_unsupported = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cfg = load_config()
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported({
                **cfg,
                "allow_unsupported": config.allow_unsupported,
            }),
        )
        save_plan(plan, cfg)
        print(json.dumps(plan["deployment"], indent=2))
        return 0


class ValidateCLI(scfg.DataConfig):
    profile = scfg.Value(None, type=str)
    allow_unsupported = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cfg = load_config()
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported({
                **cfg,
                "allow_unsupported": config.allow_unsupported,
            }),
        )
        save_plan(plan, cfg)
        print(json.dumps(plan["validated"], indent=2))
        return 0 if plan["validated"]["ok"] else 2


class RenderCLI(scfg.DataConfig):
    profile = scfg.Value(None, type=str)
    allow_unsupported = scfg.Value(False, isflag=True)
    yes = scfg.Value(
        False,
        isflag=True,
        short_alias=["y"],
        help="Apply rendered changes without prompting. Without this, render shows a per-file diff and asks for confirmation.",
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cfg = load_config()
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported({
                **cfg,
                "allow_unsupported": config.allow_unsupported,
            }),
        )
        ensure_renderable(plan)
        save_plan(plan, cfg)
        render_from_lock(plan, assume_yes=bool(config.yes))
        print(f"Wrote {plan_path(cfg)}")
        print(f"Rendered Compose into {generated_dir(cfg)}")
        print(f"Rendered mounted runtime files into {runtime_dir_for_config(cfg)}")
        return 0


class UpCLI(scfg.DataConfig):
    allow_unsupported = scfg.Value(False, isflag=True)
    detach = scfg.Value(False, isflag=True, short_alias=['d'])
    yes = scfg.Value(False, isflag=True, short_alias=["y"], help="If `up` triggers a re-render, apply changes without prompting.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cfg = load_config()
        if render_is_stale(cfg):
            RenderCLI.main(
                argv=False,
                profile=None,
                allow_unsupported=config.allow_unsupported,
                yes=config.yes,
            )
        compose_up(
            cfg["runtime"]["compose_cmd"],
            generated_dir(cfg) / "docker-compose.yml",
            generated_dir(cfg) / ".env",
            detach=config.detach,
            remove_orphans=True,
        )
        return 0


class DownCLI(scfg.DataConfig):
    @classmethod
    def main(cls, argv=1, **kwargs):
        cls.cli(argv=argv, data=kwargs)
        cfg = load_config()
        compose_down(
            cfg["runtime"]["compose_cmd"],
            generated_dir(cfg) / "docker-compose.yml",
            runtime_env_path(cfg),
        )
        return 0


class SwitchCLI(scfg.DataConfig):
    profile = scfg.Value(None, type=str, position=1)
    apply = scfg.Value(False, isflag=True)
    allow_unsupported = scfg.Value(False, isflag=True)
    yes = scfg.Value(False, isflag=True, short_alias=["y"], help="Apply rendered changes without prompting.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cfg = load_config()
        cfg["active_profile"] = config.profile
        save_yaml(config_path(), cfg)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported({
                **cfg,
                "allow_unsupported": config.allow_unsupported,
            }),
        )
        ensure_renderable(plan)
        save_plan(plan, cfg)
        render_from_lock(plan, assume_yes=bool(config.yes))
        if config.apply:
            compose_down(
                cfg["runtime"]["compose_cmd"],
                generated_dir(cfg) / "docker-compose.yml",
                generated_dir(cfg) / ".env",
            )
            compose_up(
                cfg["runtime"]["compose_cmd"],
                generated_dir(cfg) / "docker-compose.yml",
                generated_dir(cfg) / ".env",
                detach=False,
                remove_orphans=True,
            )
        print(f"Switched active_profile to {config.profile}")
        return 0


class ExplainCLI(scfg.DataConfig):
    file = scfg.Value(None, type=str)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
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


class BenchmarkCLI(scfg.DataConfig):
    model = scfg.Value(None, type=str, required=True)
    base_url = scfg.Value(None, type=str)
    api_key = scfg.Value(None, type=str)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        prompts_path = config_root() / "benchmark_prompts.json"
        if not prompts_path.exists():
            prompts_path = Path.cwd() / "benchmark_prompts.json"
        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        cfg = load_config()
        env = parse_env_file(runtime_env_path(cfg))
        base_url = config.base_url or f"http://127.0.0.1:{cfg['ports']['litellm']}/v1"
        api_key = config.api_key or env.get("LITELLM_MASTER_KEY", "")
        data = run_benchmark(base_url, api_key, config.model, prompts)
        print(json.dumps(data, indent=2))
        return 0


class ManageCLI(scfg.ModalCLI):
    description = (
        "Primary workflow: init -> edit config.yaml -> render -> up. "
        "Advanced commands like resolve/validate/lock still exist for inspection."
    )

    init = InitCLI
    resolve = ResolveCLI
    validate = ValidateCLI
    render = RenderCLI
    up = UpCLI
    down = DownCLI
    switch = SwitchCLI
    explain = ExplainCLI
    benchmark = BenchmarkCLI


def main(argv=1, **kwargs):
    return ManageCLI.main(argv=argv, **kwargs)


if __name__ == '__main__':
    main()
