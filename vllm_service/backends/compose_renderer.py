from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from jinja2 import BaseLoader, Environment

from ..config import normalized_output, normalized_state
from ..diff_prompt import confirm_writes
from ..env_utils import ensure_secret, parse_env_file, write_env_file


def _template(name: str) -> str:
    return files("vllm_service").joinpath(f"templates/{name}").read_text(encoding="utf-8")


def render_compose_artifacts(lock_data: dict, *, assume_yes: bool = True) -> None:
    """Render component-aware Compose artifacts for the resolved stack."""
    deployment = dict(lock_data.get("deployment", {}))
    deployment["state"] = normalized_state(deployment.get("state", {}))
    deployment["output"] = normalized_output(deployment.get("output"))
    generated = Path(deployment["output"]["generated_dir"])
    generated.mkdir(parents=True, exist_ok=True)
    runtime_dir = Path(deployment["state"]["runtime"])
    runtime_dir.mkdir(parents=True, exist_ok=True)

    env_path = generated / ".env"
    existing = parse_env_file(env_path)
    env_values: dict[str, str] = {}

    frontends = deployment.get("frontends", {}) or {}
    gateways = deployment.get("gateways", {}) or {}
    providers = deployment.get("providers", {}) or {}

    if (frontends.get("open_webui") or {}).get("enabled"):
        env_values.update(
            {
                "OPENWEBUI_POSTGRES_DB": existing.get("OPENWEBUI_POSTGRES_DB", "openwebui"),
                "OPENWEBUI_POSTGRES_USER": existing.get("OPENWEBUI_POSTGRES_USER", "openwebui"),
                "OPENWEBUI_POSTGRES_PASSWORD": ensure_secret(existing, "OPENWEBUI_POSTGRES_PASSWORD"),
                "WEBUI_SECRET_KEY": ensure_secret(existing, "WEBUI_SECRET_KEY"),
            }
        )

    if (gateways.get("litellm") or {}).get("enabled"):
        env_values.update(
            {
                "LITELLM_POSTGRES_DB": existing.get("LITELLM_POSTGRES_DB", "litellm"),
                "LITELLM_POSTGRES_USER": existing.get("LITELLM_POSTGRES_USER", "litellm"),
                "LITELLM_POSTGRES_PASSWORD": ensure_secret(existing, "LITELLM_POSTGRES_PASSWORD"),
                "LITELLM_MASTER_KEY": ensure_secret(existing, "LITELLM_MASTER_KEY", prefix="sk-"),
            }
        )

    if (providers.get("vllm") or {}).get("enabled"):
        env_values.update(
            {
                "VLLM_BACKEND_API_KEY": ensure_secret(existing, "VLLM_BACKEND_API_KEY"),
                "HF_TOKEN": existing.get("HF_TOKEN", ""),
            }
        )

    # Preserve unknown/user-supplied keys, but let managed keys above win.
    for key, value in existing.items():
        env_values.setdefault(key, value)

    env = Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)
    normalized_lock = dict(lock_data)
    normalized_lock["deployment"] = deployment
    ctx = {"lock": normalized_lock}
    compose = env.from_string(_template("docker-compose.yml.j2")).render(**ctx) + "\n"

    compose_fpath = generated / "docker-compose.yml"
    planned: dict[Path, str] = {compose_fpath: compose}

    lite_llm_config_fpath = runtime_dir / "litellm_config.yaml"
    if (gateways.get("litellm") or {}).get("enabled"):
        litellm_cfg = env.from_string(_template("litellm_config.yaml.j2")).render(**ctx) + "\n"
        planned[lite_llm_config_fpath] = litellm_cfg

    if not confirm_writes(planned, assume_yes=assume_yes, title="Pending compose render"):
        raise SystemExit("Aborted by user; no files were written.")

    write_env_file(env_path, env_values)
    for path, text in planned.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
