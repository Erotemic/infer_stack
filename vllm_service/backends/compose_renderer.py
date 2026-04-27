from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from jinja2 import BaseLoader, Environment

from ..config import normalized_state
from ..env_utils import ensure_secret, parse_env_file, write_env_file


def _template(name: str) -> str:
    return files("vllm_service").joinpath(f"templates/{name}").read_text(encoding="utf-8")


def _resolve_postgres_db_names(existing: dict[str, str]) -> tuple[str, str]:
    """Pick database names for Open WebUI and LiteLLM.

    Backward compatibility: if a legacy ``POSTGRES_DB`` is present and the
    new ``OPENWEBUI_POSTGRES_DB`` key is absent, seed the Open WebUI database
    name from the legacy value so existing chats remain associated with the
    same database. ``LITELLM_POSTGRES_DB`` defaults to ``litellm``.
    """
    legacy = existing.get("POSTGRES_DB", "").strip()
    openwebui_db = existing.get("OPENWEBUI_POSTGRES_DB", "").strip()
    if not openwebui_db:
        openwebui_db = legacy or "openwebui"
    litellm_db = existing.get("LITELLM_POSTGRES_DB", "").strip() or "litellm"
    return openwebui_db, litellm_db


def render_compose_artifacts(root: Path, lock_data: dict) -> None:
    generated = root / "generated"
    generated.mkdir(parents=True, exist_ok=True)

    deployment = dict(lock_data.get("deployment", {}))
    deployment["state"] = normalized_state(root, deployment.get("state", {}))
    runtime_dir = Path(deployment["state"]["runtime"])
    runtime_dir.mkdir(parents=True, exist_ok=True)

    env_path = generated / ".env"
    existing = parse_env_file(env_path)
    openwebui_db, litellm_db = _resolve_postgres_db_names(existing)

    env_values = {
        "POSTGRES_USER": existing.get("POSTGRES_USER", "openwebui"),
        "POSTGRES_PASSWORD": ensure_secret(existing, "POSTGRES_PASSWORD"),
        "OPENWEBUI_POSTGRES_DB": openwebui_db,
        "LITELLM_POSTGRES_DB": litellm_db,
        "LITELLM_MASTER_KEY": ensure_secret(existing, "LITELLM_MASTER_KEY"),
        "VLLM_BACKEND_API_KEY": ensure_secret(existing, "VLLM_BACKEND_API_KEY"),
        "WEBUI_SECRET_KEY": ensure_secret(existing, "WEBUI_SECRET_KEY"),
        "HF_TOKEN": existing.get("HF_TOKEN", ""),
    }
    # Keep the legacy POSTGRES_DB in sync if it was already present, so older
    # external tooling that still reads it sees a sensible value. Never
    # introduce it for fresh installs.
    if "POSTGRES_DB" in existing:
        env_values["POSTGRES_DB"] = openwebui_db

    write_env_file(env_path, env_values)

    env = Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)
    normalized_lock = dict(lock_data)
    normalized_lock["deployment"] = deployment
    ctx = {"lock": normalized_lock}
    compose = env.from_string(_template("docker-compose.yml.j2")).render(**ctx)
    litellm_cfg = env.from_string(_template("litellm_config.yaml.j2")).render(**ctx)

    compose_fpath = generated / "docker-compose.yml"
    lite_llm_config_fpath = runtime_dir / "litellm_config.yaml"
    compose_fpath.write_text(compose + "\n", encoding="utf-8")
    lite_llm_config_fpath.write_text(litellm_cfg + "\n", encoding="utf-8")
