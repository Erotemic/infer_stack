from __future__ import annotations

from pathlib import Path

import yaml

from vllm_service.backends.compose_renderer import render_compose_artifacts
from vllm_service.backends.kubeai_renderer import render_kubeai_artifacts
from vllm_service.config import initial_config
from vllm_service.config import save_yaml
from vllm_service.contracts import build_profile_contract, load_profile_contract
from vllm_service.hardware import simulate_inventory
from vllm_service.resolver import resolve
from vllm_service.validator import validate_resolved


def _cfg(tmp_path: Path, *, backend: str = "compose") -> dict:
    cfg = initial_config()
    cfg["backend"] = backend
    cfg["state"] = {
        "hf_cache": "state/hf-cache",
        "open_webui": "state/open-webui",
        "postgres": "state/postgres",
        "runtime": "state/runtime",
    }
    cfg["ports"] = {"litellm": 14000, "open_webui": 13000, "postgres": 15432}
    return cfg


def _write_root_config(tmp_path: Path, *, backend: str = "compose") -> Path:
    cfg = _cfg(tmp_path, backend=backend)
    save_yaml(tmp_path / "config.yaml", cfg)
    save_yaml(tmp_path / "models.yaml", {"models": {}, "profiles": {}})
    return tmp_path


def _deployment(tmp_path: Path, profile_name: str, *, backend: str = "compose", inventory: str = "4x96") -> dict:
    cfg = _cfg(tmp_path, backend=backend)
    deployment = resolve(tmp_path, cfg, inventory=simulate_inventory(inventory), profile_name=profile_name)
    validated = validate_resolved(deployment)
    assert validated["ok"], validated
    return deployment


def test_profile_resolution_uses_named_serving_profile(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, "qwen2-72b-instruct-tp2-balanced")
    assert deployment["serving_profile"]["public_name"] == "qwen2-72b-instruct-tp2-balanced"
    assert deployment["serving_profile"]["logical_model_name"] == "qwen/qwen2-72b-instruct"
    assert deployment["services"][0]["tensor_parallel_size"] == 2
    assert "qwen2-72b-instruct-tp2-balanced" in deployment["router"]["aliases"]


def test_legacy_profile_alias_resolves_to_canonical_profile(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, "helm-qwen2-72b-instruct")
    assert deployment["serving_profile"]["name"] == "qwen2-72b-instruct-tp2-balanced"


def test_kubeai_render_uses_profile_identity(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, "qwen2-72b-instruct-tp2-balanced", backend="kubeai")
    render_kubeai_artifacts(tmp_path, {"deployment": deployment})
    models = list(yaml.safe_load_all((tmp_path / "generated" / "kubeai" / "models.yaml").read_text()))
    assert models[0]["metadata"]["name"] == "qwen2-72b-instruct-tp2-balanced"
    assert models[0]["metadata"]["annotations"]["vllm-service/logical-model-name"] == "qwen/qwen2-72b-instruct"
    assert models[0]["spec"]["resourceProfile"] == "gpu-tp2-balanced:2"
    assert "--tensor-parallel-size=2" in models[0]["spec"]["args"]
    assert "--served-model-name=qwen2-72b-instruct-tp2-balanced" in models[0]["spec"]["args"]


def test_compose_render_includes_profile_labels_and_aliases(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, "gpt-oss-20b-chat")
    render_compose_artifacts(tmp_path, {"deployment": deployment})
    compose_text = (tmp_path / "generated" / "docker-compose.yml").read_text()
    litellm_text = (tmp_path / "state" / "runtime" / "litellm_config.yaml").read_text()
    assert 'vllm_service.public_name: "gpt-oss-20b-chat"' in compose_text
    assert "openai/gpt-oss-20b" in litellm_text
    assert "gpt-oss-20b-chat" in litellm_text


def test_profile_contract_is_generic_and_backend_agnostic(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, "qwen2-72b-instruct-tp2-balanced")
    contract = build_profile_contract(deployment)
    service = contract["services"][0]
    assert contract["kind"] == "serving-profile-contract"
    assert contract["profile"]["public_name"] == "qwen2-72b-instruct-tp2-balanced"
    assert service["access"]["default"]["kind"] == "openai-compatible"
    assert service["access"]["additional"][0]["kind"] == "vllm-direct"
    assert service["access"]["additional"][0]["auth_env_name"] == "VLLM_API_KEY"
    assert "client_spec" not in str(contract)
    assert "model_deployments" not in str(contract)


def test_profile_contract_for_kubeai_uses_public_profile_name(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, "qwen2-72b-instruct-tp2-balanced", backend="kubeai")
    contract = build_profile_contract(deployment)
    access = contract["services"][0]["access"]["default"]
    assert access["kind"] == "openai-compatible"
    assert access["request_model_name"] == "qwen2-72b-instruct-tp2-balanced"
    assert access["base_url"].endswith("/openai/v1")


def test_load_profile_contract_uses_public_loader_for_qwen(tmp_path: Path) -> None:
    root = _write_root_config(tmp_path)
    contract = load_profile_contract(
        "qwen2-72b-instruct-tp2-balanced",
        root=root,
        simulate_hardware_spec="2x96",
    )
    assert contract["profile"]["public_name"] == "qwen2-72b-instruct-tp2-balanced"
    assert contract["services"][0]["model"]["logical_model_name"] == "qwen/qwen2-72b-instruct"


def test_compose_uses_separate_databases_for_litellm_and_openwebui(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, "gpt-oss-20b-chat")
    render_compose_artifacts(tmp_path, {"deployment": deployment})
    compose_text = (tmp_path / "generated" / "docker-compose.yml").read_text()
    env_text = (tmp_path / "generated" / ".env").read_text()
    env_kv = dict(
        line.split("=", 1)
        for line in env_text.splitlines()
        if line and "=" in line and not line.startswith("#")
    )
    assert env_kv["OPENWEBUI_POSTGRES_DB"] == "openwebui"
    assert env_kv["LITELLM_POSTGRES_DB"] == "litellm"
    litellm_block = compose_text.split("litellm:", 1)[1].split("open-webui:", 1)[0]
    openwebui_block = compose_text.split("open-webui:", 1)[1]
    assert "${LITELLM_POSTGRES_DB}" in litellm_block
    assert "${OPENWEBUI_POSTGRES_DB}" not in litellm_block
    assert "${OPENWEBUI_POSTGRES_DB}" in openwebui_block
    assert "${LITELLM_POSTGRES_DB}" not in openwebui_block
    assert "postgres-init:" in compose_text
    assert "CREATE DATABASE" in compose_text


def test_env_rewrite_preserves_unknown_key_value_pairs(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / ".env").write_text(
        "# user comment\n"
        "POSTGRES_USER=openwebui\n"
        "POSTGRES_PASSWORD=keepme\n"
        "VERBOSE=1\n"
        "CUSTOM_THING=abc\n"
        "HTTP_PROXY=http://proxy:3128\n",
        encoding="utf-8",
    )
    deployment = _deployment(tmp_path, "gpt-oss-20b-chat")
    render_compose_artifacts(tmp_path, {"deployment": deployment})
    text = (generated / ".env").read_text()
    assert "VERBOSE=1" in text
    assert "CUSTOM_THING=abc" in text
    assert "HTTP_PROXY=http://proxy:3128" in text
    assert "# user comment" in text
    env_kv = dict(
        line.split("=", 1)
        for line in text.splitlines()
        if line and "=" in line and not line.startswith("#")
    )
    assert env_kv["POSTGRES_PASSWORD"] == "keepme"


def test_env_migrates_legacy_postgres_db_to_openwebui_db(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / ".env").write_text(
        "POSTGRES_DB=openwebui\nPOSTGRES_USER=openwebui\nPOSTGRES_PASSWORD=keepme\n",
        encoding="utf-8",
    )
    deployment = _deployment(tmp_path, "gpt-oss-20b-chat")
    render_compose_artifacts(tmp_path, {"deployment": deployment})
    env_text = (generated / ".env").read_text()
    env_kv = dict(
        line.split("=", 1)
        for line in env_text.splitlines()
        if line and "=" in line and not line.startswith("#")
    )
    assert env_kv["OPENWEBUI_POSTGRES_DB"] == "openwebui"
    assert env_kv["LITELLM_POSTGRES_DB"] == "litellm"
    assert env_kv.get("POSTGRES_DB") == "openwebui"


def test_load_profile_contract_uses_public_loader_for_gpt_oss_variants(tmp_path: Path) -> None:
    root = _write_root_config(tmp_path)
    completions = load_profile_contract(
        "gpt-oss-20b-completions",
        root=root,
        simulate_hardware_spec="1x96",
    )
    chat = load_profile_contract(
        "gpt-oss-20b-chat",
        root=root,
        simulate_hardware_spec="1x96",
    )
    assert completions["services"][0]["protocol"]["mode"] == "completions"
    assert chat["services"][0]["protocol"]["mode"] == "chat"
