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


def _split_compose_blocks(compose_text: str) -> dict[str, str]:
    """Split a rendered docker-compose.yml into a {service_name: body} dict.

    Service blocks start at column 2 (two-space indent) followed by the name
    and a colon. Splitting this way avoids substring confusion between e.g.
    ``open-webui:`` and ``postgres-open-webui:``.
    """
    import re
    pattern = re.compile(r"(?m)^  ([a-zA-Z0-9_-]+):\s*$")
    matches = list(pattern.finditer(compose_text))
    blocks: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(compose_text)
        blocks[name] = compose_text[start:end]
    return blocks


def test_compose_uses_separate_postgres_services(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, "gpt-oss-20b-chat")
    render_compose_artifacts(tmp_path, {"deployment": deployment})
    compose_text = (tmp_path / "generated" / "docker-compose.yml").read_text()
    blocks = _split_compose_blocks(compose_text)

    assert "postgres-open-webui" in blocks
    assert "postgres-litellm" in blocks
    assert "postgres-init" not in blocks
    assert "postgres-init" not in compose_text
    assert "shared-postgress" not in compose_text

    owui_pg = blocks["postgres-open-webui"]
    ll_pg = blocks["postgres-litellm"]
    assert "${OPENWEBUI_POSTGRES_DB}" in owui_pg
    assert "${OPENWEBUI_POSTGRES_USER}" in owui_pg
    assert "${OPENWEBUI_POSTGRES_PASSWORD}" in owui_pg
    assert "${LITELLM_POSTGRES_DB}" not in owui_pg
    assert "${LITELLM_POSTGRES_DB}" in ll_pg
    assert "${LITELLM_POSTGRES_USER}" in ll_pg
    assert "${LITELLM_POSTGRES_PASSWORD}" in ll_pg
    assert "${OPENWEBUI_POSTGRES_DB}" not in ll_pg


def test_compose_router_and_ui_point_at_their_own_postgres(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, "gpt-oss-20b-chat")
    render_compose_artifacts(tmp_path, {"deployment": deployment})
    compose_text = (tmp_path / "generated" / "docker-compose.yml").read_text()
    blocks = _split_compose_blocks(compose_text)

    litellm_block = blocks["litellm"]
    openwebui_block = blocks["open-webui"]
    assert "@postgres-litellm:5432/${LITELLM_POSTGRES_DB}" in litellm_block
    assert "@postgres-open-webui:" not in litellm_block
    assert "@postgres-open-webui:5432/${OPENWEBUI_POSTGRES_DB}" in openwebui_block
    assert "@postgres-litellm:" not in openwebui_block


def test_compose_env_uses_distinct_db_names_users_and_passwords(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, "gpt-oss-20b-chat")
    render_compose_artifacts(tmp_path, {"deployment": deployment})
    env_text = (tmp_path / "generated" / ".env").read_text()
    env_kv = dict(
        line.split("=", 1)
        for line in env_text.splitlines()
        if line and "=" in line and not line.startswith("#")
    )
    assert env_kv["OPENWEBUI_POSTGRES_DB"] == "openwebui"
    assert env_kv["LITELLM_POSTGRES_DB"] == "litellm"
    assert env_kv["OPENWEBUI_POSTGRES_USER"] == "openwebui"
    assert env_kv["LITELLM_POSTGRES_USER"] == "litellm"
    assert env_kv["OPENWEBUI_POSTGRES_PASSWORD"]
    assert env_kv["LITELLM_POSTGRES_PASSWORD"]
    assert env_kv["OPENWEBUI_POSTGRES_PASSWORD"] != env_kv["LITELLM_POSTGRES_PASSWORD"]
    # Old shared schema must not be emitted automatically.
    assert "POSTGRES_DB" not in env_kv
    assert "POSTGRES_USER" not in env_kv
    assert "POSTGRES_PASSWORD" not in env_kv


def test_env_rewrite_preserves_unknown_key_value_pairs(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / ".env").write_text(
        "# user comment\n"
        "OPENWEBUI_POSTGRES_PASSWORD=keepme\n"
        "VERBOSE=1\n"
        "CUSTOM_THING=abc\n"
        "HTTP_PROXY=http://proxy:3128\n"
        "NO_PROXY=localhost,127.0.0.1\n",
        encoding="utf-8",
    )
    deployment = _deployment(tmp_path, "gpt-oss-20b-chat")
    render_compose_artifacts(tmp_path, {"deployment": deployment})
    text = (generated / ".env").read_text()
    assert "VERBOSE=1" in text
    assert "CUSTOM_THING=abc" in text
    assert "HTTP_PROXY=http://proxy:3128" in text
    assert "NO_PROXY=localhost,127.0.0.1" in text
    assert "# user comment" in text
    env_kv = dict(
        line.split("=", 1)
        for line in text.splitlines()
        if line and "=" in line and not line.startswith("#")
    )
    # Existing managed values are preserved (password not rotated).
    assert env_kv["OPENWEBUI_POSTGRES_PASSWORD"] == "keepme"


def test_helm_pythia_profiles_render_as_completions(tmp_path: Path) -> None:
    for profile_name in ("helm-pythia-6.9b", "helm-pythia-2.8b-v0", "helm-pythia-1b-v0"):
        deployment = _deployment(tmp_path, profile_name, inventory="1x96")
        render_compose_artifacts(tmp_path, {"deployment": deployment})
        compose_text = (tmp_path / "generated" / "docker-compose.yml").read_text()
        assert 'vllm_service.protocol_mode: "completions"' in compose_text, profile_name
        assert deployment["services"][0]["protocol_mode"] == "completions", profile_name


def test_helm_base_model_profiles_render_as_completions(tmp_path: Path) -> None:
    for profile_name in ("helm-llama-2-7b", "helm-mistral-7b-v0.1", "helm-falcon-7b"):
        deployment = _deployment(tmp_path, profile_name, inventory="1x96")
        assert deployment["services"][0]["protocol_mode"] == "completions", profile_name


def test_validation_rejects_chat_protocol_for_completions_only_model(tmp_path: Path) -> None:
    from vllm_service.config import initial_config

    cfg = initial_config()
    cfg["backend"] = "compose"
    cfg["state"] = {
        "hf_cache": "state/hf-cache",
        "open_webui": "state/open-webui",
        "postgres_open_webui": "state/postgres-open-webui",
        "postgres_litellm": "state/postgres-litellm",
        "runtime": "state/runtime",
    }
    cfg["ports"] = {"litellm": 14000, "open_webui": 13000, "postgres": 15432}
    cfg["profiles"] = {
        "broken-pythia-chat": {
            "description": "Synthetic invalid profile: chat on a completions-only model.",
            "base_model": "pythia-1b-v0",
            "public_name": "broken-pythia-chat",
            "protocol_mode": "chat",
        }
    }
    deployment = resolve(tmp_path, cfg, inventory=simulate_inventory("1x96"), profile_name="broken-pythia-chat")
    report = validate_resolved(deployment)
    assert report["ok"] is False
    joined = " | ".join(report["errors"])
    assert "broken-pythia-chat" in joined
    assert "completions" in joined
    assert "pythia-1b-v0" in joined


def test_validation_accepts_chat_for_chat_capable_profile(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, "qwen2-5-7b-instruct-turbo-default", inventory="1x96")
    report = validate_resolved(deployment)
    assert report["ok"] is True, report
    assert deployment["services"][0]["protocol_mode"] == "chat"


def test_kubeai_protocol_validation_applies(tmp_path: Path) -> None:
    from vllm_service.config import initial_config

    cfg = initial_config()
    cfg["backend"] = "kubeai"
    cfg["resource_profiles"] = {
        "gpu-single-default": {
            "node_selector": {"nvidia.com/gpu.product": "X"},
            "requests": {"nvidia.com/gpu": 1},
            "limits": {"nvidia.com/gpu": 1},
        },
    }
    cfg["profiles"] = {
        "broken-pythia-chat-kubeai": {
            "description": "Synthetic invalid profile: chat on a completions-only model.",
            "base_model": "pythia-1b-v0",
            "public_name": "broken-pythia-chat-kubeai",
            "protocol_mode": "chat",
            "resource_profile": "gpu-single-default:1",
        }
    }
    deployment = resolve(
        tmp_path, cfg, inventory=simulate_inventory("1x96"),
        profile_name="broken-pythia-chat-kubeai",
    )
    report = validate_resolved(deployment)
    assert report["ok"] is False
    joined = " | ".join(report["errors"])
    assert "broken-pythia-chat-kubeai" in joined
    assert "completions" in joined


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
