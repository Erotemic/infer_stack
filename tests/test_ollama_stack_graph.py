from __future__ import annotations

from pathlib import Path

import yaml

from vllm_service.backends.compose_renderer import render_compose_artifacts
from vllm_service.config import initial_config
from vllm_service.hardware import simulate_inventory
from vllm_service.resolver import resolve
from vllm_service.validator import validate_resolved


def _cfg(tmp_path: Path, profile: str, backend: str = "compose") -> dict:
    cfg = initial_config()
    cfg["backend"] = backend
    cfg["active_profile"] = profile
    cfg["output"]["generated_dir"] = str(tmp_path / "generated")
    for key in ["hf_cache", "vllm_cache", "open_webui", "postgres_open_webui", "postgres_litellm", "ollama", "runtime"]:
        cfg["state"][key] = str(tmp_path / "state" / key)
    return cfg


def test_ollama_direct_renders_no_litellm(tmp_path: Path) -> None:
    dep = resolve(_cfg(tmp_path, "ollama-direct"), inventory=simulate_inventory("2x11"))
    assert dep["providers"]["ollama"]["enabled"] is True
    assert dep["gateways"]["litellm"]["enabled"] is False
    assert dep["frontends"]["open_webui"]["provider"] == "ollama"
    assert dep["services"] == []
    assert validate_resolved(dep)["ok"] is True
    render_compose_artifacts({"deployment": dep})
    compose = (tmp_path / "generated" / "docker-compose.yml").read_text()
    assert "ollama:" in compose
    assert "litellm:" not in compose
    assert "postgres-litellm:" not in compose
    assert "OLLAMA_BASE_URL: http://ollama:11434" in compose
    assert not (tmp_path / "state" / "runtime" / "litellm_config.yaml").exists()


def test_ollama_gateway_routes_through_litellm(tmp_path: Path) -> None:
    dep = resolve(_cfg(tmp_path, "ollama-qwen3.5-4b-gateway"), inventory=simulate_inventory("2x11"))
    assert dep["gateways"]["litellm"]["enabled"] is True
    assert dep["gateways"]["litellm"]["routes"]["qwen3.5-4b"]["provider"] == "ollama"
    render_compose_artifacts({"deployment": dep})
    cfg_doc = yaml.safe_load((tmp_path / "state" / "runtime" / "litellm_config.yaml").read_text())
    entry = cfg_doc["model_list"][0]
    assert entry["model_name"] == "qwen3.5-4b"
    assert entry["litellm_params"]["model"] == "ollama_chat/qwen3.5:4b"


def test_mixed_gateway_renders_ollama_vllm_and_litellm(tmp_path: Path) -> None:
    dep = resolve(_cfg(tmp_path, "mixed-ollama-smollm"), inventory=simulate_inventory("3x11"))
    assert dep["providers"]["ollama"]["enabled"] is True
    assert dep["providers"]["vllm"]["runtimes"]
    assert dep["gateways"]["litellm"]["enabled"] is True
    render_compose_artifacts({"deployment": dep})
    compose = (tmp_path / "generated" / "docker-compose.yml").read_text()
    assert "ollama:" in compose
    assert "vllm-smollm:" in compose
    assert "litellm:" in compose
    litellm = (tmp_path / "state" / "runtime" / "litellm_config.yaml").read_text()
    assert "ollama_chat/qwen3.5:4b" in litellm
    assert "openai/" in litellm


def test_kubeai_rejects_ollama_provider(tmp_path: Path) -> None:
    dep = resolve(_cfg(tmp_path, "ollama-direct", backend="kubeai"), inventory=simulate_inventory("2x11"))
    report = validate_resolved(dep)
    assert report["ok"] is False
    assert any("Ollama provider" in err for err in report["errors"])
