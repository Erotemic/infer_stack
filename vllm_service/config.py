from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from .catalog import normalize_model_catalog, normalize_profile_catalog
from .hardware import detect_inventory

CONFIG_FILE = Path("config.yaml")
MODELS_FILE = Path("models.yaml")
GENERATED_DIR = Path("generated")
PLAN_FILE = GENERATED_DIR / "plan.yaml"
KUBEAI_GENERATED_DIR = GENERATED_DIR / "kubeai"
KUBEAI_VALUES_FILE = KUBEAI_GENERATED_DIR / "kubeai-values.yaml"
KUBEAI_LOCAL_VALUES_FILE = Path("kubeai-values.local.yaml")

RESOLVED_FILE = PLAN_FILE
LOCK_FILE = PLAN_FILE

PINNED_IMAGES = {
    "postgres": "postgres:16.8",
    "open_webui": "ghcr.io/open-webui/open-webui:v0.8.6",
    "litellm": "ghcr.io/berriai/litellm:v1.82.3-stable",
    "vllm": "vllm/vllm-openai:v0.19.1",
}

DEFAULT_PORTS = {
    "litellm": 14000,
    "open_webui": 13000,
    "postgres": 15432,
}


def _default_storage_root() -> Path:
    preferred = Path("/data/service/docker/vllm-stack")
    if preferred.parent.exists():
        return preferred
    return Path.cwd() / "state"


def default_state_paths() -> dict[str, str]:
    storage_root = _default_storage_root()
    return {
        "hf_cache": str(storage_root / "hf-cache"),
        "vllm_cache": str(storage_root / "vllm-cache"),
        "open_webui": str(storage_root / "open-webui"),
        "postgres_open_webui": str(storage_root / "postgres-open-webui"),
        "postgres_litellm": str(storage_root / "postgres-litellm"),
        "runtime": str(storage_root / "runtime"),
    }


def default_cluster_config() -> dict[str, Any]:
    return {
        "namespace": "kubeai",
        "kubeai_release_name": "kubeai",
        "kubeai_chart": "kubeai/kubeai",
        "service_name": "kubeai",
        "ingress": {
            "enabled": False,
            "class_name": "traefik",
            "host": "",
            "path_prefix": "/",
            "tls_secret_name": "",
        },
    }


def default_resource_profiles() -> dict[str, Any]:
    return {
        "gpu-single-default": {
            "limits": {"nvidia.com/gpu": 1},
            "requests": {"nvidia.com/gpu": 1},
        },
        "gpu-tp2-balanced": {
            "limits": {"nvidia.com/gpu": 2},
            "requests": {"nvidia.com/gpu": 2},
        },
        "gpu-tp2-maxctx": {
            "limits": {"nvidia.com/gpu": 2},
            "requests": {"nvidia.com/gpu": 2},
        },
    }


def kubeai_values_path(root: Path) -> Path:
    return root / KUBEAI_VALUES_FILE


def kubeai_local_values_path(root: Path) -> Path:
    return root / KUBEAI_LOCAL_VALUES_FILE


def resource_profiles_to_kubeai_values(resource_profiles: dict[str, Any] | None) -> dict[str, Any]:
    values: dict[str, Any] = {"resourceProfiles": {}}
    for name, spec in (resource_profiles or {}).items():
        item: dict[str, Any] = {}
        if spec.get("node_selector"):
            item["nodeSelector"] = deepcopy(spec["node_selector"])
        if spec.get("requests"):
            item["requests"] = deepcopy(spec["requests"])
        if spec.get("limits"):
            item["limits"] = deepcopy(spec["limits"])
        if spec.get("tolerations"):
            item["tolerations"] = deepcopy(spec["tolerations"])
        if spec.get("runtime_class_name"):
            item["runtimeClassName"] = spec["runtime_class_name"]
        if spec.get("scheduler_name"):
            item["schedulerName"] = spec["scheduler_name"]
        if spec.get("image_name"):
            item["imageName"] = spec["image_name"]
        values["resourceProfiles"][name] = item
    return values


def kubeai_values_to_resource_profiles(values_doc: dict[str, Any] | None) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for name, spec in ((values_doc or {}).get("resourceProfiles", {}) or {}).items():
        profiles[name] = deepcopy(spec)
    return profiles


def load_kubeai_resource_profiles(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = kubeai_local_values_path(root)
    if not path.exists():
        return {}, {}, path
    values_doc = load_yaml(path)
    return kubeai_values_to_resource_profiles(values_doc), values_doc, path


def save_kubeai_resource_profiles(root: Path, values_doc: dict[str, Any]) -> Path:
    path = kubeai_local_values_path(root)
    save_yaml(path, values_doc)
    return path


def normalized_state(root: Path, state: dict[str, Any] | None) -> dict[str, str]:
    normalized = deepcopy(default_state_paths())
    for key, value in (state or {}).items():
        if value in (None, ""):
            continue
        p = Path(value)
        if not p.is_absolute():
            p = root / p
        normalized[key] = str(p)
    return normalized


def normalized_cluster(config: dict[str, Any] | None) -> dict[str, Any]:
    return deep_merge(default_cluster_config(), config or {})


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _load_template_yaml(name: str) -> dict[str, Any]:
    text = files("vllm_service").joinpath(f"templates/{name}").read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def builtin_models_catalog() -> dict[str, Any]:
    return _load_template_yaml("default-models.yaml")


def builtin_profiles_catalog() -> dict[str, Any]:
    return _load_template_yaml("default-profiles.yaml")


def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def merged_catalogs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    built_models = builtin_models_catalog() if config.get("catalog", {}).get("builtin_models", True) else {}
    built_profiles = builtin_profiles_catalog() if config.get("catalog", {}).get("builtin_profiles", True) else {}
    user_path = root / config.get("catalog", {}).get("user_models_file", str(MODELS_FILE))
    user = load_yaml(user_path) if user_path.exists() else {}
    return {
        "models": deep_merge(built_models.get("models", {}), user.get("models", {})),
        "profiles": deep_merge(built_profiles.get("profiles", {}), user.get("profiles", {})),
    }


def normalized_catalogs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    catalogs = merged_catalogs(root, config)
    models = normalize_model_catalog(catalogs.get("models", {}))
    raw_profiles = {**catalogs.get("profiles", {}), **deepcopy(config.get("profiles", {}))}
    profiles = normalize_profile_catalog(raw_profiles, models)
    return {
        "models": models,
        "profiles": profiles,
    }


def initial_config() -> dict[str, Any]:
    inventory = detect_inventory()
    default_profile = "qwen-mixed" if inventory.get("gpu_count", 0) >= 4 else "workstation-safe"
    return {
        "name": "local-llm-stack",
        "backend": "compose",
        "active_profile": default_profile,
        "catalog": {
            "builtin_models": True,
            "builtin_profiles": True,
            "user_models_file": str(MODELS_FILE),
        },
        "policy": {
            "require_fit_validation": True,
            "reserve_display_gpu": "auto",
            "forbid_reserved_gpu_use": False,
            "require_homogeneous_multi_gpu_groups": True,
            "minimum_vram_headroom_gib": 2,
            "allow_unsupported_render": False,
        },
        "runtime": {
            "compose_cmd": "docker compose",
            "target_inventory": "auto",
        },
        "ports": deepcopy(DEFAULT_PORTS),
        "images": deepcopy(PINNED_IMAGES),
        "state": default_state_paths(),
        "cluster": default_cluster_config(),
        "resource_profiles": default_resource_profiles(),
        "profiles": {},
    }
