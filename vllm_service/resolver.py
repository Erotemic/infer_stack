from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .catalog import canonical_profile_name, normalize_model_catalog, normalize_profile_catalog
from .config import (
    load_kubeai_resource_profiles,
    merged_catalogs,
    normalized_cluster,
    normalized_state,
    resource_profiles_to_kubeai_values,
)
from .hardware import detect_inventory


def _available_gpu_indices(inventory: dict[str, Any], reserve_display_gpu: str | bool | None) -> list[int]:
    gpus = deepcopy(inventory.get("gpus", []))
    if reserve_display_gpu == "auto":
        return [g["index"] for g in gpus if not g.get("display_active")]
    if reserve_display_gpu is True:
        return [g["index"] for g in gpus if not g.get("display_active")]
    return [g["index"] for g in gpus]


def _first_fit(available: list[int], count: int) -> tuple[list[int], str | None]:
    if len(available) < count:
        return available[:], f"need {count} GPUs but only {len(available)} available"
    return available[:count], None


def _runtime_value(service: dict[str, Any], model: dict[str, Any], key: str, default: Any) -> Any:
    runtime = service.get("runtime", {})
    if key in runtime:
        return runtime[key]
    if key in service:
        return service[key]
    return model.get("defaults", {}).get(key, default)


def _resolve_service(
    service: dict[str, Any],
    models: dict[str, Any],
    inventory: dict[str, Any],
    policy: dict[str, Any],
    used: set[int],
) -> dict[str, Any]:
    model_key = service["base_model"]
    if model_key not in models:
        raise KeyError(f"Unknown model: {model_key}")
    model = deepcopy(models[model_key])
    placement = deepcopy(service.get("placement", {}))
    topology = deepcopy(service.get("topology", {}))
    available = [i for i in _available_gpu_indices(inventory, policy.get("reserve_display_gpu", "auto")) if i not in used]

    if "tp" in topology and "tensor_parallel_size" not in topology:
        topology["tensor_parallel_size"] = topology["tp"]
    if "dp" in topology and "data_parallel_size" not in topology:
        topology["data_parallel_size"] = topology["dp"]

    strategy = placement.get("strategy", "first_fit")
    placement_error = None
    gpu_indices: list[int] = []
    if strategy in {"exact", "multi_gpu", "single_gpu"}:
        gpu_indices = list(placement.get("gpu_indices", []))
        if not gpu_indices:
            placement_error = f"service {service['service_name']} uses {strategy} placement but no gpu_indices were provided"
    else:
        gpu_count = int(placement.get("gpu_count", topology.get("tensor_parallel_size", model.get("preferred_gpu_count", 1)) or 1))
        gpu_indices, placement_error = _first_fit(available, gpu_count)

    used.update(gpu_indices)
    tp = int(topology.get("tensor_parallel_size", max(1, len(gpu_indices) or placement.get("gpu_count", 1))))
    dp = int(topology.get("data_parallel_size", 1))

    tool_calling = deepcopy(model.get("tool_calling", {}))
    tool_calling = {**tool_calling, **deepcopy(service.get("tool_calling", {}))}
    tool_call_parser = tool_calling.get("parser")
    enable_auto_tool_choice = bool(tool_calling.get("auto", False) and tool_call_parser)

    hf_model_id = service.get("hf_model_id", model.get("hf_model_id", ""))
    model_url = service.get("url") or model.get("url") or (f"hf://{hf_model_id}" if hf_model_id else "")
    benchmark_transport = deepcopy(service.get("benchmark_transport", service.get("transport", {})))

    return {
        "service_name": service["service_name"],
        "profile_name": service["profile_name"],
        "profile_public_name": service["public_name"],
        "kubernetes_name": service["kubernetes_name"],
        "model_ref": model_key,
        "hf_model_id": hf_model_id,
        "model_url": model_url,
        "logical_model_name": service.get("logical_model_name", model.get("logical_model_name", model_key)),
        "served_model_name": service.get("served_model_name", model.get("served_model_name", model_key)),
        "served_aliases": deepcopy(service.get("served_aliases", [])),
        "protocol_mode": service.get("protocol_mode", "chat"),
        "supported_protocols": list(model.get("supported_protocols", ["chat", "completions"])),
        "modalities": model.get("modalities", ["text"]),
        "features": deepcopy(model.get("features", ["TextGeneration"])),
        "engine": str(service.get("engine", model.get("engine", "VLLM"))).upper(),
        "memory_class_gib": model.get("memory_class_gib"),
        "min_vram_gib_per_replica": model.get("min_vram_gib_per_replica", 0),
        "context_window": model.get("context_window"),
        "tokenizer_name": service.get("tokenizer_name", model.get("tokenizer_name", service.get("logical_model_name", model_key))),
        "notes": deepcopy(model.get("notes", [])) + deepcopy(service.get("notes", [])),
        "audit_notes": deepcopy(service.get("audit_notes", [])) + deepcopy(model.get("caveats", [])),
        "tags": deepcopy(service.get("tags", [])),
        "gpu_indices": gpu_indices,
        "tensor_parallel_size": tp,
        "data_parallel_size": dp,
        "resource_profile": service.get("resource_profile", model.get("resource_profile", "")),
        "min_replicas": int(service.get("min_replicas", model.get("defaults", {}).get("min_replicas", 0))),
        "max_replicas": int(service.get("max_replicas", model.get("defaults", {}).get("max_replicas", 1))),
        "priority_class_name": service.get("priority_class_name", model.get("priority_class_name")),
        "max_model_len": int(_runtime_value(service, model, "max_model_len", 32768)),
        "gpu_memory_utilization": float(_runtime_value(service, model, "gpu_memory_utilization", 0.9)),
        "enable_prefix_caching": bool(_runtime_value(service, model, "enable_prefix_caching", True)),
        "max_num_batched_tokens": int(_runtime_value(service, model, "max_num_batched_tokens", 8192)),
        "max_num_seqs": int(_runtime_value(service, model, "max_num_seqs", 16)),
        "thinking_history_policy": model.get("thinking_history_policy", "keep_final_only"),
        "placement": placement,
        "topology": topology,
        "placement_error": placement_error,
        "enable_auto_tool_choice": enable_auto_tool_choice,
        "tool_call_parser": tool_call_parser,
        "extra_args": deepcopy(service.get("extra_args", model.get("defaults", {}).get("extra_args", []))),
        "benchmark_transport": benchmark_transport,
    }


def _resolve_router_aliases(profile: dict[str, Any], services: list[dict[str, Any]]) -> dict[str, str]:
    explicit = deepcopy(profile.get("router", {}).get("aliases", {}))
    if explicit:
        return explicit
    aliases: dict[str, str] = {}
    for service in services:
        for alias in service.get("served_aliases", []):
            aliases[alias] = service["service_name"]
    return aliases


def resolve(
    root: Path,
    config: dict[str, Any],
    inventory: dict[str, Any] | None = None,
    profile_name: str | None = None,
) -> dict[str, Any]:
    raw_catalogs = merged_catalogs(root, config)
    models = normalize_model_catalog(raw_catalogs.get("models", {}))
    raw_profiles = {**deepcopy(raw_catalogs.get("profiles", {})), **deepcopy(config.get("profiles", {}))}
    profiles = normalize_profile_catalog(raw_profiles, models)
    catalogs = {"models": models, "profiles": profiles}
    inventory = deepcopy(inventory) if inventory is not None else detect_inventory()
    effective_profile_name = canonical_profile_name(profile_name or config.get("active_profile"))
    if effective_profile_name not in profiles:
        raise KeyError(f"Unknown profile: {effective_profile_name}")
    profile = deepcopy(profiles[effective_profile_name])
    if profile.get("kind") == "invalid-profile":
        raise KeyError(f"Profile {effective_profile_name!r} is invalid: {profile.get('catalog_error', 'unknown error')}")

    merged_policy = {**deepcopy(config.get("policy", {})), **deepcopy(profile.get("policy", {}))}
    backend = str(config.get("backend", "compose")).lower()
    used: set[int] = set()
    services = []
    for service in profile.get("services", []):
        normalized_service = deepcopy(service)
        normalized_service["profile_name"] = profile["name"]
        service_used = set(used) if backend == "compose" else set()
        resolved_service = _resolve_service(normalized_service, catalogs.get("models", {}), inventory, merged_policy, service_used)
        if backend == "compose":
            used.update(resolved_service.get("gpu_indices", []))
        services.append(resolved_service)

    aliases = _resolve_router_aliases(profile, services)
    primary_service = services[0] if services else {}

    serving_profile = {
        "name": profile["name"],
        "public_name": profile["public_name"],
        "kind": profile.get("kind", "serving-profile"),
        "description": profile.get("description", ""),
        "base_model": profile.get("base_model", primary_service.get("model_ref", "")),
        "logical_model_name": profile.get("logical_model_name", primary_service.get("logical_model_name", "")),
        "served_model_name": profile.get("served_model_name", primary_service.get("served_model_name", "")),
        "served_aliases": deepcopy(primary_service.get("served_aliases", profile.get("served_aliases", []))),
        "protocol_mode": profile.get("protocol_mode", primary_service.get("protocol_mode", "chat")),
        "engine": profile.get("engine", primary_service.get("engine", "VLLM")),
        "resource_profile": profile.get("resource_profile", primary_service.get("resource_profile", "")),
        "service_name": profile.get("service_name", primary_service.get("service_name", "")),
        "kubernetes_name": profile.get("kubernetes_name", primary_service.get("kubernetes_name", "")),
        "tags": deepcopy(profile.get("tags", [])),
        "audit_notes": deepcopy(profile.get("audit_notes", [])),
        "notes": deepcopy(profile.get("notes", [])),
        "benchmark_transport": deepcopy(profile.get("benchmark_transport", {})),
    }

    if backend == "kubeai":
        resource_profiles, resource_profiles_values, resource_profiles_path = load_kubeai_resource_profiles(root)
        if resource_profiles:
            resource_profile_source = str(resource_profiles_path)
        else:
            resource_profiles = deepcopy(config.get("resource_profiles", {}))
            resource_profiles_values = deepcopy({"resourceProfiles": resource_profiles_to_kubeai_values(resource_profiles)["resourceProfiles"]})
            resource_profile_source = "config.yaml.resource_profiles"
    else:
        resource_profiles = deepcopy(config.get("resource_profiles", {}))
        resource_profiles_values = deepcopy({"resourceProfiles": resource_profiles_to_kubeai_values(resource_profiles)["resourceProfiles"]})
        resource_profile_source = "config.yaml.resource_profiles"

    return {
        "schema_version": 4,
        "source": {
            "config_file": "config.yaml",
            "active_profile": effective_profile_name,
        },
        "backend": backend,
        "images": deepcopy(config.get("images", {})),
        "ports": deepcopy(config.get("ports", {})),
        "policy": merged_policy,
        "vllm": {
            "enable_responses_api_store": bool(profile.get("vllm", {}).get("enable_responses_api_store", False)),
            "logging_level": str(profile.get("vllm", {}).get("logging_level", "INFO")),
        },
        "state": normalized_state(root, config.get("state", {})),
        "cluster": normalized_cluster(config.get("cluster", {})),
        "resource_profiles": resource_profiles,
        "resource_profiles_values": resource_profiles_values,
        "resource_profiles_source": resource_profile_source,
        "inventory": inventory,
        "profile": serving_profile,
        "serving_profile": deepcopy(serving_profile),
        "services": services,
        "router": {
            "enabled": True,
            "type": "litellm" if backend == "compose" else "kubeai",
            "aliases": aliases,
        },
        "open_webui": {
            "enabled": backend == "compose",
            "auth": True,
        },
    }
