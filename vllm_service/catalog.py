from __future__ import annotations

from copy import deepcopy
from typing import Any


PROFILE_NAME_ALIASES = {
    "helm-qwen2-72b-instruct": "qwen2-72b-instruct-tp2-balanced",
    "helm-qwen2.5-7b-instruct": "qwen2-5-7b-instruct-turbo-default",
    "helm-qwen2.5-72b-instruct": "qwen2-5-72b-instruct-tp2-balanced",
    "helm-gpt-oss-20b": "gpt-oss-20b-completions",
    "helm-vicuna-7b-v1.3": "vicuna-7b-v1-3-no-chat-template",
}


def sanitize_name(value: str) -> str:
    value = value.strip().lower()
    out = []
    prev_dash = False
    for char in value:
        if char.isalnum():
            out.append(char)
            prev_dash = False
            continue
        if not prev_dash:
            out.append("-")
            prev_dash = True
    sanitized = "".join(out).strip("-")
    return sanitized or "profile"


def canonical_profile_name(name: str) -> str:
    return PROFILE_NAME_ALIASES.get(name, name)


def normalize_model_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, raw in (catalog or {}).items():
        entry = deepcopy(raw)
        hf_model_id = entry.get("hf_model_id", "")
        url = entry.get("url") or (f"hf://{hf_model_id}" if hf_model_id else "")
        defaults = deepcopy(entry.get("defaults", {}))
        notes = list(entry.get("notes", []) or [])
        caveats = list(entry.get("caveats", []) or [])
        supported_protocols = entry.get("supported_protocols")
        if supported_protocols is None:
            supported_protocols = ["chat", "completions"]
        else:
            supported_protocols = [str(p) for p in supported_protocols]
        normalized[key] = {
            "key": key,
            "canonical_key": sanitize_name(entry.get("canonical_key", key)),
            "hf_model_id": hf_model_id,
            "url": url,
            "family": entry.get("family", ""),
            "modalities": list(entry.get("modalities", ["text"])),
            "supported_protocols": supported_protocols,
            "tokenizer_name": entry.get("tokenizer_name") or entry.get("tokenizer") or entry.get("served_model_name") or key,
            "logical_model_name": entry.get("logical_model_name") or entry.get("served_model_name") or key,
            "served_model_name": entry.get("served_model_name") or entry.get("logical_model_name") or key,
            "memory_class_gib": entry.get("memory_class_gib"),
            "min_vram_gib_per_replica": entry.get("min_vram_gib_per_replica", 0),
            "preferred_gpu_count": entry.get("preferred_gpu_count", 1),
            "context_window": entry.get("context_window"),
            "defaults": defaults,
            "engine": str(entry.get("engine", "VLLM")).upper(),
            "resource_profile": entry.get("resource_profile", ""),
            "priority_class_name": entry.get("priority_class_name"),
            "tool_calling": deepcopy(entry.get("tool_calling", {})),
            "thinking_history_policy": entry.get("thinking_history_policy", "keep_final_only"),
            "features": deepcopy(entry.get("features", ["TextGeneration"])),
            "safe_defaults": deepcopy(entry.get("safe_defaults", defaults)),
            "notes": notes,
            "caveats": caveats,
        }
    return normalized


def _infer_protocol_mode(profile_name: str, logical_model_name: str, raw: dict[str, Any]) -> str:
    explicit = raw.get("protocol_mode") or raw.get("protocol")
    if explicit:
        return str(explicit)
    tags = {str(tag) for tag in raw.get("tags", [])}
    if "completions" in tags:
        return "completions"
    if "chat" in tags:
        return "chat"
    name = f"{profile_name} {logical_model_name}".lower()
    if "completion" in name:
        return "completions"
    return "chat"


def _served_aliases(public_name: str, logical_model_name: str, served_model_name: str, aliases: list[str] | None = None) -> list[str]:
    ordered: list[str] = []
    for candidate in [public_name, logical_model_name, served_model_name, *(aliases or [])]:
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _normalize_service_from_profile(public_name: str, raw: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    logical_model_name = (
        raw.get("logical_model_name")
        or raw.get("helm_model_name")
        or raw.get("model_name")
        or model.get("logical_model_name")
        or model["served_model_name"]
    )
    served_model_name = raw.get("served_model_name") or logical_model_name
    protocol_mode = _infer_protocol_mode(public_name, logical_model_name, raw)
    return {
        "service_name": sanitize_name(raw.get("service_name") or raw.get("runtime_service_name") or public_name),
        "base_model": raw.get("base_model") or raw.get("model"),
        "public_name": public_name,
        "kubernetes_name": sanitize_name(public_name),
        "logical_model_name": logical_model_name,
        "served_model_name": served_model_name,
        "served_aliases": _served_aliases(public_name, logical_model_name, served_model_name, raw.get("served_aliases")),
        "protocol_mode": protocol_mode,
        "engine": str(raw.get("engine", model.get("engine", "VLLM"))).upper(),
        "resource_profile": raw.get("resource_profile", model.get("resource_profile", "")),
        "placement": deepcopy(raw.get("placement", {})),
        "topology": deepcopy(raw.get("topology", {})),
        "runtime": deepcopy(raw.get("runtime", {})),
        "extra_args": deepcopy(raw.get("extra_args", [])),
        "min_replicas": int(raw.get("min_replicas", model.get("defaults", {}).get("min_replicas", 0))),
        "max_replicas": int(raw.get("max_replicas", model.get("defaults", {}).get("max_replicas", 1))),
        "priority_class_name": raw.get("priority_class_name", model.get("priority_class_name")),
        "tags": list(raw.get("tags", []) or []),
        "audit_notes": list(raw.get("audit_notes", []) or []),
        "notes": list(raw.get("notes", []) or []),
        "benchmark_transport": deepcopy(raw.get("benchmark_transport", raw.get("transport", {}))),
    }


def _normalize_legacy_profile(name: str, raw: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    services = deepcopy(raw.get("services", []))
    aliases = deepcopy(raw.get("router", {}).get("aliases", {}))
    normalized_services = []
    for index, service in enumerate(services):
        service_name = sanitize_name(service.get("service_name") or service.get("name") or f"{name}-{index + 1}")
        model_key = service.get("base_model") or service.get("model")
        if model_key not in models:
            raise KeyError(f"Unknown model: {model_key}")
        model = models[model_key]
        served_model_name = service.get("served_model_name") or model["served_model_name"]
        logical_model_name = service.get("logical_model_name") or next(
            (alias for alias, target in aliases.items() if target == (service.get("service_name") or service.get("name"))),
            served_model_name,
        )
        public_name = sanitize_name(service.get("public_name") or (name if len(services) == 1 else f"{name}-{service_name}"))
        protocol_mode = _infer_protocol_mode(public_name, logical_model_name, service)
        normalized_services.append(
            {
                "service_name": service_name,
                "base_model": model_key,
                "public_name": public_name,
                "kubernetes_name": sanitize_name(public_name),
                "logical_model_name": logical_model_name,
                "served_model_name": served_model_name,
                "served_aliases": _served_aliases(public_name, logical_model_name, served_model_name, list(aliases.keys())),
                "protocol_mode": protocol_mode,
                "engine": str(service.get("engine", model.get("engine", "VLLM"))).upper(),
                "resource_profile": service.get("resource_profile", model.get("resource_profile", "")),
                "placement": deepcopy(service.get("placement", {})),
                "topology": deepcopy(service.get("topology", {})),
                "runtime": deepcopy(service.get("runtime", {})),
                "extra_args": deepcopy(service.get("extra_args", [])),
                "min_replicas": int(service.get("min_replicas", model.get("defaults", {}).get("min_replicas", 0))),
                "max_replicas": int(service.get("max_replicas", model.get("defaults", {}).get("max_replicas", 1))),
                "priority_class_name": service.get("priority_class_name", model.get("priority_class_name")),
                "tags": list(raw.get("tags", []) or []),
                "audit_notes": list(raw.get("audit_notes", []) or []),
                "notes": list(raw.get("notes", []) or []),
                "benchmark_transport": deepcopy(raw.get("benchmark_transport", raw.get("transport", {}))),
            }
        )
    primary = normalized_services[0] if normalized_services else None
    return {
        "name": name,
        "public_name": primary["public_name"] if primary else name,
        "description": raw.get("description", ""),
        "kind": "legacy-stack-profile" if len(normalized_services) > 1 else "serving-profile",
        "base_model": primary["base_model"] if primary else "",
        "logical_model_name": primary["logical_model_name"] if primary else "",
        "served_model_name": primary["served_model_name"] if primary else "",
        "served_aliases": primary["served_aliases"] if primary else [],
        "protocol_mode": primary["protocol_mode"] if primary else "chat",
        "engine": primary["engine"] if primary else "VLLM",
        "resource_profile": primary["resource_profile"] if primary else "",
        "service_name": primary["service_name"] if primary else sanitize_name(name),
        "kubernetes_name": primary["kubernetes_name"] if primary else sanitize_name(name),
        "services": normalized_services,
        "policy": deepcopy(raw.get("policy", {})),
        "vllm": deepcopy(raw.get("vllm", {})),
        "router": {"aliases": aliases},
        "benchmark_transport": deepcopy(raw.get("benchmark_transport", raw.get("transport", {}))),
        "tags": list(raw.get("tags", []) or []),
        "audit_notes": list(raw.get("audit_notes", []) or []),
        "notes": list(raw.get("notes", []) or []),
    }


def normalize_profile_catalog(catalog: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for original_name, raw in (catalog or {}).items():
        name = original_name
        profile = deepcopy(raw)
        try:
            if "services" in profile:
                normalized[name] = _normalize_legacy_profile(name, profile, models)
                continue
            model_key = profile.get("base_model") or profile.get("model")
            if model_key not in models:
                raise KeyError(f"Unknown model: {model_key}")
            model = models[model_key]
            public_name = sanitize_name(profile.get("public_name") or name)
            service = _normalize_service_from_profile(public_name, profile, model)
            normalized[name] = {
                "name": name,
                "public_name": public_name,
                "description": profile.get("description", ""),
                "kind": "serving-profile",
                "base_model": service["base_model"],
                "logical_model_name": service["logical_model_name"],
                "served_model_name": service["served_model_name"],
                "served_aliases": service["served_aliases"],
                "protocol_mode": service["protocol_mode"],
                "engine": service["engine"],
                "resource_profile": service["resource_profile"],
                "service_name": service["service_name"],
                "kubernetes_name": service["kubernetes_name"],
                "services": [service],
                "policy": deepcopy(profile.get("policy", {})),
                "vllm": deepcopy(profile.get("vllm", {})),
                "router": {"aliases": {alias: service["service_name"] for alias in service["served_aliases"]}},
                "benchmark_transport": deepcopy(profile.get("benchmark_transport", profile.get("transport", {}))),
                "tags": list(profile.get("tags", []) or []),
                "audit_notes": list(profile.get("audit_notes", []) or []),
                "notes": list(profile.get("notes", []) or []),
            }
        except KeyError as ex:
            normalized[name] = {
                "name": name,
                "public_name": sanitize_name(profile.get("public_name") or name),
                "description": profile.get("description", ""),
                "kind": "invalid-profile",
                "catalog_error": str(ex),
                "services": [],
                "policy": deepcopy(profile.get("policy", {})),
                "vllm": deepcopy(profile.get("vllm", {})),
                "router": deepcopy(profile.get("router", {})),
                "benchmark_transport": deepcopy(profile.get("benchmark_transport", profile.get("transport", {}))),
                "tags": list(profile.get("tags", []) or []),
                "audit_notes": list(profile.get("audit_notes", []) or []),
                "notes": list(profile.get("notes", []) or []),
            }
    return normalized


def profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": profile["name"],
        "public_name": profile["public_name"],
        "kind": profile["kind"],
        "base_model": profile.get("base_model", ""),
        "logical_model_name": profile.get("logical_model_name", ""),
        "served_model_name": profile.get("served_model_name", ""),
        "protocol_mode": profile.get("protocol_mode", "chat"),
        "engine": profile.get("engine", "VLLM"),
        "resource_profile": profile.get("resource_profile", ""),
        "description": profile.get("description", ""),
        "tags": profile.get("tags", []),
    }
