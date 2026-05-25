from __future__ import annotations

from copy import deepcopy
from typing import Any

from .catalog import canonical_profile_name, normalize_ollama_models, normalize_stack_profiles, normalize_vllm_models, sanitize_name
from .config import (
    load_kubeai_resource_profiles,
    merged_catalogs,
    normalized_cluster,
    normalized_output,
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


def _runtime_value(runtime: dict[str, Any], model: dict[str, Any], key: str, default: Any) -> Any:
    runtime_cfg = runtime.get("runtime", {}) or {}
    if key in runtime_cfg:
        return runtime_cfg[key]
    if key in runtime:
        return runtime[key]
    return model.get("defaults", {}).get(key, default)


def _enabled_value(value: Any, default: bool = False) -> bool:
    if value == "auto":
        return default
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _resolve_gpu_indices(
    *,
    name: str,
    placement: dict[str, Any],
    topology: dict[str, Any],
    preferred_gpu_count: int,
    available: list[int],
) -> tuple[list[int], str | None]:
    strategy = placement.get("strategy", "first_fit")
    if strategy in {"exact", "multi_gpu", "single_gpu"}:
        gpu_indices = list(placement.get("gpu_indices", []))
        if not gpu_indices:
            return [], f"{name} uses {strategy} placement but no gpu_indices were provided"
        return gpu_indices, None
    gpu_count = int(placement.get("gpu_count", topology.get("tensor_parallel_size", preferred_gpu_count) or 1))
    return _first_fit(available, gpu_count)


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _resolve_vllm_runtime(
    *,
    profile: dict[str, Any],
    runtime_name: str,
    runtime: dict[str, Any],
    models: dict[str, Any],
    inventory: dict[str, Any],
    policy: dict[str, Any],
    used: set[int],
    backend: str,
) -> dict[str, Any]:
    model_key = runtime.get("model") or runtime.get("base_model")
    if not model_key and runtime.get("hf_model_id"):
        model_key = sanitize_name(runtime["hf_model_id"])
        models = dict(models)
        models[model_key] = {
            "key": model_key,
            "hf_model_id": runtime["hf_model_id"],
            "url": f"hf://{runtime['hf_model_id']}",
            "served_model_name": runtime.get("served_model_name") or model_key,
            "logical_model_name": runtime.get("logical_model_name") or model_key,
            "tokenizer_name": runtime.get("tokenizer_name") or model_key,
            "supported_protocols": ["chat", "completions"],
            "modalities": ["text"],
            "features": ["TextGeneration"],
            "defaults": {},
            "preferred_gpu_count": 1,
            "min_vram_gib_per_replica": 0,
            "resource_profile": "",
            "notes": [],
            "caveats": [],
        }
    if model_key not in models:
        raise KeyError(f"Unknown vLLM model: {model_key}")
    model = deepcopy(models[model_key])
    placement = deepcopy(runtime.get("placement", {}))
    topology = deepcopy(runtime.get("topology", {}))
    available = [i for i in _available_gpu_indices(inventory, policy.get("reserve_display_gpu", "auto")) if i not in used]
    if "tp" in topology and "tensor_parallel_size" not in topology:
        topology["tensor_parallel_size"] = topology["tp"]
    if "dp" in topology and "data_parallel_size" not in topology:
        topology["data_parallel_size"] = topology["dp"]
    gpu_indices, placement_error = _resolve_gpu_indices(
        name=f"vLLM runtime {runtime_name}",
        placement=placement,
        topology=topology,
        preferred_gpu_count=int(model.get("preferred_gpu_count", 1) or 1),
        available=available,
    )
    if backend == "compose":
        used.update(gpu_indices)
    tp = int(topology.get("tensor_parallel_size", max(1, len(gpu_indices) or placement.get("gpu_count", 1))))
    dp = int(topology.get("data_parallel_size", 1))

    tool_calling = _merge(model.get("tool_calling", {}), runtime.get("tool_calling", {}))
    tool_call_parser = tool_calling.get("parser")
    tool_calling_on = bool(tool_calling.get("enabled", tool_calling.get("auto", False)))
    enable_auto_tool_choice = bool(tool_calling_on and tool_call_parser)

    reasoning = _merge(model.get("reasoning", {}), runtime.get("reasoning", {}))
    reasoning_enabled = bool(reasoning.get("enabled", False))
    reasoning_parser = reasoning.get("parser") if reasoning_enabled else None
    reasoning_expose_to_openwebui = bool(reasoning.get("expose_to_openwebui", reasoning_enabled))

    chat_compat = _merge(model.get("chat_compat", {}), runtime.get("chat_compat", {}))
    chat_compat_enabled = bool(chat_compat.get("enabled", False))
    chat_compat_strategy = str(chat_compat.get("strategy", "flat_messages")) if chat_compat_enabled else None

    hf_model_id = runtime.get("hf_model_id", model.get("hf_model_id", ""))
    served_model_name = runtime.get("served_model_name") or model.get("served_model_name") or runtime_name
    logical_model_name = runtime.get("logical_model_name") or model.get("logical_model_name") or served_model_name
    public_name = runtime.get("public_name") or runtime_name
    protocol_mode = runtime.get("protocol_mode") or runtime.get("protocol") or "chat"
    model_url = runtime.get("url") or model.get("url") or (f"hf://{hf_model_id}" if hf_model_id else "")

    return {
        "provider": "vllm",
        "runtime_name": runtime_name,
        "service_name": sanitize_name(runtime_name),
        "compose_service_name": f"vllm-{sanitize_name(runtime_name)}",
        "container_name": f"vllm-{sanitize_name(runtime_name)}",
        "profile_name": profile["name"],
        "profile_public_name": public_name,
        "kubernetes_name": sanitize_name(public_name),
        "model_ref": model_key,
        "hf_model_id": hf_model_id,
        "model_url": model_url,
        "logical_model_name": logical_model_name,
        "served_model_name": served_model_name,
        "served_aliases": [],
        "protocol_mode": protocol_mode,
        "supported_protocols": list(model.get("supported_protocols", ["chat", "completions"])),
        "modalities": model.get("modalities", ["text"]),
        "features": deepcopy(model.get("features", ["TextGeneration"])),
        "engine": "VLLM",
        "memory_class_gib": model.get("memory_class_gib"),
        "min_vram_gib_per_replica": model.get("min_vram_gib_per_replica", 0),
        "context_window": model.get("context_window"),
        "tokenizer_name": runtime.get("tokenizer_name", model.get("tokenizer_name", logical_model_name)),
        "notes": deepcopy(model.get("notes", [])) + deepcopy(runtime.get("notes", [])),
        "audit_notes": deepcopy(runtime.get("audit_notes", [])) + deepcopy(model.get("caveats", [])),
        "tags": deepcopy(runtime.get("tags", [])),
        "gpu_indices": gpu_indices,
        "tensor_parallel_size": tp,
        "data_parallel_size": dp,
        "resource_profile": runtime.get("resource_profile", model.get("resource_profile", "")),
        "min_replicas": int(runtime.get("min_replicas", model.get("defaults", {}).get("min_replicas", 0))),
        "max_replicas": int(runtime.get("max_replicas", model.get("defaults", {}).get("max_replicas", 1))),
        "priority_class_name": runtime.get("priority_class_name", model.get("priority_class_name")),
        "max_model_len": int(_runtime_value(runtime, model, "max_model_len", 32768)),
        "gpu_memory_utilization": float(_runtime_value(runtime, model, "gpu_memory_utilization", 0.9)),
        "enable_prefix_caching": bool(_runtime_value(runtime, model, "enable_prefix_caching", True)),
        "max_num_batched_tokens": int(_runtime_value(runtime, model, "max_num_batched_tokens", 8192)),
        "max_num_seqs": int(_runtime_value(runtime, model, "max_num_seqs", 16)),
        "thinking_history_policy": model.get("thinking_history_policy", "keep_final_only"),
        "placement": placement,
        "topology": topology,
        "placement_error": placement_error,
        "enable_auto_tool_choice": enable_auto_tool_choice,
        "tool_call_parser": tool_call_parser,
        "extra_args": deepcopy(runtime.get("extra_args", model.get("defaults", {}).get("extra_args", []))),
        "reasoning_enabled": reasoning_enabled,
        "reasoning_parser": reasoning_parser,
        "reasoning_expose_to_openwebui": reasoning_expose_to_openwebui,
        "chat_compat_enabled": chat_compat_enabled,
        "chat_compat_strategy": chat_compat_strategy,
        "benchmark_transport": deepcopy(runtime.get("benchmark_transport", runtime.get("transport", {}))),
        "publish_port": bool(runtime.get("publish_port", False)),
        "host_port": runtime.get("host_port"),
    }


def _resolve_ollama_model_tag(model_ref: str, ollama_models: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if model_ref in ollama_models:
        model = deepcopy(ollama_models[model_ref])
        return model["tag"], model
    return str(model_ref), {"key": model_ref, "tag": str(model_ref), "served_model_name": sanitize_name(str(model_ref)), "defaults": {}}


def _collect_ollama_needed(profile: dict[str, Any]) -> bool:
    p = profile.get("providers", {}).get("ollama", {}) or {}
    if _enabled_value(p.get("enabled"), default=False):
        return True
    for route in (profile.get("routes", {}) or {}).values():
        if str(route.get("provider", "")).lower() == "ollama":
            return True
    return False


def _resolve_ollama_provider(profile: dict[str, Any], config: dict[str, Any], inventory: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    profile_ollama = deepcopy(profile.get("providers", {}).get("ollama", {}) or {})
    config_ollama = deepcopy(config.get("ollama", {}) or {})
    merged = _merge(config_ollama, profile_ollama)
    enabled = _enabled_value(merged.get("enabled"), default=_collect_ollama_needed(profile))
    if not enabled:
        return {"enabled": False, "routes": {}}

    gpu_indices = merged.get("gpu_indices", "auto")
    placement_error = None
    if gpu_indices == "auto" or gpu_indices is None:
        available = _available_gpu_indices(inventory, policy.get("reserve_display_gpu", "auto"))
        gpu_indices = available
    else:
        gpu_indices = list(gpu_indices)
    return {
        "enabled": True,
        "service_name": "ollama",
        "base_url": "http://ollama:11434",
        "host_port": int(config.get("ports", {}).get("ollama", 11434)),
        "gpu_indices": gpu_indices,
        "publish_port": bool(merged.get("publish_port", False)),
        "host": str(merged.get("host", "0.0.0.0:11434")),
        "keep_alive": str(merged.get("keep_alive", "2m")),
        "context_length": int(merged.get("context_length", 4096)),
        "num_parallel": int(merged.get("num_parallel", 1)),
        "max_loaded_models": int(merged.get("max_loaded_models", 1)),
        "max_queue": int(merged.get("max_queue", 8)),
        "placement_error": placement_error,
        "routes": {},
    }


def _resolve_routes(profile: dict[str, Any], vllm_runtimes: dict[str, Any], ollama_models: dict[str, Any]) -> dict[str, Any]:
    routes: dict[str, Any] = {}
    for alias, raw in (profile.get("routes", {}) or {}).items():
        route = deepcopy(raw) or {}
        provider = str(route.get("provider", "vllm")).lower()
        aliases = route.get("aliases") or [alias]
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [str(a) for a in aliases]
        if alias not in aliases:
            aliases.insert(0, alias)
        for public_alias in aliases:
            if provider == "vllm":
                runtime_name = route.get("runtime") or route.get("service") or route.get("target")
                if runtime_name is None and len(vllm_runtimes) == 1:
                    runtime_name = next(iter(vllm_runtimes))
                if runtime_name not in vllm_runtimes:
                    raise KeyError(f"route {alias!r} references unknown vLLM runtime {runtime_name!r}")
                rt = vllm_runtimes[runtime_name]
                routes[public_alias] = {
                    "alias": public_alias,
                    "provider": "vllm",
                    "runtime": runtime_name,
                    "service_name": rt["service_name"],
                    "upstream_service_name": rt.get("compose_service_name", rt["service_name"]),
                    "served_model_name": rt["served_model_name"],
                    "protocol_mode": rt["protocol_mode"],
                    "max_model_len": rt["max_model_len"],
                    "chat_compat_enabled": rt.get("chat_compat_enabled", False),
                    "chat_compat_strategy": rt.get("chat_compat_strategy"),
                }
            elif provider == "ollama":
                model_ref = route.get("model") or route.get("tag") or public_alias
                tag, model = _resolve_ollama_model_tag(str(model_ref), ollama_models)
                defaults = deepcopy(model.get("defaults", {}))
                defaults.update(deepcopy(route.get("options", {})))
                max_model_len = int(route.get("max_model_len") or defaults.get("num_ctx") or model.get("context_window") or 4096)
                routes[public_alias] = {
                    "alias": public_alias,
                    "provider": "ollama",
                    "upstream_model": tag,
                    "model_ref": str(model_ref),
                    "protocol_mode": "chat",
                    "max_model_len": max_model_len,
                    "options": defaults,
                }
            else:
                raise KeyError(f"route {alias!r} uses unknown provider {provider!r}")
    return routes


def _resolve_litellm(profile: dict[str, Any], routes: dict[str, Any], providers: dict[str, Any], backend: str) -> dict[str, Any]:
    if backend == "kubeai":
        return {"enabled": False, "base_url": "", "routes": {}}
    cfg = deepcopy(profile.get("gateways", {}).get("litellm", {}) or {})
    route_providers = {route.get("provider") for route in routes.values()}
    needs_gateway = bool(routes) and (backend == "compose")
    if len(route_providers) > 1:
        needs_gateway = True
    enabled = _enabled_value(cfg.get("enabled"), default=needs_gateway)
    return {
        "enabled": bool(enabled),
        "base_url": "http://litellm:4000/v1" if enabled else "",
        "routes": routes if enabled else {},
    }


def _resolve_open_webui(profile: dict[str, Any], gateways: dict[str, Any], providers: dict[str, Any], backend: str, config: dict[str, Any]) -> dict[str, Any]:
    if backend == "kubeai":
        return {"enabled": False, "auth": False, "provider": "none"}
    raw = deepcopy(config.get("open_webui", {}) or {})
    raw = _merge(raw, profile.get("frontends", {}).get("open_webui", {}) or {})
    default_enabled = backend == "compose" and (gateways.get("litellm", {}).get("enabled") or providers.get("ollama", {}).get("enabled"))
    enabled = _enabled_value(raw.get("enabled"), default=bool(default_enabled))
    provider = str(raw.get("provider", "auto"))
    if provider == "auto":
        if gateways.get("litellm", {}).get("enabled"):
            provider = "litellm"
        elif providers.get("ollama", {}).get("enabled"):
            provider = "ollama"
        else:
            provider = "none"
    return {"enabled": enabled, "auth": bool(raw.get("auth", False)), "provider": provider}


def _serving_profile(profile: dict[str, Any], vllm_runtimes: dict[str, Any], routes: dict[str, Any]) -> dict[str, Any]:
    first_rt = next(iter(vllm_runtimes.values()), {}) if vllm_runtimes else {}
    first_route = next(iter(routes.values()), {}) if routes else {}
    public = first_route.get("alias") or first_rt.get("profile_public_name") or profile["name"]
    return {
        "name": profile["name"],
        "public_name": public,
        "kind": profile.get("kind", "stack"),
        "description": profile.get("description", ""),
        "base_model": first_rt.get("model_ref", first_route.get("model_ref", "")),
        "logical_model_name": first_rt.get("logical_model_name", first_route.get("alias", "")),
        "served_model_name": first_rt.get("served_model_name", first_route.get("upstream_model", "")),
        "served_aliases": list(routes.keys()),
        "protocol_mode": first_rt.get("protocol_mode", first_route.get("protocol_mode", "chat")),
        "engine": "mixed" if vllm_runtimes and any(r.get("provider") == "ollama" for r in routes.values()) else ("VLLM" if vllm_runtimes else "OLLAMA"),
        "resource_profile": first_rt.get("resource_profile", ""),
        "service_name": first_rt.get("service_name", ""),
        "kubernetes_name": first_rt.get("kubernetes_name", sanitize_name(profile["name"])),
        "tags": deepcopy(profile.get("tags", [])),
        "audit_notes": deepcopy(profile.get("audit_notes", [])),
        "notes": deepcopy(profile.get("notes", [])),
        "benchmark_transport": {},
    }


def _resolve_access(deployment: dict[str, Any]) -> dict[str, Any]:
    ports = deployment.get("ports", {})
    access: dict[str, Any] = {}
    litellm = deployment["gateways"]["litellm"]
    ollama = deployment["providers"]["ollama"]
    vllm_runtimes = deployment["providers"]["vllm"].get("runtimes", {})
    if deployment.get("backend") == "kubeai":
        ingress = deployment.get("cluster", {}).get("ingress", {}) or {}
        base = f"http://{ingress['host']}/openai/v1" if ingress.get("enabled") and ingress.get("host") else "http://127.0.0.1:8000/openai/v1"
        access["default"] = {"kind": "openai-compatible", "base_url": base, "auth_env_name": "KUBEAI_OPENAI_API_KEY"}
    elif litellm.get("enabled"):
        access["default"] = {"kind": "openai-compatible", "base_url": f"http://127.0.0.1:{ports.get('litellm', 14042)}/v1", "auth_env_name": "LITELLM_MASTER_KEY"}
    elif ollama.get("enabled"):
        access["default"] = {"kind": "ollama-native", "base_url": f"http://127.0.0.1:{ports.get('ollama', 11434)}", "auth_required": False}
        access["ollama_openai"] = {"kind": "openai-compatible", "base_url": f"http://127.0.0.1:{ports.get('ollama', 11434)}/v1", "auth_required": False}
    elif vllm_runtimes:
        name, rt = next(iter(vllm_runtimes.items()))
        port = rt.get("host_port") or 18000
        access["default"] = {"kind": "openai-compatible", "base_url": f"http://127.0.0.1:{port}/v1", "auth_env_name": "VLLM_BACKEND_API_KEY"}
    else:
        access["default"] = {"kind": "none", "base_url": ""}
    for idx, (name, rt) in enumerate(vllm_runtimes.items()):
        port = rt.get("host_port") or (18000 + idx)
        access[f"vllm_{name}"] = {"kind": "openai-compatible", "base_url": f"http://127.0.0.1:{port}/v1", "auth_env_name": "VLLM_BACKEND_API_KEY"}
    return access


def resolve(config: dict[str, Any], inventory: dict[str, Any] | None = None, profile_name: str | None = None) -> dict[str, Any]:
    raw_catalogs = merged_catalogs(config)
    vllm_models = normalize_vllm_models(raw_catalogs.get("vllm_models", raw_catalogs.get("models", {})))
    ollama_models = normalize_ollama_models(raw_catalogs.get("ollama_models", {}))
    profiles = normalize_stack_profiles(raw_catalogs.get("profiles", {}), vllm_models, ollama_models)
    inventory = deepcopy(inventory) if inventory is not None else detect_inventory()
    effective_profile_name = canonical_profile_name(profile_name or config.get("active_profile"))
    if effective_profile_name not in profiles:
        raise KeyError(f"Unknown profile: {effective_profile_name}")
    profile = deepcopy(profiles[effective_profile_name])
    if profile.get("kind") == "invalid-profile":
        raise KeyError(f"Profile {effective_profile_name!r} is invalid: {profile.get('catalog_error', 'unknown error')}")

    backend = str(config.get("backend", "compose")).lower()
    merged_policy = _merge(config.get("policy", {}), profile.get("policy", {}))
    images = deepcopy(config.get("images", {}))
    ports = deepcopy(config.get("ports", {}))
    state = normalized_state(config.get("state", {}))
    output = normalized_output(config.get("output", {}))

    used: set[int] = set()
    vllm_cfg = deepcopy(profile.get("providers", {}).get("vllm", {}) or {})
    runtime_cfgs = deepcopy(vllm_cfg.get("runtimes", {}) or {})
    vllm_runtimes: dict[str, Any] = {}
    for runtime_name, runtime in runtime_cfgs.items():
        resolved_rt = _resolve_vllm_runtime(
            profile=profile,
            runtime_name=str(runtime_name),
            runtime=runtime,
            models=vllm_models,
            inventory=inventory,
            policy=merged_policy,
            used=used,
            backend=backend,
        )
        vllm_runtimes[str(runtime_name)] = resolved_rt

    ollama_provider = _resolve_ollama_provider(profile, config, inventory, merged_policy)
    routes = _resolve_routes(profile, vllm_runtimes, ollama_models)
    for alias, route in routes.items():
        if route.get("provider") == "vllm" and route.get("runtime") in vllm_runtimes:
            aliases = vllm_runtimes[route["runtime"]].setdefault("served_aliases", [])
            if alias not in aliases:
                aliases.append(alias)
    ollama_routes = {k: v for k, v in routes.items() if v.get("provider") == "ollama"}
    if ollama_routes and not ollama_provider.get("enabled"):
        ollama_provider = _resolve_ollama_provider({**profile, "providers": {**profile.get("providers", {}), "ollama": {"enabled": True}}}, config, inventory, merged_policy)
    ollama_provider["routes"] = ollama_routes

    providers = {
        "ollama": ollama_provider,
        "vllm": {"enabled": bool(vllm_runtimes), "runtimes": vllm_runtimes},
    }
    litellm = _resolve_litellm(profile, routes, providers, backend)
    gateways = {"litellm": litellm}
    frontends = {"open_webui": _resolve_open_webui(profile, gateways, providers, backend, config)}
    serving_profile = _serving_profile(profile, vllm_runtimes, routes)

    if backend == "kubeai":
        resource_profiles, resource_profiles_values, resource_profiles_path = load_kubeai_resource_profiles()
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

    # Compatibility aliases. New templates should use providers/gateways/frontends.
    services = list(vllm_runtimes.values())
    router_aliases = {alias: route.get("service_name", route.get("upstream_model", "")) for alias, route in routes.items()}

    deployment = {
        "schema_version": 5,
        "source": {"config_file": "config.yaml", "active_profile": effective_profile_name},
        "backend": backend,
        "images": images,
        "ports": ports,
        "policy": merged_policy,
        "vllm": {
            "enable_responses_api_store": bool(profile.get("vllm", {}).get("enable_responses_api_store", False)),
            "logging_level": str(profile.get("vllm", {}).get("logging_level", "INFO")),
        },
        "state": state,
        "output": output,
        "cluster": normalized_cluster(config.get("cluster", {})),
        "resource_profiles": resource_profiles,
        "resource_profiles_values": resource_profiles_values,
        "resource_profiles_source": resource_profile_source,
        "inventory": inventory,
        "providers": providers,
        "gateways": gateways,
        "frontends": frontends,
        "access": {},
        "profile": serving_profile,
        "serving_profile": deepcopy(serving_profile),
        "services": services,
        "router": {"enabled": litellm.get("enabled"), "type": "litellm" if litellm.get("enabled") else "none", "aliases": router_aliases},
        "open_webui": frontends["open_webui"],
    }
    deployment["access"] = _resolve_access(deployment)
    return deployment
