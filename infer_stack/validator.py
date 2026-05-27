from __future__ import annotations

from typing import Any


def validate_resolved(resolved: dict[str, Any]) -> dict[str, Any]:
    inventory = resolved.get("inventory", {})
    gpu_map = {g["index"]: g for g in inventory.get("gpus", [])}
    policy = resolved.get("policy", {})
    backend = resolved.get("backend", "compose")
    errors: list[str] = []
    warnings: list[str] = []
    used_ports: set[int] = set()

    providers = resolved.get("providers", {}) or {}
    gateways = resolved.get("gateways", {}) or {}
    frontends = resolved.get("frontends", {}) or {}
    ollama = providers.get("ollama", {}) or {}
    vllm = providers.get("vllm", {}) or {}
    vllm_runtimes = vllm.get("runtimes", {}) or {}
    litellm = gateways.get("litellm", {}) or {}
    open_webui = frontends.get("open_webui", {}) or {}
    routes = (litellm.get("routes") or {}) if litellm.get("enabled") else {}

    if backend == "kubeai":
        if ollama.get("enabled"):
            errors.append("backend=kubeai does not support the Ollama provider yet")
        if litellm.get("enabled"):
            errors.append("backend=kubeai does not render the LiteLLM gateway yet")
        if open_webui.get("enabled"):
            errors.append("backend=kubeai does not render the Open WebUI frontend yet")
        profiles = resolved.get("resource_profiles", {})
        if not profiles:
            source = resolved.get("resource_profiles_source", "kubeai-values.local.yaml")
            errors.append(
                "No local KubeAI resource profiles were loaded for validation. "
                f"Expected them at {source}. "
                "Run `python manage.py kubeai-sync-resource-profiles --from-file values-kubeai-local-gpu.yaml` first."
            )

    if not ollama.get("enabled") and not vllm_runtimes and not open_webui.get("enabled") and not litellm.get("enabled"):
        warnings.append("resolved profile has no enabled providers, gateways, or frontends")

    if litellm.get("enabled"):
        route_providers = {route.get("provider") for route in routes.values()}
        if not routes:
            warnings.append("LiteLLM is enabled but has no routes")
        if "ollama" in route_providers and not ollama.get("enabled"):
            errors.append("LiteLLM has Ollama routes but the Ollama provider is disabled")
        if "vllm" in route_providers and not vllm_runtimes:
            errors.append("LiteLLM has vLLM routes but no vLLM runtimes are enabled")

    # Mixed direct backends with Open WebUI and no LiteLLM are intentionally not
    # supported in this first graph rewrite.
    if open_webui.get("enabled") and not litellm.get("enabled") and ollama.get("enabled") and vllm_runtimes:
        if open_webui.get("provider") != "ollama":
            errors.append("Open WebUI with mixed Ollama+vLLM direct providers requires LiteLLM")
        else:
            warnings.append("Open WebUI is connected only to Ollama; vLLM runtimes are raw/direct endpoints")

    if open_webui.get("enabled"):
        provider = open_webui.get("provider")
        if provider == "litellm" and not litellm.get("enabled"):
            errors.append("Open WebUI provider=litellm but LiteLLM is disabled")
        if provider == "ollama" and not ollama.get("enabled"):
            errors.append("Open WebUI provider=ollama but Ollama is disabled")

    if ollama.get("enabled"):
        if ollama.get("placement_error"):
            errors.append(f"ollama placement failed: {ollama['placement_error']}")
        for idx in ollama.get("gpu_indices", []) or []:
            if idx not in gpu_map:
                errors.append(f"ollama references missing gpu index {idx}")
            elif gpu_map[idx].get("display_active"):
                warnings.append(f"ollama uses display-active GPU {idx}")
        if ollama.get("publish_port"):
            port = int(resolved.get("ports", {}).get("ollama", 11434))
            if port in used_ports:
                errors.append(f"duplicate host port assignment: {port}")
            used_ports.add(port)

    seen_service_names: set[str] = set()
    for svc in vllm_runtimes.values():
        if svc["service_name"] in seen_service_names:
            errors.append(f"duplicate service name: {svc['service_name']}")
        seen_service_names.add(svc["service_name"])

        if backend == "kubeai":
            resource_profile = str(svc.get("resource_profile", "")).strip()
            if not resource_profile:
                errors.append(f"vLLM runtime {svc['runtime_name']} is missing resource_profile for kubeai backend")
            else:
                profile_name = resource_profile.split(":", 1)[0]
                if profile_name not in resolved.get("resource_profiles", {}):
                    errors.append(f"vLLM runtime {svc['runtime_name']} references unknown resource profile {profile_name!r}")

        if not svc.get("hf_model_id"):
            errors.append(f"vLLM runtime {svc['runtime_name']} is missing hf_model_id")
        if not svc.get("served_model_name"):
            errors.append(f"vLLM runtime {svc['runtime_name']} is missing served_model_name")
        protocol_mode = svc.get("protocol_mode")
        supported = list(svc.get("supported_protocols") or [])
        if supported and protocol_mode not in supported:
            errors.append(
                f"vLLM runtime {svc['runtime_name']} requests protocol_mode={protocol_mode}, "
                f"but model {svc.get('model_ref')} supports only {supported}."
            )
        if svc.get("placement_error"):
            errors.append(f"vLLM runtime {svc['runtime_name']} placement failed: {svc['placement_error']}")
        gpu_indices = svc.get("gpu_indices", [])
        if not gpu_indices:
            warnings.append(f"vLLM runtime {svc['runtime_name']} has no concrete GPU assignment in the rendered plan")
            continue
        if svc.get("tensor_parallel_size", 1) > len(gpu_indices):
            errors.append(f"vLLM runtime {svc['runtime_name']} has tensor_parallel_size larger than assigned GPU count")
        tp = max(1, int(svc.get("tensor_parallel_size", 1)))
        per_gpu_need = float(svc.get("min_vram_gib_per_replica", 0)) / tp
        headroom = float(policy.get("minimum_vram_headroom_gib", 0))
        for idx in gpu_indices:
            if idx not in gpu_map:
                errors.append(f"vLLM runtime {svc['runtime_name']} references missing gpu index {idx}")
                continue
            gpu = gpu_map[idx]
            if policy.get("reserve_display_gpu") == "auto" and gpu.get("display_active") and policy.get("forbid_reserved_gpu_use"):
                errors.append(f"vLLM runtime {svc['runtime_name']} uses display-active GPU {idx}")
            elif gpu.get("display_active"):
                warnings.append(f"vLLM runtime {svc['runtime_name']} uses display-active GPU {idx}")
            if gpu.get("memory_gib", 0) < (per_gpu_need + headroom):
                errors.append(
                    f"vLLM runtime {svc['runtime_name']} estimates {per_gpu_need} GiB + {headroom} GiB headroom on GPU {idx}, "
                    f"but only {gpu.get('memory_gib')} GiB is available"
                )
        if len(gpu_indices) > 1 and policy.get("require_homogeneous_multi_gpu_groups"):
            names = {gpu_map[idx]["name"] for idx in gpu_indices if idx in gpu_map}
            mems = {gpu_map[idx]["memory_gib"] for idx in gpu_indices if idx in gpu_map}
            if len(names) > 1 or len(mems) > 1:
                errors.append(f"vLLM runtime {svc['runtime_name']} uses a heterogeneous multi-GPU group")

    if backend == "compose" and litellm.get("enabled"):
        port = int(resolved.get("ports", {}).get("litellm", 14042))
        if port in used_ports:
            errors.append(f"duplicate host port assignment: {port}")
        used_ports.add(port)

    return {"ok": not errors, "errors": errors, "warnings": warnings}
