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

    alias_targets = resolved.get("router", {}).get("aliases", {})
    service_names = {svc["service_name"] for svc in resolved.get("services", [])}
    for alias, target in alias_targets.items():
        if target not in service_names:
            errors.append(f"router alias {alias!r} points to unknown service {target!r}")

    if backend == "kubeai":
        profiles = resolved.get("resource_profiles", {})
        if not profiles:
            source = resolved.get("resource_profiles_source", "kubeai-values.local.yaml")
            errors.append(
                "No local KubeAI resource profiles were loaded for validation. "
                f"Expected them at {source}. "
                "Run `python manage.py kubeai-sync-resource-profiles --from-file values-kubeai-local-gpu.yaml` first."
            )
        for svc in resolved.get("services", []):
            resource_profile = str(svc.get("resource_profile", "")).strip()
            if not resource_profile:
                errors.append(f"service {svc['service_name']} is missing resource_profile for kubeai backend")
            else:
                profile_name = resource_profile.split(":", 1)[0]
                if profile_name not in profiles:
                    errors.append(f"service {svc['service_name']} references unknown resource profile {profile_name!r}")

    seen_service_names: set[str] = set()
    for svc in resolved.get("services", []):
        if svc["service_name"] in seen_service_names:
            errors.append(f"duplicate service name: {svc['service_name']}")
        seen_service_names.add(svc["service_name"])

        if not svc.get("profile_public_name"):
            errors.append(f"service {svc['service_name']} is missing profile_public_name")
        if not svc.get("logical_model_name"):
            errors.append(f"service {svc['service_name']} is missing logical_model_name")
        if not svc.get("served_model_name"):
            errors.append(f"service {svc['service_name']} is missing served_model_name")
        protocol_mode = svc.get("protocol_mode")
        if not protocol_mode:
            errors.append(f"service {svc['service_name']} is missing protocol_mode")
        else:
            supported = list(svc.get("supported_protocols") or [])
            if supported and protocol_mode not in supported:
                profile_name = svc.get("profile_name") or svc.get("profile_public_name") or svc["service_name"]
                model_ref = svc.get("model_ref") or svc.get("base_model") or "<unknown-model>"
                supported_repr = ", ".join(supported)
                errors.append(
                    f"Profile {profile_name} requests protocol_mode={protocol_mode}, "
                    f"but model {model_ref} supports only [{supported_repr}]. "
                    f"Use protocol_mode: {supported[0]} or provide an explicit chat template."
                )

        if svc.get("placement_error"):
            errors.append(f"service {svc['service_name']} placement failed: {svc['placement_error']}")
        gpu_indices = svc.get("gpu_indices", [])
        if not gpu_indices:
            warnings.append(f"service {svc['service_name']} has no concrete GPU assignment in the rendered plan")
            continue
        if svc.get("tensor_parallel_size", 1) > len(gpu_indices):
            errors.append(f"service {svc['service_name']} has tensor_parallel_size larger than assigned GPU count")

        tp = max(1, int(svc.get("tensor_parallel_size", 1)))
        per_gpu_need = float(svc.get("min_vram_gib_per_replica", 0)) / tp
        headroom = float(policy.get("minimum_vram_headroom_gib", 0))
        for idx in gpu_indices:
            if idx not in gpu_map:
                errors.append(f"service {svc['service_name']} references missing gpu index {idx}")
                continue
            gpu = gpu_map[idx]
            if policy.get("reserve_display_gpu") == "auto" and gpu.get("display_active") and policy.get("forbid_reserved_gpu_use"):
                errors.append(f"service {svc['service_name']} uses display-active GPU {idx}")
            elif gpu.get("display_active"):
                warnings.append(f"service {svc['service_name']} uses display-active GPU {idx}")
            if gpu.get("memory_gib", 0) < (per_gpu_need + headroom):
                errors.append(
                    f"service {svc['service_name']} estimates {per_gpu_need} GiB + {headroom} GiB headroom on GPU {idx}, but only {gpu.get('memory_gib')} GiB is available"
                )

        if len(gpu_indices) > 1 and policy.get("require_homogeneous_multi_gpu_groups"):
            names = {gpu_map[idx]["name"] for idx in gpu_indices if idx in gpu_map}
            mems = {gpu_map[idx]["memory_gib"] for idx in gpu_indices if idx in gpu_map}
            if len(names) > 1 or len(mems) > 1:
                errors.append(f"service {svc['service_name']} uses a heterogeneous multi-GPU group")

    if backend == "compose":
        port = int(resolved.get("ports", {}).get("litellm", 14000))
        if port in used_ports:
            errors.append(f"duplicate host port assignment: {port}")
        used_ports.add(port)

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }
