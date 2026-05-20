from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import GENERATED_DIR_NAME, KUBEAI_GENERATED_SUBDIR, normalized_output
from .exporters import benchmark_bundle_dir, helm_bundle_dir
from .profile_runtime import default_base_url


def verify_profile(root: Path, deployment: dict[str, Any]) -> dict[str, Any]:
    service = deployment.get("services", [])[0] if deployment.get("services") else {}
    profile_name = deployment["serving_profile"]["name"]
    output_cfg = deployment.get("output")
    if output_cfg:
        output_root = Path(normalized_output(root, output_cfg)["generated_dir"])
    else:
        output_root = root / GENERATED_DIR_NAME
    bundle_dir = benchmark_bundle_dir(root, profile_name, generated_dir=output_root)
    legacy_bundle_dir = helm_bundle_dir(root, profile_name, generated_dir=output_root)
    expected = {
        "public_name": deployment["serving_profile"]["public_name"],
        "logical_model_name": service.get("logical_model_name", ""),
        "protocol_mode": service.get("protocol_mode", deployment["serving_profile"].get("protocol_mode", "")),
        "served_model_name": service.get("served_model_name", deployment["serving_profile"].get("served_model_name", "")),
        "endpoint_base_url": default_base_url(deployment),
        "generated_artifacts": {
            "compose": str(output_root / "docker-compose.yml"),
            "kubeai_models": str(output_root / KUBEAI_GENERATED_SUBDIR / "models.yaml"),
            "benchmark_bundle_dir": str(bundle_dir),
            "legacy_helm_bundle_dir": str(legacy_bundle_dir),
        },
    }
    checks = {
        "has_services": bool(deployment.get("services")),
        "has_public_name": bool(expected["public_name"]),
        "has_logical_model_name": bool(expected["logical_model_name"]),
        "has_protocol_mode": bool(expected["protocol_mode"]),
    }
    return {
        "ok": all(checks.values()),
        "profile": expected,
        "checks": checks,
    }
