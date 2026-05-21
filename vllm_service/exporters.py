from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from .config import normalized_output
from .paths import data_root
from .profile_runtime import default_base_url, export_transport_config

# Transitional compatibility layer:
# benchmark bundle ownership now lives in helm_audit. Keep this exporter only to
# avoid breaking near-term operator workflows while the audit-side integration
# becomes the primary path.


def _resolve_generated_dir(generated_dir: Path | None) -> Path:
    if generated_dir is not None:
        p = Path(generated_dir)
        if not p.is_absolute():
            p = data_root() / p
        return p
    return Path(normalized_output(None)["generated_dir"])


def benchmark_bundle_dir(profile_name: str, *, generated_dir: Path | None = None) -> Path:
    return _resolve_generated_dir(generated_dir) / "benchmark" / profile_name


def helm_bundle_dir(profile_name: str, *, generated_dir: Path | None = None) -> Path:
    return _resolve_generated_dir(generated_dir) / "helm" / profile_name


def _generated_dir_for_deployment(deployment: dict) -> Path:
    return Path(normalized_output(deployment.get("output"))["generated_dir"])


def _maybe_data_relative(target: Path) -> str:
    """Format ``target`` as a path relative to ``data_root()`` when possible."""
    try:
        return str(target.resolve().relative_to(data_root().resolve()))
    except ValueError:
        return str(target.resolve())


def _service_endpoint_shape(service: dict, deployment: dict, *, base_url: str | None) -> dict:
    transport = export_transport_config(service, deployment, base_url=base_url)
    resolved_base_url = transport["base_url"] or default_base_url(deployment, explicit=base_url)
    protocol_mode = service.get("protocol_mode", "chat")
    return {
        "base_url": resolved_base_url,
        "models_path": f"{resolved_base_url}/models",
        "chat_completions_path": f"{resolved_base_url}/chat/completions",
        "completions_path": f"{resolved_base_url}/completions",
        "preferred_request_path": f"{resolved_base_url}/{'completions' if protocol_mode == 'completions' else 'chat/completions'}",
        "request_model_name": transport["request_model_name"],
        "transport_kind": transport["transport_kind"],
    }


def _benchmark_model_deployment(service: dict, deployment: dict, *, base_url: str | None) -> dict:
    transport = export_transport_config(service, deployment, base_url=base_url)
    args: dict[str, str] = {"base_url": transport["base_url"]}
    if transport["transport_kind"] == "vllm-direct":
        args["vllm_model_name"] = transport["request_model_name"]
    else:
        args["api_key"] = transport["api_key_value"]
        args["openai_model_name"] = transport["request_model_name"]

    return {
        "name": transport["deployment_name"],
        "model_name": service["logical_model_name"],
        "tokenizer_name": service["tokenizer_name"],
        "max_sequence_length": int(service["max_model_len"]),
        "client_spec": {
            "class_name": transport["client_class"],
            "args": args,
        },
    }


def _manifest_template(
    *,
    experiment_name: str,
    description: str,
    model_name: str,
    model_deployment_name: str,
    model_deployments_fpath: str,
    max_eval_instances: int,
) -> dict:
    return {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "description": description,
        "run_entries": [f"ifeval:model={model_name},model_deployment={model_deployment_name}"],
        "max_eval_instances": max_eval_instances,
        "suite": experiment_name,
        "mode": "compute_if_missing",
        "materialize": "symlink",
        "backend": "tmux",
        "devices": 0,
        "tmux_workers": 1,
        "local_path": "prod_env",
        "precomputed_root": None,
        "require_per_instance_stats": True,
        "model_deployments_fpath": model_deployments_fpath,
        "enable_huggingface_models": [],
        "enable_local_huggingface_models": [],
    }


def _write_legacy_alias(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def export_benchmark_bundle(deployment: dict, *, base_url: str | None = None, output_dir: Path | None = None) -> dict:
    services = deployment.get("services", [])
    if len(services) != 1:
        raise ValueError("Benchmark bundle export currently expects a single-service serving profile")
    service = services[0]
    if not service:
        raise ValueError("Cannot export a benchmark bundle for a profile with no resolved services")

    default_dir = output_dir is None
    deployment_generated = _generated_dir_for_deployment(deployment)
    bundle_dir = (
        output_dir
        or benchmark_bundle_dir(
            deployment["serving_profile"]["name"],
            generated_dir=deployment_generated,
        )
    ).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    endpoint_shape = _service_endpoint_shape(service, deployment, base_url=base_url)
    transport = export_transport_config(service, deployment, base_url=base_url)

    model_deployments = {"model_deployments": [_benchmark_model_deployment(service, deployment, base_url=base_url)]}
    model_deployments_path = bundle_dir / "model_deployments.yaml"
    model_deployments_path.write_text(yaml.safe_dump(model_deployments, sort_keys=False), encoding="utf-8")

    benchmark_section = {
        "deployment_name": transport["deployment_name"],
        "suggested_client_class": transport["client_class"],
        "model_deployments_path": str(model_deployments_path),
        "model_deployments_repo_relative": _maybe_data_relative(model_deployments_path),
    }
    bundle = {
        "target": "crfm_helm_benchmark",
        "profile": {
            "name": deployment["serving_profile"]["name"],
            "public_name": deployment["serving_profile"]["public_name"],
            "logical_model_name": service["logical_model_name"],
            "served_model_name": service["served_model_name"],
            "served_aliases": service.get("served_aliases", []),
            "base_model": service["model_ref"],
            "hf_model_id": service["hf_model_id"],
            "tokenizer_name": service["tokenizer_name"],
            "protocol_mode": service["protocol_mode"],
            "engine": service["engine"],
            "resource_profile": service["resource_profile"],
            "kubernetes_name": service["kubernetes_name"],
        },
        "benchmark_transport": {
            "backend": deployment["backend"],
            "router_type": deployment.get("router", {}).get("type", ""),
            "transport_kind": transport["transport_kind"],
            "request_model_name": transport["request_model_name"],
            "api_key_env": transport["api_key_env"],
            "api_key_value": transport["api_key_value"],
            "endpoint_shape": endpoint_shape,
        },
        "benchmark": benchmark_section,
        "helm": dict(benchmark_section),
        "artifacts": {
            "model_deployments": str(model_deployments_path),
        },
        "notes": service.get("audit_notes", []),
    }
    bundle_path = bundle_dir / "bundle.yaml"
    bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")

    model_deployments_fpath = _maybe_data_relative(model_deployments_path)
    benchmark_smoke_manifest = _manifest_template(
        experiment_name=f"{deployment['serving_profile']['name']}-smoke",
        description=f"Machine-local benchmark smoke manifest for {deployment['serving_profile']['public_name']}.",
        model_name=service["logical_model_name"],
        model_deployment_name=transport["deployment_name"],
        model_deployments_fpath=model_deployments_fpath,
        max_eval_instances=5,
    )
    benchmark_full_manifest = _manifest_template(
        experiment_name=f"{deployment['serving_profile']['name']}-full",
        description=f"Machine-local benchmark full manifest for {deployment['serving_profile']['public_name']}.",
        model_name=service["logical_model_name"],
        model_deployment_name=transport["deployment_name"],
        model_deployments_fpath=model_deployments_fpath,
        max_eval_instances=1000,
    )
    benchmark_smoke_path = bundle_dir / "benchmark_smoke_manifest.yaml"
    benchmark_full_path = bundle_dir / "benchmark_full_manifest.yaml"
    benchmark_smoke_path.write_text(yaml.safe_dump(benchmark_smoke_manifest, sort_keys=False), encoding="utf-8")
    benchmark_full_path.write_text(yaml.safe_dump(benchmark_full_manifest, sort_keys=False), encoding="utf-8")

    smoke_manifest_path = bundle_dir / "smoke_manifest.yaml"
    full_manifest_path = bundle_dir / "full_manifest.yaml"
    _write_legacy_alias(benchmark_smoke_path, smoke_manifest_path)
    _write_legacy_alias(benchmark_full_path, full_manifest_path)

    legacy_bundle_dir = None
    if default_dir:
        legacy_bundle_dir = helm_bundle_dir(
            deployment["serving_profile"]["name"],
            generated_dir=deployment_generated,
        ).resolve()
        legacy_bundle_dir.mkdir(parents=True, exist_ok=True)
        for path in [
            model_deployments_path,
            bundle_path,
            benchmark_smoke_path,
            benchmark_full_path,
            smoke_manifest_path,
            full_manifest_path,
        ]:
            _write_legacy_alias(path, legacy_bundle_dir / path.name)

    return {
        "bundle_dir": bundle_dir,
        "bundle_path": bundle_path,
        "model_deployments_path": model_deployments_path,
        "benchmark_smoke_manifest_path": benchmark_smoke_path,
        "benchmark_full_manifest_path": benchmark_full_path,
        "smoke_manifest_path": smoke_manifest_path,
        "full_manifest_path": full_manifest_path,
        "legacy_bundle_dir": legacy_bundle_dir,
        "bundle": bundle,
    }


def export_helm_bundle(deployment: dict, *, base_url: str | None = None, output_dir: Path | None = None) -> dict:
    return export_benchmark_bundle(deployment, base_url=base_url, output_dir=output_dir)
