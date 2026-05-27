from __future__ import annotations

from typing import Any


def vllm_args(service: dict[str, Any]) -> list[str]:
    args = [
        f"--served-model-name={service['served_model_name']}",
        f"--tensor-parallel-size={service['tensor_parallel_size']}",
        f"--data-parallel-size={service['data_parallel_size']}",
        f"--max-model-len={service['max_model_len']}",
        f"--gpu-memory-utilization={service['gpu_memory_utilization']}",
        f"--max-num-batched-tokens={service['max_num_batched_tokens']}",
        f"--max-num-seqs={service['max_num_seqs']}",
        "--disable-log-requests",
    ]
    if service.get("enable_prefix_caching"):
        args.append("--enable-prefix-caching")
    if service.get("enable_auto_tool_choice"):
        args.append("--enable-auto-tool-choice")
        if service.get("tool_call_parser"):
            args.append(f"--tool-call-parser={service['tool_call_parser']}")
    args.extend(service.get("extra_args", []))
    return args


def default_base_url(deployment: dict[str, Any], *, explicit: str | None = None) -> str:
    if explicit:
        return explicit.rstrip("/")
    access = deployment.get("access", {}).get("default") or {}
    if access.get("base_url"):
        return str(access["base_url"]).rstrip("/")
    backend = deployment.get("backend", "compose")
    if backend == "kubeai":
        ingress = deployment.get("cluster", {}).get("ingress", {}) or {}
        host = ingress.get("host", "")
        if ingress.get("enabled") and host:
            return f"http://{host}/openai/v1"
        return "http://127.0.0.1:8000/openai/v1"
    if (deployment.get("gateways", {}).get("litellm") or {}).get("enabled"):
        return f"http://127.0.0.1:{deployment.get('ports', {}).get('litellm', 14042)}/v1"
    if (deployment.get("providers", {}).get("ollama") or {}).get("enabled"):
        return f"http://127.0.0.1:{deployment.get('ports', {}).get('ollama', 11434)}"
    return f"http://127.0.0.1:{deployment.get('ports', {}).get('litellm', 14042)}/v1"


def suggested_client_class(protocol_mode: str, transport_kind: str) -> str:
    protocol = protocol_mode.lower()
    kind = transport_kind.lower()
    if kind == "vllm-direct":
        if protocol == "completions":
            return "helm.clients.vllm_client.VLLMClient"
        return "helm.clients.vllm_client.VLLMChatClient"
    if protocol == "completions":
        return "helm.clients.openai_client.OpenAILegacyCompletionsClient"
    return "helm.clients.openai_client.OpenAIClient"


def export_transport_config(service: dict[str, Any], deployment: dict[str, Any], *, base_url: str | None = None) -> dict[str, Any]:
    transport = dict(service.get("benchmark_transport", service.get("transport", {})) or {})
    backend = str(deployment.get("backend", "compose")).lower()
    protocol = service.get("protocol_mode", "chat")

    if backend == "kubeai":
        transport_kind = "openai-compatible"
    else:
        transport_kind = str(transport.get("kind") or "openai-compatible")

    if backend == "kubeai":
        resolved_base_url = default_base_url(deployment, explicit=base_url)
    elif base_url:
        resolved_base_url = base_url.rstrip("/")
    elif transport.get("base_url"):
        resolved_base_url = str(transport["base_url"]).rstrip("/")
    else:
        resolved_base_url = default_base_url(deployment)

    if backend == "kubeai":
        request_model_name = service["profile_public_name"]
        deployment_name = f"kubeai/{service['profile_public_name']}-local"
        api_key_value = str(transport.get("api_key_placeholder") or "EMPTY")
        api_key_env = str(transport.get("api_key_env") or "KUBEAI_OPENAI_API_KEY")
    elif transport_kind == "vllm-direct":
        request_model_name = str(transport.get("request_model_name") or service["hf_model_id"] or service["served_model_name"])
        deployment_name = str(transport.get("deployment_name") or f"vllm/{service['profile_public_name']}-local")
        api_key_value = str(transport.get("api_key_placeholder") or "EMPTY")
        api_key_env = str(transport.get("api_key_env") or "VLLM_API_KEY")
    else:
        request_model_name = str(transport.get("request_model_name") or service["served_model_name"])
        deployment_name = str(transport.get("deployment_name") or f"litellm/{service['profile_public_name']}-local")
        api_key_value = str(transport.get("api_key_placeholder") or "SET_LITELLM_MASTER_KEY_IN_RUNTIME_BUNDLE")
        api_key_env = str(transport.get("api_key_env") or "LITELLM_MASTER_KEY")

    return {
        "transport_kind": transport_kind,
        "backend": backend,
        "base_url": resolved_base_url,
        "deployment_name": deployment_name,
        "request_model_name": request_model_name,
        "api_key_env": api_key_env,
        "api_key_value": api_key_value,
        "client_class": suggested_client_class(protocol, transport_kind),
        "public_name": service["profile_public_name"],
        "logical_model_name": service["logical_model_name"],
        "served_model_name": service["served_model_name"],
        "kubernetes_name": service["kubernetes_name"],
    }
