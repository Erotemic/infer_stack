# Stack graph profiles

Profiles now describe a local LLM stack as four independent pieces:

- `providers`: inference servers such as `vllm` and `ollama`.
- `gateways`: optional routing/API unification layers such as `litellm`.
- `frontends`: optional user interfaces such as `open_webui`.
- `routes`: optional public model aliases exposed through a gateway.

This makes Ollama direct mode simple while still allowing mixed Ollama+vLLM stacks behind LiteLLM.

## Direct Ollama

`ollama-direct` starts one Ollama daemon and Open WebUI connected directly to it. It does not render LiteLLM, does not render `postgres-litellm`, and does not require predeclared model routes.

```bash
vllm-stack setup --backend compose --profile ollama-direct
vllm-stack render --yes --simulate-hardware 2x11
vllm-stack up -d

docker compose -f generated/docker-compose.yml --env-file generated/.env exec ollama \
  ollama pull qwen3.5:4b
```

## Ollama through LiteLLM

`ollama-qwen3.5-4b-gateway` starts Ollama plus LiteLLM and exposes a stable OpenAI-compatible model alias through LiteLLM.

```yaml
routes:
  qwen3.5-4b:
    provider: ollama
    model: qwen3.5-4b
```

LiteLLM renders that as `ollama_chat/qwen3.5:4b` with `api_base: http://ollama:11434`.

## Mixed Ollama + vLLM

`mixed-ollama-smollm` demonstrates one shared Ollama daemon plus one vLLM runtime behind LiteLLM.

```yaml
routes:
  home-assistant-local:
    provider: ollama
    model: qwen3.5-4b
  smollm2-135m:
    provider: vllm
    runtime: smollm
```

## Raw servers

`raw-ollama-vllm` starts the backends without Open WebUI or LiteLLM. This is useful for debugging direct provider endpoints.
