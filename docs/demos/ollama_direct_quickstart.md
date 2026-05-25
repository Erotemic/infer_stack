# Quickstart — direct Ollama on a dual GTX 1080 Ti host

This quickstart brings up the simplest local stack for Pascal-era GPUs:

```text
Open WebUI -> Ollama
```

There is no LiteLLM router and no predeclared model registry. Ollama manages
its own model store under `state.ollama`, and models are pulled with the
Ollama CLI or through Open WebUI.

## Prerequisites

- Docker with the NVIDIA container runtime.
- `nvidia-smi` works on the host and from GPU-enabled containers.
- `vllm-stack` installed in your Python environment.

This path is useful for GTX 1080 Ti systems because Ollama supports Pascal
GPUs, while current official vLLM images require newer CUDA compute capability.

## 1. Set up the profile

```bash
mkdir -p /data/service/docker/vllm-stack
vllm-stack setup \
  --backend compose \
  --profile ollama-direct \
  --state-root /data/service/docker/vllm-stack \
  --generated-dir /data/service/docker/vllm-stack/generated
```

For a two-card 1080 Ti machine, the built-in profile pins Ollama to GPUs
`[0, 1]` and uses conservative daemon settings:

```yaml
keep_alive: 2m
context_length: 4096
num_parallel: 1
max_loaded_models: 1
max_queue: 8
```

## 2. Render and start

```bash
vllm-stack render --yes --simulate-hardware 2x11
vllm-stack up -d
```

Expected rendered services:

```text
postgres-open-webui
ollama
open-webui
```

Not expected in this profile:

```text
postgres-litellm
litellm
vllm-*
```

Open WebUI is available at <http://127.0.0.1:13000>. Ollama is published on
<http://127.0.0.1:11434> because the direct profile sets `publish_port: true`.

## 3. Pull a model

Start with a small model before trying larger ones:

```bash
docker compose -f /data/service/docker/vllm-stack/generated/docker-compose.yml \
  --env-file /data/service/docker/vllm-stack/generated/.env \
  exec ollama ollama pull qwen3.5:4b
```

Other good initial candidates for 11 GiB cards:

```bash
ollama pull qwen3.5:2b
ollama pull qwen3.5:9b-q4_K_M
ollama pull gemma4:e2b
```

Run a direct API check:

```bash
curl -s http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5:4b",
    "messages": [{"role": "user", "content": "Say hello from Ollama."}],
    "stream": false
  }' | python3 -m json.tool
```

Or use Ollama's OpenAI-compatible endpoint:

```bash
curl -s http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5:4b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }' | python3 -m json.tool
```

## 4. Day-to-day operations

```bash
vllm-stack ps
vllm-stack logs -f ollama
vllm-stack down
vllm-stack up -d
```

Inspect loaded models and where they live:

```bash
docker compose -f /data/service/docker/vllm-stack/generated/docker-compose.yml \
  --env-file /data/service/docker/vllm-stack/generated/.env \
  exec ollama ollama list

docker compose -f /data/service/docker/vllm-stack/generated/docker-compose.yml \
  --env-file /data/service/docker/vllm-stack/generated/.env \
  exec ollama ollama ps
```

## 5. When to add LiteLLM

Stay direct when the machine is just a local Ollama server or Home Assistant
backend. Switch to an Ollama gateway profile when you want a stable OpenAI
`/v1` route name that can later move to vLLM:

```bash
vllm-stack switch ollama-qwen3.5-4b-gateway --apply
```

That changes the shape to:

```text
Open WebUI -> LiteLLM -> Ollama
```

For mixed Ollama+vLLM routing, use a profile such as `mixed-ollama-smollm` as
a template.
