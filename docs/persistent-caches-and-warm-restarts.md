# Persistent caches and warm restarts

Provider-specific on-disk state keeps restart time short without changing model
semantics or runtime performance. vLLM runtimes use Hugging Face and vLLM
caches. Ollama uses its own model store. None of these replaces the cost of
moving model weights into GPU memory — that work happens whenever the provider
process loads a model.

## What gets cached

### Ollama model store — `state.ollama -> /root/.ollama`

Ollama downloads GGUF/model blobs into `/root/.ollama`. The Compose template
mounts `state.ollama` there whenever the Ollama provider is enabled. Direct
Ollama profiles therefore survive `down`, `up`, `switch`, and `render` without
re-pulling models.

Ollama model residency is controlled by daemon/request settings such as
`keep_alive` / `OLLAMA_KEEP_ALIVE`. A short keep-alive can let a mostly idle
home-assistant-style backend unload models after use; the next request pays the
load cost again.

### Hugging Face cache — `state.hf_cache -> /root/.cache/huggingface`

The Hugging Face Hub client downloads model weights, tokenizers, and config
files into this directory the first time a model is requested. Persisting the
mount across container restarts means the second start of the same model
reuses the on-disk weights instead of re-downloading them. For 70B+ models
this can be the difference between minutes and hours of cold start.

This mount existed before the vLLM cache was added; it is unchanged.

### vLLM cache — `state.vllm_cache -> /root/.cache/vllm`

vLLM stores compiled artifacts under `VLLM_CACHE_ROOT` (defaulted to
`/root/.cache/vllm` inside the container). The most expensive entries are
`torch.compile` graphs and CUDA-graph captures keyed by the engine
configuration (model architecture, tensor-parallel size, max context length,
dtype, etc.). On a cold container with an empty cache, vLLM has to
re-compile; on a warm restart against the same configuration, those artifacts
are reused.

Default host path: `/data/service/docker/infer-stack/vllm-cache`. Override via
`state.vllm_cache` in `config.yaml` or by editing the rendered deployment
plan. The path is created on first volume mount; no manual `mkdir` is
required.

The cache key is keyed on the engine configuration. Changing `max_model_len`,
`tensor_parallel_size`, `gpu_memory_utilization`, the optimization level,
eager/compile behaviour, or the model itself will (correctly) miss the cache
and trigger a recompile. **Do not** add `--enforce-eager` or `-O0` to shave
seconds off the restart — that sacrifices steady-state throughput to skip a
one-time cost the cache already amortises.

### Secondary compiler caches

The Compose template also persists the other obvious startup caches that sit
outside `VLLM_CACHE_ROOT`:

- `state.torch_cache -> /root/.cache/torch` for PyTorch / TorchInductor
  artifacts, with `TORCH_HOME` and `TORCHINDUCTOR_CACHE_DIR` set explicitly.
- `state.triton_cache -> /root/.cache/triton` for Triton kernels, with
  `TRITON_CACHE_DIR` set explicitly.
- `state.cuda_cache -> /root/.cache/nvidia/ComputeCache` for NVIDIA driver
  JIT artifacts, with `CUDA_CACHE_PATH` set explicitly.

These caches reduce repeated compile/JIT work, but they do not make model
switching instantaneous. A vLLM process still has to import Python modules,
construct the engine, deserialize the selected model, allocate KV cache, move
weights to the GPU, and run any cache-missed graph/cuda-graph setup.

## What is not cached

Model weights still have to be loaded from `/root/.cache/huggingface` (CPU
RAM / page cache) into GPU HBM after every container restart. There is no
way around this without keeping the engine process alive — i.e. avoiding
the restart in the first place. See the "minimal restart" guidance below.

Switching a single vLLM runtime from model A to model B still means replacing
the vLLM engine process, because vLLM serves one model configuration per
process in this stack. That is why even tiny models can take tens of seconds
to come back healthy: the expensive work is not just downloading weights.

## Shared memory: `ipc: host`

vLLM uses shared memory for tensor-parallel communication and worker IPC.
Docker's default `--shm-size` (64 MiB) is far too small. The Compose
template sets `ipc: host` on every vLLM service, matching the upstream vLLM
Docker guidance and giving the engine the host's shared-memory budget.

If your environment forbids host IPC (multi-tenant cluster policy, etc.),
replace `ipc: host` in the rendered Compose with an explicit `shm_size`
sized for your tensor-parallel topology — typically a few GiB for TP > 1.

## Minimal-restart workflow

LiteLLM deliberately does **not** depend on provider health in the rendered
Compose file. That prevents Compose from restarting LiteLLM every time a vLLM
runtime container is replaced during a model swap. The CLI refreshes LiteLLM's
route table through its admin API when possible; smoke tests retry briefly
while the selected upstream model finishes loading.

When you are iterating on a single profile and don't want to bounce the whole
stack:

```bash
# Pull refreshed images through the rendered Compose wrapper.
infer-stack pull vllm-<profile>

# Restart just the one service through the wrapper. This avoids bouncing
# Postgres / LiteLLM / Open WebUI.
infer-stack restart vllm-<profile>
```

For a full stack restart (e.g. after `infer-stack render` against a new
profile), `infer-stack down && infer-stack up -d` is the safe path; persistent
volumes and bind mounts (Postgres, Open WebUI data, Ollama model store, and
vLLM caches) are not touched by `down`.

For direct Ollama model pulls, operate on the shared daemon instead of replacing
the container:

```bash
infer-stack ollama-pull qwen3.5:4b
```

## Verifying the cache is being reused

After a warm restart, a populated `state.vllm_cache` directory will contain
subdirectories keyed by configuration hash. The vLLM startup logs print
"Using cached compiled graph" (or similar; exact wording varies by version)
when the cache is hit. If you see a long compile pass on every restart,
check that the volume mount is actually pointing at the persistent path and
not at an anonymous Docker volume.
