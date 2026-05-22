# Persistent caches and warm restarts

Two on-disk caches keep restart time short without changing model semantics or
runtime performance. Both are mounted into every rendered vLLM Compose
service. Neither replaces the cost of moving model weights into GPU memory —
that work happens on every container start.

## What gets cached

### Hugging Face cache — `state.hf_cache → /root/.cache/huggingface`

The Hugging Face Hub client downloads model weights, tokenizers, and config
files into this directory the first time a model is requested. Persisting the
mount across container restarts means the second start of the same model
reuses the on-disk weights instead of re-downloading them. For 70B+ models
this can be the difference between minutes and hours of cold start.

This mount existed before the vLLM cache was added; it is unchanged.

### vLLM cache — `state.vllm_cache → /root/.cache/vllm`

vLLM stores compiled artifacts under `VLLM_CACHE_ROOT` (defaulted to
`/root/.cache/vllm` inside the container). The most expensive entries are
`torch.compile` graphs and CUDA-graph captures keyed by the engine
configuration (model architecture, tensor-parallel size, max context length,
dtype, etc.). On a cold container with an empty cache, vLLM has to
re-compile; on a warm restart against the same configuration, those artifacts
are reused.

Default host path: `/data/service/docker/vllm-stack/vllm-cache`. Override via
`state.vllm_cache` in `config.yaml` or by editing the rendered deployment
plan. The path is created on first volume mount; no manual `mkdir` is
required.

The cache key is keyed on the engine configuration. Changing `max_model_len`,
`tensor_parallel_size`, `gpu_memory_utilization`, the optimization level,
eager/compile behaviour, or the model itself will (correctly) miss the cache
and trigger a recompile. **Do not** add `--enforce-eager` or `-O0` to shave
seconds off the restart — that sacrifices steady-state throughput to skip a
one-time cost the cache already amortises.

## What is not cached

Model weights still have to be loaded from `/root/.cache/huggingface` (CPU
RAM / page cache) into GPU HBM after every container restart. There is no
way around this without keeping the engine process alive — i.e. avoiding
the restart in the first place. See the "minimal restart" guidance below.

## Shared memory: `ipc: host`

vLLM uses shared memory for tensor-parallel communication and worker IPC.
Docker's default `--shm-size` (64 MiB) is far too small. The Compose
template sets `ipc: host` on every vLLM service, matching the upstream vLLM
Docker guidance and giving the engine the host's shared-memory budget.

If your environment forbids host IPC (multi-tenant cluster policy, etc.),
replace `ipc: host` in the rendered Compose with an explicit `shm_size`
sized for your tensor-parallel topology — typically a few GiB for TP > 1.

## Minimal-restart workflow

When you are iterating on a single profile and don't want to bounce the whole
stack:

```bash
# Pull the new image first so the outage starts only after the layers are local.
docker compose -f generated/docker-compose.yml pull vllm-<profile>

# Recreate just the one service. --no-deps avoids restarting Postgres /
# litellm / open-webui. --force-recreate ensures the new image and any
# changed config are picked up even if Compose thinks the service is "up".
docker compose -f generated/docker-compose.yml up -d \
  --no-deps --force-recreate vllm-<profile>
```

For a full stack restart (e.g. after `vllm-stack render` against a new
profile), `vllm-stack down && vllm-stack up -d` is the safe path; persistent
volumes (Postgres, OWUI data, both caches) are not touched by `down`.

## Verifying the cache is being reused

After a warm restart, a populated `state.vllm_cache` directory will contain
subdirectories keyed by configuration hash. The vLLM startup logs print
"Using cached compiled graph" (or similar; exact wording varies by version)
when the cache is hit. If you see a long compile pass on every restart,
check that the volume mount is actually pointing at the persistent path and
not at an anonymous Docker volume.
