# Quickstart — RTX 3090 workstation (GPU 1)

Brings the vllm-stack up on a single-GPU workstation where GPU 0 is reserved
for the desktop and GPU 1 is free for inference. We start with the tiniest
possible model to prove the plumbing works, then switch up to progressively
larger models without re-running setup.

Ports: LiteLLM on **14042**, Open WebUI on **13000**

After setup, all state and generated artifacts live under
`/data/service/docker/vllm-stack/`:

```
/data/service/docker/vllm-stack/
  generated/            ← docker-compose.yml, .env, plan.yaml, litellm config
  hf-cache/             ← downloaded model weights (survives stack restarts)
  vllm-cache/           ← compiled torch graphs (survives stack restarts)
  open-webui/           ← chat history
  postgres-open-webui/  ← open-webui database
  postgres-litellm/     ← litellm database
  runtime/              ← runtime bind-mount configs

~/.config/vllm_service/
  config.yaml           ← profile / backend / path settings
  models.yaml           ← custom model overrides
```

---

## Prerequisites

- Docker with the NVIDIA container runtime (`nvidia-smi` visible inside containers)
- `vllm-stack` installed in your Python environment
- A Hugging Face token in your shell (only needed for gated models like Gemma):

```bash
export HF_TOKEN=hf_...
```

The first profile in this guide (`gpt2-single`) is fully public and does **not**
need `HF_TOKEN`.

---

## 1. First-time setup with GPT-2 (smallest possible)

Start with `gpt2-single` — GPT-2 124M, ~250 MB download, no auth, completions-only.
The whole point is to validate the stack end-to-end before committing to a large
download or a complex model.

Create the data root and write `~/.config/vllm_service/config.yaml`:

```bash
mkdir -p /data/service/docker/vllm-stack
vllm-stack setup \
  --backend compose \
  --profile gpt2-single \
  --state-root /data/service/docker/vllm-stack \
  --generated-dir /data/service/docker/vllm-stack/generated
```

Render and bring it up:

```bash
vllm-stack render --yes
vllm-stack up -d
```

vLLM downloads GPT-2 (~250 MB) and compiles a torch graph (cached). Expect
~30–60 seconds total on first start, a few seconds on warm restart.

Watch it come up:

```bash
vllm-stack ps         # show container status
vllm-stack logs -f    # tail all logs, Ctrl-C to stop tailing (containers stay up)
```

Once all containers are healthy, smoke-test the API:

```bash
vllm-stack smoke-test
```

This sends a single completion request to LiteLLM and prints the model response.
On success you'll see a small JSON blob with `"finish_reason": "length"` or
similar. (Open WebUI is up at <http://127.0.0.1:13000> but you'll probably want
a chat-capable model — switch profiles below.)

Manual API check, in case you want to see the bytes:

```bash
curl -s http://127.0.0.1:14042/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(vllm-stack env --key LITELLM_MASTER_KEY)" \
  -d '{
    "model": "gpt2",
    "prompt": "Once upon a time, ",
    "max_tokens": 32
  }' | python3 -m json.tool
```

---

## 2. Switch to another tiny model (SmolLM2 135M, chat-capable)

GPT-2 is base-model only — no chat template, no Open WebUI. Step up to
**SmolLM2 135M Instruct** to get a real chat-completions endpoint at the
smallest possible cost (~270 MB):

```bash
vllm-stack switch smollm2-135m-single --apply
```

`switch` re-points the active profile in `config.yaml`, re-renders the compose
file, and `--apply` cycles the stack to match (`down` → `up -d`). GPT-2 stays
in `hf-cache/`, so flipping back to it later costs nothing.

Smoke test again — this time chat-completions:

```bash
vllm-stack smoke-test
```

Then open <http://127.0.0.1:13000> in a browser — Open WebUI now has a chat
model wired up.

---

## 3. Switch to Qwen3.5 9B (real workstation model)

Once the plumbing is proven, switch to a properly-sized reasoning model. The
canonical 3090 profile is `workstation-safe` (Qwen3.5 9B, GPU 0 not assumed
to be free — uses `first_fit` placement, lands on the first usable GPU):

```bash
vllm-stack switch workstation-safe --apply
```

First start downloads ~18 GB and warms the torch.compile cache; subsequent
restarts are warm. The reasoning parser is engaged automatically (Qwen3.5 emits
`<think>…</think>` blocks; vLLM strips them and exposes them in a separate
`reasoning` field).

Smoke test the larger model:

```bash
vllm-stack smoke-test
```

If you'd rather pin to a specific GPU index (e.g. GPU 1 because GPU 0 drives a
display), use the `gemma4-e4b-3090-workstation` profile instead, which bakes
`gpu_indices: [1]` into the placement.

---

## Day-to-day workflow

**Stop** (preserves all data and cached weights):

```bash
vllm-stack down
```

**Start after a reboot** (warm restart):

```bash
vllm-stack up -d
```

**Tail logs:**

```bash
vllm-stack logs
```

**Container status:**

```bash
vllm-stack ps
```

**Run the smoke test** to verify the router and the backend are both healthy:

```bash
vllm-stack smoke-test
```

**Read a secret from .env** (e.g. for `curl`):

```bash
vllm-stack env --key LITELLM_MASTER_KEY
```

---

## Troubleshooting

### `Cannot start stack: required host ports are already bound`

`vllm-stack up` does a pre-flight check on ports 14042 (litellm) and 13000
(open-webui) before pulling images or waiting for vLLM to load, so this fails
immediately rather than after a long startup.

Find what is holding the port:

```bash
ss -tlnp 'sport = :14042'                    # show socket + process (if visible)
sudo lsof -nP -iTCP:14042 -sTCP:LISTEN       # works even when ss can't see the owner
docker ps --filter publish=14042             # leftover container from another project
```

Common cases:

- **Leftover container from a previous vllm-stack run** — `vllm-stack down`, or
  `docker stop litellm && docker rm litellm` if the generated compose file is
  gone.
- **Another project's container** publishes the same port — stop it, or change
  this stack's ports:
  `vllm-stack setup --litellm-port 14001 --open-webui-port 13001` then
  `vllm-stack render --yes` and `vllm-stack up -d`.
- **A non-Docker service** is bound to the port — find the PID via the
  commands above and stop it.

### Smoke test errors

`vllm-stack smoke-test` maps the common requests-library failures onto
one-line remediation hints:

- *Could not connect: nothing is listening yet* — give `vllm-stack up` a few
  more seconds; `vllm-stack ps` to confirm litellm is running.
- *Connection was closed before a response* — router is up but an upstream
  isn't ready; `vllm-stack logs vllm-*` to watch model loading.
- *401/403 from the API* — the key in `.env` doesn't match the running
  container. If you re-rendered after the container started, restart with
  `vllm-stack down && vllm-stack up -d`.
- *503 from the API* — vLLM engine is still loading; `vllm-stack logs vllm-*`.

### Wiping state to start fresh

Delete postgres data, open-webui history, and runtime configs while keeping
cached model weights (avoids a re-download):

```bash
vllm-stack purge --yes
```

Delete everything including the model cache (forces re-download on next start):

```bash
vllm-stack purge --yes --delete-cache
```
