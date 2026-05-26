# vLLM Service

`vllm_service` manages **named stack profiles** for local and Kubernetes-backed inference.

A stack profile is a small graph made from:

* **providers** — inference runtimes such as vLLM and Ollama
* **gateways** — optional API routers such as LiteLLM
* **frontends** — optional UIs such as Open WebUI
* **routes** — optional public model aliases exposed through a gateway

This repo can render those profiles through two backends:

* **Compose** for local single-host serving. Compose supports vLLM, Ollama, optional LiteLLM, and optional Open WebUI.
* **KubeAI** for Kubernetes-backed vLLM serving. KubeAI support is vLLM-only for now.

The direct Ollama path can run without LiteLLM and without predeclaring models. vLLM profiles still use explicit runtimes, placement, and runtime settings.

## Main commands

```bash
vllm-stack setup --backend compose --profile ollama-direct
# or: vllm-stack setup --backend compose --profile qwen2-5-7b-instruct-turbo-default
vllm-stack list-profiles
vllm-stack describe-profile <profile>
vllm-stack validate
vllm-stack render
vllm-stack up -d
vllm-stack deploy
vllm-stack switch <profile> --apply  # re-render and converge; no separate up needed
vllm-stack status
vllm-stack smoke-test
```

The CLI is built on [`scriptconfig`](https://gitlab.kitware.com/utils/scriptconfig),
so every subcommand is also importable as a Python class — useful for
notebooks, tests, and other scripts:

```python
from vllm_service.cli import RenderCLI, SmokeTestCLI

RenderCLI.main(argv=False, profile="qwen2-5-7b-instruct-turbo-default", yes=True)
SmokeTestCLI.main(argv=False, model="qwen/qwen2.5-7b-instruct-turbo")
```

`manage.py` and `vllm-stack` are aliases for the same entry point;
shell examples below use `vllm-stack`.

## Operating the rendered Compose stack

Once the stack is up, common docker compose operations are available as
`vllm-stack` subcommands so you don't have to `cd` into the rendered
output directory or repeat the `-f docker-compose.yml --env-file .env`
flags. They all resolve the rendered location via the same
`output.generated_dir` chain as the rest of the CLI.

```bash
vllm-stack ps                              # docker compose ps
vllm-stack ps -a                           # include stopped
vllm-stack logs -f open-webui              # follow one service
vllm-stack logs --tail=200 litellm vllm-*  # tailored backlog
vllm-stack restart open-webui              # restart specific services
vllm-stack stop                            # stop everything (no remove)
vllm-stack start                           # start back up
vllm-stack pull                            # refresh images
```

For Ollama model management inside the rendered Ollama service, prefer the
CLI wrappers:

```bash
vllm-stack ollama-pull smollm2:135m
vllm-stack ollama-list
vllm-stack ollama-ps
```

For other interactive one-shot commands inside a container, use
`vllm-stack logs`, `vllm-stack ps`, `vllm-stack restart`, or fall back to raw
Compose only when no wrapper exists.

On the KubeAI backend these wrappers raise ``NotImplementedError`` —
use the equivalent ``kubectl`` commands in the meantime.

## Inspect a profile before running it

```bash
vllm-stack describe-profile qwen2-5-7b-instruct-turbo-default --format yaml
```

## Stack profile model

Profiles are written as stack graphs. The main sections are `providers`, `gateways`, `frontends`, and `routes`. For details and examples, see [docs/stack-graph-profiles.md](docs/stack-graph-profiles.md).

Common shapes:

```text
Open WebUI -> Ollama                         # ollama-direct, no LiteLLM
Open WebUI -> LiteLLM -> vLLM                # classic vLLM compose profiles
Open WebUI -> LiteLLM -> Ollama              # Ollama with stable aliases
Open WebUI -> LiteLLM -> Ollama + vLLM       # mixed migration / test stacks
Ollama API + vLLM API directly               # raw backend profiles
```

Custom provider models and custom profiles live in the configured `catalog.user_models_file`, which defaults to `~/.config/vllm_service/models.yaml`. New files should prefer provider-specific top-level keys:

```yaml
vllm_models:
  my-vllm-model:
    hf_model_id: org/model

ollama_models:
  my-ollama-model:
    tag: qwen3.5:4b

profiles:
  my-stack:
    providers: {}
    gateways: {}
    frontends: {}
    routes: {}
```

`models:` is still interpreted as a vLLM model catalog for convenience, but new docs and recipes use `vllm_models:` / `ollama_models:`.

## Where config and rendered artifacts live

`vllm-stack` follows XDG basedir conventions, so where you invoke it
from never changes which config it reads or where it writes rendered
artifacts:

| What | Default location | Override |
| --- | --- | --- |
| `config.yaml`, `models.yaml`, `kubeai-values.local.yaml` | `~/.config/vllm_service/` (resp. `$XDG_CONFIG_HOME`) | `VLLM_SERVICE_CONFIG_DIR` env var, or `--config-dir` |
| Rendered `generated/` (docker-compose.yml, plan.yaml, kubeai/*) and `state/` (hf-cache, postgres volumes, bind mounts) | `~/.local/share/vllm_service/` (resp. `$XDG_DATA_HOME`) | `VLLM_SERVICE_DATA_DIR` env var, or `--data-dir` |

Per-knob overrides still apply on top:

* `--generated-dir /path/to/out` (or `VLLM_SERVICE_GENERATED_DIR`, or
  `output.generated_dir` in `config.yaml`) only moves the rendered
  artifacts.
* `--state-root /path` (or `VLLM_SERVICE_STATE_ROOT`) moves all
  `state.*` paths together. Individual `state.runtime` etc. can be
  overridden in `config.yaml`.

Examples:

```bash
# Point at a checkout-local config for ad-hoc experiments.
VLLM_SERVICE_CONFIG_DIR=$PWD vllm-stack setup --backend compose --profile <p>

# Send rendered artifacts somewhere other than ~/.cache.
vllm-stack render --data-dir /srv/vllm-stack

# Or just the rendered output, leaving state/ in the cache dir.
vllm-stack render --generated-dir /tmp/scratch-out
```

`--config-dir` / `--data-dir` are per-subcommand flags (they live on
every subcommand), so they appear **after** the subcommand name on the
CLI. For "set once, applies to everything" use the env vars instead.

## Constraining placement to specific GPUs

If some of your GPUs are tied up by other work, restrict the planner
(and the rendered ``device_ids``) to the subset you want it to use:

```bash
# Only place onto GPU 1 (e.g. GPU 0 is running a display).
vllm-stack render --yes --profile test-single-11gb --allowed-gpus 1

# Or pin a TP=2 profile to physical GPUs 1 and 3.
vllm-stack render --yes --profile test-multi-gpu --allowed-gpus 1,3
```

``--allowed-gpus`` (or ``VLLM_SERVICE_ALLOWED_GPUS=1,3``) filters the
detected inventory before placement — real indices are preserved, so
the rendered compose stack pins ``device_ids: ["1", "3"]`` to those
exact physical GPUs. Useful for integration tests that need to share a
host with other jobs.

## Demos / integration recipes

End-to-end examples under [docs/demos/](docs/demos/) are written as
markdown tutorials. The CI smoke test is runnable with pytest-codeblocks:

```bash
pytest --codeblocks docs/demos/ci_smoke_test.md
```

Each ``bash`` block is a self-contained shell snippet you can also
copy-paste into a terminal. See
[docs/demos/ci_smoke_test.md](docs/demos/ci_smoke_test.md) for the
``setup → describe → validate → render`` flow on the smallest test
profiles.

For a real running vLLM stack on a workstation, see
[docs/demos/quickstart.md](docs/demos/quickstart.md). For direct Ollama on a dual GTX 1080 Ti style host, see
[docs/demos/ollama_direct_quickstart.md](docs/demos/ollama_direct_quickstart.md). For a focused GPU-1 backend switch test, see [docs/demos/smollm2_gpu1_backend_switch.md](docs/demos/smollm2_gpu1_backend_switch.md).

User-supplied paths on the CLI (`--file`, `--from-file`,
`--resource-profiles-file`, `--output-dir`) still resolve against the
current working directory — they're meant to behave as typed.

---

## Backend 1: Compose

Use Compose for local single-host deployments. It can render direct Ollama stacks, vLLM stacks, mixed Ollama+vLLM stacks, and raw backend-only stacks.

### Getting started

Prerequisite: Docker and the `docker compose` plugin must be installed.

```bash
# Direct Ollama, no LiteLLM and no predeclared models.
vllm-stack setup --backend compose --profile ollama-direct
vllm-stack validate --simulate-hardware 2x11
vllm-stack render --yes --simulate-hardware 2x11
vllm-stack up -d

# Classic vLLM through LiteLLM/Open WebUI.
vllm-stack setup --backend compose --profile qwen2-5-7b-instruct-turbo-default
vllm-stack validate
vllm-stack render
vllm-stack up -d
```

### Test that it is responding

When LiteLLM is enabled, the default Compose front door is:

```text
http://127.0.0.1:14042/v1
```

When using `ollama-direct`, Open WebUI talks to Ollama directly and the Ollama API is available at:

```text
http://127.0.0.1:11434
http://127.0.0.1:11434/v1
```

unless you changed the relevant ports in config.

Wait until the active profile can serve a real request through its resolved default endpoint:

```bash
vllm-stack wait-ready
```

`wait-ready` is stronger than Docker Compose health: it probes the user-facing
LiteLLM, Ollama, or direct vLLM access surface and, by default, requires a tiny
generation/completion to succeed. The smoke test runs this readiness probe by
default before issuing its normal test request:

```bash
vllm-stack smoke-test
```

For direct Ollama profiles, pull a model first and then smoke-test that model:

```bash
vllm-stack ollama-pull qwen3.5:4b
vllm-stack ollama-list
vllm-stack smoke-test --model qwen3.5:4b
```

For LiteLLM profiles, `smoke-test` reads the rendered `.env` automatically and
uses the active profile's resolved OpenAI-compatible front door. You can inspect
individual secrets when needed:

```bash
vllm-stack env LITELLM_MASTER_KEY
vllm-stack env VLLM_BACKEND_API_KEY
```

When you intentionally want the old quick behavior, skip the readiness wait:

```bash
vllm-stack smoke-test --no-wait --model gpt2
```

### Stop it

```bash
vllm-stack down
```

`down` never removes named volumes. The Postgres data directory and the
Open WebUI volume are preserved across `down`, `up`, `switch`, and `render`.

### Open WebUI authentication

By default Open WebUI runs with `WEBUI_AUTH=False` — no login screen,
anyone who can reach the port gets straight into the UI. This is the
expected behavior for a local dev box. To re-enable login/signup, set
in `config.yaml`:

```yaml
open_webui:
  auth: true
```

and re-render. Existing accounts stored in the `postgres-open-webui`
volume are preserved across the toggle.

### Persistent state and database layout

Compose renders stateful services only when their components are enabled:

* `postgres-open-webui` — rendered only when Open WebUI is enabled. It stores chats, accounts, and settings in `state.postgres_open_webui`.
* `postgres-litellm` — rendered only when LiteLLM is enabled. It stores router state in `state.postgres_litellm`.
* `ollama` — rendered only when the Ollama provider is enabled. Its model store is `state.ollama`, mounted at `/root/.ollama`.
* vLLM runtimes mount `state.hf_cache` for Hugging Face weights and `state.vllm_cache` for compiled artifacts.

Each Postgres container has its own `POSTGRES_DB`, `POSTGRES_USER`, and
`POSTGRES_PASSWORD`, sourced from component-specific `.env` keys. There is no shared Postgres instance and no `postgres-init` bootstrap service.

Open WebUI chat history is **not** tied to the model currently being served, so after a profile switch old chats may reference model IDs the current gateway no longer advertises — that is expected.

### Operational tips

Prefer scoping commands to specific services rather than relying on
container names. Use only the services rendered by the active profile:

```bash
# LiteLLM gateway profile
vllm-stack logs -f litellm

# Direct Ollama profile
vllm-stack logs -f ollama

# Ollama model store helpers
vllm-stack ollama-list
vllm-stack ollama-ps
```

You do not need to delete any volume during normal operation. If you
ever want a destructive reset, do it explicitly with
`docker compose down -v` against `generated/docker-compose.yml` — the
toolchain itself never does this.

### Custom .env values are preserved

`generated/.env` is rewritten non-destructively. Any `KEY=value` pair
you add manually (for example `VERBOSE=1`, `HF_HOME=/data/hf`, or any
key this program does not yet know about) is preserved across
`render`, `setup`, `switch`, `up`, and `deploy`. Comments and the order
of existing lines are preserved where practical.

### Switching profiles

```bash
vllm-stack switch <profile> --apply
```

`switch --apply` re-renders from the updated `config.yaml`, then brings the
stack up convergently with `--remove-orphans` so a separate `vllm-stack up` is
not needed. Components/runtimes that are no longer in the rendered compose file
are dropped. Compose preserves existing containers whose service definitions did
not change. For vLLM-to-vLLM profile switches, unchanged Open WebUI stays up;
LiteLLM is refreshed through its admin API when possible. The live refresh
path treats LiteLLM's "model not found in db" response for config-backed
models as non-fatal, so switching aliases can add the new route without
tearing LiteLLM down. That can temporarily leave stale config-backed aliases
in `/v1/models`; restart LiteLLM manually only when you want to clean those
up. If Compose already created or recreated LiteLLM while converging the new
stack, no extra router refresh is attempted because the new container has
already loaded the freshly rendered YAML. Profiles that do not render
LiteLLM, such as direct Ollama profiles, skip the router refresh path even if
an old `runtime/litellm_config.yaml` file remains from a previous profile.
Switches that change Open WebUI's provider wiring, such as
`Open WebUI -> LiteLLM` to `Open WebUI -> Ollama`, necessarily recreate
Open WebUI because its environment changes. Postgres volumes and provider
caches are left untouched. vLLM runtime containers are named after their
Compose service, for example `vllm-chat`, so `docker ps` and
`vllm-stack logs vllm-chat` clearly identify them as vLLM containers.

### Protocol modes for base vs. instruct models

Profiles declare a `protocol_mode` (`chat` or `completions`) that the
served model must support. Models also declare which protocols they
support via `supported_protocols`. Validation runs before render and
fails with an actionable message if a profile asks for `chat` on a
completions-only model.

Practical guidance:

* Instruct/chat models (with a chat template) can use either, but
  default to `chat`.
* Base models like Pythia, Llama-2 base, Mistral-v0.1 base, and Falcon
  base do not define a chat template. Their HELM profiles use
  `protocol_mode: completions` and the `smoke-test` command will
  exercise `/v1/completions` for them.
* The rendered LiteLLM config uses `text-completion-openai/<served>`
  as the upstream provider for completions-only services. That means
  even chat-shaped requests sent through Open WebUI to a Pythia model
  get translated by LiteLLM into upstream `/v1/completions` calls — no
  second vLLM container is needed to support Open WebUI for a
  completions-only model.
* Open WebUI is still a chat UI, so prompt formatting matters.
  HELM/eval clients should call `/v1/completions` directly for exact
  prompt control rather than going through the chat frontend.

### Chat-shaped clients on top of completions models

Some clients (e.g. InspectAI / Inspect Evals stock MMLU tasks) only
speak `/v1/chat/completions` and cannot be reconfigured. For those
cases, profiles can opt into a LiteLLM-only adapter:

```yaml
chat_compat:
  enabled: true
  strategy: flat_messages
```

When set on a `protocol_mode: completions` service, the rendered
LiteLLM config keeps the `text-completion-openai/<served>` upstream
and adds LiteLLM's documented prompt-template fields
(`initial_prompt_value` / `roles` / `final_prompt_value`) so chat
messages get flattened into a plain prompt — no role labels, messages
joined by `\n` — before being forwarded to vLLM `/v1/completions`.

This is **not** a chat tune; the model is still a base model and
prompt formatting still matters for evaluation. Use it only when a
chat-shaped client cannot be changed. The vLLM container is not
restarted, no `--chat-template` is rendered, and the adapter takes
effect after a `litellm`-only restart:

```bash
vllm-stack render
vllm-stack restart litellm
```

The built-in `pythia-inspect-mmlu-compat` profile is a ready-made
example; see
[`recipies/compose_pythia_inspect_mmlu_compat.md`](recipies/compose_pythia_inspect_mmlu_compat.md).

### Reasoning / thinking models

Models can declare reasoning support in the catalog:

```yaml
reasoning:
  enabled: true
  parser: qwen3
  expose_to_openwebui: true
```

Profiles can override or set the same field per service. When a
service has `reasoning.enabled: true` and a `parser`, the renderer
adds `--reasoning-parser <parser>` to that vLLM container's command
line — that flag alone enables reasoning extraction in the current
vLLM CLI. You do not need to repeat it by hand in `extra_args`.

Open WebUI sees reasoning content via two paths:

1. Inline `<think>...</think>` tags emitted by the model.
2. Structured `reasoning_content` fields when LiteLLM normalizes them.

The LiteLLM template keeps `merge_reasoning_content_in_choices: true`
on chat-mode entries so Open WebUI can display reasoning in the
streamed response. To test reasoning end-to-end:

```bash
# Non-streaming CLI smoke test:
vllm-stack smoke-test \
  --model qwen3.6-35b-a3b \
  --prompt "Think step by step: 17*23"

# For streaming inspection, read the key with the CLI wrapper:
LITELLM_MASTER_KEY=$(vllm-stack env LITELLM_MASTER_KEY)
curl -N http://127.0.0.1:14042/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","stream":true,
       "messages":[{"role":"user","content":"Think step by step: 17*23"}]}'
```

In Open WebUI, reasoning shows up best with streaming enabled in the
chat settings.

---

## Backend 2: KubeAI

Use KubeAI when you want Kubernetes-managed serving.

### Important rules

1. **Use the same namespace everywhere.** The namespace in `vllm-stack setup --namespace ...` must match the namespace where the KubeAI Helm release already exists.
2. **Prefer the repo-driven path.** The normal path is `setup` -> `validate` -> `render` -> `deploy` -> `status`.
3. **`kubectl port-forward` stays in the foreground.** Leave it running in one terminal and send requests from another.
4. **The first request can take a while.** `/openai/v1/models` may work before chat completions work. The first completion may trigger pod creation, image pull, model load, and compile warmup.
5. **On the current repo version, KubeAI still needs a live workaround after deploy.** The renderer currently produces a `Model` spec that needs a small manual patch to work with the KubeAI version used in these notes.

### KubeAI prerequisites

You need:

* a working Kubernetes cluster
* `kubectl`
* Helm

If you want a quick local single-node cluster, K3s is a good option.

Install K3s:

```bash
curl -sfL https://get.k3s.io | sh -
# or pin
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION='v1.34.3+k3s1' sh -
```

Make `kubectl` usable without `sudo`:

```bash
sudo mkdir -p /etc/rancher/k3s/config.yaml.d
printf 'write-kubeconfig-mode: "0644"\n' | \
  sudo tee /etc/rancher/k3s/config.yaml.d/10-kubeconfig-mode.yaml >/dev/null
sudo systemctl restart k3s
kubectl get nodes
```

Install Helm:

```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/83a46119086589a593a62ca544982977a60318ca/scripts/get-helm-4
chmod 700 get_helm.sh
./get_helm.sh
helm version
```

### NVIDIA GPU support

Install the NVIDIA device plugin and GPU Feature Discovery so Kubernetes can expose GPU resources and labels:

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm upgrade -i nvdp nvdp/nvidia-device-plugin \
  --version 0.17.1 \
  --namespace nvidia-device-plugin \
  --create-namespace \
  --set gfd.enabled=true \
  --set runtimeClassName=nvidia
```

Check that GPU support is working:

```bash
kubectl -n nvidia-device-plugin get pods
kubectl get node "$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')" \
  -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'
kubectl get nodes --show-labels | tr ',' '\n' | grep 'nvidia.com/' || true
```

You want to see a non-empty `nvidia.com/gpu` count and `nvidia.com/*` labels such as product and memory.

### KubeAI Helm repository

```bash
helm repo add kubeai https://www.kubeai.org
helm repo update
```

## Determine which namespace to use

Before doing anything else, discover whether a `kubeai` release already exists and which namespace it uses.

```bash
KUBEAI_NAMESPACE="$(helm list -A | awk '$1=="kubeai"{print $2; exit}')"
if [ -z "${KUBEAI_NAMESPACE}" ]; then
  KUBEAI_NAMESPACE=default
fi
echo "Using KubeAI namespace: ${KUBEAI_NAMESPACE}"
```

If a release already exists, **reuse that namespace**.

Sanity-check the cluster:

```bash
kubectl get nodes
kubectl get crd models.kubeai.org || true
helm list -A | grep kubeai || true
kubectl -n "${KUBEAI_NAMESPACE}" get pods || true
kubectl get node "$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')" \
  -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'
```

---

## Generate the local KubeAI resource-profile file

Generate a local KubeAI resource-profile file from the labels on this machine.

For the built-in serving profiles in this repo, keep these names aligned:

* `gpu-single-default`
* `gpu-tp2-balanced`
* `gpu-tp2-maxctx`

**Important:** include GPU `requests`, GPU `limits`, and `runtimeClassName: nvidia`. Without those, the model pod can land on the GPU node but still start without `libcuda.so.1` available inside the container.

```bash
PRODUCT="$(kubectl get nodes -o jsonpath='{.items[0].metadata.labels.nvidia\.com/gpu\.product}')"
MEMORY="$(kubectl get nodes -o jsonpath='{.items[0].metadata.labels.nvidia\.com/gpu\.memory}')"

cat > values-kubeai-local-gpu.yaml <<EOF
resourceProfiles:
  gpu-single-default:
    nodeSelector:
      nvidia.com/gpu.product: "${PRODUCT}"
      nvidia.com/gpu.memory: "${MEMORY}"
    requests:
      nvidia.com/gpu: 1
    limits:
      nvidia.com/gpu: 1
    runtimeClassName: nvidia

  gpu-tp2-balanced:
    nodeSelector:
      nvidia.com/gpu.product: "${PRODUCT}"
      nvidia.com/gpu.memory: "${MEMORY}"
    requests:
      nvidia.com/gpu: 2
    limits:
      nvidia.com/gpu: 2
    runtimeClassName: nvidia

  gpu-tp2-maxctx:
    nodeSelector:
      nvidia.com/gpu.product: "${PRODUCT}"
      nvidia.com/gpu.memory: "${MEMORY}"
    requests:
      nvidia.com/gpu: 2
    limits:
      nvidia.com/gpu: 2
    runtimeClassName: nvidia
EOF

cat values-kubeai-local-gpu.yaml
```

Sync that file so `validate` and `render` use the same local resource-profile data:

```bash
vllm-stack kubeai-sync-resource-profiles --from-file values-kubeai-local-gpu.yaml
```

---

## Example 1: single-GPU system

Use this example on a 1-GPU workstation.

```bash
vllm-stack setup \
  --backend kubeai \
  --profile qwen2-5-7b-instruct-turbo-default \
  --namespace "${KUBEAI_NAMESPACE}"

vllm-stack list-profiles
vllm-stack describe-profile qwen2-5-7b-instruct-turbo-default --format yaml
vllm-stack validate
vllm-stack render
vllm-stack deploy
vllm-stack status
```

### Current live workaround for the single-GPU example

On the current repo version, apply this live patch after `deploy`.

This patch does four things:

* keeps the model warm with `minReplicas: 1`
* changes `resourceProfile` from `gpu-single-default` to `gpu-single-default:1`
* makes the served model name match the public profile name
* avoids the duplicate `--served-model-name` mismatch that causes 404s on completions

```bash
kubectl -n "${KUBEAI_NAMESPACE}" patch model qwen2-5-7b-instruct-turbo-default --type merge -p '{
  "spec": {
    "minReplicas": 1,
    "resourceProfile": "gpu-single-default:1",
    "args": [
      "--served-model-name=qwen2-5-7b-instruct-turbo-default",
      "--tensor-parallel-size=1",
      "--data-parallel-size=1",
      "--max-model-len=32768",
      "--gpu-memory-utilization=0.9",
      "--max-num-batched-tokens=8192",
      "--max-num-seqs=16",
      "--disable-log-requests",
      "--enable-prefix-caching"
    ]
  }
}'

kubectl -n "${KUBEAI_NAMESPACE}" delete pod -l model=qwen2-5-7b-instruct-turbo-default
```

If you run `vllm-stack render` or `vllm-stack deploy` again on the current repo version, re-apply this live patch.

---

## Example 2: four-GPU system

On a 4-GPU host, do the same **single-GPU smoke test first** to verify the cluster, KubeAI, runtime class, and model plumbing. That exact sequence worked on a 4-GPU machine during bring-up.

```bash
vllm-stack setup \
  --backend kubeai \
  --profile qwen2-5-7b-instruct-turbo-default \
  --namespace "${KUBEAI_NAMESPACE}"

vllm-stack validate
vllm-stack render
vllm-stack deploy
vllm-stack status

kubectl -n "${KUBEAI_NAMESPACE}" patch model qwen2-5-7b-instruct-turbo-default --type merge -p '{
  "spec": {
    "minReplicas": 1,
    "resourceProfile": "gpu-single-default:1",
    "args": [
      "--served-model-name=qwen2-5-7b-instruct-turbo-default",
      "--tensor-parallel-size=1",
      "--data-parallel-size=1",
      "--max-model-len=32768",
      "--gpu-memory-utilization=0.9",
      "--max-num-batched-tokens=8192",
      "--max-num-seqs=16",
      "--disable-log-requests",
      "--enable-prefix-caching"
    ]
  }
}'

kubectl -n "${KUBEAI_NAMESPACE}" delete pod -l model=qwen2-5-7b-instruct-turbo-default
```

After the 7B smoke test works, move up to larger profiles such as `qwen2-72b-instruct-tp2-balanced`. On the current repo version, apply the same kind of live patch after deploy: keep `minReplicas: 1`, append `:1` to the chosen `resourceProfile`, and make the single effective `--served-model-name` match the public profile name.

---

## Test that KubeAI is responding

If you are not exposing ingress yet, port-forward the service.

**This command stays in the foreground.** Run it in one terminal and leave it there:

```bash
kubectl -n "${KUBEAI_NAMESPACE}" port-forward svc/kubeai 8000:80
```

Then use another terminal for requests.

### First check: `/models`

```bash
curl http://127.0.0.1:8000/openai/v1/models
```

If that works, the KubeAI front door is alive.

### Then try the smoke test

```bash
vllm-stack smoke-test \
  --base-url http://127.0.0.1:8000/openai/v1 \
  --model qwen2-5-7b-instruct-turbo-default
```

### Or test chat completions directly

```bash
time curl http://127.0.0.1:8000/openai/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2-5-7b-instruct-turbo-default",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 8
  }'
```

### What to expect on the first request

Common first-request behavior:

* `/openai/v1/models` works before completions work
* a completion request causes KubeAI to create a model-serving pod
* that pod may spend time in `ContainerCreating` while the image is pulled
* the model then spends more time loading and warming up
* the first completion can be much slower than later ones

That is not automatically a failure. Watch the system state while the first request is happening:

```bash
watch -n 1 'kubectl -n '"${KUBEAI_NAMESPACE}"' get pods; echo; kubectl -n '"${KUBEAI_NAMESPACE}"' get models'
```

---

## Debugging checks

### Check the live Model object

```bash
kubectl -n "${KUBEAI_NAMESPACE}" describe model qwen2-5-7b-instruct-turbo-default
kubectl -n "${KUBEAI_NAMESPACE}" get model qwen2-5-7b-instruct-turbo-default -o yaml | grep -E 'minReplicas|maxReplicas|resourceProfile'
```

### Check the current model pod

```bash
kubectl -n "${KUBEAI_NAMESPACE}" describe pod "$(kubectl -n "${KUBEAI_NAMESPACE}" get pods -o name | grep 'model-qwen2-5-7b-instruct-turbo-default' | tail -n 1 | cut -d/ -f2)"
```

### Tail KubeAI controller logs

```bash
kubectl -n "${KUBEAI_NAMESPACE}" logs deploy/kubeai --tail=200 -f
```

### Tail model-server logs

```bash
kubectl -n "${KUBEAI_NAMESPACE}" logs -f "$(kubectl -n "${KUBEAI_NAMESPACE}" get pods -o name | grep 'model-qwen2-5-7b-instruct-turbo-default' | tail -n 1 | cut -d/ -f2)" -c server
```

If the model pod restarted, inspect the previous crash:

```bash
kubectl -n "${KUBEAI_NAMESPACE}" logs "$(kubectl -n "${KUBEAI_NAMESPACE}" get pods -o name | grep 'model-qwen2-5-7b-instruct-turbo-default' | tail -n 1 | cut -d/ -f2)" -c server --previous
```

### Check recent events

```bash
kubectl -n "${KUBEAI_NAMESPACE}" get events --sort-by=.lastTimestamp | tail -n 40
```

### Common bad states and what they mean

* `invalid resource profile: "gpu-single-default", should match <name>:<multiple>`
  * append `:1` in the live `Model` spec
* `libcuda.so.1: cannot open shared object file`
  * the pod landed on the GPU node without actually requesting a GPU; fix the resource-profile file to include GPU requests, limits, and `runtimeClassName: nvidia`
* `/models` works but completions 404 with `The model ... does not exist.`
  * the served model name does not match the public profile name; apply the live args patch above
* startup probe fails with `connection refused`
  * the model pod may still be pulling the image, loading the model, or warming up

---

## Which backend should I start with?

Start with **Compose** if you want:

* the fastest path to a working local server
* easy inspection of generated files
* simple single-host iteration

Move to **KubeAI** when you want:

* vLLM runtimes on Kubernetes
* KubeAI’s OpenAI-compatible front door
* profile deployment through Kubernetes artifacts

KubeAI rendering is vLLM-only for now. Profiles that enable Ollama, LiteLLM, or Open WebUI are rejected for `--backend kubeai`.

A good workflow is:

1. inspect a profile with `describe-profile`
2. run it with Compose when you want the simplest local deployment
3. move to KubeAI when you want Kubernetes-backed serving

Compose is the better fit when you already know which profile you want. KubeAI has more first-request overhead because it may need to create pods, pull images, load the model, and warm up the backend.



## vLLM startup caches

Generated Compose mounts persist Hugging Face, vLLM, PyTorch/TorchInductor,
Triton, and CUDA JIT caches. Warm starts avoid redownloading and redoing many
compile/JIT steps, but a vLLM model swap still creates a new engine process and
must reload weights into GPU memory.

### Diagnosing profile switches and readiness

`docker compose` health only means that a container-level healthcheck passed.  It
is not the same thing as "the routed model can answer a request through the
active access surface."  This matters most when switching between two vLLM
profiles that reuse the same runtime service name: the old vLLM process exits,
Docker starts the replacement process, and LiteLLM may remain up while returning
upstream connection errors until vLLM finishes loading the new model.

Use the dedicated readiness and diagnostics commands after a switch:

```bash
vllm-stack switch gpt2-single --apply --yes
vllm-stack wait-ready --model gpt2
vllm-stack smoke-test --model gpt2
```

For debugging, use:

```bash
vllm-stack diagnose --model gpt2 --generation
vllm-stack diagnose --logs --tail 80
```

`diagnose` prints the resolved provider/gateway/frontend graph, rendered Compose
service state, LiteLLM route probes, direct provider probes, and optional recent
logs.  It is intended to distinguish an actual LiteLLM outage from the more
common case where LiteLLM is running but its upstream vLLM runtime is still
booting.

The Compose service-state diagnostics include Docker's exit code, OOM-killed
flag, restart count, and actual container name.  This is important because
`litellm exited with code 137` usually means Docker sent SIGKILL, commonly from
an OOM kill or a forced container replacement, whereas LiteLLM returning HTTP
500 with `Cannot connect to host vllm-*` means LiteLLM is still running but the
upstream vLLM runtime is not ready yet.
