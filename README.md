# vLLM Service

`vllm_service` manages **named serving profiles** for local and Kubernetes-backed inference.

This repo can render and run those profiles through two backends:

* **Compose** for local single-host serving
* **KubeAI** for Kubernetes-backed serving

A serving profile bundles the model, the public served name, placement, and runtime settings.

## Main commands

```bash
python manage.py setup --backend compose --profile qwen2-5-7b-instruct-turbo-default
python manage.py list-profiles
python manage.py describe-profile <profile>
python manage.py validate
python manage.py render
python manage.py up -d
python manage.py deploy
python manage.py switch <profile> --apply
python manage.py status
python manage.py smoke-test
```

## Inspect a profile before running it

```bash
python manage.py describe-profile qwen2-5-7b-instruct-turbo-default --format yaml
```

## Where rendered artifacts live

Rendered Compose / KubeAI artifacts go into a **machine-wide** output
directory rather than the repo checkout, so multiple users can develop
against the same repo on one host without stomping on each other's
generated files.

Default target:

```text
/data/service/docker/vllm-stack/generated/
  docker-compose.yml
  .env
  plan.yaml
  kubeai/...
```

The default kicks in whenever `/data/service/docker/` exists on the
host (the same convention `state.*` paths use). On other machines —
including CI and tests — the renderer falls back to `./generated/`
relative to the repo checkout, preserving the old behavior.

Override per user, in order of precedence:

1. `--generated-dir /path/to/out` on any command that accepts overrides
   (`setup`, `render`, `up`, `deploy`, `switch`, ...).
2. `VLLM_SERVICE_GENERATED_DIR=/path/to/out` env var.
3. `output.generated_dir` in `config.yaml` (persisted by `setup` from
   whichever of the above was in effect when you ran it).

Examples:

```bash
python manage.py setup --backend compose --profile <p> \
  --generated-dir /home/$USER/vllm-out
VLLM_SERVICE_GENERATED_DIR=/tmp/scratch python manage.py render
```

`config.yaml`, `models.yaml`, and `kubeai-values.local.yaml` remain
user-local in your working directory — only the **output** of the
renderer moves to the shared location.

---

## Backend 1: Compose

Use Compose when you know exactly which profile you want and want the lowest-friction local deployment.

### Getting started

Prerequisite: Docker and the `docker compose` plugin must be installed.

```bash
python manage.py setup --backend compose --profile qwen2-5-7b-instruct-turbo-default
python manage.py validate
python manage.py render
python manage.py up -d
```

### Test that it is responding

```bash
python manage.py smoke-test
```

The default Compose front door is:

```text
http://127.0.0.1:14000/v1
```

unless you changed the LiteLLM port in config.

List models:

```bash
curl http://127.0.0.1:14000/v1/models
```

Send a request:

```bash
curl http://127.0.0.1:14000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen/qwen2.5-7b-instruct-turbo",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }'
```

### Stop it

```bash
python manage.py down
```

`down` never removes named volumes. The Postgres data directory and the
Open WebUI volume are preserved across `down`, `up`, `switch`, and `render`.

### Persistent state and database layout

Compose runs **two** separate Postgres containers:

* `postgres-open-webui` — used only by Open WebUI for chats, accounts,
  and settings. Backed by `state.postgres_open_webui`.
* `postgres-litellm` — used only by LiteLLM for router state. Backed by
  `state.postgres_litellm`.

Each container has its own `POSTGRES_DB`, `POSTGRES_USER`, and
`POSTGRES_PASSWORD`, sourced from `.env` keys:

* `OPENWEBUI_POSTGRES_DB` / `_USER` / `_PASSWORD`
* `LITELLM_POSTGRES_DB` / `_USER` / `_PASSWORD`

There is no shared Postgres instance and no `postgres-init` bootstrap
service — each container creates its own database the first time its
volume is initialized.

Open WebUI chat history lives in the Open WebUI Postgres
container/volume. LiteLLM state lives in the LiteLLM Postgres
container/volume. Chat history is **not** tied to the model currently
being served, so after a profile switch old chats may reference model
IDs the router no longer advertises — that is expected.

### Operational tips

Prefer scoping commands to specific services rather than relying on
container names:

```bash
docker compose -f generated/docker-compose.yml --env-file generated/.env logs -f litellm
docker compose -f generated/docker-compose.yml --env-file generated/.env exec postgres-open-webui psql -U "$OPENWEBUI_POSTGRES_USER" "$OPENWEBUI_POSTGRES_DB"
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
python manage.py switch <profile> --apply
```

`switch --apply` re-renders from the updated `config.yaml`, brings the
stack up convergently with `--remove-orphans` (so vLLM services that
are no longer in the rendered compose file are dropped), and — if the
LiteLLM router config changed — force-recreates the `litellm` and
`open-webui` containers in place so they pick up the new alias list.
Both Postgres volumes are left untouched.

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
python manage.py render
docker compose -f generated/docker-compose.yml --env-file generated/.env \
  up -d --no-deps --force-recreate litellm
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
# Direct LiteLLM curl (streaming):
source generated/.env
curl -N http://127.0.0.1:14000/v1/chat/completions \
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

1. **Use the same namespace everywhere.** The namespace in `python manage.py setup --namespace ...` must match the namespace where the KubeAI Helm release already exists.
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
python manage.py kubeai-sync-resource-profiles --from-file values-kubeai-local-gpu.yaml
```

---

## Example 1: single-GPU system

Use this example on a 1-GPU workstation.

```bash
python manage.py setup \
  --backend kubeai \
  --profile qwen2-5-7b-instruct-turbo-default \
  --namespace "${KUBEAI_NAMESPACE}"

python manage.py list-profiles
python manage.py describe-profile qwen2-5-7b-instruct-turbo-default --format yaml
python manage.py validate
python manage.py render
python manage.py deploy
python manage.py status
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

If you run `python manage.py render` or `python manage.py deploy` again on the current repo version, re-apply this live patch.

---

## Example 2: four-GPU system

On a 4-GPU host, do the same **single-GPU smoke test first** to verify the cluster, KubeAI, runtime class, and model plumbing. That exact sequence worked on a 4-GPU machine during bring-up.

```bash
python manage.py setup \
  --backend kubeai \
  --profile qwen2-5-7b-instruct-turbo-default \
  --namespace "${KUBEAI_NAMESPACE}"

python manage.py validate
python manage.py render
python manage.py deploy
python manage.py status

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
python manage.py smoke-test \
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

* the same profile model on Kubernetes
* KubeAI’s OpenAI-compatible front door
* profile deployment through Kubernetes artifacts

A good workflow is:

1. inspect a profile with `describe-profile`
2. run it with Compose when you want the simplest local deployment
3. move to KubeAI when you want Kubernetes-backed serving

Compose is the better fit when you already know which profile you want. KubeAI has more first-request overhead because it may need to create pods, pull images, load the model, and warm up the backend.

