# Compose recipe: Pythia 6.9B + Pythia 2.8B + Qwen3.6-35B-A3B on a 4x96GB host

This recipe uses the built-in `pythia-qwen3.6-mixed-4x96` profile to serve
three models behind a single LiteLLM router on a machine with **4 x 96GB
GPUs**:

| Service       | Model                       | GPUs   | Protocol     |
|---------------|-----------------------------|--------|--------------|
| `qwen36-35b`  | `Qwen/Qwen3.6-35B-A3B`      | 0, 1   | chat (TP2, reasoning) |
| `pythia-69b`  | `EleutherAI/pythia-6.9b`    | 2      | completions  |
| `pythia-28b`  | `EleutherAI/pythia-2.8b-v0` | 3      | completions  |

The Qwen service has `--enable-reasoning --reasoning-parser qwen3`
auto-rendered by the profile. The two Pythia services are routed
through LiteLLM as `text-completion-openai/...` so chat-shaped requests
from Open WebUI are translated into upstream `/v1/completions` calls.

---

## 1. Setup, render, run

```bash
cd /path/to/vllm_service
python manage.py setup --backend compose --profile pythia-qwen3.6-mixed-4x96
python manage.py render
python manage.py up -d
```

If you already have the stack running on a different profile:

```bash
python manage.py switch pythia-qwen3.6-mixed-4x96 --apply
```

`switch --apply` re-renders, brings the stack up convergently, and
recreates `litellm` + `open-webui` in place so the new alias list takes
effect. Postgres volumes are not touched.

---

## 2. Set your Hugging Face token (Qwen3.6 is gated)

```bash
grep '^HF_TOKEN=' generated/.env
```

Edit `generated/.env` and set `HF_TOKEN=...`. Unknown `.env` keys
(yours and ours) are preserved across re-renders.

---

## 3. Verify the routes

```bash
source generated/.env
curl -s http://127.0.0.1:14000/v1/models \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" | jq '.data[].id'
```

Expected aliases:

- `qwen3.6-35b-a3b`
- `eleutherai/pythia-6.9b`
- `eleutherai/pythia-2.8b-v0`

Smoke-test each (the smoke-test picks `/v1/chat/completions` or
`/v1/completions` based on the profile's protocol):

```bash
python manage.py smoke-test --model qwen3.6-35b-a3b
python manage.py smoke-test --model eleutherai/pythia-6.9b
python manage.py smoke-test --model eleutherai/pythia-2.8b-v0
```

---

## 4. Open WebUI

Open `http://127.0.0.1:13000`. All three model aliases will appear.
Reasoning streams from the Qwen3.6 service and is displayed inline
when chat streaming is enabled. The two Pythia models will respond,
but remember they are base/completions models — prompt formatting
matters and chat-style multi-turn use will not produce great results
without explicit prompt strategies.

For exact prompt control on the Pythia models, prefer direct
`/v1/completions` against LiteLLM:

```bash
curl -s http://127.0.0.1:14000/v1/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"eleutherai/pythia-6.9b","prompt":"The capital of France is","max_tokens":16}'
```

---

## Summary

- Three vLLM containers, one shared LiteLLM router, separate Postgres
  containers for Open WebUI and LiteLLM.
- Reasoning flags are emitted from profile metadata, not hand-written.
- Completions-only models are usable from chat clients via LiteLLM's
  `text-completion-openai` provider, with no extra vLLM containers.
