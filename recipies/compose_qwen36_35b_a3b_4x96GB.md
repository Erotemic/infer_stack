# Compose recipe: Qwen3.6-35B-A3B on a 4x96GB host

This is the shortest end-to-end example for serving **Qwen/Qwen3.6-35B-A3B** on a machine with **4 x 96GB GPUs** using the **Compose** backend.

This profile uses:

- one model instance
- GPUs 0 and 1
- tensor parallel size 2
- text-only mode
- native **262,144** token context
- the Qwen reasoning parser
- higher batching and concurrency for more users
- Open WebUI on top of LiteLLM

---

## 1. Start from the repo root

```bash
cd /path/to/vllm_service
```

---

## 2. Initialize Compose config

```bash
python manage.py setup --backend compose
```

---

## 3. Write the local model and profile definitions

Overwrite `models.yaml` with:

```bash
cat > models.yaml <<'EOF'
models:
  qwen3.6-35b-a3b-local:
    hf_model_id: Qwen/Qwen3.6-35B-A3B
    tokenizer_name: Qwen/Qwen3.6-35B-A3B
    served_model_name: qwen3.6-35b-a3b-262k
    family: qwen3.6
    modalities: [text]
    memory_class_gib: 80
    min_vram_gib_per_replica: 24
    preferred_gpu_count: 2
    context_window: 262144
    defaults:
      max_model_len: 262144
      gpu_memory_utilization: 0.95
      enable_prefix_caching: false
      max_num_batched_tokens: 8192
      max_num_seqs: 8

profiles:
  qwen3.6-35b-a3b-tp2-262k-local:
    description: "Single Qwen3.6-35B-A3B service across 2 GPUs at 262k context."
    vllm:
      enable_responses_api_store: false
      logging_level: INFO
    services:
      - service_name: qwen36-35b
        model: qwen3.6-35b-a3b-local
        served_model_name: qwen3.6-35b-a3b-262k
        placement:
          strategy: exact
          gpu_indices: [0, 1]
        topology:
          tensor_parallel_size: 2
        runtime:
          max_model_len: 262144
          gpu_memory_utilization: 0.95
          max_num_batched_tokens: 8192
          max_num_seqs: 8
          enable_prefix_caching: false
        extra_args:
          - --language-model-only
          - --reasoning-parser
          - qwen3

    router:
      aliases:
        qwen3.6-35b-a3b-262k: qwen36-35b
EOF
```

---

## 4. Switch to the local profile

```bash
python manage.py switch qwen3.6-35b-a3b-tp2-262k-local
```

Optional sanity check before rendering:

```bash
python manage.py describe-profile qwen3.6-35b-a3b-tp2-262k-local --format yaml
```

---

## 5. Render the stack

```bash
python manage.py render
```

This creates:

- `generated/docker-compose.yml`
- `generated/.env`

---

## 6. Set your Hugging Face token

Add your token to `generated/.env`:

```bash
grep '^HF_TOKEN=' generated/.env
```

Then edit the file and set:

```text
HF_TOKEN=your_token_here
```

---

## 7. Start the stack

From the repo root:

```bash
docker compose -f generated/docker-compose.yml --env-file generated/.env up -d
```

---

## 8. Verify the backend

Check that the model is exposed:

```bash
curl http://127.0.0.1:18000/v1/models \
  -H "Authorization: Bearer $(grep '^VLLM_BACKEND_API_KEY=' generated/.env | cut -d= -f2-)"
```

You should see:

- `qwen3.6-35b-a3b-262k`

---

## 9. Open the UI

Open:

```text
http://127.0.0.1:13000
```

On first run, create the Open WebUI admin account.

Then select the model:

- `qwen3.6-35b-a3b-262k`

---

## 10. Optional repo smoke test

From the repo root:

```bash
python manage.py smoke-test \
  --model qwen3.6-35b-a3b-262k \
  --prompt "Say hello in one sentence."
```

---

## Summary

This profile serves:

- `Qwen/Qwen3.6-35B-A3B`
- on 2 of the 4 x 96GB GPUs
- with tensor parallel size 2
- at 262,144 token context
- with `max_num_batched_tokens: 8192`
- with `max_num_seqs: 8`
- with `--reasoning-parser qwen3`
- through both the direct backend and Open WebUI
