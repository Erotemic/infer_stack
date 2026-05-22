# Compose recipe: Qwen3.5-122B-A10B-FP8 on a 4x96GB host

This is the shortest working end-to-end example for serving **Qwen/Qwen3.5-122B-A10B-FP8** on a machine with **4 x 96GB GPUs** using the **Compose** backend.

This profile uses:

- one model instance
- all 4 GPUs
- tensor parallel size 4
- text-only mode
- native **262,144** token context
- Open WebUI on top of LiteLLM

---

## Quick path: use the built-in profile

Available out of the box as the built-in profile
`qwen3.5-122b-a10b-fp8-tp4-4x96`. To skip writing local YAML:

```bash
vllm-stack setup --backend compose --profile qwen3.5-122b-a10b-fp8-tp4-4x96
vllm-stack render --yes
vllm-stack up -d
```

The rest of this recipe walks through the same deployment manually.

---

## 1. Start from the repo root

```bash
cd /path/to/vllm_service
```

Optional sanity check:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
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
  qwen3.5-122b-a10b-fp8-local:
    hf_model_id: Qwen/Qwen3.5-122B-A10B-FP8
    tokenizer_name: Qwen/Qwen3.5-122B-A10B-FP8
    served_model_name: qwen3.5-122b-a10b-fp8-262k
    family: qwen3.5
    modalities: [text]
    memory_class_gib: 80
    min_vram_gib_per_replica: 80
    preferred_gpu_count: 4
    context_window: 262144
    defaults:
      max_model_len: 262144
      gpu_memory_utilization: 0.95
      enable_prefix_caching: false
      max_num_batched_tokens: 1024
      max_num_seqs: 1

profiles:
  qwen3.5-122b-a10b-fp8-tp4-262k-local:
    description: "Single Qwen3.5-122B-A10B-FP8 service across all 4 GPUs at 262k context."
    vllm:
      enable_responses_api_store: false
      logging_level: INFO
    services:
      - service_name: qwen-122b-fp8
        model: qwen3.5-122b-a10b-fp8-local
        served_model_name: qwen3.5-122b-a10b-fp8-262k
        placement:
          strategy: exact
          gpu_indices: [0, 1, 2, 3]
        topology:
          tensor_parallel_size: 4
        runtime:
          max_model_len: 262144
          gpu_memory_utilization: 0.95
          max_num_batched_tokens: 1024
          max_num_seqs: 1
          enable_prefix_caching: false
        extra_args:
          - --language-model-only
          - --reasoning-parser
          - qwen3

    router:
      aliases:
        qwen3.5-122b-a10b-fp8-262k: qwen-122b-fp8
EOF
```

---

## 4. Switch to the local profile

```bash
python manage.py switch qwen3.5-122b-a10b-fp8-tp4-262k-local
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
curl http://127.0.0.1:18000/v1/models   -H "Authorization: Bearer $(grep '^VLLM_BACKEND_API_KEY=' generated/.env | cut -d= -f2-)"
```

You should see:

- `qwen3.5-122b-a10b-fp8-262k`

---

## 9. Open the UI

Open:

```text
http://127.0.0.1:13000
```

On first run, create the Open WebUI admin account.

Then select the model:

- `qwen3.5-122b-a10b-fp8-262k`

---

## 10. Optional repo smoke test

From the repo root:

```bash
python manage.py smoke-test   --model qwen3.5-122b-a10b-fp8-262k   --prompt "Say hello in one sentence."
```

---

## Summary

This profile serves:

- `Qwen/Qwen3.5-122B-A10B-FP8`
- on 4 x 96GB GPUs
- with tensor parallel size 4
- at 262,144 token context
- through both the direct backend and Open WebUI
