# LoRA Integration Plan — vllm_service (dev prototype)

Date: 2026-05-20

Summary
-------
Prototype a host-side LoRA merge workflow that: (1) detects a LoRA hint
in a serving profile, (2) merges the LoRA into the base model on the host
using `peft`/`transformers`, (3) writes a merged HF-style model directory
under the runtime `hf_cache`, and (4) renders the compose stack to point
the vLLM container at that merged path. This avoids modifying vLLM images
or runtime containers for the first iteration.

Goals
-----
- Minimal invasiveness: no vLLM image rebuilds.
- Developer-friendly: tools live under `dev/lora/` for experimentation.
- Idempotent, safe merges with clear error messages and a fallback path.

Approach (brief)
-----------------
- Add a small helper `dev/lora/merge_lora.py` that loads base + LoRA via
  `transformers` + `peft`, applies the adapter, and saves the merged
  model to a host path under the resolved `hf_cache` state.
- During `render` (compose backend) call the helper for services that
  declare a `lora:` hint and rewrite the service `hf_model_id` to the
  merged path (container path under `/root/.cache/huggingface/...`).

Detailed Steps
--------------
1. Add dev utilities
   - `submodules/vllm_service/dev/lora/merge_lora.py` (core API).
   - `submodules/vllm_service/dev/lora/cli.py` (thin CLI wrapper for manual testing).
   - `submodules/vllm_service/dev/lora/requirements-dev.txt` (pins: transformers, peft, safetensors, torch).
   - `submodules/vllm_service/dev/lora/README.md` (usage & limitations).

2. Propagate profile hint
   - Update resolver to preserve a `lora` hint on resolved services
     (small, non-breaking pass-through). Example field: `svc['lora']`.

3. Render-time merge
   - Modify `render_compose_artifacts` (compose renderer) to, before
     templating, iterate services and for any with `lora` call the
     merge helper.
   - Host output location: `${deployment.state.hf_cache}/merged_loras/<profile>/<service>/<revision>/`.
   - Set `svc['hf_model_id']` to the container-side path
     `/root/.cache/huggingface/merged_loras/<profile>/<service>/<revision>` so
     the compose template (already mounting `hf_cache`) exposes it.

4. Idempotency and caching
   - Merge helper should skip if output exists and `revision`/hash matches.
   - Provide `--force` and `--dry-run` flags on the CLI.

5. Tests
   - Unit: mock `transformers`/`peft` to verify expected calls and files written.
   - Integration: smoke merge using a tiny HF test model or a local fake repo.
   - Render test: call `render_compose_artifacts` on a plan with `lora` and assert
     `docker-compose.yml` references `/root/.cache/huggingface/merged_loras/...`.

6. Documentation
   - Document profile syntax examples in `dev/lora/README.md` and add a short
     example to `templates/default-profiles.yaml` as an example (optional).

Files to create / modify
------------------------
- Create: `submodules/vllm_service/dev/lora/merge_lora.py`  -- merge API.
- Create: `submodules/vllm_service/dev/lora/cli.py`         -- developer CLI.
- Create: `submodules/vllm_service/dev/lora/requirements-dev.txt` -- dev deps.
- Create: `submodules/vllm_service/dev/lora/README.md`      -- instructions.
- Modify: `submodules/vllm_service/vllm_service/resolver.py` -- propagate `lora` hint.
- Modify: `submodules/vllm_service/vllm_service/backends/compose_renderer.py` -- call merge helper and patch `svc['hf_model_id']`.

Merge helper: design notes
--------------------------
- Implementation (preferred):
  - Use `transformers.AutoModelForCausalLM` (or AutoModel depending on family)
    and `peft.PeftModel.from_pretrained(lora_path, model=base_model)`.
  - After applying the adapter, call `model.save_pretrained(out_dir)` and
    copy tokenizer files (via `AutoTokenizer.from_pretrained(base)` and
    `tokenizer.save_pretrained(out_dir)`).
  - Use `safetensors` when possible for adapter weights.

- Outputs:
  - Host dir: `<hf_cache>/merged_loras/<profile>/<service>/<revision>/`
  - Container path (rendered into compose): `/root/.cache/huggingface/merged_loras/...`

Profile syntax (example)
------------------------
Add one of these to a profile service in your `config.yaml` / `models.yaml`:

```yaml
services:
  - service_name: myservice
    base_model: qwen/qwen2.5-7b-instruct-turbo
    lora:
      repo: hf-user/my-lora
      revision: v1
      files: ["adapter.safetensors"]   # optional
      merge_strategy: peft
```

Render integration flow (summary)
---------------------------------
1. `python manage.py render --profile <profile-with-lora>` builds plan.
2. `render_compose_artifacts` normalizes state and computes `hf_cache`.
3. For services with `lora`, call `merge_lora.merge(...)` to produce host dir.
4. Replace `svc['hf_model_id']` with container path under `/root/.cache/huggingface/...`.
5. Render templates: compose mounts `hf_cache` so vLLM sees the merged model.

Fallbacks & limitations
-----------------------
- Quantized models (bitsandbytes / gptq / awq / fp8) may not be mergeable via
  `peft`. In those cases require a pre-merged artifact (user-provided local path)
  or mark the service as unsupported and surface a clear error.
- Mismatched tokenizer/architecture will fail — helper should validate and
  produce actionable errors.
- Merging large models is CPU/GPU and disk intensive; document requirements.

Developer commands (quick)
--------------------------
Install dev deps (example):

```bash
python -m pip install -r submodules/vllm_service/dev/lora/requirements-dev.txt
```

Manual merge test:

```bash
python submodules/vllm_service/dev/lora/cli.py \
  --base qwen/qwen2.5-7b-instruct-turbo \
  --lora hf-user/lora-repo --revision v1 \
  --out generated/hf-cache/merged_loras/qwen/myprofile-serviceA
```

Render after adding `lora` to profile:

```bash
cd submodules/vllm_service
python manage.py render --profile <profile-with-lora>
```

Timeline
--------
- Prototype (dev folder, merge helper, resolver+renderer hook, README, unit tests): ~6–16 hours.
- Hardening (CI, quantized handling, kubeai path): additional 1–3 days.

Next steps
----------
Confirm and I will implement the prototype files under
`submodules/vllm_service/dev/lora/` and the small resolver/renderer hooks.
