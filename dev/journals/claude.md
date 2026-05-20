## 2026-05-20 15:09:24 -0400

Model: claude-opus-4-7[1m] (Opus 4.7, 1M context).

User intent: rendered Compose / KubeAI artifacts were being written to
`./generated/` inside the repo checkout, which meant two users developing
against the same checkout on one host would stomp on each other's files.
The request is to move the canonical output location off-tree and to a
machine-wide directory while still letting each user keep their own
input profiles (`config.yaml` / `models.yaml` / KubeAI local values).
Suggested default: `/data/service/docker/vllm-stack/generated`.

What I changed:

- New `output.generated_dir` section in `config.yaml`, with a default
  resolver that mirrors `_default_storage_root`: prefer
  `/data/service/docker/vllm-stack/generated` when `/data/service/docker`
  exists, else fall back to `./generated`. Override precedence: CLI flag
  `--generated-dir` → env var `VLLM_SERVICE_GENERATED_DIR` →
  `config.yaml` → default. `setup` bakes the resolved value into
  `config.yaml` so it's visible/editable.
- Plumbed `output.generated_dir` (absolute) into the resolved deployment
  dict alongside `state` and `cluster`. Both backend renderers
  (`compose_renderer`, `kubeai_renderer`), `kubeai_ops`, `exporters`,
  and `verification` read from there. Each falls back to the old
  `<root>/generated` layout when the deployment doesn't carry an
  `output` section, so direct test callers that build a deployment by
  hand keep working without needing to know about the new field.
- CLI `generated_dir()`, `plan_path()`, `kubeai_generated_dir()` now
  take cfg; threaded cfg through every call site. The KubeAI README
  emitted by the renderer references the actual rendered path instead
  of the hard-coded `generated/kubeai/...` so the printed instructions
  stay correct when the output is off-tree.
- Tests: `test_serving_profiles._cfg` pins
  `output.generated_dir = "generated"` so the resolver-populated
  deployment dict points at `tmp_path/generated` on dev machines where
  `/data/service/docker` exists. `test_cli_setup.run_cli` sets the env
  var so the subprocess flow + persisted config.yaml stay anchored on
  `tmp_path`. Added three new targeted tests covering custom output
  dir for compose, kubeai, and `normalized_output` path anchoring.

State of mind / reflections:

The fallback choice was the central design decision. The renderers
could (a) always require `deployment["output"]` to be populated, or (b)
fall back to `<root>/generated` when missing. Option (b) preserved a
lot of direct-call test surface that builds the deployment dict
manually, and matched how `normalized_state` already behaves
(callers can omit the section and get sensible defaults). The cost is
two near-identical fallback stanzas in the renderers and kubeai_ops,
which I considered consolidating into a helper but left inline because
each call site reads slightly different bits of the deployment.

What might break: existing operator workflows running on a host where
`/data/service/docker` already exists will see new renders go to
`/data/service/docker/vllm-stack/generated` even without explicit
config changes. That's the desired behavior, but anyone with a running
stack pinned to `./generated/docker-compose.yml` would need to either
take it down via the old path or override `output.generated_dir` to
`generated` in `config.yaml`. The README documents this; I didn't add a
migration script because the old `down`/cleanup flow is unchanged when
operators point the same `--generated-dir` at the previous location.

Pre-existing test failures: 13 tests in `test_cli_setup.py` fail
identically with my changes stashed — they exercise `render` without
`--yes` via subprocess, which hangs on the Rich confirm prompt. Not
in scope here; flagged but not fixed.

Design takeaways:
1. Output dir is a deployment-shaped fact, not a CLI fact. Putting it
   in the resolved deployment alongside `state` made every backend and
   downstream verifier converge on one source without each rediscovering
   it from cfg/root.
2. When introducing a config section that already has a sensible
   "anchor on root" interpretation, mimicking the existing
   `normalized_state` pattern saves a downstream surprise: relative
   paths in config behave the same way state paths do.
3. Tests that exercise renderers directly (no CLI) are a load-bearing
   constraint when changing where artifacts land. Preserving a
   `<root>/<dirname>` fallback in the renderer itself avoided a sweeping
   test refactor and kept the abstraction useful for ad-hoc tooling.
