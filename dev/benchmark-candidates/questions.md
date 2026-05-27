# Benchmark questions

## Q001 — Docker Compose `depends_on` cascades recreates into dependents

Tags: `lifecycle-ordering`, `dependency-boundary`, `docker-compose`

Source:
- Buggy state: any commit in `91b2586..32be174^` (the `depends_on` line
  has existed since the compose template was introduced in 91b2586).
- Fix commit: 32be174 ("Hot-swap models without dropping Open WebUI") —
  see `infer_stack/templates/docker-compose.yml.j2`.
- Conversation: 2026-05-22 entry in `dev/journals/claude.md`.

### Prompt

You maintain a small docker-compose stack with three services in a chain:

```yaml
services:
  postgres-ui:
    image: postgres:16
    healthcheck:
      test: ["CMD-SHELL", "pg_isready"]
      interval: 10s

  router:
    image: my-router:latest
    container_name: router
    depends_on:
      postgres-ui:
        condition: service_healthy
    volumes:
      - ./router-config.yaml:/app/config.yaml:ro
    # ... (long-running, multi-minute startup)

  ui:
    image: my-ui:latest
    container_name: ui
    depends_on:
      postgres-ui:
        condition: service_healthy
      router:
        condition: service_started
    environment:
      ROUTER_URL: http://router:4000
    # ... (browser-facing, holds long-lived WebSocket sessions)
```

Users connect to `ui` via long-lived WebSocket sessions. An operator
periodically swaps the router's upstream config: rewrites
`router-config.yaml` on disk, then runs `docker compose up -d`. Compose
detects no change to the `router` container spec itself (only the
mounted file content changed), so they expect a no-op.

In production they instead observe that **`ui` recreates on every
`docker compose up -d` after `router-config.yaml` is edited**, dropping
all open WebSocket sessions. Diffing the rendered compose files
byte-for-byte shows the `ui` block is identical. Container labels
including `com.docker.compose.config-hash` are also identical.

Implement the minimal change to the compose file that keeps `ui` running
across these refreshes, without breaking initial startup ordering.
Briefly explain (≤4 sentences) why your change works and which Compose
mechanism caused the original bounce.

### Expected answer

Drop `router: {condition: service_started}` from `ui.depends_on`. The
final block becomes:

```yaml
  ui:
    image: my-ui:latest
    container_name: ui
    depends_on:
      postgres-ui:
        condition: service_healthy
    environment:
      ROUTER_URL: http://router:4000
```

Explanation that should be present:

1. The trigger isn't config-hash equality on `ui` itself. Compose v2's
   recreate planner re-evaluates dependents of a recreated service via
   the `depends_on` graph and will recreate them even when their own
   resolved config is byte-identical. This is not visible in the
   `com.docker.compose.config-hash` label.
2. `condition: service_started` is documented as a startup-ordering
   hint, so it's surprising that it doubles as a recreate trigger —
   that's the trap.
3. `postgres-ui: service_healthy` is kept because `ui` genuinely cannot
   start without its database. Losing the router dependency is safe as
   long as `ui` retries connections to the router (most browser-facing
   apps already do).
4. After the change, `compose up -d` post-config-rewrite recreates only
   the router. `ui` keeps its TCP listeners and any in-flight
   WebSockets continue.

Acceptable variant: keep `depends_on: router` but set
`required: false` (Compose v2.20+) — same effect on the cascade. Adding
`restart: true` is wrong; that *increases* the cascade.

### Validation

```bash
# 1. Bring the stack up.
docker compose up -d
docker inspect ui --format='{{.State.StartedAt}}' > /tmp/ui-started

# 2. Edit router config and apply.
echo "# touched" >> router-config.yaml
docker compose up -d

# 3. ui must NOT have been recreated.
test "$(docker inspect ui --format='{{.State.StartedAt}}')" = "$(cat /tmp/ui-started)"
```

Expected result: command 3 exits 0 (timestamp unchanged) after the fix.
Before the fix, command 3 fails (timestamp advanced — `ui` was
recreated).

### Why this was easy to miss

- The Compose docs describe `depends_on` as a *startup ordering*
  mechanism. Nothing in the public docs warns that it doubles as a
  recreate-cascade trigger.
- The `com.docker.compose.config-hash` label on each container is
  identical across renders, so the obvious diagnostic ("did compose
  detect a config change?") points the wrong way.
- The fix feels like removing a real invariant: the dep *is* needed for
  cold-start ordering, just not for ongoing operation. You have to
  trust the dependent's retry behavior to delete the line.
- The cascade does not require `--always-recreate-deps`. It happens
  with a plain `docker compose up -d` in the version range tested
  (Compose v2.x).

### Notes

The real-repo version of this had four layers obscuring the diagnosis
(see 2026-05-22 entry in `dev/journals/claude.md`):

1. The user's first hypothesis was force-recreate in our fallback path —
   that was a real bug, but fixing it didn't stop the bounce.
2. The live-refresh path (which avoided container recreate entirely)
   was failing for an unrelated reason (`store_model_in_db: False`
   meant `/model/delete` couldn't find YAML-loaded models in the DB),
   so the fallback kept firing.
3. Even with the fallback narrowed to recreate *only* the router
   container, the dependent UI still bounced — that's the question
   above.
4. Diffing two renders byte-for-byte to confirm `ui`'s block was
   identical was what isolated the issue to the `depends_on` cascade
   rather than to anything we control in the rendered config.

Related composition candidate: "preserve user sessions during a
profile/config swap" — combines this question with (a) `os.environ/...`
references needing resolution before forwarding to a router admin API
and (b) `store_model_in_db: True` being required for the admin API to
support `/model/delete` on YAML-loaded models. Both are LiteLLM-specific
sub-invariants; consider as standalone questions once the corpus grows.


## Q002 — XDG basedir: pick `data` for persistent state, not `cache`

Tags: `xdg-basedir`, `persistence`, `pattern-following`

Source:
- Buggy state: `b844c9f..f0f9197^` (paths.py was introduced in b844c9f
  using `type='cache'` for the data root).
- Fix commit: f0f9197 ("Default data_root to XDG_DATA_HOME, not
  XDG_CACHE_HOME") — see `infer_stack/paths.py`.

### Prompt

You're adding feature support to a Python project that already has
this module for resolving default directories:

```python
# app/paths.py
from pathlib import Path
import ubelt as ub

def _default_config_root() -> Path:
    """Where config.yaml lives. Defaults to ~/.config/myapp."""
    return Path(ub.Path.appdir("myapp", type="config"))

def _default_runtime_root() -> Path:
    """Where rendered artifacts and compiled assets are cached.

    Holds the rendered docker-compose.yml, model weight caches, and
    compiled torch graphs. All entries can be regenerated from config.
    """
    return Path(ub.Path.appdir("myapp", type="cache"))
```

(`ub.Path.appdir(name, type=X)` returns an XDG-compliant directory:
`type="config"` → `~/.config/<name>`,
`type="cache"` → `~/.cache/<name>`,
`type="data"` → `~/.local/share/<name>`,
respecting the corresponding `XDG_*_HOME` env vars.)

The feature you're adding is a self-hosted chat application stack
deployed via Docker Compose. The rendered stack now needs three new
host directories for bind-mount volumes:

1. `litellm-config/` — a yaml file the router reads at startup.
   Re-generated by the CLI on every `render` from upstream config.
2. `postgres-data/` — PostgreSQL data directory for the chat
   application. Stores user accounts, chat history, and per-user
   settings. Lost on delete, not recoverable from anywhere else.
3. `model-weights/` — Hugging Face cache directory. Holds downloaded
   safetensors files. Re-downloadable from the Hugging Face Hub on
   demand (slow but always possible).

Add a third helper `_default_state_root()` that returns the parent
directory under which these three subdirectories will live (so the
volumes resolve to `<state_root>/litellm-config`, `<state_root>/postgres-data`,
`<state_root>/model-weights`). Pick the `type=` argument for
`ub.Path.appdir` and justify the choice in ≤3 sentences.

### Expected answer

```python
def _default_state_root() -> Path:
    """Root for bind-mount state. Houses both regenerable caches and
    non-regenerable user data (postgres), so the whole tree gets `data`
    semantics to avoid system cleanup tools wiping the database.
    """
    return Path(ub.Path.appdir("myapp", type="data"))
```

Justification that should be present:

1. The combined tree contains `postgres-data/` which is **not**
   regenerable from anywhere — accounts and chat history exist only
   there. That alone disqualifies `type="cache"`.
2. The XDG specification permits system cleanup tools to remove
   `~/.cache` contents at any time; `~/.local/share` is reserved for
   persistent user data.
3. The right granularity is the *least regenerable* directory in the
   tree. Splitting the three subdirs across `data` and `cache` is also
   acceptable (postgres → data, model-weights and litellm-config →
   cache) but adds tracking surface for no operational benefit when
   they're all bind-mounted into one stack.

**Wrong answers**:

- `type="cache"`, citing "the existing `_default_runtime_root` uses
  cache so I'll follow that pattern" — the existing helper is for
  *regenerable* artifacts; the new helper isn't.
- `type="config"` — these are run-time volumes, not user-editable
  configuration.
- A custom `~/.myapp-state` path that bypasses XDG — abandons the
  ergonomics XDG provides (env-var override, distro conventions).

### Validation

```bash
python -c "
import os, tempfile, pathlib
os.environ['XDG_DATA_HOME'] = '/tmp/xdg-data-test'
os.environ.pop('XDG_CACHE_HOME', None)
from app.paths import _default_state_root
assert str(_default_state_root()).startswith('/tmp/xdg-data-test'), \
    f'state_root must honor XDG_DATA_HOME, got: {_default_state_root()}'
print('OK:', _default_state_root())
"
```

Expected result: command prints `OK: /tmp/xdg-data-test/myapp` and
exits 0. With the wrong (cache) answer, the assertion fails because
`appdir(type='cache')` honors `XDG_CACHE_HOME`, not `XDG_DATA_HOME`.

### Why this was easy to miss

- The natural cognitive move is to **follow the existing pattern**.
  The file already has one `_default_*_root` using `type="cache"`,
  so adding another with `type="cache"` reads as consistent.
- The docstring on `_default_runtime_root` says "cached", which is
  *correct* for that helper but visually reinforces the wrong choice
  for the new one.
- `ub.Path.appdir` accepts `type="cache"` silently — no validation
  error, no type-checker warning, no runtime symptom on the developer
  machine.
- The blast radius only shows up later, in production, when a system
  cleanup tool or a user running `rm -rf ~/.cache/*` to reclaim disk
  wipes the postgres database. The bug is invisible until the data
  is already gone.

### Notes

The real-repo version was the inverse direction: an existing
`_default_data_root` was implemented with `type="cache"` from the
start (this was the "follow the existing config pattern" trap — the
sibling `_default_config_root` correctly used `type="config"`), and
the mistake wasn't noticed until reviewing the data layout months
later. The fix was a one-line change in the helper plus updating the
docstring to explain why `data` was chosen, so a future maintainer
wouldn't "fix" it back to `cache`.

This belongs to the same invariant family as `pattern-following`
mistakes: an agent reading the existing code looks for cues to be
consistent and ends up consistent with the wrong neighbor. Worth
adding more questions to this family as they appear: e.g., copying a
type annotation from a sibling field when the new field has different
nullability semantics, or copying a `cache_control` header from an
existing endpoint when the new endpoint shouldn't be cached.
