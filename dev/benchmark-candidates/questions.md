# Benchmark questions

## Q001 — Docker Compose `depends_on` cascades recreates into dependents

Tags: `lifecycle-ordering`, `dependency-boundary`, `docker-compose`

Source:
- Buggy state: any commit in `91b2586..32be174^` (the `depends_on` line
  has existed since the compose template was introduced in 91b2586).
- Fix commit: 32be174 ("Hot-swap models without dropping Open WebUI") —
  see `vllm_service/templates/docker-compose.yml.j2`.
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
