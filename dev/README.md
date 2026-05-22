# Developer memory — `dev/`

This directory is long-running engineering memory for this repository.

It exists so an agent arriving cold can learn from the repository's past
maintenance mistakes before making the same ones again. Anything in `dev/`
should help future agents preserve project-specific invariants, diagnose
confusing failures faster, or turn real repair work into benchmark-quality
questions.

`dev/` is not a TODO list, feature log, design-doc tree, or scratch directory.
It is curated memory for lessons that remain useful across sessions.

## What belongs here

Put something in `dev/` when it is durable, transferable, and useful to a
future maintainer or agent without the original conversation context.

Good candidates include:

- a real agent or human mistake that exposed a project-specific invariant;
- a refactor failure that required repo context to understand;
- a bug that took significant effort to diagnose;
- a test, fixture, API, module-boundary, build, packaging, or runtime invariant
  that future patches are likely to violate;
- a distilled benchmark question based on a real repair.

Do not put routine plans, active TODOs, feature status, generated logs, or raw
scratch notes here. Keep those in the working tree, issue tracker, project
TODOs, or temporary files.

## Recommended layout

```text
dev/
  README.md                 # This file: purpose and operating rules
  SEARCH.md                 # How agents should search this memory
  benchmark-candidates/     # Distilled hard questions from real mistakes
    README.md               # Benchmark workflow, quality bar, prompt levels
    index.md                # Routing by invariant / failure class
    questions.md            # Main single-issue benchmark corpus
    compositions.md         # Multi-invariant benchmark prompts
  journals/                 # Debugging postmortems and durable lessons
    index.md                # Symptom router
    lessons-learned.md      # Aggregate entries, newest first
```

Projects may split `questions.md` into topic files when the corpus grows, such
as `api-questions.md`, `ui-questions.md`, `database-questions.md`,
`build-questions.md`, or `migration-questions.md`.

Prefer fewer, better files over a sprawling archive nobody trusts enough to
read.

## Agent entry protocol

When working in this repository, an agent should use `dev/` as follows:

1. **Before a risky change**, search `dev/SEARCH.md` and the relevant
   benchmark/journal indexes for the area being touched.
2. **During the change**, keep a short note of surprising failures, review
   comments, confusing tests, and fixes that depended on repo-specific context.
3. **After fixing the issue**, decide whether the failure teaches a durable
   invariant.
4. **If yes**, add either a benchmark candidate, a journal entry, or both.
5. **Before handoff**, run the validation command recorded with the entry, or
   explain why it could not be run.

The goal is not to document every mistake. The goal is to make the next agent
less likely to repeat the mistakes that matter.

---

# `dev/benchmark-candidates/`

## Purpose

`benchmark-candidates/` turns real maintenance work into self-contained
software-engineering benchmark questions.

A benchmark candidate captures the context an agent had *before* making a
mistake. It asks whether another model can preserve the relevant invariant
while doing the original task, ideally before seeing the compiler error, test
failure, lint failure, runtime symptom, or review comment that exposed the bug.

The corpus should be useful in two ways:

1. **Operational memory:** future agents can read it before similar work and
   avoid known traps.
2. **Evaluation data:** maintainers can later package the entries into a real
   benchmark suite for measuring agent reliability on realistic repo tasks.

## When to add a benchmark candidate

Add an entry when all of these are true:

- The mistake came from a real patch, review, debugging session, or failed
  agent attempt.
- The fix required repository context, not just generic language knowledge.
- Another competent model could plausibly make the same mistake from the same
  starting context.
- The invariant can be stated compactly.
- There is a validation command or precise expected patch shape.

Examples of good invariant classes:

- module-boundary and re-export behavior;
- public API compatibility;
- fixture and relative-path resolution;
- code-generation or schema compatibility;
- serialization and migration behavior;
- test-only interface requirements;
- async, concurrency, transaction, or lifecycle ordering;
- UI state, input handling, and event routing;
- packaging, deployment, CI, or platform-specific assumptions;
- determinism, snapshot, golden-file, and record/replay expectations.

## Benchmark workflow

1. **Fix the project first.** Do not write speculative benchmark entries for
   unverified guesses.
2. **Save the failure evidence.** Preserve the exact compiler, test, lint,
   runtime, CI, review, or user-visible failure that exposed the issue.
3. **Reconstruct the pre-error context.** Ask what the agent knew before the
   mistake. The strongest prompt starts there, not at the final error.
4. **Distill the invariant.** Convert the mistake into a compact rule, such as:
   "annotations move with the item they annotate," "facade exports must preserve
   the old public API," or "fixtures referenced by relative paths move with the
   file that resolves them."
5. **Write a hard question.** Include enough surrounding code shape, file
   layout, and task context that the original mistake is tempting.
6. **Write the expected answer.** State the minimal patch shape and the
   reasoning the candidate is testing.
7. **Write validation.** Include the command, test, static check, or manual
   assertion that catches the bug.
8. **Tag by invariant.** Prefer tags that describe the failure class, not only
   the surface technology.

## Candidate quality bar

A candidate is worth keeping when it is:

- **Realistic:** based on an actual maintenance event.
- **Contextual:** requires project-specific reasoning or API awareness.
- **Minimal:** removes irrelevant files and details while preserving the trap.
- **Checkable:** has a validation command or exact expected patch shape.
- **Non-leaky:** does not reveal the answer through the prompt wording unless
  the candidate is explicitly an error-repair task.
- **Transferable:** teaches a pattern likely to recur in this repo or others
  with similar architecture.

Avoid entries that merely say "remember to run tests" or "the model made a
syntax error." Those are useful only when tied to a specific invariant.

## Prompt levels

For each significant issue, prefer the highest fair prompt level.

### Level A: pre-error operation

The model receives the original task and surrounding context, before the
failure is known.

Example:

> Split this facade module into private child modules while preserving behavior,
> public API, tests, fixtures, and generated assets.

This is the best level for testing planning, invariant enumeration, and
mistake avoidance.

### Level B: error repair

The model receives the error, failing test, review comment, or runtime symptom
and must fix it.

Example:

> After the refactor, this test fails because a fixture path no longer resolves.
> Diagnose the cause and patch the code without changing the fixture layout.

This is useful for debugging ability, but it is less predictive of whether the
model would have avoided the bug.

### Level C: distilled micro-question

The model receives a tiny example that isolates the rule.

Example:

> In this language/framework, which file determines the base path for a relative
> fixture include after code is moved into a child module?

This is useful for focused unit evaluation, but less repo-specific.

## Entry template

Use this template for single-issue questions.

```markdown
## QNNN — <short title>

Tags: `<invariant-tag>`, `<surface-tag>`, `<repo-area>`

Source:
- Commit/PR/branch/conversation: <link or identifier>
- Original failure evidence: <where the exact output or review comment lives>

### Prompt

<The benchmark question. Prefer Level A when fair. Include the relevant file
layout, code shape, task, and constraints. Do not leak the final error unless
this is a Level B candidate.>

### Expected answer

<The minimal correct patch shape. Explain what must move, stay public, remain
private, be regenerated, be re-exported, be tested, or be left unchanged.>

### Validation

```bash
<command that catches the bug>
```

Expected result:

<What should pass, fail, or be asserted.>

### Why this was easy to miss

<Name the cognitive trap: e.g. local compile success hid an integration API
break; moving a file changed relative-path semantics; a new test made a
production derive look necessary; two independent migrations produced
misleading errors.>

### Notes

<Optional: variants, related questions, follow-up composition candidates.>
```

## Composition candidates

Some failures matter because several invariants are in flight at once. Put
these in `benchmark-candidates/compositions.md`.

A composition is worth keeping when it tests abilities not fully measured by
the component questions, such as:

- enumerating risk categories before editing;
- preserving several independent invariants in one patch;
- recognizing interference between fixes;
- attributing errors correctly when multiple migrations happen together;
- choosing validation that covers the combined change.

Composition template:

```markdown
## C-NNN — <short title>

Composes: Q001, Q004, Q009
Tags: `<composition-tag>`, `<repo-area>`

### Prompt

Plan and implement this refactor as one PR. Before writing code, enumerate the
categories of mistake you will explicitly guard against.

<Repo context and task.>

### Expected answer

<What a complete solution must preserve across all component invariants.>

### Validation

```bash
<commands>
```

### Why this composition is harder than its parts

<Explain the interaction between invariants.>
```

---

# `dev/journals/`

## Purpose

`journals/` records durable debugging lessons from issues that took real effort
to diagnose.

A good journal entry is optimized for search by future symptom language. It
should help an agent or maintainer recognize "I have seen this kind of failure
before" much earlier than the original debugger did.

## When to add a journal entry

Add one when:

- the diagnosis took significant time;
- the symptom was misleading;
- the root cause is likely to recur;
- the fix depends on repo-specific architecture, data flow, lifecycle, or
  tooling;
- future agents would benefit from the symptom wording.

Do not write long narratives. The best journal entries are compact and
searchable.

## Journal entry template

```markdown
## YYYY-MM-DD — <symptom phrase>

Area: `<repo-area>`
Tags: `<symptom-tag>`, `<root-cause-tag>`, `<surface-tag>`

### Symptom

<Use the words someone would search for while still confused. Include exact
error snippets only when short and useful.>

### Root cause

<The actual cause, with enough repo context to make it understandable.>

### Fix

<The patch shape or operational fix.>

### Validation

```bash
<command or manual check>
```

### Takeaway

<One durable lesson future agents should remember.>
```

---

# `dev/SEARCH.md`

Create `SEARCH.md` as the routing layer so agents do not have to read all of
`dev/` before every task.

It should contain:

- the main repo areas and which memory files to check for each;
- grep/ripgrep patterns for common symptoms;
- tag clusters and where they live;
- instructions for when to add new benchmark candidates or journals.

Suggested starter:

```markdown
# Search guide for `dev/`

Use this file before large refactors, confusing debugging sessions, migrations,
or changes touching old failure-prone areas.

## Fast search

```bash
rg -n "<term>" dev/
```

Search by:

- file/module/component name;
- framework/library name;
- symptom phrase;
- failing test name;
- invariant tag;
- platform;
- error keyword.

## Routing

| Working on | Search terms | Start here |
|---|---|---|
| Module or package refactor | `module-refactor`, `visibility`, `public-api` | `benchmark-candidates/index.md` |
| Fixtures, snapshots, assets | `fixtures`, `relative-paths`, `golden-file`, `snapshot` | `benchmark-candidates/index.md` |
| UI or input state | `ui`, `input`, `state`, `event` | `benchmark-candidates/index.md` |
| CI, packaging, deployment | `ci`, `packaging`, `deployment`, `platform` | `journals/index.md` |
| Confusing runtime bug | symptom words from the failure | `journals/index.md` |

## Add memory when

- a real fix exposes a transferable invariant;
- debugging took long enough that the next agent should not rediscover it;
- a review comment identifies a project rule not written elsewhere;
- a benchmark candidate could test whether another model avoids the same trap.
```

---

# Tag taxonomy

Tags should cluster entries by the invariant being tested. The invariant tag is
what makes the corpus useful for evaluating model failures. Surface tags are
still useful, but secondary.

Suggested clusters:

- **Refactor invariants:** `module-refactor`, `visibility`, `public-api`,
  `annotations`, `doc-comments`, `file-layout`, `relative-paths`,
  `fixtures`, `asset-paths`, `test-layout`.
- **API and compatibility:** `backwards-compatibility`, `schema-change`,
  `serialization`, `migration`, `deprecation`, `versioning`,
  `generated-code`.
- **State and event flow:** `state-machine`, `event-design`,
  `multi-source-input`, `edge-vs-held-state`, `lifecycle-ordering`,
  `cross-system-signal`.
- **Determinism and tests:** `determinism`, `record-replay`, `snapshot-test`,
  `golden-file`, `ci-fixture`, `flaky-test`, `off-by-one`.
- **Concurrency and async:** `async`, `race-condition`, `deadlock`,
  `transaction-boundary`, `cancellation`, `retry`, `idempotency`.
- **Platform and packaging:** `filesystem`, `permissions`, `deployment`,
  `packaging`, `mobile`, `desktop`, `web`, `linux`, `macos`, `windows`.
- **Architecture seams:** `architecture-seam`, `code-shape-tradeoff`,
  `layering`, `facade`, `dependency-boundary`, `ownership-boundary`.

When in doubt, tag with both the invariant and the surface, for example:
`relative-paths`, `fixtures`, `python`, or `public-api`, `typescript`,
`package-exports`.

---

# Relationship to the rest of the repository

Use this rule of thumb:

```text
TODO / issue tracker    -> what is in flight
CHANGELOG / FEATURES    -> what shipped
docs/                   -> how the system works today
src/ / packages/        -> the product itself
tools/                  -> build, authoring, or maintenance tooling
dev/                    -> durable engineering memory and benchmark candidates
```

Ask this before adding anything to `dev/`:

> Would a brand-new agent landing in this repository benefit from reading this
> cold, without the original chat context?

If yes, `dev/` is probably right. If no, use a TODO, issue, docs page, scratch
file, or conversation memory instead.

---

# Maintenance rules

- Keep entries short enough that future agents will actually read them.
- Prefer exact commands over vague advice.
- Preserve failure evidence, but do not dump giant logs inline.
- Link to commits, PRs, issues, branches, or conversations when possible.
- Favor invariant names over one-off symptom names.
- Delete or consolidate weak entries when the corpus gets noisy.
- Treat `dev/` as curated memory, not a junk drawer.
