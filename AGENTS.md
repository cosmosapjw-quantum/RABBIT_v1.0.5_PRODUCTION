# Repository Guidelines

## Mandatory Augmented Type-I No-QKE Pre-Read
Before editing the augmented Type-I PSTF no-QKE programme, read
`docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`.  That document is
the controlling anti-drift rule for this line: do not add another standalone
diagnostic/readiness/manifest/hash/figure gate unless the same PR deletes or
consolidates older gate plumbing and directly moves a runtime physics, solver,
or performance blocker.  QKE remains out of scope, public production support
must not be claimed. Rust AOT is the active implementation and repeated-run
design target; SciPy/BDF remains the temporary number-of-record until the
Rust endpoint authority gate passes. JAX is a frozen local parity/AD/Jacobian
oracle, not a forward-development or runtime-promotion target.

Also read `bbn_codex_anti_drift_cost_effective_policy.md`.  Every PR in this
line must report added/deleted/net lines, token-use availability, blocker
movement ratio, and a cost-effectiveness verdict.  If exact token counters are
not exposed by the harness, report `UNAVAILABLE` with the reason instead of
inventing a number.

The BD397 external audit adds a stricter anti-local-minimum rule.  Do not spend
another PR optimizing an already-cheap segment, adding policy knobs, or adding
telemetry/readiness plumbing when the measured activation/cold endpoint blocker
or an identified physics-correctness blocker remains untouched.  Segment-only
benchmarks must be labeled segment-only and cannot be reported as endpoint
progress.  A PR that adds new code surface must either move the measured
activation/cold wall, fix a physics readout/calibration/parity blocker, or
delete/consolidate more obsolete wrapper surface than it adds.

## Project Structure & Module Organization
`src/rabbit/` contains the installable package. Key areas are `config/` for capability registries and runtime policy, `jax/` plus `solver/` for backend implementations, and `transport/`, `thermo/`, `weak/`, and `network/` for physics blocks. `inference/` holds forward-solver and sampling entrypoints. `tests/` mirrors the package with `test_*.py` modules and fixtures under `tests/fixtures/`. Use `docs/audit/` for phase notes and profiling writeups, and `scripts/` for release, benchmark, figure, and registry-sync tasks. Treat `audit_outputs/`, `diagnostic_outputs/`, `production_report_cache/`, and most of `figures/` as generated unless a task explicitly targets them.

## Build, Test, and Development Commands
`python -m pip install -e .` installs the base package.
`python -m pip install -e ".[dev]"` installs pytest plus optional JAX, plotting, and inference dependencies.
`pytest -q` runs the default suite from `tests/`.
`pytest -m "production and not slow" -q --tb=line` runs the portable release gates used by `scripts/package_release.py`.
`pytest -m gold -q` runs the locked BBN regression gates.
`PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q tests/test_jax_runtime_fallback.py tests/test_jax_typeI_characteristic_parity.py` is the frozen CPU-JAX parity-oracle smoke bundle; it is not a backend-promotion gate.
`python scripts/render_capability_tables.py --apply` regenerates capability tables and registry-backed docs.
`python scripts/sync_test_counts.py` refreshes test-count blocks in `README.md` and `STATUS.md`.
`python scripts/package_release.py` runs release preflight checks and builds the release zip.

## Coding Style & Naming Conventions
Target Python 3.10+ with 4-space indentation. Follow existing naming: `snake_case` for modules, functions, and tests; `PascalCase` for classes and dataclasses; `UPPER_CASE` for constants. Match surrounding style by keeping module docstrings, `from __future__ import annotations`, and type hints on public APIs. No formatter or linter is configured in `pyproject.toml`, so keep edits consistent with nearby code and avoid unnecessary rewrites.

## Testing Guidelines
Name tests `tests/test_*.py` and keep reusable fixtures under `tests/fixtures/`. Reuse the repo markers declared in `pyproject.toml`: `gold`, `production`, `release_smoke`, `jax`, `slow`, and `build_env_only`. Physics, backend-dispatch, or registry changes should ship with at least one focused regression test plus the relevant parity/runtime gate. Frozen JAX-oracle checks pin `JAX_PLATFORMS=cpu`; new JAX timing, driver, endpoint, or runtime-policy work is forbidden on this programme.

## Generated Docs, Commits, and PRs
`README.md`, `STATUS.md`, `SUPPORTED_CAPABILITIES.md`, and `PROMOTION_GATES.md` include registry-generated sections. Update `src/rabbit/config/backend_capabilities.py` or `src/rabbit/config/feature_capabilities.py` first, then rerun the render/count scripts instead of hand-editing generated blocks. Current history uses short imperative subjects with phase tags, for example `PR-A: eliminate explicit J_j state in JAX char driver` or `PR-JL: compact characteristic state and harden runtime fallback`. PRs should list scope, affected backends, commands run, and any refreshed audit notes or generated docs.

# Research Harness Rules

## Non-negotiable rules

- Never report validation as completed unless the relevant command/script actually ran.
- Never present toy/demo/mock numerical output as research evidence.
- Mark unverified claims explicitly.
- Do not silently change physical conventions, units, signs, gauge choices, frame choices, or boundary conditions.
- Prefer small, reviewable patches.
- After changes, report:
  - changed files,
  - commands run,
  - validation passed/failed/skipped,
  - remaining risks.

## Claim status vocabulary

Use exactly these labels where applicable:

- IMPLEMENTED: code exists and has been executed/tested.
- VALIDATED: independently checked by test, benchmark, derivation, or reproducible artifact.
- DERIVED: mathematically derived in the document, with assumptions stated.
- SPECIFIED: defined in a design/spec document but not yet implemented.
- PROPOSED: plausible research direction, not yet derived or implemented.
- SPECULATIVE: physically/mathematically interesting but currently unsupported.
- DEPRECATED: superseded and should not guide implementation.
- FORBIDDEN: explicitly disallowed pattern or assumption.

## Scientific checks

For physics/math changes, check:
- assumptions,
- conventions,
- dimensions/units,
- signs,
- limiting cases,
- boundary/initial conditions,
- positivity/normalization,
- numerical stability,
- reproducibility.

## Artifact rules

When creating or updating research artifacts:
- update the handoff packet if the project state changed;
- update the claim ledger if claim status changed;
- update the validation ledger if a command/test/build ran;
- preserve source files for figures and tables.

## Mandatory shared-context protocol for subagent workflows

This repository uses spec-driven development and evidence-bearing subagent audits. `AGENTS.md` contains durable policy only. Volatile project state, PR state, evidence, assignments, and results live under `.agent-harness/`.

### 1. Canonical context and compulsory bootstrap

- Before spawning any subagent, the main agent MUST ensure that `.agent-harness/context/CONTEXT_INDEX.json` and `.agent-harness/generated/CONTEXT_PACK.md` are current by running:
  `python3 .agent-harness/scripts/build_context_pack.py`
- Before spawning a subagent for an assignment, the main agent MUST mint that
  assignment's single-use admission receipt:
  `python3 .agent-harness/scripts/admit_agent.py --assignment-id <ASSIGNMENT_ID>`
  (add `--expect-agent-id <id>` whenever the parent chooses the agent id).
- Every spawned subagent MUST receive a spawn header containing all five fields:
  `RUN_ID`, `ASSIGNMENT_ID`, `CONTEXT_VERSION`, `INDEPENDENCE_MODE`, and
  `ADMISSION_TOKEN`. `SubagentStart` never receives the spawn prompt, so the
  token is the only thing that binds one agent to one assignment; a receipt is
  single-use and is consumed at `SubagentStop` with the writing `agent_id` and
  the result SHA-256 recorded in `runs/<RUN_ID>/ADMISSIONS.jsonl`.
- Every subagent MUST load, in this order:
  1. `.agent-harness/generated/CONTEXT_PACK.md`
  2. `.agent-harness/runs/<RUN_ID>/assignments/<ASSIGNMENT_ID>.json`
  3. only the role-specific and evidence files named by that assignment.
- Assignment schema v2 separates the live Codex `runtime_agent_type` from the
  logical `review_role`. The compatibility field `agent_type` MUST equal the
  runtime event type. A supported `default` runtime may carry a CAS or adjudicator
  `review_role` only when `new_assignment.py` seals the exact role files and result
  template; neither the assignment nor result may impersonate a dedicated runtime.
- The `SubagentStart` hook injects the shared context contract automatically. A subagent MUST NOT begin repo-wide exploration before validating that its assignment context version matches the current context index.
- If the spawn header or assignment file is missing, stale, or inconsistent, the subagent MUST stop substantive work and report the contract violation.

### 2. Context tiers and independence

Use explicit context tiers rather than relying on hidden parent-thread state:

- Tier 0 — shared core: specification pointer, conventions, symbol table, frozen decisions, changed-surface map, gate definitions, claim/evidence indices, tool availability, and non-goals. All subagents receive this tier.
- Tier 1 — assignment slice: claim IDs, files, symbols, tests, datasets, allowed tools, and expected output schema for one bounded task.
- Tier 2 — sibling results: withheld by default. An agent may read sibling results only when its assignment has `independence_mode = "adjudication"` or explicitly lists those result paths.

`blind-results` means that agents share the problem definition, conventions, assumptions, target form, and test vectors, but MUST NOT inspect another solver or reviewer’s derivation, verdict, or result before submitting their own.

### 3. Spawn budget and topology

- Maximum concurrently active subagents: 4.
- Maximum total subagents per run: 8.
- Maximum nesting depth: 2.
- Depth-2 delegation is allowed only when the parent assignment explicitly grants `may_spawn = true`, lists the unowned claim IDs to delegate, and registers the child assignment before spawning it.
- The main agent MUST create every assignment through `.agent-harness/scripts/new_assignment.py`; unregistered subagents are forbidden.
- A second wave is allowed when it covers new evidence, a disjoint failure class, a genuine independent derivation, or a disputed claim. A differently named reviewer repeating the same evidence and question is not a new wave.

### 4. No duplicate discovery by default

- The main agent or one designated context mapper owns repo mapping and shared evidence collection.
- Other agents MUST use the context pack and assignment slice. They MUST NOT rescan the whole repository unless the assignment sets `discovery_mode = "independent"` and explains why independent discovery is epistemically required.
- Re-reading a file is allowed when the agent verifies a cited claim, but broad rediscovery must be reported as a context defect and added to the shared evidence map rather than repeated by every agent.
- Common command output, large logs, diffs, and generated artifacts must be stored once and referenced by path plus hash; do not paste the same long output into multiple agent prompts.

### 5. Evidence and result contract

- Findings are keyed by stable `claim_id`; prose similarity is not a distinct finding.
- Every substantive verdict MUST include exact evidence references, assumptions used, tool/version information, and a reproducible command or proof artifact when applicable.
- Each subagent writes only to its unique result path declared in the assignment. It must not edit shared context, specs, gates, or another agent’s result.
- Before stopping, every subagent MUST write a result envelope and end its final message with one line of the form:
  `HARNESS_RESULT: {"assignment_id":"...","context_version":"...","status":"pass|fail|inconclusive|error","result_path":"...","admission_proof":"<ADMISSION_TOKEN from the spawn header>"}`
- The main agent deduplicates by `(claim_id, evidence_fingerprint, verdict)` before adjudication.

### 6. Spec and gate authority

- The current specification and gate registry are authoritative for scope, pass/fail criteria, and required evidence.
- Agents may challenge a gate, but must record that as a separate meta-finding; they may not silently redefine success criteria.
- Implementation may begin only after the relevant assignment identifies its governing spec clauses and gates.
- Final acceptance requires machine-readable gate results plus a human-readable adjudication note.

### 7. Four-axis CAS cross-validation

For mathematical or physical claims requiring CAS verification, the default independent axes are:

1. Wolfram Language + xAct
2. SageMath + Singular
3. Lean + mathlib or project proof libraries
4. SymPy, with high-precision numerical checks where useful

All four axes share the same `CAS_CONTRACT.json`: mathematical statement, conventions, domains, assumptions, branch choices, target canonical form, invariants, test vectors, tolerances, and forbidden shortcuts. Until adjudication, each axis MUST NOT read another axis’s scripts, derivation, or result. Agreement without assumption/branch alignment is not counted as cross-validation.

### 8. Write ownership

- Parallel agents may write only isolated evidence or result artifacts under their assignment directories.
- One designated main writer owns production code, shared documentation, specifications, and gate files.
- The adjudicator reads normalized result envelopes; it does not redo every full analysis unless a disputed claim requires a targeted rerun.
