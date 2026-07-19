# RABBIT BBN Codex Cost-Effective Anti-Drift Policy

Date: 2026-06-07

Purpose: keep the augmented Type-I no-QKE BBN line focused on endpoint,
physics, numerical, and performance blocker movement per line of code and per
available token budget. This policy is active with `AGENTS.md` and
`docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`.

## Hard Scope

- QKE remains out of scope.
- Do not claim public-production or publication-ready support.
- Preserve raw negative abundances, negative `Y_p`, NaN, rejected candidates,
  failed Newton/Rodas attempts, and failed endpoint artifacts.
- Rust AOT is the active implementation and repeated-run design target.
  SciPy/BDF remains the temporary number-of-record until the frozen Rust
  endpoint authority gate passes; JAX is a frozen local parity/AD/Jacobian
  oracle and receives no new physics, driver, endpoint, or runtime optimization.
- Do not add readiness, manifest, hash, figure, or claim-wrapper gates unless
  the same PR deletes or consolidates older surface and moves a runtime physics,
  solver, endpoint, or performance blocker.

## Required PR Cost Line

Every PR in this line must report:

```text
added_lines:
deleted_lines:
net_lines:
files_touched:
token_use_exact:
token_use_basis:
runtime_behavior_changed: yes/no
physics_behavior_changed: yes/no
known_blocker_reduced: yes/no
blocker_movement_ratio: 0.00..1.00
validation_strengthened: yes/no
cost_effectiveness_verdict: ACCEPT / ACCEPT_WITH_LIMITS / FAILURE_MODE_RELOCATION / NO_PROGRESS / DRIFT
```

Token rule:

- If the harness/API exposes exact token use, report it.
- If exact token use is not exposed, write `token_use_exact: UNAVAILABLE` and
  explain the missing counter in `token_use_basis`.
- Do not invent or back-compute exact token counts from transcript length.

## Line Budget

- `net_lines <= 80`: preferred.
- `80 < net_lines <= 200`: acceptable with direct test/artifact evidence.
- `200 < net_lines <= 400`: warning; must include blocker movement or net
  deletion/consolidation.
- `net_lines > 400`: presumed drift unless the PR moves a hard endpoint,
  physics, runtime, or validation blocker and explains why it could not be
  split.

Docs and packets count as cost. If a policy or audit packet is requested, keep
it short and remove stale/redundant surface where possible.

Migration duplication is temporary. Every Rust PR reports cumulative Rust
source additions and superseded Python/JAX deletions. By R-06, active source
LOC must be lower than the PORT-00 baseline; otherwise the migration has not
closed even when parity and timing pass.

## Blocker Movement Ratio

Use this as an honest self-audit score, not as a success claim.

- `0.00`: no measured movement; wrapper/doc/schema-only growth.
- `0.25`: a real code path or failure mode became executable or better
  localized, but endpoint/performance did not improve.
- `0.50`: one known blocker partially moved. Example: same-case wall, memory,
  step count, JVP/source calls, or endpoint temperature improves by at least
  about 10%, or a previously blocked physical path completes a focused real run.
- `0.75`: a hard blocker moved substantially. Example: same-physics endpoint
  becomes materially colder, exit 137 disappears, or a dominant wall bucket
  drops by at least about 30%.
- `1.00`: a previously impossible configured full e2e BBN run reaches endpoint
  with raw state preserved and required parity/floor checks passing.

## Verdicts

- `ACCEPT`: proportional code cost and measured blocker/physics/validation
  progress.
- `ACCEPT_WITH_LIMITS`: useful local progress, but not an endpoint or dominant
  blocker milestone.
- `FAILURE_MODE_RELOCATION`: not solved, but failure moved later or became
  more diagnostic under the same relevant physics.
- `NO_PROGRESS`: no measurable progress; reduce scope or revert.
- `DRIFT`: code/context grew without proportional scientific gain.

## Candidate Selection

Before a nontrivial edit, compare at most three executable candidates:

```text
Candidate:
files:
net_lines_estimate:
expected_blocker_or_physics_gain:
main_risk:
minimal_test:
```

Choose the candidate with the best blocker movement per line. Prefer deletion,
reuse, caching, vectorization, or a focused harness over new abstraction.

## Simplicity Rules

- Prefer one small pure function over a class/registry/config surface.
- Prefer reusing existing network, thermo, transport, and solver APIs over
  duplicating physics formulas.
- A new flag is justified only if it enables a real comparative run or replaces
  an older flag.
- Do not add import-only, schema-only, count-lock, or pass-through tests.
- Segment-only benchmarks must be labeled segment-only and cannot be reported
  as endpoint progress.

## Minimum Evidence

For each PR, run the smallest relevant harness and report pass/fail/skipped:

- endpoint or staged solver PR: focused staged/AP65 run or explicit skip reason;
- physics convention PR: conservation, sign, limiting-case, and raw-state tests;
- performance PR: same-case wall/counter artifact, not toy speedup;
- policy/audit PR: no solver validation claim, and line-cost disclosed.

Raw failed artifacts must be preserved. A failed run can count as progress only
when it relocates the blocker or exposes a more specific failure mechanism.

## One-Line Rule

Write the shortest auditable patch that moves one physical or numerical
uncertainty boundary, then prove it with the smallest hard BBN run.
