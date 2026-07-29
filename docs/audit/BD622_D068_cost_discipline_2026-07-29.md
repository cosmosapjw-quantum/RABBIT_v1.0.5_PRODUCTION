# BD622 D-068 — forward-only cost discipline and the shared trajectory core

Date: 2026-07-29
Lane: D-065 remedy lane 2 (cost), owner-granted 2026-07-29
Freeze: `ed7bc49` (shared core), `0fb94f2` (first driver built on it)

Status: **IMPLEMENTED — NO GATE MOVEMENT — RECORD WRITTEN RETROSPECTIVELY**

## 0. Why this record is late, and what that costs

D-068 shipped inside commit `ed7bc49` alongside D-067 and received no audit
report, no `INDEX.md` anchor, and no row in `FROZEN_DECISIONS.md`,
`DECISION_LOG.md`, or `VALIDATION_LEDGER.md`. It was found missing during the
D-070 survey, two decisions later.

That is a provenance weakness and it is stated here rather than smoothed over.
Every claim below is re-derived from the committed bytes and from measurements
taken at D-070, not from a contemporaneous record — because there was none.

The same survey found the hook-fixture count written as `39`, `35`, and
`12 -> 35` across four surfaces. Counting `def test_` at each commit resolves it:
**35 at the D-067 seal `ed7bc49`, 39 at `07e3507`**, which added four regression
fixtures for the three defects the round-3 review found. So the frozen D-067 row
and the D-065-era ledger text are *temporally correct*, not wrong, and have been
annotated rather than rewritten. The genuine defect was narrower and worse: the
D-067 report asserted `39 pass` in its prose and `12 -> 35` in its own cost block
**at the same commit**. A surface that contradicts itself cannot be resolved by
precedence rules, only by measurement.

A lane without an SSOT row is a lane that did not happen as far as any later
auditor is concerned, and a number that disagrees with itself is worse than a
number that is merely stale. `check_ssot_consistency.py` (D-070) fails closed on
both, and `SSOT_FACTS.json` pins the measured value with the commit it holds at.

## 1. What D-065 found

Finding F-D065-06 returned `G-F10-SCOPE=PASS` with cost verdict **DRIFT**:

```text
added_lines: 16297      deleted_lines: 75       net_lines: 16222
files_touched: 93       production_source_lines_changed: 0
blocker_movement_ratio: 0.25
cost_effectiveness_verdict: DRIFT
```

with three named mechanisms: the D-060/D-061 oracles are ~99.5% identical, the
D-062/D-063 trajectory drivers ~98.5% identical, and report bodies are duplicated
into stdout logs. Required cost accounting was absent from the D-064 closeout
surfaces entirely.

Re-measured at D-070, normalising only the decision-id strings:

| Pair | Total lines | Differing lines | Identical |
|---|---|---|---|
| `d060_covariance_metrology_r3_oracle.py` / `d061_..._r4_oracle.py` | 3050 | 26 | 99.1% |
| `d062_independent_trajectory_r2.py` / `d063_..._r3.py` | 1270 | 50 | 96.1% |

The audit's characterisation holds. Each reissue was a whole-file copy with a
handful of changed constants.

## 2. What D-068 changed

**`scripts/audit/_trajectory_core.py`** (472 lines) — one module holding the
pieces every trajectory driver had been copying: `Setup` / `build_setup`,
`Deadline`, `Stats`, `unpack`, `make_rhs`, `run_integration`,
`comoving_energies`, `coupled_residuals`, `checkpoint_states`,
`endpoint_summary`, `pair_densities`, `spectral_moments`, `anchor_moments`,
`equilibrium_tail_fraction`, and `Reporter`.

It was authored fresh rather than extracted, and d060–d063 were left untouched.
A retroactive refactor would have rewritten the bytes that preserved-FAIL records
D-060 and D-062, and PASS records D-061 and D-063, cite as evidence. Verified
before freeze: the core's `make_rhs`, `comoving_energies`, and `pair_densities`
reproduce the frozen d063 implementations **bitwise**.

**`scripts/audit/_f10c2_anchors.py`** (103 lines) — the completed-catalogue
anchors and the Rust spectrum, single-sourced. `spectrum_digest()` was verified
equal to d063's embedded
`3df5a9907f8a7e7e5168a4659504387f16a0c795878a833df97f7a7e2613fa58` before use.

**Canonical single-report output** — `Reporter` writes the whole report to the
`--out` path, atomically (`tmp` + `os.replace`), once per phase, and **never**
into stdout. `Reporter.log` emits only `[{elapsed:9.1f}s] {message}` progress
lines. Measured effect:

| Run | Report | stdout log | stdout content |
|---|---|---|---|
| d063 r3 (before) | 42,138 B | 55,022 B | progress **plus a full copy of the report body** |
| d069 r4 (after) | pending | 5,522 B at 3.5 h | progress only |

**Mandatory cost fields** — every closeout from D-068 forward carries the
`added/deleted/net lines`, `files_touched`, `runtime_behavior_changed`,
`physics_behavior_changed`, `blocker_movement_ratio`, and verdict block.
D-070 makes this mechanical: `.agent-harness/scripts/cost_report.py` computes
every measurable field from git with a per-area breakdown, and **refuses to run**
without an explicit `--blocker-movement` and a non-empty justification, because
that ratio is a judgement and must carry the words of whoever made it.

## 3. What this cycle actually cost — the honest arithmetic

D-068 **added 575 lines and deleted none.** Worse, the first driver built on the
core is not smaller than the one it replaces:

```text
d063 standalone                                    651 lines
d069 driver 624 + core 472 + anchors 103         1199 lines
```

So on its own cycle D-068 made the number the audit complained about *larger*.
That is not a defect of the accounting; it is what forward-only means, and the
owner chose forward-only with the reason on the record: the duplicated files are
the byte-provenance of preserved FAIL and PASS evidence, and deleting them to
improve a line count would destroy the audit trail D-065 demanded.

The saving is entirely prospective and lands on the *next* driver, which inherits
575 lines it does not have to copy. The D-071 driver is the first place it can be
measured, and it will be measured there rather than asserted here.

## 4. What is deliberately not claimed

- **No gate moves.** D-068 touches no physics, no solver, no production source,
  and no gate or claim status.
- **The DRIFT verdict is not repaired by this lane.** DRIFT is a verdict on a
  completed chain; spend cannot be un-spent. `blocker_movement_ratio` rises only
  when a blocker actually moves, which is D-070's business, not D-068's.
- **No historical deletion**, so the ~99.1% and ~96.1% duplications the audit
  measured are still present in the tree and will still be found by the next
  auditor. They are preserved on purpose.
- The bitwise-equivalence check covers `make_rhs`, `comoving_energies`, and
  `pair_densities` against d063 only. It is not a proof that the core reproduces
  every d060–d063 behaviour.

## 5. Cost

```text
added_lines: 575
deleted_lines: 0
net_lines: 575
files_touched: 2
production_source_lines_changed: 0
token_use_exact: UNAVAILABLE — no reliable stage-scoped counter
token_use_basis: harness/API exposes no reliable per-stage counter
runtime_behavior_changed: no — new modules, no existing driver altered
physics_behavior_changed: no
known_blocker_reduced: no — this lane is process discipline, not evidence
blocker_movement_ratio: 0.0
cost_effectiveness_verdict: DRIFT on this cycle by construction; the saving is
  prospective and is first measurable at the D-071 driver
```
