# BD303 External Optimization / Speedup Audit Report

Target: RABBIT augmented Type-I PSTF no-QKE BBN solver, after BD282 / BD300 / BD301 / BD302.
Method: CRAG (Retrieve / Grade / Correct / Generate) + Chain-of-Code + role subagents.
Scope discipline: optimization audit only — not solver validation, not publication readiness, not public-production support. QKE out of scope. Target stays CPU-JAX + in-tree Rodas5P/AP65.

Evidence basis: every artifact claim below was re-derived directly from the raw JSON in `artifacts/`, not copied from the packet's forensics prose. Where I could only infer rather than measure, it is labelled **DERIVED**.

---

## 1. Executive Verdict

**`TELEMETRY_INCOMPLETE`.**

I reach the same top-line verdict the packet self-assigned, but on stronger evidence and with three refinements that change the action it implies:

1. **The gap is a surfacing/aggregation bug over *existing* instrumentation, not absent instrumentation.** The phase-2 corrector wall *is* timed at the call site (`_record_wall_seconds_stat`, `augmented_continuous_ap65_rhs.py:9382`) and 8 nonzero per-call samples survive into the artifact, but the per-row total `phase2_corrector_wall_seconds_total` is never written onto the span-row dict, so the summary reads `0.0`. The fix is small and surgical, not a re-instrumentation.

2. **A *partial measured* attribution is recoverable right now** from fields the summarizer ignores. Per-step `attempt_wall_seconds_samples` is complete (sample count == `attempt_count` in all four rows) and reconstructs **98.8%** of total wall. The "393 s residual unattributed" is not dark time — it is per-step execution time the summary does not decompose.

3. **The case for doing phase-2 reuse *first* is weaker than the decision frame assumes**, because the activation-row Jacobian is *already* host-lagged (60 builds, 59 reuses, stride 2; finite-difference path off). BD282's claim that the corrector "rebuilds its Jacobian on every iteration (~24.6k rebuilds)" is **contradicted** for the outer Jacobian and **unconfirmed** for the inner corrector.

**Consequence for the PR-C-vs-PR-D question (Required Work item 5):** on *measured* component wall, the ordering cannot be decided, because the phase-2/rejected/host/JAX walls are `0.0`/absent. The only directly-measured dominant component is collision-payload build (283 s, 42%). Therefore the correct first move is **neither PR-C nor PR-D**: it is a telemetry-surfacing fix that makes the existing per-call timers reach the span row. If forced to pick on today's evidence, **PR-D has the stronger measured target but an unproven achievable reduction**, and **PR-C risks duplicating machinery that already runs**. This is `TELEMETRY_INCOMPLETE`, not `PHASE2_FIRST`/`COLLISION_FIRST`.

---

## 2. Claim Ledger

Verdicts: SUPPORTED / PARTIAL / CONTRADICTED / UNTESTED / FORBIDDEN-OK (forbidden item correctly respected). "Measured" = re-derived from artifact this audit; "Source" = confirmed by reading code; "Derived" = inferred.

| ID | Packet claim | My verdict | Evidence |
|---|---|---|---|
| P1 | Fresh q4 rows real; final component JSON lost in BD301. | **SUPPORTED** (Measured) | BD301 progress summary = 4 rows, row wall 677.027 s; traceback `NameError: name '_sum_float' is not defined`. BD301 is failure provenance only. |
| P2 | `_sum_float` patch lets BD302 write clean final JSON. | **PARTIAL** (Measured) | BD302 final JSON is valid and parses; crash is gone. But the patched aggregator returns `0.0` for every phase-2 wall total (see §4), so "clean JSON" ≠ "complete attribution". |
| P3 | q4 bottleneck bimodal: payload pre-activation, phase-2/replay at activation. | **SUPPORTED** (Measured) | Payload share by row: 67% / 71% / 71% / **15%**. Activation row (3) carries all 15,603 phase-2 Newton evals and all 41 rejections while payload drops to 15%. Bimodality confirmed. |
| P4 | Phase-2 Jacobian rebuild likely first target; needs fresh timer split. | **CONTRADICTED in part / UNTESTED for wall** (Source+Measured) | Outer Jacobian is host-lagged: `host_lagged_jacobian_refresh_count=60`, `reuse_count=59`, `stride=2`, `frozen_source_finite_difference_jacobian_evaluation_count=0`. Inner corrector shows `newton_jacobian_evaluation_count_total=15603` but that is an *evaluation* count, not a confirmed *rebuild* count, and its wall is `0.0`/unmeasured. The premise "rebuild every iteration" is not supported by BD302. |
| P5 | Collision payload reuse=0 is a major pre-activation target. | **SUPPORTED for reuse=0; mechanism diagnosed** (Source+Measured) | `stage_collision_payload_reuse_count=0`, `auto_reuse_count=0`, `current_state_build_count` = full evaluation count in every row. Cause is the reuse gate, not a bug (see §3 / item 6). |
| P6 | Dense LU not the q4 bottleneck (W/J `[62,62]`). | **SUPPORTED** (Measured) | `jacobian_shape=[62,62]`, `linear_system_is_dense=True`, backend `scipy_lu_factor`, outer `factorization_count=153` (one per step attempt), `solve_count=1224`. Dense solve is not a wall target at q4. |
| P7 | JAX compile/runtime unresolved unless BD302 records explicit values. | **SUPPORTED** (Measured+Source) | `jax_compile_seconds` / `jax_runtime_seconds` set to `None`-with-reason at terminal depth only (`...rhs.py:17224–17230`); **absent at attribution-row depth**. `check_component_wall_attribution.py` prints FAIL for exactly these two fields. |
| P8 | Diagnostic hot-loop trace is a memory/clarity target, not primary wall. | **PARTIAL/UNTESTED** (no measured trace-wall) | No measured trace-wall field exists; `tracemalloc_peak=903,903,197 B`. Cannot rank as wall target. Treat as cleanup only. |
| P9 | q9/q10 forbidden in this packet. | **FORBIDDEN-OK** (Source) | q9/q10 appear only as prose in docs/manifest; **no command script executes them**. `COMMANDS_SLOW_OPTIONAL.sh` explicitly excludes them. Respected. |
| P10 | No optimization default-on before PR-B parity + `N_eff_3T>=3.0`. | **FORBIDDEN-OK, with a guard-integrity caveat** | Both design probes are `default_on_allowed:false`, `SPECIFIED_NOT_IMPLEMENTED`. Caveat: the wall-attribution checker *prints* FAIL but *exits 0* (§3) — verify the real blockers exit nonzero, or "default-on blocker" is advisory only. |

---

## 3. Commands Run

CPU-JAX environment was available (Python 3.12.3, numpy 2.4.4, scipy 1.17.1). **JAX is not installed in the audit sandbox**, and the clean q4 reproduction is a single-run-per-packet ~11.5 min job that the packet already executed — so the pytest/JAX commands and the q4 rerun are marked `NOT RUN` by design. All JSON-only helper scripts were run.

| Command | Status | Result |
|---|---|---|
| `python3 scripts/extract_q4_profile_table.py artifacts` | **RUN / PASS** | Reproduced the 4-row table exactly (137.64 / 131.28 / 66.42 / 341.14 s; rejects 0/0/0/41; N_eff 9.20→3.19; Yp 1e-30→0.1233). |
| `python3 scripts/check_component_wall_attribution.py artifacts/bd302_clean_profile` | **RUN / FAIL-as-expected, but exit 0** | Prints `FAIL missing value-or-reason for jax_compile_seconds, jax_runtime_seconds`; **exit code 0**. Confirms `TELEMETRY_INCOMPLETE`; flags a non-blocking-gate hazard. |
| `python3 scripts/summarize_packet_artifacts.py` | **RUN / PASS-but-defective** | Wrote summary. **Double-counts the final JSON**: reports `wall_seconds=1352.98` (= 2×676.49) and payload `566.34` (= 2×283.17), `rows_seen=9`, because it sums nested duplicate row containers. The progress-summary view is correct (676.49). |
| Direct JSON re-derivation of BD299 / BD301 / BD302 (this audit) | **RUN / PASS** | All headline numbers reconciled (§4). |
| `pytest tests/test_summarize_perf_artifacts.py` | **NOT RUN** | No JAX; packet ledger records `6 passed`. |
| `pytest ...::test_fb70_phase2_summary_metrics_sums_wall_timers` | **NOT RUN** (read by source) | Test exists and passes per ledger — but it is a false-comfort test (§4). |
| `pytest test_augmented_continuous_ap65_{rhs,full_bbn_span_ladder,...}` | **NOT RUN** | No JAX; packet ledger records 229/229 etc. |
| Clean BD302 q4 rerun (`/usr/bin/time -v ... span_ladder.py`) | **NOT RUN** (already executed once per packet) | Re-running would violate the once-per-packet rule and add no evidence. |
| q9 / q10 | **NOT RUN — FORBIDDEN** | Not requested, not executed. |

---

## 4. Artifact Forensics

### BD299 baseline (re-derived)
q4 bounded partial. Payload build wall present per-row (e.g. row 0 = 94.74 s); phase-2 Newton counts = 0 in early rows (network not yet firing). Confirms BD299 as the *payload-wall* baseline and the phase-2 *count* hotspot, and **not** a component-wall source. Consistent with the forensics doc.

### BD301 (re-derived)
4 rows completed, row wall 677.027 s, then `NameError: _sum_float`. Provenance only. Consistent.

### BD302 (re-derived — this is where the audit diverges from the packet's framing)

Row table (from `rows[0].span_rows[]`):

| row | N_span | wall s | payload s | payload share | n_rej | phase-2 Newton iters | N_eff_3T | Yp |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | [0.0,1.0] | 137.64 | 92.25 | 67% | 0 | 0 | 9.203 | 1e-30 |
| 1 | [1.0,2.0] | 131.28 | 93.37 | 71% | 0 | 0 | 5.028 | 1e-30 |
| 2 | [2.0,2.5] | 66.42 | 46.86 | 71% | 0 | 0 | 3.552 | 1e-30 |
| 3 | [2.5,2.75] | 341.14 | 50.68 | **15%** | 41 | 15,603 | 3.190 | 0.1233 |
| **Σ** | | **676.49** | **283.17** | **42%** | 41 | 15,603 | — | — |

**Finding A — the residual is not dark.** `attempt_wall_seconds_samples` is *complete* (len == `attempt_count`: 200/200, 201/201, 101/101, 153/153). Σ attempt walls = **668.62 s = 98.8% of 676.49 s**. The summary's "393 s residual unattributed" is simply per-step execution time it never decomposes.

**Finding B — the phase-2 wall total is a wiring bug, not missing instrumentation.**
- `phase2_corrector_wall_seconds` (per-call) appears **64×** in the artifact; 8 nonzero samples sit in `span_rows[3].phase2_conservative_extent_corrector_samples[]` (~0.012 s each).
- `phase2_corrector_wall_seconds_total` appears **once**, at `rows[0].span_summary`, value **0.0**, because `_phase2_summary_metrics._sum_float` (`...span_ladder.py:9111`) reads `row.get("phase2_corrector_wall_seconds_total")` and the per-row dict never carries that key.
- `_record_wall_seconds_stat` (`...rhs.py:10366`) *correctly* accumulates `_total/_count/_max` into `trace_stats`, and is called for the corrector at `...rhs.py:9382`. So the timer is captured but never surfaced onto the span row that the summary reduces.
- **Root cause of the false green:** `test_fb70_phase2_summary_metrics_sums_wall_timers` (`tests/...span_ladder.py:2329`) feeds the aggregator two *synthetic* rows that already contain `phase2_corrector_wall_seconds_total`, and asserts `3.25`. It tests summation, never population. It passes while real runs yield `0.0`.

**Finding C — taxonomy of the gap (what's actually missing vs. mis-routed):**

| Component | Timed at call site? | Surfaced to span row? | In artifact? | Class |
|---|:--:|:--:|:--:|---|
| Collision payload build | yes | yes | 283 s | **MEASURED** |
| Per-step attempt wall | yes | yes | complete (98.8%) | **MEASURED** |
| Phase-2 corrector wall | yes (`rhs.py:9382`) | **no** | per-sample only; total 0.0 | **NOT SURFACED** |
| Phase-2 Newton sub-walls (resid/Jac/solve) | partial (`rhs.py:9413`) | **no** | total 0.0 | **NOT SURFACED** |
| Rejected attempt/replay wall | consumer keys only (`span_ladder.py:6661`) | **no** | absent | **NOT SURFACED / likely not timed** |
| Host Jacobian / host JVP wall | consumer keys only (`span_ladder.py:6653`) | **no** | absent | **NOT SURFACED / likely not timed** |
| JAX compile / runtime | `None`+reason at terminal depth (`rhs.py:17224`) | **no** (at attribution depth) | absent at attribution depth | **NOT SURFACED to attribution depth** |

**Finding D — raw evidence preserved, no observable inflation.** `raw_boundary_provenance_preserved=True`; raw-negative containers present (0 negatives this run); `physical_observable_violations=[]`; `passed=False`, `rows_full_bbn_completed=None/0`, `completion_class=completed_hot_endpoint`. Top-level flags: `public_dispatch_ready=False`, `production_smc_validation_ready=False`, `physical_full_bbn_span_ready=False`, `qke_scope=out_of_scope`, `promotion_decision=not_promoted`. `Yp=0.1233`, `D/H=2.50e-3` are hot-endpoint intermediates (T_γ≈0.07 MeV), correctly *not* presented as final BBN observables and *not* clipped.

---

## 5. Component Wall Attribution Table

q4 partial (676.49 s total wall). "Measured" = directly timed and surfaced; "Derived" = reconstructed from per-attempt walls; "Unmeasured" = dedicated timer 0.0/absent.

| Component | Wall | Share | Basis |
|---|---:|---:|---|
| Collision payload build (all rows) | 283.17 s | 41.9% | **MEASURED** (`dynamic_collision_payload_build_wall_seconds_total`) |
| Activation-row phase-2 / network work | ~228 s | ~34% | **DERIVED** — excess of the 123 "network-on" attempts (mean 2.55 s) over the 0.69 s network-off baseline; scales with 15,603 Newton evals (~14.6 ms/eval). *Not* split among Jacobian/solve/residual/source. |
| Setup / compile / non-Newton per-step / overhead | ~165 s | ~24% | **DERIVED** residual after the two above |
| — of which phase-2 *Jacobian build* | unknown | — | **UNMEASURED**; only 60 builds occur (lagged), so likely small |
| — rejected-step replay (41 rejections) | unknown | — | **UNMEASURED** (timer absent) |
| — host JVP / dispatch / materialization | unknown | — | **UNMEASURED** (timer absent) |
| — JAX compile vs runtime | unknown | — | **UNMEASURED** (None+reason, not at attribution depth) |
| Dense linear solve | negligible | — | **MEASURED-by-shape**: 62×62, 153 factorizations |

The single load-bearing unknown is the **split of the ~228 s activation-row chunk**. PR-C only pays off if that chunk is dominated by *re-buildable* Jacobian/factorization work; the evidence (lagging already on, FD off) suggests much of it may be *irreducible* residual/source evaluation inside the Newton loop.

---

## 6. Optimization Ranking

Ranked by *measured-target strength × achievable-reduction confidence × physics safety*, not by raw count.

1. **PR-C0 — Telemetry surfacing fix (prerequisite, not optional).** Surface the already-accumulated `trace_stats[*_wall_seconds_total]` (phase-2 corrector + Newton resid/Jac/solve) onto each `span_rows[]` entry; add `rejected_step_*` and `host_jvp/host_jacobian` call-site timers; emit `jax_compile/runtime` value-or-reason *at attribution depth*. Add an **integration** test asserting a real q4 run yields nonzero phase-2 wall (the current unit test cannot catch the 0.0). Fix `summarize_packet_artifacts.py` double-count and make `check_component_wall_attribution.py` exit nonzero on FAIL. **This is the only work that unblocks the PR-C/PR-D decision.**
2. **PR-D — Collision payload reuse (clearest measured target; reduction unproven).** Target = 283 s (42%), concentrated 67–71% in pre-activation rows. Mechanism must replace the magnitude gate (`|dA| ≤ 1e-6`, see §3/item 6) with a **state-change tolerance** on Δpayload across a stage. Risk: the payload depends on continuously-varying T_ν / finite-mass scale, so achievable reuse may be small even with a correct gate. Requires source-budget parity + raw collision-source diagnostics unchanged.
3. **PR-C — Phase-2 corrector reuse (target unconfirmed; partially already done).** Only after PR-C0 confirms the phase-2 wall is real *and* dominated by re-buildable work. If the cost is the 15,603 *linear solves* rather than the (already-lagged, 60) Jacobian builds, the right lever is **factorization/chord reuse of the inner corrector**, not "Jacobian reuse." Do not rebuild machinery that `host_lagged_jacobian_*` already provides.
4. **Rejection/controller tuning at activation.** 41 rejections + 41 replays in row 3; replay wall unmeasured. Defer until PR-C0 times replay.
5. **Hot-loop diagnostic diet.** Memory/clarity only; `tracemalloc_peak≈0.9 GB`. No measured wall justification. Cleanup, not speedup.
6. **Structured/low-rank solve.** Not at q4 (62×62). Revisit only on an *approved* high-q run where host solve wall is shown to dominate.

**Rewrite verdict: REJECTED.** No stable compiled-worthy residual kernel is demonstrated above ~50–70% wall after parity; the one thing a rewrite trivially accelerates (dense LU) is 62×62 and negligible; and the dominant measured cost (payload build) is an *orchestration/reuse* problem, not a hot-loop arithmetic kernel. Concurs with BD282's `STAY_PYTHON_JAX_AND_OPTIMIZE`.

---

## 7. Required Parity Guards (default-on blockers — keep)

- **PR-B LRS/non-LRS parity** must pass before *any* optimization default flips. Unchanged.
- **`N_eff_3T ≥ 3.0` floor tripwire.** Baseline trajectory 9.20 → 5.03 → 3.55 → **3.19** (monotone decreasing toward the floor). Any optimization that perturbs the activation row must re-assert ≥ 3.0.
- **Raw observable preservation:** `Yp`, `D/H`, `Σ_H`, `N_eff_3T`, conservation deltas, raw-negative/nonfinite containers — byte-for-byte unchanged within tolerance; no clipping.
- **New guard required:** the floor/parity checks must **exit nonzero on failure**. The shipped `check_component_wall_attribution.py` prints FAIL and exits 0 — if the real blockers are wired the same way, "default-on blocker" is advisory only and will not stop CI.

---

## 8. Revised Max-6 PR Plan

Aligned to the existing namespace (PR-B = parity; PR-C = phase-2; PR-D = payload). Each PR must, per `AGENTS.md` anti-drift, *move a runtime blocker and consolidate plumbing*, not add a gate.

| PR | Scope | Moves which blocker | Gate to pass |
|---|---|---|---|
| **PR-C0** | Surface existing wall timers onto span rows; add replay/host-JVP call-site timers; JAX value-or-reason at attribution depth; integration test for nonzero phase-2 wall; fix summarizer double-count; checker exits nonzero. | Component-wall attribution (the BD303 blocker) | Real q4 yields finite, nonnegative, non-overlapping phase-2/payload/replay/host walls summing to ≤ total. |
| **PR-B** | LRS/non-LRS FLRW-limit `N_eff_3T` parity. | Endpoint physics blocker | Parity within tolerance; floor ≥ 3.0. |
| **PR-D** | Opt-in state-change-tolerance collision payload reuse (replaces the `1e-6` magnitude gate). | The 283 s payload wall | A/B: fewer builds, source budgets + raw observables within tolerance, parity held, **off by default**. |
| **PR-C** | Opt-in inner-corrector chord / periodic factorization reuse (only if PR-C0 shows re-buildable phase-2 wall dominates). | Activation-row phase-2 wall | A/B: lower phase-2 wall, raw abundances/conservation unchanged, **off by default**. |
| **PR-E** | Activation-row rejection/controller tuning. | 41 rejections + replays | Fewer rejections, total wall not shifted, stability preserved. |
| **PR-F** | Diagnostic-lite + memory/IO consolidation (only after E). | RSS/artifact churn | Smaller artifacts, all raw boundary evidence retained. |

Structured/low-rank solve is intentionally **not** in the six; it is gated on an approved high-q run.

---

## 9. Missing Files / Evidence

- **No component-wall attribution** for phase-2 corrector, rejected replay, host JVP/Jacobian, or JAX compile-vs-runtime at the `span_rows[].h_refinement_attempts[]` depth. This is the core gap; everything in §8 PR-C0 targets it.
- **No inner-corrector factorization counter** (the outer one exists). Cannot tell whether the 15,603 inner solves re-factorize or reuse — decisive for PR-C.
- **No measured rejected-replay wall or trace-row wall** — items 3 and 5 of the ranking cannot be ordered on measured evidence.
- **No design-probe execution.** `phase2_jacobian_reuse_design_probe.json` and `collision_payload_reuse_design_probe.json` are both `SPECIFIED_NOT_IMPLEMENTED`; the phase-2 probe's own `required_baseline` ("BD302 q4 component attribution with phase2/Newton wall") does not exist.
- **JAX persistent cache excluded** (acceptable; `/usr/bin/time -v` + JSON metadata substitute).
- q9/q10 artifacts intentionally absent (correct).

---

## 10. Red-Team Objections

Each red-team question, answered against this audit — including against my own reasoning.

- **"Is this just another gate / manifest / hash / figure / claim wrapper?"** The *packet* largely is packaging — but it honestly says so and preserves raw artifacts. **PR-C0 is not a wrapper**: it surfaces existing timers and deletes a false-green test path, i.e. it moves the attribution blocker. PR-B/PR-D/PR-C move physics/perf blockers. PR-C0 must consolidate (fix summarizer + checker) to satisfy `AGENTS.md`, not add a 13th gate.
- **"Does the recommendation move a runtime physics/solver/performance blocker?"** Yes: PR-C0 → attribution; PR-B → endpoint parity; PR-D → 283 s payload; PR-C → activation-row phase-2. None is cosmetic.
- **"Did profiling prove the optimization target?"** Honestly: **only for payload (283 s, measured).** The phase-2 target is **DERIVED, not proven** — including my own ~228 s estimate, which rests on the network-off baseline being a valid proxy and on attempt-wall bimodality. Treat it as a hypothesis until PR-C0.
- **"Is a rewrite being proposed without evidence?"** No. Rewrite is rejected; the dominant cost is orchestration/reuse, and the dense kernel is 62×62.
- **"Are q9/q10 being smuggled into the plan?"** No. They appear only as prose; no command runs them; not in §8.
- **"Are negative abundances or `Yp` hidden by presentation?"** No. Raw-negative containers preserved, `physical_observable_violations=[]`, `Yp`/`D/H` shown as hot-endpoint intermediates with `passed=False`/`not_promoted`. Not clipped, not promoted.
- **"Is private diagnostic support inflated into public production support?"** No. All four readiness flags are `False`; `qke_scope=out_of_scope`.
- **"Are docs / README / manifests treated as proof?"** No — they were treated as packaging. The two tooling defects I found (summarizer 2× double-count; checker exits 0 on FAIL) are exactly the failure mode of trusting tooling output over raw data; both are flagged for repair.

**Objection turned on my own analysis:** my activation-row phase-2 estimate (~228 s) and the implied ~14.6 ms/eval are **DERIVED**. If the inner corrector's per-eval cost is dominated by collision-source contraction rather than Jacobian/factorization, PR-C's lever is wrong and the residual is largely irreducible. That uncertainty is precisely why the verdict is `TELEMETRY_INCOMPLETE` and why PR-C0 is the gating prerequisite rather than PR-C or PR-D.
