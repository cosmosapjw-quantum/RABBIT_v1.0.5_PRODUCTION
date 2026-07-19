# External Performance / Optimization Audit Report

**Packet:** `BD282_external_performance_optimization_audit_packet_2026-06-02`
**Target:** RABBIT augmented Type-I PSTF no-QKE BBN solver
**Branch / HEAD (per packet):** `feature/v2-f5-closed-model-events` @ `064afe4`
**Auditor environment:** Linux x86_64, Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, JAX 0.10.1 (CPU), no GPU. Sandbox network restricted; no `/usr/bin/time`.
**Method:** CRAG claim extraction + Chain-of-Code probes; the 11 prompt roles were simulated sequentially (no subagent runtime available, matching the packet's own `ROLE_SKILL_USAGE_AND_FALLBACKS.json` note).

A blunt caveat up front: this is a **performance** audit. It does **not** validate endpoint physics. The strongest reason *not* to optimize aggressively right now is physics, not performance: the solver reaches no full-BBN endpoint row in any shipped q4 artifact, and the `N_eff_3T` / LRS-vs-non-LRS parity question is unresolved. Every recommendation below is therefore gated on "do not make wrong physics faster."

---

## 1. Executive Verdict

**`STAY_PYTHON_JAX_AND_OPTIMIZE`** — with two hard conditions: (i) the highest-value optimizations are algorithmic/orchestration changes that stay in Python/JAX, and (ii) they must be opt-in and parity-gated until the endpoint `N_eff_3T`/parity blocker is closed.

A full C++/Rust/Julia rewrite is **not justified now**, and on the evidence it would not address the actual costs. The one thing a compiled language trivially accelerates — the dense linear solve — is a **62×62** system at q4 and is already negligible. The real costs are (a) **collision payload rebuilds with zero reuse** and (b) a **phase-2 nuclear-network Newton corrector that rebuilds its Jacobian on every iteration** (~24.6k rebuilds concentrated in a single span row). Both are fixable in-language.

I partially **disagree with the packet's own internal `pr_acceleration_plan.md`**, which makes "block/low-rank endpoint solver wiring" the acceleration core (PR-1). At q4 that is the *least* justified change (see §8). The justified first moves are payload reuse and phase-2 Jacobian reuse.

---

## 2. One-Page Summary

The headline q4 numbers reproduce **exactly** from the shipped artifacts. `bd299_q4_activation_probe`: total wall **752.1 s**, dynamic collision payload build wall **307.4 s (40.9%)**, **5,471** payload builds, **5,520** host linear solves, **24,595** phase-2 Newton iterations, **0** full-BBN endpoint rows, best `T_gamma ≈ 0.070 MeV` (target is `< 0.01 MeV`). All four shipped q4 variants agree within noise (wall 718–769 s; payload 39–41%; Newton 20.8k–27.8k; **reuse = 0 in every run**; two of four *fail* outright).

The decisive new finding is that **cost is bimodal across the span ladder**, which the headline single number hides:

| span row | T_final (MeV) | wall (s) | rejected | payload (s) | **payload %** | phase-2 Newton iters |
|---|---:|---:|---:|---:|---:|---:|
| 0 (pre-activation) | 0.309 | 139.3 | 0 | 94.7 | **68.0%** | 0 |
| 1 (pre-activation) | 0.132 | 135.1 | 0 | 96.6 | **71.5%** | 0 |
| 2 (pre-activation) | 0.087 | 69.2 | 0 | 49.2 | **71.1%** | 0 |
| 3 (network activation) | 0.070 | 408.4 | 56 | 66.8 | **16.4%** | **24,595** |

Before the nuclear network activates (`Yp = 1e-30`), collision payload build dominates (~70% of wall). At activation (`Yp` jumps to 0.19, T≈0.07 MeV), the row balloons to 408 s — **54% of total wall** — and payload drops to 16%; the remainder is phase-2 corrector + 56 rejected-step replays. **Neither "collision is the bottleneck" nor "phase-2 is not the bottleneck" is correct globally; each is true in a different regime.**

Memory at q4 peaks at **~3.78 GB RSS** (`ru_maxrss_kb = 3,961,216`; `VmHWM` ≈ same), while `tracemalloc_peak` is only **~473 MB**. So ≥ ~2.9 GB (≈76%) is **non-Python** (JAX/XLA executables, collision arrays, BLAS) — and it is *not* the LU factors (a 62×62 matrix is ~30 KB). The reported >20 GB at q9/q10 (prior reports, untested here) is therefore almost certainly **q-dependent collision tensors**, not dense LU.

Telemetry status correction: contrary to the internal re-audit's H17, **per-attempt memory and linear-system-backend telemetry IS present and populated** in the bd299 raw artifacts (nested under `span_rows[].h_refinement_attempts[]`). The gap is that the shipped `summarize_perf_artifacts.py` reads only top-level `rows` and therefore **surfaces none of it** (0 hits). What is genuinely missing is **per-component wall timing** (phase-2 corrector, rejected-step replay, JVP, JAX compile-vs-runtime): only ~41% of wall is directly attributed; the dominant ~59% is *inferred* to be phase-2/replay but cannot be proven from the artifact.

---

## 3. Claim Ledger

Verdict vocabulary: SUPPORTED / CONTRADICTED / PARTIAL / STALE / UNTESTED. "Measured" = reproduced from artifact/probe this session; "Source" = confirmed by reading code; "Inferred" = derived, not directly timed.

| ID | Claim | Verdict | Evidence (this audit) |
|---|---|---|---|
| P1 | Slowdown is **not** primarily Python language overhead | **SUPPORTED** | tracemalloc ≤ ~0.9 GB vs RSS ~3.8 GB; dominant costs are payload-build numerics and 24.6k Newton-Jacobian rebuilds, not Python glue. Source + measured. |
| P2 | q4 ≈ 752 s total, ≈ 307 s payload build | **SUPPORTED** | Exact match: 752.101 s / 307.378 s in `bd299_q4_activation_probe`. Per-row totals reconcile. Measured. |
| P3 | Payload builds occur thousands of times | **SUPPORTED** | 5,471 builds = 5,471 source evals; mean 0.046 s/build. Measured. |
| P4 | Phase-2 Newton/corrector is a major multiplier | **SUPPORTED (regime-specific)** | 24,595 Newton iters, **all** in the activation row (408 s). Internal audit's "not a leading blocker" (H23) is **refuted** for the activation regime. Measured + source. |
| P5 | Rejected-step coupling amplifies payload/corrector | **SUPPORTED** | The only row with rejections (56) is the only expensive/Newton-heavy row; 188 attempts → 56 rejected (30%). Measured. |
| P6 | AP65 endpoint still uses dense LU | **SUPPORTED** | Runtime telemetry: `linear_system_backend = scipy_lu_factor`, `is_dense = True`, `low_rank_active = False`; source `_factorized_linear_solver` (ap65_rhs.py:14324) via scipy `lu_factor`/`lu_solve`; **but W is 62×62** (see §8 caveat). |
| P7 | Dense LU / memory not row-logged unless BD282 landed | **STALE / resolved** | BD282 telemetry **has** landed and **is populated** per attempt (`linear_system_backend`, `ru_maxrss_kb`, `vmhwm_kb`, `tracemalloc_*`, W/J shapes). Measured. |
| P8 | Low-rank/Woodbury/block exist but endpoint routing unproven | **SUPPORTED (stronger: unwired)** | Algebra tests pass (B1: 2, B2: 11). No caller of the Woodbury/low-rank solver outside its own module + tests; host backend enum has no structured option. Source + measured. |
| P9 | JAX compile vs runtime not separated | **SUPPORTED** | No `compile`/`runtime` fields anywhere in artifacts. Indirectly: 11 algebraic block-sparse tests take 231 s (compile-bound). Measured. |
| P10 | Diagnostic JSON/row shaping is in/near hot loops | **SUPPORTED but immaterial to wall** | Source: per-state `_json_sha256` (ap65_rhs.py:9915), `_payload_trace_row` appended per source eval (10916), `_json_safe` dict copies. Churn probe: 2,000 rows → 0.19 s, ~1 MB → ~5.5k rows ≈ <1 s. Real, but a memory-clarity issue, not a wall driver. |
| P11 | q-dependent collision tensors (not state size) drive memory | **PARTIAL / SUPPORTED-by-inference** | At q4, W is 62×62 (LU ≠ memory); ~2.9 GB is non-Python; state A-block = `n_species·n_modes·n_q` scales linearly in q (ap65_rhs.py:12012). q9/q10 >20 GB UNTESTED but consistent with collision tensors, not LU. |
| P12 | tracemalloc captures Python share, not all JAX/XLA | **SUPPORTED** | `tracemalloc_peak ≈ 473 MB` vs `ru_maxrss ≈ 3.78 GB`. Measured. |
| P13 | `ru_maxrss`/`VmHWM` are process-level, need unit tags | **SUPPORTED** | Both present with `ru_maxrss_unit = kilobytes`, `memory_telemetry_platform` tagged. Measured. |
| P14 | Defer q9/q10 until row-level telemetry exists | **PARTIAL** | Row-level *memory/backend* telemetry now exists; per-component **wall** telemetry does not. Defer remains right until wall-attribution lands. |
| P15 | Whole-language rewrite premature | **SUPPORTED** | No stable >50–70% compiled-kernel candidate; the dense-solve target is 62×62; physics unresolved. |
| P16 | Selective compiled kernels plausible only after stable bottleneck ID | **SUPPORTED** | Dominant cost shifts by regime; shapes for the heavy work are not yet pinned to a single stable contract. |
| P17 | Optimization must be parity-gated | **SUPPORTED (policy)** | Endorsed; see §12, §14. |
| P18 | Speedup before `N_eff_3T`/floor resolved can entrench wrong physics | **SUPPORTED (policy)** | Endorsed; binding constraint on sequencing. |
| (open) | `selected_stage_collision_payload_reuse_total = 0` — why? | **ANSWERED** | Policy `auto_small_collision_reuse` falls back to `current_state` rebuild every time (`auto_current_state_count = 1316`, `auto_reuse_count = 0`). Reuse heuristic never fires for the q4 dynamic case. Measured. |

---

## 4. Commands Run and Outputs

All exit codes captured. Tests run with `PYTHONPATH=src JAX_PLATFORMS=cpu`.

| # | Command | Exit | Result |
|---|---|---|---|
| 1 | `pip install -e . --no-deps` (+ `jax[cpu]`, pytest) | 0 | `rabbit` and `rabbit.jax.solver_jax_rodas5p` import OK |
| 2 | `pytest ...::test_woodbury_stage_linear_solve_matches_dense ...::test_low_rank_rodas_step_matches_dense_step_for_linear_rhs` | 0 | **2 passed in 5.02 s** |
| 3 | `pytest -q tests/test_block_sparse_jacobian.py` | 0 | **11 passed in 230.95 s** (compile-bound) |
| 4 | `pytest -q test_augmented_pstf_distribution.py test_augmented_collision_bridge.py test_three_temperature_closure_invariants.py` | 0 | **86 passed in 4.76 s** |
| 5 | `python scripts/run_dense_vs_woodbury_ab_probe.py` | 0 | synthetic 24×24, `max_abs_error 4.2e-17`, passes — **toy algebra only** |
| 6 | `python scripts/run_memory_telemetry_probe.py` | 0 | smoke: 8 MB alloc shows in tracemalloc; RSS 19 MB baseline — **mechanism only** |
| 7 | `python scripts/run_diagnostic_payload_churn_probe.py` | 0 | 2,000 fake rows → json.dumps **0.186 s**, ~1 MB peak — quantifies churn as negligible |
| 8 | `python scripts/run_row_serialization_probe.py` | 0 | negative `Yp = -0.01` serialized **unclipped** (no truncation) |
| 9 | `python scripts/summarize_perf_artifacts.py artifacts/` | 0 | 8 summaries; reproduces wall/payload/Newton; **0 memory/shape fields surfaced** |
| 10 | `python scripts/run_collision_payload_accounting_probe.py artifacts/` | 0 | 8 matches; **reuse = 0 in all** |
| 11 | `run_augmented_continuous_ap65_full_bbn_span_ladder.py --help` | 0 | full flag set discoverable; host backend enum = `{scipy_lu_factor, numpy_solve_per_stage, scipy_gmres_dense_operator}` (no structured option) |
| 12 | bounded pre-activation smoke (`--N-span-end-ladder 1.0 --q-nodes 4 --max-steps 24 --wall-time-budget-seconds 90`) | 0 | **failed at physics gate in 1.1 s**, 0 span rows, `wall_seconds_total = 0.0`, violations emitted honestly — **not runtime evidence**; bare defaults ≠ documented bd299 config, and I declined to guess the full source-refresh flag set |

Probes 5–8 are **synthetic** (self-labeled "not physics/endpoint evidence"); they confirm only that the algebra/serialization mechanisms behave, not that they help RABBIT. The shipped "cheap" suite is almost entirely synthetic; the only RABBIT-specific cheap evidence is the unit tests (2–4), which are algebraic and not endpoint runs.

---

## 5. Profiling Attribution Table (bd299 q4, total = 752.1 s)

| Component | Wall | Share | Confidence |
|---|---:|---:|---|
| Dynamic collision payload build | 307.4 s | **40.9%** | **Measured** (artifact timer) |
| Host dense LU (690 fact. + 5,520 solves of 62×62) | < ~0.1 s | < 0.02% | **Estimated** (62³/3 flops; no timer) |
| Diagnostic JSON/SHA256/trace-row serialization | < ~1 s | < 0.2% | **Estimated** (churn-probe extrapolation) |
| Full-JVP Jacobian (320 evals, frozen source → no payload rebuild) | seconds | low | **Estimated** (no timer) |
| Phase-2 corrector (24,595 Newton-Jacobian rebuilds + 24,595 dense solves + 25,299 residual evals + 67,793 AB2-predictor attempts) + 56 rejected-step replays | bulk of remaining **~444 s** | **~59% (residual)** | **Inferred** (row wall − payload − negligible LU; no phase-2/replay timer) |
| JAX compile (first-call) | unknown | unknown | **Not measured** (P9) |

**The single most important profiling fact: only 40.9% of wall is directly attributed.** The dominant ~59% is inferred to be phase-2-corrector-plus-replay (it is concentrated in the one activation row whose payload is only 16%), but the artifact has **no timer** for it. By Amdahl, an optimization targeting only collision payload caps at ~41% of wall even if it went to zero — and only ~16% in the regime that actually dominates total wall.

---

## 6. Memory Attribution Table (bd299 q4)

| View | Pre-activation row | Activation row (peak) | Interpretation |
|---|---:|---:|---|
| `ru_maxrss_kb` (process RSS, monotonic) | 2,247,380 (~2.14 GB) | **3,961,216 (~3.78 GB)** | grows toward endpoint |
| `VmHWM` | ~2.15 GB | ~3.78 GB | matches RSS |
| `tracemalloc_peak_bytes` (Python objects) | 915,498,814 (~873 MB) | 496,489,366 (~473 MB) | per-attempt peak; **far below RSS** |
| Implied non-Python (JAX/XLA/BLAS/arrays) | ~1.3 GB | **~3.3 GB (≈76%)** | dominant share |
| Dense LU factors (W = 62×62, float64) | ~30 KB | ~30 KB | **negligible** |

Memory is dominated by non-Python allocations. Dense LU is not it. Diagnostic churn (tracemalloc ≤ ~0.9 GB, and dropping) is not it either. The likely driver — and the right thing to confirm before any q9/q10 run — is the q-dependent collision tensor set (P11). The prior >20 GB q9/q10 figure is **UNTESTED here**; do not act on it without logging `W_shape` and collision-array bytes at q9/q10 first (cheap).

---

## 7. Runtime Spine Bottleneck Map

```
CLI  run_augmented_continuous_ap65_full_bbn_span_ladder.py
  -> span ladder (14,815 LOC)         [orchestrates per-N spans; chains restarts]
  -> AP65 RHS (20,992 LOC)            [HOTSPOTS LIVE HERE]
       |- dynamic collision payload build  ~5,471x, 0.046 s ea  -> 40.9% wall  (PRE-ACTIVATION DOMINANT)
       |   reuse policy auto_small_collision_reuse -> reuse=0, always current_state rebuild
       |- frozen-source full-JVP Jacobian   320 evals, lagged stride 2, ~50% reuse  (no payload rebuild)
       |- dense W (62x62) -> _factorized_linear_solver(scipy lu) 690 fact / 5,520 solve  -> ~0% wall
       |- phase-2 BE/BDF2 network corrector  24,595 Newton iters in ONE row     -> bulk of remaining ~59% (ACTIVATION DOMINANT)
       |     rebuilds network Jacobian EVERY Newton iteration (no chord/lag); np.linalg.solve(J,-r)  (ap65_rhs.py:3449)
       |     AB2 predictor: 67,793 attempts, 10,842 displacement-guard rejects (16%)
       |- rejected host steps  56 (30% of attempts) -> replay/rebuild amplification
  -> transport/collision bridge (5,401 + 5,421 LOC)   [payload numerics]
  -> jax/solver_jax_rodas5p.py (2,019 LOC)            [low-rank/Woodbury algebra EXISTS, UNWIRED]
```

Structural debt corroborated and **still growing**: RHS 19,678 → **20,992** LOC, span ladder 13,359 → **14,815**, `validation/` 94,297 → **97,135** LOC across 71 modules since the internal re-audit. The anti-drift concern (plumbing outgrowing physics) is empirically holding.

---

## 8. Solver / Linear-Algebra Path Verdict

**Dense LU is confirmed at the endpoint, and it does not matter at q4.** `W_shape = J_shape = [62, 62]`, float64. A 62×62 factorization is microseconds and ~30 KB; 690 factorizations + 5,520 solves total well under a second of the 752 s. The host `linear-system-backend-policy` enum exposes only `scipy_lu_factor`, `numpy_solve_per_stage`, and `scipy_gmres_dense_operator` — **all dense**; there is no structured/low-rank/Woodbury/block option, and runtime shows `iterative_solve_count = 0` (the BD288 GMRES scaffold is inert).

The low-rank/Woodbury/block-sparse code in `solver_jax_rodas5p.py` is **algebraically correct** (B1: 2 passed; B2: 11 passed) but **unwired**: no caller outside its own module and tests, and the host `low_rank_active` field is a telemetry constant fixed to `False`. **Block-JVP is purely a Jacobian *assembly* policy; the assembled J still feeds a dense W → dense LU. No structured solve reaches the endpoint** (answers Q5 and Q6 unambiguously).

State scaling: A-block size = `n_species · n_modes · n_q` (ap65_rhs.py:12012), so W grows **linearly** in q. At q4 → 62; even q10 lands in the low hundreds — still trivial for dense LU. **Therefore the "dense LU dominates at larger q" hypothesis is unsupported at q4 and implausible at q9/q10 on flop/memory grounds.** Wiring Woodbury/low-rank into the host would optimize a solve that is already ~0% of wall. This is why I rank it last (§14) and disagree with the internal plan's PR-1 framing. The cheap, decisive falsifier is to log `W_shape` at q9/q10 — if it is still a few hundred rows, close the structured-solver track entirely.

---

## 9. Collision Payload Verdict

Payload build is the **pre-activation** dominant cost (~70% of wall in rows 0–2) and ~41% of total wall. 5,471 builds, mean 0.046 s, max 0.54 s; `source_evaluations = builds` (one build per source eval). **Reuse never fires** in any of the four variants: `auto_small_collision_reuse` resolves to `current_state` rebuild every time. This is the largest *cleanly attributed* lever. The open question "why is reuse zero" is answered: the auto-reuse heuristic's fallback path is taken unconditionally for the q4 dynamic case.

Risk: payload reuse changes physics if current-state semantics are not honored. The corrector/RHS evaluate *current-state* sources, so reuse must be gated by a state-change tolerance and must preserve the raw source diagnostics. This is a real-but-bounded engineering change in Python/JAX, not a rewrite.

---

## 10. Phase-2 Corrector Verdict

**This is the dominant cost in the regime that dominates total wall, and it is under-instrumented.** 24,595 Newton iterations occur entirely in the single activation row (408 s, 54% of total wall). Source inspection (ap65_rhs.py ~3400–3470) shows the backward-Euler network Newton loop **rebuilds the network Jacobian on every iteration** (`jacobian_evaluation_count == iteration_count == linear_solve_count == 24,595`) and solves with `np.linalg.solve(J, -residual)`. There is no chord/modified-Newton reuse and no cross-substep Jacobian caching. The AB2 predictor fires 67,793 times with 10,842 (16%) displacement-guard rejections — i.e., the predictor frequently overshoots and is clamped.

The network Jacobian is small (nuclide-sized), so the per-iteration *solve* is cheap; the cost is the **24,595 Jacobian assemblies** (network kinetics/flux evaluations). Caching/lagging the network Jacobian (chord iteration with periodic refresh, mirroring the host's lagged-Jacobian stride) is the highest-leverage *algorithmic* fix for the activation regime. Recent PRs (BD296–BD298) tuned predictor throttles and guard counters but did **not** reduce the iteration count: bd298 22,857 vs bd299 24,595 — same order of magnitude, and the `window`/`damped` variants (27,818 / 20,828 iters) still **failed**. Tuning guards is not reducing the cost.

Do **not** delete the corrector (no falsification justifies that). Make it cheaper per step.

---

## 11. JAX Compile / Runtime Verdict

**Not separated** (P9 SUPPORTED). No artifact field distinguishes XLA compile from execution. Indirect evidence that compile is non-trivial: 11 purely algebraic block-sparse tests took 231 s, dominated by per-test tracing/compilation. The runner exposes `--jax-compilation-cache-dir` and a persistent-cache threshold, so a persistent compilation cache is available but its effect is unmeasured. Before any port decision, wrap compiled kernels with `block_until_ready()` timing and record `compile_seconds` vs `runtime_seconds` per row. Until then, no raw wall comparison across jacobian-policy or backend modes is trustworthy.

---

## 12. Diagnostic JSON / Object-Churn Verdict

**Present in the hot loop, immaterial to wall, modestly relevant to memory and clarity.** Source confirms per-state `_json_sha256` fingerprinting and per-source-eval `_payload_trace_row` construction with `_json_safe` dict copies (~5,471 trace rows at q4). The churn probe bounds the cost: 2,000 rows serialize in 0.186 s / ~1 MB, so ~5.5k rows cost well under a second of 752 s. The `payload_metadata_policy = hot_loop_minimal` mode is active yet `inner_loop_trace_rows_suppressed = 0`, so the "minimal" mode is not actually suppressing rows. Moving trace/fingerprint emission to the accepted-step (row) boundary is a low-risk cleanliness/memory win (consistent with guardrail BD2) but is **not** a wall optimization and must not be sold as one.

---

## 13. Language / Runtime Decision Tree

```
Is endpoint physics (N_eff_3T, LRS/non-LRS parity) resolved?
  NO  -> STOP. Do not default-optimize. Parity + telemetry first.  [CURRENT STATE]
  YES -> Is one numerical kernel repeatedly > 50-70% of wall with a STABLE shape/contract?
           NO  -> STAY_PYTHON_JAX_AND_OPTIMIZE (payload reuse, phase-2 Jacobian reuse, fewer rejections)
           YES -> Is it separated from Python-object + JAX-compile overhead, parity-gated,
                  behind a backend policy with reference fallback, with reproducible before/after?
                    NO  -> add timers/parity first; do not port
                    YES -> SELECTIVE_PORT_LATER (one kernel only: most likely collision pairwise
                           quadrature/source assembly, NOT the 62x62 dense solve)
Full rewrite -> REJECTED now (endpoint unresolved; no >50-70% stable kernel; dense-solve target trivial;
                a rewrite would reproduce dynamic-payload/dense-LU architecture).
```

Where the candidate kernel would be, *if* the threshold is ever met: the **pairwise νν collision quadrature / radial-angular source assembly** (real numerical work, ~41% of wall pre-activation, and JAX-Pallas/Numba/C++-able *if* its shape is frozen). The dense solve is explicitly **not** a port candidate.

---

## 14. Top 5 Optimizations (ranked)

Scored: expected speedup / correctness risk / implementation difficulty / current test coverage.

| Rank | Optimization | Expected speedup | Correctness risk | Difficulty | Test coverage today | Notes |
|---|---|---|---|---|---|---|
| **1** | **Phase-2 network-Jacobian reuse (chord/modified-Newton + periodic refresh, cross-substep cache)** | High in activation regime (the row = 54% of total wall; targets 24.6k Jac rebuilds) | Medium (must preserve raw abundances + mass/charge conservation) | Medium | Partial (mass/charge deltas logged; needs parity test) | Mirror the host's lagged-Jacobian stride. Direct answer to the open question. |
| **2** | **Collision payload reuse that actually fires (state-change-tolerance gate, honor current-state semantics)** | High pre-activation (≤41% of wall; the only cleanly-attributed lever) | Medium (reuse can change sources if state drift not bounded) | Medium | Weak (bridge unit tests exist; no reuse-parity test) | Fix why `auto_small_collision_reuse` → 0. Keep raw source diagnostics. |
| **3** | **Reduce host rejected steps near network activation (controller/step-size tuning around the Yp turn-on)** | Medium (56 rejections → replay amplification in the 408 s row) | Low–Medium | Medium | Weak | Rejections are the multiplier that couples payload + corrector cost. |
| **4** | **Per-component wall + compile/runtime telemetry (phase-2, replay, JVP, XLA compile)** | Speedup = 0, but **unblocks** all of the above and any port decision | Very low | Low | n/a | Highest *information* value; only 41% of wall is attributed today. Cheap. |
| **5** | **Move trace/SHA256/`_json_safe` emission to row boundary; make `hot_loop_minimal` actually suppress** | Low wall, modest memory | Low | Low | n/a | Cleanliness + memory; do not oversell. |

**Explicitly de-prioritized:** wiring Woodbury/low-rank/block-sparse into the host solve. At a 62×62 (and q-linear) W it optimizes ~0% of wall. Keep the algebra as a tested library; do not route it to the endpoint until/unless `W_shape` at q9/q10 proves otherwise.

---

## 15. Recommended PR Sequence (≤ 6)

Sequencing is constrained by P18: **physics-parity work and telemetry must precede default-on optimization.** I reorder the internal `pr_acceleration_plan.md` accordingly.

1. **PR-A — Per-component wall + compile/runtime telemetry.** Add timers around: phase-2 corrector, rejected-step replay, full-JVP Jacobian, payload build (already timed), and XLA compile vs runtime (`block_until_ready`). Also fix `summarize_perf_artifacts.py` to descend into `span_rows[].h_refinement_attempts[]` so existing memory/backend fields are surfaced. Cheap, very low risk, unblocks everything. *(This is the legitimate "instrumentation" PR; it moves a real blocker — wall attribution — so it does not violate the anti-drift rule.)*
2. **PR-B — Controlled LRS/non-LRS FLRW-limit `N_eff_3T` parity** (= internal PR-2). The binding physics blocker. Must land before any optimization is made default.
3. **PR-C — Phase-2 network-Jacobian reuse (chord + periodic refresh), parity-gated, opt-in.** Targets the activation-regime dominant cost. Roll back on any drift in `Yp`/`D-H`/mass-charge residual.
4. **PR-D — Collision payload reuse that fires, behind a state-change tolerance, opt-in.** Targets the pre-activation dominant cost; preserve raw source diagnostics.
5. **PR-E — God-module split of AP65 RHS, then span ladder** (= internal PR-4 then PR-5), fail-closed on missing init/source policy. Enables safe future physics/solver edits; halts the LOC growth documented in §7.
6. **PR-F — Physics-invariant tests + collisional `ell_max=2` fencing + Teff/plumbing cleanup after call-graph check** (= internal PR-3 + PR-6 folded). Cheap hardening; deletes obsolete gates per the anti-drift consolidation rule.

Structured-solver (Woodbury/low-rank) endpoint wiring is **not** in the six. Revisit only if PR-A's q9/q10 `W_shape` telemetry shows the host matrix has become large.

---

## 16. Missing Evidence / Files

- **`BD281_external_reaudit_report_2026-06-02.md`** — referenced as a packet input but **absent** (placeholder `MISSING_BD281...` only). Several internal-audit verdicts cite it; cannot be independently checked.
- **`bd292_q4_laguerre_dynamic_full_endpoint_probe.json`** (19.7 MB) — a *full-endpoint* q4 probe exists in the repo but was **excluded** from the packet (`SKIPPED_LARGE_ARTIFACTS.md`). The packet ships only *hot-endpoint* (T≈0.07 MeV) q4 probes. The actual full-endpoint q4 evidence is therefore not auditable here.
- **q9/q10 artifacts** (bd274/275/276, ~5–10 MB each) — skipped; the >20 GB memory claim is **UNTESTED** in this packet.
- **bd278 endpoint-matrix shards** (30 MB) — skipped; the `N_eff_3T ≈ 2.994` parity evidence relies on prior read-only probes, not re-runnable here.
- **No per-component wall telemetry** in any shipped artifact (phase-2/replay/JVP/compile) — the single most consequential gap; ~59% of q4 wall is unattributed.
- **Fresh bounded q4 run** — not performed. Bare-default flags fail the physics gate in ~1 s; reproducing the documented bd299 configuration requires the full source-refresh flag set, which I declined to guess (per packet instruction). Headline numbers were instead reproduced from shipped artifacts (exact).

---

## 17. Exact Next Commands

Cheap, in priority order (all `PYTHONPATH=src JAX_PLATFORMS=cpu`):

```bash
# 1. Re-run the unit/algebra gates (already green here) before any change:
pytest -q tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_woodbury_stage_linear_solve_matches_dense \
         tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_low_rank_rodas_step_matches_dense_step_for_linear_rhs
pytest -q tests/test_block_sparse_jacobian.py
pytest -q tests/test_three_temperature_closure_invariants.py

# 2. Confirm the reuse-never-fires finding directly from a real artifact:
python scripts/run_collision_payload_accounting_probe.py artifacts/   # expect reuse=0 in all rows

# 3. Reproduce the headline q4 decomposition (use the raw artifact, not the extract,
#    and descend into span_rows[].h_refinement_attempts[] for memory/backend fields):
python - <<'PY'
import json; d=json.load(open('artifacts/raw/diagnostic_outputs/bd299_q4_activation_probe.json'))
for r in d['rows'][0]['span_rows']:
    a=r['h_refinement_attempts'][-1]
    print(round(r['T_final_MeV'],4), round(r['wall_seconds'],1),
          round(a['dynamic_collision_payload_build_wall_seconds_total'],1),
          a['phase2_conservative_extent_corrector_newton_iteration_count_total'],
          a['W_shape'], a['ru_maxrss_kb'], a['tracemalloc_peak_bytes'])
PY
```

Medium (only after a maintainer confirms exact flags; do **not** guess):

```bash
python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py --help   # re-read enums
# Reconstruct the bd299 invocation from the row metadata (jacobian-policy,
# stage-collision-payload-policy, all source-refresh-* policies, phase2-* policies,
# h-max, q-nodes, N-mu/N-phi, atol/rtol, max-steps, wall-time-budget) and add
# --jax-compilation-cache-dir <dir>. Wrap with the python `time` builtin (no /usr/bin/time here).
# A/B exactly one variable at a time (payload policy, or jacobian policy, or corrector Jacobian-reuse).
```

Forbidden until PR-A telemetry exists **and** the user accepts slow/high-memory execution: any q9/q10 run.

---

## 18. Red-Team Objections (against my own conclusions)

1. **q4 is not q9/q10.** My strongest engineering claim (dense LU is negligible) rests on `W = 62×62` at q4 and a linear-in-q scaling argument. If `n_modes` (angular) grows super-linearly at high q, or the host couples additional dense blocks, W could be larger than I project. *Mitigation:* PR-A logs `W_shape` at q9/q10; this is cheap and decisive. I flag the dense-LU verdict as q4-proven, q9/q10-extrapolated.
2. **~59% of wall is inferred, not measured.** I attribute the activation-row remainder to phase-2 + replay by subtraction (row wall − payload − negligible LU). The artifact has no phase-2/replay timer; an unseen cost (e.g., first-call XLA compile concentrated in that row, or rejected-step overhead larger than I assume) could be misattributed. *Mitigation:* PR-A timers; treat my phase-2 ranking as a strong hypothesis, not a measurement.
3. **Optimizing now can entrench wrong physics.** Two of four shipped variants *fail*, no run reaches endpoint, and `N_eff_3T` ranges absurdly across span rows (9.58, 5.23, 3.70, 3.32 at the four T-values) — the corrector/closure is clearly not yet physically settled in the activation region. Speeding up the corrector or caching payloads before parity is closed risks making an unphysical trajectory cheaper to produce. *This is why §15 puts parity (PR-B) ahead of all default-on optimization and keeps PR-C/PR-D opt-in.*
4. **Payload "reuse" may be physically load-bearing.** That reuse never fires might be *correct* — the dynamic q4 source may genuinely change every step, so any reuse would alter physics. *Mitigation:* PR-D must be a state-change-tolerance gate validated against a no-reuse parity baseline, not an unconditional cache.
5. **The audit packet itself is plumbing.** Per the project's own anti-drift guardrail, a "performance audit packet" that ships synthetic probes and telemetry schemas but no endpoint run could be another instance of evidence-plumbing outgrowing physics. The honest reading: the *real* blocker is still the unreached `T_gamma < 0.01 MeV` endpoint and the parity gap; performance work is secondary and must not become the headline of progress.
6. **JAX version skew.** I ran JAX 0.10.1; the artifacts were produced on an unrecorded (older, pinned `>=0.4.20`) JAX. Compile times and some numerics can differ. My executed results (unit tests, synthetic probes) are version-current; the *artifact* numbers are from the project's own run and were only reproduced by re-reading, not re-execution.

---

*Prepared as an independent external performance/optimization audit. Endpoint physics is explicitly out of scope and is not validated by anything herein. Synthetic and partial-span evidence is labeled as such throughout; no toy output is presented as research evidence.*
