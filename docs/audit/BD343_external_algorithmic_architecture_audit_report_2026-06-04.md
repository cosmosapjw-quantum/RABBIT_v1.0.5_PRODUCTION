# BD343 External Algorithmic Architecture Audit Report

**Subject:** RABBIT augmented Type-I PSTF no-QKE BBN solver (AP65 / Rodas5P path)
**Packet:** `BD343_full_context_algorithmic_architecture_audit_packet_2026-06-04`
**Date:** 2026-06-04
**Method:** CRAG claim grading + Chain-of-Code artifact verification + 5-role internal debate + Best-of-N experiment selection
**Scope honored:** QKE out of scope; no public-production/publication claim; raw negative/nonfinite evidence preserved; no default-on optimization recommended before PR-B parity + cold `N_eff_3T >= 3.0` floor; no q9/q10/high-q runs recommended except marked optional.

All numeric claims below were recomputed from packet JSON with the commands quoted in §6/§7. No numbers were taken on faith from packet prose.

---

## 1. Executive Verdict

**`SOURCE_RESPONSE_FIRST`** — executed through a *bounded* runtime seam, not a re-architecture.

The single most decision-relevant fact in this packet is that the implicit operator's collision-source response is, by construction, **damping-only and diagonal-only**. The diagonal collision Jacobian (`_frozen_source_collision_dA_diagonal_jacobian`, ap65_rhs.py:15770) activates a diagonal entry *only* where `A * dA < 0` and then clips it to `[-cap, 0]`. It mathematically cannot represent (a) growth-direction response (`A * dA > 0`), or (b) any off-diagonal coupling across q-nodes, species, modes, temperatures, or shear. Every other Jacobian policy in the hot path is a "frozen-source" base plus one of these damping-oriented add-ons.

Until a finite-difference reference quantifies how much of the true dynamic source-response norm this misses, you **cannot rationally order** phase-2 redesign, payload caching, or a solver bakeoff, because all three are plausibly *downstream symptoms* of the source-response gap:

- the largest wall bucket (phase-2 corrector, 181.4 s) is consistent with a conservative network corrector doing work the main implicit step's Jacobian cannot;
- the sharpest wall lever measured (accepted-recovery, BD340→BD341, −174 s total of which **−160 s is in phase-2**) shows that *step rejections* inflate phase-2, and a Jacobian blind to growth/coupling modes is a textbook cause of rejections.

Three boundary conditions on this verdict:

1. **It is not `DEEP_REARCHITECTURE_REQUIRED`.** There is no measured stable residual compute kernel above 50–70% wall after parity. The residual is 23.4% and is largely orchestration/telemetry (§6, §7), not an irreducible numerical core.
2. **It is not `SOLVER_BAKEOFF_FIRST`.** Outer linear solve is 0.45 s (0.11%), the boundary forbids it before source-response semantics are fixed, and a bakeoff is presently *impossible* without unifying two divergent Rodas5P steppers (§5).
3. **It necessarily front-loads one bounded `KEEP_AP65_SPLIT_RUNTIME` PR (PR-0).** The FD reference function the verdict depends on (`dynamic_collision_source_response_reference_np`) is *called* by ap65_rhs.py but its module is **absent from the packet** (§11), the audit path is `nonlrs_s2`-only with default policy `"none"`, and the span ladder reaches into **22 private RHS helpers**. You cannot run the source-response reference cleanly without a minimal typed seam. That seam is the enabling cost of `SOURCE_RESPONSE_FIRST`, not a competing program.

---

## 2. Claim Ledger (CRAG)

Labels: IMPLEMENTED / VALIDATED / PARTIAL / UNTESTED / CONTRADICTED / FORBIDDEN. "VALIDATED" requires a test/artifact/derivation in-packet, not prose.

| # | Claim | Grade | Evidence (code path / artifact) |
|---|---|---|---|
| C1 | BD342 q4 produces real exclusive component-wall attribution | **VALIDATED** | `bd342_component_wall_summary.json` → exclusive 6 sum = 328.6576 s = `attributed_wall_seconds_total`; residual 100.156 s = `residual_unattributed_wall_seconds`; recomputed in §6 |
| C2 | AP65 RHS is a runtime/solver/payload/telemetry mega-module | **VALIDATED** | `wc -l` = 24,827 lines, 277 top-level defs/classes; ladder 16,890 lines |
| C3 | Dense LU is not the q4 bottleneck | **VALIDATED (CONTRADICTS the "optimize LU" path)** | `outer_linear_system` = 0.4511 s; W/J shape `[62,62]`; `scipy_lu_factor`; `linear_system_low_rank_active=false` |
| C4 | Phase-2 corrector is the largest exclusive bucket | **VALIDATED** | `phase2_corrector` = 181.355 s (42.29%) |
| C5 | Phase-2 *wall* is dominated by orchestration, not Newton linear algebra | **VALIDATED (new finding)** | Named Newton algebra (jac+solve+resid+flux) = 17.06 s (9.4% of phase-2); bookkeeping+AB2 pred+AB2 guard = 105.79 s (58.3%); ~50 s inside `newton_solve_call` unattributed to named sub-walls (§6) |
| C6 | Payload reuse now fires | **VALIDATED** | `stage_collision_payload_state_reuse_total=2630`, `..._reuse_evaluation_total=4319`, `..._current_state_total=1689` |
| C7 | Provider build dominates the radial factory | **VALIDATED** | `payload_pstf_radial_factory_provider_build` = 81.017 s = 61.5% of `payload` (131.78 s), 89.7% of `radial_factory` (90.28 s) |
| C8 | Provider build is *cacheable* state-independent scaffolding | **UNTESTED** | No ON/OFF output-equality artifact; provider-build source is in the **missing** `dynamic_collision_runtime` module (§11) |
| C9 | Radial source→occupation conversion sign/coordinate is correct | **PARTIAL** | `distribution_rhs_to_augmented_rhs` (augmented_pstf_distribution.py): `df/dA=-f(1-f)`, `dA=-df/max(f(1-f),eps_f)` — math correct & focused tests exist; small-`f`/clipped limiting magnitude is approximate (eps_f floor, §3) |
| C10 | 3T heavy-neutrino bank heat-capacity factor is correct | **PARTIAL (VALIDATED-by-inspection)** | `coupled_3T_rhs*`: `d2 = 2 * _drho_nu_pair_dT(T_nu_x)`, bank energy / bank capacity; focused invariant tests claimed, not endpoint |
| C11 | q-weight convention (energy vs raw Laguerre) is consistent | **PARTIAL** | `_default_q_energy_grid` returns `(q, w*exp(q)*q**3)` from `laggauss` — correct energy (q³) moment; explicit raw-weight round-trip recovery not located in snapshot |
| C12 | Implicit operator contains an *adequate* dynamic collision source response | **UNTESTED → leaning CONTRADICTED** | Diagonal Jacobian is provably damping-only + diagonal-only (ap65_rhs.py:15810–15815); no FD norm comparison run |
| C13 | A finite-difference dynamic source-response reference exists | **PARTIAL** | `dynamic_collision_source_response_reference_np` is *called* (ap65_rhs.py:1577) but its module (`dynamic_collision_runtime`) is **not in the packet**; audit policy default `"none"`; `nonlrs_s2`-only |
| C14 | BD342 is endpoint-validated physics | **CONTRADICTED (correctly self-denied)** | `passed=false`, `physical_full_bbn_span_ready=false`, violation `resolution_ladder_failed_or_nonendpoint_rows`; stops at `T_gamma≈0.0699 MeV` vs `full_bbn_endpoint_MeV=0.01` |
| C15 | Raw negative/nonfinite evidence is preserved (not clipped) | **VALIDATED** | `observable_policy = raw_solver_bbn_observables_no_truncation_or_sign_repair`; all `raw_candidate_negative_*` counts = 0; `nonfinite_rejection_count=0` |
| C16 | A cold-endpoint `N_eff_3T >= 3.0` floor under the real collision config exists | **UNTESTED** | Only cold-complete artifact (q14/q15 BD220) ran `collision_projection_policy = flrw_monopole_only` (§3) → monopole-projected, not non-LRS S2; `controlled_flrw_lrs_nonlrs_parity = None` |
| C17 | Accepted-recovery is a real, policy-sensitive wall lever | **VALIDATED** | BD340 (no policy) 602.0 s / phase-2 344.6 s vs BD341 428.0 s / phase-2 184.1 s (§7) |
| C18 | A whole-language/whole-solver rewrite is justified now | **UNTESTED → reject** | No residual-kernel-dominance evidence; CRAG external-code ledger grades this UNSUPPORTED |
| C19 | QKE / off-diagonal coherence | **FORBIDDEN (out of scope)** | Packet boundary |
| C20 | Span ladder couples to RHS internals | **VALIDATED** | Ladder imports 22 private `_`-helpers vs 2 public names from ap65_rhs (§5) |

---

## 3. Physics Consistency Ledger

Answers to the packet's eight physics questions, each graded and tied to code.

| Q | Item | Finding | Grade |
|---|---|---|---|
| P1 | Signs/dims of `dQ_nue_pair_N`, `dQ_nux_bank_N`, `dA_collision` across modules | `coupled_3T_rhs` sets `dT_source=-dQ_total_N/drdT_plasma` (plasma loses what ν gains), `dT_nue=-T+dQ_nue/d1`, `dT_nux=-T+dQ_nux/d2`; `nu_nu` source enforces `dQ_nux_bank=-dQ_nue_pair` (energy-conserving). Signs internally coherent in the inspected files | **PARTIAL** (consistent in nudec_coupled; cross-module `dA_collision` sign lives partly in the **missing** runtime module) |
| P2 | Heavy-neutrino bank heat-capacity factor | Correct: `d2 = 2 * _drho_nu_pair_dT(T_nu_x)` = 2-pair (4-state) capacity; `dQ_nux_bank` is bank energy. Same factor in both RHS variants | **VALIDATED-by-inspection** |
| P3 | q-weight convention between energy and raw Laguerre weights | `energy_weight = w * exp(q) * q**3`; `exp(q)` cancels Laguerre `exp(-q)`, `q**3` is the energy moment measure. Forward direction verified | **PARTIAL** (round-trip recovery not located) |
| P4 | `distribution_rhs_to_augmented_rhs` near small/clipped occupations | `dA = -df/max(f(1-f), eps_f)`, `eps_f=1e-12`. Sign correct; where `f(1-f) < eps_f` the magnitude is **floored**, i.e. dA is under-estimated in the deep Pauli-blocked / near-empty tails | **PARTIAL** (correct sign, biased magnitude in clipped tails) |
| P5 | `collision_projection_policy: flrw_monopole_only` | It is a **projection** (collision source onto FLRW monopole), used as a stabilizing/private approximation — not a derived physical assumption. It is currently load-bearing: the only cold-complete artifact (q14/q15) used it | **PARTIAL / temporary approximation** |
| P6 | Fixed diagonal / S2 non-LRS preserves needed angular collision info | Cannot be affirmed. The diagonal damping Jacobian discards anisotropic response, and the cold floor exists only under monopole projection. Angular-source residual under non-LRS S2 is unmeasured at the endpoint | **UNTESTED** |
| P7 | Raw negative/nonfinite/positivity diagnostics preserved everywhere | BD342 surfaces explicit raw-negative and nonfinite counters (all 0 here) and the run is `..._no_truncation_or_sign_repair`. No path observed hiding failures behind summaries | **VALIDATED** (for this run) |
| P8 | `N_eff_3T` clearly marked pre-asymptotic above cold endpoint | `bbn_observables/N_eff_method = 3T_asymptotic_temperature_ratio`; per-span values swing (first span `N_eff=9.58`, `Yp=1e-30`) while the bounded aggregate is `N_eff=3.32`, `Yp=0.123` at 0.0699 MeV — visibly pre-asymptotic | **VALIDATED** |

**Physics note on Yp:** BD342's `Yp=0.123` at `T_gamma≈0.0699 MeV` is *mid-synthesis* (deuterium bottleneck barely open), not a 50% deficit against canonical `Yp≈0.247`. It must **not** be read as a defect; it is simply pre-endpoint. The earlier "isotropic endpoint ~33% low" observation is a different run and not evidenced here.

---

## 4. Numerical Methods Critique

**4.1 The implicit Jacobian is structurally incomplete (primary issue).**
`_frozen_source_collision_dA_diagonal_jacobian` builds the collision contribution as:

```
damping = finite & (|A| > floor) & (A*dA < 0)     # restoring directions only
diag[damping] = dA[damping] / A[damping]            # negative by the mask
diag = clip(diag, -cap, 0.0)                         # strictly non-positive
J[A_idx, A_idx] = diag                               # diagonal only
```

This is **safe but blind**: it never injects spurious instability into the Newton operator (good), but it sees neither growth modes nor any cross-coupling (q×q, species×species, mode×mode, A×T_γ/T_ν, A×Σ). If the true source response near q4 activation carries weight in those directions, the implicit step is solving with a deficient operator and the corrector/rejection machinery absorbs the difference. This is the mechanism that links the diagonal Jacobian to both the 181 s phase-2 bucket and the BD340 rejection explosion.

**4.2 Phase-2 wall is an orchestration cost, not a stiffness-solve cost.**
Decomposed (§6): the actual implicit linear algebra inside phase-2 is 17.06 s. The dominant 105.79 s is `step_attempt_bookkeeping` (65.92 s) + AB2 predictor (23.67 s) + AB2 residual guard (16.19 s), plus ~50 s inside `newton_solve_call` not attributed to any named Newton timer. So two distinct questions must be separated:
- *Is phase-2 numerically necessary?* — open; answered only after the FD reference (E1) shows whether the main step's Jacobian is adequate.
- *Why is phase-2 wall large?* — measured: orchestration/bookkeeping/predictor-guard, refined vs coarse adaptive pair (114.5 s refined + 59.1 s coarse), not dense linear algebra.

**4.3 Two divergent Rodas5P implementations.**
The in-tree JAX solver (`solver_jax_rodas5p.py`) has structured seams: `_rodas5p_step_low_rank`, `_rodas5p_step_custom_linear_solver`, `_rodas5p_step_schur`. The AP65 hot path uses its own `_host_rodas5p_step` (ap65_rhs.py:16860) with local Jacobian policy and local dense solve, **bypassing** those seams. A same-RHS/same-Jacobian solver bakeoff (E4) is therefore not currently runnable — the step interface is not shared.

**4.4 Rejection controller principled vs tuned.**
Accepted-recovery cuts rejections 45→1 (BD340→BD341/342) and the gain is concentrated in phase-2 wall. This is a real lever, but BD340 is a confounded control (cache-bound + no accepted-recovery), so the *isolated* causal effect of accepted-recovery is not cleanly measured. The controller currently looks tuned to the activation region, not derived.

**4.5 q-weight / occupation handling.** Forward energy-moment convention is correct (§3 P3). The `eps_f` Pauli floor in the occupation→logit conversion is a magnitude approximation in clipped tails; acceptable for stability, but it is one more place a FD source-response reference should probe.

---

## 5. Code Architecture Critique

**5.1 The mega-module is the structural reason experiments feel like surgery — confirmed quantitatively.**
`augmented_continuous_ap65_rhs.py` = 24,827 lines / 277 top-level defs, owning solver policy, payload eval + cache, phase-2, host Rodas5P stepping, Jacobian/JVP policy, linear-solver selection, trace building, wall timers, **and** JSON claim-boundary text. The span ladder imports **22 private `_`-prefixed helpers** from it (rejection ceilings, phase-2 Newton policies, payload-reuse tolerances, cache keys) against only 2 public names. Any source-response or solver experiment must reach into these privates — this is the binding architectural constraint, and it directly blocks E1/E4.

**5.2 Namespace leakage.**
Host/NumPy dynamic source-response probes live under `jax/augmented_typeI_replay.py` (4,780 lines) despite the JAX namespace; the structured JAX solver seams in `solver_jax_rodas5p.py` are unused by the AP65 host stepper. Two Rodas5P codepaths is the most expensive smell because it forbids a clean bakeoff.

**5.3 Telemetry schema is hand-mirrored.**
`scripts/summarize_perf_artifacts.py` mirrors dozens of field names manually; runtime schema changes require parallel edits in scripts/tests. This couples telemetry evolution to runtime edits.

**5.4 The minimal seam (what PR-0 should expose).** Just enough typed surface to run E1 without private imports:

```
JacobianProvider        : frozen_source_jvp | finite_difference_reference | structured_source_response
CollisionPayloadProvider: evaluate_base / evaluate_stage(reuse_policy) / source_response_operator(state)
AP65RHSKernel           : evaluate_rhs(N, y, payload)  → compact stats, no JSON
StepController           : host Rodas5P step + rejection policy + phase-2 hook
ArtifactBuilder          : compact records → JSON, outside the hot loop
```

This is a runtime/Jacobian/source-response boundary, not a file-size cosmetic. Do **not** create new readiness/hash/figure/manifest gates while doing it (anti-drift history).

---

## 6. Component Wall Table — Exclusive vs Overlapping

**Verified with:**
```
python3 -c "import json; s=json.load(open('artifacts/bd342_radial_factory_timers_q4/bd342_component_wall_summary.json'));
cwa=s['component_wall_attribution']; comp=cwa['components']; g=lambda k: comp.get(k,{}).get('wall_seconds',0);
ex=g('phase2_corrector')+g('payload')+g('host_jacobian')+g('outer_linear_system')+g('jax_compile')+g('jax_runtime');
print(ex, cwa['attributed_wall_seconds_total'], cwa['total_wall_seconds'], cwa['total_wall_seconds']-ex)"
# -> 328.657579 328.657579 428.813504 100.155925
```

### Exclusive buckets (these sum to the attributed total; safe to add)

| Component | Wall (s) | % of 428.81 |
|---|---:|---:|
| phase2_corrector | 181.355 | 42.29% |
| payload | 131.784 | 30.73% |
| host_jacobian (= host_jvp_jacobian) | 15.068 | 3.51% |
| outer_linear_system | 0.451 | 0.11% |
| jax_compile | 0.000 (unavailable) | 0.00% |
| jax_runtime | 0.000 (unavailable) | 0.00% |
| **Attributed total** | **328.658** | **76.64%** |
| **residual_unattributed** | **100.156** | **23.36%** |
| **Selected total** | **428.814** | **100%** |

### Overlapping sub-timers (diagnostic substructure — **do NOT add to exclusive buckets**)

Phase-2 internal (note `row_count` differs — these are not independent rows):

| Sub-timer | Wall (s) | Relation |
|---|---:|---|
| phase2_adaptive_pair | 173.712 | ⊂ phase2_corrector |
| phase2_step_attempt | 173.559 | ≈ adaptive_pair |
| ↳ phase2_refined_step_attempt | 114.486 | partitions step_attempt |
| ↳ phase2_coarse_step_attempt | 59.072 | partitions step_attempt |
| phase2_newton_solve_call | 67.177 | the Newton portion |
| ↳ named Newton algebra (jac 5.882 + linsolve 1.904 + resid 7.337 + flux 1.933) | **17.056** | only 9.4% of phase-2 |
| ↳ unattributed inside newton_solve_call | ~50.12 | **gap** |
| phase2_step_attempt_bookkeeping | 65.921 | orchestration |
| phase2_ab2_rhs_predictor + ab2_residual_guard | 39.865 | orchestration |
| **orchestration subtotal (bookkeeping+AB2)** | **105.79** | **58.3% of phase-2** |

Payload internal:

| Sub-timer | Wall (s) | Relation |
|---|---:|---|
| payload_pstf_radial_factory | 90.282 | ⊂ payload |
| ↳ provider_build | 81.017 | 61.5% of payload |
| ↳ process_config 8.198 / radial_grid_kwargs 5.029 / validation 0.593 / others ~0.39 | 14.20 | rest of factory |
| **payload NOT inside radial_factory** | **41.50** | **gap (31.5% of payload, no sub-timer)** |

Replay/rejected timers (`rejected_step_attempt = rejected_step_replay = 3.351`) are replay-overlap, excluded from residual arithmetic by design.

**Three unattributed pockets to chase (≈191 s combined):** 100.16 s top-level residual + 41.50 s payload-not-in-factory + ~50.12 s in-Newton. These dominate the "where does the wall go" question more than any single named bucket after phase-2.

---

## 7. Ablation Interpretation

**Verified with:**
```
for tag in bd340_reverted_cache_q4 bd341_accepted_recovery_q4 bd342_radial_factory_timers_q4; do
  python3 -c "import json,glob;s=json.load(open(glob.glob('artifacts/$tag/*component_wall_summary.json')[0]));c=s['component_wall_attribution'];print('$tag',round(c['total_wall_seconds'],1),round(c['components']['phase2_corrector']['wall_seconds'],1),round(c['components']['payload']['wall_seconds'],1))"; done
# bd340 602.0 344.6 135.6  | bd341 428.0 184.1 129.9 | bd342 428.8 181.4 131.8
```

| Run | Total (s) | phase-2 (s) | payload (s) | Read |
|---|---:|---:|---:|---|
| BD340 (no accepted-recovery; cache-bound) | 602.0 | 344.6 | 135.6 | confounded negative control; ~45 activation rejections |
| BD341 (accepted-recovery) | 428.0 | 184.1 | 129.9 | recovered baseline; 1 rejection |
| BD342 (+ factory subtimers) | 428.8 | 181.4 | 131.8 | preferred current evidence |

**Key inference:** the BD340→BD341 delta is **−174 s total**, of which **−160.5 s is phase-2** and only −5.7 s is payload. Rejections do not spread evenly across the run — they concentrate in phase-2 (each rejected host step replays the conservative corrector). This is the empirical anchor for the causal chain *deficient main-step Jacobian → step rejections → phase-2/rejection wall*. It is **suggestive, not proof**, because BD340 confounds accepted-recovery with cache-bounding; the clean test is an accepted-recovery-only ablation (PR-D / E2 design).

**What the ablations already falsify:**
- Dense-LU-first (0.45 s) — dead.
- Phase-2-Newton-Jacobian-reuse-first — named Newton algebra is 17 s; reusing it cannot move the 181 s bucket.
- "Just add telemetry" — telemetry is now adequate at the exclusive level; the open questions are physics/orchestration, not visibility.

**What remains genuinely open after ablation:** whether phase-2 (and the rejections that feed it) are *necessary physics* or *compensation* — settle via E1 before E2; whether provider-build is cacheable (E3); the 191 s of unattributed pockets (E5).

---

## 8. Top 3 Next Experiments (Best-of-N)

**Candidate pool (6):** E1 FD source-response reference; E2 phase-2 ablate; E3 payload provider cache-boundary; E4 same-RHS solver bakeoff; E5 artifact/runtime separation; E6 LRS/non-LRS/q-grid ladder.

**Rejected:**
- **E4 (solver bakeoff)** — rejected as a *now* experiment. Forbidden by boundary before source-response semantics are fixed; outer solve is 0.45 s; and it is not even runnable (two divergent Rodas5P steppers, §5.3). It cannot falsify the current blocker.
- **E6 (q9/q10/high-q ladder)** — rejected. q4/q5 evidence is sufficient to test source-response semantics; high-q needs explicit approval and would not falsify the diagonal-Jacobian hypothesis any better than q4/q5. (Marked **optional**, justified only if E1 at q4/q5 shows the missing response norm grows with q.)

**Top 3 (each with precise support/falsify outcomes):**

**E1 — Finite-difference dynamic source-response reference (q4/q5 fixed states).**
Compute FD response of `dQ_nue_pair_N`, `dQ_nux_bank_N`, `dA_collision` (and represented angular/high-q source moments) at fixed states; compare the captured fraction of the FD response L2 norm by: frozen JVP, standard-3T add-on, diagonal `dA/A`, and a candidate structured operator. Correlate residual norm with rejection/phase-2 windows.
- *Supports `SOURCE_RESPONSE_FIRST` if:* the diagonal/frozen policies capture < ~0.7 of FD norm **and** the missing norm sits in directions (growth, off-diagonal, anisotropic) that line up with rejection-heavy windows.
- *Falsifies it if:* captured fraction ≥ ~0.9 (especially if ≥0.9 of the norm is in `A*dA<0` diagonal directions) → the Jacobian is adequate; reorder to E2/E3.
- *Prerequisite:* recover/inspect `dynamic_collision_runtime` (§11) and expose the reference via PR-0; current path is `nonlrs_s2`-only with policy default `"none"`.

**E3 — Payload provider cache-boundary probe.**
Split provider/factory construction into state-independent scaffold vs state-dependent quadrature; instrument both; add a pure memoization cache with ON/OFF **output-equality** check (tight numerical equality of `dQ`/`dA`) at fixed state.
- *Supports a 2–3 PR payload win if:* a large fraction of the 81 s provider-build is state-independent and ON/OFF equality holds bitwise/tight.
- *Falsifies cacheability if:* equality fails OR the cost is genuinely temperature-dependent quadrature (cache key must include live thermo state) → payload caching drops down the queue.

**E2 — Phase-2 corrector ablation (run AFTER E1).**
Bounded q4/q5 across: current / no-phase-2 / tighter Newton+conservation tol / phase-2-only-on-accepted-activation / no-AB2-predictor / no-residual-guard / full-step-only vs adaptive-pair. Metrics: wall, rejects, raw + corrected abundance deltas, conservation residual, finite-state, endpoint row class, component wall.
- *Supports phase-2 as compensation if:* a stronger main-step source-response (from E1) shrinks phase-2 need without breaking conservation.
- *Supports phase-2 as necessary physics if:* removing/tightening it degrades conservation or abundance fidelity at any wall.
- *Bonus falsifier for the §8/E1 priority:* if stripping bookkeeping+AB2 (with output parity preserved) cuts phase-2 181→<80 s at unchanged rejection count, then phase-2 wall was orchestration, and its perf weight is independent of the Jacobian question.

---

## 9. Proposed PR Sequence (≤6)

Ordering respects the boundary: diagnostic/opt-in experiments may precede PR-B; **no optimization goes default-on before PR-B parity + cold floor.**

| PR | Title | Content | Proof / Falsification | Scale |
|---|---|---|---|---|
| **PR-0** | Bounded runtime/source-response seam | Expose `JacobianProvider` + `CollisionPayloadProvider.source_response_operator` + a public `dynamic_collision_source_response_reference` callable for q4 LRS-projected **and** non-LRS-S2 configs; stop the ladder importing the 22 private RHS helpers for source-response/step control. No new gates. | Ladder runs E1 via public interface; private-import count for source-response → 0 | 2–3 PRs of work, scoped as one seam |
| **PR-A** | E1 FD source-response reference | Wire FD reference; emit captured-norm comparison (frozen/3T-addon/diagonal/structured) + per-window rejection correlation as a compact artifact | Captured fraction threshold (§8) decides whether source-response or phase-2/payload leads next | 1–2 PRs |
| **PR-B** | Cold parity + floor (boundary-mandated gate) | Cold-endpoint LRS/non-LRS parity **and** `N_eff_3T >= 3.0` cold floor under the **actual** collision projection (not `flrw_monopole_only`); preserve raw negatives | If floor fails → physics-correctness preempts all optimization; default-on stays forbidden | 2–5 PRs + expensive runs |
| **PR-C** | E3 provider cache boundary | Scaffold/quadrature split + ON/OFF output-equality cache | Decides 2–3 PR payload win vs unavoidable thermo quadrature | 2–3 PRs |
| **PR-D** | E2 phase-2 ablation | Controlled phase-2 policy matrix (after PR-A) | Classifies phase-2 as necessary physics vs compensator | 2–4 PRs |
| **PR-E** | Artifact/runtime separation | Move JSON expansion out of hot loop; keep compact counters; attribute the 100 s residual + 41.5 s payload gap + 50 s in-Newton gap | Shows whether the 191 s of pockets are diagnostic tax vs structural | 1–3 PRs |

PR-A, PR-C, and PR-B's expensive runs can proceed in parallel once PR-0 lands; PR-D is gated on PR-A.

---

## 10. Conditions Under Which a Solver/Library Rewrite Becomes Justified

A solver-family change (E4 → adopt BDF/Radau/CVODE/W-method) is justified **only if all hold:**
1. PR-0 has unified the step interface so the *same* RHS + *same* Jacobian/linear operator can be fed to each solver (today this is impossible — two Rodas5P steppers).
2. PR-A has either fixed or ruled out the source-response gap (a bakeoff over a deficient Jacobian measures the wrong thing).
3. PR-E has removed the orchestration/telemetry tax, so the residual reflects numerics, not bookkeeping.
4. With (1)–(3) done, a same-math bakeoff shows an external stiff solver winning decisively on accepted/rejected steps, wall, and conservation — i.e. a **stable residual compute kernel above ~50–70% wall after parity** that Rodas5P/host-stepping demonstrably loses on.

A **library/platform** rewrite (e.g. toward LINX-style JAX kernels) is justified **only additionally if:** the FD reference shows the true source response needs dense off-diagonal coupling across q×species×mode that the in-tree JAX low-rank/Woodbury/Schur seams cannot exploit, **and** differentiability/throughput (not correctness) is the binding constraint after parity. Absent these, the CRAG external-code ledger correctly grades a rewrite UNSUPPORTED.

A pre-emptive falsifier for §1: if a same-RHS external BDF/Radau/CVODE run beats Rodas5P on wall/rejections **before** any source-response change, then `SOLVER_BAKEOFF_FIRST` should move ahead of `SOURCE_RESPONSE_FIRST`.

---

## 11. Missing Files / Evidence

**Critical:**
- **`rabbit.validation.dynamic_collision_runtime` is absent from the snapshot.** ap65_rhs.py imports it as `collision_runtime` (line 22) and depends on it for `dynamic_collision_source_response_reference_np` (the FD reference E1 hinges on), `build_nonlrs_dynamic_collision_payload_from_state_np`, `build_dynamic_collision_payload_from_restart_kwargs`, `source_refresh_config_for_trace_role`, `BOUNDARY_TRACE_ROLES`. **The 81 s provider-build implementation and the FD source-response reference are therefore both un-inspectable in this packet.** This is the largest evidence gap.
- Other unbundled imports referenced by inspected code: `rabbit.thermo.rate_prefactors` (`gamma_nu_nu_over_H`), `rabbit.validation.truncation_guards` (`enforce_positive_typeI_omega`).

**Important:**
- No executed FD-vs-policy comparison artifact (audit policy default `"none"`); no JSON quantifying captured response fraction.
- Cold-endpoint evidence exists **only** under `flrw_monopole_only` (q14/q15 BD220); no cold non-LRS-S2 collision-projection floor; no LRS/non-LRS parity artifact (`controlled_flrw_lrs_nonlrs_parity = None`).
- JAX compile vs runtime not separated (explicit unavailable reason) — host-JVP vs JAX-compile cannot be distinguished.
- Full test tree excluded (focused tests only) — cannot judge whether tests lock stable physics contracts or private-helper behavior.
- Three unattributed wall pockets (100 s residual, 41.5 s payload-not-in-factory, ~50 s in-Newton) have no sub-timer evidence.
- Absolute cold `N_eff_3T`/`Yp` values for q14/q15 not surfaced in the curated fields (only resolution deltas, which are ~1e-13 / ~1e-7).

---

## 12. Red-Team Objections + What Would Falsify This Conclusion

**O1 (performance critic):** Phase-2 wall is 58% orchestration and only 9.4% Newton linear algebra, so the *wall* problem is orchestration, not Jacobian quality. E1 is a physics-honesty errand, not a perf win; PR-E/E2 would cut more wall faster.
→ *Concession:* partly correct on **wall**. SOURCE_RESPONSE_FIRST is justified on **decision value and physics honesty**, plus the rejection→phase-2 link, not on being the biggest single wall cut. *Falsifier:* E2 with stripped bookkeeping/AB2 (output parity held) cutting phase-2 to <80 s at unchanged rejections → phase-2 was overhead; source-response loses its wall rationale (physics rationale survives).

**O2 (rejection-controller objection):** Accepted-recovery already shows rejections drive phase-2; a cheaper controller fix might dissolve phase-2 cost with no Jacobian work.
→ *Falsifier:* an accepted-recovery-only ablation (no cache confound) that holds rejections at 1 across activation and collapses phase-2 → the "weak Jacobian → rejections" chain weakens, and PHASE2/controller work precedes source-response.

**O3 (physics critic):** The diagonal-damping Jacobian being blind to growth modes may be harmless if the true q4 response is dominated by `A*dA<0` damping directions.
→ *Falsifier:* E1 shows ≥0.9 of FD response norm in diagonal `A*dA<0` directions → diagonal Jacobian is adequate; verdict flips to PHASE2/payload first.

**O4 (architecture critic):** The 22 private imports may be a stable de-facto API; a typed extraction is churn for no runtime gain.
→ *Falsifier:* a 1–2 PR source-response bakeoff implemented **without** PR-0 extraction (the falsifier the packet's own architecture ledger names) → extraction unnecessary; demote PR-0.

**O5 (radical critic, the other direction):** Maybe the whole AP65 staging is wrong and a PRIMAT-style staged thermo+network with a standard stiff network solve would dissolve phase-2 entirely.
→ *Concession:* plausible long-run; but unprovable now and forbidden as a first move (no parity, no same-RHS bakeoff, missing runtime module). *Falsifier for delaying it:* after PR-A/PR-C/PR-E, if the residual numerical core is small and phase-2 is necessary physics that maps cleanly onto a standard network integrator, a staged-reformulation PR becomes the rational next step over more AP65 tuning.

**What would overturn the entire `SOURCE_RESPONSE_FIRST` verdict (any one):**
- E1 captured-fraction ≥ ~0.9 → semantics already adequate (→ PHASE2 or PAYLOAD first).
- Same-RHS external solver beats Rodas5P before any source-response change → `SOLVER_BAKEOFF_FIRST`.
- Provider-build proven unavoidable thermodynamic quadrature → payload caching demoted, but does not by itself promote any other candidate above source-response.
- Cold PR-B floor fails under the real projection → physics correctness preempts all optimization; the verdict becomes "fix endpoint stability first," optimization stays forbidden.

---

*End of BD343 external algorithmic architecture audit. This evaluates algorithmic/architectural direction only; it validates neither final abundances nor public/solver readiness, consistent with the packet's stated boundaries.*
