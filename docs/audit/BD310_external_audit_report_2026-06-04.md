# BD310 External Audit Report — RABBIT augmented Type-I no-QKE AP65 solver

Date: 2026-06-04
Auditor scope: current AP65 q4 blocker after BD303 and PR-C0 … PR-D3.
Method: CRAG + Chain-of-Code. Every claim below was graded against raw JSON
artifacts and source code in the packet, not against README/doc prose. Where the
documents and the raw evidence disagree, the raw evidence wins and the disagreement
is stated explicitly.

Boundaries honored: QKE out of scope; no publication/public-production claim; no
whole-language rewrite recommended; CPU-JAX + in-tree Rodas5P/AP65 retained; no
optimization default-on before PR-B parity and `N_eff_3T >= 3.0`; raw
negative/nonfinite evidence preserved; no standalone readiness/manifest/hash/figure
gate recommended.

---

## 1. Executive verdict

**`PR_B_FIRST`** — for the *default-on decision*. The LRS/non-LRS FLRW-limit
`N_eff_3T` parity is the binding constraint, it is unresolved, and no performance
PR may be promoted past it. This is not a default of convenience: the q4 evidence
independently shows the two attempted optimization families (PR-C Jacobian reuse,
PR-D payload reuse) cannot be *correctly evaluated* yet, so doing them first would
be optimizing a path whose physics is still in question, on telemetry that cannot
score them.

**Secondary verdict, performance axis: `TELEMETRY_STILL_INCOMPLETE` for phase-2,
`PR_D4`-leaning for payload.** Two distinct sub-findings:

- The phase-2 corrector bucket is **90.7–91.0 % un-subtimed beyond Newton** in
  every post-C0 run. PR-C0 surfaced the *top-level* partition but did **not**
  instrument the dominant 244–283 s of phase-2 work. PR-C4 cannot be aimed until
  that is split. So for phase-2 specifically, telemetry is still incomplete.
- Payload attribution *is* clean (285 s, the direct build timer). The PR-D2 failure
  is not a telemetry failure and not proof the mechanism is impossible: the reuse
  **tolerance is misconfigured** (code default `rtol = atol = 0.0`; the BD309 run
  used an effective scale of ~2.4e-7 against a background moving ~0.4 %). That is a
  one-line experiment away from a real answer, then a `PR_D4` sub-block split.

The single highest-leverage *correct* next action is PR-B. The single
highest-leverage *performance* lever, once physics clears, is collision-payload
build cost (42 % exclusive wall), attacked by (a) fixing the reuse tolerance and
(b) sub-block reuse — not by more Jacobian caching.

---

## 2. Claim ledger

Grades: SUPPORTED / PARTIAL / CONTRADICTED / UNTESTED / FORBIDDEN, each with the
raw basis.

| # | Claim (from packet docs) | Grade | Raw basis |
|---|---|---|---|
| C1 | CPU-JAX/Rodas5P/AP65 remains the target | **SUPPORTED** | `W_shape = J_shape = jacobian_shape = linear_system_matrix_shape = [62,62]` in all 4 span rows (BD304). Host JVP/Jacobian wall = 15.1 s (≈2 %). No residual kernel >50–70 % survives the corrected partition. |
| C2 | QKE out of scope | **FORBIDDEN boundary** | Honored. No QKE path touched. |
| C3 | q4 bounded replay validates full BBN | **FORBIDDEN / false** | `passed=false`, `violations=['resolution_ladder_failed_or_nonendpoint_rows']`; span 3 ends `T_final_MeV = 0.0699`, target `<=0.01`. Correctly labeled in the artifact. |
| C4 | PR-C0 moved a real blocker | **PARTIAL** | True at top level: payload/phase2/host/residual are now populated and form an exact exclusive partition (sum = total to machine precision). **But** 90.7–91.0 % of the phase-2 bucket remains un-subtimed, so the move is partial, not complete. |
| C5 | PR-C1/PR-C3 phase-2 reuse speeds q4 | **CONTRADICTED** | Total 678.3 → 703.3 → 725.6 s. PR-C3 cut `jac_eval` 15603→9704 but raised `newton_iters` 15603→17178 (+10.1 %) and total +7.0 %. Measured Jacobian-assembly saving = `jac_wall` 7.66→4.90 s (−2.76 s) against +47 s total. |
| C6 | PR-D0/PR-D2 payload reuse speeds q4 | **CONTRADICTED** | Payload wall stayed 283.8–284.5 s; reuse fired 14/4585. |
| C7 | …because payload reuse is hard/impossible | **CONTRADICTED (re-diagnosed)** | BD309: `max_abs_delta_max = 0.00396`, `max_scaled_delta_max = 16207.8` ⇒ effective scale ≈ 2.4e-7. Code default `_STAGE_COLLISION_PAYLOAD_REUSE_STATE_DEFAULT_RTOL = 0.0`, `…_ATOL = 0.0`. The criterion is ~16000× too tight; the background is only ~0.4 % different. The reuse *count* is an artifact of tolerance, not of physics. |
| C8 | PR-B parity/floor is the default-on blocker | **SUPPORTED** | Unresolved LRS/non-LRS `N_eff_3T` split; the floor is physically motivated (see §5). |
| C9 | Structured/low-rank solve is the current q4 target | **CONTRADICTED for q4** | `[62,62]` everywhere; revisit only with high-q evidence. |
| C10 | Dense LU is the q4 bottleneck | **CONTRADICTED** | Host Jacobian/JVP wall = 15 s of 678 s. |
| C11 | Component attribution is "real enough" | **PARTIAL** | Real for payload/host; the prose's 5-way additive framing (payload+phase2+rejected+host+residual) is **misleading** — those five sum to 706–778 s, exceeding the 678–726 s total by 28–53 s, because rejected-replay overlaps payload and phase2 (the summarizer itself excludes it from residual arithmetic). |
| C12 | Raw negative/nonfinite evidence preserved | **SUPPORTED** | `selected_phase2_..._raw_newton_trial_negative_count_total`, `..._raw_candidate_negative_*`, distribution `eps_f` floor preserves rather than clips; summarizer raises on negative residual rather than clamping. |
| C13 | Heavy-bank `dQ_nux_bank_N / (2·dρ_pair/dT)` convention is correct | **SUPPORTED** | `nudec_coupled.py`: `d2 = 2.0*_drho_nu_pair_dT(T_nu_x)`; bank energy = 2 pairs ⇒ dρ_bank/dT = 2·dρ_pair/dT; degeneracy applied exactly once. Local test confirms analytic `dT_νx/dN = −0.75·T_νx`. |
| C14 | `N_eff_3T` is a valid floor tripwire | **PARTIAL → conditional** | Definition is correct and = 3 at equal T. But it is only meaningful at the **cold endpoint**; per-span values are 9.20/5.03/3.55/3.19 (still in the e± annihilation tail at 0.07 MeV). Floor must be checked at `T_γ <= 0.01 MeV`, not on hot rows. |

---

## 3. Raw artifact forensics and q4 ablation interpretation

### 3.1 Corrected component-wall table (exclusive partition)

Extracted directly from `rows[0]` `selected_*` fields. The exclusive partition is
**payload + phase2_corrector + host_jacobian + residual = total** (verified to
machine precision). Rejected-replay is a *cross-cutting overlapping slice* that
lives inside payload + phase2 + residual and is therefore listed separately, not
added.

| case | condition | total s | payload s | phase2 s | host s | residual s | phase2 non-Newton | rejected (overlap) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BD304 | PR-C0 baseline | 678.31 | 284.97 | 269.24 | 15.12 | 108.99 | 244.56 (90.8 %) | 136.49 (20.1 %) |
| BD305 | PR-C1 periodic reuse | 703.26 | 285.61 | 294.43 | 15.12 | 108.10 | 267.38 (90.8 %) | 147.09 (20.9 %) |
| BD307 | PR-C3 cross-solve reuse | 725.59 | 290.31 | 310.58 | 15.11 | 109.59 | 282.69 (91.0 %) | 162.11 (22.3 %) |
| BD308 | PR-D0 full-state reuse | 708.14 | 283.83 | 301.98 | 15.02 | 107.31 | 274.03 (90.7 %) | 150.81 (21.3 %) |
| BD309 | PR-D2 thermo reuse | 712.68 | 284.50 | 306.16 | 15.03 | 106.98 | 277.83 (90.7 %) | 152.93 (21.5 %) |

Counters (BD304): `payload_builds = 5210` = `source_evaluations = 5210`;
`stage_source_evaluations = 4585`; `step_count = 614`; `newton_iters = 15603`;
`newton_jac_eval = 15603`; `linear_factorizations = 655`; `linear_solves = 5240`;
`max_rss ≈ 3.79 GB` (flat across BD304–BD309).

### 3.2 What the corrected partition changes

The docs present payload (284) **and** phase2 (269) **and** rejected (136) **and**
host (15) **and** residual (108) as if they were a five-way budget. They are not:
their sum (706 s for BD304) exceeds the 678 s total. The summarizer is actually
correct (`residual_basis = "payload + phase2_corrector + host_jacobian +
jax_compile + jax_runtime; reported_components surfaces overlapping/replay subtimers
but excludes them from residual arithmetic"`), but the human-facing prose mis-states
it. The corrected reading is:

- **Payload build ≈ 42 % (285 s)** — biggest single exclusive cost.
- **Phase-2 corrector ≈ 40 % (269 s)**, of which **only ~9 % is measured** (Newton
  total 24.7 s; the three measured sub-walls jacobian 7.66 + residual 3.15 +
  linear_solve 0.58 = 11.4 s). **~244 s (36 % of the whole run) is un-attributed
  inside phase-2.**
- **Host Jacobian ≈ 2 % (15 s)**.
- **Residual ≈ 16 % (109 s)** — non-payload RHS, source→mode conversion, EOS,
  telemetry, dispatch.
- **Rejected-step replay ≈ 20 % (136 s)** is a *slice across* the above, not a
  fifth bucket; reducing rejected steps reduces payload/phase2/residual together.

### 3.3 PR-C ablation (Jacobian reuse) — why it failed

PR-C3 reused 7474 Jacobians (`jac_eval` 15603→9704) and saved 2.76 s of Jacobian
assembly. It simultaneously raised Newton iterations by 1575 (+10.1 %), payload
builds by 74, and phase-2 wall by 41 s. **Verdict: wrong target, and a convergence
perturbation.** Stale Jacobians lengthened Newton, which lengthened replay. You
cannot win 2.76 s of a 678 s run by spending iteration count. This is not "reuse
cadence needs tuning"; it is "the Jacobian/linear-algebra subpath is 9 % of the
bucket, so no caching strategy on it can matter."

### 3.4 PR-D ablation (payload reuse) — why it barely fired

BD309 thermo-background scope: state vector = `[N, y[0:5]]` (e-fold time + 5
background scalars). Decision: reuse iff `max_scaled_delta <= 1`, with
`scaled_delta = |Δ| / (atol + rtol·max(|base|,|stage|))`. Observed
`max_abs_delta = 0.00396`, `max_scaled_delta = 16207.8` ⇒ effective scale ≈ 2.4e-7.
The module defaults are literally `rtol = atol = 0.0` (which mathematically forces
reuse only on bit-identical states). **Verdict: misconfigured tolerance, not a
physics wall.** At a physically reasonable thermo tolerance (~1e-2), the scaled
delta would be ~0.4 and reuse would fire on most stages. The honest open question
is whether reuse at that tolerance preserves the collision **source budget** —
which is exactly the controlled experiment to run.

---

## 4. Physics consistency critique

**Three-temperature no-QKE closure: internally consistent for the stated private
diagnostic scope.** `nudec_coupled.py` carries one ν_e pair + a two-pair ν_x bank
at a single T_νx; energy gains enter as `dT_i/dN = −T_i + dQ_i/(dρ_i/dT)` with
`dρ_νx_bank/dT = 2·dρ_pair/dT`. Degeneracy is applied once. The plasma equation
`dT_γ/dN = dT_base − dQ_total/(dρ_em/dT)` with `dT_base` from entropy conservation
(`S(T_γ)` route at μ=0) is standard; total energy plasma+ν is conserved by
construction since `dQ_total` is the same transferred quantity with opposite sign.
`coupled_3T_rhs_from_collision_moments` preserves this when sourced by the bridge.
The ν-ν equilibration source is energy-neutral by construction
(`dQ_nux_bank = −dQ_nue_pair`, `T_common⁴ = (T_νe⁴ + 2T_νx⁴)/3`) and vanishes at
equal T (test-confirmed). The distribution layer is exact:
`f = sigmoid(−(q+A))`, `df/dA = −f(1−f)`, `dA/dN = −(df/dN)/max(f(1−f), eps_f)`,
with an overflow-safe Fermi-Dirac. Net: the closure is sound.

**`N_eff_3T` proxy and floor: correct definition, conditional validity.**
`N_eff = (T_νe/T_std)⁴ + 2(T_νx/T_std)⁴`, `T_std = T_γ(4/11)^{1/3}`; = 3 at equal
T. The floor `>= 3.0` is **physically motivated**, not arbitrary: in the zero-shear
FLRW limit, standard non-instantaneous decoupling gives `N_eff ≈ 3.044 > 3`, and a
value *below* 3 means neutrinos ended colder than the instantaneous-decoupling
reference — i.e. the neutrino sector lost energy it should have kept. The proxy
denominator `T_std` is path-independent (it depends only on `T_γ`), so a
path-dependent `N_eff_3T` split can only come from path-dependent `T_νe, T_νx`, i.e.
from the **collision-source energy actually deposited into neutrinos** — not from
the proxy. Therefore the `2.994` (non-LRS) vs `3.105–3.115` (LRS) split is a
**path/parity bug, not a proxy-definition failure.** Caveat: `N_eff_3T` is only
meaningful at the cold endpoint; the q4 per-span values (9.20/5.03/3.55/3.19) are
mid-annihilation transients and must not be floor-checked.

**Most probable mechanism of the parity split (concrete, falsifiable).** The LRS
and non-LRS paths use *different* angular machinery. The non-LRS path runs a
**fixed-S2 projection** (`NonLRSS2Grid`) that explicitly *measures discarded angular
source power* (`nonlrs_fixed_s2_projection_residual_l2`,
`…_residual_relative_l2`, "angular source power discarded by the retained fixed S2
modes"), whereas the LRS path uses a staged route. If the retained S2 modes do not
carry the full energy moment when higher angular modes are truncated, the non-LRS
path **loses neutrino energy** → colder ν → `N_eff_3T < 3`. The bridge already has
energy-moment-preserving switches for the nu-nu rows
(`conserve_offdiagonal_nunu_pair_energy_moments`, `conserve_identical_nunu_moments`);
the same enforcement appears **not** to be applied to the non-LRS fixed-S2
projection. This is directly testable (§6, PR-B).

**One tail-conditioning artifact to log (not a blocker).** The `eps_f = 1e-12` floor
in `dA/dN = −(df/dN)/max(f(1−f), eps_f)` caps the high-q mode response where
`f(1−f) ≪ 1e-12`. This biases high-q tail dynamics by an arbitrary floor. It does
not enter `N_eff_3T` (which is temperature-based), but it can affect augmented-mode
energy bookkeeping; keep the raw nonfinite/floor-hit telemetry on so the bias is
auditable.

---

## 5. Numerical algorithm critique

- **Dense LU correctly de-prioritized at q4.** `[62,62]` in all spans; host wall
  15 s. Structured/low-rank solve should re-enter the top-three plan only when q9/q10
  evidence (not yet permitted to run for this decision) shows W/J growing enough
  that factorization wall is a measured top-three cost. Trigger condition to write
  down: host-Jacobian + factorization wall ≳ 25 % of total at the tested q.
- **Phase-2 corrector is the real numerical opacity.** ~244 s (36 % of run) inside
  the corrector is un-subtimed. From the source, each corrector step runs an
  adaptive embedded pair (`_phase2_backward_euler_network_adaptive_pair`,
  `_phase2_bdf2_newton_network_adaptive_pair`), a conservative-extent attempt, and a
  full-network window-reference (`_phase2_full_network_window_reference_*`), each of
  which assembles the nuclear-network kinetics payload
  (`_phase2_standard_network_kinetics_payload`) and fluxes **outside** the timed
  Newton sub-walls. The 614-step, zero-substep-retry window-reference counters
  (`…window_reference_step_count_total = 614`, `…adaptive_substep_retry = 0`) rule
  out "thousands of fine substeps"; the cost is per-step kinetics/flux assembly and
  embedded-pair orchestration repeated O(few) times per corrector step. **This must
  be subtimed before any phase-2 redesign.**
- **Rejected-step replay (20–22 %) is coupled to phase-2, not independent.** It
  grows monotonically with the phase-2 wall across the ablations (136→147→162 s as
  phase-2 269→294→310 s). It is most consistent with activation-row stiffness driving
  step rejection, where each rejected attempt re-runs the corrector. It is therefore
  unlikely to be a pure controller-knob win in isolation; it should move when phase-2
  per-step cost or the activation-window step policy moves.
- **No rewrite justification.** After correcting the partition, the largest stable
  exclusive kernel is payload build at 42 %; nothing sits above 50–70 % residual.
  **Reject whole-language rewrite explicitly.**
- **JAX compile/runtime split** is still unavailable as a clean real split and
  remains folded into "residual"; this limits cross-ablation precision but is not the
  current decision driver.

---

## 6. Numerical method — the controlled experiments that actually decide

### PR-B minimal controlled LRS/non-LRS pair (run this first)

Hold **everything** identical and vary only the LRS flag with shear set to exactly
zero: same q-grid (q4), same angular grid/label, same
`collision_source_composition_policy` (the q4 `weak_rate_corrections` +
`neutrino_collision_terms` set), same `h_max`, same `chain_restart_handoff`, same
`initial_np_policy` / `initial_A_monopole_offset`, same `electron_chemical_potential`
policy and `qed_correction_model`, **run to the cold endpoint `T_γ <= 0.01 MeV`**
(not the 0.07 MeV bounded replay). Record per row: `N_eff_3T`, `T_nu_e`, `T_nu_x`,
the per-species neutrino energy budget, and
`nonlrs_fixed_s2_projection_residual_relative_l2`.

Decision rule:
- If `N_eff_3T(non-LRS) − N_eff_3T(LRS)` tracks the time-integrated discarded
  angular energy ⇒ **fixed-S2 projection truncation**; fix = extend the existing
  `conserve_*_energy_moments` enforcement to the non-LRS fixed-S2 projection so the
  retained modes carry the full energy moment.
- If the discarded residual is ~0 yet they still split ⇒ the bug is in the EOS
  derivative path, initialization, or handoff; bisect those next.
- Either way, the floor `N_eff_3T >= 3.0` is only evaluated at the cold endpoint.

---

## 7. Code architecture critique

The recorded git hotspots (`augmented_continuous_ap65_rhs.py`, the span-ladder,
their test files, generated capability docs) and the file sizes confirm the debt:
`augmented_continuous_ap65_rhs.py` ≈ 23k lines / 944 KB and
`augmented_continuous_ap65_full_bbn_span_ladder.py` ≈ 16k lines mix RHS physics,
phase-2 corrector, telemetry, payload-reuse policy, and artifact I/O in one unit.
This is what makes the phase-2 opacity possible: the corrector's kinetics/flux work
is not isolated enough to time.

Smallest risk-reducing extractions, each tied to a runtime/physics blocker (not
cosmetic):

1. **Phase-2 corrector → its own module with exclusive subtimers** (kinetics-payload
   assembly, flux build, embedded-pair, window-reference, Newton). This rides with
   PR-C4 and is the prerequisite for it. Highest value.
2. **Collision-payload build → sub-block boundaries** (T-independent geometry /
   q-grid kinematics vs T-dependent occupation/rate factors), with a per-sub-block
   timer. This is the PR-D4 prerequisite and is where the 42 % lives.
3. **Span-ladder artifact I/O → separated from the run orchestration**, so the
   physics path can be reasoned about without the JSONL/JSON writer. Do this only
   alongside (1) or (2), never as a standalone refactor with no preserved q4 evidence.

Do not split for tidiness; split exactly the two surfaces (phase-2, payload) whose
opacity is currently blocking the performance decision.

---

## 8. Test-quality critique

Mixed, leaning toward schema/policy/telemetry locks plus self-referential golden
values, with a thin but real physics-invariant layer.

- **Genuinely protective (keep):** `test_three_temperature_closure_invariants.py`
  checks the analytic `dT_νx/dN = −0.75·T_νx`, directional plasma cooling under ν
  heating, ν_e untouched, and ν-ν vanishing at equal T — real physics, not schema.
  The distribution conversion tests and any `assert_allclose` math-identity tests are
  similarly protective.
- **Schema/policy/count locks (low protection):** in
  `test_augmented_continuous_ap65_rhs.py` (240 test fns) there are ~178 string-enum
  equalities (policy/scope/contract names), ~46 `wall_seconds`-field presence checks,
  and ~11 exact `_count ==` locks (e.g. `accepted_count == 1`). These break on benign
  refactors and protect no physics.
- **Self-referential golden values (false comfort risk):** `Yp == pytest.approx(...)`
  golden locks reproduce the code's *own prior output*; they guard against drift but
  certify nothing physical, since the generating run is itself unvalidated.
- **The decisive missing test:** there is no end-to-end **energy-conservation**
  invariant and no **LRS/non-LRS FLRW-limit parity** test. The closure test never
  integrates and never touches the non-LRS path, so it structurally cannot catch the
  `N_eff_3T` split. Add a cold-endpoint zero-shear parity test as part of PR-B.

---

## 9. Recommended PR plan (max 6, ordered)

1. **PR-B — LRS/non-LRS zero-shear parity to the cold endpoint.** The default-on
   gate. Includes the controlled pair of §6 and a new energy-conservation +
   parity regression test (§8). Likely fix: energy-moment-preserving non-LRS fixed-S2
   projection. No optimization promoted until `N_eff_3T(LRS) ≈ N_eff_3T(non-LRS) ≥ 3.0`
   at `T_γ <= 0.01 MeV`.
2. **PR-D4-prep — payload sub-block timers + reuse-tolerance experiment.** First
   correct the defaults (`rtol/atol` no longer 0.0) and sweep
   `rtol ∈ {1e-7,1e-5,1e-3,1e-2}` recording (reuse fraction, wall, **source-budget
   delta**, `N_eff_3T` delta). Simultaneously add a per-sub-block payload timer
   (geometry/kinematics vs occupation/rate). Opt-in only.
3. **PR-D4 — sub-block payload reuse.** Reuse the T-independent geometry/kinematics
   sub-blocks unconditionally; rebuild only the T-dependent rate/occupation factors;
   gate the latter on a *component-specific* tolerance with a source-budget parity
   assertion. This is where the 42 % is.
4. **PR-C4-prep — phase-2 exclusive subtimers** (kinetics payload, flux,
   embedded-pair, window-reference, Newton), extracted into a phase-2 module. Until
   this lands, PR-C4 is unaimable.
5. **PR-C4 — phase-2 redesign** *only if* (4) shows a reusable/avoidable dominant
   sub-cost; the design must prove **fewer Newton iterations and lower total wall**,
   not merely fewer Jacobian evaluations.
6. **PR-E — activation-window controller/`h_max` tuning** *after* (4)/(5), once
   rejected-replay wall is tied to a specific controller decision rather than to
   phase-2 stiffness.

Default-on condition (write into the gate): all of (i) PR-B parity at the cold
endpoint with `N_eff_3T ≥ 3.0` on both paths; (ii) source-budget parity preserved
under the optimization (per-component `dQ` within tolerance); (iii) raw
negative/nonfinite evidence still serialized; (iv) no regression in cold-endpoint
`Yp`/`D/H` beyond a stated physical tolerance.

---

## 10. Exact commands to run next

From the repository root (not inside the packet). Re-verify the corrected partition
and the reuse-tolerance diagnosis the audit is built on:

```bash
# 1. Re-derive the exclusive partition + phase-2 non-Newton fraction from raw JSON
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/check_component_wall_attribution.py \
  diagnostic_outputs/bd304_pr_c0_component_wall_q4_after_newton_patch/bd304_q4_activation_probe.json
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/check_component_wall_attribution.py \
  diagnostic_outputs/bd309_pr_d2_thermo_reuse_q4/bd309_q4_pr_d2_thermo_reuse.json

# 2. Confirm reuse-tolerance defaults and observed deltas
grep -n "_STAGE_COLLISION_PAYLOAD_REUSE_STATE_DEFAULT_RTOL\|_DEFAULT_ATOL" \
  src/rabbit/validation/augmented_continuous_ap65_rhs.py

# 3. PR-B controlled pair to the COLD endpoint (zero shear, identical everything,
#    LRS vs non-LRS). Use the span-ladder runner with stop_at_T_gamma_MeV<=0.01 and
#    matched q4/angular/source/h_max/handoff/init; record N_eff_3T, T_nu_e, T_nu_x,
#    per-species nu energy budget, nonlrs_fixed_s2_projection_residual_relative_l2.
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python \
  scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py --help   # confirm exact flags first

# 4. Focused tests that protect physics/telemetry (not count locks)
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_three_temperature_closure_invariants.py
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_augmented_pstf_distribution.py
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_summarize_perf_artifacts.py
```

(The packet `scripts/extract_q4_ablation_table.py --repo-root .` reproduces the
ablation table from final JSON; it agrees with the corrected partition above except
for the misleading 5-way prose framing, which lives in the docs, not the extractor.)

---

## 11. Red-team objections

- **"PR-B is a stall; the user wants speed."** PR-B is the only action whose value is
  independent of the broken perf telemetry. Optimizing now risks freezing source
  budgets on a path that already fails the zero-shear floor — i.e. making a wrong
  answer faster, the exact thing the packet's own guardrail forbids.
- **"You re-diagnosed PR-D2 as a tolerance bug — maybe a loose tolerance silently
  corrupts the source budget."** Possible, and that is *why* PR-D4-prep measures the
  source-budget delta per tolerance, and why PR-D4 reuses only T-independent
  sub-blocks. The audit does not claim loose whole-payload reuse is safe; it claims
  the *14/4585 count* proves nothing about feasibility because the tolerance was ~2e-7.
- **"The N_eff split could be the proxy, not physics."** Rejected on the math: `T_std`
  is path-independent, so a path-dependent `N_eff_3T` must come from path-dependent
  `T_νe/T_νx`. The proxy cannot manufacture a split.
- **"The phase-2 244 s might be irreducible real network work."** Maybe. PR-C4-prep
  exists precisely to find out before any redesign; if the 244 s is irreducible
  kinetics, PR-C4 is dropped and effort moves to payload + rejection. The audit does
  not assume phase-2 is reducible — it assumes it is currently *unmeasured*.
- **"Single unreplicated runs."** Correct, and the verdict respects that: the
  ablations are strong enough only to *reject* speedup claims (effects of +7 to +10 %
  exceed run-to-run noise of ~minutes on 11–12 min runs) and to *re-diagnose*
  mechanisms, not to rank micro-optimizations. The decisive PR-B comparison should be
  run ≥2× per arm.

---

## 12. Missing files / evidence

- **Cold-endpoint (`T_γ <= 0.01 MeV`) runs for both LRS and non-LRS** at matched
  settings — the only data that can resolve the `N_eff_3T` floor. The packet
  intentionally excludes q9/q10 and full-endpoint validations; for PR-B this exclusion
  must be lifted (for the LRS/non-LRS pair specifically, still at q4).
- **Per-sub-block payload timers** — none exist; needed to know what fraction of the
  285 s is T-independent and thus safely reusable.
- **Phase-2 exclusive subtimers** beyond Newton — none exist; ~244 s is dark.
- **Source-budget (`dQ_nue_pair`, `dQ_nux_bank`) parity series under reuse** — needed
  to certify any payload reuse as physics-safe.
- **A clean JAX compile vs runtime split** — still folded into residual.
- **The non-LRS angular-energy residual time series**
  (`nonlrs_fixed_s2_projection_residual_relative_l2` over the run) paired with the
  per-species ν energy budget — would directly confirm or kill the projection-loss
  hypothesis.

---

### Bottom line

PR-C0's partition is real at the top level and overturns the old "393 s dark
residual," but the human-facing prose over-claims it: the phase-2 bucket is ~91 %
un-instrumented and the five named components are not additive. Against the corrected
partition, PR-C (Jacobian reuse) was aimed at 9 % of phase-2 and correctly failed;
PR-D (payload reuse) was aimed at the right 42 % but throttled by a ~2e-7 tolerance
(code default 0.0), so its failure is a configuration artifact, not a physics verdict.
The binding constraint is unchanged and unresolved: the zero-shear LRS/non-LRS
`N_eff_3T` parity, most plausibly an energy-non-conserving non-LRS fixed-S2 angular
projection. Do **PR-B first**, add the two missing timer surfaces alongside the
payload/phase-2 PRs, and reject the rewrite.
