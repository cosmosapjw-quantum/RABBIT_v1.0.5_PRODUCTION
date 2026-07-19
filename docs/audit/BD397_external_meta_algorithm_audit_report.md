# BD397 External Meta-Algorithm Breakthrough Audit — Report

**Target:** RABBIT augmented Type-I PSTF no-QKE BBN solver
**Source snapshot:** HEAD `13ed46e`, branch `feature/v2-f5-closed-model-events`
**Date:** 2026-06-07
**Auditor mode:** External adversarial audit (optimization / physics / architecture only). Not a solver-validation, publication-readiness, or public-production claim.

---

## How this audit was conducted

- Read all required pre-reads (README, context-independent brief, physics/numerics brief, development history, bottleneck matrix, ablation matrix, auditor question list, prior reports BD345/346/357/373, remediation notes).
- Verified the structural claims directly against the source (line counts, function counts, policy-knob surface).
- Ran the three prescribed test commands. **All pass: 3 / 78 / 2.**
  - `test_three_temperature_closure_invariants.py` → 3 passed
  - `test_augmented_collision_bridge.py` → 78 passed
  - `test_j04_jax_rodas5p.py::woodbury + low-rank vs dense` → 2 passed
- Ran original physics experiments (standalone closure integration, proxy-vs-stop-temperature curve, clean-vs-artifact continuation) to localize the `N_eff_3T` discrepancy.
- Performed the mandated external/CRAG comparison against LINX, PRIMAT, PRyMordial, AlterBBN, NUDEC_BSM, and the precision N_eff literature.
- Worked in a private copy; the original repository is untouched.

Supporting runnable artifacts (delivered alongside this report):
- `scripts/neff_proxy_convergence.py` — proxy-vs-T_gamma curve (experiment E1)
- `patches/neff_asymptotic_extension.py` — asymptotic N_eff readout fix (patch P1, runs)
- `patches/staged_operator_split_prototype.py` — staged background/network skeleton (sketch B1)
- `FINDINGS_SUMMARY.json` — machine-readable summary

---

## 1. Executive verdict

**`PHASE2_STAGED_REWRITE` + `NEFF_CLOSURE_FIRST` + `DEFLATE_NOW`**, with **`PR_B_PARITY_FIRST` as the gating precondition** (not a competitor). I reject `CURRENT_ARCHITECTURE_ACCEPTABLE` and `PAYLOAD_FACTORY_FIRST`.

The entire wall-time problem lives in the activation window N ∈ [2.5, 3.0], where the fused AP65 + phase-2 path either takes ~325 s or hits the 420 s wall budget **without completing**, while the ~10 most recent PRs (BD389→396) optimized the *cheapest* window [4.5, 4.75] from 33.8 s to 17.3 s. The fused architecture is a self-inflicted wound: the 3T background (`hubble_3T`, `coupled_3T_rhs`) has **no dependence on nuclear abundances**, so the conventional staged background→network split used by LINX/PRIMAT/PRyMordial/AlterBBN is physically valid here, including in Bianchi-I. Separately, the `N_eff_3T ≈ 3.1149` "discrepancy" decomposes into two causes, one of which is a real ~0.7% calibration gap between the collision bridge and the calibrated energy-transfer table. Both the staged rewrite and the N_eff fix are blocked behind the still-missing PR-B parity plus a *meaningful* cold floor. The codebase (197.8k src LOC, a 25.8k-line RHS, ~12k LOC of publication machinery in a branch that forbids publication claims) is overgrown enough that deflation should run in parallel.

---

## 2. Claim ledger

Labels: IMPLEMENTED / VALIDATED / DERIVED / SPECIFIED / PROPOSED / SPECULATIVE / DEPRECATED / FORBIDDEN.

| Claim | Label | Evidence |
|---|---|---|
| Dense LU / linear solve is not a bottleneck at q4 | **VALIDATED** | `extract_packet_metrics.py`: linear wall 0.4–3 s, <0.1% of total in every run incl. 2808 s cold |
| Collision payload reuse is real (q4 payload wall fell) | **VALIDATED** | BD332 vs earlier baseline; payload ~130 s vs prior no-reuse; counters reconcile |
| Phase-2 corrector is the dominant wall bucket in full/cold runs | **VALIDATED** | 47–55% of cold-run wall (`bd345 cold_current` p2 = 1539/2809 s; `bd349` p2 = 1293/2725 s) |
| FLRW zero-shear invariant is restored after BD346 | **VALIDATED** | Σ_H ~1e-25 in fixed probes; projection guard code read directly (§5) |
| FLRW monopole projection cannot hide real Bianchi shear | **VALIDATED** | `_project_flrw_monopole_distribution_source_modes` raises unless \|Σ±\| ≤ tol and input anisotropic A-modes ≤ tol; logs removed L2 |
| 3T closure is calibrated to N_eff ≈ 3.034 | **VALIDATED** | Standalone integration → 3.0345; matches `nudec_tables._C_RATE` docstring and STATUS.md "ap_unified 3.0345" |
| `N_eff_3T ≈ 3.1149` is a diagnostic proxy, not physical N_eff | **DERIVED** (correct) | Already tagged `diagnostic_proxy_not_physical_N_eff` in artifacts; reproduced |
| Cold-row 3.32 / 3.70 / 5.23 / 9.58 values are endpoint truncation | **DERIVED** (mine) | `neff_proxy_convergence.py`: proxy monotone in T_gamma_stop; matches artifact rows at matched T_gamma |
| Residual ~0.08 N_eff excess is a bridge-vs-table calibration gap | **DERIVED** (mine, new) | Continuing clean state → 3.0345 vs artifact state → 3.1150; ν sector +0.665% hotter at matched T_gamma |
| Activation window N ∈ [2.5, 2.75] completes within budget | **FORBIDDEN / false** | BD379 `wall_time_budget_exceeded_count=1, completed_count=0` |
| Recent segment wins generalize to full endpoint | **SPECULATIVE** | All BD389–396 wins are on [4.5, 4.75]; no post-fix cold endpoint telemetry exists |
| Branch is publication / production ready | **FORBIDDEN** (correctly self-denied) | STATUS.md AP76/AP79 audit = `not_promoted`; scope rules forbid it |
| Staged operator-split solve is faster/correct here | **PROPOSED** | Physically justified (abundances don't enter background); not yet prototyped |
| `fix_plot.py` at root, multiple `figure_renderer_v2` | **DEPRECATED candidates** | root_snapshot + 12 publication modules |

---

## 3. Failure map

| What fails | Where | When | Why | What changes it |
|---|---|---|---|---|
| Activation-window non-completion | fused AP65 + `_phase2_*` in `augmented_continuous_ap65_rhs.py` | N ∈ [2.5, 2.75] cold/no-bakeoff | Stiff deuterium-bottleneck ignition inside a fused RHS that also carries thermo + shear + PSTF; Newton/retry/substep machinery microsteps | Staged network solve on frozen background (PR-1) |
| ~18–32% unattributed wall | `component_wall_attribution` in cold span rows | All long runs | Telemetry attributes only 68–82% of wall; the rest is Python dispatch/marshalling between host and JAX | Module extraction; pure-JAX small kernels (PR-5) |
| Post-fix cold endpoint has empty telemetry | `bd346_…/cold_current_after_bridge_fix/…summary.json` | After FLRW fix | Most physically relevant artifact records `attributed_wall_seconds_total=0.0, depth=none` | Re-run cold endpoint with telemetry (§11) |
| Residual N_eff excess (+0.08) | bridge moments → `coupled_3T_rhs_from_collision_moments` | All full runs | Collision bridge injects ~0.67% more e→ν energy than the calibrated `total_energy_transfer` table | Re-anchor bridge energy moment to table prefactor (PR-2) |
| Proxy inflation (3.3–9.6) | `N_eff_from_3T` read at span end | N ≤ 2.75 rows | e⁺e⁻ annihilation incomplete at T_gamma ~0.05 MeV; photon unheated | Asymptotic-tail readout (PR-3) |
| Vacuous cold floor | `N_eff_3T >= 3.0` tripwire | At current endpoints | Proxy is 3.3–9.6 there, so `>=3.0` is trivially passed | Evaluate at asymptotia as two-sided band 3.00–3.06 (PR-3) |

---

## 4. Component wall and endpoint table

Per-window cost (no-bakeoff cap1024), reconstructed from the `bd384`–`bd396` perf summaries:

| N window | wall (s) | phase-2 share | status | what it is |
|---|---|---|---|---|
| [2.5, 2.75] | 420.0 | ~93% | **budget exceeded, not complete** | activation / deuterium ignition |
| [2.75, 3.0] | 325.7 | ~85% | slow | activation tail |
| [3.0, 3.5] | 62.6 | low | ok | post-activation |
| [3.5, 4.0] | 58.6 | low | ok | post-activation |
| [4.0, 4.75] | 82.7 | low | ok | post-activation |
| [4.5, 4.75] | 33.8 → **17.3** | ~22% | ok (BD389→396 optimized) | cheapest tail window |

Full cold/q4 endpoints (span rows):

| Run | total wall (s) | phase-2 | payload | residual unattributed | host JVP | linear |
|---|---|---|---|---|---|---|
| `bd345` cold_current | 2808.9 | 1539.2 (55%) | 662.1 | 524.7 (19%) | 80.6 | 2.2 |
| `bd349` non-LRS split-pairwise cold | 2724.6 | 1293.2 (47%) | 422.6 | 885.3 (**32%**) | 120.4 | 3.1 |
| `bd348` non-LRS cold checkpoint | 1378.9 | 682.9 (50%) | 189.7 | 431.8 (31%) | 72.8 | 1.6 |
| q4 endpoints (`bd332`/`bd333`/`bd334`) | ~420 | ~170–180 (~42%) | ~130 (~31%) | ~95–100 (~24%) | ~15 | ~0.4 |

The effort/cost mismatch is the headline: optimization is concentrated on the one window that was already cheap, and on host JVP (~3.5% of cold wall).

---

## 5. Physics consistency audit

**Signs / units / energy conservation.** The 3T closure in `nudec_coupled.py` is internally consistent: one ν+ν̄ pair denominator `_drho_nu_pair_dT`, heavy bank carried as 2 pairs with `d2 = 2·dρ/dT`, energy-conserving `dQ_nux = -dQ_nue` in the equilibration source. The 78-test bridge suite and 3-test closure suite pass.

**FLRW limit — restored and safe (answer to physics Q4).** `_flrw_invariant_radial_projection_applicable` returns true only when `|Σ₊|, |Σ₋| ≤ tol` and `_source_modes_aniso_abs_max(A_modes) ≤ tol`; the projector raises `ValueError` otherwise and logs `removed_aniso_distribution_source_l2`. The projection therefore **cannot mask physical anisotropic stress** in Bianchi runs — at nonzero shear it is inapplicable and does not fire. It only removes spurious l ≥ 1 source modes arising from the discrete representation of an exactly isotropic distribution (the BD345 bug). One gap: there is no test proving a *small-but-real* shear (e.g. Σ ~ 1e-3) is preserved. Add a near-FLRW Bianchi case to the suite.

**N_eff (physics Q2/Q3) — the central result, decomposed.**

1. The standalone closure, integrated from T = 10 MeV to T_gamma ≪ m_e, gives **N_eff = 3.0345** (`neff_proxy_convergence.py`). This is exactly the correct no-QKE classical Boltzmann target. The literature value for Boltzmann energy transport ignoring oscillations is an increase in N_eff of ~0.034 (Mangano et al. 2002; Birrell et al. 2015; Grohs et al. 2016; Pitrou et al. 2018), versus the full-SM benchmark N_eff^SM = 3.0440 ± 0.0002 (Bennett et al. 2020/2021; de Salas & Pastor; Akita & Yamaguchi; Froustey et al.) with oscillations and finite-temperature QED.

2. The cold-row values 3.32 / 3.70 / 5.23 / 9.58 are **pure truncation**: the proxy is read at N ≤ 2.75 where T_gamma ≈ 0.05 MeV and e⁺e⁻ annihilation is incomplete, so the photon has not received its full (11/4)^(1/3) entropy boost. Reproduced against artifacts (artifact T_gamma = 0.0874 → 3.70; my integration at matched T_gamma → 3.67).

3. The residual gap between the *asymptotic* full-solver value (**3.115**) and the calibrated closure (**3.0345**) is **real and ~0.08**, caused by the collision bridge injecting **+0.665%** more e→ν energy than `total_energy_transfer` at matched T_gamma. Continuing the clean closure state asymptotes to 3.0345; continuing the artifact state asymptotes to 3.1150. STATUS.md claims a "collision-energy-contract pass" locks the AP source to the canonical Mangano prefactor — that contract is evidently not closing the ~0.7% gap, which is itself a finding.

So `N_eff_3T ≈ 3.1149` is **not a closure bug, not a deep physics result, and not merely mislabeled**: it is (a) endpoint truncation plus (b) a bridge/table energy-transfer calibration mismatch. The diagnostic label is already correct; the fix is to read at asymptotia and re-anchor the bridge moment.

**FLRW-limit validations that must pass before any Bianchi extension is trusted (physics Q5):** zero-shear invariance under perturbed-but-isotropic initial data; asymptotic N_eff convergence to the calibrated target; near-FLRW small-shear preservation (currently untested); and a same-state staged-vs-fused parity on Yp / D/H.

---

## 6. ODE / differentiation / solver audit

**Rodas5P suitability (numerics Q2).** Appropriate for the *background* (smooth, stiff-ish, low-D). The problem is forcing the *fused* system (thermo + shear + PSTF + network) through it during activation, where the deuterium-bottleneck Jacobian manifold is exactly the stiffness STATUS.md admits "the implicit step controller cannot follow without microstepping." That admission is an argument for staging, not for more phase-2 knobs.

**Fused vs staged (numerics Q1) — decision: stage.** Conventional codes split precisely because background quantities and abundances can be evaluated independently during radiation domination (LINX, arXiv:2408.14538). The abundances never enter `hubble_3T` / `coupled_3T_rhs`, so this holds for RABBIT, and shear couples only within the background sector, so it holds in Bianchi-I too. Staging puts the stiff ~10-D nuclide solve on a frozen background and removes the need for IMEX-inside-Rodas5P (the constraint the team locked against). Sketch in `patches/staged_operator_split_prototype.py`.

**Differentiation (numerics Q3).** The FD / host-JVP / frozen-source-JAX / lagged-Jacobian hybrid is overengineered. Host JVP is ~4.9 s of a 17.3 s segment but only ~15 s of a 420 s cold run (≈3.5%), so it is *not* the lever — the lagged-Jacobian PRs (BD395/396) optimized a non-bottleneck. Make the single source of truth the staged network's analytic/AD Jacobian in **log-abundance coordinates** (numerics Q4): evolving lnY makes positivity structural and deletes the corrective-replay-after-negative machinery.

---

## 7. Overengineering / overformalism audit

This is the most clear-cut track. Quantified:

- `augmented_continuous_ap65_rhs.py`: **25,795 lines, 340 functions, 96 `_phase2_*` functions, 25+ phase-2 policy/seed/cap knobs**, including telltale double-prefixed `_phase2_phase2_local_error_*` (mechanical accretion).
- `validation/` subtree: **106,135 LOC** — over half the 197.8k-line codebase — for a branch with no validated endpoint.
- **~12,072 LOC of `publication` / `readiness` machinery** (`augmented_publication_readiness/bundle/matrix/artifacts/plots`, `figure_renderer_v2`, `claim_readiness_review`) in a branch whose scope rules forbid publication claims. This is the single largest deletion candidate.
- Test suite: **4,436 tests, only 116 gold physics gates** (~2.6%). Count-lock tests confirmed: `plot_count == 5`, `total_plot_count == 6`, `case_count == 6`, `pairwise_comparison_count == 3`, `len(types_covered) == 6`. These break on cosmetic changes and protect no physics.
- STATUS.md itself is an exhibit: a multi-thousand-word AP0–AP81 / FB-02–FB-89 enumeration of evidence/gates/witnesses/provenance/bundles. The honesty discipline is genuine (`not_promoted` is stated), but it is buried under evidence plumbing.

**Top deletion / extraction candidates:**
1. Freeze and move the 12k-LOC publication/readiness suite out of the active path.
2. Split `augmented_continuous_ap65_rhs.py` into `background/`, `network/`, `phase2/`, `telemetry/` modules — target <3k LOC each.
3. Delete count-lock/schema-lock tests; replace with behavior/physics assertions.
4. Collapse the 25+ phase-2 knobs to the ≤3 that staging will still need.

**Goal/method mismatch (overeng Q5): yes.** The research goal is a Bianchi-I no-QKE BBN endpoint; the method has become "accrete telemetry and policy knobs around a fused RHS." The 96-function phase-2 layer and the publication apparatus are method artifacts, not physics progress.

---

## 8. PR roadmap (≤6 near-term)

| PR | Title | ≤5-PR feasible? | Success metric |
|---|---|---|---|
| **PR-1** | Staged background→network prototype behind a parity harness | **Breakthrough-class**; the *prototype* fits ≤1 PR, full default-switch does not | Staged vs fused parity on `bd349` cold endpoint: Yp, D/H, Σ_H, asymptotic N_eff within tol; staged wall on [2.5,3.0] < fused |
| **PR-2** | Re-anchor collision-bridge energy moment to `total_energy_transfer` prefactor | **Yes** | Asymptotic full-solver N_eff → 3.034 ± 0.003; ν-sector overheating at matched T_gamma < 0.05% |
| **PR-3** | Asymptotic N_eff readout + two-sided cold floor 3.00–3.06 | **Yes** (`neff_asymptotic_extension.py`) | Floor fires meaningfully; in-span and asymptotic both reported |
| **PR-4** | PR-B LRS/non-LRS parity at a fresh cold endpoint with full telemetry | **Yes** | Post-fix cold endpoint artifact with attributed ≥95%; LRS/non-LRS Σ_H parity |
| **PR-5** | Deflation I: freeze/move 12k-LOC publication suite; delete count-lock tests | **Yes** | src LOC −~12k; physics-gate fraction up; suite still green |
| **PR-6** | log-abundance network coordinates (positivity-structural) | **Yes** (lands inside PR-1's network) | No negative-candidate replay path needed; activation completes |

**Bottleneck classification:** phase-2 corrector = **breakthrough** (staging), not ≤5-PR patchable; N_eff excess = **≤5 PR** (calibration); non-LRS cold stall = **breakthrough** (subsumed by staging + state-dimension reduction); payload factory = **≤5 PR** but **deprioritize** (not the cold-run lever); host Jacobian = **already over-optimized, stop**; code size = partial ≤5 PR, full >5 PR.

---

## 9. Best three experiments per major blocker (after BoN)

**Phase-2 / activation:**
- (E-a) staged-vs-fused parity on `bd349` cold endpoint — the decisive experiment.
- (E-b) wall vs N-window for the *staged* solver to confirm [2.5, 3.0] drops below 420 s.
- (E-c) Radau-vs-Kvaerno-vs-Rodas5P on the *isolated* log-abundance network only, behind the parity harness.

**N_eff closure:**
- (E-d) direct comparison of bridge `dQ_nue_pair_N` / `dQ_nux_bank_N` against `total_energy_transfer` at matched (T_gamma, T_nu) to confirm the +0.67% locus.
- (E-e) asymptotic readout across q = 4/6/8 to separate angular-resolution from calibration.
- (E-f) two-sided floor regression.

**Overengineering:**
- (E-g) delete count-lock tests and measure suite-green retention.
- (E-h) extract `phase2/` module and diff behavior.
- (E-i) freeze publication suite and confirm no live-path import breaks.

---

## 10. Sample code

Delivered as files:
- `patches/staged_operator_split_prototype.py` (B1) — the central recommendation: verified physical decoupling, log-abundance network, parity-harness shape.
- `patches/neff_asymptotic_extension.py` (P1) — runs and reproduces the 3.1149 → asymptotic decomposition.
- `scripts/neff_proxy_convergence.py` (E1) — the proxy-vs-T_gamma curve.

These are prototype/pseudocode skeletons plus working diagnostics, not drop-in patches to the 25.8k-line module.

---

## 11. Missing evidence / files

1. **A post-fix cold endpoint artifact with populated telemetry** — the only included cold-after-fix summary has zero attribution. Request `diagnostic_outputs/bd346_*/cold_current_after_bridge_fix/` rerun with component walls.
2. **A same-HEAD reuse-on vs reuse-off payload ablation** — still absent (the ablation matrix admits this).
3. **The `total_energy_transfer`-vs-bridge `dQ` comparison artifact** at matched temperatures (E-d).
4. **Any completed activation-window run** — all included activation probes (BD378/379/383) time out.
5. **`diagnostic_outputs/` is excluded (~1.2 GB)** — acceptable, but the four items above should be requested by path.

---

## 12. Red-team objections to my own conclusion

- **"Staging breaks in Bianchi runs where shear back-reacts."** Shear enters H, but the *nuclear network* never sees shear directly — only H and T. Staging stays valid; the rebuttal would only bite if anisotropic stress coupled into nuclide rates, which it does not in this model.
- **"The +0.665% ν overheating could be q4 angular under-resolution, not a bridge calibration bug."** Possible — that is exactly why E-e (asymptotic readout across q) is listed. The evidence shows the offset exists at the full-solver level; it does not yet prove the *mechanism* is the prefactor vs the q grid. Labeled DERIVED, not VALIDATED.
- **"Deflating the publication suite destroys evidence."** It should be *frozen and moved*, not deleted — raw negative/nonfinite evidence preservation is untouched; only the premature publication apparatus leaves the active path.
- **"A staged rewrite is a multi-month breakthrough; keep patching phase-2."** The artifacts argue the opposite: ~10 PRs of patching moved only the cheapest window. The prototype + parity harness is ≤1–2 PRs to *decide*; if parity holds, the payoff is the activation window that currently does not complete at all.
- **"Maybe the cold floor at 3.0 is fine as a smoke check, not a physics gate."** Then it should be labeled a smoke check, not presented as the `N_eff_3T >= 3.0` physics tripwire the scope rules invoke — at N = 2.75 it cannot discriminate anything.

**Scope check worth surfacing:** the recent BD389–396 commits are individually clean engineering, but the artifact evidence shows they optimized a non-bottleneck window and a non-bottleneck (host Jacobian). That is the strongest single signal that the project is in a local minimum the staged rewrite is designed to escape.

---

## References (external comparison / CRAG anchors)

- LINX — *A Fast, Differentiable, and Extensible Big Bang Nucleosynthesis Package*, arXiv:2408.14538. Staged background→network architecture; comparator for AD strategy and API boundaries.
- PRIMAT — arXiv:1909.12046. Precision BBN thermodynamics / reaction-rate benchmark.
- PArthENoPE — arXiv:0705.0290. Public coupled-abundance BBN ODE solver.
- AlterBBN v2 — arXiv:1806.11095. Fast C BBN code, standard + alternative cosmologies.
- PRyMordial — PMC11266446. Python early-universe/BBN package, thermal background + weak-rate workflows.
- NUDEC_BSM — arXiv:2001.04466 (and v2, arXiv:2511.04747). Neutrino-decoupling comparator; momentum-averaged energy-transfer rate that RABBIT's `nudec_tables.py` cites (Escudero 2019).
- Precision N_eff: arXiv:2012.02726 (N_eff^SM = 3.0440 ± 0.0002); arXiv:2008.01074; review arXiv:2301.12299 (no-oscillation Boltzmann transport → ΔN_eff ≈ 0.034).
