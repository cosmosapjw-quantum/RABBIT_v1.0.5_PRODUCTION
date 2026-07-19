# Hypothesis Falsification Matrix

Date: 2026-06-02

## 2026-06-17 Current-Head Addendum

BD490/BD491 supersede the original BD279/BD280 PR-B parity uncertainty for the
specific current-head q4 controlled zero-shear FLRW pair.

- BD490
  (`diagnostic_outputs/bd490_pr_b_collision_on_parity_current_head/bd490_split_pairwise_cold_endpoint_current_head.json`)
  reached cold endpoint for LRS/non-LRS collision-on rows and passed
  LRS/non-LRS `N_eff_3T` parity, but failed the floor/band check because both
  rows ended high (`N_eff_3T ~= 3.11496`).
- BD491
  (`diagnostic_outputs/bd491_pr_b_thermal_collision_on_split_current_head/bd491_q4_thermal_start_lrs_nonlrs_collision_on_parity_current_head.json`)
  repeated the controlled pair with the thermal-start path and reports
  `default_on_blocker_passed=true`,
  `default_on_blocker_status=passed_pr_b_neff_floor_and_lrs_nonlrs_parity`,
  `N_eff_3T_floor_pair_passed=true`, and
  `N_eff_3T_parity_passed=true`.
- BD491 endpoint values are LRS `N_eff_3T=3.0348008780946367`,
  non-LRS `N_eff_3T=3.0348087179727026`, and
  `delta.N_eff_3T=7.839878065851735e-06` at the controlled pair level.

Scope: this closes the current-head q4 thermal-start controlled PR-B
floor/parity blocker only. It does not validate QKE, public production, high-q
convergence, nonzero-shear anisotropic transport, all settings, or default-on
optimization.

BD490/BD491 are not a single-knob ablation. Treat BD491 as the current passing
thermal-start controlled-pair baseline, not proof that thermal start alone
caused the floor/parity change. The top-level BD491 artifact still has
`passed=false`; the controlled PR-B pair/floor object is the scoped evidence.

Verdict labels: SUPPORTED, CONTRADICTED, PARTIAL, STALE, UNTESTED, UNKNOWN.

| ID | Claim / Hypothesis | Source Report | Code / Artifact Evidence | Probe / Test Command | Result | Verdict | Next Action |
|---|---|---|---|---|---|---|---|
| H1 | BD279 packet lacked the modules needed to close energy/N_eff questions. | BD279 lines 13-24 | HEAD contains `src/rabbit/thermo/nudec_coupled.py`, `nudec_tables.py`, EOS, and `transport/augmented_pstf_distribution.py`. | `ls`, `rg`, direct source reads | Modules exist in full repo. | STALE for HEAD; SUPPORTED for BD279 packet | Mark as packet-completeness issue only. |
| H2 | Logit convention is `f=sigmoid(-(q+A))`; `dA/dN=-df/(f(1-f))`. | BD279 lines 79-80; BD280 lines 63 | `augmented_pstf_distribution.py:216-245`; existing tests in `test_augmented_pstf_distribution.py`. | `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_augmented_pstf_distribution.py` | 10 distribution tests included in 14-test bundle, all passed. | SUPPORTED | Keep; add endpoint moment tests later. |
| H3 | Radial and angular collision paths convert occupation source to `dA`, not direct `C_modes -> dA`. | BD279 lines 79-80; BD280 lines 63-65 | `augmented_collision_bridge.py` calls `distribution_rhs_to_augmented_rhs` in radial/angular paths; tests include radial source conversion. | Targeted bridge tests in 14-test bundle | Passed. | SUPPORTED locally | Add regression for high-q Pauli amplification on radial path. |
| H4 | Energy closure is applied in occupation-source space before `dA`. | BD279 line 83; BD280 line 65 | `_apply_standard_3t_plasma_energy_closure` mutates `distribution_source_modes` before `dA`. | `pytest ... test_standard_3t_multimode_energy_closure_precedes_logit_conversion` | Passed. | SUPPORTED | Add final `f(1-f) q^3` moment check in a future PR. |
| H5 | Heavy-bank degeneracy is applied exactly once in active bridge/3T path. | BD279 line 82; BD280 line 64 | `BANK_DEGENERACY[NUX]=4`; bridge divides per-species target by 4 and scalar path multiplies once; 3T denominator uses `2*drho_pair`. BD491 controlled pair also passes floor/parity under the thermal-start path. | Existing bridge test plus new `test_positive_nux_bank_heat_source_warms_nux_relative_to_free_streaming`; BD491 controlled pair artifact. | Local test passed; BD491 pair passed PR-B floor/parity. | SUPPORTED in local path and current-head controlled pair scope | Continue with nonzero-shear/ell and high-q checks; do not infer public-production validation. |
| H6 | Positive `dQ_nux_bank_N` warms heavy neutrino sector relative to free streaming. | User requirement; BD279 heavy-bank suspicion | `coupled_3T_rhs_from_collision_moments` uses `dQ_nux_bank_N/(2*drho_pair)`. | `PYTHONPATH=src ... pytest -q tests/test_three_temperature_closure_invariants.py` | 3 passed; heated `dT_nux` > baseline. | SUPPORTED | Commit test; include in PR-2 invariant suite. |
| H7 | Python and JAX `N_eff_3T` definitions match. | Main audit question B.5/B.6 | Python `N_eff_from_3T`; JAX `N_eff_from_3T_jax`. | `tests/test_three_temperature_closure_invariants.py::test_python_and_jax_neff_3t_definitions_match` | Passed. | SUPPORTED as code definition consistency | Physical definition/proxy status remains PARTIAL. |
| H8 | `N_eff_3T ~= 2.994` is q/angular discretization. | BD279 hypothesis to reject | BD278 shard 1 read-only probe: 25 rows, min `2.9938263948`, max `2.9938277863`, spread `1.39e-6` across q/angular/shear. | Python JSON extractor over `diagnostic_outputs/bd278_endpoint_matrix_shards/bd278_endpoint_matrix_shard_1_of_4.json` | Spread too small for leading q/angular explanation. | CONTRADICTED/PARTIAL | Close discretization-only branch; run parity/hmax controlled probes. |
| H9 | `N_eff_3T ~= 2.994` is an LRS/non-LRS FLRW-limit parity problem. | BD279 lines 62-64 | BD490 controlled pair passed LRS/non-LRS parity but failed floor high (`N_eff_3T ~= 3.11496`). BD491 thermal-start controlled pair passed both floor and parity with `delta.N_eff_3T=7.839878065851735e-06`. | BD490 and BD491 current-head controlled pair artifacts. | Original uncontrolled `2.994` parity suspicion is superseded for current-head q4 thermal-start controlled scope. | STALE for BD279 artifact; CONTRADICTED as current-head controlled-pair leading blocker | Do not route next work through PR-B pair/floor. Extend to nonzero-shear/ell and high-q before broad default changes. |
| H10 | `N_eff_3T ~= 2.994` is caused by coarse `h_max` or restart/chaining differences. | BD279 line 64; BD280 solver notes | BD490/BD491 use controlled pair metadata, and BD491 passes the scoped PR-B pair/floor object. The comparison is not a single-knob ablation: `h_max`, initial A/np policy, activation/projection policy, and thermal-start setup differ. | BD490 vs BD491 artifact comparison. | h-max/restart remains a possible numerical sensitivity axis, but the old parity/floor suspicion no longer routes the next current-head q4 thermal-start work. | PARTIAL; no longer leading current-head blocker | Keep h_max sweeps only for future convergence studies, not as the next PR-B blocker. |
| H11 | Zero-shear FLRW submanifold is preserved at endpoint. | BD279 lines 91-95; BD280 line 67 | BD491 controlled pair reports LRS `Sigma_H=4.570206882454147e-31`, non-LRS `Sigma_H=3.3286755172789884e-31`, and fixed-S2 projection residual relative L2 max `1.3727219546475864e-15` on the non-LRS row. | Artifact probe; `test_bd217_nonlrs_radial_source_preserves_flrw_monopole_invariant`; BD491 controlled pair artifact. | Existing test passed; BD491 endpoint zero-shear drift is machine-floor scale for this controlled scope. | SUPPORTED in controlled thermal-start zero-shear scope | Structural no-projection and nonzero-shear anisotropic invariants remain separate checks. |
| H12 | Monopole projection may be hiding anisotropic collision source. | Red-team | `collision_projection_policy="flrw_monopole_only"` exists. | Not run | Not yet falsified. | PARTIAL/UNTESTED | Add no-projection FLRW source test and one q4 row if feasible. |
| H13 | AP65 endpoint path still uses dense `W=I/(gamma*h)-J` LU. | BD280 lines 77-88; solver auditor | `augmented_continuous_ap65_rhs.py:14024-14031`; `_factorized_linear_solver`. q4 endpoint profiles still show small W/J (`[62,62]`) and non-dominant outer linear solve. | Source inspection; BD491 profile summary. | Dense path confirmed; q4 does not justify structured-solve priority by itself. | SUPPORTED but not current q4 target | Revisit low-rank/block wiring only if high-q W/J evidence grows or a captured AP65-like system shows a dominant solve bucket. |
| H14 | Low-rank/Woodbury and block-sparse solver pieces exist and are algebraically valid. | BD280 lines 77-88, 128 | `solver_jax_rodas5p.py:89-120`, `195-230`, `548-579`; `tests/test_j04_jax_rodas5p.py`, `tests/test_block_sparse_jacobian.py`. | Low-rank 4 targeted tests; `tests/test_block_sparse_jacobian.py` | Low-rank 4 passed; block-sparse 11 passed. | SUPPORTED | Endpoint wiring PR. |
| H15 | Low-rank/Woodbury is wired into AP65 endpoint host. | BD280 says no | AP65 host still dense; low-rank function doc says public solver does not route through path. BD491 q4 does not make this the next blocker because outer solve wall is small relative to phase2/payload/residual. | Source inspection; BD491 profile summary. | Not wired, but not current q4 priority. | CONTRADICTED as implemented; PARTIAL for priority | Keep as conditional high-q/large-WJ work, not a default next PR. |
| H16 | Rejection/stiffness dominated by `geometry_thermo`. | BD279 lines 101-110 | External reports cite BD278 row telemetry; local artifact probe not exhaustive but report line and artifact fields support. | Read-only report/artifact inspection | Supported by report; not recomputed fully. | PARTIAL/SUPPORTED | Extract telemetry into plan; do not optimize collision payload alone. |
| H17 | RSS/VmHWM is already recorded per endpoint row. | Performance question D | `rg` for `ru_maxrss`, `VmHWM`, `tracemalloc`, `peak_rss`, `max_rss` in BD278 artifacts returned no hits. | `rg -n "ru_maxrss|VmHWM|tracemalloc|peak_rss|max_rss|memory" diagnostic_outputs/...` | No fields found. | CONTRADICTED | Add row-level memory instrumentation. |
| H18 | Large memory is Python-native overhead, not algorithmic. | User concern | No attribution fields. Structural dense solve and q-dependent collision arrays are both plausible. | No profiling run yet | Unknown. | UNKNOWN | Profile before language rewrite. |
| H19 | AP65 RHS and span ladder are god modules. | BD280 lines 32-38, 95-97 | `wc -l`: RHS 19,678, span ladder 13,359, tests 13,707 and 12,952. | `wc -l ...` | Confirmed. | SUPPORTED | Extract modules after physics/solver critical path. |
| H20 | Validation/evidence plumbing outgrew physics. | BD280 lines 95-108 | `src/rabbit/validation` 94,297 LOC across 71 modules; git hotspots show docs/status/registry and validation tests dominate. | `wc -l`, git hotspot commands | Confirmed. | SUPPORTED | Consolidate/delete in PRs that move runtime blockers. |
| H21 | Teff can be deleted immediately. | BD280 deletion candidate | `rg` found import-reachable Teff references in config, transport, weak, jax, tests. | `rg -n "Teff|teff" ...` | Import-reachable. | PARTIAL | Delete only after call graph and compatibility rejection tests. |
| H22 | Count-lock tests prove physics. | BD280 test critique | Tests include exact matrix count locks; packet smoke tests intentionally did not validate solver. | Test/source inspection | Count locks are not physics evidence. | CONTRADICTED | Replace with invariant/property tests over time. |
| H23 | Phase-2 corrector is current primary culprit. | User constraints / external reports | BD491-style profiles put phase2/corrector wall ahead of outer dense linear solve and behind/alongside payload; BD491 floor/parity success shows phase2 is not the `N_eff` physics blocker. | Source/artifact inspection; BD491 profile summary. | Supported as a performance attribution target after BD491; contradicted as a physics endpoint/floor explanation. | PARTIAL: performance target, not physics blocker | Re-measure component wall on BD491-style endpoint before optimizing; do not present phase2 work as an `N_eff` fix. |
| H24 | CPU-JAX/Rodas5P should be abandoned for another language now. | User memory/perf concern | Dense/block-low-rank options untested at endpoint; profiling missing. | No rewrite evidence | Premature. | CONTRADICTED as next step | Stay CPU-JAX/Rodas5P; profile and wire block/low-rank first. |

## Probe Output Summaries

- `tests/test_three_temperature_closure_invariants.py`: 3 passed in 0.98s.
- Low-rank/Woodbury targeted `tests/test_j04_jax_rodas5p.py`: 4 passed in 3.35s.
- `tests/test_block_sparse_jacobian.py`: 11 passed in 32.28s.
- PSTF distribution plus selected collision bridge tests: 14 passed in 2.15s.
- BD278 shard JSON probe: 25 rows, `N_eff_3T` spread `1.3914228254e-6`.

## Next Minimal Falsification Steps

1. Nonzero-shear/ell convergence study on the production collision-on path.
2. No-projection FLRW collision-source invariant.
3. Component-wall residual attribution on the BD491-style thermal-start
   endpoint run.
4. AP65 dense-LU vs block/low-rank endpoint microprobe only if high-q W/J
   evidence grows beyond the q4 `[62,62]` regime.
5. `h_max` convergence sweep as a convergence check, not as the leading PR-B
   parity explanation.
