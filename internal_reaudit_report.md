# Internal Re-Audit Report

Date: 2026-06-02

Scope: RABBIT augmented Type-I PSTF no-QKE BBN solver, current branch
`feature/v2-f5-closed-model-events`, after external reports
`BD279_external_audit_report_2026-06-02.md` and
`BD280_external_full_code_audit_report_2026-06-02.md`.

## 2026-06-17 Supersession Note

This report is a historical BD279/BD280 re-audit. Its conclusion that
LRS/non-LRS FLRW-limit `N_eff_3T` parity was unresolved is superseded for the
current-head q4 thermal-start controlled pair by BD490/BD491. In particular,
BD491
(`diagnostic_outputs/bd491_pr_b_thermal_collision_on_split_current_head/bd491_q4_thermal_start_lrs_nonlrs_collision_on_parity_current_head.json`)
reports `default_on_blocker_status=passed_pr_b_neff_floor_and_lrs_nonlrs_parity`
with LRS `N_eff_3T=3.0348008780946367`, non-LRS
`N_eff_3T=3.0348087179727026`, and `delta.N_eff_3T=7.839878065851735e-06`.

The supersession scope is narrow: QKE, public production, high-q convergence,
nonzero-shear anisotropic transport, all settings, and optimization default-on
decisions remain outside this report and outside the BD491 evidence.

Routing rule: use this report only as a historical 2026-06-02 diagnosis. For
current implementation ordering, use the BD491 evidence plus the updated
`hypothesis_falsification_matrix.md`, `pr_acceleration_plan.md`, and
`MAIN_PR_LIST_RECOMMENDATION.md`. The body below is intentionally preserved, but
its parity and dense-LU priorities are no longer controlling current-head q4
work.

## Executive Summary

1. **BD279 and BD280 are not mutually exclusive.** BD280 is right that the local
   collision/logit/3T algebra is mostly healthy. BD279 is right that endpoint
   `N_eff_3T ~= 2.994` is not physically settled and should not be dismissed as
   discretization noise.
2. **Historical 2026-06-02 conclusion, superseded in scope:** the then-strongest
   unresolved physics blocker was the LRS/non-LRS FLRW-limit parity gap. BD490
   and BD491 now supersede that for the current-head q4 thermal-start controlled
   pair/floor object.
3. **Historical 2026-06-02 solver concern, demoted for q4:** dense AP65 host
   linear solving was suspected before later wall evidence. Current q4 work
   should not prioritize dense-LU replacement unless high-q or captured-system
   profiling proves it dominant.
4. **The architecture blocker is real and measurable.** `augmented_continuous_ap65_rhs.py`
   is 19,678 lines, the span ladder is 13,359 lines, and `validation/` is
   94,297 lines across 71 modules. This supports the external "god-module and
   evidence plumbing" diagnosis.
5. **One safe patch was added locally in this audit pass:** a fast 3T closure
   invariant test file covering Python/JAX `N_eff_3T` consistency, positive
   heavy-bank heating sign, and equal-temperature nu-nu equilibration
   neutrality. It is not committed yet.

## Claim Conflict Ledger: BD279 vs BD280 vs Code Evidence

| Claim | Source | Code / Artifact Evidence | Verdict | Internal Resolution |
|---|---|---|---|---|
| BD279 packet omitted thermo/N_eff modules, so energy/N_eff could not be closed from that packet. | BD279 lines 13-24 | Current repo contains `src/rabbit/thermo/nudec_coupled.py`, `nudec_tables.py`, EOS, and `augmented_pstf_distribution.py`. | STALE for full repo; SUPPORTED for BD279 packet | Do not treat this as a HEAD source gap. Treat it as a prior packet-completeness gap. |
| Standard FLRW anchor and q3 radial collision-on are near standard BBN. | BD279 lines 32-33 | BD199 summary records anchor `N_eff_3T=3.114862...` and q3 radial `N_eff_3T=3.105381...`. | SUPPORTED by artifact | Anchor physics is improved, but `N_eff_3T` remains a 3T proxy. |
| BD278 shard rows complete and sit at `N_eff_3T ~= 2.99383`. | BD279 lines 35, 59-64 | Read-only JSON probe on `diagnostic_outputs/bd278_endpoint_matrix_shards/bd278_endpoint_matrix_shard_1_of_4.json`: 25 rows, min `2.9938263948`, max `2.9938277863`, spread `1.39e-6`. | SUPPORTED | This strongly weakens a q/angular-discretization-only explanation. |
| Heavy-bank/parity suspicion remains despite local degeneracy correctness. | BD279 lines 66-71 | `nudec_coupled.py` uses `dQ_nux_bank/(2*drho_pair)`, bridge divides per-species NUX target by bank degeneracy and multiplies once for scalar dQ. New test confirms positive bank heat warms `T_nu_x` relative to free streaming. BD491 later passes the scoped q4 thermal-start controlled pair/floor object. | PARTIAL historically; superseded for current q4 controlled pair | Local channel sign/degeneracy is supported; current q4 controlled pair/floor is no longer unresolved, but broader settings remain open. |
| Local physics core is healthy. | BD280 lines 14, 63-66 | Existing tests plus new test: logit conversion, occupation-space closure, 3T equations, heavy-bank sign pass. | SUPPORTED locally | Do not promote this to endpoint physical validation. |
| Dense solve/low-rank unwired is the top engineering blocker. | BD280 lines 16, 77-88, 128 | `solver_jax_rodas5p.py` has low-rank/Woodbury helpers and says the public solver does not route through that path; AP65 host still forms dense `W` at `augmented_continuous_ap65_rhs.py:14029`. Low-rank parity tests pass. Later q4 wall evidence demotes this from immediate priority. | SUPPORTED historically; PARTIAL as current priority | Treat as conditional high-q/large-WJ work, not the next q4 blocker. |
| Continuous-vs-piecewise ambiguity exists. | BD280 lines 16, 77 | BD278 endpoint rows use phase-2 activation/corrector fields; older continuous single-RHS blocker language remains in docs/status. | PARTIAL/SUPPORTED | Plan must label primary target path versus diagnostic path in code/docs. |
| Test suite is too count/skeleton heavy. | BD280 lines 103-108 | Current endpoint matrix tests include `len(cases)==98`; BD280 packet tests were intentionally smoke-only. | SUPPORTED | Replace count locks gradually with invariant/property tests. |
| `ell_max=2` is exact. | Current conventions plus BD279/BD280 critique | `config/conventions.py` and grid config still present exactness language; augmented no-QKE docs and AP65 claim boundary describe fixed diagonal three-mode S2 projection rather than generic ell/m convergence. | CONTRADICTED for collisional augmented runtime | Fence exactness to the valid collisionless/free-streaming regime. |

## Supported Claims

- IMPLEMENTED/SUPPORTED: logit convention
  `f = sigmoid(-(q + A))` and `dA/dN = -df/dN / max(f(1-f), eps)`.
  Evidence: `src/rabbit/transport/augmented_pstf_distribution.py:216-245`;
  `tests/test_augmented_pstf_distribution.py` ran and passed.
- IMPLEMENTED/SUPPORTED: standard-3T radial energy closure is applied to
  occupation source modes before conversion to `dA`. Evidence:
  `src/rabbit/transport/augmented_collision_bridge.py:2224-2310`; selected
  bridge tests ran and passed.
- IMPLEMENTED/SUPPORTED: active heavy-bank scalar path applies the bank
  degeneracy once in the bridge and the 3T denominator uses two pairs.
  Evidence: `augmented_collision_bridge.py:2196-2220`, `2712-2746`;
  `nudec_coupled.py:296-300`; new invariant test.
- IMPLEMENTED/SUPPORTED: low-rank/Woodbury and block-sparse solver pieces exist
  and pass focused algebraic tests.
- SUPPORTED: AP65 host dense LU remains the actual endpoint solve surface.
- SUPPORTED: architecture and validation evidence-plumbing debt are major
  development-speed blockers.

## Contradicted Or Stale Claims

- STALE: "Full repo lacks thermo/N_eff modules." True only for BD279 audit
  packet contents, not HEAD.
- CONTRADICTED: "`ell_max=2` exact" if applied to the collisional augmented
  runtime. It can be retained only as a collisionless/free-streaming statement.
- CONTRADICTED/PARTIAL: "`N_eff_3T ~= 2.994` is just q/angular discretization."
  BD278 spread across q/angular/shear in shard 1 is only `1.39e-6`.
- PARTIAL: "Projection-fixed zero-shear endpoint proves full FLRW invariant."
  Current endpoint shear is tiny, but structural invariance without projection
  remains untested.

## Unresolved Claims

- Whether `N_eff_3T ~= 2.994` is caused by non-LRS wiring, h-max/chaining
  differences, source-composition policy, or a subtle 3T proxy interpretation.
- Whether q9/q10 memory >20 GB is dominated by dense Jacobian/LU, collision
  kernel arrays, JAX compilation/runtime caches, Python diagnostics, or a
  mixture. Current endpoint artifacts lack RSS/VmHWM/tracemalloc fields.
- Whether block-JVP or low-rank/Woodbury can be safely wired into AP65 endpoint
  rows without changing endpoint observables.
- Whether continuous single-RHS should remain a primary target or be explicitly
  demoted to diagnostic while the piecewise phase-1/phase-2 corrector path is
  the current endpoint target.

## Physics Invariants Verdict

Local algebraic invariants are in good shape:

- logit sign and Pauli scaling: SUPPORTED;
- energy closure before logit mapping: SUPPORTED;
- 3T equation signs and denominators: SUPPORTED;
- heavy-bank positive heat response: SUPPORTED by new test;
- nu-nu equal-temperature neutrality: SUPPORTED by new test.

Endpoint-level physics remains PARTIAL globally, but not because the current-head
q4 thermal-start controlled LRS/non-LRS pair is unresolved. BD491 supplies that
scoped pair/floor evidence. The next physics falsifiers are broader setting
coverage, nonzero-shear/ell convergence, and unprojected FLRW source invariants.

## Solver/Stiffness Verdict

The in-tree CPU-JAX/Rodas5P/AP65 target remains valid. Existing code supports
block-sparse Jacobian construction and low-rank/Woodbury stage solve algebra,
and tests pass, but AP65 still materializes dense `W` and uses LU. That is an
implementation gap, not a current q4 priority unless profiling shows the solve
bucket dominates in a larger high-q regime.

The error telemetry from BD279/BD280 points to `geometry_thermo` dominance, so
optimizing only collision payload construction would be too narrow. The solver
PR must include stage-solution parity, endpoint observable parity, and memory
instrumentation.

## Performance/Memory Verdict

The memory concern is real but not yet attributed in artifacts. Current reports
cite 6.83 GB for a shard and >20 GB for q9/q10, but row-level RSS/VmHWM fields
are missing. Before any language rewrite, add per-row RSS/VmHWM/tracemalloc and
separate JAX compile, runtime, dense Jacobian/LU, collision arrays, retained
caches, and JSON payload construction.

Do not recommend Rust/C++/Julia/Pallas as the next move. The next move is
post-BD491 component-wall/residual attribution and nonzero-shear/ell evidence;
block/low-rank solve wiring is conditional on high-q or large-WJ profiling.

## Architecture/Test Verdict

The architecture debt is load-bearing:

- AP65 RHS and span ladder are too large for reliable physics changes.
- validation evidence plumbing dominates active runtime code.
- Teff is deprecated and import-reachable, so deletion must be call-graph-gated.
- tests overuse count/schema locks and underuse physics invariants.

Small invariant tests are acceptable because they directly protect physics
contracts and do not add claim/readiness gate plumbing.

## Red-Team Objections

1. **Projection may hide an anisotropic-source bug.** Passing zero-shear rows
   under `flrw_monopole_only` projection does not prove the unprojected
   collision source preserves the FLRW submanifold.
2. **Fresh HEAD may differ from packet artifacts.** BD279 did not rerun the
   solver; BD280 ran only minimal packet tests. A q4 current-tree fresh run is
   needed before making endpoint claims.
3. **Solver optimization can preserve wrong physics faster.** Dense-vs-Woodbury
   parity is necessary but not sufficient; LRS/non-LRS parity and `N_eff_3T`
   definition must be pinned in the same acceleration sequence.
4. **Silent defaults can mimic physical changes.** Past fixed `Xn0=0.13` /
   `A0=1e-5` regressions show that endpoint rows must serialize and validate
   all initialization and source-policy choices.

## Final Prioritized Blockers

1. Broader setting coverage beyond the BD491 q4 thermal-start controlled
   pair/floor object.
2. Nonzero-shear/ell convergence and unprojected FLRW source invariants.
3. Missing per-row RSS/VmHWM/tracemalloc and compile/runtime attribution.
4. AP65 RHS and span-ladder god modules.
5. Overgrown evidence/readiness/figure plumbing and deprecated Teff surfaces.
6. Weak count-lock tests in place of invariant/property tests.
7. Conditional high-q dense-LU/low-rank/block solve evidence if W/J grows beyond
   the q4 regime.

## Commands Run

```bash
rg -n "BD279|BD280|augmented Type-I|no-QKE|anti-drift|Rodas5P|Teff" /home/cosmosapjw/.codex/memories/MEMORY.md
sed -n '1,220p' .../skills/using-superpowers/SKILL.md
sed -n '1,220p' .../skills/systematic-debugging/SKILL.md
sed -n '1,220p' .../skills/codebase-recon/SKILL.md
sed -n '1,180p' .../skills/test-driven-development/SKILL.md
sed -n '1,180p' .../skills/brooks-audit/SKILL.md
sed -n '1,260p' BD279_external_audit_report_2026-06-02.md
sed -n '1,260p' BD280_external_full_code_audit_report_2026-06-02.md
sed -n '1,220p' AGENTS.md
sed -n '1,180p' docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md
git rev-list --count HEAD
git log --oneline -30
git log --format=format: --name-only --since="1 year ago" | sort | uniq -c | sort -nr | head -20
git log -i -E --grep="fix|bug|broken|repair|tighten" --name-only --format='' --since="1 year ago" | sort | uniq -c | sort -nr | head -20
rg -n "N_eff_3T|N_eff_from_3T|3T_asymptotic|asymptotic_temperature|instantaneous|neff" src/rabbit tests scripts diagnostic_outputs/bd278_endpoint_matrix_shards diagnostic_outputs/bd199_flrw_collision_audit -g '!*.zip'
rg -n "distribution_rhs_to_augmented_rhs|standard_3t|dQ_nux_bank_N|BANK_DEGENERACY|coupled_3T_rhs" src/rabbit/transport src/rabbit/thermo src/rabbit/collisions tests/test_augmented_collision_bridge.py -g '*.py'
rg -n "low_rank|Woodbury|block_jvp|GMRES|JFNK|linear_solve|jacobian_policy" src/rabbit tests scripts -g '*.py'
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_three_temperature_closure_invariants.py
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_public_linear_solver_hooks_support_low_rank_payload tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_low_rank_materialization_matches_dense_assembly tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_woodbury_stage_linear_solve_matches_dense tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_low_rank_rodas_step_matches_dense_step_for_linear_rhs
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_block_sparse_jacobian.py
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_augmented_pstf_distribution.py tests/test_augmented_collision_bridge.py::test_pstf_radial_distribution_source_converts_to_logit_rhs tests/test_augmented_collision_bridge.py::test_standard_3t_scalar_temperature_source_excludes_internal_nunu_redistribution tests/test_augmented_collision_bridge.py::test_standard_3t_multimode_energy_closure_precedes_logit_conversion tests/test_augmented_collision_bridge.py::test_bd217_nonlrs_radial_source_preserves_flrw_monopole_invariant
```

## Files Changed

- `tests/test_three_temperature_closure_invariants.py`
- `internal_reaudit_report.md`
- `hypothesis_falsification_matrix.md`
- `skill_and_subagent_usage.md`
- `pr_acceleration_plan.md`
