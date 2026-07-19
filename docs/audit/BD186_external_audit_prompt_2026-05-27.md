# External Audit Prompt: BD186 Augmented Type-I No-QKE Collision Blocker

You are asked to audit a Python/JAX research codebase for anisotropic BBN with
augmented Type-I PSTF neutrino transport and diagonal no-QKE collision terms.
Please treat the attached documents and source files as the complete context.
Do not assume public production support.  QKE is out of scope.  Do not suggest
output clipping of negative abundances or negative `Y_p`; raw candidate states
must be preserved.

## What The Code Is Trying To Do

The branch attempts private diagnostic full-BBN runs for Bianchi Type-I
anisotropic neutrino transport.  The host solver is CPU-JAX/Rodas5P.  The
activated 9-species BBN network is not solved inside the host Rodas stages; it
is applied as a split coupled implicit BE/BDF2/Newton corrector in
`Y = X/A_mass`.  Dynamic neutrino collision terms are diagonal/no-QKE and enter
through a dynamic collision source that returns thermo energy transfer and
`dA_modes` for the distribution modes.

The current high-level architecture is:

```text
Rodas5P host step
  geometry + temperatures + A_modes + raw X_phase2 shell
  live weak rates from reconstructed neutrino monopoles
  collisionless non-LRS q/mu/phi transport
  optional dynamic collision dQ and dA_modes

post-accepted phase-2 corrector
  9-species BE/BDF2/Newton network solve in Y=X/A_mass
  physical-X block-max error norm
  raw negative candidate telemetry preserved
```

## Current Status

Fresh BD186 evidence:

1. Strict isotropic full-BBN private run succeeds:
   `diagnostic_outputs/bd186_external_audit/isotropic_sigma0_equalT_full_bbn_N4p8_max512.json`
   with `sigma_plus0=sigma_minus0=0`,
   `T_gamma0=T_nu_e0=T_nu_x0=0.8 MeV`, `N_span_end=4.8`,
   `max_steps=512`, `T_final_MeV=0.00914138906397148`,
   `Yp=0.16412585886748188`, `D/H=2.123358549828114e-05`,
   `Sigma_H=4.64e-15`, and `physical_full_bbn_span_ready=true`.

2. Non-LRS no-collision angular-grid full endpoint rows succeed but do not
   pass geometry resolution:
   `diagnostic_outputs/bd186_external_audit/nonlrs_angular_ablation_N4p8.json`.
   The two rows have small abundance deltas but `Sigma_H` differs by about
   `0.012155211646054737`.

3. Teff ablation is deprecated and must not be used as current evidence.
   A current LRS collisionless neutrino distribution ell diagnostic was run:
   `diagnostic_outputs/bd186_external_audit/lrs_collisionless_ell_ablation_sigma0p02_N0p2.json`.
   It converges at `ell_max=4` over a short collisionless span.  This is not a
   dynamic non-LRS full-BBN ell endpoint.

4. A current LRS 3T weak/network ell attempt failed before artifact output:
   `ValueError: X must contain only finite values` during SciPy Radau numerical
   Jacobian probing.  Treat it as failed evidence.

5. q14 dynamic collision full-span probing still fails:
   `diagnostic_outputs/bd186_external_audit/bd185_q14_collision_N4p8_cache_bound_probe.json`.
   It now writes an artifact instead of being killed before output, but
   `physical_full_bbn_span_ready=false`.  It records 344 dynamic collision
   payload build attempts and about 137.7 selected wall seconds.

## Key Code References

Please inspect these functions and contracts:

- `src/rabbit/jax/augmented_typeI_replay.py`
  - `_build_live_source_grid_jax`: q/angle grid, raw Laguerre and energy weights.
  - `_dynamic_collision_source_payload_from_restart_state_np`: dynamic collision
    source construction, metadata policy, `dA_modes` payload.
  - `_live_source_rhs_vector`: distribution reconstruction, live weak rates,
    stress feedback, q/mu/phi transport, `transport_dA + collision.dA_modes`.

- `src/rabbit/validation/augmented_continuous_ap65_rhs.py`
  - `_phase2_backward_euler_network_residual_Y`: network residual in `Y`.
  - `_phase2_backward_euler_network_residual_jacobian_Y`: analytic network
    Jacobian.
  - `_frozen_source_collision_dA_diagonal_jacobian`: damping-only collision
    diagonal Jacobian approximation.
  - `_run_step_cap_row`: host Rodas5P adaptive stepping and failure telemetry.

- `src/rabbit/jax/nonlinear_transport.py`
  - `build_q_diff_matrix`: current local finite-difference q derivative.
  - `q_diff_manufactured_error_summary`: q manufactured checks.

- `src/rabbit/transport/augmented_nonlrs_transport.py`
  - fixed three-mode diagonal non-LRS transport basis and S2 angular grid.

- `src/rabbit/transport/angular_decomposition.py`
  - generic angular/ell decomposition contracts that are not fully wired into
    the current dynamic non-LRS full-BBN endpoint.

- `src/rabbit/collisions/pstf_contractions.py`
  - six-monomial PSTF angular contraction tables and mode-space results.

- `src/rabbit/collisions/pstf_process_catalog.py`
  - PSTF radial collision source, moment weights, and projection/correction.

- `src/rabbit/network/abundances_standard.py`
  - 9-species, 31-reaction PRIMAT AC2024 network and flux convention.

## Questions To Answer

1. Is the current blocker diagnosis correct?
   Specifically, is q14/q15 dynamic Laguerre non-LRS failure primarily a dynamic
   collision source/Jacobian problem, rather than weak-rate history or the split
   phase-2 network corrector?

2. Is the current dynamic collision source algebra and implementation
   dimensionally and sign-consistent?
   Check q raw/energy weights, radial source normalization, p4 interpolation,
   moment projection, and units of `dA_modes` and `dQ_*_N`.

3. Is the host Jacobian missing a structurally important derivative?
   `collision.dA_modes` is added to `dA`, but only a damping-style diagonal
   approximation exists.  Would a local-q/species/mode block Jacobian or
   finite-difference/JVP source-response Jacobian be the next correct solver
   change?

4. Is the current fixed three-mode diagonal non-LRS live representation adequate
   for staging, or must generic ell/m dynamic full-BBN support be implemented
   before collision-on results can be interpreted?

5. Is this a Python-native performance limit?
   Distinguish Python metadata/JSON retention, NumPy/Python collision source
   construction, JAX JVP/LU solve cost, and true numerical stiffness.  Say
   whether a targeted compiled kernel is justified and where.  Do not recommend
   a full rewrite unless the attached evidence supports it.

6. What is overengineered?
   Critically evaluate whether artifact schemas, policy matrices, readiness
   language, and helper wrappers have grown beyond what is needed to move the
   runtime physics blocker.  Identify what should be deleted, split, or frozen.

7. What exact next PR should be implemented?
   Provide a minimal but substantive runtime-moving change, the files it should
   touch, and the focused tests/artifacts that would falsify it.

## Constraints

- QKE remains out of scope.
- Do not claim public production support.
- Keep CPU-JAX/Rodas5P as the repeated-run/backend target unless you can show a
  decisive reason not to.
- Do not add another standalone readiness/manifest/hash/figure gate as the main
  change.
- Do not hide negative abundances or negative `Y_p` by output truncation.
- Prefer a physics/solver/performance fix over another evidence-only wrapper.

## Attached Files To Inspect

Core docs:

- `AGENTS.md`
- `docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`
- `docs/TYPEI_AUGMENTED_NOQKE_FULL_E2E_BBN_PLAN.md`
- `docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- `docs/audit/BD186_augmented_typeI_noqke_external_audit_overview_2026-05-27.md`
- `docs/audit/BD186_current_collision_blocker_external_review_packet_2026-05-27.md`

Source:

- `src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
- `src/rabbit/validation/augmented_continuous_ap65_rhs.py`
- `src/rabbit/jax/augmented_typeI_replay.py`
- `src/rabbit/jax/solver_jax_rodas5p.py`
- `src/rabbit/jax/nonlinear_transport.py`
- `src/rabbit/transport/augmented_collision_bridge.py`
- `src/rabbit/transport/augmented_nonlrs_transport.py`
- `src/rabbit/transport/angular_decomposition.py`
- `src/rabbit/transport/augmented_typeI_weak_network.py`
- `src/rabbit/collisions/pstf_contractions.py`
- `src/rabbit/collisions/pstf_process_catalog.py`
- `src/rabbit/collisions/deterministic_reference.py`
- `src/rabbit/network/abundances_standard.py`
- `src/rabbit/weak/live_rates.py`
- `src/rabbit/jax/weak_live_jax.py`
- `src/rabbit/jax/nudec_coupled_jax.py`

Tests and scripts:

- `scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py`
- `scripts/run_augmented_3t_convergence_artifact.py`
- `tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py`
- `tests/test_augmented_continuous_ap65_rhs.py`
- `tests/test_jax_augmented_typeI_replay.py`
- `tests/test_angular_decomposition.py`
- `tests/test_augmented_convergence.py`
- `tests/test_standard_network.py`

Artifacts:

- `diagnostic_outputs/bd186_external_audit/artifact_summary.json`
- `diagnostic_outputs/bd186_external_audit/isotropic_sigma0_equalT_full_bbn_N4p8_max512.json`
- `diagnostic_outputs/bd186_external_audit/isotropic_full_bbn_N4p8_max512.json`
- `diagnostic_outputs/bd186_external_audit/nonlrs_angular_ablation_N4p8.json`
- `diagnostic_outputs/bd186_external_audit/lrs_collisionless_ell_ablation_sigma0p02_N0p2.json`
- `diagnostic_outputs/bd186_external_audit/bd184_q14_collision_N4p8_auto_chain_probe2.json`
- `diagnostic_outputs/bd186_external_audit/bd185_q14_collision_N4p8_cache_bound_probe.json`
- `diagnostic_outputs/bd186_external_audit/git_log_last100.txt`
