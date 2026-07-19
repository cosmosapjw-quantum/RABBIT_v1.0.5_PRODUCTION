# PR-AP4 Piecewise Terminal QED Forwarding

Date: 2026-05-17

Scope:
- Fix AP4/AP65 `piecewise_frozen` terminal source re-evaluation so it forwards
  the selected scalar QED correction model to
  `evaluate_augmented_nonlrs_nonlinear_combined_collision_3T_source`.
- Preserve the staged no-QKE and no-public-dispatch boundaries.

Implementation:
- `evaluate_augmented_nonlrs_nonlinear_combined_collision_3T_source` now accepts
  `qed_correction_model`, validates it, and records finite/exact scalar-QED
  one-hot diagnostics on the returned combined source.
- `_run_nonlrs_combined_piecewise_frozen_case` forwards
  `spec.qed_correction_model` during terminal source diagnostics, matching the
  model used by the subspan solves.

Verification:
- TDD RED:
  `PYTHONPATH=src pytest -q tests/test_augmented_stability_envelope.py -k piecewise_frozen_charge_neutrality_handoff`
  first failed with missing `qed_correction_model` in the terminal source kwargs.
- GREEN:
  `PYTHONPATH=src pytest -q tests/test_augmented_stability_envelope.py -k piecewise_frozen_charge_neutrality_handoff`
  -> `1 passed, 77 deselected`.
- Real AP4/AP65 charge-neutral plus exact scalar-QED named-preset run:
  `PYTHONPATH=src python scripts/run_augmented_nonlrs_combined_full_span_3t_candidate_gate.py --preset piecewise_physical_preview --electron-chemical-potential-mode charge_neutrality --qed-correction-model exact_finite_mu_scalar --output /tmp/rabbit_ap4_piecewise_physical_preview_charge_neutral_exact_qed_after_terminal_forward.json`
  -> `passed=true`, `violations=[]`, `span_count=2`, `max_span_length=0.001`,
  `source_evaluation_max=3`, `electron_chemical_potential_abs_max=3.298436883301101e-10`,
  `electron_charge_asymmetry_density_abs_max_MeV3=6.618704871052225e-11`,
  and `collision_dA_abs_max_final=0.00026527558184847477`.

Claim boundary:
- This closes terminal diagnostic forwarding for an already selected scalar-QED
  control.  It is not anisotropic/tensor QED response, QKE, public dispatch,
  production SMC validation, or promotion-grade full-BBN support.
