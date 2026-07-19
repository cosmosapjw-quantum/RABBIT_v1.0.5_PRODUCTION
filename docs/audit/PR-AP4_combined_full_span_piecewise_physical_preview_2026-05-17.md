# PR-AP4 Combined Piecewise Physical Preview Preset

Date: 2026-05-17

Scope:
- Promote the already passing AP4/AP65 nonuniform `piecewise_frozen`
  combined angular+`pstf_radial` source-refresh path into a named
  `piecewise_physical_preview` preset.
- Keep the preset diagnostic-only: no public dispatch, no production SMC
  validation, no QKE, and no promotion-grade full-BBN claim.

Implementation:
- Added preset constants in `src/rabbit/validation/augmented_stability.py`.
- `piecewise_physical_preview` resolves to `N_span=(0,1e-4),(0,1e-3)`,
  `source_update_policy=piecewise_frozen`,
  `source_update_subspan_ends=(1e-6,1e-4,1e-3)`, `method=Radau`,
  `max_pstf_radial_source_evaluations=8`, and `max_nfev=10000`.
- Artifact inputs split routine numeric coverage from isolated diagnostics:
  routine `N_span=(0,1e-4)`, isolated diagnostic `N_span=(0,1e-3)`.
- Artifact inputs also record
  `piecewise_physical_preview_supported_electron_modes=[fixed, charge_neutrality]`.
- Artifact inputs also record
  `piecewise_physical_preview_supported_qed_correction_models=[finite_mu_scaled, exact_finite_mu_scalar]`.
- Artifact inputs also record the four validated electron-bath/scalar-QED
  control combinations, including the charge-neutral plus exact scalar-QED row.
- The combined-source artifact supported-claims ledger now records the named
  preset, charge-neutral finite-mass e-/e+ compatibility, and exact scalar-QED
  cross-control compatibility while preserving the stage-scoped blockers.

Verification:
- TDD RED:
  `PYTHONPATH=src pytest -q tests/test_augmented_stability_envelope.py -k piecewise_physical_preview`
  first failed because the CLI rejected the new preset.
- Focused tests:
  `PYTHONPATH=src pytest -q tests/test_augmented_stability_envelope.py -k "piecewise_physical_preview or physical_preview_dry_run or piecewise_frozen_dry_run or piecewise_frozen_subspan_ladder"`
  -> `4 passed, 72 deselected`.
- Real AP4/AP65 numeric artifact:
  `PYTHONPATH=src python scripts/run_augmented_nonlrs_combined_full_span_3t_candidate_gate.py --preset piecewise_physical_preview --output /tmp/rabbit_ap4_piecewise_physical_preview.json`
  -> `passed=true`, `violations=[]`, `span_count=2`, `max_span_length=0.001`,
  `source_evaluation_max=3`, `radial_grid_cache_entries=45`.
- Real AP4/AP65 charge-neutral numeric artifact:
  `PYTHONPATH=src python scripts/run_augmented_nonlrs_combined_full_span_3t_candidate_gate.py --preset piecewise_physical_preview --electron-chemical-potential-mode charge_neutrality --output /tmp/rabbit_ap4_piecewise_physical_preview_charge_neutral.json`
  -> `passed=true`, `violations=[]`, `span_count=2`, `max_span_length=0.001`,
  `source_evaluation_max=3`, `electron_chemical_potential_abs_max=3.298436883974794e-10`,
  and `electron_charge_asymmetry_density_abs_max_MeV3=6.618704874587679e-11`.
- Real AP4/AP65 exact scalar-QED numeric artifact:
  `PYTHONPATH=src python scripts/run_augmented_nonlrs_combined_full_span_3t_candidate_gate.py --preset piecewise_physical_preview --qed-correction-model exact_finite_mu_scalar --output /tmp/rabbit_ap4_piecewise_physical_preview_exact_qed.json`
  -> `passed=true`, `violations=[]`, `span_count=2`, `max_span_length=0.001`,
  `source_evaluation_max=3`, `qed_correction_model=exact_finite_mu_scalar`,
  and `radial_grid_cache_entries=45`.
- Real AP4/AP65 charge-neutral plus exact scalar-QED numeric artifact:
  `PYTHONPATH=src python scripts/run_augmented_nonlrs_combined_full_span_3t_candidate_gate.py --preset piecewise_physical_preview --electron-chemical-potential-mode charge_neutrality --qed-correction-model exact_finite_mu_scalar --output /tmp/rabbit_ap4_piecewise_physical_preview_charge_neutral_exact_qed.json`
  -> `passed=true`, `violations=[]`, `span_count=2`, `max_span_length=0.001`,
  `source_evaluation_max=3`, `qed_correction_model=exact_finite_mu_scalar`,
  `electron_chemical_potential_abs_max=3.298436883301101e-10`, and
  `electron_charge_asymmetry_density_abs_max_MeV3=6.618704871052225e-11`.

Concrete numeric rows:
- `N_span=(0,1e-4)`: `T_gamma_final=0.7999214320680265`,
  `H_rate_s_final=0.43146274191619854`, `Xn_final=0.13000591435767977`,
  `nfev=31`, `source_update_subspan_count=2`,
  `collision_dA_abs_max_final=0.00026527543953436006`, and
  `radial_offdiagonal_nunu_pair_max_abs_energy_residual_final=8.205631676526035e-20`.
- `N_span=(0,1e-3)`: `T_gamma_final=0.7992146796753832`,
  `H_rate_s_final=0.43068978083203713`, `Xn_final=0.13005888045355307`,
  `nfev=47`, `source_update_subspan_count=3`,
  `collision_dA_abs_max_final=0.000265067738217371`, and
  `radial_offdiagonal_nunu_pair_max_abs_energy_residual_final=7.326834993749698e-20`.

Charge-neutral concrete row:
- `N_span=(0,1e-3)`: `T_gamma_final=0.7992146727272548`,
  `H_rate_s_final=0.43068977699536865`, `Xn_final=0.13005888045236302`,
  `electron_chemical_potential_MeV_final=3.295370971985368e-10`,
  `electron_charge_asymmetry_density_MeV3_final=6.59880533199606e-11`,
  `source_update_charge_asymmetry_state_handoff=1`,
  `collision_dA_abs_max_final=0.0002650678248380589`, and
  `radial_offdiagonal_nunu_pair_max_abs_energy_residual_final=8.216219588366713e-20`.

Exact scalar-QED concrete row:
- `N_span=(0,1e-3)`: `T_gamma_final=0.7992145483449656`,
  `H_rate_s_final=0.4302626189334139`, `Xn_final=0.13005893883614722`,
  `nfev=47`, `qed_correction_model_exact_finite_mu_scalar=1`,
  `collision_dA_abs_max_final=0.0002650691502775048`, and
  `radial_offdiagonal_nunu_pair_max_abs_energy_residual_final=9.634999775017666e-20`.

Charge-neutral plus exact scalar-QED concrete row:
- `N_span=(0,1e-3)`: `T_gamma_final=0.7992145483449652`,
  `H_rate_s_final=0.4302626189334136`, `Xn_final=0.13005893883614483`,
  `electron_chemical_potential_MeV_final=3.295370300138031e-10`,
  `electron_charge_asymmetry_density_MeV3_final=6.598801808206643e-11`,
  `source_update_charge_asymmetry_state_handoff=1`,
  `qed_correction_model_exact_finite_mu_scalar=1`,
  `collision_dA_abs_max_final=0.0002650691502786733`, and
  `radial_offdiagonal_nunu_pair_max_abs_energy_residual_final=9.634999775017666e-20`.

Claim boundary:
- This is a named operator-split source-refresh preview for existing AP4/AP65
  physics.  It is not fully live-RHS collision coupling, public dispatch,
  production SMC evidence, QKE, or promotion-grade full-BBN span support.
