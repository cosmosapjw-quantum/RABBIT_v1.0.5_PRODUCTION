# PR-AP4 Combined Full-Span Piecewise Refinement

Date: 2026-05-17

Scope:
- Add an AP4/AP65 diagnostic artifact and CLI for piecewise source-refresh
  refinement over the combined angular+`pstf_radial` non-LRS 3T solve.
- Compare two nonuniform `piecewise_frozen` source-refresh schedules over the
  same physical-preview `N_span=(0,1e-3)` span:
  coarse `(1e-6,1e-4,1e-3)` and refined `(1e-6,1e-5,1e-4,1e-3)`.
- Preserve no-QKE, no-public-dispatch, and no-production-SMC boundaries.

Implementation:
- `build_augmented_nonlrs_combined_full_span_piecewise_refinement_artifact`
  runs the existing AP4/AP65 candidate gate once per schedule with a shared
  radial-grid cache and records refined-minus-reference observable deltas.
- `write_augmented_nonlrs_combined_full_span_piecewise_refinement_artifact`
  writes the JSON artifact with deterministic sorted output.
- `scripts/run_augmented_nonlrs_combined_full_span_piecewise_refinement.py`
  exposes a smoke-scale CLI runner with dry-run metadata and summary output.

Verification:
- TDD RED:
  `PYTHONPATH=src pytest -q tests/test_augmented_stability_envelope.py -k piecewise_refinement`
  first failed because the builder did not exist.
- TDD RED:
  `PYTHONPATH=src pytest -q tests/test_augmented_stability_envelope.py -k piecewise_refinement_script`
  first failed because the CLI script did not exist.
- Focused GREEN:
  `PYTHONPATH=src pytest -q tests/test_augmented_stability_envelope.py -k piecewise_refinement`
  -> `2 passed, 76 deselected`.
- Real numeric artifact:
  `PYTHONPATH=src python scripts/run_augmented_nonlrs_combined_full_span_piecewise_refinement.py --output /tmp/rabbit_ap4_piecewise_refinement.json`
  -> `passed=true`, `violations=[]`, `schedule_count=2`,
  `source_evaluation_max=4`, `nfev_max=56`, and
  `radial_grid_cache_entries=72`.

Concrete refined-minus-coarse deltas at `N_span=(0,1e-3)`:
- `T_gamma_final=-1.1280976153216216e-12`
- `T_nu_e_final=2.4635848916432224e-12`
- `T_nu_x_final=5.268008251846368e-13`
- `Xn_final=2.3314683517128287e-15`
- `collision_dA_abs_max_final=4.4915406029882865e-14`
- `radial_offdiagonal_nunu_pair_max_abs_energy_residual_final=2.0117032497289633e-21`
- `source_evaluations=1`
- `nfev=9`

Claim boundary:
- This is operator-split source-refresh refinement evidence.  It is not a
  continuous live-RHS collision solve, public dispatch, production SMC
  validation, QKE, or promotion-grade full-BBN support.
