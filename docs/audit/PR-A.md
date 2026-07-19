# PR-A Audit Verdict

## Summary

PR-A removes the explicit `J_j` ODE state from the JAX characteristic
driver and reconstructs the angular Jacobian analytically from
`(X0, S, μ_j)`. The physics surface is unchanged; the audit resolved the
only subtle point, namely that the production transport weight is the
forward Jacobian `J_j = dμ_j/dμ_{j,0}` required by eq (55), not the
inverse form suggested by the raw OCR text around eq (51).

## Checklist

- [x] Paper-equation provenance checked from `RABBIT_report.pdf` pages
      16-18 and recorded in `PR-A_stage1.md`.
- [x] No mock / placeholder / TODO logic introduced.
- [x] New primitive coverage added:
      `tests/test_pr_a_analytic_jacobian.py`.
- [x] Tier-1 parity regression remains green:
      `tests/test_jax_typeI_characteristic_parity.py`
      previously rerun with PR-A changes in place.
- [x] Tier-2 parity regression remains green:
      `tests/test_jax_typeI_characteristic_tier2.py`
      previously rerun with PR-A changes in place.
- [x] Cross-backend matched-physics regression green:
      `tests/test_cross_backend_regression.py` → 4 passed.
- [x] State/Jacobian size reduced as intended:
      tier-1 phase-2 state dim = 25, tier-2 phase-2 = 27,
      dense Jacobian shape = `(25, 25)`.
- [x] RHS identity cache remains stable:
      repeated `_get_char_rhs(...)` returns the same `id(rhs_fn)`.
- [x] Warm CPU timing on the local sandbox remains in the expected band:
      1.66 s min / 1.70 s mean with
      `JAX_PLATFORMS=cpu JAX_COMPILATION_CACHE_DIR=/tmp/rabbit_jax_cache`.
- [x] Documentation synchronized:
      `ROADMAP_STATE_OF_RECORD.md`,
      `ROADMAP_PR_CATALOG.md`,
      `JAX_CHAR_GPU_OPTIMIZATION_PLAN.md`,
      `docs/audit/PR-A_stage{1,2,3}.md`.

## Numerical evidence

- `tests/test_pr_a_analytic_jacobian.py` → 4 passed.
- `tests/test_jax_typeI_characteristic_parity.py` +
  `tests/test_jax_typeI_characteristic_tier2.py` +
  `tests/test_pr_a_analytic_jacobian.py` → 50 passed.
- `tests/test_cross_backend_regression.py` → 4 passed.
- `tests/test_advanced_envelope_lock.py` +
  `tests/test_inference_hierarchy_lock.py` → 116 passed after syncing
  the already-merged 9-backend registry / 19-key capability catalog.
- Consolidated green regression bundle
  (`test_pr_a_analytic_jacobian`, tier-1 parity, tier-2 parity,
  cross-backend regression, advanced envelope lock, inference hierarchy
  lock) → **170 passed in 349.89 s**.
- Direct rerun of the documented stable-red quartet
  (`test_supported_capabilities_mentions_features`,
  `test_classB_typeV_bbn_gold`, `test_jax_flrw_gold`,
  `test_anisotropy_signal_parity`) still yields exactly **4 failures**,
  matching the baseline recorded in `ROADMAP_STATE_OF_RECORD.md`.
- Tier-1 roadmap parity stays at |ΔY_p| ≤ 4 × 10⁻⁸.
- Tier-2 roadmap parity stays at |ΔY_p| ≤ 7 × 10⁻⁸.

## Adversarial review verdict

Cold adversarial local review found no off-by-one layout bug, no stale
three-value unpack of `characteristic_rhs_jax`, no sign error in the
analytic Jacobian, and no mock/stub logic. The only non-physics issue
surfaced by the fast regression sweep was stale 7-backend envelope
tests; the registry has already had 9 backends since the characteristic
surfaces landed, so those tests were synced as part of closing this PR.

Verdict: **pass**.
