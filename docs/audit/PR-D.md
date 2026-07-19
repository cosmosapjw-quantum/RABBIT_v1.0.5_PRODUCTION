# PR-D Audit

## Scope

Promote `backend="auto"` from the bounded JAX linearized tier-1 surface to the
bounded JAX exact-characteristic default surface, while preserving SciPy
fallback outside the documented CPU-first Type-I envelope.

## Static self-audit

- Registry truth now maps `CAPABILITY_BY_BACKEND["auto"]` to
  `jax_typeI_characteristic_tier1`.
- Runtime auto resolution stays bounded:
  - pure LRS Type I only
  - Teff disabled
  - collisions disabled
  - thermo tier in `{1, 2}`
  - `N_q >= 20`
- Explicit `backend="jax"` remains the linearized PSTF surface for backward
  compatibility.
- Low-`N_q` auto calls still fall back to SciPy, so existing conservative
  fallback behavior is preserved.

## Docs/contracts updated

- Registry-generated docs were re-rendered after the flip.
- `ROADMAP_STATE_OF_RECORD.md`, `ROADMAP_PR_CATALOG.md`,
  `JAX_CANONICAL_ARCHITECTURE.md`, and
  `JAX_MAIN_CANONICAL_PROMOTION_ROADMAP_2026-04-18.md` were synced to the new
  default-surface story.
- `SUPPORTED_CAPABILITIES.md` regained the literal `Inference` string through
  the feature registry so registry-sync no longer fails on that known doc gap.

## Verification

- `tests/test_production_gates.py::TestGateP15_ReleaseParity`:
  `4 passed in 40.50s`
- `tests/test_registry_sync.py tests/test_registry_doc_sync.py tests/test_pr07_inference_production.py tests/test_pr08_model_comparison_guardrail.py`:
  `43 passed, 3 skipped in 11.27s`
- Initial auto-dispatch + hierarchy + propagation + doc-sync bundle:
  `193 passed, 3 skipped, 1 failed in 226.48s`
  - the lone failure was the pre-existing `SUPPORTED_CAPABILITIES.md`
    `"Inference"` literal gap; fixed via the feature registry and then
    revalidated with the focused registry/doc suite above.
