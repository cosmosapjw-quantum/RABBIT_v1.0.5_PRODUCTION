<!-- BEGIN:README_HEADER -->
# RABBIT v1.0.0

Type I BBN research runtime with publication validation in progress.
<!-- END:README_HEADER -->

<!-- BEGIN:IDENTITY -->
RABBIT provides a **canonical Type I BBN core** in the registry-runtime sense only, plus a candidate/substrate perimeter. Registry maturity, gold regression, and the historical SciPy/JAX candidate parity result are not publication validation; the publication PR programme remains authoritative.
<!-- END:IDENTITY -->

<!-- BEGIN:CORE_SUMMARY -->
## Active Runtime Core (NOT publication validation)

- **FLRW BBN baseline**: Runtime-regression baseline; gold and prior cross-code checks are not publication validation
- **Type I anisotropic BBN**: Runtime Type-I regression surface; finite-shear publication validation remains open
- **Weak rates (Born-Sirlin-FM)**: Runtime weak-rate regressions; not Stage-I collision validation
- **9-species nuclear network**: 31-reaction runtime network; BBN publication anchors remain open
- **Backends**: `scipy`

## Frozen Legacy Diagnostics

- **Teff spectral hardening legacy kernel**: deprecated legacy kernel; no public forward-solver runtime or promotion path

## Candidate Features (code works, partial validation, NOT publication-locked)

**Candidate-strong** (BBN-verified in documented regime):
- Tilted scalar BBN (documented-scope: principal-axis tilted candidate slice, v0 in [0, 1e-3], CL0-CL3; runtime-guarded)

**Candidate-layered** (multi-layer validation, partial BBN):
- Class A curved transport (documented-scope: 6-type; Type I exact characteristic option, curved cells reduced transport κ-cascade; CL0-CL3)
- Class B BBN (documented-scope: 6-label reduced-mask h-locked representative gold candidate)

**Diagnostic / Exploratory** (component-level only):
- AD diagnostic gradients (diagnostic: Diagnostic gradients (custom_vjp/FD parity 8.2e-8); Type I, {eta, tau_n}; not full differentiable solver)
- Tier-3 full-collision preflight surface (classical Boltzmann; no QKE) (diagnostic; AP-form unification (combine spectral_relaxation anisotropy stability with projected_physical grid scaling): RESOLVED in PR-T3B canonical milestone via collision_mode='ap_unified_preflight' (anisotropy spread ~7e-5, grid spread <1e-4))
- Inference / PE framework / model comparison (candidate: PE framework on Type I {Yp, D/H}; full-forward null scan locked; evidence exploratory)
<!-- END:CORE_SUMMARY -->

<!-- BEGIN:README_BACKENDS_HEADER -->
## Backends (from `backend_capabilities.py`)
<!-- END:README_BACKENDS_HEADER -->

<!-- BEGIN:BACKEND_TABLE -->
| Backend | Tier | CL | Surface class |
|---|---|---|---|
| `auto` | **canonical** | 0–3 | canonical |
| `scipy` | **canonical** | 0–3 | canonical |
<!-- END:BACKEND_TABLE -->

<!-- BEGIN:README_QUICKSTART -->
## Quick start

```python
from rabbit.inference.forward_likelihood import canonical_forward_solver

# Standard BBN (FLRW baseline)
pred = canonical_forward_solver(Sigma_H=0.0, backend="auto")
print(f"Yp = {pred.Yp:.6f}, D/H = {pred.DH:.2e}")

# Type I anisotropic BBN
pred = canonical_forward_solver(Sigma_H=0.05, backend="auto")

# JAX endpoint dispatch names are retired. Frozen low-level JAX
# component oracles remain non-dispatchable metadata only.
```
<!-- END:README_QUICKSTART -->

<!-- BEGIN:TEST_COUNTS -->
Overlapping marker subsets: **106 gold** BBN regression gates | 244 release smoke | `@production`: 443 total, 393 production-and-not-slow | build-env total: 2538 tests across 279 files
<!-- END:TEST_COUNTS -->

<!-- BEGIN:README_FOOTER -->
**Note**: Counts above are **build-environment metadata**, not portable invariants.
Optional dependencies (JAX, BlackJAX) affect test collection.
Run `make sync-counts` to update for your environment.
See `docs/RENDER_PROVENANCE.json` for exact build provenance.
<!-- END:README_FOOTER -->
