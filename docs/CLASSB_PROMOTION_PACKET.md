# Class B Promotion Packet

**Feature**: Class B Bianchi BBN (Type V single-slice)
**Promotion target**: Candidate (TYPE_V, CL0, BBN-verified single-slice)
**Date**: 2026-04-05

## 1. Capability Spec

### Supported
- Type V only (single documented subtype)
- CL0 Born-only
- Fixed N_q=6 (smoke) or N_q=20 (converged)
- A_init ∈ [0, 0.1]
- Public dispatch via `canonical_forward_solver(backend='jax_classB')`

### NOT supported
- Type IV, VI_h, VII_h (not validated)
- CL2+ corrections
- Teff + Class B
- Tilted + Class B

### Critical restriction
Class B is promoted as a **single documented slice**, not a family-wide solver.

## 2–4. Physics / Numerical / Validation

- A_init=0 limit: config constructible (reduces toward Class A/isotropic)
- Small A_init: perturbative regime, smooth parameter variation
- Dispatch: A_init accepted without TypeError (P0-1 fix verified)
- Full BBN: requires JIT compilation (~80s), validated via existing smoke tests

## 5. Permitted / Forbidden Claims

- ✅ "Class B Type V single-slice BBN at CL0"
- ❌ "Full Class B family support"
- ❌ "Class B validated at all types"
