# B.HR re-scoped — the even-ladder transport is already hyperbolic in q (CFL has no target)

**Date:** 2026-06-30
**Purpose:** The plan's **B.HR** item called for a *globally hyperbolic regularization (Cai–Fan–Li)
operator-projection in front of the transport derivative* of the LRS neutrino moment system,
"required for B2/B4/B5." A 3-angle code+spec mapping + a measurement shows **B.HR as named is
mis-targeted**: there is no Grad-type transport-hyperbolicity loss to repair. This records the
re-scoping and the diagnostic that replaces it.

---

## Why Cai–Fan–Li does not apply here

The collisionless even-ell shear-advection RHS
(`transport/typeI_even_ladder_hierarchy.py:43`, faithful to
`neutrino_decoupling_PSTF_HM_generalization_ko.md` lines 294–312) is

    RHS = Sigma_H [ M_stream @ (q d/dq) F + 1.5 M_angle @ F ],

with `F = (F_0, F_2, …, F_lmax)` and `M_stream` / `M_angle` **constant** pentadiagonal-in-ℓ matrices
(the A/B/C and tilde Legendre-recurrence coefficients).

1. **No spatial flux Jacobian.** Cai–Fan–Li repairs a Grad system whose *spatial* flux Jacobian
   `dF/dU` loses real eigenvalues near equilibrium. Bianchi I is spatially homogeneous
   (`D_a = ω_ab = 0`) — there is no `∂_x` transport term, so there is no spatial flux Jacobian and
   nothing for a CFL projection to act on.
2. **The only derivative is `q d/dq`, with a constant coupling matrix.** Well-posedness reduces to a
   STATIC question: does `M_stream` (the matrix multiplying the derivative) have a real,
   diagonalizable spectrum (real characteristic q-advection speeds)? That is a one-time check per
   `ell_max`, not a runtime regularization.
3. **The hierarchy is closed by ℓ_max truncation (Ma–Bertschinger), not a Grad–Hermite closure.**
   The residual concern is truncation leakage (`D_ℓ = 4 + (ℓ-2)/2`, `validation/transport_convergence.py`),
   which CFL does not address.

## Measurement — M_stream is real-spectrum and well-conditioned

`src/rabbit/transport/even_ladder_wellposedness.py` assembles `M_stream` / `M_angle` from the exact
live coefficients (a faithfulness test pins matrix·vector == the live RHS to 1e-12) and computes the
spectrum:

| ℓ_max | n | M_stream spectrum | cond(eigenbasis) | hyperbolic in q? |
|------:|--:|-------------------|-----------------:|:----------------:|
| 2 | 2 | real, ∈ [−0.33, 0.61] | 2.17 | ✅ |
| 4 | 3 | real, ∈ [−0.41, 0.80] | 2.87 | ✅ |
| 6 | 4 | real, ∈ [−0.45, 0.88] | 3.46 | ✅ |
| 8 | 5 | real, ∈ [−0.47, 0.92] | 3.96 | ✅ |
| 12 | 7 | real, ∈ [−0.48, 0.96] | 4.83 | ✅ |

**The even-ladder q-advection is already hyperbolic** for ℓ_max well beyond the production span. No
regularization is needed; a CFL module would be dead code with no failing test to justify it.

## The M_angle trap (disambiguated, not "fixed")

`M_angle` — the **zeroth-order** algebraic source (multiplies no derivative) — is **not** a
hyperbolicity object: hyperbolicity is a property of the principal (highest-derivative) symbol,
which is `M_stream ⊗ (q d/dq)` only, so the gate keys exclusively on `M_stream`.

**Audit correction (2026-06-30).** The earlier wording called `M_angle` "harmless bounded
oscillation" — that is wrong, and under-stated the source. Its spectrum carries a **measured,
strictly positive real-part growth rate**: `max Re(eig) ≈ +0.27` across ℓ_max = 2…12 (an `F_ℓ`
eigenmode then grows in the RHS at rate `1.5 Σ_H · Re`), with a single exact zero mode from the
source-free `F_0` row and **no decaying mode**.
At ℓ_max = 2 the spectrum is **purely real** (max |Im| = 0) — pure growth, zero oscillation; the
oscillatory imaginary parts appear only at ℓ_max ≥ 4 and grow with ℓ_max (|Im| up to ~3.9 at
ℓ_max = 12). So `M_angle` is a **genuine shear-driven growth source**, not a harmless oscillation;
its amplitude stays bounded in practice because the band-limited logit realizability lock caps the
`A_modes` (B5/AP62), **not** because the source is non-growing. The diagnostic
`angle_source_eigenvalues` exposes this spectrum, and
`test_angle_source_is_a_growth_source_not_harmless_oscillation` pins BOTH facts — the growth (real
part) and the oscillation (imag part) — so the source is never mistaken for a transport
ill-posedness **and** never mislabelled as non-growing.

## Scope update (auto-managed)

- **B.HR (CFL regularization) — CLOSED / re-scoped.** Replaced by the reusable hyperbolicity gate
  `even_ladder_wellposedness.q_advection_is_hyperbolic`, which B2/B4/B5 should call as a precondition
  when extending the even-ladder operator (e.g. new ℓ ranges or coefficient changes).
- **Correction to the plan dependency** "B.HR required for B2/B4/B5": B2/B4/B5 actually depend on
  (a) this q-advection real-spectrum property (now provided + gated) and (b) **B5's own M_N
  realizability** at strong shear (the band-limited logit, `augmented_pstf_distribution.py`) — the
  genuine strong-shear well-posedness mechanism. Neither is the Cai–Fan–Li construction.

*Caveat: the diagnostic applies to the Type-I / even-ladder operator. The curved Class A/B operators
use a scalar-κ proxy (`jax/rhs_classA.py`), not the exact m-decomposed hierarchy; the real-spectrum
claim is not over-generalized to them here.*
