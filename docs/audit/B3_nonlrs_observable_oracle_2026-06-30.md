# B3-lite — non-LRS characteristic observables certified exact at Σ₋ ≠ 0

**Date:** 2026-06-30
**Purpose:** PR#15 established the LRS characteristic-ray path as the exact shear-faithful reference.
This extends that exact-reference status into the **non-LRS** (biaxial, Σ₋ ≠ 0) sector by closing the
one validation hole that kept the non-LRS observable path at "candidate" scope: the angular
extractors that feed the RHS stresses were pinned only in the LRS reduction (Σ₋ → 0).

**Method:** an independent first-principles numpy oracle. The diagonal Type-I stretch
`A = diag(e^{-(S₊+√3S₋)}, e^{-(S₊−√3S₋)}, e^{2S₊})` acts on the initial direction `n₀`; the
transported FD argument rescales `q → q·‖A n₀‖` (so `e^{2I} = ‖A n₀‖`) and the S² push-forward
Jacobian is `J = det(A)/‖A n₀‖³ = ‖A n₀‖⁻³` (det A = 1, trace-free shear). Computing `‖A n₀‖`
directly from the geometry (a Euclidean `√Σ`, *not* the production log-domain `logaddexp` helpers)
gives a structurally independent check of the closed-form `I`, `J`, and the monopole/stress assembly.

## Result — exact to machine precision at Σ₋ ≠ 0

At (S₊, S₋) ∈ {(0.15, 0.08), (0.20, −0.05)} on a 12×16 S² grid:

| check | production vs first-principles oracle |
|-------|---------------------------------------|
| `e^{2I}` = ‖A n₀‖ | rel. err ~2e-16 |
| `J` = ‖A n₀‖⁻³ | rel. err ~7e-16 |
| `extract_monopole_S2` assembly | abs. err ~1e-16 |
| `extract_stress_minus_S2` (m=2) vs independent (1−μ²)cos2φ shape | rel. err ~1e-16 |
| `extract_stress_plus_S2` (m=0) vs independent P₂(μ) shape | rel. err ~2e-16 |

Plus the physical/consistency gates:
- **LRS reduction:** at Σ₋ = 0 the non-LRS monopole reproduces the trusted LRS `extract_monopole_jax`
  (PR#15's reference) to ~1e-16, for N_φ ∈ {1, 4, 16} — the azimuthal quadrature correctly integrates
  out the absent m=2 structure, not merely ignores φ. The m=2 stress vanishes (~6e-17).
- **Σ₋ signal:** Σ₋ moves the energy-density anisotropy by ~53% (δρ 0.230 → 0.352) and the m=2 stress
  flips sign with sign(Σ₋) — a stub ignoring the minus sector fails decisively.
- **Quadrature convergence:** the integrated m=2 stress at (12,16) vs (24,32) agrees to 6.6e-4.

## Scope (honest)

Validates the S² quadrature assembly + the I/J closed forms + the measure push-forward at Σ₋ ≠ 0. It
does **not** re-validate the exact-diagonal-Type-I physics (the ODE test `test_pr_n1_nonlrs_primitives`
covers that) nor the absolute `−(√3/2)` m=2 normalization convention (anchored by the AP62/AP63
LRS-reduction rate tests); the stress oracle independently supplies only the angular *shape*.

## Scope update (auto-managed)

- **The non-LRS characteristic observable path is now exact-reference-grade at Σ₋ ≠ 0** — the one hole
  blocking promotion from "candidate" is closed at the observable level. Combined with PR#15, the
  characteristic-ray path is the shear-faithful reference in **both** LRS and non-LRS sectors.
- **B4 (wiring collisions into the JAX characteristic path) remains a multi-PR effort**, not started:
  it needs a jnp twin of the numpy-only `PhysicalCollisionOperator` (`projected_operator.py:228-363`)
  + a jnp `teff_collision_bridge` + JAX char-RHS collision dispatch. The collisional reference today
  is the SciPy characteristic driver (`forward_likelihood.py:2243-2247` blocks JAX-char collisions).
- **Regression gate:** `tests/test_nonlrs_char_observable_oracle.py`.
