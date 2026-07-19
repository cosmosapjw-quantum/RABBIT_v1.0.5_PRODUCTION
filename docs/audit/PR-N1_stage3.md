# PR-N1 Stage 3 — Self Verification / Derivation

## Forward-map derivation

In orthogonal Type I the integrated shear generator is diagonal in the
fixed Wainwright-Hsu frame:

`diag(S_+ + sqrt(3) S_-, S_+ - sqrt(3) S_-, -2 S_+)`.

Because these diagonal operators commute at all times, the path-ordered
exponential collapses to an axis-wise stretch:

- `x -> x exp(-(S_+ + sqrt(3) S_-))`
- `y -> y exp(-(S_+ - sqrt(3) S_-))`
- `z -> z exp( 2 S_+)`

Renormalising the stretched unit vector gives the exact non-LRS
direction map.  At `S_- = 0` this reduces immediately to the existing
LRS logistic map `X = X_0 exp(6 S_+)`.

## ODE cross-check

For constant shear, the unit-direction ODE is

`n_i' = (kappa - lambda_i) n_i`,
`kappa = sum_j lambda_j n_j^2`,

with `lambda = (Sigma_+ + sqrt(3) Sigma_-,
Sigma_+ - sqrt(3) Sigma_-, -2 Sigma_+)`.

PR-N1 compares the analytic map against `solve_ivp` on three
representative constant-shear cases and locks the final direction
vector to absolute error `< 3e-11`.

## Symmetry audit

- `Sigma_- = 0` restores axial symmetry, so the `Pi_-` kernel must
  integrate to zero over the uniform midpoint `phi` grid.
- The exact x↔y exchange is `Sigma_- -> -Sigma_-` with
  `phi -> pi/2 - phi`.  Under that transformation `Pi_+` is invariant
  and `Pi_-` flips sign.

## Verdict

The delivered PR-N1 formulas are internally coherent, reduce to the
existing LRS primitives at `N_phi = 1`, and do not require any driver
or Jacobian changes yet.
