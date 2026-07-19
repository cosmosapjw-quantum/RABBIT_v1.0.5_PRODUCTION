# PR-A Stage 3 — Derivation / Limit / Pole Audit

## 1. First-principles derivation

The production transport weight is defined as the forward Jacobian

`J = dμ/dμ₀`.

Start from the analytic direction map written in terms of

`X = μ²/(1-μ²) = X₀ e^{6S}`,  `X₀ = μ₀²/(1-μ₀²)`.

Differentiate with respect to `μ` and `μ₀`:

- `dX/dμ = 2μ / (1-μ²)^2`
- `dX₀/dμ₀ = 2μ₀ / (1-μ₀²)^2`
- for fixed `S`, `dX/dμ₀ = e^{6S} dX₀/dμ₀`

Therefore

`dμ/dμ₀ = (dμ/dX)(dX/dμ₀)`

`= ((1-μ²)^2 / (2μ)) * e^{6S} * (2μ₀ / (1-μ₀²)^2)`

`= e^{6S} μ₀ (1-μ²)^2 / (μ (1-μ₀²)^2)`.

Now use `e^{6S} = X/X₀ = μ²(1-μ₀²) / (μ₀²(1-μ²))` to simplify:

`dμ/dμ₀ = μ (1-μ²) / (μ₀ (1-μ₀²))`.

Taking absolute values on the sign-preserving `μ<0` and `μ>0` branches
gives the exact production formula implemented in JAX:

`J = |μ|(1-μ²) / (|μ₀|(1-μ₀²))`.

Check against the carried ODE:

`d/dN ln J = d/dN ln|μ| + d/dN ln(1-μ²)`

`= 3Σ(1-μ²) - 6Σμ²`

`= 3Σ(1-3μ²)`,

so

`dJ/dN = 3Σ(1-3μ²)J`.

This matches eq (55) exactly.

## 2. Dimensional analysis

- `J` is an angular Jacobian, hence dimensionless.
- `S = ∫Σ dN` is dimensionless.
- `μ`, `μ₀` are direction cosines, hence dimensionless.

Verdict: the implemented formula is dimensionally consistent.

## 3. Limit checks

- At `S = 0`, `μ = μ₀`, so `J = 1`. This matches the initial condition.
- As `S → +∞`, rays compress toward `|μ| → 1`; then
  `1-μ² → 0` and `J → +∞`, representing angular compression into the
  anisotropy axis. Physical and expected.

## 4. NaN / pole audit

`1-μ² = 0` only at `|μ| = 1`. The Gauss-Legendre ray grid never places a
node at the pole, so the singularity is not sampled. The implementation
still floors denominators at `1e-30` to keep the JAX graph safe.

## 5. Sign audit

For constant positive `Σ`, the directly integrated ODE solution for
`J(N)` stays strictly positive on every ray. The analytic regression
test `tests/test_pr_a_analytic_jacobian.py` matches that numerical ODE
solution at max absolute error `< 1e-9`; no negative Jacobian was
observed.

## Verdict

The post-PR-A JAX implementation is correct in the production-code
convention `J = dμ/dμ₀`. The report OCR around eq (51) is ambiguous, but
eq (55) plus direct numerical ODE parity fixes the sign convention
uniquely.
