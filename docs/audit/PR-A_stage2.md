# PR-A Stage 2 — External Verification

## Web search results

The requested external searches were run.

1. Query: `Bianchi Type I BBN characteristic ray angular Jacobian`
   Result quality: low. Returned general Bianchi I / CMB / recombination
   papers, but no direct external treatment of the exact RABBIT ray
   quadrature Jacobian. Nothing contradictory was found.
2. Query: `dJ/dN = 3 Sigma (1-3mu^2) J Bianchi`
   Result quality: low. No clean published hit with the exact ODE form
   surfaced in search results. Closest hits were general Bianchi I
   geodesic / dynamical-systems references.
3. Query: `Wainwright Ellis Bianchi Type I geodesic direction cosine mu 3 Sigma mu (1-mu^2)`
   Result quality: low-to-moderate. Search results pointed to general
   Bianchi dynamical-systems literature, but not to a directly quotable
   public page for this exact transport-weight formula.

Representative non-contradictory context returned by search:

- Phys. Rev. D 112, 023553 (2025), "Constraining the locally rotationally
  symmetric Bianchi type I model with self-consistent recombination
  history and observables" — general LRS Bianchi I null-geodesic context.
- Physics Reports 775-777 (2018), "Dynamical systems applied to
  cosmology" — general pointer back to Wainwright-Ellis style Bianchi
  dynamical-systems treatments.

Verdict: **no contradictory external formula was found**. The web was
low-yield for this exact quantity, so the sign convention must be fixed
internally from the repo derivation and numerical parity.

## Independent calculus check from eq (48)

Starting from the report's direction equation

`dμ/dN = 3 Σ μ (1-μ²)`,

we obtain

- `d/dN ln|μ| = 3 Σ (1-μ²)`
- `d/dN ln(1-μ²) = -2 μ (dμ/dN)/(1-μ²) = -6 Σ μ²`

Therefore

`d/dN ln(|μ|(1-μ²)) = 3 Σ (1-3μ²)`.

This matches the carried ODE for the production-code Jacobian,

`dJ/dN = 3 Σ (1-3μ²) J`,

so the forward Jacobian must satisfy

`J ∝ |μ|(1-μ²)`.

Matching the initial condition `J(0)=1` gives

`J = |μ|(1-μ²) / (|μ₀|(1-μ₀²))`.

Verdict: the exponent/sign implied by the ODE is the forward-measure
form used in `jacobian_jax`, not the inverse form suggested by the raw
OCR text of eq (51).
