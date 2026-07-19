# PR-N1 Stage 2 — External / Literature Verification

## Sources

- NIST DLMF, spherical harmonics definitions and orthogonality:
  https://dlmf.nist.gov/14.30
- DLMF explicit representation:
  https://dlmf.nist.gov/14.30.E3
- Lebedev-Laikov sphere quadrature reference:
  https://www.mathnet.ru/eng/dan3035
- Wainwright & Ellis chapter metadata for non-tilted class-A Bianchi
  models:
  https://www.cambridge.org/core/books/dynamical-systems-in-cosmology/bianchi-cosmologies-nontilted-class-a-models/72A05DB5C8B85C79379582C762090691

## What was checked

- The real `|m| = 2` harmonic used for the non-LRS stress channel is
  proportional to `sin^2(theta) cos(2 phi)`.
  Verdict: VERIFIED from the standard spherical-harmonic normalization
  in DLMF §14.30; PR-N1 uses the repository’s project convention
  `(sqrt(3)/2) sin^2(theta) cos(2 phi)`.

- A tensor-product `Gauss-Legendre(mu) x midpoint(phi)` grid is a
  legitimate temporary S^2 rule.
  Verdict: VERIFIED as a standard quadrature construction; Lebedev is
  still the deferred optimisation path, not the correctness baseline.

- External literature did not supply a more useful open-access closed
  formula for the exact orthogonal-Type-I direction map than the one
  already implied by the internal diagonal-shear basis.  That formula
  was therefore validated by direct ODE cross-check in Stage 3.

## External-verification verdict

No external source contradicted the additive PR-N1 design.  The only
correction to the original prompt is the symmetry statement: the exact
x↔y exchange associated with `Sigma_minus -> -Sigma_minus` is
`phi -> pi/2 - phi`, not `phi -> pi - phi`.
