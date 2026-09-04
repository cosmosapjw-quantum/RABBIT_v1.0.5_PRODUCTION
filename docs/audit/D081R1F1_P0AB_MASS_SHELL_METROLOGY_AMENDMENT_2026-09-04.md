# D-081R1F1 P0AB near-stationary tangent-invariant metrology amendment

**Date:** 2026-09-04  
**Scope:** test metrology only; no production formula, branch, quadrature, or threshold change

## Preserved failures

The first invariant run failed at supported sample `2564` under the local
massive-shell tangent ratio

\[
R_{\rm local}^{(4)}
=
\frac{|E_4 E_{4,T}-p_4 p_{4,T}|}
{|E_4 E_{4,T}|+|p_4 p_{4,T}|}
\le 2\times10^{-12}.
\]

The diagnostic replay recorded

```text
Rust local ratio:   6.90998553394061958e-12
Python local ratio: 2.83593356793396332e-11
Rust primal shell ratio:   1.09861278171695725e-16
Python primal shell ratio: 1.09861278171695725e-16
Rust energy-tangent residual: 0
```

After correcting only that denominator, the next run failed the analogous
massless-leg local ratio at sample `2876`. This establishes a common test-domain
problem: when a physical tangent is nearly stationary, the sum of the local
derivative terms approaches zero and is not a stable scale for an identity
whose underlying primal kinematic quantities remain O(MeV) or O(MeV^2).

The frozen D-080A Python authority itself fails the original massive local gate
by more than Rust. The production formulas are therefore not identified as the
cause of either failure.

## Corrected characteristic scales

Every differentiated invariant keeps the frozen cap

\[
R\le 2\times10^{-12},
\]

but is normalized by the corresponding primal invariant scale per unit
temperature. The old local contribution ratios remain raw conditioning
diagnostics.

### Energy conservation

\[
E_{2,T}-E_{3,T}-E_{4,T}=0,
\]

\[
S_{E,T}=\frac{|E_2|+|E_3|+|E_4|}{T_\gamma},
\qquad
R_{E,T}=\frac{|E_{2,T}-E_{3,T}-E_{4,T}|}{S_{E,T}}.
\]

### Massless outgoing leg

\[
E_{3,T}-|p_3|_T=0,
\]

\[
S_{3,T}=\frac{|E_3|+|p_3|}{T_\gamma},
\qquad
R_{3,T}=\frac{|E_{3,T}-|p_3|_T|}{S_{3,T}}.
\]

### Massive outgoing leg

\[
\frac{d}{dT_\gamma}(E_4^2-p_4^2-m_e^2)
=2(E_4E_{4,T}-p_4p_{4,T})=0,
\]

\[
S_{4,T}=\frac{E_4^2+p_4^2+m_e^2}{T_\gamma},
\]

\[
R_{4,T}=\frac{2|E_4E_{4,T}-p_4p_{4,T}|}{S_{4,T}}.
\]

### Minkowski-product identities

For an identity \(d_a+d_b-d_c=0\), use

\[
S_{d,T}=\frac{|d_a|+|d_b|+|d_c|}{T_\gamma},
\qquad
R_{d,T}=\frac{|d_{a,T}+d_{b,T}-d_{c,T}|}{S_{d,T}}.
\]

For \(d_{12}-d_{34}=0\), omit the third term in the scale.

These scales have the same dimensions as the differentiated residuals and do
not collapse at a stationary tangent. They are not fitted to the observed Rust
outputs.

## Independent gates retained

- direct Rust/Python D-080A array parity `<=1e-7`;
- Rust-centered primal derivative witnesses `<=1e-7`;
- primal mass-shell validation;
- raw local tangent-invariant ratios, recorded but not used as primary gates;
- exact support and matrix-correction branch identity;
- normalized branch-margin parity;
- release check and strict Clippy.

A one-percent mutation at the largest nonzero tangent component must still be
killed by the corrected invariant metric. No production source is modified by
this amendment, and no numerical threshold is increased.
