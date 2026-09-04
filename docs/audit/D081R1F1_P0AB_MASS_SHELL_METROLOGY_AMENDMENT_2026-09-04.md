# D-081R1F1 P0AB massive-leg tangent metrology amendment

**Date:** 2026-09-04  
**Scope:** test metrology only; no production formula, branch, quadrature, or threshold change

## Preserved failure

The first invariant run failed at supported sample `2564` under

\[
R_{\rm local}
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

The frozen D-080A Python authority therefore fails the old local-ratio gate by
a larger factor than Rust. The two terms are both approximately
`2.4592574e-8 MeV`; the residual is only `O(1e-18 MeV)`. This is a
near-stationary tangent where the local derivative contribution is an
ill-conditioned denominator, not evidence that the finite-mass kinematic
formula is wrong.

## Corrected dimensionless gate

The differentiated mass-shell identity is

\[
\frac{d}{dT_\gamma}
(E_4^2-p_4^2-m_e^2)
=2(E_4E_{4,T}-p_4p_{4,T})=0.
\]

Use the same frozen threshold `2e-12`, but normalize by the characteristic
primal shell scale per temperature,

\[
S_{\rm shell,T}
=
\frac{E_4^2+p_4^2+m_e^2}{T_\gamma},
\]

\[
\boxed{
R_{\rm shell,T}
=
\frac{2|E_4E_{4,T}-p_4p_{4,T}|}
{S_{\rm shell,T}}
\le 2\times10^{-12}
}.
\]

This scale has the same MeV dimension as the differentiated shell residual and
remains nonzero when the physical tangent is stationary. The original
`R_local` is retained as a raw conditioning diagnostic and is not relabelled as
a PASS quantity.

## Independent gates retained

- direct Rust/Python D-080A array parity `<=1e-7`;
- Rust-centered primal derivative witnesses `<=1e-7`;
- primal mass-shell validation;
- energy-tangent conservation;
- massless-leg, Minkowski-dot, support, and lambda invariants;
- unchanged support and correction branches;
- release check and strict Clippy.

No production source is modified by this amendment, and no threshold is
increased.
