# D-081R1F1 P0AB invariant-reference metrology amendment

**Date:** 2026-09-04  
**Scope:** test-contract correction only; no production formula, quadrature,
support predicate, branch policy, or numerical-physics tolerance change

## Preserved failures and independent diagnosis

The amended P0AB GREEN lane had one remaining failure:

```text
Rust invariant:
d23_T + d24_T - d12_T

sample:
27028

Rust characteristic-scale ratio:
3.46217020326870728e-12

frozen cap:
2.0e-12
```

A read-only deterministic D-080A Python diagnostic was then executed before
this amendment. It found:

```text
Python at the same sample:
3.38258008365333466e-12

Python maximum d12_T-d34_T ratio:
8.08613644371259742e-12

Python maximum d23_T+d24_T-d12_T ratio:
2.00746092203014612e-11
```

The exact D-080A array oracle is deterministic with SHA-256

```text
b145b898d4e2160c2fd72c3abc5830f7afaf85b03511b5a0b8ce0b4426c9a32d
```

and the diagnostic workflow is `33886256124`.

Thus the universal `2e-12` hard gate is outside the numerical image of the
frozen Python authority for two cancellation-sensitive Minkowski-dot
identities. The formulas are not thereby invalid. The identities remain exact
in real arithmetic:

\[
(d_{12})_T-(d_{34})_T=0,
\]

\[
(d_{23})_T+(d_{24})_T-(d_{12})_T=0.
\]

For the second identity the undifferentiated relation is

\[
d_{23}+d_{24}=d_{12}+m_e^2,
\]

and \(m_e\) is constant.

## Contract correction

The old universal hard cap is **retired**, not widened, for the two
cancellation-sensitive dot-product identities. Raw local and
characteristic-scale residuals remain recorded.

The stable identities retain the unchanged hard cap:

```text
energy tangent conservation
massless outgoing shell
massive outgoing shell
d13_T + d14_T - d12_T

cap:
2e-12
```

The cancellation-sensitive identities use the frozen Python authority and the
exact linear-functional propagation bound.

Let

\[
I_R=L(x_R),\qquad I_P=L(x_P),
\]

where \(L\) is one of the linear functionals

\[
L_{12,34}(x)=x_{12}-x_{34},
\]

\[
L_{23,24,12}(x)=x_{23}+x_{24}-x_{12}.
\]

Then

\[
\boxed{
|I_R-I_P|
\le
\sum_j |x_{R,j}-x_{P,j}|+\eta_{\rm eval}
}
\]

with a prospective binary64 evaluation allowance

\[
\eta_{\rm eval}
=
128\,\epsilon_{\rm mach}
\left(
\sum_j|x_{R,j}|
+
\sum_j|x_{P,j}|
+
S
\right).
\]

The factor 128 is a conservative evaluation budget for the final stored-value
linear functional; it does not cover or excuse errors in the tangent arrays.
Those arrays remain independently gated by the direct D-080A cross-language
test at `1e-7` and by the Rust-centered primal derivative witness.

For the massive-shell product identity, the corresponding product-propagation
bound is retained even though that identity remains under the `2e-12` stable
cap.

## Falsification

The correction is admissible only if all of the following hold:

1. direct D-080A base/tangent array parity passes;
2. the four stable identities pass `2e-12` in both Rust and Python;
3. the two sensitive identities satisfy the stored-array propagation bound;
4. raw Rust and Python residual maxima are printed;
5. a one-percent massless-leg mutation remains killed by the stable gate;
6. a one-percent `d24_T` mutation at the most sensitive supported sample
   violates the unmutated propagation envelope;
7. support and correction branches remain identical;
8. release check and strict Clippy pass.

This amendment does not claim bitwise invariant zeros. It establishes that the
Rust residual is no larger than the independently frozen Python residual plus
the error propagated from the directly measured cross-language tangent-array
differences.

## Classification

```text
FORMULA DEFECT:
not found

OLD UNIVERSAL INVARIANT GATE:
invalid for two frozen-authority outputs

THRESHOLD INCREASE:
none

GATE-DOMAIN CORRECTION:
yes

PRODUCTION SOURCE CHANGE:
none in this amendment

CLAIM CEILING:
P0A/P0B fixed-branch primitive validation only
```
