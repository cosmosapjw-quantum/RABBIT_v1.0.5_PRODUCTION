# D-081R1F1 P0A/P0B margin-test amendment

**Date:** 2026-09-04  
**Failed GREEN workflow:** `33879599975`  
**Status:** `TEST_ORACLE_OPERATION_GRAPH_CORRECTION / NO_THRESHOLD_CHANGE`

## Preserved result

The first adversarial GREEN run passed the exact finite-Simpson EOS derivative test and failed before direct Python-oracle execution at the internal normalized support-margin assertion.

The production margin is computed from the exact `invariant_s` value used by the admitted support predicate:

\[
s=(E_1+E_2)^2-|\mathbf p_1+\mathbf p_2|^2.
\]

The failed test reconstructed an algebraically equivalent value from the stored Minkowski product,

\[
s=m_e^2+2d_{12}.
\]

These expressions are equal in exact arithmetic but do not share the same binary64 operation graph. At large electron momentum the direct total-energy/total-momentum subtraction is cancellation-sensitive, so requiring the two reconstructions to agree at `64 eps` does not test the declared branch-margin definition.

## Amendment

The threshold remains exactly `64*f64::EPSILON`. The test oracle is corrected to reproduce the same direct `s` and Källén operation order used by the support predicate, using the frozen incoming-angle rule and batch flattening order. The separate D-080A direct-oracle test remains unchanged at its prospectively frozen `1e-7` cross-language gate.

No production source, physics coefficient, quadrature, support predicate, branch policy, retained state, or downstream JVP is changed by this amendment.

## Classification

```text
FIRST GREEN FAILURE:
FORMULA/TEST-ORACLE OPERATION-GRAPH MISMATCH

EOS DISCRETE DERIVATIVE:
PASS IN FAILED RUN

THRESHOLD RELAXATION:
NO

PRODUCTION SEMANTICS CHANGE:
NO
```
