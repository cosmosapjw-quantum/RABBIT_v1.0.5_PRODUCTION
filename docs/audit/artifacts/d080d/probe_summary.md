# D-080D deterministic square static-Jacobian probe

- classification: `EXPLICIT_SQUARE_STATIC_JACOBIAN`
- explicit thermal matrix: `[26, 26]`
- thermal direction-1 best residual: `5.3256293759435533e-10`
- thermal direction-2 best residual: `2.0013236843808799e-10`
- weak-tail best residual: `1.9628771326375855e-10`
- exact elapsed-column norm: `0.0000000000000000e+00`
- maximum matrix-vs-direct-JVP residual: `2.0641343597740823e-15`

Residuals normalize spectral, photon-temperature, and elapsed-output blocks separately.
The full matrix is structurally singular because stored elapsed time is passive;
the associated BDF matrix identity is symbolic/static evidence only, not a solver claim.

- retained-state directional residual: `2.7851573363078990e-07`
- retained-state scope: directional only; no order-60 explicit matrix was assembled.
