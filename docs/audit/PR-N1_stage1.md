# PR-N1 Stage 1 — Internal Verification

## Claims checked

- `characteristic_rays_jax.py` is the LRS reduction anchor.
  Verdict: VERIFIED at
  [`src/rabbit/jax/characteristic_rays_jax.py`](../src/rabbit/jax/characteristic_rays_jax.py).
  The existing JAX LRS primitives expose the reference `μ(S)`,
  `J(S)`, `I(S)`, `Π_+`, and `f̃₀(q)` conventions.

- Generic Type I already carries a second shear amplitude and a second
  quadrupole channel on the PSTF side.
  Verdict: VERIFIED at
  [`src/rabbit/transport/typeI_hierarchy.py`](../src/rabbit/transport/typeI_hierarchy.py)
  and
  [`src/rabbit/transport/projectors.py`](../src/rabbit/transport/projectors.py).
  The hierarchy has explicit `Σ_-` sourcing and the projectors already
  expose `Pi_minus`.

- The current production characteristic driver is still LRS-only.
  Verdict: VERIFIED at
  [`src/rabbit/jax/driver_typeI_char.py`](../src/rabbit/jax/driver_typeI_char.py),
  where `JAXTypeICharConfig` still rejects `Sigma_H_minus != 0`.

## Internal consistency verdict

- Pure additive `src/rabbit/jax/characteristic_rays_nonlrs_jax.py`
  is the correct scope for PR-N1.
- No driver/index layout change is needed yet.
- The right regression targets are:
  LRS reduction, constant-shear forward-map ODE parity,
  `Pi_minus -> 0` at `Sigma_minus = 0`, and symmetry under x↔y exchange.
