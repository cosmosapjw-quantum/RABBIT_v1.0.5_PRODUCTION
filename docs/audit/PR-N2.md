# PR-N2 Audit — Non-LRS Characteristic Driver Integration

## Verdict

Pass, with candidate-scope caveats.

PR-N2 wires the non-LRS S² characteristic primitives into the JAX driver as an
explicit opt-in backend, `jax_characteristic_nonlrs`. The landed surface is
compact: it does **not** carry explicit ray-state ODE blocks. Instead it
reconstructs the generic-Type-I direction map, characteristic intensity shift,
and S² Jacobian analytically from the accumulated shear integrals `(S_+, S_-)`.

## What Landed

- `transport_mode="characteristic_nonlrs"` in `driver_typeI_char.py` and
  `driver_typeI.py`
- analytic `intensity_shift_nonlrs_jax(...)` and `jacobian_nonlrs_jax(...)`
- candidate backend capability `jax_characteristic_nonlrs`
- inference dispatch through `canonical_forward_solver(...)`
- registry-generated docs regenerated from the updated backend registry

## Static Adversarial Checks

- The non-LRS S² Jacobian matches `jax.jacfwd` of the forward map at
  `|ΔJ| < 1e-11`.
- On the exact LRS slice (`N_phi=1`, `Sigma_- = 0`), the non-LRS driver reduces
  to the LRS characteristic driver at `|ΔY_p| ≈ 1.1e-8`,
  `|ΔD/H| ≈ 1.2e-10` for the regression cell used in `tests/test_pr_n2_nonlrs_driver.py`.
- `N_phi=1` is now explicitly reserved for the LRS reduction slice; generic
  `Sigma_- != 0` runs require `N_phi >= 2`.
- The `Pi_-` kernel sign was corrected to match the generic linearized source
  convention.

## Dynamic Validation

Focused regression:

- `tests/test_pr_n1_nonlrs_primitives.py`
- `tests/test_pr_n2_nonlrs_driver.py`
- characteristic parity / tier-2 / registry / inference bundle

Result:

- `237 passed, 3 skipped`

## Scope Honesty

The new surface is **candidate**, not canonical:

- tier-1 only
- collisionless only
- explicit opt-in only
- not selected by `backend="auto"`

At small generic shear (`Sigma_+ = Sigma_- = 0.05`), the exact non-LRS
characteristic signal agrees in **sign** with the generic linearized PSTF
reference but is about `4x` larger in magnitude on the current locked cell.
That is recorded as a bounded candidate behavior, not promoted parity.
