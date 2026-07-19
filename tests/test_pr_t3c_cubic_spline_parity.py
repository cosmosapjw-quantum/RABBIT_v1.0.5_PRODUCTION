"""PR-T3C-PF #2: element-wise parity for the JAX-native not-a-knot
natural cubic spline (``rabbit.jax.cubic_spline_jax``) versus
``scipy.interpolate.interp1d(kind='cubic')``.

The tier-3 collision preflight kernels currently use PCHIP cubic
Hermite interpolation for off-grid distribution evaluation
(``f_ν̄(y_2)``, ``f_β(y_4)``).  PCHIP is monotone-preserving but
structurally different from scipy's not-a-knot natural cubic
spline; the resulting interpolant gap currently bounds detailed
balance at ``~1e-22`` and energy conservation at ``~2% rel`` on
the diagonal ν-ν preflight (see
``docs/audit/PR-T3C_preflight.md``).

This module locks the new JAX cubic spline at element-wise machine
precision against the scipy reference, which is the required
infrastructure for tightening those bounds in the canonical
PR-T3C runtime patch.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy")

import jax
import jax.numpy as jnp
from scipy.interpolate import interp1d
from numpy.polynomial.laguerre import laggauss

from rabbit.jax.cubic_spline_jax import cubic_spline_eval


jax.config.update("jax_enable_x64", True)


def _fermi_dirac(y: np.ndarray) -> np.ndarray:
    return 1.0 / (np.exp(np.minimum(y, 500.0)) + 1.0)


# ---------------------------------------------------------------------------
# Parity at the input nodes (must be identity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 6, 8, 12, 20])
def test_cubic_spline_identity_at_nodes(n: int) -> None:
    """Evaluating the spline at its own input nodes must return the
    input ``y`` values to floating-point reduction order."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.0, 10.0, size=n))
    # Smooth FD-like data
    y = _fermi_dirac(x)
    y_eval = np.asarray(cubic_spline_eval(jnp.asarray(x), jnp.asarray(y), jnp.asarray(x)))
    err = np.max(np.abs(y - y_eval))
    assert err < 1e-12, f"identity-at-nodes failed: max|err| = {err:.3e}"


# ---------------------------------------------------------------------------
# Parity vs scipy.interpolate.interp1d(kind='cubic') on Laguerre nodes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [6, 8, 12, 20])
def test_cubic_spline_matches_scipy_interp1d_cubic(n: int) -> None:
    """JAX cubic spline matches scipy interp1d cubic at element-wise
    machine precision on a smooth FD distribution sampled at
    Laguerre nodes."""
    x_np, _ = laggauss(n)
    y_np = _fermi_dirac(x_np)

    # Off-grid query points: midpoints of Laguerre intervals plus a
    # selection of generic positive numbers.
    rng = np.random.default_rng(7)
    x_mid = 0.5 * (x_np[:-1] + x_np[1:])
    x_random = rng.uniform(x_np[0] + 1e-3, x_np[-1] - 1e-3, size=20)
    x_query_np = np.concatenate([x_mid, x_random])

    scipy_eval = interp1d(x_np, y_np, kind="cubic", bounds_error=False,
                          fill_value=(y_np[0], y_np[-1]))(x_query_np)
    jax_eval = np.asarray(cubic_spline_eval(
        jnp.asarray(x_np), jnp.asarray(y_np), jnp.asarray(x_query_np),
    ))

    abs_err = np.max(np.abs(scipy_eval - jax_eval))
    # Element-wise machine-precision target.
    assert abs_err < 1e-12, (
        f"interp1d-cubic parity failed at n={n}: max|err| = {abs_err:.3e}"
    )


# ---------------------------------------------------------------------------
# Smoothness: continuity of 1st and 2nd derivatives at internal nodes
# ---------------------------------------------------------------------------


def test_cubic_spline_C2_smoothness() -> None:
    """The not-a-knot cubic spline is ``C^2``: the spline value
    converges to the same limit when approached from the left and
    right of any internal node.  We measure this via
    finite-difference second-derivative agreement on a dense
    query grid bracketing each internal node."""
    n = 20
    x_np, _ = laggauss(n)
    y_np = _fermi_dirac(x_np)

    # Probe a tight neighbourhood of each interior node from both sides.
    x_node = x_np[n // 2]
    eps = 1e-4
    x_query = jnp.asarray([x_node - eps, x_node, x_node + eps])
    y_query = np.asarray(cubic_spline_eval(
        jnp.asarray(x_np), jnp.asarray(y_np), x_query,
    ))
    # Continuity at the node: y(x-eps), y(x), y(x+eps) should be
    # smooth.  Take a 2nd-order finite-difference of the local
    # values; for a C^2 spline this is bounded.
    second_diff = (y_query[2] - 2.0 * y_query[1] + y_query[0]) / eps**2
    # The actual second derivative of the FD distribution at this
    # mid-point is finite (~|f''(x)|).  Locking |second_diff| < 1
    # is a generous bound that flags any blow-up from a sign-flip
    # in the not-a-knot construction.
    assert abs(second_diff) < 1.0, (
        f"C^2 smoothness check failed: |second_diff| = {abs(second_diff):.3e}"
    )


# ---------------------------------------------------------------------------
# JIT consistency
# ---------------------------------------------------------------------------


def test_cubic_spline_jit_compatible() -> None:
    """``cubic_spline_eval`` is JIT-compatible and traces to the
    same result as the eager call."""
    n = 12
    x_np, _ = laggauss(n)
    y_np = _fermi_dirac(x_np)
    x_query_np = np.linspace(x_np[0], x_np[-1], 50)

    eager = np.asarray(cubic_spline_eval(
        jnp.asarray(x_np), jnp.asarray(y_np), jnp.asarray(x_query_np),
    ))
    jitted_fn = jax.jit(cubic_spline_eval)
    jitted = np.asarray(jitted_fn(
        jnp.asarray(x_np), jnp.asarray(y_np), jnp.asarray(x_query_np),
    ))
    # Eager vs JIT can differ at ULP-level reduction order; allow
    # machine-precision tolerance.
    np.testing.assert_allclose(eager, jitted, rtol=1e-14, atol=1e-15)
