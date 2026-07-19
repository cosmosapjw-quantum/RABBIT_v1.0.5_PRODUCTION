"""JAX-native not-a-knot natural cubic spline interpolation.

Mirrors ``scipy.interpolate.interp1d(kind='cubic')`` /
``scipy.interpolate.CubicSpline(bc_type='not-a-knot')`` at
element-wise machine precision.  The not-a-knot boundary
condition forces the third derivative to be continuous at the
first and last interior nodes, which is scipy's default.

Used by the tier-3 collision preflight modules to evaluate
neutrino monopoles on off-grid Gauss-Legendre quadrature nodes
without the structural PCHIP-vs-cubic interpolant gap that
currently bounds detailed balance and energy conservation in
the bank-core preflight (see ``docs/audit/PR-T3C_preflight.md``).

The construction solves a small dense linear system for the
second derivatives at nodes; for the typical Laguerre grid
(``N_q ≤ 32``) the cost is negligible compared to a single
collision-kernel evaluation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@jax.jit
def _solve_not_a_knot_M(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Return second derivatives ``M_i`` at the input nodes.

    Solves the linear system ``A @ M = b`` where ``A`` is the
    not-a-knot natural-cubic-spline tridiagonal-with-boundary
    operator on ``x``.

    The interior rows enforce ``2nd-derivative continuity`` at
    each internal node:

        h_{i-1}/6 M_{i-1} + (h_{i-1}+h_i)/3 M_i + h_i/6 M_{i+1}
            = s_i - s_{i-1}

    where ``s_i = (y_{i+1} - y_i) / h_i``.

    The boundary rows enforce ``not-a-knot`` (3rd-derivative
    continuous at ``x_1`` and ``x_{n-1}``):

        -h_1     M_0   + (h_0 + h_1) M_1   - h_0     M_2     = 0
        -h_{n-1} M_{n-2} + (h_{n-2} + h_{n-1}) M_{n-1} - h_{n-2} M_n = 0

    The system is solved via a dense ``jnp.linalg.solve``; ``n``
    is small (Laguerre grids ``≤ 32``), so the dense cost is
    negligible.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    n = x.shape[0]
    h = x[1:] - x[:-1]                # (n-1,)
    s = (y[1:] - y[:-1]) / h          # (n-1,)

    # Build dense (n, n) tridiagonal-with-boundary matrix A
    A = jnp.zeros((n, n), dtype=jnp.float64)
    b = jnp.zeros(n, dtype=jnp.float64)

    # Boundary row 0 (not-a-knot at left)
    A = A.at[0, 0].set(-h[1])
    A = A.at[0, 1].set(h[0] + h[1])
    A = A.at[0, 2].set(-h[0])
    # b[0] = 0

    # Boundary row n-1 (not-a-knot at right)
    A = A.at[n - 1, n - 3].set(-h[n - 2])
    A = A.at[n - 1, n - 2].set(h[n - 3] + h[n - 2])
    A = A.at[n - 1, n - 1].set(-h[n - 3])
    # b[n-1] = 0

    # Interior rows i = 1, ..., n-2
    interior_idx = jnp.arange(1, n - 1)
    h_minus = h[:-1]   # h_{i-1} for i=1..n-2
    h_plus = h[1:]     # h_i for i=1..n-2

    def _set_interior(A_in, b_in):
        A_out = A_in
        b_out = b_in
        for i in range(1, n - 1):
            A_out = A_out.at[i, i - 1].set(h[i - 1] / 6.0)
            A_out = A_out.at[i, i].set((h[i - 1] + h[i]) / 3.0)
            A_out = A_out.at[i, i + 1].set(h[i] / 6.0)
            b_out = b_out.at[i].set(s[i] - s[i - 1])
        return A_out, b_out

    A, b = _set_interior(A, b)

    M = jnp.linalg.solve(A, b)
    return M


def _eval_cubic_segment(
    x: jnp.ndarray,
    y: jnp.ndarray,
    M: jnp.ndarray,
    x_query: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the cubic-spline polynomial at ``x_query`` using
    pre-computed second derivatives ``M`` at the nodes.

    Standard formulation (Numerical Recipes §3.3):

        p_i(x) = A * y_i + B * y_{i+1} + (1/6)(A^3 - A) h_i^2 M_i
                 + (1/6)(B^3 - B) h_i^2 M_{i+1}

    where ``A = (x_{i+1} - x) / h_i``, ``B = 1 - A``,
    ``h_i = x_{i+1} - x_i``, on the segment ``i`` containing ``x``.

    For ``x_query`` outside ``[x[0], x[-1]]`` we clamp to the
    nearest segment (matching ``interp1d``'s default extrapolation
    via the boundary segment polynomial).
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    M = jnp.asarray(M, dtype=jnp.float64)
    x_query = jnp.asarray(x_query, dtype=jnp.float64)

    # Locate the bracketing segment: idx in [0, n-2].
    idx = jnp.clip(
        jnp.searchsorted(x, x_query, side="right") - 1,
        0,
        x.shape[0] - 2,
    )
    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    M0 = M[idx]
    M1 = M[idx + 1]
    h_seg = x1 - x0
    h_seg = jnp.where(h_seg == 0.0, 1.0, h_seg)  # safety
    A_coef = (x1 - x_query) / h_seg
    B_coef = 1.0 - A_coef
    return (
        A_coef * y0
        + B_coef * y1
        + ((A_coef**3 - A_coef) * M0 + (B_coef**3 - B_coef) * M1) * (h_seg**2) / 6.0
    )


def cubic_spline_eval(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_query: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the not-a-knot natural cubic spline through
    ``(x, y)`` at ``x_query``.

    This is the JAX-native equivalent of
    ``scipy.interpolate.interp1d(x, y, kind='cubic')(x_query)``.

    Parameters
    ----------
    x : (n,) float array; strictly increasing.
    y : (n,) float array.
    x_query : float scalar or 1D array; query points.

    Returns
    -------
    Spline values at ``x_query``, same shape as ``x_query``.
    """
    M = _solve_not_a_knot_M(x, y)
    return _eval_cubic_segment(x, y, M, x_query)


def cubic_spline_eval_with_clamp(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_query: jnp.ndarray,
    fill_low: float | jnp.ndarray,
    fill_high: float | jnp.ndarray,
) -> jnp.ndarray:
    """``cubic_spline_eval`` with scipy-style boundary clamping.

    Inside ``[x[0], x[-1]]`` the not-a-knot cubic spline is
    evaluated.  Outside the range the result is clamped to
    ``fill_low`` (below ``x[0]``) or ``fill_high`` (above
    ``x[-1]``), matching the semantics of
    ``scipy.interpolate.interp1d(x, y, kind='cubic',
    bounds_error=False, fill_value=(fill_low, fill_high))``.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    x_query = jnp.asarray(x_query, dtype=jnp.float64)
    spline_value = cubic_spline_eval(x, y, x_query)
    fill_low_arr = jnp.asarray(fill_low, dtype=jnp.float64)
    fill_high_arr = jnp.asarray(fill_high, dtype=jnp.float64)
    out = jnp.where(x_query < x[0], fill_low_arr, spline_value)
    out = jnp.where(x_query > x[-1], fill_high_arr, out)
    return out


__all__ = ["cubic_spline_eval", "cubic_spline_eval_with_clamp"]
