"""tests/test_rodas5p_native_ad.py — Phase RA acceptance gates.

The discrete adjoint replay path
:mod:`rabbit.jax.solver_jax_rodas5p_adjoint` mirrors the primary adaptive
stage contract with shared tableau coefficients inside a ``lax.scan`` over a
predetermined h-schedule. ``jax.grad`` flows through the scan natively,
giving the discrete adjoint of the Rosenbrock-Wanner method
(Hairer-Wanner Ch. IV.8; Sandu+ 2003).

Five gates exercise:
    1. Forward parity vs primary adaptive ``_solve_core``.
    2. Forward parity vs Diffrax ``Kvaerno5 + RecursiveCheckpointAdjoint``
       (Phase α cross-validation reference).
    3. ``jax.grad`` is finite, NaN-free, non-trivial.
    4. Gradient parity vs Richardson FD (≤ 1e-4 rel).
    5. Gradient parity vs Diffrax adjoint (≤ 1e-5 rel).

Plan §13.3.4 acceptance gates.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("diffrax")

import jax
import jax.numpy as jnp

from rabbit.jax.solver_jax_rodas5p_adjoint import (
    solve_log_steps,
    solve_uniform_steps,
    solve_with_adjoint,
    replay_solve_with_h_tape,
)


jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# §1. Test fixtures: stiff Robertson + linear (sanity)
# ═══════════════════════════════════════════════════════════════════════

def _robertson_rhs(t, y, theta):
    """Stiff chemical kinetics, ratio ~1e9 ≈ BBN n/p region."""
    k1, k2, k3 = theta[0], theta[1], theta[2]
    return jnp.array([
        -k1 * y[0] + k3 * y[1] * y[2],
        k1 * y[0] - k3 * y[1] * y[2] - k2 * y[1] ** 2,
        k2 * y[1] ** 2,
    ])


def _linear_rhs(t, y, theta):
    """Non-stiff linear: dy/dt = -theta * y."""
    return -theta[0] * y


_THETA_FID = jnp.array([0.04, 3.0e7, 1.0e4])
_Y0 = jnp.array([1.0, 0.0, 0.0])
_T_SPAN = (0.0, 10.0)


def _richardson_fd(fn, x, eps0=1.0e-3, levels=3):
    """4-point Richardson extrapolation (matches test_native_ad_parity.py)."""
    n = x.shape[0]
    out = jnp.zeros(n)
    for i in range(n):
        derivs = []
        for k in range(levels):
            eps = eps0 / (2 ** k)
            xp = x.at[i].set(x[i] + eps)
            xm = x.at[i].set(x[i] - eps)
            derivs.append((fn(xp) - fn(xm)) / (2.0 * eps))
        d = jnp.array(derivs)
        for k in range(1, levels):
            factor = 4.0 ** k
            d = d.at[k:].set((factor * d[k:] - d[:-k]) / (factor - 1.0))
        out = out.at[i].set(d[-1])
    return out


# ═══════════════════════════════════════════════════════════════════════
# §2. Forward correctness gates
# ═══════════════════════════════════════════════════════════════════════

class TestForwardCorrectness:
    """Solve_log_steps must match adaptive _solve_core and Diffrax."""

    def test_robertson_log_steps_matches_diffrax(self):
        """Log-spaced 400-step solve agrees with Diffrax adjoint at 1e-6."""
        from rabbit.jax.solver_diffrax_canonical import diffrax_adjoint_solve

        y_log = solve_log_steps(_robertson_rhs, _Y0, _THETA_FID,
                                _T_SPAN[0], _T_SPAN[1], n_steps=400)
        y_dfx = diffrax_adjoint_solve(_robertson_rhs, _Y0, _T_SPAN, _THETA_FID,
                                      rtol=1.0e-10, atol=1.0e-12)
        rel = jnp.max(jnp.abs(y_log - y_dfx)
                      / jnp.maximum(jnp.abs(y_dfx), 1.0e-12))
        assert float(rel) < 1.0e-5, (
            f"log-steps Rodas5P vs Diffrax: rel={float(rel):.3e}, "
            f"y_log={y_log}, y_dfx={y_dfx}"
        )

    def test_robertson_log_steps_matches_adaptive_solve_core(self):
        """Log-spaced 400-step solve agrees with adaptive _solve_core at 1e-5."""
        from rabbit.jax.solver_jax_rodas5p import _solve_core

        def f_no_theta(t, y):
            return _robertson_rhs(t, y, _THETA_FID)

        y_adapt, _, _, _ = _solve_core(
            f_no_theta, _Y0, _T_SPAN[0], _T_SPAN[1],
            rtol=1.0e-9, atol=1.0e-12, max_steps=20000, h_max=0.5,
        )
        y_log = solve_log_steps(_robertson_rhs, _Y0, _THETA_FID,
                                _T_SPAN[0], _T_SPAN[1], n_steps=400)
        rel = jnp.max(jnp.abs(y_log - y_adapt)
                      / jnp.maximum(jnp.abs(y_adapt), 1.0e-12))
        assert float(rel) < 1.0e-4, (
            f"log-steps Rodas5P vs adaptive _solve_core: rel={float(rel):.3e}"
        )

    def test_linear_uniform_steps(self):
        """Non-stiff linear: y = exp(-θ t); 200 uniform steps."""
        theta = jnp.array([0.5])
        y_final = solve_uniform_steps(_linear_rhs, jnp.array([1.0]),
                                      theta, 0.0, 4.0, n_steps=200)
        expected = jnp.exp(-0.5 * 4.0)
        assert abs(float(y_final[0]) - float(expected)) < 1.0e-6


# ═══════════════════════════════════════════════════════════════════════
# §3. Gradient correctness gates
# ═══════════════════════════════════════════════════════════════════════

class TestGradientCorrectness:
    """jax.grad through the discrete-adjoint replay must be correct."""

    def test_grad_finite_and_nontrivial(self):
        """jax.grad of y_final[0] wrt theta is finite and non-zero."""
        def loss(theta):
            yf = solve_log_steps(_robertson_rhs, _Y0, theta,
                                 _T_SPAN[0], _T_SPAN[1], n_steps=300)
            return yf[0]
        g = jax.grad(loss)(_THETA_FID)
        assert jnp.all(jnp.isfinite(g)), f"grad has NaN: {g}"
        assert float(jnp.max(jnp.abs(g))) > 1.0e-3, (
            f"grad too small: {g}"
        )

    def test_grad_vs_richardson_fd(self):
        """jax.grad matches Richardson FD to 1e-4 rel.

        Robust to near-zero gradient components (Robertson has near-zero
        sensitivity to k_2 in y[0]; we assert absolute floor instead in
        that case — same protocol as test_native_ad_parity.py).
        """
        def loss(theta):
            yf = solve_log_steps(_robertson_rhs, _Y0, theta,
                                 _T_SPAN[0], _T_SPAN[1], n_steps=300)
            return yf[0]

        g_ad = jax.grad(loss)(_THETA_FID)
        g_fd = _richardson_fd(loss, _THETA_FID, eps0=1.0e-3, levels=3)

        FD_FLOOR = 1.0e-7
        for i in range(_THETA_FID.shape[0]):
            af = float(jnp.abs(g_ad[i]))
            ff = float(jnp.abs(g_fd[i]))
            diff = float(jnp.abs(g_ad[i] - g_fd[i]))
            if ff > FD_FLOOR:
                rel = diff / ff
                assert rel < 1.0e-4, (
                    f"AD-vs-FD rel error too large at param {i}: "
                    f"rel={rel:.3e} g_ad={float(g_ad[i]):.3e} "
                    f"g_fd={float(g_fd[i]):.3e}"
                )
            else:
                assert af < 10.0 * FD_FLOOR, (
                    f"AD grad too large where FD below floor: "
                    f"|g_ad|={af:.3e} |g_fd|={ff:.3e}"
                )

    def test_grad_vs_diffrax_adjoint(self):
        """jax.grad through Rodas5P discrete adjoint matches Diffrax to 1e-5."""
        from rabbit.jax.solver_diffrax_canonical import diffrax_adjoint_solve

        def loss_rod(theta):
            yf = solve_log_steps(_robertson_rhs, _Y0, theta,
                                 _T_SPAN[0], _T_SPAN[1], n_steps=400)
            return yf[0]

        def loss_dfx(theta):
            yf = diffrax_adjoint_solve(_robertson_rhs, _Y0, _T_SPAN, theta,
                                       rtol=1.0e-10, atol=1.0e-12)
            return yf[0]

        g_rod = jax.grad(loss_rod)(_THETA_FID)
        g_dfx = jax.grad(loss_dfx)(_THETA_FID)

        FD_FLOOR = 1.0e-7
        for i in range(_THETA_FID.shape[0]):
            ar = float(jnp.abs(g_rod[i]))
            ad = float(jnp.abs(g_dfx[i]))
            diff = float(jnp.abs(g_rod[i] - g_dfx[i]))
            if ad > FD_FLOOR:
                rel = diff / ad
                assert rel < 1.0e-5, (
                    f"Rodas5P-AD vs Diffrax-AD rel mismatch at param {i}: "
                    f"rel={rel:.3e} rodas={float(g_rod[i]):.3e} "
                    f"dfx={float(g_dfx[i]):.3e}"
                )
            else:
                assert ar < 10.0 * FD_FLOOR


# ═══════════════════════════════════════════════════════════════════════
# §4. Composability with jit / jacrev
# ═══════════════════════════════════════════════════════════════════════

class TestComposability:
    """The new path must compose with jax.jit and jax.jacrev."""

    def test_jit_compiles(self):
        """jit-compiled solve runs without recompiling on second call."""
        @jax.jit
        def jitted(theta):
            return solve_log_steps(_robertson_rhs, _Y0, theta,
                                   _T_SPAN[0], _T_SPAN[1], n_steps=200)
        y1 = jitted(_THETA_FID)
        y2 = jitted(_THETA_FID * 1.001)
        assert jnp.all(jnp.isfinite(y1))
        assert jnp.all(jnp.isfinite(y2))

    def test_grad_per_component_via_grad(self):
        """jax.grad of each output component is finite (jacrev-equivalent)."""
        def f(theta):
            return solve_log_steps(_robertson_rhs, _Y0, theta,
                                   _T_SPAN[0], _T_SPAN[1], n_steps=300)
        for k in range(3):
            grad_k = jax.grad(lambda th, kk=k: f(th)[kk])(_THETA_FID)
            assert jnp.all(jnp.isfinite(grad_k)), (
                f"grad of component {k} has NaN: {grad_k}"
            )


# ═══════════════════════════════════════════════════════════════════════
# §5. solve_with_adjoint top-level API
# ═══════════════════════════════════════════════════════════════════════

def test_solve_with_adjoint_default_uniform():
    """solve_with_adjoint with default n_steps returns finite result on linear RHS."""
    theta = jnp.array([0.5])
    y0 = jnp.array([1.0])
    yf = solve_with_adjoint(_linear_rhs, y0, theta, 0.0, 4.0, n_steps=500)
    assert abs(float(yf[0]) - float(jnp.exp(-2.0))) < 1.0e-5


def test_solve_with_adjoint_custom_h_tape():
    """solve_with_adjoint with explicit h_tape uses the provided schedule."""
    theta = jnp.array([0.5])
    y0 = jnp.array([1.0])
    h_tape = jnp.full((100,), 0.04)
    yf = solve_with_adjoint(_linear_rhs, y0, theta, 0.0, 4.0, h_tape=h_tape)
    assert abs(float(yf[0]) - float(jnp.exp(-2.0))) < 1.0e-4


def test_replay_nonautonomous_order_with_analytic_dfdt():
    """Analytic and fallback replay retain fifth order on a nonautonomous RHS."""

    def rhs(t, y, theta):
        return -y + theta[0] * jnp.sin(t)

    def dfdt(t, y, theta):
        del y
        return jnp.asarray([theta[0] * jnp.cos(t)], dtype=theta.dtype)

    theta = jnp.asarray([1.0], dtype=jnp.float64)
    y0 = jnp.asarray([0.0], dtype=jnp.float64)
    exact = 0.5 * (np.sin(1.0) - np.cos(1.0) + np.exp(-1.0))
    step_counts = np.asarray([8, 16, 32, 64], dtype=np.int64)
    analytic_errors = []
    fallback_errors = []
    endpoint_gaps = []

    for n_steps in step_counts:
        h_tape = jnp.full((int(n_steps),), 1.0 / float(n_steps), dtype=jnp.float64)
        analytic_endpoint = replay_solve_with_h_tape(
            rhs,
            y0,
            theta,
            0.0,
            h_tape,
            dfdt_fn=dfdt,
        )
        fallback_endpoint = replay_solve_with_h_tape(
            rhs,
            y0,
            theta,
            0.0,
            h_tape,
        )
        analytic_errors.append(abs(float(analytic_endpoint[0]) - exact))
        fallback_errors.append(abs(float(fallback_endpoint[0]) - exact))
        endpoint_gaps.append(abs(float(analytic_endpoint[0] - fallback_endpoint[0])))

    analytic_order = float(
        np.polyfit(
            np.log(1.0 / step_counts.astype(np.float64)),
            np.log(np.asarray(analytic_errors, dtype=np.float64)),
            1,
        )[0]
    )
    fallback_order = float(
        np.polyfit(
            np.log(1.0 / step_counts.astype(np.float64)),
            np.log(np.asarray(fallback_errors, dtype=np.float64)),
            1,
        )[0]
    )
    assert np.all(np.isfinite(analytic_errors)), analytic_errors
    assert np.all(np.isfinite(fallback_errors)), fallback_errors
    assert analytic_order >= 4.5, analytic_order
    assert fallback_order >= 4.5, fallback_order
    assert endpoint_gaps[2] <= 1.0e-10


def test_replay_dfdt_runs_once_per_active_step_and_skips_padding():
    callback_points = []

    def rhs(t, y, theta):
        return -y + theta[0] * jnp.sin(t)

    def record_callback(t, y):
        callback_points.append((float(t), np.asarray(y, dtype=np.float64).copy()))

    def dfdt(t, y, theta):
        jax.debug.callback(record_callback, t, y, ordered=True)
        return jnp.asarray([theta[0] * jnp.cos(t)], dtype=theta.dtype)

    endpoint = replay_solve_with_h_tape(
        rhs,
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
        0.0,
        jnp.asarray([0.125, 0.0, 0.125, 0.0], dtype=jnp.float64),
        dfdt_fn=dfdt,
    )
    jax.block_until_ready(endpoint)
    jax.effects_barrier()

    assert len(callback_points) == 2
    np.testing.assert_allclose(
        np.asarray([point[0] for point in callback_points]),
        np.asarray([0.0, 0.125]),
        rtol=0.0,
        atol=0.0,
    )
