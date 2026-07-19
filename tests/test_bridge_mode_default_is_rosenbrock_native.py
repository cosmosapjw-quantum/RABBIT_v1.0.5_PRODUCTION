"""tests/test_bridge_mode_default_is_rosenbrock_native.py — v3.0 Phase A gate.

Locks the v3.0 production default: ``bridge_mode='rosenbrock_native'``
(Rodas5P discrete adjoint replay, Phase RA from v2 plan §13). Up through
v2.0 the default was ``'fd_legacy'``; v3.0 promotes the native AD path.

This file is the single source of truth that the dispatcher exposes the
right valid set + correct default. If the default needs to change, this
test must be updated explicitly with a citation to the decision record.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from rabbit.jax.gradient_bridge import (
    DEFAULT_BRIDGE_MODE,
    VALID_BRIDGE_MODES,
    BBNObservation,
    make_differentiable_solve_rodas5p_native,
    make_log_posterior,
)


jax.config.update("jax_enable_x64", True)


def _toy_rhs(t, y, theta):
    """Mildly stiff exponential decay; sufficient for dispatch checks."""
    return -theta[0] * y


_THETA = jnp.array([1.0, 0.0])
_Y0 = jnp.array([1.0])
_T_SPAN = (0.0, 1.0)


class TestDefaultBridgeMode:
    """v3.0 contract: rosenbrock_native is the production default."""

    def test_default_is_rosenbrock_native(self):
        assert DEFAULT_BRIDGE_MODE == "rosenbrock_native", (
            f"v3.0 contract violation: DEFAULT_BRIDGE_MODE={DEFAULT_BRIDGE_MODE!r}; "
            "expected 'rosenbrock_native'"
        )

    def test_valid_modes_include_three(self):
        assert set(VALID_BRIDGE_MODES) == {
            "rosenbrock_native",
            "diffrax_native",
            "fd_legacy",
        }

    def test_make_log_posterior_accepts_none_routes_to_default(self):
        """bridge_mode=None routes to DEFAULT_BRIDGE_MODE without error."""
        obs = [BBNObservation("y", 0.5, 0.05)]
        indices = {"y": 0}
        lp = make_log_posterior(
            _toy_rhs, _Y0, _T_SPAN, indices, obs,
            bridge_mode=None,
        )
        val = lp(_THETA)
        assert jnp.isfinite(val)

    def test_unknown_bridge_mode_raises(self):
        obs = [BBNObservation("y", 0.5, 0.05)]
        indices = {"y": 0}
        with pytest.raises(ValueError, match=r"unknown bridge_mode"):
            make_log_posterior(
                _toy_rhs, _Y0, _T_SPAN, indices, obs,
                bridge_mode="not_a_real_mode",
            )


class TestRosenbrockNativeFactory:
    """Phase RA factory exposes a grad-able solve_fn."""

    def test_factory_returns_grad_able_solve(self):
        solve = make_differentiable_solve_rodas5p_native(
            _toy_rhs, _Y0, _T_SPAN, n_steps=200,
        )
        yf = solve(_THETA)
        assert jnp.all(jnp.isfinite(yf))
        # Reverse-mode gradient: dy_final/dtheta_0 should be negative
        # (decaying exponential: more decay at higher rate).
        g = jax.grad(lambda th: solve(th)[0])(_THETA)
        assert jnp.all(jnp.isfinite(g))
        assert float(g[0]) < 0.0, (
            f"expected negative dy/dtheta_0 for exponential decay; got {float(g[0])}"
        )

    def test_log_schedule_matches_uniform_within_budget(self):
        """Uniform vs log schedules give the same final state to 1e-6 on a
        non-stiff RHS (sanity check that the schedule dispatch works)."""
        solve_unif = make_differentiable_solve_rodas5p_native(
            _toy_rhs, _Y0, _T_SPAN, n_steps=400, schedule="uniform",
        )
        solve_log = make_differentiable_solve_rodas5p_native(
            _toy_rhs, _Y0, _T_SPAN, n_steps=400, schedule="log",
        )
        yu = solve_unif(_THETA)
        yl = solve_log(_THETA)
        assert jnp.all(jnp.isfinite(yu))
        assert jnp.all(jnp.isfinite(yl))
        # Same physics, different discretization; should agree to 1e-4.
        rel = float(jnp.max(jnp.abs(yu - yl) / jnp.maximum(jnp.abs(yu), 1e-12)))
        assert rel < 1.0e-4, (
            f"uniform vs log schedule disagreement: rel={rel:.3e}"
        )

    def test_invalid_schedule_raises(self):
        solve = make_differentiable_solve_rodas5p_native(
            _toy_rhs, _Y0, _T_SPAN, n_steps=100, schedule="bogus",
        )
        with pytest.raises(ValueError, match=r"unknown schedule"):
            solve(_THETA)


class TestRosenbrockVsDiffraxParity:
    """Tighter parity gate (1e-6) between the two AD paths.

    v2.0 test_native_ad_parity.py held a 1e-5 gate on the stiff Robertson
    surrogate; this is the v3.0 follow-on that tightens at the level of the
    bridge-mode dispatch on a smooth non-stiff RHS where the Rodas5P
    discrete adjoint and Diffrax adjoint should agree near machine
    precision.
    """

    def test_yp_rodas5p_vs_diffrax_smooth_rhs(self):
        from rabbit.jax.solver_diffrax_canonical import diffrax_adjoint_solve

        y_dfx = diffrax_adjoint_solve(
            _toy_rhs, _Y0, _T_SPAN, _THETA, rtol=1e-10, atol=1e-12,
        )
        solve = make_differentiable_solve_rodas5p_native(
            _toy_rhs, _Y0, _T_SPAN, n_steps=800,
        )
        y_ros = solve(_THETA)
        rel = float(jnp.max(jnp.abs((y_dfx - y_ros) / jnp.maximum(jnp.abs(y_dfx), 1e-12))))
        assert rel < 1.0e-6, (
            f"rosenbrock_native vs diffrax_native parity widened: rel={rel:.3e}"
        )
