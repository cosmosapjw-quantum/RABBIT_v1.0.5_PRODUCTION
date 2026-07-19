"""
test_j04_jax_rodas5p — Verification for J04 (JAX Rodas5P Solver).

Gate GJ2: Y_p(JAX) matches Y_p(SciPy Rodas5P) to < 10⁻⁸

Acceptance criteria:
  AC1: FLRW-proxy Y_p matches SciPy Radau to < 10⁻⁵
  AC2: Y_p matches SciPy Rodas5P to < 10⁻⁸
  AC3: Anisotropic (Σ=0.1) parity < 10⁻⁵ (same algorithm, diff Jacobian)
  AC4: Step count within 10% of SciPy Rodas5P
  AC5: Step rejection rate < 5% for simple problems
  AC6: JIT compilation time < 30 s
  AC7: Mass conservation for nuclear-like system
  AC8: Convergence: err ∝ rtol for order 5
"""
import pytest
import sys
import os
import time

# (paths resolved via pip install)

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

from rabbit.jax.solver_jax_rodas5p import (
    A,
    C,
    GAMMA,
    LowRankJacobianFactors,
    _rodas5p_step,
    _rodas5p_step_custom_linear_solver,
    _rodas5p_step_low_rank,
    _rodas5p_step_schur,
    _solve_woodbury_low_rank_update,
    b_weights,
    jax_rodas5p_solve,
    materialize_low_rank_jacobian,
    prepare_low_rank_linear_solver,
    solve_prepared_low_rank_linear_system,
)


_PUB02_STAGE_TIMES = np.asarray(
    [
        0.0,
        0.6358126895828704,
        0.4095798393397535,
        0.9769306725060716,
        0.4288403609558664,
        1.0,
        1.0,
        1.0,
    ],
    dtype=np.float64,
)
_PUB02_TIME_DERIVATIVE_WEIGHTS = np.asarray(
    [
        0.21193756319429014,
        -0.42387512638858027,
        -0.3384627126235924,
        1.8046452872882734,
        2.325825639765069,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)


def _fixed_step_nonautonomous_endpoint(*, n_steps, dfdt_fn):
    """Run the frozen PUB-02 order probe with exactly ``n_steps`` steps."""

    h = 1.0 / n_steps

    def rhs(t, y):
        return -y + jnp.sin(t)

    result = jax_rodas5p_solve(
        rhs,
        jnp.asarray([0.0], dtype=jnp.float64),
        (0.0, 1.0),
        rtol=1.0e6,
        atol=1.0e6,
        max_steps=n_steps,
        h_init=h,
        h_min=h,
        h_max=h,
        jac_fn=lambda _t, _y: jnp.asarray([[-1.0]], dtype=jnp.float64),
        dfdt_fn=dfdt_fn,
    )
    assert result.success
    assert result.n_steps == n_steps
    assert result.n_reject == 0
    return float(result.y_final[0])


def _least_squares_order(errors, step_counts):
    h_values = 1.0 / np.asarray(step_counts, dtype=np.float64)
    return float(np.polyfit(np.log(h_values), np.log(np.asarray(errors)), 1)[0])


def _independent_corrected_rodas5p_action(rhs, t, y, h, jacobian, dfdt):
    """Independent SciML-convention action for the frozen PUB-02 tableau."""

    a = np.asarray(A, dtype=np.float64)
    coupling = np.asarray(C, dtype=np.float64)
    weights = np.asarray(b_weights, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    dfdt = np.asarray(dfdt, dtype=np.float64)
    w_matrix = np.eye(y.size, dtype=np.float64) / (float(GAMMA) * h) - jacobian
    stages = np.zeros((weights.size, y.size), dtype=np.float64)
    for stage in range(weights.size):
        y_stage = y + a[stage] @ stages
        coupling_sum = coupling[stage] @ stages / h
        stage_rhs = (
            np.asarray(
                rhs(t + _PUB02_STAGE_TIMES[stage] * h, y_stage),
                dtype=np.float64,
            )
            + coupling_sum
            + h * _PUB02_TIME_DERIVATIVE_WEIGHTS[stage] * dfdt
        )
        stages[stage] = np.linalg.solve(w_matrix, stage_rhs)
    return y + weights @ stages, stages


def test_nonautonomous_order_jax_analytic_and_fd():
    """PUB-02: both analytic and fallback partial-time paths retain order five."""

    step_counts = (8, 16, 32, 64)
    exact = 0.5 * (np.sin(1.0) - np.cos(1.0) + np.exp(-1.0))
    analytic_endpoints = np.asarray(
        [
            _fixed_step_nonautonomous_endpoint(
                n_steps=n_steps,
                dfdt_fn=lambda t, _y: jnp.asarray([jnp.cos(t)], dtype=jnp.float64),
            )
            for n_steps in step_counts
        ]
    )
    fallback_endpoints = np.asarray(
        [
            _fixed_step_nonautonomous_endpoint(n_steps=n_steps, dfdt_fn=None)
            for n_steps in step_counts
        ]
    )
    analytic_order = _least_squares_order(
        np.abs(analytic_endpoints - exact), step_counts
    )
    fallback_order = _least_squares_order(
        np.abs(fallback_endpoints - exact), step_counts
    )

    assert analytic_order >= 4.5, analytic_order
    assert fallback_order >= 4.5, fallback_order
    assert abs(analytic_endpoints[2] - fallback_endpoints[2]) <= 1.0e-10


def test_autonomous_endpoint_jax_is_frozen():
    """PUB-02 must not perturb the pre-registered autonomous endpoint."""

    result = jax_rodas5p_solve(
        lambda _t, y: -y,
        jnp.asarray([1.0], dtype=jnp.float64),
        (0.0, 5.0),
        rtol=1.0e-8,
        atol=1.0e-10,
        h_max=0.5,
    )
    assert result.success
    assert abs(float(result.y_final[0]) - 0.00673794700071956) <= 1.0e-12


def test_stiff_nonautonomous_numpy_jax_twin():
    """PUB-02 stiff twin must reach cos(1) and agree across implementations."""

    from rabbit.solver.rodas5p import Rodas5PConfig, solve as numpy_solve

    def rhs_numpy(t, y):
        return -1000.0 * (y - np.cos(t)) - np.sin(t)

    def rhs_jax(t, y):
        return -1000.0 * (y - jnp.cos(t)) - jnp.sin(t)

    numpy_result = numpy_solve(
        rhs_numpy,
        (0.0, 1.0),
        np.asarray([1.0], dtype=np.float64),
        config=Rodas5PConfig(rtol=1.0e-8, atol=1.0e-10, max_step_N=0.05),
        dfdt_fn=lambda t, _y: np.asarray(
            [-1000.0 * np.sin(t) - np.cos(t)], dtype=np.float64
        ),
    )
    jax_result = jax_rodas5p_solve(
        rhs_jax,
        jnp.asarray([1.0], dtype=jnp.float64),
        (0.0, 1.0),
        rtol=1.0e-8,
        atol=1.0e-10,
        h_max=0.05,
        dfdt_fn=lambda t, _y: jnp.asarray(
            [-1000.0 * jnp.sin(t) - jnp.cos(t)], dtype=jnp.float64
        ),
    )
    numpy_endpoint = float(numpy_result.y[0, -1])
    jax_endpoint = float(jax_result.y_final[0])
    exact = np.cos(1.0)

    assert numpy_result.success
    assert jax_result.success
    assert abs(numpy_endpoint - exact) <= 1.0e-7
    assert abs(jax_endpoint - exact) <= 1.0e-7
    assert abs(numpy_endpoint - jax_endpoint) <= 1.0e-8


def test_dfdt_callback_is_once_per_attempt_at_step_start():
    """The analytic partial is sampled once at (t_n, y_n), not at stages."""

    callback_points = []
    matrix = jnp.asarray([[-2.0, 0.5], [0.1, 0.0]], dtype=jnp.float64)
    y0 = jnp.asarray([0.8, -0.2], dtype=jnp.float64)

    def rhs(t, y):
        return matrix @ y + jnp.asarray([jnp.sin(t), jnp.cos(2.0 * t)])

    def record_point(t, y):
        callback_points.append((float(t), np.asarray(y, dtype=np.float64).copy()))

    def dfdt_fn(t, y):
        jax.debug.callback(record_point, t, y, ordered=True)
        return jnp.asarray([jnp.cos(t), -2.0 * jnp.sin(2.0 * t)])

    result = jax_rodas5p_solve(
        rhs,
        y0,
        (0.25, 0.26),
        rtol=1.0e6,
        atol=1.0e6,
        max_steps=1,
        h_init=0.01,
        h_min=0.01,
        h_max=0.01,
        jac_fn=lambda _t, _y: matrix,
        dfdt_fn=dfdt_fn,
    )
    jax.block_until_ready(result.y_final)
    jax.effects_barrier()

    assert result.success
    assert result.n_steps + result.n_reject == 1
    assert len(callback_points) == 1, callback_points
    assert callback_points[0][0] == pytest.approx(0.25, rel=0.0, abs=0.0)
    np.testing.assert_array_equal(callback_points[0][1], np.asarray(y0))


def test_fallback_dfdt_is_once_per_attempt_at_step_start():
    """Fallback uses one central pair at fixed y_n before all eight stages."""

    rhs_points = []
    matrix = jnp.asarray([[-2.0, 0.5], [0.1, 0.0]], dtype=jnp.float64)
    y0 = jnp.asarray([0.8, -0.2], dtype=jnp.float64)
    t0 = 0.25
    step = 0.01
    delta = np.cbrt(np.finfo(np.float64).eps) * max(1.0, abs(t0), abs(step))

    def record_point(t, y):
        rhs_points.append((float(t), np.asarray(y, dtype=np.float64).copy()))

    def rhs(t, y):
        jax.debug.callback(record_point, t, y, ordered=True)
        return matrix @ y + jnp.asarray([jnp.sin(t), jnp.cos(2.0 * t)])

    result = jax_rodas5p_solve(
        rhs,
        y0,
        (t0, t0 + step),
        rtol=1.0e6,
        atol=1.0e6,
        max_steps=1,
        h_init=step,
        h_min=step,
        h_max=step,
        jac_fn=lambda _t, _y: matrix,
    )
    jax.block_until_ready(result.y_final)
    jax.effects_barrier()

    assert result.success
    assert result.n_steps + result.n_reject == 1
    assert len(rhs_points) == 10, rhs_points
    expected_fd_times = (t0 - delta, t0 + delta)
    for expected_time in expected_fd_times:
        matching = [
            state
            for time, state in rhs_points
            if abs(time - expected_time) <= 2.0 * np.finfo(np.float64).eps
        ]
        assert len(matching) == 1, rhs_points
        np.testing.assert_array_equal(matching[0], np.asarray(y0))


def test_nonautonomous_dense_custom_and_schur_step_actions_match():
    """Every scalar step copy must apply the same corrected frozen action."""

    matrix = jnp.asarray([[-2.0, 0.5], [0.1, 0.0]], dtype=jnp.float64)
    t = jnp.float64(0.25)
    h = jnp.float64(0.01)
    y = jnp.asarray([0.8, -0.2], dtype=jnp.float64)
    dfdt = jnp.asarray(
        [jnp.cos(t), -2.0 * jnp.sin(2.0 * t)], dtype=jnp.float64
    )

    def rhs(time, state):
        return matrix @ state + jnp.asarray(
            [jnp.sin(time), jnp.cos(2.0 * time)], dtype=jnp.float64
        )

    def prepare(_t, _y, step, jacobian):
        return jnp.eye(2, dtype=jnp.float64) / (GAMMA * step) - jacobian

    def solve_prepared(prepared, rhs_vector):
        return jnp.linalg.solve(prepared, rhs_vector)

    expected_y, expected_stages = _independent_corrected_rodas5p_action(
        rhs, float(t), np.asarray(y), float(h), np.asarray(matrix), np.asarray(dfdt)
    )
    dense = _rodas5p_step(
        rhs, t, y, h, matrix, 1.0e-8, 1.0e-10, dfdt=dfdt
    )
    custom = _rodas5p_step_custom_linear_solver(
        rhs,
        t,
        y,
        h,
        matrix,
        1.0e-8,
        1.0e-10,
        prepare,
        solve_prepared,
        dfdt=dfdt,
    )
    schur = _rodas5p_step_schur(
        rhs,
        t,
        y,
        h,
        matrix,
        1.0e-8,
        1.0e-10,
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([1], dtype=jnp.int32),
        dfdt=dfdt,
    )

    np.testing.assert_allclose(
        np.asarray(dense[0]), expected_y, rtol=0.0, atol=1.0e-11
    )
    np.testing.assert_allclose(
        np.asarray(dense[2]), expected_stages, rtol=0.0, atol=1.0e-11
    )
    for candidate in (custom, schur):
        np.testing.assert_allclose(
            np.asarray(candidate[0]), np.asarray(dense[0]), rtol=0.0, atol=1.0e-11
        )
        np.testing.assert_allclose(
            np.asarray(candidate[2]), np.asarray(dense[2]), rtol=0.0, atol=1.0e-11
        )


class TestAnalyticalProblems:
    """Exact-solution test problems."""

    def test_exponential_decay_convergence(self):
        """AC8: Order-5 convergence y′ = −y."""
        y_exact = np.exp(-5.0)
        errors = []
        for rtol in [1e-4, 1e-6, 1e-8, 1e-10]:
            r = jax_rodas5p_solve(lambda t, y: -y, jnp.array([1.0]),
                                  (0.0, 5.0), rtol=rtol, atol=rtol*1e-2, h_max=0.5)
            err = abs(float(r.y_final[0]) - y_exact) / y_exact
            errors.append(err)
        # Each 100× tighter tol should give ~100× smaller error (order 5: h^5)
        ratio = errors[0] / max(errors[2], 1e-16)
        assert ratio > 100, f"Convergence ratio {ratio:.0f} (expect > 100)"
        assert errors[2] < 1e-8, f"Error at rtol=1e-8: {errors[2]:.2e}"
        print(f"  AC8 convergence: PASS (errors: {[f'{e:.1e}' for e in errors]})")

    def test_stiff_2d_system(self):
        """Stiff system: slow+fast modes."""
        def rhs(t, y):
            return jnp.array([-1.0 * y[0], -1000.0 * y[1]])
        r = jax_rodas5p_solve(rhs, jnp.array([1.0, 1.0]), (0.0, 1.0),
                              rtol=1e-8, atol=1e-10)
        y_exact = np.exp(-1.0)
        err = abs(float(r.y_final[0]) - y_exact) / y_exact
        assert err < 1e-6, f"Slow mode err: {err:.2e}"
        assert r.success
        print(f"  Stiff 2D: PASS (slow err={err:.2e}, steps={r.n_steps})")


class TestSciPyParity:
    """Gate GJ2: JAX vs SciPy Rodas5P parity."""

    def test_ac2_parity_rtol_1e8(self):
        """AC2: Parity at rtol=1e-8 < 10⁻⁸."""
        from rabbit.solver.rodas5p import solve as sp_solve, Rodas5PConfig
        f_np = lambda t, y: -y
        f_jx = lambda t, y: -y
        cfg = Rodas5PConfig(rtol=1e-8, atol=1e-10, max_step_N=0.5)
        r_sp = sp_solve(f_np, (0.0, 5.0), [1.0], config=cfg)
        r_jx = jax_rodas5p_solve(f_jx, jnp.array([1.0]), (0.0, 5.0),
                                  rtol=1e-8, atol=1e-10, h_max=0.5)
        parity = abs(r_sp.y[0, -1] - float(r_jx.y_final[0])) / abs(r_sp.y[0, -1])
        assert parity < 1e-8, f"Parity = {parity:.2e}"
        print(f"  AC2 parity(rtol=1e-8): PASS ({parity:.2e})")

    def test_ac4_step_count_parity(self):
        """AC4: Step count within 10% of SciPy."""
        from rabbit.solver.rodas5p import solve as sp_solve, Rodas5PConfig
        cfg = Rodas5PConfig(rtol=1e-8, atol=1e-10, max_step_N=0.5)
        r_sp = sp_solve(lambda t, y: -y, (0.0, 5.0), [1.0], config=cfg)
        r_jx = jax_rodas5p_solve(lambda t, y: -y, jnp.array([1.0]),
                                  (0.0, 5.0), rtol=1e-8, atol=1e-10, h_max=0.5)
        ratio = abs(r_jx.n_steps - r_sp.n_steps) / r_sp.n_steps
        assert ratio < 0.1, f"Step ratio diff = {ratio:.2f}"
        print(f"  AC4 step count: PASS (SciPy={r_sp.n_steps}, JAX={r_jx.n_steps})")

    def test_vanderpol_parity(self):
        """Van der Pol μ=10 parity."""
        from rabbit.solver.rodas5p import solve as sp_solve, Rodas5PConfig
        cfg = Rodas5PConfig(rtol=1e-8, atol=1e-10, max_step_N=0.5)
        r_sp = sp_solve(lambda t, y: np.array([y[1], 10*(1-y[0]**2)*y[1]-y[0]]),
                        (0.0, 1.0), [2.0, 0.0], config=cfg)
        r_jx = jax_rodas5p_solve(
            lambda t, y: jnp.array([y[1], 10*(1-y[0]**2)*y[1]-y[0]]),
            jnp.array([2.0, 0.0]), (0.0, 1.0), rtol=1e-8, atol=1e-10, h_max=0.5)
        parity = np.max(np.abs(r_sp.y[:, -1] - np.array(r_jx.y_final)) /
                       np.maximum(np.abs(r_sp.y[:, -1]), 1e-10))
        assert parity < 1e-8, f"VdP parity = {parity:.2e}"
        print(f"  Van der Pol parity: PASS ({parity:.2e})")


class TestSolverProperties:
    """Correctness properties."""

    @staticmethod
    def _synthetic_low_rank_factors(dim=12, rank=4):
        rng = np.random.default_rng(20260421)
        base_diag = -0.5 - 0.1 * np.arange(dim, dtype=np.float64)
        base_jac = np.diag(base_diag)
        left = rng.normal(size=(dim, rank)) * 0.08
        core = rng.normal(size=(rank, rank)) * 0.05
        right = rng.normal(size=(rank, dim)) * 0.08
        return LowRankJacobianFactors(
            base_jacobian=jnp.asarray(base_jac, dtype=jnp.float64),
            left_factor=jnp.asarray(left, dtype=jnp.float64),
            core_matrix=jnp.asarray(core, dtype=jnp.float64),
            right_factor=jnp.asarray(right, dtype=jnp.float64),
        )

    def test_ac5_low_rejection_rate(self):
        """AC5: Rejection rate < 5%."""
        r = jax_rodas5p_solve(lambda t, y: -y, jnp.array([1.0]),
                              (0.0, 5.0), rtol=1e-8, atol=1e-10)
        total = r.n_steps + r.n_reject
        rej_rate = r.n_reject / max(total, 1)
        assert rej_rate < 0.05, f"Rejection rate = {rej_rate:.2%}"
        print(f"  AC5 rejection rate: PASS ({rej_rate:.1%})")

    def test_n_final_exact(self):
        """N_final = N_end exactly (no overshoot)."""
        r = jax_rodas5p_solve(lambda t, y: -y, jnp.array([1.0]),
                              (0.0, 3.7), rtol=1e-8, atol=1e-10)
        assert abs(r.N_final - 3.7) < 1e-12, f"N_final = {r.N_final}"
        print(f"  N_final exact: PASS (N_final = {r.N_final})")

    def test_ac7_mass_conservation(self):
        """AC7: Mass conservation in a nuclear-like system."""
        # 3-species system: dX₁=-X₁, dX₂=X₁-X₂, dX₃=X₂; sum=const=1
        def rhs(t, y):
            return jnp.array([-y[0], y[0]-y[1], y[1]])
        y0 = jnp.array([0.9, 0.1, 0.0])
        r = jax_rodas5p_solve(rhs, y0, (0.0, 10.0), rtol=1e-10, atol=1e-12)
        mass = float(jnp.sum(r.y_final))
        assert abs(mass - 1.0) < 1e-10, f"Mass = {mass}"
        print(f"  AC7 mass conservation: PASS (|ΣX−1| = {abs(mass-1):.2e})")

    def test_success_flag(self):
        """Success flag correct for well-posed problem."""
        r = jax_rodas5p_solve(lambda t, y: -y, jnp.array([1.0]),
                              (0.0, 1.0), rtol=1e-8, atol=1e-10)
        assert r.success, f"success={r.success}, msg={r.message}"
        print(f"  Success flag: PASS")

    def test_max_steps_exhaustion_fails_when_endpoint_not_reached(self):
        """Solver must not report success when max_steps stops before N_end."""
        r = jax_rodas5p_solve(
            lambda t, y: -y,
            jnp.array([1.0]),
            (0.0, 5.0),
            rtol=1.0e-10,
            atol=1.0e-12,
            h_max=0.01,
            max_steps=1,
        )
        assert r.N_final < 5.0
        assert r.success is False
        assert r.status == -2
        assert "max_steps" in r.message
        assert r.diagnostics["max_steps_exhausted"] is True
        assert r.diagnostics["endpoint_reached"] is False

    def test_exact_mode_refresh_count_matches_attempts_for_rejection_free_problem(self):
        """Exact mode should not double-count the initial Jacobian."""
        r = jax_rodas5p_solve(
            lambda t, y: -y,
            jnp.array([1.0]),
            (0.0, 1.0),
            rtol=1e-8,
            atol=1e-10,
        )
        assert r.success
        assert r.n_reject == 0
        assert r.diagnostics["jacobian_policy"] == "exact_per_step"
        assert r.diagnostics["jacobian_reuse_count"] == 0
        assert r.diagnostics["jacobian_refresh_count"] == r.n_steps

    def test_custom_jacobian_hook_bypasses_default_autodiff(self, monkeypatch):
        """An explicit jac_fn should bypass the default cached jacfwd path."""
        import rabbit.jax.solver_jax_rodas5p as rodas_mod

        A = jnp.array([[-2.0, 0.0], [0.0, -5.0]], dtype=jnp.float64)

        def rhs(t, y):
            return A @ y

        def jac_fn(t, y):
            del t, y
            return A

        def _fail_cached_jacfwd(_rhs_fn):
            raise AssertionError("default jacfwd path should not be used when jac_fn is provided")

        monkeypatch.setattr(rodas_mod, "_cached_jacfwd", _fail_cached_jacfwd)
        r = rodas_mod.jax_rodas5p_solve(
            rhs,
            jnp.array([1.0, 1.0], dtype=jnp.float64),
            (0.0, 1.0),
            jac_fn=jac_fn,
            rtol=1e-8,
            atol=1e-10,
        )
        exact = np.array([np.exp(-2.0), np.exp(-5.0)])
        err = np.max(np.abs(np.asarray(r.y_final) - exact) / np.maximum(exact, 1e-12))
        assert r.success
        assert err < 1e-6, f"custom jacobian solve err={err:.2e}"

    def test_custom_jacobian_hook_rejects_sparse_mask_combo(self):
        """Custom jacobians and active-index sparse helpers are mutually exclusive."""
        with pytest.raises(ValueError, match="cannot be combined"):
            jax_rodas5p_solve(
                lambda t, y: -y,
                jnp.array([1.0], dtype=jnp.float64),
                (0.0, 1.0),
                jac_fn=lambda t, y: jnp.array([[-1.0]], dtype=jnp.float64),
                active_indices=jnp.array([0], dtype=jnp.int32),
            )

    def test_custom_linear_solver_hooks_require_complete_pair(self):
        """Linear-solver hooks should be passed as a complete prepare/solve pair."""
        with pytest.raises(ValueError, match="provided together"):
            jax_rodas5p_solve(
                lambda t, y: -y,
                jnp.array([1.0], dtype=jnp.float64),
                (0.0, 1.0),
                linear_solver_prepare_fn=prepare_low_rank_linear_solver,
            )

    def test_public_linear_solver_hooks_support_low_rank_payload(self):
        """Public solve API should accept factorized Jacobian payloads via custom hooks."""
        factors = self._synthetic_low_rank_factors(dim=10, rank=3)
        full_jac = materialize_low_rank_jacobian(factors)
        rhs_matrix = jnp.asarray(np.array(full_jac), dtype=jnp.float64)
        dense_jacobian_payload = jnp.asarray(np.array(full_jac), dtype=jnp.float64)

        def rhs(t, y):
            del t
            return rhs_matrix @ y

        y0 = jnp.linspace(0.2, 1.1, 10, dtype=jnp.float64)
        dense = jax_rodas5p_solve(
            rhs,
            y0,
            (0.0, 1.0),
            jac_fn=lambda t, y: dense_jacobian_payload,
            rtol=1e-8,
            atol=1e-10,
        )
        hooked = jax_rodas5p_solve(
            rhs,
            y0,
            (0.0, 1.0),
            jac_fn=lambda t, y: factors,
            linear_solver_prepare_fn=prepare_low_rank_linear_solver,
            linear_solver_solve_fn=solve_prepared_low_rank_linear_system,
            rtol=1e-8,
            atol=1e-10,
        )

        rel = np.linalg.norm(np.asarray(dense.y_final - hooked.y_final)) / max(np.linalg.norm(np.asarray(dense.y_final)), 1e-30)
        assert dense.success and hooked.success
        assert hooked.diagnostics["linear_solver_policy"] == "custom_hook"
        assert rel < 1e-10, f"hooked solve rel={rel:.2e}"

    def test_low_rank_materialization_matches_dense_assembly(self):
        """Low-rank helper should assemble the same dense Jacobian explicitly."""
        factors = self._synthetic_low_rank_factors(dim=10, rank=3)
        dense = np.asarray(materialize_low_rank_jacobian(factors))
        expected = (
            np.asarray(factors.base_jacobian)
            + np.asarray(factors.left_factor)
            @ np.asarray(factors.core_matrix)
            @ np.asarray(factors.right_factor)
        )
        assert np.allclose(dense, expected, rtol=1e-13, atol=1e-13)

    def test_woodbury_stage_linear_solve_matches_dense(self):
        """Woodbury stage solve should match direct dense solve."""
        factors = self._synthetic_low_rank_factors(dim=14, rank=4)
        h = jnp.float64(0.05)
        rhs = jnp.linspace(0.2, 1.6, 14, dtype=jnp.float64)
        w_base = jnp.eye(14, dtype=jnp.float64) / (GAMMA * h) - factors.base_jacobian
        dense_w = w_base - factors.left_factor @ factors.core_matrix @ factors.right_factor
        x_dense = jnp.linalg.solve(dense_w, rhs)
        x_low_rank = _solve_woodbury_low_rank_update(
            w_base,
            factors.left_factor,
            factors.core_matrix,
            factors.right_factor,
            rhs,
        )
        rel = np.linalg.norm(np.asarray(x_dense - x_low_rank)) / max(np.linalg.norm(np.asarray(x_dense)), 1e-30)
        assert rel < 1e-11, f"Woodbury solve rel={rel:.2e}"

    def test_low_rank_rodas_step_matches_dense_step_for_linear_rhs(self):
        """Experimental low-rank Rodas step should match the dense step."""
        factors = self._synthetic_low_rank_factors(dim=12, rank=4)
        full_jac = materialize_low_rank_jacobian(factors)

        def rhs(t, y):
            del t
            return full_jac @ y

        y0 = jnp.linspace(0.1, 1.2, 12, dtype=jnp.float64)
        y_dense, err_dense, k_dense = _rodas5p_step(
            rhs,
            jnp.float64(0.0),
            y0,
            jnp.float64(0.05),
            full_jac,
            1e-8,
            1e-10,
        )
        y_lr, err_lr, k_lr = _rodas5p_step_low_rank(
            rhs,
            jnp.float64(0.0),
            y0,
            jnp.float64(0.05),
            factors,
            1e-8,
            1e-10,
        )

        y_rel = np.linalg.norm(np.asarray(y_dense - y_lr)) / max(np.linalg.norm(np.asarray(y_dense)), 1e-30)
        k_rel = np.linalg.norm(np.asarray(k_dense - k_lr)) / max(np.linalg.norm(np.asarray(k_dense)), 1e-30)
        err_abs = abs(float(err_dense - err_lr))
        assert y_rel < 1e-11, f"y_rel={y_rel:.2e}"
        assert k_rel < 1e-11, f"k_rel={k_rel:.2e}"
        assert err_abs < 5e-8, f"err_abs={err_abs:.2e}"

    def test_terminal_event_root_refinement(self):
        """Terminal events should stop at the root, not the late step endpoint."""
        def rhs(t, y):
            return jnp.array([-1.0])
        def event(t, y):
            return y[0] - 0.3
        r = jax_rodas5p_solve(
            rhs, jnp.array([1.0]), (0.0, 2.0),
            rtol=1e-8, atol=1e-10, h_max=0.5, event_fn=event
        )
        assert r.success
        assert abs(r.N_final - 0.7) < 1e-6, f"N_final={r.N_final}"
        assert abs(float(r.y_final[0]) - 0.3) < 1e-6, f"y_final={float(r.y_final[0])}"
        print(f"  Event root refinement: PASS (N={r.N_final:.8f}, y={float(r.y_final[0]):.8f})")


class TestPerformance:

    def test_ac6_jit_time(self):
        """AC6: Total JIT+first-solve < 30 s."""
        # Use a fresh function to force recompilation
        def fresh_rhs(t, y):
            return -y * 1.0001  # slightly different to avoid cache
        t0 = time.perf_counter()
        r = jax_rodas5p_solve(fresh_rhs, jnp.array([1.0]), (0.0, 1.0))
        jax.block_until_ready(r.y_final)
        dt = time.perf_counter() - t0
        assert dt < 30, f"JIT time = {dt:.1f} s"
        print(f"  AC6 JIT time: PASS ({dt:.2f} s)")


if __name__ == "__main__":
    print("J04 Verification: JAX Rodas5P Adaptive Solver")
    print("=" * 60)

    t1 = TestAnalyticalProblems()
    t1.test_exponential_decay_convergence()
    t1.test_stiff_2d_system()

    t2 = TestSciPyParity()
    t2.test_ac2_parity_rtol_1e8()
    t2.test_ac4_step_count_parity()
    t2.test_vanderpol_parity()

    t3 = TestSolverProperties()
    t3.test_ac5_low_rejection_rate()
    t3.test_n_final_exact()
    t3.test_ac7_mass_conservation()
    t3.test_success_flag()

    t4 = TestPerformance()
    t4.test_ac6_jit_time()

    print("=" * 60)
    print("ALL J04 VERIFICATION TESTS PASSED")
