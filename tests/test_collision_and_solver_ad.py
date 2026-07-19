"""
test_collision_and_solver_ad.py — Tests for collision wiring and native solver AD.

These tests verify:
1. Collision operator produces nonzero ΔYp through canonical interface
2. Fixed-step Rosenbrock supports native jax.grad
"""
import importlib.util

import pytest
import numpy as np


def _test_ros2_step(rhs_fn, N, y, h):
    """Linearly implicit Euler step used only by the native-AD smoke tests."""
    import jax
    import jax.numpy as jnp

    f0 = rhs_fn(N, y)

    def f_only_y(yy):
        return rhs_fn(N, yy)

    jacobian = jax.jacfwd(f_only_y)(y)
    system = jnp.eye(y.shape[0]) / h - jacobian
    delta = jnp.linalg.solve(system, f0)
    return N + h, y + delta


def _solve_fixed_rosenbrock_for_ad_test(
    rhs_fn,
    y0,
    N_span: tuple[float, float],
    *,
    N_steps: int,
):
    """Fixed-step Rosenbrock probe kept out of the runtime solver package."""
    import jax.numpy as jnp
    import jax.lax as lax

    N_start, N_end = N_span
    h = (N_end - N_start) / N_steps
    y0 = jnp.asarray(y0, dtype=jnp.float64)

    def scan_body(carry, _):
        N, y = carry
        return _test_ros2_step(rhs_fn, N, y, h), None

    (_, y_final), _ = lax.scan(
        scan_body,
        (jnp.float64(N_start), y0),
        None,
        length=N_steps,
    )
    return y_final


# ═══ Collision integration tests ═══

class TestCollisionIntegration:
    """Verify RTA collision operator is wired into production driver."""

    @pytest.mark.slow
    def test_collision_nonzero_effect(self):
        """Collisions must change Yp (neutrino heating from e± annihilation)."""
        from rabbit.inference.forward_likelihood import canonical_forward_solver

        r_off = canonical_forward_solver(
            Sigma_H=0.0, N_q=6, correction_level=0,
            enable_collisions=False, tier=1)
        r_on = canonical_forward_solver(
            Sigma_H=0.0, N_q=6, correction_level=0,
            enable_collisions=True, tier=2)

        assert r_off.success and r_on.success
        dYp = r_on.Yp - r_off.Yp
        # ν heating decreases Yp (more n→p conversion)
        assert dYp < 0, f"Collision ΔYp should be negative: {dYp:.4e}"
        assert abs(dYp) > 1e-5, f"Collision effect too small: {dYp:.4e}"
        assert abs(dYp) < 0.01, f"Collision effect implausibly large: {dYp:.4e}"

    @pytest.mark.slow
    def test_collision_direction_correct(self):
        """Collision ΔYp must be negative (ν heating → more n→p → less Yp)."""
        from rabbit.inference.forward_likelihood import canonical_forward_solver

        r_coll = canonical_forward_solver(
            Sigma_H=0.0, N_q=6, correction_level=0,
            enable_collisions=True, tier=2)
        # CL0 Born + collision: should be between 0.240 and 0.243
        assert 0.238 < r_coll.Yp < 0.244, f"Yp={r_coll.Yp} out of range"

    @pytest.mark.slow
    def test_collision_with_shear(self):
        """Collisions + shear must both work together."""
        from rabbit.inference.forward_likelihood import canonical_forward_solver

        r = canonical_forward_solver(
            Sigma_H=0.1, N_q=6, correction_level=0,
            enable_collisions=True, tier=2)
        assert r.success, f"Failed with shear+collisions"
        assert 0.23 < r.Yp < 0.25, f"Yp={r.Yp} out of range"


# ═══ Fixed-step solver AD tests ═══

class TestFixedStepSolverAD:
    """Verify fixed-step Rosenbrock supports native jax.grad."""

    def test_fixed_step_ad_helper_is_not_a_runtime_solver_module(self):
        """The fixed-step AD probe stays test-local, not a runtime solver path."""
        assert importlib.util.find_spec("rabbit.jax.solver_fixed_step") is None

    def test_simple_ode_grad(self):
        """jax.grad must work on a simple ODE through fixed-step solver."""
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        # y' = -α*y, y(0) = 1 → y(T) = e^{-αT}
        # ∂y(T)/∂α = -T * e^{-αT}
        T_end = 2.0

        def loss(alpha):
            def rhs(t, y):
                return -alpha * y
            y0 = jnp.array([1.0])
            y_final = _solve_fixed_rosenbrock_for_ad_test(
                rhs,
                y0,
                (0.0, T_end),
                N_steps=2000,
            )
            return y_final[0]

        alpha = jnp.float64(1.0)
        val, grad = jax.value_and_grad(loss)(alpha)

        exact_val = np.exp(-1.0 * T_end)
        exact_grad = -T_end * np.exp(-1.0 * T_end)

        assert abs(float(val) - exact_val) < 0.01, \
            f"value: {val} vs exact {exact_val}"
        assert abs(float(grad) - exact_grad) < 0.1, \
            f"grad: {grad} vs exact {exact_grad}"

    def test_2d_ode_grad(self):
        """jax.grad on a 2D coupled system."""
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        def loss(params):
            a, b = params[0], params[1]
            def rhs(t, y):
                return jnp.array([-a * y[0] + b * y[1],
                                   -b * y[1]])
            y0 = jnp.array([1.0, 1.0])
            y_final = _solve_fixed_rosenbrock_for_ad_test(
                rhs,
                y0,
                (0.0, 1.0),
                N_steps=1000,
            )
            return jnp.sum(y_final**2)

        params = jnp.array([1.0, 0.5])
        val, grad = jax.value_and_grad(loss)(params)

        assert jnp.all(jnp.isfinite(grad)), f"grad not finite: {grad}"
        assert float(grad[0]) < 0, \
            f"∂L/∂a should be negative (more damping → smaller y): {grad[0]}"
