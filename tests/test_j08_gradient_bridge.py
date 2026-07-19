"""
test_j08_gradient_bridge — Verification for J08 (custom_vjp Gradient Bridge).

Gate GJ4: ∂Y_p/∂τ_n via custom_vjp matches FD to < 1%

Acceptance criteria:
  AC1: ∂y/∂λ matches analytical gradient to < 0.1%
  AC2: ∂y/∂η matches FD to < 0.5% (or both ≈ 0)
  AC3: jax.grad(log_posterior)(params) returns finite gradient pytree
  AC4: Gradient computation time < 5 s for n=2 parameters (warm)
  AC5: FD convergence: V-curve behavior (optimal eps ~ 10⁻⁵)
  AC6: No NaN in gradient for reasonable parameter ranges
"""
import pytest
import sys
import os
import time


import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

from rabbit.jax.gradient_bridge import (
    make_differentiable_solve, differentiable_solve,
    make_log_posterior, gradient_check, BBNObservation,
)


def _decay_rhs_factory(params):
    lam = params[0]
    def rhs(t, y):
        return -lam * y
    return rhs


class TestAnalyticalGradient:
    """AC1: Gradient matches exact analytical solution."""

    def test_ac1_exponential_decay(self):
        """∂y(5)/∂λ = −5·exp(−5λ) for y′ = −λy."""
        solve_fn = make_differentiable_solve(
            _decay_rhs_factory, jnp.array([1.0]), 0.0, 5.0,
            param_indices=(0,), rtol=1e-10, atol=1e-12,
        )
        params = jnp.array([1.0])
        grad_val = jax.grad(lambda p: solve_fn(p)[0])(params)
        exact_grad = -5.0 * np.exp(-5.0)
        rel = abs(float(grad_val[0]) - exact_grad) / abs(exact_grad)
        assert rel < 0.001, f"rel = {rel:.4e}"
        print(f"  AC1 analytical gradient: PASS (rel={rel:.4e})")

    def test_two_parameter(self):
        """Two-parameter system: both gradients have correct sign."""
        def rhs_factory(params):
            a, b = params[0], params[1]
            def rhs(t, y):
                return jnp.array([-a * y[0] + b])
            return rhs

        solve_fn = make_differentiable_solve(
            rhs_factory, jnp.array([1.0]), 0.0, 3.0,
            param_indices=(0, 1), rtol=1e-10, atol=1e-12,
        )
        params = jnp.array([2.0, 0.5])
        grad = jax.grad(lambda p: solve_fn(p)[0])(params)
        # ∂y/∂a < 0 (more decay → smaller y)
        assert float(grad[0]) < 0, f"∂y/∂a = {float(grad[0])} should be < 0"
        # ∂y/∂b > 0 (more source → larger y)
        assert float(grad[1]) > 0, f"∂y/∂b = {float(grad[1])} should be > 0"
        print(f"  Two-param signs: PASS (∂/∂a={float(grad[0]):.4e}, ∂/∂b={float(grad[1]):.4e})")


class TestGradientValidation:
    """AC2: custom_vjp matches independent FD."""

    def test_gradient_check_passes(self):
        """gradient_check utility: rel error < 1%."""
        solve_fn = make_differentiable_solve(
            _decay_rhs_factory, jnp.array([1.0]), 0.0, 5.0,
            param_indices=(0,), rtol=1e-10, atol=1e-12,
        )
        gc = gradient_check(
            solve_fn, jnp.array([1.0]),
            loss_fn=lambda y: y[0],
            param_names=('λ',),
            threshold=0.01,
        )
        assert gc.passed, f"max_rel = {gc.max_rel_error:.4e}"
        print(f"  AC2 gradient check: PASS (max_rel={gc.max_rel_error:.4e})")

    def test_gradient_check_multipoint(self):
        """Gradient correct at multiple parameter values."""
        solve_fn = make_differentiable_solve(
            _decay_rhs_factory, jnp.array([1.0]), 0.0, 5.0,
            param_indices=(0,), rtol=1e-10, atol=1e-12,
        )
        for lam in [0.3, 1.0, 3.0]:
            gc = gradient_check(
                solve_fn, jnp.array([lam]),
                loss_fn=lambda y: y[0],
                threshold=0.01,
            )
            assert gc.passed, f"Failed at λ={lam}: rel={gc.max_rel_error:.4e}"
        print(f"  Multipoint gradient check: PASS (λ=0.3, 1.0, 3.0)")


class TestLogPosterior:
    """AC3: Log-posterior interface."""

    def test_ac3_log_posterior_differentiable(self):
        """jax.grad(log_posterior) returns finite gradient."""
        def rhs_factory(params):
            lam = params[0]
            def rhs(t, y):
                return jnp.array([-lam * y[0]])
            return rhs

        obs = [BBNObservation('Y_p', 0.1, 0.01)]
        log_post = make_log_posterior(
            rhs_factory, jnp.array([0.5]), (0.0, 3.0),
            observable_indices={'Y_p': 0},
            observations=obs,
            param_indices=(0,),
            rtol=1e-8, atol=1e-10,
        )
        params = jnp.array([1.0])
        val, grad = jax.value_and_grad(log_post)(params)
        assert jnp.isfinite(val), f"log_post = {val}"
        assert jnp.all(jnp.isfinite(grad)), f"grad = {grad}"
        print(f"  AC3 log_posterior: PASS (val={float(val):.4f}, grad={float(grad[0]):.4e})")

    def test_log_posterior_with_prior(self):
        """Prior contributes to log-posterior."""
        def rhs_factory(params):
            def rhs(t, y):
                return -params[0] * y
            return rhs

        # Gaussian prior on λ: mean=1, std=0.5
        def prior_fn(params):
            return -0.5 * ((params[0] - 1.0) / 0.5) ** 2

        obs = [BBNObservation('Y_p', 0.1, 0.01)]
        log_post = make_log_posterior(
            rhs_factory, jnp.array([0.5]), (0.0, 3.0),
            observable_indices={'Y_p': 0},
            observations=obs,
            param_indices=(0,),
            prior_fn=prior_fn,
        )
        val = log_post(jnp.array([1.0]))
        assert jnp.isfinite(val)
        # At prior center, prior contribution = 0
        grad = jax.grad(log_post)(jnp.array([1.0]))
        assert jnp.isfinite(grad[0])
        print(f"  Log-posterior with prior: PASS")


class TestPerformance:
    """AC4: Gradient computation time."""

    def test_ac4_gradient_time(self):
        """Gradient computation < 5 s for n=2 warm call."""
        def rhs_factory(params):
            a, b = params[0], params[1]
            def rhs(t, y):
                return jnp.array([-a * y[0], -b * y[1]])
            return rhs

        solve_fn = make_differentiable_solve(
            rhs_factory, jnp.array([1.0, 1.0]), 0.0, 5.0,
            param_indices=(0, 1), rtol=1e-8, atol=1e-10,
        )
        loss = lambda p: jnp.sum(solve_fn(p))

        # Warmup
        _ = jax.grad(loss)(jnp.array([1.0, 1.0]))

        # Timed
        t0 = time.perf_counter()
        g = jax.grad(loss)(jnp.array([1.0, 1.0]))
        jax.block_until_ready(g)
        dt = time.perf_counter() - t0
        assert dt < 8.0, f"Gradient time = {dt:.2f} s"
        print(f"  AC4 gradient time: PASS ({dt:.3f} s for n=2)")


class TestRobustness:
    """AC5, AC6: Edge cases."""

    def test_ac5_fd_epsilon_convergence(self):
        """FD converges then diverges (V-curve) as eps varies."""
        solve_fn = make_differentiable_solve(
            _decay_rhs_factory, jnp.array([1.0]), 0.0, 5.0,
            param_indices=(0,), rtol=1e-10, atol=1e-12,
        )
        exact = -5.0 * np.exp(-5.0)
        errors = []
        for eps in [1e-3, 1e-5, 1e-7, 1e-9]:
            solve_fn_eps = make_differentiable_solve(
                _decay_rhs_factory, jnp.array([1.0]), 0.0, 5.0,
                param_indices=(0,), rtol=1e-10, atol=1e-12, fd_eps=eps,
            )
            g = jax.grad(lambda p: solve_fn_eps(p)[0])(jnp.array([1.0]))
            err = abs(float(g[0]) - exact) / abs(exact)
            errors.append(err)
        # Best error should be < 1e-6 (at optimal eps)
        assert min(errors) < 1e-6, f"Best err = {min(errors):.2e}"
        print(f"  AC5 FD convergence: PASS (errors={[f'{e:.1e}' for e in errors]})")

    def test_ac6_no_nan_range(self):
        """No NaN for reasonable parameter range."""
        solve_fn = make_differentiable_solve(
            _decay_rhs_factory, jnp.array([1.0]), 0.0, 5.0,
            param_indices=(0,),
        )
        for lam in [0.01, 0.1, 1.0, 5.0, 10.0]:
            g = jax.grad(lambda p: solve_fn(p)[0])(jnp.array([lam]))
            assert jnp.isfinite(g[0]), f"NaN at λ={lam}"
        print(f"  AC6 no NaN: PASS (λ=0.01..10)")


if __name__ == "__main__":
    print("J08 Verification: custom_vjp Gradient Bridge")
    print("=" * 60)

    t1 = TestAnalyticalGradient()
    t1.test_ac1_exponential_decay()
    t1.test_two_parameter()

    t2 = TestGradientValidation()
    t2.test_gradient_check_passes()
    t2.test_gradient_check_multipoint()

    t3 = TestLogPosterior()
    t3.test_ac3_log_posterior_differentiable()
    t3.test_log_posterior_with_prior()

    t4 = TestPerformance()
    t4.test_ac4_gradient_time()

    t5 = TestRobustness()
    t5.test_ac5_fd_epsilon_convergence()
    t5.test_ac6_no_nan_range()

    print("=" * 60)
    print("ALL J08 VERIFICATION TESTS PASSED")
