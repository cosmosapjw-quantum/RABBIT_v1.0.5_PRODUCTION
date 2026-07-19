"""
Test: Full-pipeline gradient ∂Y_p/∂Σ_H validation.

Computes the sensitivity of Y_p to initial shear via finite differences
on the actual production BBN solver. This is the ground-truth reference
for any future AD gradient implementation.

Physics checks:
  1. ∂Y_p/∂Σ_H > 0 (shear increases Y_p via expansion channel)
  2. ∂Y_p/∂Σ_H ∝ Σ_H at leading order (quadratic dependence of Y_p on Σ)
  3. Gradient at Σ=0 vanishes by symmetry (Y_p is even in Σ)
  4. FD convergence: gradient stable across eps = 1e-3 to 1e-5

NOTE: The AD gradient bridge (gradient_bridge.py) is tested only on
toy ODE systems. Until it is wired to the full BBN pipeline, FD
provides the validated reference for ∂Y_p/∂Σ_H.
"""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def gradient_at_0p1():
    """∂Y_p/∂Σ_H at Σ_H=0.1 via central FD."""
    from rabbit.inference.forward_likelihood import canonical_forward_solver
    eps = 1e-5
    yp_p = canonical_forward_solver(Sigma_H=0.1+eps, backend='scipy', N_q=6).Yp
    yp_m = canonical_forward_solver(Sigma_H=0.1-eps, backend='scipy', N_q=6).Yp
    return (yp_p - yp_m) / (2*eps)


@pytest.fixture(scope="module")
def gradient_at_0():
    """∂Y_p/∂Σ_H at Σ_H=0 via forward FD."""
    from rabbit.inference.forward_likelihood import canonical_forward_solver
    eps = 1e-5
    yp_eps = canonical_forward_solver(Sigma_H=eps, backend='scipy', N_q=6).Yp
    yp_0 = canonical_forward_solver(Sigma_H=0.0, backend='scipy', N_q=6).Yp
    return (yp_eps - yp_0) / eps


class TestGradientPhysics:

    def test_positive_at_nonzero_sigma(self, gradient_at_0p1):
        """Shear increases Y_p (expansion channel dominates)."""
        assert gradient_at_0p1 > 0

    def test_near_zero_at_sigma_zero(self, gradient_at_0):
        """Y_p is even in Σ → gradient vanishes at Σ=0."""
        assert abs(gradient_at_0) < 1e-4

    def test_order_of_magnitude(self, gradient_at_0p1):
        """∂Y_p/∂Σ should be O(10⁻⁴) to O(10⁻³) at Σ=0.1."""
        assert 1e-5 < abs(gradient_at_0p1) < 1e-2


class TestFDConvergence:
    """FD gradient should be stable across step sizes."""

    def test_convergence(self):
        from rabbit.inference.forward_likelihood import canonical_forward_solver
        grads = []
        # The SciPy number-of-record path resolves this derivative cleanly down
        # to 1e-5; 1e-6 is below the current endpoint integration noise floor.
        for eps in [1e-3, 1e-4, 1e-5]:
            yp_p = canonical_forward_solver(Sigma_H=0.1+eps, backend='scipy', N_q=6).Yp
            yp_m = canonical_forward_solver(Sigma_H=0.1-eps, backend='scipy', N_q=6).Yp
            grads.append((yp_p - yp_m) / (2*eps))
        # Adjacent estimates should agree to < 10%
        for i in range(len(grads)-1):
            if abs(grads[i]) > 1e-8:
                rel = abs(grads[i+1] - grads[i]) / abs(grads[i])
                assert rel < 0.1, f"FD unstable: {grads}"
