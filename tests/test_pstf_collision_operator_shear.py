"""tests/test_pstf_collision_operator_shear.py — Stage 2 nonperturbative shear response.

Demonstrates that the LRS collision operator captures the anisotropic (shear) response
that the isotropic operator cannot: with a dipole amplitude Gamma_1 != 0 the L=1 moment
Chat_1 is non-zero, and the exact trilinear kernel makes it NONPERTURBATIVE in the
anisotropy amplitude.

By parity (L + l_i + l_j + l_k even), a single dipole amplitude alpha gives
    Chat_1(alpha) = a alpha + b alpha^3   (no even powers),
so Chat_1(2 alpha)/Chat_1(alpha) = 2 only if b = 0 (pure linearization). The exact
operator has b != 0: the linear part a is well converged (~0.3% across resolution) and
the cubic part b is robustly NEGATIVE (saturation) -- so the ratio is strictly below 2.
The precise cubic magnitude needs finer quadrature than a unit test affords (it is ~1% of
the linear response); that convergence belongs to the transport-wired Stage-2 study. Here
we pin the robust, resolution-stable facts: non-zero dipole response + sub-linear (b<0).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rabbit.collisions.kernels import G_F_MEV
from rabbit.collisions.pstf_collision_operator import lrs_collision_moment_CL

_GF2 = G_F_MEV ** 2


def _nunu_xpoly_factory(p1, p2, p3):
    p1p2 = p1 * p2
    coeffs = (32 * _GF2 * p1p2 ** 2, -64 * _GF2 * p1p2 ** 2, 32 * _GF2 * p1p2 ** 2)
    return lambda y: coeffs


def _momentum_grid(n=5, pmin=0.3, pmax=6.0):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (pmax - pmin)
    mid = 0.5 * (pmax + pmin)
    return mid + half * nodes, half * weights


def _dipole_gamma(alpha):
    """Monopole G_0 = 1 - 2 f_FD plus a dipole amplitude alpha * exp(-(p-1.5)^2)."""
    fd = lambda p: 1.0 / (1.0 + math.exp(p))
    return lambda p: np.array([1.0 - 2.0 * fd(p), alpha * math.exp(-((p - 1.5) ** 2))])


def test_dipole_response_is_finite_nonzero():
    """A dipole anisotropy drives a finite non-zero L=1 moment (isotropic gives 0)."""
    p_nodes, p_weights = _momentum_grid()
    c1 = lrs_collision_moment_CL(1, _dipole_gamma(0.2), 1.5, p_nodes, p_weights,
                                 _nunu_xpoly_factory, ell_max=1, n_y=14, n_theta=12)
    assert math.isfinite(c1)
    assert abs(c1) > 1e-26


def _analytic_ratio(p_nodes, p_weights, n_y):
    c_a = lrs_collision_moment_CL(1, _dipole_gamma(0.2), 1.5, p_nodes, p_weights,
                                  _nunu_xpoly_factory, ell_max=1, n_y=n_y, use_analytic_f3=True)
    c_2a = lrs_collision_moment_CL(1, _dipole_gamma(0.4), 1.5, p_nodes, p_weights,
                                   _nunu_xpoly_factory, ell_max=1, n_y=n_y, use_analytic_f3=True)
    return c_2a / c_a


def test_response_is_nonperturbative_sublinear():
    """Chat_1(2a)/Chat_1(a) < 2: the trilinear kernel saturates the response (b<0).

    Pure linearization gives exactly 2; the exact operator is strictly below it. With the
    CAS-EXACT F^(3) (use_analytic_f3) the cubic is no longer theta-quadrature-limited -- the
    ratio is the converged value ~1.9445, n_y-stable (the earlier numerical kernel drifted with
    n_theta). The tight band excludes both the linear value 2.0 and any quadrature artifact.
    """
    p_nodes, p_weights = _momentum_grid()
    ratio = _analytic_ratio(p_nodes, p_weights, n_y=24)
    assert 1.93 < ratio < 1.96  # converged sub-linear cubic (exact F^(3), no theta-quad drift)


def test_nonperturbative_cubic_is_ny_converged():
    """The analytic (exact-F^(3)) cubic ratio is stable under n_y refinement -- a genuine
    convergence, not the n_theta-limited drift of the numerical kernel."""
    p_nodes, p_weights = _momentum_grid()
    r16 = _analytic_ratio(p_nodes, p_weights, n_y=16)
    r32 = _analytic_ratio(p_nodes, p_weights, n_y=32)
    assert abs(r16 - r32) < 5e-4, f"analytic cubic not n_y-converged: {r16} vs {r32}"
