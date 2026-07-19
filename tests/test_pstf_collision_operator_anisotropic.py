"""tests/test_pstf_collision_operator_anisotropic.py — Stage 2 LRS L>0 collision operator.

Extends the isotropic operator to the full axisymmetric multipole hierarchy
(neutrino_large_anisotropy_exact_PSTF_collision_ko.md, eq 1111-1135):

    Chat_L(p1) = (2/(2pi)^4)(1/2E1)(1/8) int p2^2 dp2/2E2 int p3^2 dp3/2E3 bracket_L,

with bracket_L the m=0 PSTF contraction (pstf_lrs_moment_kernel) of the reduced kernels
F^(1)/F^(3) against the axisymmetric amplitudes Gamma^(i)_l = g^(i)_{l0}/sqrt(4pi) of each
leg's distribution. This is the LRS production engine.

Cheap rigorous gates (all-monopole inputs only touch the L=0 kernels):
  - isotropic reduction: with only Gamma_0 the L=0 moment equals the validated isotropic
    operator, and Chat_{L>0} = 0 (the m=0 contraction zeroes L>0 at all-monopole);
  - detailed balance: at Fermi-Dirac equilibrium bracket_0 = 8 Lambda_r F_0 = 0, so
    Chat_L = 0 for every L.
The high-ell shear response (expensive trilinear F^(3)) is exercised separately.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rabbit.collisions.kernels import G_F_MEV
from rabbit.collisions.pstf_collision_operator import (
    isotropic_collision_moment_C0,
    lrs_collision_moment_CL,
)

_GF2 = G_F_MEV ** 2


def _nunu_xpoly_factory(p1, p2, p3):
    p1p2 = p1 * p2
    coeffs = (32 * _GF2 * p1p2 ** 2, -64 * _GF2 * p1p2 ** 2, 32 * _GF2 * p1p2 ** 2)
    return lambda y: coeffs


def _fermi_dirac(mu=0.0, T=1.0):
    return lambda p: 1.0 / (1.0 + math.exp((p - mu) / T))


def _momentum_grid(n=6, pmin=0.3, pmax=6.0):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (pmax - pmin)
    mid = 0.5 * (pmax + pmin)
    return mid + half * nodes, half * weights


def _isotropic_gamma(f):
    """Gamma spectrum with only the monopole Gamma_0 = G_0 = 1 - 2 f."""
    return lambda p: np.array([1.0 - 2.0 * f(p)])


def test_isotropic_reduction_to_C0():
    """All-monopole: Chat_0 (L=0) equals the validated isotropic operator."""
    p_nodes, p_weights = _momentum_grid()
    f = _fermi_dirac()
    # break equilibrium so C0 is non-zero and the comparison is meaningful
    g = lambda p: min(0.999, f(p) * (1.0 + 0.25 * math.exp(-((p - 1.5) ** 2))))
    gamma = _isotropic_gamma(g)
    for p1 in (0.8, 2.0):
        c_iso = isotropic_collision_moment_C0(g, p1, p_nodes, p_weights, _nunu_xpoly_factory, n_y=24)
        c_L0 = lrs_collision_moment_CL(0, gamma, p1, p_nodes, p_weights, _nunu_xpoly_factory,
                                       ell_max=0, n_y=24, n_theta=20)
        assert c_iso != 0.0
        # ratio vs 1.0 so the tolerance bites (~1e-24 values swamp pytest.approx's abs default)
        assert c_L0 / c_iso == pytest.approx(1.0, rel=1e-9)


def test_isotropic_reduction_with_high_ell_kernels_computed():
    """ell_max=1 forces the trilinear F^(3) high-ell kernels to be COMPUTED; with a
    monopole-only amplitude they must contract to zero and leave Chat_0 = the isotropic
    operator. Exercises the real F^(3) path (the all-monopole/DB tests do not)."""
    p_nodes, p_weights = _momentum_grid(n=5)
    g = lambda p: min(0.999, _fermi_dirac()(p) * (1.0 + 0.25 * math.exp(-((p - 1.5) ** 2))))
    gamma = _isotropic_gamma(g)  # length-1 spectrum -> monopole only, ell_max=1 allowed
    c_iso = isotropic_collision_moment_C0(g, 1.5, p_nodes, p_weights, _nunu_xpoly_factory, n_y=20)
    c_L0 = lrs_collision_moment_CL(0, gamma, 1.5, p_nodes, p_weights, _nunu_xpoly_factory,
                                   ell_max=1, n_y=20, n_theta=16)
    assert c_iso != 0.0
    assert c_L0 / c_iso == pytest.approx(1.0, rel=1e-8)


def test_isotropic_higher_moments_vanish():
    """All-monopole: Chat_{L>0} = 0 (the m=0 contraction zeroes L>0)."""
    p_nodes, p_weights = _momentum_grid()
    g = lambda p: min(0.999, _fermi_dirac()(p) * (1.0 + 0.25 * math.exp(-((p - 1.5) ** 2))))
    gamma = _isotropic_gamma(g)
    for L in (1, 2):
        cL = lrs_collision_moment_CL(L, gamma, 1.5, p_nodes, p_weights, _nunu_xpoly_factory,
                                     ell_max=0, n_y=24, n_theta=20)
        assert abs(cL) < 1e-30


@pytest.mark.slow
def test_analytic_f3_matches_numerical_operator():
    """use_analytic_f3 (CAS-exact F^(3) where generated) matches the numerical operator within
    the theta-quadrature precision -- the analytic kernel is exact, the numerical converges to it.

    Slow: the numerical-fallback path runs the full theta-quadrature F^(3) at ell_max=1."""
    p_nodes, p_weights = _momentum_grid(n=5)
    fd = _fermi_dirac()
    gamma = lambda p: np.array([1.0 - 2.0 * fd(p), 0.3 * math.exp(-((p - 1.5) ** 2))])
    common = dict(ell_max=1, n_y=20, n_theta=48)
    c_num = lrs_collision_moment_CL(1, gamma, 1.5, p_nodes, p_weights, _nunu_xpoly_factory,
                                    use_analytic_f3=False, **common)
    c_ana = lrs_collision_moment_CL(1, gamma, 1.5, p_nodes, p_weights, _nunu_xpoly_factory,
                                    use_analytic_f3=True, **common)
    assert abs(c_num) > 1e-26
    assert c_ana == pytest.approx(c_num, rel=1e-4)


def test_ell_max_truncation_fails_loud():
    """Amplitude carrying multipoles beyond ell_max must raise, not silently truncate."""
    p_nodes, p_weights = _momentum_grid()
    f = _fermi_dirac()
    # dipole-carrying spectrum (rank 1) but ell_max=0 -> silent loss -> must raise
    gamma = lambda p: np.array([1.0 - 2.0 * f(p), 0.05])
    with pytest.raises(ValueError):
        lrs_collision_moment_CL(1, gamma, 1.5, p_nodes, p_weights, _nunu_xpoly_factory,
                                ell_max=0, n_y=16, n_theta=12)


def test_detailed_balance_all_L():
    """Fermi-Dirac equilibrium -> bracket_0 = 8 Lambda_r F_0 = 0 -> Chat_L = 0 for all L."""
    p_nodes, p_weights = _momentum_grid()
    f = _fermi_dirac(mu=0.2, T=1.1)
    gamma = _isotropic_gamma(f)
    for L in (0, 1, 2):
        cL = lrs_collision_moment_CL(L, gamma, 1.5, p_nodes, p_weights, _nunu_xpoly_factory,
                                     ell_max=0, n_y=24, n_theta=20)
        assert abs(cL) < 1e-30, f"L={L}: {cL}"
