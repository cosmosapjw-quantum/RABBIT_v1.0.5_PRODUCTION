"""tests/test_pstf_reduced_kernel_trilinear.py — Stage 0 trilinear reduced kernel F^(3).

The n=3 (large-anisotropy) reduced kernel folds the generalized HM angular polynomial
Pi^(3)(x,y) into the body-shape integral
(neutrino_large_anisotropy_exact_PSTF_collision_ko.md, eq 595-606):

    F^(3;ijk)_{L;li lj lk;J}(p1,p2,p3)
        = int dy int dx Theta(D) Theta(E4-m4)/sqrt(D) W_r(x,y) Pi^(3)(x,y).

Pi^(3) is evaluated numerically on the body frame, so the inner x-integral uses the
non-singular substitution x = x_c + r cos(theta) (dx/sqrt(D) = dtheta/sqrt(-a)) that the
analytic I_n is built on -- spectrally convergent, no 1/sqrt(D) endpoint blow-up.

Stage-0 GATE (kernel level), each an INDEPENDENT anchor:
  - all-monopole F^(3)_{0;000;0} = F_0 (Pi^(3)=1);
  - weak-anisotropy F^(3)_{L;L00} = F^(1)_{L,i} (two rank-0 legs -> P_L(mu_{1i})),
    matched to the separately-built Legendre-inserted kernel.
"""

from __future__ import annotations

import pytest

from rabbit.collisions.kernels import G_F_MEV
from rabbit.collisions.pstf_reduced_kernel import (
    reduced_kernel_F0,
    reduced_kernel_F_legendre,
    reduced_kernel_F3,
    reduced_kernel_F3_analytic,
)

_E = (1.2, 0.9, 0.8)
_P = (1.2, 0.9, 0.8)
_M4 = 0.0
_GF2 = G_F_MEV ** 2


def _nunu_xpoly(y):
    e1e2 = _E[0] * _E[1]
    p1p2 = _P[0] * _P[1]
    return (32 * _GF2 * e1e2 ** 2, -64 * _GF2 * e1e2 * p1p2, 32 * _GF2 * p1p2 ** 2)


def test_F3_all_monopole_equals_F0():
    """Pi^(3)_{0;000;0}=1 -> F^(3) reproduces the scalar F_0 (theta-quadrature vs I_n)."""
    F0 = reduced_kernel_F0(_E, _P, _M4, _nunu_xpoly, n_y=48)
    assert F0 != 0.0
    F3 = reduced_kernel_F3(0, 0, (0, 0, 0), (1, 2, 3), _E, _P, _M4, _nunu_xpoly,
                           n_y=48, n_theta=40)
    assert F3 / F0 == pytest.approx(1.0, rel=1e-9)  # ratio so the tolerance bites


def test_F3_two_monopole_legs_equals_legendre_kernel():
    """F^(3)_{L;L00} with legs (i,1,1) reduces to the Legendre kernel F_{L,i}."""
    for (leg_i, L) in [(2, 1), (2, 2), (3, 1), (4, 1), (4, 2)]:
        F1 = reduced_kernel_F_legendre(L, leg_i, _E, _P, _M4, _nunu_xpoly, n_y=48)
        assert F1 != 0.0
        F3 = reduced_kernel_F3(L, L, (L, 0, 0), (leg_i, 1, 1), _E, _P, _M4, _nunu_xpoly,
                               n_y=48, n_theta=48)
        assert F3 / F1 == pytest.approx(1.0, rel=1e-8), f"leg={leg_i} L={L}"  # ratio bites


def test_F3_genuine_trilinear_is_finite_real():
    # a genuinely tripolar entry (234, ranks 1,1,2) must be a finite real number
    F3 = reduced_kernel_F3(2, 2, (1, 1, 2), (2, 3, 4), _E, _P, _M4, _nunu_xpoly,
                           n_y=32, n_theta=32)
    assert isinstance(F3, float)
    assert abs(F3) < float("inf")


def test_F3_below_threshold_zero():
    assert reduced_kernel_F3(0, 0, (0, 0, 0), (1, 2, 3),
                             (0.5, 0.4, 0.8), (0.5, 0.4, 0.8), 2.0, _nunu_xpoly) == 0.0


def test_F3_analytic_matches_numerical_theta_quad():
    """CAS-exact analytic F^(3) (closed-form I_n) == numerical theta-quad F^(3) at high n_theta."""
    cases = [(0, 0, (0, 0, 0), (1, 2, 3)), (2, 2, (2, 0, 0), (2, 1, 1)),
             (2, 2, (1, 1, 2), (2, 3, 4)), (1, 1, (1, 1, 1), (2, 3, 4))]
    for (L, J, ranks, legs) in cases:
        ana = reduced_kernel_F3_analytic(L, J, ranks, legs, _E, _P, _M4, _nunu_xpoly, n_y=48)
        num = reduced_kernel_F3(L, J, ranks, legs, _E, _P, _M4, _nunu_xpoly, n_y=48, n_theta=64)
        # analytic is exact; numerical theta-quad converges to it -> agree at the quad precision
        assert ana == pytest.approx(num, rel=1e-5), f"{(L, J, ranks, legs)}: ana={ana} num={num}"


def test_F3_analytic_reductions_machine_precision():
    """Analytic F^(3) reductions to F_0 / F_legendre are MACHINE precision (both closed-form)."""
    F0 = reduced_kernel_F0(_E, _P, _M4, _nunu_xpoly, n_y=48)
    a000 = reduced_kernel_F3_analytic(0, 0, (0, 0, 0), (1, 2, 3), _E, _P, _M4, _nunu_xpoly, n_y=48)
    assert a000 / F0 == pytest.approx(1.0, rel=1e-12)
    for (leg, L) in [(2, 1), (2, 2), (4, 2)]:
        F1 = reduced_kernel_F_legendre(L, leg, _E, _P, _M4, _nunu_xpoly, n_y=48)
        a = reduced_kernel_F3_analytic(L, L, (L, 0, 0), (leg, 1, 1), _E, _P, _M4, _nunu_xpoly, n_y=48)
        assert a / F1 == pytest.approx(1.0, rel=1e-12), f"leg={leg} L={L}"
