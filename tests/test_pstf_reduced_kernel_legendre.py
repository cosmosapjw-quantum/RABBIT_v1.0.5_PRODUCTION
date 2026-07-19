"""tests/test_pstf_reduced_kernel_legendre.py — Stage 0 Legendre-inserted kernel F_{ell,i}.

The n=1 (linearized / external-leg) reduced kernel is the HM scalar kernel with a
Legendre weight on the relative angle mu_{1i} = n_1 . n_i inserted
(neutrino_decoupling_PSTF_HM_generalization_ko.md, eq 794-841):

    F_{ell,i}^(r)(p1,p2,p3) = int dy int dx Theta(D) Theta(E4-m4)/sqrt(D)
                                  W_r(x,y) P_ell(mu_{1i}),

with mu_{11}=1, mu_{12}=x, mu_{13}=y, mu_{14}=(p1+p2 x - p3 y)/p4 -- each at most linear
in x, so W_r P_ell is still polynomial in x and the inner integral stays the analytic
I_n. These F_{ell,i} are the F^(1) inputs the LRS moment contraction consumes.

Self-consistency anchors (exact): F_{0,i} = F_0 for every leg (P_0 = 1), and
F_{ell,1} = F_0 for every ell (mu_{11}=1 -> P_ell(1)=1, eq 832/838).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_legendre

from rabbit.collisions.kernels import G_F_MEV
from rabbit.collisions.pstf_reduced_kernel import (
    dee_coefficients,
    reduced_kernel_F0,
    reduced_kernel_F_legendre,
)

_E = (1.2, 0.9, 0.8)
_P = (1.2, 0.9, 0.8)
_M4 = 0.0
_GF2 = G_F_MEV ** 2


def _nunu_xpoly(y):
    e1e2 = _E[0] * _E[1]
    p1p2 = _P[0] * _P[1]
    return (32 * _GF2 * e1e2 ** 2, -64 * _GF2 * e1e2 * p1p2, 32 * _GF2 * p1p2 ** 2)


def test_F_ell0_equals_F0_all_legs():
    """ell=0 (P_0=1) reproduces the scalar F_0 for every leg."""
    F0 = reduced_kernel_F0(_E, _P, _M4, _nunu_xpoly, n_y=48)
    for leg in (1, 2, 3, 4):
        Fl = reduced_kernel_F_legendre(0, leg, _E, _P, _M4, _nunu_xpoly, n_y=48)
        assert Fl / F0 == pytest.approx(1.0, rel=1e-12)  # ratio so the tolerance bites


def test_F_ell_leg1_equals_F0_all_ell():
    """mu_{11}=1 -> P_ell(1)=1, so the leg-1 Legendre kernel collapses to F_0."""
    F0 = reduced_kernel_F0(_E, _P, _M4, _nunu_xpoly, n_y=48)
    for ell in (1, 2, 3):
        Fl = reduced_kernel_F_legendre(ell, 1, _E, _P, _M4, _nunu_xpoly, n_y=48)
        assert Fl / F0 == pytest.approx(1.0, rel=1e-12)  # ratio so the tolerance bites


def _F_legendre_quad_reference(ell, leg, n_y):
    """F_{ell,i} with the x-integral by scipy.quad on the NON-SINGULAR theta substitution
    x = x_c + r cos(theta) (independent of the analytic I_n); machine-accurate (a naive
    quad over the 1/sqrt(D) endpoint singularity is only ~2-6% accurate)."""
    E1, E2, E3 = _E
    p1, p2, p3 = _P
    p4 = math.sqrt((E1 + E2 - E3) ** 2 - _M4 ** 2)  # on-shell leg-4 momentum
    nodes, weights = np.polynomial.legendre.leggauss(n_y)
    total = 0.0
    for yk, wk in zip(nodes, weights):
        a, b, c = dee_coefficients(yk, _E, _P, _M4)
        disc = b * b - 4 * a * c
        if a >= 0 or disc <= 0:
            continue
        r = math.sqrt(disc) / (2 * abs(a))
        xc = -b / (2 * a)
        lo, hi = max(-1.0, xc - r), min(1.0, xc + r)
        if lo >= hi:
            continue
        theta_lo = math.acos(min(1.0, max(-1.0, (lo - xc) / r)))
        theta_hi = math.acos(min(1.0, max(-1.0, (hi - xc) / r)))
        inv = 1.0 / math.sqrt(-a)
        A = _nunu_xpoly(yk)

        def mu1i(x):
            if leg == 1:
                return 1.0
            if leg == 2:
                return x
            if leg == 3:
                return yk
            return (p1 + p2 * x - p3 * yk) / p4

        def integrand(th):
            x = xc + r * math.cos(th)
            w = sum(A[n] * x ** n for n in range(len(A)))
            return w * eval_legendre(ell, mu1i(x)) * inv

        val, _ = quad(integrand, theta_hi, theta_lo)
        total += wk * val
    return total


def test_F_legendre_leg2_matches_quadrature():
    for ell in (1, 2, 3):
        Fl = reduced_kernel_F_legendre(ell, 2, _E, _P, _M4, _nunu_xpoly, n_y=64)
        ref = _F_legendre_quad_reference(ell, 2, 64)
        assert ref != 0.0
        assert Fl / ref == pytest.approx(1.0, rel=1e-9)  # ratio so the tolerance bites


def test_F_legendre_leg4_matches_quadrature():
    for ell in (1, 2):
        Fl = reduced_kernel_F_legendre(ell, 4, _E, _P, _M4, _nunu_xpoly, n_y=64)
        ref = _F_legendre_quad_reference(ell, 4, 64)
        assert ref != 0.0
        assert Fl / ref == pytest.approx(1.0, rel=1e-9)  # ratio so the tolerance bites
