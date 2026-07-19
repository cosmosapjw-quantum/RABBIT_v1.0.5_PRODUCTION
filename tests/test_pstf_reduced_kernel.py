"""tests/test_pstf_reduced_kernel.py — Stage 0 isotropic reduced kernel F_0 assembly.

Closes the radial half of Stage 0: the isotropic HM scalar reduced kernel
(neutrino_decoupling_PSTF_HM_generalization_ko.md, eq 612-632)

    F_0^(r)(p1,p2,p3) = int_{-1}^1 dy int_{-1}^1 dx Theta(D) Theta(E4-m4)/sqrt(D) W_r(x,y),

with D = a(y) x^2 + b(y) x + c(y). Every HM invariant chi_ij is at most LINEAR in x
(eq 586-602), so W_r is at most quadratic in x and the inner integral is the analytic
I_n closed form (pstf_x_integral); only the 1-D y integral is numerical (Gauss-Legendre).

Pinned against: (1) the tested body-frame Jacobian D (a x^2+b x+c == frame.D); (2) the
W_r x-polynomial vs chi_invariants; (3) analytic-x F_0 == per-y scipy.quad of W_r/sqrt(D)
(independent x-integration); (4) F_0 > 0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from rabbit.collisions.kernels import G_F_MEV
from rabbit.collisions.pstf_body_frame import collision_body_frame
from rabbit.collisions.pstf_reduced_kernel import (
    dee_coefficients,
    reduced_kernel_F0,
)

# massless asymmetric config (p4 = 1.3)
_E = (1.2, 0.9, 0.8)
_P = (1.2, 0.9, 0.8)
_M4 = 0.0
_GF2 = G_F_MEV ** 2


def _nunu_xpoly(y):
    """W(nu nu) = 32 G_F^2 chi12^2, chi12 = E1E2 - p1 p2 x (massless): A0,A1,A2."""
    e1e2 = _E[0] * _E[1]
    p1p2 = _P[0] * _P[1]
    return (32 * _GF2 * e1e2 ** 2, -64 * _GF2 * e1e2 * p1p2, 32 * _GF2 * p1p2 ** 2)


def test_dee_coefficients_match_frame_jacobian():
    """a x^2 + b x + c reproduces the tested body-frame Jacobian D(x,y)."""
    for y in (-0.4, 0.0, 0.3, 0.55):
        a, b, c = dee_coefficients(y, _E, _P, _M4)
        for x in (-0.3, 0.1, 0.45):
            fr = collision_body_frame(E1=_E[0], p1=_P[0], E2=_E[1], p2=_P[1],
                                      E3=_E[2], p3=_P[2], m4=_M4, x=x, y=y)
            if not fr.valid:
                continue
            assert a * x * x + b * x + c == pytest.approx(fr.D, abs=1e-12)
            assert a < 0.0


def test_nunu_xpoly_matches_chi_invariants():
    from rabbit.collisions.pstf_matrix_element import chi_invariants, scrW_nu_nu_diagonal
    for y in (-0.2, 0.3):
        A0, A1, A2 = _nunu_xpoly(y)
        for x in (-0.25, 0.15):
            fr = collision_body_frame(E1=_E[0], p1=_P[0], E2=_E[1], p2=_P[1],
                                      E3=_E[2], p3=_P[2], m4=_M4, x=x, y=y)
            if not fr.valid:
                continue
            chi = chi_invariants(fr, 0, _E, _P)
            assert A0 + A1 * x + A2 * x * x == pytest.approx(scrW_nu_nu_diagonal(chi), rel=1e-12)


def _F0_quad_reference(xpoly, energies, momenta, m4, n_y):
    """F_0 with the inner x-integral by scipy.quad on the NON-SINGULAR theta substitution
    x = x_c + r cos(theta) (dx/sqrt(D) = dtheta/sqrt(-a)): an independent x-integration
    (numerical theta-Gauss vs the analytic I_n), accurate to machine precision. A naive
    quad over the 1/sqrt(D) endpoint singularity is only ~2% accurate -- not a usable gate.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_y)
    E1, E2, E3 = energies
    if E1 + E2 - E3 <= m4:
        return 0.0
    total = 0.0
    for yk, wk in zip(nodes, weights):
        a, b, c = dee_coefficients(yk, energies, momenta, m4)
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
        A = xpoly(yk)
        val, _ = quad(lambda th: sum(A[n] * (xc + r * math.cos(th)) ** n
                                     for n in range(len(A))) * inv, theta_hi, theta_lo)
        total += wk * val
    return total


def test_reduced_kernel_F0_analytic_matches_quadrature():
    F0_analytic = reduced_kernel_F0(_E, _P, _M4, _nunu_xpoly, n_y=48)
    F0_quad = _F0_quad_reference(_nunu_xpoly, _E, _P, _M4, n_y=48)
    assert F0_analytic > 0.0 and F0_quad != 0.0
    # ratio vs 1.0 so the tolerance BITES (these are ~1e-20 -- a rel= on the raw value is
    # swamped by pytest.approx's 1e-12 abs default). Analytic I_n vs theta-quad: ~machine.
    assert F0_analytic / F0_quad == pytest.approx(1.0, rel=1e-10)


def test_reduced_kernel_F0_y_convergence():
    """Gauss-Legendre y-integral converges as n_y grows."""
    coarse = reduced_kernel_F0(_E, _P, _M4, _nunu_xpoly, n_y=24)
    fine = reduced_kernel_F0(_E, _P, _M4, _nunu_xpoly, n_y=96)
    finer = reduced_kernel_F0(_E, _P, _M4, _nunu_xpoly, n_y=192)
    assert abs(fine - finer) < abs(coarse - finer)
    assert finer > 0.0


def test_reduced_kernel_F0_below_threshold_zero():
    # E4 = E1+E2-E3 < m4 -> kinematically forbidden -> F_0 = 0
    assert reduced_kernel_F0((0.5, 0.4, 0.8), (0.5, 0.4, 0.8), 2.0, _nunu_xpoly) == 0.0
