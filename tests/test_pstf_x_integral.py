"""tests/test_pstf_x_integral.py — Stage 0 closed-form x-integral of the reduced kernel.

After the body-frame polynomial reduction, the reduced kernel's inner x-integral is
(neutrino_large_anisotropy_exact_PSTF_collision_ko.md, "x 적분의 폐형식"):

    W_r(x,y) Pi(x,y) = sum_n A_n(y) x^n,
    F = int dy sum_n A_n(y) I_n(y),
    I_n(y) = int_{x_lo}^{x_hi} x^n dx / sqrt(a x^2 + b x + c),

with D = a x^2 + b x + c (a < 0), [x_lo,x_hi] = [-1,1] cap {D >= 0}. The closed form
uses x = x_c + r cos(theta), x_c = -b/2a, r = sqrt(b^2-4ac)/2|a|:

    I_n = (1/sqrt(-a)) sum_k C(n,k) x_c^{n-k} r^k J_k(theta_hi, theta_lo),
    J_k(alpha,beta) = int_alpha^beta cos^k theta d theta.

This keeps the large-anisotropy kernel at TWO energy integrals (like isotropic HM):
the x-integral is analytic, only y is integrated numerically at precompute time. The
closed form is pinned here against direct numerical quadrature.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from rabbit.collisions.pstf_x_integral import (
    cos_power_integral,
    reduced_x_integral,
)


# ── J_k(alpha,beta) = int_alpha^beta cos^k theta d theta ───────────────────

def test_cos_power_integral_matches_quadrature():
    for (alpha, beta) in [(0.0, math.pi), (0.3, 2.1), (1.0, 1.0), (2.0, 0.4)]:
        for k in range(0, 7):
            ref, _ = quad(lambda t: math.cos(t) ** k, alpha, beta)
            assert cos_power_integral(k, alpha, beta) == pytest.approx(ref, abs=1e-11)


# ── I_n closed form vs numerical quadrature ────────────────────────────────

# (a, b, c): a<0 (downward parabola), Delta>0, varied root placement
_QUADRATICS = [
    (-1.0, 0.0, 0.5),   # roots +-0.707 strictly inside [-1,1]
    (-2.0, 1.0, 0.8),   # roots [-0.430, 0.930] inside
    (-1.0, 0.0, 2.0),   # roots +-1.414 -> clipped to [-1,1] on both ends
    (-1.0, 0.6, 1.2),   # asymmetric, left root clipped
    (-3.0, -0.4, 0.9),  # asymmetric, negative b
]


def _domain(a, b, c):
    disc = b * b - 4 * a * c
    assert disc > 0 and a < 0
    r = math.sqrt(disc) / (2 * abs(a))
    xc = -b / (2 * a)
    lo = max(-1.0, xc - r)
    hi = min(1.0, xc + r)
    return lo, hi


def test_reduced_x_integral_matches_quadrature():
    for (a, b, c) in _QUADRATICS:
        lo, hi = _domain(a, b, c)
        for n in range(0, 5):
            ref, _ = quad(lambda x: x ** n / math.sqrt(a * x * x + b * x + c), lo, hi)
            got = reduced_x_integral(n, a, b, c)
            assert got == pytest.approx(ref, abs=1e-9, rel=1e-9), f"n={n} a,b,c={a,b,c}"


def test_reduced_x_integral_n0_is_arcsin_span():
    # int dx/sqrt(c - x^2) over the full root span = pi  (a=-1,b=0 -> roots +-sqrt(c))
    assert reduced_x_integral(0, -1.0, 0.0, 0.5) == pytest.approx(math.pi, abs=1e-12)


def test_reduced_x_integral_empty_domain_is_zero():
    # Delta < 0 (D<0 for all x): no physical interval
    assert reduced_x_integral(0, -1.0, 0.0, -0.5) == 0.0
    assert reduced_x_integral(2, -1.0, 0.0, -0.5) == 0.0
    # roots entirely outside [-1,1] (both > 1): empty intersection
    assert reduced_x_integral(0, -1.0, 6.0, -8.5) == 0.0


def test_reduced_x_integral_odd_symmetry():
    # symmetric domain (b=0) -> odd n vanish, even n match quad
    assert reduced_x_integral(1, -1.0, 0.0, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert reduced_x_integral(3, -1.0, 0.0, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_reduced_x_integral_rejects_nonnegative_a():
    with pytest.raises(ValueError):
        reduced_x_integral(0, 1.0, 0.0, 0.5)
