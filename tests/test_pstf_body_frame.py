"""tests/test_pstf_body_frame.py — Stage 0 generalized HM angular polynomial.

The exact all-orders PSTF collision multipole factorizes the SO(3) global rotation
out of the HM body frame (neutrino_large_anisotropy_exact_PSTF_collision_ko.md,
"HM body frame" + "Haar 평균과 Wigner-Eckart 축약"): the collision-square shape is
fixed by (x, y, s) with x = n1·n2, y = n1·n3, and the two on-shell azimuth roots
s = ±. The Haar integral collapses to the generalized angular polynomial

    Pi^(3;ijk)_{L;li lj lk;J}(x,y) = (8 pi^2 / sqrt(4 pi (2L+1)))
        sum_{s=±} sum_{mi+mj+mk=0}
            <li mi, lj mj | J, mi+mj> <J, mi+mj; lk mk | L 0>
            prod_a Y_{la ma}(n_a^(s)),

normalized so all-monopole inputs give Pi = 1, and reducing for one multipole leg
to the Legendre-inserted kernel Pi^(1;i)_L = P_L(mu_{1i}). This pins the angular core
(spherical harmonics, body-frame kinematics, CG double-coupling) against those exact
limits — independent of the radial reduced-kernel integral.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import eval_legendre

from rabbit.collisions.pstf_body_frame import (
    spherical_harmonic,
    collision_body_frame,
    body_scalar_T,
    angular_polynomial_n1,
    angular_polynomial_n3,
)

SQRT_4PI = math.sqrt(4.0 * math.pi)


# ── spherical harmonics (Condon-Shortley, complex) ────────────────────────

def _vec(theta, phi):
    return np.array([math.sin(theta) * math.cos(phi),
                     math.sin(theta) * math.sin(phi),
                     math.cos(theta)])


def test_spherical_harmonic_closed_forms():
    theta, phi = 0.7, 1.3
    n = _vec(theta, phi)
    ct, st = math.cos(theta), math.sin(theta)
    assert spherical_harmonic(0, 0, n) == pytest.approx(1.0 / SQRT_4PI, abs=1e-13)
    assert spherical_harmonic(1, 0, n) == pytest.approx(math.sqrt(3 / (4 * math.pi)) * ct, abs=1e-13)
    y11 = -math.sqrt(3 / (8 * math.pi)) * st * np.exp(1j * phi)
    assert spherical_harmonic(1, 1, n) == pytest.approx(y11, abs=1e-13)
    y20 = math.sqrt(5 / (16 * math.pi)) * (3 * ct * ct - 1)
    assert spherical_harmonic(2, 0, n) == pytest.approx(y20, abs=1e-13)
    y22 = math.sqrt(15 / (32 * math.pi)) * st * st * np.exp(2j * phi)
    assert spherical_harmonic(2, 2, n) == pytest.approx(y22, abs=1e-13)


def test_spherical_harmonic_negative_m_symmetry():
    n = _vec(1.1, -0.6)
    for ell in (1, 2, 3):
        for m in range(1, ell + 1):
            lhs = spherical_harmonic(ell, -m, n)
            rhs = (-1) ** m * np.conj(spherical_harmonic(ell, m, n))
            assert lhs == pytest.approx(rhs, abs=1e-13)


def test_spherical_harmonic_addition_theorem():
    # sum_m Y*_{Lm}(a) Y_{Lm}(b) = (2L+1)/(4pi) P_L(a·b)
    a, b = _vec(0.5, 0.3), _vec(2.0, -1.1)
    mu = float(np.dot(a, b))
    for L in (1, 2, 3, 4):
        s = sum(np.conj(spherical_harmonic(L, m, a)) * spherical_harmonic(L, m, b)
                for m in range(-L, L + 1))
        assert s.imag == pytest.approx(0.0, abs=1e-12)
        assert s.real == pytest.approx((2 * L + 1) / (4 * math.pi) * eval_legendre(L, mu), abs=1e-12)


# ── HM body frame kinematics ──────────────────────────────────────────────

def _frame():
    # massless symmetric config, validity verified: B = sqrt(0.5) < 1
    return collision_body_frame(E1=1.0, p1=1.0, E2=1.0, p2=1.0, E3=1.0, p3=1.0,
                                m4=0.0, x=0.2, y=0.5)


def test_body_frame_valid_and_geometry():
    fr = _frame()
    assert fr.valid
    # n1·n2 = x, n1·n3 = y (both roots)
    assert float(np.dot(fr.n1, fr.n2)) == pytest.approx(0.2, abs=1e-13)
    for s in (0, 1):
        assert float(np.dot(fr.n1, fr.n3[s])) == pytest.approx(0.5, abs=1e-13)
        for v in (fr.n1, fr.n2, fr.n3[s], fr.n4[s]):
            assert float(np.dot(v, v)) == pytest.approx(1.0, abs=1e-12)


def test_body_frame_momentum_energy_conservation():
    fr = _frame()
    p1, p2, p3, p4 = 1.0, 1.0, 1.0, fr.p4
    assert fr.E4 == pytest.approx(1.0, abs=1e-13)  # E1+E2-E3
    for s in (0, 1):
        lhs = p1 * fr.n1 + p2 * fr.n2
        rhs = p3 * fr.n3[s] + p4 * fr.n4[s]
        assert np.allclose(lhs, rhs, atol=1e-12)


def test_body_frame_jacobian_and_invalid():
    fr = _frame()
    B = fr.B
    D_expect = 4 * 1.0 * 1.0 * (1 - 0.2 ** 2) * (1 - 0.5 ** 2) * (1 - B ** 2)
    assert fr.D == pytest.approx(D_expect, abs=1e-13)
    assert fr.D > 0
    # |B|>1 -> closed on-shell quadrilateral impossible -> invalid (B~1.005 here)
    bad = collision_body_frame(E1=1.0, p1=1.0, E2=0.8, p2=0.8, E3=0.9, p3=0.9,
                               m4=0.0, x=0.1, y=0.2)
    assert abs(bad.B) > 1.0
    assert not bad.valid


def test_body_frame_collinear_edge_fails_closed():
    """|x|=1 (n2 collinear with n1) -> denom 0 -> fail closed, no NaN-as-valid."""
    for (x, y) in ((1.0, 0.5), (-1.0, 0.5), (0.2, 1.0), (0.2, -1.0)):
        fr = collision_body_frame(E1=1.0, p1=1.0, E2=1.0, p2=1.0, E3=1.0, p3=1.0,
                                  m4=0.0, x=x, y=y)
        assert not fr.valid


# ── generalized HM angular polynomial ─────────────────────────────────────

def test_pi_all_monopole_normalization():
    """Pi^(n)_{0;0..0} = 1 (the defining normalization, any valid x,y)."""
    fr = _frame()
    assert angular_polynomial_n1(0, 2, fr) == pytest.approx(1.0, abs=1e-13)
    val = angular_polynomial_n3(L=0, J=0, ranks=(0, 0, 0), legs=(1, 2, 3), frame=fr)
    assert val == pytest.approx(1.0, abs=1e-12)


def test_pi_n1_legendre():
    """Pi^(1;i)_L = P_L(mu_{1i})."""
    fr = _frame()
    mus = {1: 1.0, 2: 0.2, 3: 0.5, 4: float(fr.n4[0][2])}
    for i, mu in mus.items():
        for L in (1, 2, 3):
            assert angular_polynomial_n1(L, i, fr) == pytest.approx(eval_legendre(L, mu), abs=1e-12)


def test_pi_n3_two_monopole_legs_reduce_to_legendre():
    """n=3 with two rank-0 legs collapses to Pi^(1;i)_L (weak-anisotropy GATE)."""
    fr = _frame()
    for i in (2, 3, 4):
        mu = 0.2 if i == 2 else (0.5 if i == 3 else float(fr.n4[0][2]))
        for L in (1, 2):
            # multipole on leg i (rank L), monopole legs filled by leg 1; J=L forced
            val = angular_polynomial_n3(L=L, J=L, ranks=(L, 0, 0), legs=(i, 1, 1), frame=fr)
            assert val == pytest.approx(eval_legendre(L, mu), abs=1e-12)


def test_pi_n3_is_real():
    """Reality: T(beta_-) = T(beta_+)* so sum_s T is real (Condon-Shortley)."""
    fr = _frame()
    for (L, J, ranks, legs) in [(2, 2, (1, 1, 2), (2, 3, 4)),
                                (0, 1, (1, 1, 0), (2, 3, 4)),
                                (2, 2, (2, 2, 2), (2, 3, 4))]:
        raw = 0.0 + 0.0j
        pref = 8 * math.pi ** 2
        for s in (0, 1):
            dirs = (fr.direction(legs[0], s), fr.direction(legs[1], s), fr.direction(legs[2], s))
            raw += pref * body_scalar_T(L, J, ranks, dirs)
        assert raw.imag == pytest.approx(0.0, abs=1e-12)
        assert raw.real == pytest.approx(angular_polynomial_n3(L, J, ranks, legs, fr), abs=1e-12)
