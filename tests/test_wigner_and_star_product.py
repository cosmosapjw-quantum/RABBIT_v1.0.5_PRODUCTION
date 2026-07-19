"""tests/test_wigner_and_star_product.py — Stage 0 PSTF CG algebra.

Dependency-free Clebsch-Gordan / Wigner-3j / Wigner-6j (Racah formulas) and the
normalized PSTF star product z_{JM} = (1/sqrt(4pi)) sum CG x y with the monopole
convention x_00 = sqrt(4pi) X_0 (so a rank-0 scalar is the TRUE identity). Verified
against closed-form values, orthogonality, the scalar-identity property, and the
6j reordering identity (spec: neutrino_large_anisotropy_exact_PSTF_collision_ko.md).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rabbit.collisions.wigner import clebsch_gordan, wigner_3j, wigner_6j
from rabbit.collisions.pstf_trilinear_kernel import star_product, SQRT_4PI


# ── Wigner / CG closed-form anchors ───────────────────────────────────────

def test_cg_known_values():
    assert clebsch_gordan(1, 0, 1, 0, 2, 0) == pytest.approx(math.sqrt(2.0 / 3.0), abs=1e-13)
    # spin-1 singlet (Condon-Shortley): <1 1,1 -1|0 0> = +1/sqrt(3)
    assert clebsch_gordan(1, 1, 1, -1, 0, 0) == pytest.approx(1.0 / math.sqrt(3.0), abs=1e-13)
    assert clebsch_gordan(1, 1, 1, -1, 2, 0) == pytest.approx(1.0 / math.sqrt(6.0), abs=1e-13)
    # stretched state is unity
    assert clebsch_gordan(2, 2, 2, 2, 4, 4) == pytest.approx(1.0, abs=1e-13)


def test_cg_selection_rules_and_orthogonality():
    # m3 != m1+m2 vanishes
    assert clebsch_gordan(1, 1, 1, 0, 2, 0) == 0.0
    # |j1-j2| <= j3 <= j1+j2
    assert clebsch_gordan(1, 0, 1, 0, 3, 0) == 0.0
    # orthogonality over m1,m2 at fixed j3,m3
    s = sum(clebsch_gordan(2, m1, 2, -m1, 0, 0) ** 2 for m1 in range(-2, 3))
    assert s == pytest.approx(1.0, abs=1e-13)


def test_3j_known_value_standalone():
    # direct closed form (independent of the 3j<->CG conversion):
    # (1 1 2; 0 0 0) = +sqrt(2/15)
    assert wigner_3j(1, 1, 2, 0, 0, 0) == pytest.approx(math.sqrt(2.0 / 15.0), abs=1e-13)
    # (1 1 0; 0 0 0) = -1/sqrt(3)
    assert wigner_3j(1, 1, 0, 0, 0, 0) == pytest.approx(-1.0 / math.sqrt(3.0), abs=1e-13)


def test_3j_symmetry_and_6j_value():
    # 3j relation to CG
    j1, j2, j3, m1, m2 = 2, 1, 2, 1, -1
    m3 = -(m1 + m2)
    lhs = wigner_3j(j1, j2, j3, m1, m2, m3)
    rhs = (-1) ** (j1 - j2 - m3) / math.sqrt(2 * j3 + 1) * clebsch_gordan(j1, m1, j2, m2, j3, -m3)
    assert lhs == pytest.approx(rhs, abs=1e-13)
    # closed-form 6j anchor: {1 1 1; 1 1 1} = 1/6
    assert wigner_6j(1, 1, 1, 1, 1, 1) == pytest.approx(1.0 / 6.0, abs=1e-13)


# ── PSTF star product ─────────────────────────────────────────────────────

def _sph(rng, ell):
    """Random complex spherical components x_{ell m}, m=-ell..ell."""
    return rng.standard_normal(2 * ell + 1) + 1j * rng.standard_normal(2 * ell + 1)


def test_scalar_is_star_identity():
    """[X^(0) star_ell Y^(ell)] = X_0 * Y (monopole convention x_00 = sqrt(4pi) X_0)."""
    rng = np.random.default_rng(0)
    ell = 2
    X0 = 0.7
    x0 = np.array([SQRT_4PI * X0], dtype=complex)  # rank-0 spherical comp
    y = _sph(rng, ell)
    z = star_product(x0, 0, y, ell, ell)
    assert np.allclose(z, X0 * y, atol=1e-13)


def test_star_6j_reordering_identity():
    """[[X*_J Y]*_L Z] == sum_J' (-1)^... sqrt((2J+1)(2J'+1)) {6j} [X*_L[Y*_J' Z]]."""
    rng = np.random.default_rng(1)
    l1, l2, l3 = 1, 2, 1
    J, L = 2, 1
    x, y, zc = _sph(rng, l1), _sph(rng, l2), _sph(rng, l3)
    left = star_product(star_product(x, l1, y, l2, J), J, zc, l3, L)
    right = np.zeros(2 * L + 1, dtype=complex)
    for Jp in range(abs(l2 - l3), l2 + l3 + 1):
        sixj = wigner_6j(l1, l2, J, l3, L, Jp)
        if sixj == 0.0:
            continue
        phase = (-1) ** (l1 + l2 + l3 + L)
        coeff = phase * math.sqrt((2 * J + 1) * (2 * Jp + 1)) * sixj
        right = right + coeff * star_product(x, l1, star_product(y, l2, zc, l3, Jp), Jp, L)
    assert np.allclose(left, right, atol=1e-12)
