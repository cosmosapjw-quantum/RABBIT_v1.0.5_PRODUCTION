"""tests/test_pstf_lrs_moment_kernel.py — Stage 0 LRS axisymmetric collision contraction.

For LRS (axisymmetric) Bianchi I only m=0 survives, and the exact all-orders PSTF
collision hierarchy reduces to a REAL contraction over m=0 Clebsch-Gordans
(neutrino_large_anisotropy_exact_PSTF_collision_ko.md, "LRS/축대칭 특수화", eq 1088-1137):

    C_L = (1/8) [ sum_i eps_i F^(1;i)_L Gamma^(i)_L
                  + sum_{i<j<k} eps_i eps_j eps_k
                       sum_{li,lj,lk,J} F^(3;ijk)_{L;li lj lk;J}
                         <li 0, lj 0|J 0> <J 0, lk 0|L 0>
                         Gamma^(i)_{li} Gamma^(j)_{lj} Gamma^(k)_{lk} ],

with Gamma^(i)_l = g^(i)_{l0}/sqrt(4pi), Gamma_0 = G_0. This pins the moment-contraction
engine (the time-evolution step: energy convolution + PSTF star contraction) against:
  - the exact particle-hole foundation: isotropic (all-monopole) -> 8 Lambda_r F0;
  - the general complex star product (the m=0 sector must agree);
  - the weak-anisotropy reduction F^(3)_{L;L00} = F^(1)_L -> Legendre kernel;
  - the L + sum_a l_a even selection rule.
The reduced kernels F^(n) are inputs here (precomputed from W_r . Pi); this module is
the contraction only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rabbit.collisions.pstf_trilinear_kernel import star_product, lambda_r_particle_hole, SQRT_4PI
from rabbit.collisions.pstf_lrs_moment_kernel import (
    couple_m0,
    lrs_trilinear_amplitude,
    axisymmetric_collision_moment,
    LEG_SIGNS,
)


# ── real axisymmetric (m=0) star product ──────────────────────────────────

def test_couple_m0_all_monopole():
    out = couple_m0(np.array([0.7]), np.array([1.3]))
    assert out[0] == pytest.approx(0.7 * 1.3, abs=1e-13)
    assert len(out) == 1


def test_couple_m0_selection_rule():
    # <l_a 0, l_b 0 | J 0> = 0 unless l_a + l_b + J even
    a = np.array([0.0, 1.0])        # rank 1
    b = np.array([0.0, 0.0, 1.0])   # rank 2
    out = couple_m0(a, b)           # ranks 1 (x) 2 -> J in {1,2,3}; J=2 forbidden (1+2+2 odd)
    assert out[2] == pytest.approx(0.0, abs=1e-13)
    assert out[1] != 0.0 and out[3] != 0.0


def test_couple_m0_matches_general_star_product():
    """m=0 sector of the complex star product: couple_m0 output == z_{J,0}/sqrt(4pi).

    Single-rank-(la,lb) axisymmetric inputs isolate one star-product pair: build
    spherical components x_{l,m!=0}=0, x_{l,0}=sqrt(4pi) Gamma_l, run the general
    complex star_product, and compare its M=0 component (an independent code path).
    """
    rng = np.random.default_rng(3)
    for (la, lb) in [(1, 2), (2, 2), (1, 3), (2, 3)]:
        ga = np.zeros(la + 1); ga[la] = rng.standard_normal()
        gb = np.zeros(lb + 1); gb[lb] = rng.standard_normal()
        out = couple_m0(ga, gb)
        x = np.zeros(2 * la + 1, dtype=complex); x[la] = SQRT_4PI * ga[la]
        y = np.zeros(2 * lb + 1, dtype=complex); y[lb] = SQRT_4PI * gb[lb]
        for J in range(abs(la - lb), la + lb + 1):
            z = star_product(x, la, y, lb, J)
            expect = z[J] / SQRT_4PI  # M=0 component (z indexed M=-J..J, so index J)
            assert out[J] == pytest.approx(expect.real, abs=1e-12)
            assert expect.imag == pytest.approx(0.0, abs=1e-12)


# ── trilinear output amplitude Gamma^(ijk)_L ──────────────────────────────

def test_lrs_trilinear_all_monopole():
    gi, gj, gk = np.array([0.5]), np.array([-0.3]), np.array([0.8])
    out = lrs_trilinear_amplitude(gi, gj, gk)
    assert out[0] == pytest.approx(0.5 * -0.3 * 0.8, abs=1e-13)


def test_lrs_trilinear_first_pair_exchange_symmetry():
    """Bare amplitude is symmetric in the first two (coupled) legs i<->j only.

    The fixed coupling tree ((i (x) j)_J (x) k)_L makes Gamma^(ijk)_L symmetric under
    i<->j (m=0 CG symmetry forced by the li+lj+J even rule), but NOT under swapping in
    the third leg k -- that recoupling involves a 6j and is realized only by the full
    kernel-weighted contraction, not the bare amplitude.
    """
    rng = np.random.default_rng(4)
    gi = rng.standard_normal(3)
    gj = rng.standard_normal(2)
    gk = rng.standard_normal(3)
    base = lrs_trilinear_amplitude(gi, gj, gk)
    swapped = lrs_trilinear_amplitude(gj, gi, gk)  # i<->j: symmetric
    assert np.allclose(base, swapped, atol=1e-12)
    # i<->k is genuinely NOT a symmetry of the bare tree (sanity: differs)
    other = lrs_trilinear_amplitude(gk, gj, gi)
    assert not np.allclose(base, other, atol=1e-9)


def test_lrs_trilinear_selection_rule():
    # one rank-1 leg, two monopoles -> only odd L survives the parity rule; L=2 -> 0
    gi, gj, gk = np.array([0.0, 1.0]), np.array([1.0]), np.array([1.0])
    out = lrs_trilinear_amplitude(gi, gj, gk)
    assert out[1] == pytest.approx(1.0, abs=1e-12)   # P_1 channel
    if len(out) > 2:
        assert out[2] == pytest.approx(0.0, abs=1e-12)


# ── full bracket: isotropic + weak-anisotropy GATEs ───────────────────────

def _monopole_gammas(g0):
    return {i: np.array([g0[i]]) for i in range(4)}


def test_axisymmetric_isotropic_hm_gate():
    """All-monopole inputs reproduce 8 Lambda_r F0 (the exact isotropic HM limit)."""
    F0 = 0.37
    occ = [0.2, 0.6, 0.45, 0.8]
    g = [1.0 - 2.0 * f for f in occ]
    reduced_linear = {i: {0: F0} for i in range(4)}
    reduced_trilinear = {trip: {(0, 0, 0, 0, 0): F0}
                         for trip in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))}
    C0 = axisymmetric_collision_moment(0, reduced_linear, reduced_trilinear, _monopole_gammas(g))
    expect = 8.0 * lambda_r_particle_hole(*occ) * F0
    assert C0 == pytest.approx(expect, abs=1e-13)


def test_axisymmetric_weak_anisotropy_reduces_to_linear():
    """F^(3)_{L;L00}=F^(1)_L: the two-monopole-leg trilinear reproduces the linear term.

    Take a single dipole perturbation on leg 1 (Gamma^(1)_1 = d) over the monopole
    background; the L=1 moment's trilinear 234-type contributions with two rank-0 legs
    must add the same Legendre-kernel structure as the linear F^(1)_1 term.
    """
    F1 = 0.5
    g0 = [0.2, -0.1, 0.3, -0.4]
    d = 0.05
    gammas = {0: np.array([g0[0], d]), 1: np.array([g0[1]]),
              2: np.array([g0[2]]), 3: np.array([g0[3]])}
    # linear leg-0 kernel at L=1
    reduced_linear = {0: {1: F1}}
    # trilinear triples containing leg 0 with leg-0 rank 1, other two rank 0, J=1, L=1;
    # by F^(3)_{1;100}=F^(1)_1 the kernel value equals F1
    reduced_trilinear = {}
    for trip in ((0, 1, 2), (0, 1, 3), (0, 2, 3)):
        reduced_trilinear[trip] = {(1, 1, 0, 0, 1): F1}
    C1 = axisymmetric_collision_moment(1, reduced_linear, reduced_trilinear, gammas)
    # linear contribution
    lin = LEG_SIGNS[0] * F1 * d
    # trilinear: each triple (0,j,k) contributes eps0 epsj epsk * F1 * <1 0,0 0|1 0><1 0,0 0|1 0> * d*g0[j]*g0[k]
    from rabbit.collisions.wigner import clebsch_gordan
    cg = clebsch_gordan(1, 0, 0, 0, 1, 0) * clebsch_gordan(1, 0, 0, 0, 1, 0)
    tri = 0.0
    for (i, j, k) in ((0, 1, 2), (0, 1, 3), (0, 2, 3)):
        tri += LEG_SIGNS[i] * LEG_SIGNS[j] * LEG_SIGNS[k] * F1 * cg * d * g0[j] * g0[k]
    assert C1 == pytest.approx(lin + tri, abs=1e-13)
