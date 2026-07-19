"""tests/test_pstf_matrix_element.py — Stage 0 HM matrix element on the body frame.

The exact PSTF reduced kernel needs the HM microscopic matrix element W_r expressed in
the canonical body frame (neutrino_decoupling_PSTF_HM_generalization_ko.md, eq 405-474):
sign-safe Lorentz invariants

    chi_ij = -p_i . p_j = E_i E_j - p_i p_j (n_i . n_j)   [metric (-,+,+,+)],

with the body-frame directions n_i(x,y,s). For massless on-shell 2<->2 (p1+p2=p3+p4),
4-momentum conservation forces the Mandelstam pairings exactly:

    chi12 = chi34 (s),   chi13 = chi24 (t),   chi14 = chi23 (u),

a rigorous independent check of the n3/n4 construction. The nu-nu reactions are
W = 32 G_F^2 chi_a chi_b (128 for the same-flavour channel). This pins the body-frame
matrix element used to build F_0 = int Theta(D)/sqrt(D) W_r dx dy.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.collisions.kernels import G_F_MEV
from rabbit.collisions.pstf_body_frame import collision_body_frame
from rabbit.collisions.pstf_matrix_element import (
    chi_invariants,
    scrW_nu_nu_diagonal,
    scrW_nu_nubar_diagonal,
    scrW_nu_nubar_same_flavor,
)


def _massless_frame():
    # asymmetric massless config (p4 = p1+p2-p3 = 1.3), validity checked in tests.
    # energies/momenta are legs 1,2,3 only; leg 4 is taken from the frame.
    fr = collision_body_frame(E1=1.2, p1=1.2, E2=0.9, p2=0.9, E3=0.8, p3=0.8,
                              m4=0.0, x=0.1, y=0.2)
    E = (1.2, 0.9, 0.8)
    mom = (1.2, 0.9, 0.8)
    return fr, E, mom


def test_frame_is_valid_and_massless_energies():
    fr, E, mom = _massless_frame()
    assert fr.valid
    assert fr.p4 == pytest.approx(1.3, abs=1e-13)
    assert fr.E4 == fr.p4  # massless: E_4 = p_4
    assert E == mom  # massless: E_i = p_i


def test_mandelstam_pairings_both_roots():
    """chi12=chi34, chi13=chi24, chi14=chi23 (massless 4-momentum conservation)."""
    fr, E, mom = _massless_frame()
    for s in (0, 1):
        chi = chi_invariants(fr, s, E, mom)
        assert chi[(1, 2)] == pytest.approx(chi[(3, 4)], abs=1e-12)  # s
        assert chi[(1, 3)] == pytest.approx(chi[(2, 4)], abs=1e-12)  # t
        assert chi[(1, 4)] == pytest.approx(chi[(2, 3)], abs=1e-12)  # u


def test_chi_nonnegative_and_mu14_closed_form():
    fr, E, mom = _massless_frame()
    p1, p2, p3 = mom
    p4 = fr.p4
    for s in (0, 1):
        chi = chi_invariants(fr, s, E, mom)
        for v in chi.values():
            assert v >= -1e-13  # chi = E_iE_j(1 - mu_ij) >= 0 for massless
        # mu14 = (p1 + p2 x - p3 y)/p4 -> chi14 = E1E4 - p1 p4 mu14
        mu14 = (p1 + p2 * 0.1 - p3 * 0.2) / p4
        assert chi[(1, 4)] == pytest.approx(E[0] * fr.E4 - p1 * p4 * mu14, abs=1e-12)


def test_chi12_value():
    # chi12 = E1 E2 - p1 p2 x = 1.2*0.9 - 1.2*0.9*0.1
    fr, E, mom = _massless_frame()
    chi = chi_invariants(fr, 0, E, mom)
    assert chi[(1, 2)] == pytest.approx(1.2 * 0.9 * (1 - 0.1), abs=1e-13)


def test_scrW_nu_nu_values_and_ratios():
    fr, E, mom = _massless_frame()
    chi = chi_invariants(fr, 0, E, mom)
    gf2 = G_F_MEV ** 2
    assert scrW_nu_nu_diagonal(chi) == pytest.approx(32 * gf2 * chi[(1, 2)] * chi[(3, 4)], abs=0, rel=1e-13)
    assert scrW_nu_nubar_diagonal(chi) == pytest.approx(32 * gf2 * chi[(1, 4)] * chi[(2, 3)], abs=0, rel=1e-13)
    # same-flavour nu nubar is 4x the distinct-flavour channel (128 vs 32 G_F^2)
    assert scrW_nu_nubar_same_flavor(chi) == pytest.approx(4 * scrW_nu_nubar_diagonal(chi), rel=1e-13)
    # all positive
    assert scrW_nu_nu_diagonal(chi) > 0
    assert scrW_nu_nubar_same_flavor(chi) > 0


def test_chi_invariants_rejects_invalid_frame_and_bad_length():
    # invalid frame (|B|>1) must fail loud, not emit garbage from zero vectors
    bad = collision_body_frame(E1=1.0, p1=1.0, E2=0.8, p2=0.8, E3=0.9, p3=0.9,
                               m4=0.0, x=0.1, y=0.2)
    assert not bad.valid
    with pytest.raises(ValueError):
        chi_invariants(bad, 0, (1.0, 0.8, 0.9), (1.0, 0.8, 0.9))
    fr, E, mom = _massless_frame()
    with pytest.raises(ValueError):
        chi_invariants(fr, 0, (1.2, 0.9, 0.8, fr.p4), (1.2, 0.9, 0.8, fr.p4))


def test_scrW_uses_mandelstam_equivalent_pairings():
    """Since chi12=chi34 and chi14=chi23, W can be written either way (massless)."""
    fr, E, mom = _massless_frame()
    chi = chi_invariants(fr, 0, E, mom)
    gf2 = G_F_MEV ** 2
    assert scrW_nu_nu_diagonal(chi) == pytest.approx(32 * gf2 * chi[(1, 2)] ** 2, rel=1e-12)
    assert scrW_nu_nubar_diagonal(chi) == pytest.approx(32 * gf2 * chi[(1, 4)] ** 2, rel=1e-12)
