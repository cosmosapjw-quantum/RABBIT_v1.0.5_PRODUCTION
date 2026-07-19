"""rabbit.collisions.pstf_matrix_element — HM matrix element on the body frame.

Stage 0 of the nonperturbative Bianchi-I substrate (not production-wired). Expresses the
Hannestad-Madsen microscopic matrix element W_r = S_r sum_spins |M_r|^2 in the canonical
collision body frame (``neutrino_decoupling_PSTF_HM_generalization_ko.md``, eq 405-474),
the W_r(x,y) that the exact reduced kernel integrates:

    F_0^(r)(p1,p2,p3) = int dx dy Theta(D) Theta(E4-m4)/sqrt(D) W_r(x,y).

HM works in metric (+,-,-,-); in this code's (-,+,+,+) the sign-safe invariants are

    chi_ij = -p_i . p_j = E_i E_j - p_i p_j (n_i . n_j)   (>= 0 for massless),

with the body-frame directions n_i(x,y,s) from :mod:`pstf_body_frame`. For massless
on-shell 2<->2 (p1+p2=p3+p4) the Mandelstam pairings are exact: chi12=chi34 (s),
chi13=chi24 (t), chi14=chi23 (u). The neutrino-neutrino reactions are products of two
chi (the same-flavour nu-nubar channel carries 128 G_F^2, the distinct channels 32);
the electron reactions (mass + chiral couplings g_L,g_R) are a downstream addition.
"""

from __future__ import annotations

import numpy as np

from rabbit.collisions.kernels import G_F_MEV

#: Fermi constant squared in MeV^-4 (HM W_r ~ G_F^2 E^4, dimensionless after /E^4).
_GF2: float = G_F_MEV ** 2


def chi_invariants(frame, s: int, energies, momenta) -> dict:
    """Sign-safe HM Lorentz invariants chi_ij = E_i E_j - p_i p_j (n_i . n_j) at root s.

    Parameters
    ----------
    frame : :class:`pstf_body_frame.BodyFrame` (must be ``valid``).
    s : azimuth root index (0 -> beta_+, 1 -> beta_-).
    energies, momenta : length-3 sequences (legs 1,2,3). Leg 4 is taken from the frame
        (``frame.E4`` / ``frame.p4``) so the on-shell closure is the single source of
        truth -- no caller can pass an inconsistent leg-4 energy.

    Returns a dict ``{(i,j): chi_ij}`` for 1 <= i < j <= 4 (1-based leg labels).
    """
    if not frame.valid:
        raise ValueError("chi_invariants requires a valid body frame")
    if len(energies) != 3 or len(momenta) != 3:
        raise ValueError("energies/momenta give legs 1,2,3 (length 3); leg 4 is the frame")
    dirs = [frame.direction(leg, s) for leg in (1, 2, 3, 4)]
    E = [float(e) for e in energies] + [frame.E4]
    p = [float(pp) for pp in momenta] + [frame.p4]
    chi = {}
    for a in range(4):
        for b in range(a + 1, 4):
            mu = float(np.dot(dirs[a], dirs[b]))
            chi[(a + 1, b + 1)] = E[a] * E[b] - p[a] * p[b] * mu
    return chi


def scrW_nu_nu_diagonal(chi: dict) -> float:
    """W(nu_a nu_b -> nu_a nu_b), a != b: 32 G_F^2 chi12 chi34 (eq 456)."""
    return 32.0 * _GF2 * chi[(1, 2)] * chi[(3, 4)]


def scrW_nu_nubar_diagonal(chi: dict) -> float:
    """W(nu_a nubar_b -> nu_a nubar_b), a != b: 32 G_F^2 chi14 chi23 (eq 462)."""
    return 32.0 * _GF2 * chi[(1, 4)] * chi[(2, 3)]


def scrW_nu_nubar_same_flavor(chi: dict) -> float:
    """W(nu_a nubar_a -> nu_a nubar_a): 128 G_F^2 chi14 chi23 (eq 468)."""
    return 128.0 * _GF2 * chi[(1, 4)] * chi[(2, 3)]


def scrW_nu_nubar_to_other_flavor(chi: dict) -> float:
    """W(nu_a nubar_a -> nu_b nubar_b), b != a: 32 G_F^2 chi14 chi23 (eq 473)."""
    return 32.0 * _GF2 * chi[(1, 4)] * chi[(2, 3)]
