"""rabbit.collisions.pstf_lrs_moment_kernel — LRS axisymmetric collision contraction.

Stage 0 of the nonperturbative Bianchi-I substrate (not production-wired). The
time-evolution step of the exact all-orders PSTF neutrino collision: given the
precomputed energy-dependent reduced kernels F^(n) (from W_r . Pi, the body-shape
x,y integral) and the axisymmetric harmonic amplitudes Gamma^(i)_l = g^(i)_{l0}/sqrt(4pi)
(Gamma_0 = G_0 = 1 - 2 f_0), it performs the real m=0 PSTF star contraction
(``neutrino_large_anisotropy_exact_PSTF_collision_ko.md``, "LRS/축대칭 특수화", eq 1088-1137):

    C_L = (1/8) [ sum_i eps_i F^(1;i)_L Gamma^(i)_L
                  + sum_{i<j<k} eps_i eps_j eps_k
                       sum_{li,lj,lk,J} F^(3;ijk)_{L;li lj lk;J}
                         <li 0, lj 0|J 0> <J 0, lk 0|L 0>
                         Gamma^(i)_{li} Gamma^(j)_{lj} Gamma^(k)_{lk} ].

In the isotropic (all-monopole) limit the bracket collapses to 8 Lambda_r F0 (the exact
particle-hole foundation), and with the reduction F^(3)_{L;L00}=F^(1)_L the two-monopole-leg
trilinear reproduces the Legendre-inserted linear kernel. m=0 CG parity enforces the
L + sum_a l_a even selection rule. This module is the contraction ONLY; the reduced
kernels are precomputed upstream.
"""

from __future__ import annotations

import numpy as np

from rabbit.collisions.wigner import clebsch_gordan

#: Leg signs eps_i: incoming legs (1,2) -> +1, outgoing (3,4) -> -1 (0-based here).
LEG_SIGNS: tuple[int, int, int, int] = (+1, +1, -1, -1)

#: The four particle-hole triples (0-based leg labels); coefficient eps_i eps_j eps_k.
TRIPLES: tuple[tuple[int, int, int], ...] = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def couple_m0(a, b):
    """Real axisymmetric (m=0) PSTF star product of two amplitude spectra.

    c_J = sum_{la,lb} <la 0, lb 0 | J 0> a_{la} b_{lb}, for J = 0 .. (len(a)-1)+(len(b)-1).
    This is the m=0 sector of :func:`pstf_trilinear_kernel.star_product`: with axisymmetric
    spherical components x_{l0}=sqrt(4pi) a_l, the M=0 output component equals sqrt(4pi) c_J.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    La, Lb = len(a) - 1, len(b) - 1
    if La < 0 or Lb < 0:
        raise ValueError("couple_m0 requires non-empty amplitude spectra (>= monopole)")
    out = np.zeros(La + Lb + 1, dtype=np.float64)
    for la in range(La + 1):
        if a[la] == 0.0:
            continue
        for lb in range(Lb + 1):
            if b[lb] == 0.0:
                continue
            for J in range(abs(la - lb), la + lb + 1):
                cg = clebsch_gordan(la, 0, lb, 0, J, 0)
                if cg != 0.0:
                    out[J] += cg * a[la] * b[lb]
    return out


def lrs_trilinear_amplitude(gi, gj, gk):
    """Axisymmetric trilinear output amplitude Gamma^(ijk)_L (eq 1099-1106).

    = couple_m0(couple_m0(gi, gj), gk): the (i,j)->J then (J,k)->L coupling tree. By the
    6j recoupling identity the result is independent of the coupling order (leg-exchange
    symmetric). Returns the spectrum indexed by output rank L.
    """
    return couple_m0(couple_m0(gi, gj), gk)


def axisymmetric_collision_moment(L, reduced_linear, reduced_trilinear, gammas,
                                  eps=LEG_SIGNS):
    """Axisymmetric LRS collision moment C_L (the bracket of eq 1111-1135; no 1/8, no
    energy prefactor/convolution -- those are applied by the energy-integration layer).

    Parameters
    ----------
    L : output multipole rank.
    reduced_linear : dict ``i -> {Lout: F^(1;i)_Lout}`` for legs i in 0..3.
    reduced_trilinear : dict ``(i,j,k) -> {(Lout, li, lj, lk, J): F^(3;ijk)}`` for i<j<k.
    gammas : dict ``i -> array`` of axisymmetric amplitudes Gamma^(i)_l (index = rank l).
    eps : leg signs (defaults to LEG_SIGNS).
    """
    total = 0.0
    # A missing leg in `gammas` means no stored amplitude -> contributes nothing (an
    # empty spectrum); skip rather than KeyError so sparse process inputs are tolerated.
    empty = np.zeros(0)

    # Linear: sum_i eps_i F^(1;i)_L Gamma^(i)_L
    for i in range(4):
        f1 = reduced_linear.get(i, {})
        gi = gammas.get(i, empty)
        if L in f1 and L < len(gi):
            total += eps[i] * f1[L] * gi[L]

    # Trilinear: sum_{i<j<k} eps_i eps_j eps_k sum_{li lj lk J} F^(3) <..><..> g g g
    for (i, j, k) in TRIPLES:
        f3 = reduced_trilinear.get((i, j, k))
        if not f3:
            continue
        sgn = eps[i] * eps[j] * eps[k]
        gi, gj, gk = gammas.get(i, empty), gammas.get(j, empty), gammas.get(k, empty)
        for (lout, li, lj, lk, jcoup), fval in f3.items():
            if lout != L:
                continue
            # ranks beyond the stored spectrum have zero amplitude (l > l_max) -> skip
            if li >= len(gi) or lj >= len(gj) or lk >= len(gk):
                continue
            cg1 = clebsch_gordan(li, 0, lj, 0, jcoup, 0)
            if cg1 == 0.0:
                continue
            cg2 = clebsch_gordan(jcoup, 0, lk, 0, L, 0)
            if cg2 == 0.0:
                continue
            total += sgn * fval * cg1 * cg2 * gi[li] * gj[lj] * gk[lk]

    return total


def axisymmetric_collision_moment_jacobian(L, reduced_linear, reduced_trilinear, gammas,
                                           eps=LEG_SIGNS):
    """Analytic Jacobian dC_L/dGamma^(m)_l of :func:`axisymmetric_collision_moment`.

    C_L is cubic in the amplitudes, so the derivative is the same product rule as the
    production tensor Jacobian (plan A1/B.J): the linear term gives eps_i F^(1;i)_L at rank L,
    and each trilinear monomial contributes with ONE Gamma-leg dropped (three terms, the three
    legs of the triple). No AD of the kernel. Returns ``{m: array}`` with ``jac[m][l] =
    dC_L/dGamma^(m)_l`` sized to each leg's stored amplitude spectrum.
    """
    empty = np.zeros(0)
    jac = {m: np.zeros(len(gammas.get(m, empty))) for m in range(4)}

    # Linear: dC_L/dGamma^(i)_L += eps_i F^(1;i)_L
    for i in range(4):
        f1 = reduced_linear.get(i, {})
        gi = gammas.get(i, empty)
        if L in f1 and L < len(gi):
            jac[i][L] += eps[i] * f1[L]

    # Trilinear: product rule -- each monomial drops one of its three (distinct) Gamma legs
    for (i, j, k) in TRIPLES:
        f3 = reduced_trilinear.get((i, j, k))
        if not f3:
            continue
        sgn = eps[i] * eps[j] * eps[k]
        gi, gj, gk = gammas.get(i, empty), gammas.get(j, empty), gammas.get(k, empty)
        for (lout, li, lj, lk, jcoup), fval in f3.items():
            if lout != L:
                continue
            if li >= len(gi) or lj >= len(gj) or lk >= len(gk):
                continue
            cg1 = clebsch_gordan(li, 0, lj, 0, jcoup, 0)
            if cg1 == 0.0:
                continue
            cg2 = clebsch_gordan(jcoup, 0, lk, 0, L, 0)
            if cg2 == 0.0:
                continue
            coeff = sgn * fval * cg1 * cg2
            jac[i][li] += coeff * gj[lj] * gk[lk]
            jac[j][lj] += coeff * gi[li] * gk[lk]
            jac[k][lk] += coeff * gi[li] * gj[lj]

    return jac
