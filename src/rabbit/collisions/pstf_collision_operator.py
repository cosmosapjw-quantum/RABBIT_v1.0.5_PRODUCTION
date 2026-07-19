"""rabbit.collisions.pstf_collision_operator — isotropic PSTF collision operator.

Stage 0 of the nonperturbative Bianchi-I substrate (not production-wired). The first live
collision operator assembled from the exact-PSTF substrate: the isotropic HM collision
moment (``neutrino_decoupling_PSTF_HM_generalization_ko.md``, eq 616-622)

    Chat_0(p1) = (2/(2pi)^4)(1/2E1) int p2^2 dp2/2E2 int p3^2 dp3/2E3 Lambda_r^(0) F_0,

with the gain-minus-loss factor Lambda_r^(0) = (1-f1)(1-f2) f3 f4 - f1 f2 (1-f3)(1-f4) and
p4 = p1+p2-p3 on-shell. F_0 is the validated reduced kernel (:mod:`pstf_reduced_kernel`).

Detailed balance is exact: at any Fermi-Dirac equilibrium f = 1/(1+e^{(p-mu)/T}) energy
conservation p1+p2 = p3+p4 forces Lambda_r^(0) = f1 f2 f3 f4 (e^{p1+p2}-e^{p3+p4}) = 0
pointwise, so Chat_0 = 0 -- provided f4 is evaluated at the exact p4 (the operator takes f
as a callable, so no interpolation breaks the balance). This is the isotropic engine; the
anisotropic L>0 moments add the Legendre / trilinear reduced kernels and the m=0 PSTF
contraction (:mod:`pstf_lrs_moment_kernel`).
"""

from __future__ import annotations

import math

import numpy as np

from rabbit.collisions.deterministic_reference import collision_statistical_factor
from rabbit.collisions.pstf_lrs_moment_kernel import (
    TRIPLES,
    axisymmetric_collision_moment,
)
from rabbit.collisions.pstf_pi3_analytic import has_analytic_pi3 as _has_analytic_pi3
from rabbit.collisions.pstf_reduced_kernel import (
    reduced_kernel_F0,
    reduced_kernel_F3,
    reduced_kernel_F3_analytic,
    reduced_kernel_F_legendre,
)

#: 2 / (2 pi)^4 -- the HM isotropic phase-space prefactor (eq 618).
_PHASE_PREFACTOR: float = 2.0 / (2.0 * math.pi) ** 4


def isotropic_collision_moment_C0(f, p1, p_nodes, p_weights, scrW_x_coeffs_factory,
                                  *, masses=(0.0, 0.0, 0.0, 0.0), n_y: int = 48):
    """Isotropic collision moment Chat_0(p1) for a 2<->2 process (massless legs by default).

    Parameters
    ----------
    f : callable ``p -> occupancy in (0,1)`` (evaluated at the exact p4, so detailed
        balance stays exact -- pass a callable, not a grid+interpolation).
    p1 : external leg-1 momentum.
    p_nodes, p_weights : quadrature for the int_0^inf dp2 and dp3 momentum integrals.
    scrW_x_coeffs_factory : callable ``(p1,p2,p3) -> (A_0,A_1,...)``, the W_r x-polynomial
        feeding :func:`pstf_reduced_kernel.reduced_kernel_F0`.
    masses : (m1,m2,m3,m4); only m4 enters F_0's Theta(E4-m4) gate (massless default).
    n_y : Gauss order for the reduced kernel's inner y-integral.

    Returns Chat_0(p1). Configurations with p4 = p1+p2-p3 <= 0 are kinematically dropped.
    """
    m1, m2, m3, m4 = masses
    E1 = math.hypot(p1, m1)
    f1 = float(f(p1))

    acc = 0.0
    for p2, w2 in zip(p_nodes, p_weights):
        p2 = float(p2)
        E2 = math.hypot(p2, m2)
        f2 = float(f(p2))
        for p3, w3 in zip(p_nodes, p_weights):
            p3 = float(p3)
            E3 = math.hypot(p3, m3)
            # leg 4 is on-shell: E4 = E1+E2-E3, p4 = sqrt(E4^2 - m4^2) (energy, not
            # momentum, is conserved -- identical to p1+p2-p3 only when massless).
            E4 = E1 + E2 - E3
            if E4 <= m4:
                continue
            p4 = math.sqrt(E4 * E4 - m4 * m4)
            f3 = float(f(p3))
            f4 = float(f(p4))
            lam = collision_statistical_factor(f1, f2, f3, f4)
            if lam == 0.0:
                continue  # exact 0 contributes exactly 0; skip the costly F_0 eval
            F0 = reduced_kernel_F0((E1, E2, E3), (p1, p2, p3), m4,
                                   scrW_x_coeffs_factory(p1, p2, p3), n_y=n_y)
            measure = (p2 * p2 / (2.0 * E2)) * (p3 * p3 / (2.0 * E3))
            acc += float(w2) * float(w3) * measure * lam * F0

    return _PHASE_PREFACTOR * (1.0 / (2.0 * E1)) * acc


def isotropic_collision_moment_rate(f, p_nodes, p_weights, scrW_x_coeffs_factory,
                                    *, power: int = 3, masses=(0.0, 0.0, 0.0, 0.0),
                                    n_y: int = 48):
    """Momentum-weighted moment of the isotropic collision operator.

    rate = sum_i w_i p_i^power Chat_0(p_i). power=3 gives the energy-transfer rate
    (drho_nu/dt up to g/(2 pi^2)); power=2 gives the number-transfer rate. Both vanish at
    any Fermi-Dirac equilibrium (Chat_0 = 0 pointwise by detailed balance). This is the
    N_eff-relevant heating/cooling rate -- the quantity to compare against the production
    exact kernel (cross-track diagnostic for the Mangano gap).
    """
    total = 0.0
    for p1, w1 in zip(p_nodes, p_weights):
        c0 = isotropic_collision_moment_C0(f, float(p1), p_nodes, p_weights,
                                           scrW_x_coeffs_factory, masses=masses, n_y=n_y)
        total += float(w1) * float(p1) ** power * c0
    return total


def _trilinear_rank_combos(L: int, ell_max: int):
    """Valid (li,lj,lk,J) for the axisymmetric trilinear term up to ell_max.

    m=0 CG non-vanishing requires the triangle (J in [|li-lj|, li+lj], L in [|J-lk|, J+lk])
    and parity L + li + lj + lk even (else <li0,lj0|J0> or <J0,lk0|L0> = 0).
    """
    combos = []
    for li in range(ell_max + 1):
        for lj in range(ell_max + 1):
            for lk in range(ell_max + 1):
                if (L + li + lj + lk) % 2 != 0:
                    continue
                for J in range(abs(li - lj), li + lj + 1):
                    if abs(J - lk) <= L <= J + lk:
                        combos.append((li, lj, lk, J))
    return combos


def lrs_collision_moment_CL(L, gamma_spectrum, p1, p_nodes, p_weights, scrW_x_coeffs_factory,
                            *, ell_max: int = 2, masses=(0.0, 0.0, 0.0, 0.0),
                            n_y: int = 48, n_theta: int = 48, use_analytic_f3: bool = False):
    """Axisymmetric LRS collision moment Chat_L(p1) (spec eq 1111-1135).

    Chat_L = (2/(2pi)^4)(1/2E1)(1/8) int p2^2 dp2/2E2 int p3^2 dp3/2E3 bracket_L, where
    bracket_L is the m=0 PSTF contraction (:func:`pstf_lrs_moment_kernel.axisymmetric_collision_moment`)
    of the reduced kernels F^(1)/F^(3) against the per-leg amplitudes.

    Parameters
    ----------
    L : output multipole rank.
    gamma_spectrum : callable ``p -> array`` of axisymmetric amplitudes Gamma_l(p) =
        g_{l0}(p)/sqrt(4pi) (Gamma_0 = G_0 = 1 - 2 f_0). Evaluated at each leg's momentum
        (leg 4 on-shell), so detailed balance stays exact.
    p_nodes, p_weights : momentum quadrature for the dp2, dp3 integrals.
    scrW_x_coeffs_factory : ``(p1,p2,p3) -> (y -> (A_0,A_1,...))`` the W_r x-polynomial.
    ell_max : highest input multipole carried by the amplitudes (caps the trilinear sum).
    masses, n_y, n_theta : as in the reduced-kernel builders.

    The L=0 all-monopole case reproduces :func:`isotropic_collision_moment_C0` (bracket_0 =
    8 Lambda_r F_0, times 1/8); Chat_{L>0} vanishes at all-monopole.
    """
    m1, m2, m3, m4 = masses
    E1 = math.hypot(p1, m1)

    def _amplitudes(p):
        """Gamma spectrum at p, fail-loud if it carries multipoles beyond ell_max.

        ell_max caps the trilinear sum, so any leg amplitude with ranks > ell_max would
        lose contributions unnoticed -- guard EVERY leg, not just leg 1 (the spectrum
        length may vary with momentum).
        """
        g = np.asarray(gamma_spectrum(p), dtype=float)
        rank = len(g) - 1
        if rank > ell_max:
            raise ValueError(
                f"ell_max={ell_max} truncates amplitude multipoles up to rank {rank} "
                f"at p={p}; raise ell_max to >= {rank}")
        return g

    g1 = _amplitudes(p1)
    combos = _trilinear_rank_combos(L, ell_max)
    acc = 0.0
    for p2, w2 in zip(p_nodes, p_weights):
        p2 = float(p2)
        E2 = math.hypot(p2, m2)
        g2 = _amplitudes(p2)
        for p3, w3 in zip(p_nodes, p_weights):
            p3 = float(p3)
            E3 = math.hypot(p3, m3)
            E4 = E1 + E2 - E3
            if E4 <= m4:
                continue
            p4 = math.sqrt(E4 * E4 - m4 * m4)
            g3 = _amplitudes(p3)
            g4 = _amplitudes(p4)
            gammas = {0: g1, 1: g2, 2: g3, 3: g4}

            energies = (E1, E2, E3)
            momenta = (p1, p2, p3)
            xpoly = scrW_x_coeffs_factory(p1, p2, p3)

            # F^(1;i)_L for each leg (0-based i -> physical leg i+1)
            reduced_linear = {
                i: {L: reduced_kernel_F_legendre(L, i + 1, energies, momenta, m4, xpoly, n_y=n_y)}
                for i in range(4)
            }
            # F^(3;ijk)_{L; li lj lk; J} for each particle-hole triple. With use_analytic_f3,
            # use the CAS-exact closed-form kernel where the case is generated (no theta-quad
            # under-resolution), else fall back to the numerical theta-quadrature kernel.
            reduced_trilinear = {}
            for (i, j, k) in TRIPLES:
                legs = (i + 1, j + 1, k + 1)
                entries = {}
                for (li, lj, lk, J) in combos:
                    if use_analytic_f3 and _has_analytic_pi3(L, J, (li, lj, lk), legs):
                        val = reduced_kernel_F3_analytic(
                            L, J, (li, lj, lk), legs, energies, momenta, m4, xpoly, n_y=n_y)
                    else:
                        val = reduced_kernel_F3(
                            L, J, (li, lj, lk), legs, energies, momenta, m4, xpoly,
                            n_y=n_y, n_theta=n_theta)
                    entries[(L, li, lj, lk, J)] = val
                reduced_trilinear[(i, j, k)] = entries

            bracket = axisymmetric_collision_moment(L, reduced_linear, reduced_trilinear, gammas)
            measure = (p2 * p2 / (2.0 * E2)) * (p3 * p3 / (2.0 * E3))
            acc += float(w2) * float(w3) * measure * bracket

    return _PHASE_PREFACTOR * (1.0 / (2.0 * E1)) * (1.0 / 8.0) * acc
