"""rabbit.collisions.pstf_trilinear_kernel — exact all-orders PSTF collision substrate.

Stage 0 of the nonperturbative Bianchi-I substrate (not production-wired). Implements
the particle-hole reduction at the heart of the exact all-orders PSTF neutrino
collision multipole (``neutrino_large_anisotropy_exact_PSTF_collision_ko.md``).

Core identity (spec eq. 8Λ): with the particle-hole central variable G_i = 1 - 2 f_i,
the on-shell 2<->2 Fermi-blocking statistical factor Lambda_r collapses to a
LINEAR + TRILINEAR polynomial in G — the quartic f_1 f_2 f_3 f_4 and the quadratic
G-combinations cancel exactly — valid at ARBITRARY anisotropy amplitude (no
small-anisotropy assumption):

    8 Lambda_r = sum_i eps_i G_i  +  sum_{i<j<k} eps_i eps_j eps_k G_i G_j G_k,
    eps = (+1, +1, -1, -1)   for legs (1,2 incoming ; 3,4 outgoing).

Equivalently, in logit variables eta_i = ln((1-f_i)/f_i) (= q + A in the augmented
representation, ``augmented_pstf_distribution``), detailed balance is manifest:

    Lambda_r = sinh[(eta_1 + eta_2 - eta_3 - eta_4)/2] / (8 prod_i cosh(eta_i/2)).

Hence ANY common-drift Fermi-Dirac equilibrium (eta_1 + eta_2 = eta_3 + eta_4 from
4-momentum conservation with alpha_1 + alpha_2 = alpha_3 + alpha_4) annihilates the
collision term exactly — not merely the isotropic equilibrium. This is the property
a strongly-anisotropic solver must preserve.

This module is the *verified algebraic foundation*; the reduced angular kernels
(linear F^(1), trilinear F^(3)) and the S2xq grid collision evaluator build on it.
It reproduces the existing tested ``deterministic_reference.collision_statistical_factor``
(the direct gain-loss form) to machine precision — the Stage-0 isotropic-HM gate.
"""

from __future__ import annotations

import math

import numpy as np

from rabbit.collisions.wigner import clebsch_gordan

#: sqrt(4 pi) — the monopole spherical component is x_00 = SQRT_4PI * X_0, the
#: convention under which a rank-0 scalar is the true star-product identity.
SQRT_4PI: float = math.sqrt(4.0 * math.pi)

#: Leg signs eps_i: incoming legs (1,2) -> +1, outgoing legs (3,4) -> -1.
LEG_SIGNS: tuple[int, int, int, int] = (+1, +1, -1, -1)

#: The four particle-hole triples (1-based leg labels). Their coefficient in the
#: trilinear sum is eps_i eps_j eps_k (computed from LEG_SIGNS), giving
#: {123:-1, 124:-1, 134:+1, 234:+1}.
TRIPLES: tuple[tuple[int, int, int], ...] = ((1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4))


def particle_hole_G(f):
    """Particle-hole central variable G = 1 - 2 f (physical range -1 <= G <= 1)."""
    return 1.0 - 2.0 * np.asarray(f, dtype=np.float64)


def occupancy_to_logit(f):
    """Logit eta = ln((1 - f) / f) of an occupancy f in (0, 1).

    Equals q + A in the augmented representation f = 1/(1 + exp(q + A)).
    """
    f_arr = np.asarray(f, dtype=np.float64)
    return np.log1p(-f_arr) - np.log(f_arr)


def lambda_r_particle_hole(f1, f2, f3, f4):
    """Statistical factor Lambda_r via the exact particle-hole G decomposition.

    Returns (1/8)[ sum_i eps_i G_i + sum_{i<j<k} eps_i eps_j eps_k G_i G_j G_k ],
    identical to ``deterministic_reference.collision_statistical_factor`` but written
    in the linear + trilinear form the all-orders PSTF kernel contracts against.
    """
    G = [particle_hole_G(f) for f in (f1, f2, f3, f4)]
    linear = sum(s * Gi for s, Gi in zip(LEG_SIGNS, G))
    trilinear = 0.0
    for (i, j, k) in TRIPLES:
        coeff = LEG_SIGNS[i - 1] * LEG_SIGNS[j - 1] * LEG_SIGNS[k - 1]
        trilinear = trilinear + coeff * G[i - 1] * G[j - 1] * G[k - 1]
    return (linear + trilinear) / 8.0


def lambda_r_logit(eta1, eta2, eta3, eta4):
    """Statistical factor Lambda_r in logit variables (detailed balance manifest).

    Lambda_r = sinh[(eta_1 + eta_2 - eta_3 - eta_4)/2] / (8 prod_i cosh(eta_i/2));
    vanishes exactly whenever eta_1 + eta_2 = eta_3 + eta_4 (any common drift).
    """
    e = [np.asarray(x, dtype=np.float64) for x in (eta1, eta2, eta3, eta4)]
    num = np.sinh(0.5 * (e[0] + e[1] - e[2] - e[3]))
    den = (8.0 * np.cosh(0.5 * e[0]) * np.cosh(0.5 * e[1])
           * np.cosh(0.5 * e[2]) * np.cosh(0.5 * e[3]))
    return num / den


def star_product(x, l1: int, y, l2: int, J: int):
    """Normalized PSTF Clebsch-Gordan product, spherical-component form.

    Couples two irreducible PSTF tensors (given by their complex spherical
    components ``x[m1]`` for m1=-l1..l1 and ``y[m2]`` for m2=-l2..l2) into rank J:

        z_{JM} = (1/sqrt(4pi)) sum_{m1+m2=M} <l1 m1, l2 m2 | J M> x_{l1 m1} y_{l2 m2}.

    The extra 1/sqrt(4pi) (beyond the unitary CG product) makes a rank-0 scalar the
    true identity: with the monopole convention x_00 = SQRT_4PI * X_0,
    ``star_product([SQRT_4PI*X0], 0, y, l, l) == X0 * y``. Returns the length-(2J+1)
    complex array z[M], M=-J..J.
    """
    x = np.asarray(x, dtype=complex)
    y = np.asarray(y, dtype=complex)
    z = np.zeros(2 * J + 1, dtype=complex)
    for iM, M in enumerate(range(-J, J + 1)):
        acc = 0.0 + 0.0j
        for im1, m1 in enumerate(range(-l1, l1 + 1)):
            m2 = M - m1
            if abs(m2) > l2:
                continue
            cg = clebsch_gordan(l1, m1, l2, m2, J, M)
            if cg != 0.0:
                acc += cg * x[im1] * y[m2 + l2]
        z[iM] = acc / SQRT_4PI
    return z
