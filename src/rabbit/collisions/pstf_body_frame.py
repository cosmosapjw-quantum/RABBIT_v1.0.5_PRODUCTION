"""rabbit.collisions.pstf_body_frame — generalized HM angular polynomial.

Stage 0 of the nonperturbative Bianchi-I substrate (not production-wired). Implements
the body-frame factorization that makes the exact all-orders PSTF collision multipole
tractable at arbitrary anisotropy (``neutrino_large_anisotropy_exact_PSTF_collision_ko.md``).

The global rotation R in SO(3) is integrated out exactly with the Haar identity,
leaving a *shape* integral over the canonical HM body frame parametrized by

    x = n1 . n2,    y = n1 . n3,    s = ± (the two on-shell azimuth roots),

with n1 = z-hat. The Haar average collapses each multilinear collision term to the
generalized angular polynomial (dimensionless, normalized so all-monopole -> 1):

    Pi^(3;ijk)_{L; li lj lk; J}(x,y)
        = (8 pi^2 / sqrt(4 pi (2L+1))) sum_{s=±} sum_{mi+mj+mk=0}
              <li mi, lj mj | J, mi+mj> <J, mi+mj; lk mk | L 0>
              Y_{li mi}(n_i^s) Y_{lj mj}(n_j^s) Y_{lk mk}(n_k^s),

with the n=1 reduction Pi^(1;i)_L = P_L(mu_{1i}) (the existing Legendre-inserted HM
kernel) and reality T(beta_-) = T(beta_+)* (Condon-Shortley) so the stored kernel is
real. This module is the verified *angular core*; the radial reduced kernel
F^(n;r,I) = int dy dx Theta(D) Theta(E4-m4)/sqrt(D) W_r(x,y) Pi^(n)(x,y) multiplies it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import lpmv

from rabbit.collisions.wigner import clebsch_gordan

SQRT_4PI: float = math.sqrt(4.0 * math.pi)


def spherical_harmonic(ell: int, m: int, n_vec) -> complex:
    """Complex spherical harmonic Y_{ell m}(n) for a unit 3-vector n.

    Condon-Shortley convention (the (-1)^m phase rides in scipy's ``lpmv``), matching
    the Clebsch-Gordan basis in :mod:`rabbit.collisions.wigner`. Negative m via
    Y_{ell,-m} = (-1)^m conj(Y_{ell m}).
    """
    n = np.asarray(n_vec, dtype=np.float64)
    cos_theta = float(n[2])
    if m < 0:
        return (-1.0) ** m * np.conj(spherical_harmonic(ell, -m, n))
    norm = math.sqrt((2 * ell + 1) / (4.0 * math.pi)
                     * math.factorial(ell - m) / math.factorial(ell + m))
    phi = math.atan2(float(n[1]), float(n[0]))
    return norm * lpmv(m, ell, cos_theta) * np.exp(1j * m * phi)


@dataclass(frozen=True)
class BodyFrame:
    """Canonical HM body frame at fixed (p1,p2,p3,m4,x,y); ``valid`` if D > 0.

    ``n3``/``n4`` are 2-tuples indexed by the azimuth root s (0 -> beta_+, 1 -> beta_-).
    """

    n1: np.ndarray
    n2: np.ndarray
    n3: tuple
    n4: tuple
    E4: float
    p4: float
    B: float
    D: float
    valid: bool

    def direction(self, leg: int, s: int) -> np.ndarray:
        """Unit vector of leg in {1,2,3,4} at azimuth root s in {0,1}."""
        if leg == 1:
            return self.n1
        if leg == 2:
            return self.n2
        if leg == 3:
            return self.n3[s]
        if leg == 4:
            return self.n4[s]
        raise ValueError(f"leg must be 1..4, got {leg}")


def collision_body_frame(E1, p1, E2, p2, E3, p3, m4, x, y) -> BodyFrame:
    """Construct the canonical HM body frame (spec "HM body frame와 전역회전의 분리").

    n1 = (0,0,1); n2 = (sqrt(1-x^2), 0, x); n3^s = (sqrt(1-y^2) cos b_s,
    sqrt(1-y^2) sin b_s, y) with b_± = ± arccos B; n4^s = (p1 n1 + p2 n2 - p3 n3^s)/p4.
    Returns ``valid=False`` when E4 <= m4 or |B| > 1 (no on-shell closed quadrilateral).
    """
    E4 = E1 + E2 - E3
    if E4 <= m4:
        zero = np.zeros(3)
        return BodyFrame(np.array([0.0, 0.0, 1.0]), zero, (zero, zero), (zero, zero),
                         E4, 0.0, float("nan"), -1.0, False)
    p4 = math.sqrt(E4 * E4 - m4 * m4)
    sx = math.sqrt(max(0.0, 1.0 - x * x))
    sy = math.sqrt(max(0.0, 1.0 - y * y))
    n1 = np.array([0.0, 0.0, 1.0])
    n2 = np.array([sx, 0.0, x])

    if sx == 0.0 or sy == 0.0:
        # |x| or |y| = 1: n2 (or n3) collinear with n1 — degenerate, B undefined
        # (denom -> 0). Measure-zero; fail closed rather than emit NaN as valid.
        zero = np.zeros(3)
        return BodyFrame(n1, n2, (zero, zero), (zero, zero), E4, p4, float("nan"), -1.0, False)

    denom = 2.0 * p2 * p3 * sx * sy
    B = (p1 * p1 + p2 * p2 + p3 * p3 + 2 * p1 * p2 * x - 2 * p1 * p3 * y
         - 2 * p2 * p3 * x * y - p4 * p4) / denom
    if abs(B) > 1.0:
        zero = np.zeros(3)
        return BodyFrame(n1, n2, (zero, zero), (zero, zero), E4, p4, B, -1.0, False)

    D = 4.0 * p2 * p2 * p3 * p3 * (1.0 - x * x) * (1.0 - y * y) * (1.0 - B * B)
    beta = math.acos(B)
    n3 = []
    n4 = []
    for sign in (+1.0, -1.0):
        b = sign * beta
        v3 = np.array([sy * math.cos(b), sy * math.sin(b), y])
        v4 = (p1 * n1 + p2 * n2 - p3 * v3) / p4
        n3.append(v3)
        n4.append(v4)
    return BodyFrame(n1, n2, (n3[0], n3[1]), (n4[0], n4[1]), E4, p4, B, D, True)


def body_scalar_T(L: int, J: int, ranks, dirs) -> complex:
    """Body scalar T^{LJ}_{li lj lk}(n1; ni,nj,nk) in the canonical frame.

    T^{LJ} = (1/sqrt(4 pi (2L+1))) Y^{L0}_{J; li lj lk}, the rank-L tensor formed by
    coupling Y(ni) x Y(nj) -> J then x Y(nk) -> L and projecting on M=0 (n1 = z-hat).
    Only mi+mj+mk = 0 survives.
    """
    li, lj, lk = ranks
    ni, nj, nk = dirs
    acc = 0.0 + 0.0j
    for mi in range(-li, li + 1):
        yi = spherical_harmonic(li, mi, ni)
        if yi == 0:
            continue
        for mj in range(-lj, lj + 1):
            mk = -(mi + mj)
            if abs(mk) > lk:
                continue
            mJ = mi + mj
            if abs(mJ) > J:
                continue
            cg1 = clebsch_gordan(li, mi, lj, mj, J, mJ)
            if cg1 == 0.0:
                continue
            cg2 = clebsch_gordan(J, mJ, lk, mk, L, 0)
            if cg2 == 0.0:
                continue
            yj = spherical_harmonic(lj, mj, nj)
            yk = spherical_harmonic(lk, mk, nk)
            acc += cg1 * cg2 * yi * yj * yk
    return acc / math.sqrt(4.0 * math.pi * (2 * L + 1))


def angular_polynomial_n1(L: int, i: int, frame: BodyFrame) -> float:
    """Pi^(1;i)_L = P_L(mu_{1i}) — the Legendre-inserted linear HM kernel."""
    mu = float(np.dot(frame.n1, frame.direction(i, 0)))
    # Legendre P_L via the m=0 spherical harmonic normalization is overkill; use the
    # standard recurrence through numpy's evaluation for an exact closed form.
    return float(np.polynomial.legendre.legval(mu, [0] * L + [1]))


def angular_polynomial_n3(L: int, J: int, ranks, legs, frame: BodyFrame) -> float:
    """Generalized trilinear angular polynomial Pi^(3;ijk)_{L; li lj lk; J}(x,y).

    legs = (i,j,k) selects body directions in {1,2,3,4}; ranks = (li,lj,lk). Sums the
    two azimuth roots (the result is real by T(beta_-) = T(beta_+)*); returns the real
    part. All-monopole inputs give 1; two rank-0 legs reduce to angular_polynomial_n1.
    """
    pref = 8.0 * math.pi * math.pi
    total = 0.0 + 0.0j
    for s in (0, 1):
        dirs = (frame.direction(legs[0], s),
                frame.direction(legs[1], s),
                frame.direction(legs[2], s))
        total += pref * body_scalar_T(L, J, ranks, dirs)
    return float(total.real)
