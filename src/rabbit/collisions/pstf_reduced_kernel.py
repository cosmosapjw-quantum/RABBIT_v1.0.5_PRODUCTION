"""rabbit.collisions.pstf_reduced_kernel — isotropic HM scalar reduced kernel F_0.

Stage 0 of the nonperturbative Bianchi-I substrate (not production-wired). Assembles the
isotropic Hannestad-Madsen scalar reduced kernel
(``neutrino_decoupling_PSTF_HM_generalization_ko.md``, eq 612-632):

    F_0^(r)(p1,p2,p3) = int_{-1}^1 dy int_{-1}^1 dx Theta(D) Theta(E4-m4)/sqrt(D) W_r(x,y),

the body-shape integral feeding the isotropic collision term
    Chat^(r),iso = (2/(2pi)^4)(1/2E1) int p2^2 dp2/2E2 int p3^2 dp3/2E3 Lambda_r^(0) F_0.

The azimuth delta-function Jacobian gives D(x,y) = a(y) x^2 + b(y) x + c(y). Writing the
on-shell cos(beta*) numerator as K + P x (K, P below), D = 4 p2^2 p3^2 (1-y^2)(1-x^2) -
(K + P x)^2, i.e.

    a = -4 p2^2 p3^2 (1-y^2) - P^2   (< 0 always),
    b = -2 K P,
    c =  4 p2^2 p3^2 (1-y^2) - K^2,
    K = p1^2 + p2^2 + p3^2 - 2 p1 p3 y - p4^2,   P = 2 p2 (p1 - p3 y).

This direct form is numerically stable (HM's b ~ (p1 - eps/p1) is not, at p1 -> 0). Every
HM invariant chi_ij is at most linear in x, so W_r is at most quadratic in x and the inner
integral is the analytic I_n closed form (:func:`pstf_x_integral.reduced_x_integral`); only
the 1-D y integral is numerical (Gauss-Legendre). This is the SAME two-energy-integral
structure as isotropic HM -- the large-anisotropy generalization keeps it intact.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial import legendre as _np_leg
from numpy.polynomial import polynomial as _np_poly

from rabbit.collisions.pstf_x_integral import reduced_x_integral


def dee_coefficients(y: float, energies, momenta, m4: float):
    """Coefficients (a, b, c) of the azimuth Jacobian D(x, y) = a x^2 + b x + c at fixed y.

    ``energies``/``momenta`` are legs 1,2,3; leg 4 is on-shell (E4 = E1+E2-E3, p4 from m4).
    a is strictly negative (downward parabola); D >= 0 selects the physical x interval.
    """
    E1, E2, E3 = (float(e) for e in energies)
    p1, p2, p3 = (float(p) for p in momenta)
    E4 = E1 + E2 - E3
    p4_sq = E4 * E4 - m4 * m4
    K = p1 * p1 + p2 * p2 + p3 * p3 - 2.0 * p1 * p3 * y - p4_sq
    P = 2.0 * p2 * (p1 - p3 * y)
    base = 4.0 * p2 * p2 * p3 * p3 * (1.0 - y * y)
    a = -base - P * P
    b = -2.0 * K * P
    c = base - K * K
    return a, b, c


def reduced_kernel_F0(energies, momenta, m4: float, scrW_x_coeffs, *, n_y: int = 64):
    """Isotropic HM reduced kernel F_0 = int dy int dx Theta(D) Theta(E4-m4)/sqrt(D) W_r.

    Parameters
    ----------
    energies, momenta : legs 1,2,3 (leg 4 on-shell from E4 = E1+E2-E3, p4 from m4).
    m4 : leg-4 mass (Theta(E4-m4) kinematic gate).
    scrW_x_coeffs : callable ``y -> (A_0, A_1, ...)`` giving W_r(x,y) = sum_n A_n(y) x^n.
    n_y : Gauss-Legendre order for the outer y integral on [-1, 1].

    The inner x integral is the exact closed form I_n; the empty-domain case (D < 0 for
    all x at that y) contributes 0 automatically.
    """
    E1, E2, E3 = (float(e) for e in energies)
    if E1 + E2 - E3 <= m4:
        return 0.0  # kinematically forbidden: Theta(E4 - m4) = 0

    nodes, weights = np.polynomial.legendre.leggauss(int(n_y))
    total = 0.0
    for yk, wk in zip(nodes, weights):
        a, b, c = dee_coefficients(float(yk), energies, momenta, m4)
        if a >= 0.0:
            continue  # degenerate slice (measure zero); reduced_x_integral needs a < 0
        coeffs = scrW_x_coeffs(float(yk))
        inner = 0.0
        for n, An in enumerate(coeffs):
            if An != 0.0:
                inner += An * reduced_x_integral(n, a, b, c)
        total += float(wk) * inner
    return total


def _mu_1i_linear(leg: int, y: float, energies, momenta, m4: float):
    """Linear-in-x coefficients (u, v) of mu_{1i}(x,y) = n_1 . n_i = u + v x.

    mu_{11}=1, mu_{12}=x, mu_{13}=y, mu_{14}=(p1 + p2 x - p3 y)/p4.
    """
    E1, E2, E3 = (float(e) for e in energies)
    p1, p2, p3 = (float(p) for p in momenta)
    if leg == 1:
        return 1.0, 0.0
    if leg == 2:
        return 0.0, 1.0
    if leg == 3:
        return float(y), 0.0
    if leg == 4:
        p4 = math.sqrt((E1 + E2 - E3) ** 2 - m4 * m4)
        return (p1 - p3 * y) / p4, p2 / p4
    raise ValueError(f"leg must be 1..4, got {leg}")


def _legendre_mu_x_coeffs(ell: int, leg: int, y: float, energies, momenta, m4: float):
    """x-polynomial coefficients of P_ell(mu_{1i}(x,y)) (mu_{1i} = u + v x)."""
    u, v = _mu_1i_linear(leg, y, energies, momenta, m4)
    leg_in_mu = _np_leg.leg2poly([0] * ell + [1])  # P_ell as coeffs in powers of mu
    base = np.array([u, v])                          # mu = u + v x
    xpoly = np.array([0.0])
    powk = np.array([1.0])                            # (u + v x)^0
    for k, ck in enumerate(leg_in_mu):
        if k > 0:
            powk = _np_poly.polymul(powk, base)
        if ck != 0.0:
            xpoly = _np_poly.polyadd(xpoly, ck * powk)
    return xpoly


def reduced_kernel_F_legendre(ell: int, leg: int, energies, momenta, m4: float,
                              scrW_x_coeffs, *, n_y: int = 64):
    """Legendre-inserted n=1 reduced kernel F_{ell,i} = int Theta(D)/sqrt(D) W_r P_ell(mu_{1i}).

    Folds the P_ell(mu_{1i}(x,y)) weight (still polynomial in x) into the W_r x-polynomial
    and reuses :func:`reduced_kernel_F0`. F_{0,i}=F_0 (P_0=1) and F_{ell,1}=F_0 (mu_{11}=1).
    """
    def combined(y):
        w = np.asarray(scrW_x_coeffs(y), dtype=float)
        pl = _legendre_mu_x_coeffs(ell, leg, y, energies, momenta, m4)
        return _np_poly.polymul(w, pl)

    return reduced_kernel_F0(energies, momenta, m4, combined, n_y=n_y)


def reduced_kernel_F3(L, J, ranks, legs, energies, momenta, m4: float, scrW_x_coeffs,
                      *, n_y: int = 48, n_theta: int = 48):
    """Trilinear (n=3) reduced kernel F^(3;ijk)_{L;li lj lk;J} (spec eq 595-606).

    F^(3) = int dy int dx Theta(D) Theta(E4-m4)/sqrt(D) W_r(x,y) Pi^(3)(x,y), with the
    generalized HM angular polynomial Pi^(3) evaluated numerically on the body frame.
    The inner x-integral uses the substitution x = x_c + r cos(theta) (the same one the
    analytic I_n rests on): dx/sqrt(D) = dtheta/sqrt(-a), removing the 1/sqrt(D) endpoint
    singularity, so a smooth Gauss-Legendre theta rule converges spectrally.

    Reduces to F_0 at all-monopole (Pi^(3)=1) and to F_{L,i} when two legs are rank 0.
    """
    # local import: the trilinear kernel needs the body-frame angular evaluator, while
    # the scalar/Legendre kernels above are pure radial algebra (avoid a module cycle).
    from rabbit.collisions.pstf_body_frame import (
        collision_body_frame,
        angular_polynomial_n3,
    )

    E1, E2, E3 = (float(e) for e in energies)
    p1, p2, p3 = (float(p) for p in momenta)
    if E1 + E2 - E3 <= m4:
        return 0.0

    y_nodes, y_weights = np.polynomial.legendre.leggauss(int(n_y))
    th_nodes, th_weights = np.polynomial.legendre.leggauss(int(n_theta))
    total = 0.0
    for yk, wyk in zip(y_nodes, y_weights):
        yk = float(yk)
        a, b, c = dee_coefficients(yk, energies, momenta, m4)
        if a >= 0.0:
            continue
        disc = b * b - 4.0 * a * c
        if disc <= 0.0:
            continue
        r = math.sqrt(disc) / (2.0 * abs(a))
        x_c = -b / (2.0 * a)
        x_lo = max(-1.0, x_c - r)
        x_hi = min(1.0, x_c + r)
        if x_lo >= x_hi:
            continue
        # x in [x_lo, x_hi] <-> theta in [theta_hi, theta_lo] (arccos decreasing in x)
        # At an UNclipped root x_lo/x_hi = x_c -/+ r, so (x - x_c)/r = -/+1 exactly in
        # exact arithmetic; FP rounding in r can push it just past +-1, so clamp before acos.
        theta_lo = math.acos(min(1.0, max(-1.0, (x_lo - x_c) / r)))
        theta_hi = math.acos(min(1.0, max(-1.0, (x_hi - x_c) / r)))
        half = 0.5 * (theta_lo - theta_hi)
        mid = 0.5 * (theta_lo + theta_hi)
        inv_sqrt_minus_a = 1.0 / math.sqrt(-a)
        coeffs = scrW_x_coeffs(yk)

        inner = 0.0
        for tn, tw in zip(th_nodes, th_weights):
            theta = mid + half * float(tn)
            x = x_c + r * math.cos(theta)
            frame = collision_body_frame(E1, p1, E2, p2, E3, p3, m4, x, yk)
            if not frame.valid:
                continue
            pi3 = angular_polynomial_n3(L, J, ranks, legs, frame)
            w = 0.0
            for n, An in enumerate(coeffs):
                if An != 0.0:
                    w += An * x ** n
            inner += float(tw) * half * w * pi3 * inv_sqrt_minus_a
        total += float(wyk) * inner
    return total


def reduced_kernel_F3_analytic(L, J, ranks, legs, energies, momenta, m4: float,
                               scrW_x_coeffs, *, n_y: int = 64):
    """Trilinear F^(3) via the CAS-exact Pi^(3) x-polynomial + closed-form I_n (no theta-quad).

    Same integral as :func:`reduced_kernel_F3` but with Pi^(3)(x,y) supplied as an EXACT
    x-polynomial (CAS-generated, collisions/pstf_pi3_analytic.analytic_pi3_xcoeffs): the inner
    x-integral closes analytically (W_r . Pi^(3) = sum_n C_n(y) x^n -> sum_n C_n I_n), so only
    the 1-D y integral is numerical -- exact and fully converged, retiring the under-resolved
    theta-quadrature. Raises KeyError if the (L,J,ranks,legs) case was not CAS-generated.
    """
    from rabbit.collisions.pstf_pi3_analytic import analytic_pi3_xcoeffs

    coeff_fn = analytic_pi3_xcoeffs(L, J, ranks, legs)
    if coeff_fn is None:
        raise KeyError(
            f"no CAS-generated Pi^(3) x-coeffs for {(L, J, tuple(ranks), tuple(legs))}; "
            f"regenerate (python3 scripts/cas/gen_pstf_pi3.py --emit-module)")

    E1, E2, E3 = (float(e) for e in energies)
    p1, p2, p3 = (float(p) for p in momenta)
    e4 = E1 + E2 - E3
    if e4 <= m4:
        return 0.0
    p4 = math.sqrt(e4 * e4 - m4 * m4)

    y_nodes, y_weights = np.polynomial.legendre.leggauss(int(n_y))
    total = 0.0
    for yk, wyk in zip(y_nodes, y_weights):
        yk = float(yk)
        a, b, c = dee_coefficients(yk, energies, momenta, m4)
        if a >= 0.0:
            continue
        # W_r . Pi^(3) as an x-polynomial: convolve the two coefficient lists.
        w_coeffs = np.asarray(scrW_x_coeffs(yk), dtype=float)
        pi3_coeffs = np.asarray(coeff_fn(yk, p1, p2, p3, p4), dtype=float)
        prod = _np_poly.polymul(w_coeffs, pi3_coeffs)
        inner = 0.0
        for n, cn in enumerate(prod):
            if cn != 0.0:
                inner += cn * reduced_x_integral(n, a, b, c)
        total += float(wyk) * inner
    return total
