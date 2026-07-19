"""rabbit.collisions.pstf_x_integral — closed-form x-integral of the reduced kernel.

Stage 0 of the nonperturbative Bianchi-I substrate (not production-wired). After the HM
body-frame polynomial reduction, the exact all-orders PSTF reduced kernel
(``neutrino_large_anisotropy_exact_PSTF_collision_ko.md``, "x 적분의 폐형식")

    F^(n;r,I) = int dy int dx Theta(D) Theta(E4-m4)/sqrt(D) W_r(x,y) Pi^(n)(x,y)

has an INNER x-integral that is analytic, because W_r(x,y) Pi(x,y) = sum_n A_n(y) x^n
is a finite polynomial and the Jacobian D(x,y) = a(y) x^2 + b(y) x + c(y) is quadratic
in x. With the center-radius substitution x = x_c + r cos(theta) (x_c = -b/2a,
r = sqrt(b^2-4ac)/2|a|, a < 0):

    I_n = (1/sqrt(-a)) sum_{k=0}^{n} C(n,k) x_c^{n-k} r^k J_k(theta_hi, theta_lo),
    J_k(alpha,beta) = int_alpha^beta cos^k theta d theta,

evaluated over [x_lo, x_hi] = [-1,1] cap {D >= 0}. Hence the large-anisotropy kernel
keeps the SAME two energy integrals as isotropic HM at evolution time; only the 1-D y
integral is done numerically at precompute time. This module provides J_k and I_n; the
A_n(y) assembly (from W_r and Pi) and the y-integral are downstream Stage-0 pieces.
"""

from __future__ import annotations

import math


def cos_power_integral(k: int, alpha: float, beta: float) -> float:
    """J_k(alpha,beta) = int_alpha^beta cos^k(theta) d theta via the reduction formula.

    J_0 = beta - alpha; J_1 = sin(beta) - sin(alpha);
    J_k = [sin(theta) cos^{k-1}(theta)/k]_alpha^beta + (k-1)/k J_{k-2}  (k >= 2).
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    j_prev2 = beta - alpha                       # J_0
    if k == 0:
        return j_prev2
    j_prev1 = math.sin(beta) - math.sin(alpha)   # J_1
    if k == 1:
        return j_prev1
    for kk in range(2, k + 1):
        boundary = ((math.sin(beta) * math.cos(beta) ** (kk - 1)
                     - math.sin(alpha) * math.cos(alpha) ** (kk - 1)) / kk)
        j_kk = boundary + (kk - 1) / kk * j_prev2
        j_prev2, j_prev1 = j_prev1, j_kk
    return j_prev1


def reduced_x_integral(n: int, a: float, b: float, c: float) -> float:
    """I_n = int_{x_lo}^{x_hi} x^n / sqrt(a x^2 + b x + c) dx, closed form.

    D = a x^2 + b x + c with a < 0 (downward parabola; D >= 0 between its roots).
    [x_lo, x_hi] = [-1, 1] cap {D >= 0}. Returns 0.0 when that intersection is empty
    (Delta <= 0, or both roots outside [-1,1] on the same side).

    Raises ValueError for a >= 0 (out of spec: the HM Jacobian gives a < 0; the
    a -> 0 linear degeneracy is a separate branch handled where the y-grid is built).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if a >= 0.0:
        raise ValueError(f"reduced_x_integral requires a < 0 (got a={a})")
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return 0.0  # D < 0 for all x (a<0, no real roots): empty physical interval
    r = math.sqrt(disc) / (2.0 * abs(a))
    x_c = -b / (2.0 * a)
    root_lo, root_hi = x_c - r, x_c + r          # D >= 0 on [root_lo, root_hi] (a<0)
    x_lo = max(-1.0, root_lo)
    x_hi = min(1.0, root_hi)
    if x_lo >= x_hi:
        return 0.0  # roots entirely outside [-1,1]: empty intersection

    # x = x_c + r cos(theta): theta = arccos((x - x_c)/r), decreasing in x, so the
    # x in [x_lo, x_hi] maps to theta in [theta_hi, theta_lo] with theta_hi < theta_lo.
    # Clamp the arccos argument against round-off at the (possibly unclipped) roots.
    theta_lo = math.acos(min(1.0, max(-1.0, (x_lo - x_c) / r)))
    theta_hi = math.acos(min(1.0, max(-1.0, (x_hi - x_c) / r)))

    inv_sqrt_minus_a = 1.0 / math.sqrt(-a)
    total = 0.0
    for kk in range(0, n + 1):
        binom = math.comb(n, kk)
        # J_k(theta_hi, theta_lo): alpha = theta_hi, beta = theta_lo (spec convention)
        total += binom * x_c ** (n - kk) * r ** kk * cos_power_integral(kk, theta_hi, theta_lo)
    return inv_sqrt_minus_a * total
