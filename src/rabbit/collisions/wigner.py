"""rabbit.collisions.wigner — dependency-free Wigner 3j/6j + Clebsch-Gordan.

Integer-angular-momentum Racah formulas (no sympy dependency), used by the exact
all-orders PSTF collision kernel (Stage 0). Values verified against closed forms
and orthogonality in tests/test_wigner_and_star_product.py.
"""

from __future__ import annotations

import math
from functools import lru_cache


def _triangle(a: int, b: int, c: int) -> float:
    """Triangle coefficient Delta(a,b,c) = (a+b-c)!(a-b+c)!(-a+b+c)!/(a+b+c+1)!."""
    return (math.factorial(a + b - c) * math.factorial(a - b + c)
            * math.factorial(-a + b + c)) / math.factorial(a + b + c + 1)


@lru_cache(maxsize=None)
def wigner_3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """Wigner 3j symbol (integer arguments), Racah single-sum form."""
    if m1 + m2 + m3 != 0:
        return 0.0
    if not (abs(j1 - j2) <= j3 <= j1 + j2):
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    pref = ((-1) ** (j1 - j2 - m3)) * math.sqrt(_triangle(j1, j2, j3))
    pref *= math.sqrt(
        math.factorial(j1 + m1) * math.factorial(j1 - m1)
        * math.factorial(j2 + m2) * math.factorial(j2 - m2)
        * math.factorial(j3 + m3) * math.factorial(j3 - m3)
    )
    kmin = max(0, j2 - j3 - m1, j1 - j3 + m2)
    kmax = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    ssum = 0.0
    for k in range(kmin, kmax + 1):
        terms = (k, j1 + j2 - j3 - k, j1 - m1 - k, j2 + m2 - k,
                 j3 - j2 + m1 + k, j3 - j1 - m2 + k)
        if any(t < 0 for t in terms):
            continue
        denom = 1.0
        for t in terms:
            denom *= math.factorial(t)
        ssum += ((-1) ** k) / denom
    return pref * ssum


def clebsch_gordan(j1: int, m1: int, j2: int, m2: int, j3: int, m3: int) -> float:
    """<j1 m1, j2 m2 | j3 m3> from the 3j symbol."""
    if m3 != m1 + m2:
        return 0.0
    if not (abs(j1 - j2) <= j3 <= j1 + j2):
        return 0.0
    return ((-1) ** (j1 - j2 + m3)) * math.sqrt(2 * j3 + 1) * wigner_3j(j1, j2, j3, m1, m2, -m3)


@lru_cache(maxsize=None)
def wigner_6j(j1: int, j2: int, j3: int, j4: int, j5: int, j6: int) -> float:
    """Wigner 6j symbol (integer arguments), Racah single-sum form."""
    for a, b, c in ((j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3)):
        if not (abs(a - b) <= c <= a + b):
            return 0.0
    pref = math.sqrt(_triangle(j1, j2, j3) * _triangle(j1, j5, j6)
                     * _triangle(j4, j2, j6) * _triangle(j4, j5, j3))
    alpha = (j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3)
    beta = (j1 + j2 + j4 + j5, j2 + j3 + j5 + j6, j3 + j1 + j6 + j4)
    ssum = 0.0
    for t in range(max(alpha), min(beta) + 1):
        denom = 1.0
        ok = True
        for a in alpha:
            if t - a < 0:
                ok = False
                break
            denom *= math.factorial(t - a)
        if not ok:
            continue
        for b in beta:
            if b - t < 0:
                ok = False
                break
            denom *= math.factorial(b - t)
        if not ok:
            continue
        ssum += ((-1) ** t) * math.factorial(t + 1) / denom
    return pref * ssum
