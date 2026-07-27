"""Manufactured nonphysical occupation fixtures for the C-R6 orbit-chart lab."""

from fractions import Fraction

from mechanism import Dual

# (c0, c1, c2) per species: f(y) = c0 + c1*y + c2*y^2, order mu, mubar, tau, taubar.
F_ASYM = (
    (Fraction(1, 2), Fraction(-1, 16), Fraction(0)),
    (Fraction(1, 2), Fraction(-1, 32), Fraction(0)),
    (Fraction(1, 4), Fraction(1, 32), Fraction(0)),
    (Fraction(1, 4), Fraction(1, 64), Fraction(0)),
)
F_SYM = (F_ASYM[0], F_ASYM[1], F_ASYM[0], F_ASYM[1])


def occupancy(coeffs, y):
    c0, c1, c2 = coeffs
    return c0 + c1 * y + c2 * y * y


def nodal_values(coeffs4, nodes):
    return tuple(
        tuple(occupancy(c, y) for y in nodes) for c in coeffs4
    )


def sweep_coeffs(case):
    """Counter-derived dyadic coefficients; f stays inside (9/64, 37/64) on [0,4]."""
    out = []
    for s in range(4):
        m = 4 * case + s
        c0 = Fraction(1, 4) + Fraction(m % 8, 32)
        c1 = Fraction((m // 8) % 8 - 4, 256)
        c2 = Fraction((m // 64) % 4, 1024)
        out.append((c0, c1, c2))
    return tuple(out)


def dual_nodal_asym(nodes):
    """Asymmetric fixture lifted to duals with tangent delta-f_mu(y) = y/256."""
    out = []
    for s, coeffs in enumerate(F_ASYM):
        row = []
        for y in nodes:
            tangent = Fraction(y, 256) if s == 0 else Fraction(0)
            row.append(Dual(occupancy(coeffs, y), tangent))
        out.append(tuple(row))
    return tuple(out)
