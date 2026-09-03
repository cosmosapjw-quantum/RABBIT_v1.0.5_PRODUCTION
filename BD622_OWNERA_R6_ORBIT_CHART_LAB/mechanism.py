"""Repository-free nonphysical C-R6-ORBIT-CHART discriminator mechanism.

Single finalist frozen in EXPERIMENT_CONTRACT.json: two paired ordered
orientation members closed by the derived Reynolds one-half quotient.
Imports nothing from rabbit.*; every spectrum, kernel, and grid is
manufactured and nonphysical.
"""

from fractions import Fraction

YMAX = Fraction(4)
NMODES = 4
TVALS = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
RAT_NODES = (Fraction(1, 2), Fraction(3, 2), Fraction(5, 2), Fraction(7, 2))
RAT_WEIGHTS = (Fraction(1), Fraction(1), Fraction(1), Fraction(1))

SPECIES = ("mu", "mubar", "tau", "taubar")
P_PERM = (2, 3, 0, 1)
MEMBER_P = (0, 1, 2, 3)
MEMBER_M = (2, 3, 0, 1)
QUOTIENT = Fraction(1, 2)
CLOSURE = ((MEMBER_P, QUOTIENT), (MEMBER_M, QUOTIENT))
LANES = (1, 1, -1, -1)
SIGN_FORM = "gml_pppp_mm"
NATIVE_ROUNDTRIP_ENABLED = False
CANONICAL_SORT_ENABLED = True

# Modal basis b_k(y) = LegendreP_k(y/2 - 1), expanded in y (ascending powers).
BASIS_POLY = (
    (Fraction(1),),
    (Fraction(-1), Fraction(1, 2)),
    (Fraction(1), Fraction(-3, 2), Fraction(3, 8)),
    (Fraction(-1), Fraction(3), Fraction(-15, 8), Fraction(5, 16)),
)
GRAM = (Fraction(4), Fraction(4, 3), Fraction(4, 5), Fraction(4, 7))


class Dual:
    """Forward-mode dual number over exact rationals: a + eps*b."""

    __slots__ = ("a", "b")

    def __init__(self, a, b=0):
        self.a = a if isinstance(a, Fraction) else Fraction(a)
        self.b = b if isinstance(b, Fraction) else Fraction(b)

    def _lift(self, other):
        if isinstance(other, Dual):
            return other
        return Dual(other)

    def __add__(self, other):
        o = self._lift(other)
        return Dual(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __sub__(self, other):
        o = self._lift(other)
        return Dual(self.a - o.a, self.b - o.b)

    def __rsub__(self, other):
        o = self._lift(other)
        return Dual(o.a - self.a, o.b - self.b)

    def __mul__(self, other):
        o = self._lift(other)
        return Dual(self.a * o.a, self.a * o.b + self.b * o.a)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = self._lift(other)
        return Dual(self.a / o.a, (self.b * o.a - self.a * o.b) / (o.a * o.a))

    def __neg__(self):
        return Dual(-self.a, -self.b)

    def __eq__(self, other):
        o = self._lift(other)
        return self.a == o.a and self.b == o.b

    def __repr__(self):
        return "Dual(%s, %s)" % (self.a, self.b)


def kernel(y1, y2, t):
    return y1 * y1 * y2 * y2 * t * (1 - t)


def pauli_drive(f1, f2, f3, f4, form=None):
    if form is None:
        form = SIGN_FORM
    gain = (1 - f1) * (1 - f2) * f3 * f4
    loss = f1 * f2 * (1 - f3) * (1 - f4)
    if form == "gml_pppp_mm":
        return gain - loss
    if form == "lmg_mmpp":
        return loss - gain
    raise ValueError("unknown sign form: %r" % (form,))


def lanes_for(form=None):
    if form is None:
        form = SIGN_FORM
    if form == "gml_pppp_mm":
        return LANES
    if form == "lmg_mmpp":
        return tuple(-s for s in LANES)
    raise ValueError("unknown sign form: %r" % (form,))


def poly_eval(p, y):
    acc = p[-1] * (y * 0 + 1)
    for c in reversed(p[:-1]):
        acc = acc * y + c
    return acc


def poly_mul(p, q):
    out = [(p[0] * q[0]) * 0] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        for j, qj in enumerate(q):
            out[i + j] = out[i + j] + pi * qj
    return out


def poly_integral(p, upper):
    total = (p[0] * upper) * 0
    power = upper
    for n, c in enumerate(p):
        total = total + c * power / (n + 1)
        power = power * upper
    return total


def basis_at(y):
    return tuple(poly_eval(p, y) for p in BASIS_POLY)


def lagrange_eval(nodes, values, y):
    total = None
    for i, yi in enumerate(nodes):
        term = values[i]
        for j, yj in enumerate(nodes):
            if j != i:
                term = term * (y - yj) / (yi - yj)
        total = term if total is None else total + term
    return total


def lagrange_poly(xs, ys):
    total = [ys[0] * 0]
    for i, xi in enumerate(xs):
        term = [ys[i]]
        for j, xj in enumerate(xs):
            if j != i:
                term = poly_mul(term, [-xj / (xi - xj), 1 / (xi - xj)])
        if len(total) < len(term):
            total = total + [total[0] * 0] * (len(term) - len(total))
        for n, c in enumerate(term):
            total[n] = total[n] + c
    return total


def native_roundtrip(row):
    """Lossy per-member native 1/y^2 round trip through the 3-point subgrid.

    Dead code in the frozen finalist (the module flag above stays False);
    MUT-6 enables it to realize the forbidden reduction-order pattern.
    """
    sub = RAT_NODES[:3]
    vals = []
    for y in sub:
        rho = row[0] * 0
        b = basis_at(y)
        for m in range(NMODES):
            rho = rho + row[m] * b[m] / GRAM[m]
        vals.append(rho / (y * y))
    p = lagrange_poly(sub, vals)
    dens = poly_mul([Fraction(0), Fraction(0), Fraction(1)], p)
    return tuple(
        poly_integral(poly_mul(dens, list(BASIS_POLY[m])), YMAX)
        for m in range(NMODES)
    )


def generate_events(grid, scramble=False):
    nodes, _weights, tvals, ymax = grid
    events = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            for k in range(len(tvals)):
                y1, y2, t = nodes[i], nodes[j], tvals[k]
                s = y1 + y2
                y3 = t * s
                y4 = (1 - t) * s
                if 0 < y3 < ymax and 0 < y4 < ymax:
                    events.append((i, j, k, y1, y2, y3, y4))
    if scramble:
        events.reverse()
    return events


def evaluate_member(nodal, member, grid, scramble=False, form=None, trace=None):
    nodes, weights, tvals, _ymax = grid
    lanes = lanes_for(form)
    zero = nodal[0][0] * 0
    c = [[zero] * NMODES for _ in range(len(SPECIES))]
    addends = []
    for (i, j, k, y1, y2, y3, y4) in generate_events(grid, scramble):
        t = tvals[k]
        f1 = nodal[member[0]][i]
        f2 = nodal[member[1]][j]
        f3 = lagrange_eval(nodes, nodal[member[2]], y3)
        f4 = lagrange_eval(nodes, nodal[member[3]], y4)
        d = pauli_drive(f1, f2, f3, f4, form)
        w = weights[i] * weights[j] * kernel(y1, y2, t)
        ys = (y1, y2, y3, y4)
        if trace is not None:
            trace.append({"event": (i, j, k), "member": member, "y": ys,
                          "f": (f1, f2, f3, f4), "drive": d, "weight": w})
        for leg in range(4):
            b = basis_at(ys[leg])
            for m in range(NMODES):
                addends.append(
                    ((i, j, k, leg, m), member[leg], m, lanes[leg] * w * d * b[m])
                )
    if CANONICAL_SORT_ENABLED:
        addends.sort(key=lambda entry: entry[0])
    for _key, sp, m, val in addends:
        c[sp][m] = c[sp][m] + val
    if NATIVE_ROUNDTRIP_ENABLED:
        c = [list(native_roundtrip(row)) for row in c]
    return c


def sigma_member(member):
    """Label-permutation action of P on ordered members (M_PLUS <-> M_MINUS).

    The slot involution R lives in the intertwining identity
    C(Pf; M) = P.C(f; sigma(M)), not as an extra factor here.
    """
    return tuple(P_PERM[s] for s in member)


def reynolds(catalogue, quotient=None):
    """Reynolds closure operator on a member->coefficient catalogue."""
    if quotient is None:
        quotient = QUOTIENT
    out = {}
    for member, coeff in catalogue.items():
        for target in (member, sigma_member(member)):
            out[target] = out.get(target, coeff * 0) + quotient * coeff
    return out


def closed_action(nodal, grid, closure=None, scramble=False, form=None, trace=None):
    if closure is None:
        closure = CLOSURE
    zero = nodal[0][0] * 0
    out = [[zero] * NMODES for _ in range(len(SPECIES))]
    for member, q in closure:
        c = evaluate_member(nodal, member, grid, scramble, form, trace)
        for a in range(len(SPECIES)):
            for m in range(NMODES):
                out[a][m] = out[a][m] + q * c[a][m]
    return out


def permute_nodal(nodal):
    return tuple(nodal[P_PERM[s]] for s in range(len(SPECIES)))


def permute_rows(c):
    return [c[P_PERM[a]] for a in range(len(SPECIES))]


def number(c, a):
    return c[a][0]


def energy(c, a):
    return 2 * c[a][0] + 2 * c[a][1]


def s_anti(c):
    return c[2][0] + c[3][0]


RAT_GRID = (RAT_NODES, RAT_WEIGHTS, TVALS, YMAX)


def float_mirror_grid():
    return (
        tuple(float(y) for y in RAT_NODES),
        tuple(float(w) for w in RAT_WEIGHTS),
        tuple(float(t) for t in TVALS),
        float(YMAX),
    )


def gl8_grid():
    from numpy.polynomial.legendre import leggauss

    x, w = leggauss(8)
    return (
        tuple(2.0 * (xi + 1.0) for xi in x.tolist()),
        tuple(2.0 * wi for wi in w.tolist()),
        (0.25, 0.5, 0.75),
        4.0,
    )
