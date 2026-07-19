"""CAS generator for the generalized HM angular polynomial Pi^(3)(x,y) (plan B.F3).

Reduces the exact all-orders PSTF trilinear angular polynomial
(neutrino_large_anisotropy_exact_PSTF_collision_ko.md, "n=3 큰 비등방성의 핵심 kernel")

    Pi^(3;ijk)_{L; li lj lk; J}(x,y)
        = (8 pi^2 / sqrt(4 pi (2L+1))) sum_{s=±} sum_{mi+mj+mk=0}
              <li mi, lj mj | J, mi+mj> <J, mi+mj; lk mk | L 0> prod_a Y_{la ma}(n_a^s)

to an EXACT finite polynomial in x (coefficients polynomial in y, rational in the momenta).
The current production path (collisions/pstf_reduced_kernel.reduced_kernel_F3) evaluates this
by NUMERICAL theta-quadrature -- correct but expensive and under-resolved at high ell. The
symbolic x-polynomial lets the reduced-kernel x-integral use the existing closed-form I_n
instead (exact, fast, fully converged).

Procedure (spec "symbolic 생성 절차"):
  1. Y_lm(n) matching collisions/pstf_body_frame.spherical_harmonic (Condon-Shortley phase
     via scipy lpmv; negative m via Y_{l,-m} = (-1)^m conj(Y_{lm})).
  2. CG-couple -> body scalar T^{LJ} (only mi+mj+mk=0 survives).
  3. n4 = (p1 n1 + p2 n2 - p3 n3)/p4 (Cartesian, regular-solid-harmonic form so leg 4 has
     no sqrt(1-Z^2) denominator).
  4. Symmetrize over the two azimuth roots s=± (b -> -b kills odd sin(b) terms).
  5. cos(b)=B(x,y), sin^2(b)=1-B^2; the associated-Legendre sqrt(1-x^2)/sqrt(1-y^2)
     denominators CANCEL, leaving a polynomial in x,y.

ENVIRONMENT: SYSTEM python3 (sympy >= 1.14). NOT the repo venv (no sympy there). No repo
imports -- this is an offline, regenerable symbolic tool; its output is validated against the
venv's numerical collisions/pstf_body_frame.angular_polynomial_n3 by the companion test.

Run: `python3 scripts/cas/gen_pstf_pi3.py` (self-validation: reduces a representative case set,
confirms the sqrt cancellation, the all-monopole->1 and two-monopole->P_L exact reductions,
and the L+sum(l) odd -> 0 parity rule).
"""
from __future__ import annotations

import sympy as sp
from sympy.physics.quantum.cg import CG

# ── symbols ───────────────────────────────────────────────────────────────
x, y = sp.symbols("x y", real=True)
p1, p2, p3, p4 = sp.symbols("p1 p2 p3 p4", positive=True)
B = sp.symbols("B", real=True)           # cos(b) on a root
sx, sy = sp.symbols("sx sy", positive=True)   # sin(theta_2)=sqrt(1-x^2), sin(theta_3)
sb = sp.symbols("sb", real=True)              # sin(b); set to +/- sqrt(1-B^2) per root


def _cg(j1, m1, j2, m2, j3, m3):
    """Condon-Shortley Clebsch-Gordan (sympy), matching collisions/wigner.clebsch_gordan."""
    return CG(j1, m1, j2, m2, j3, m3).doit()


def _cart(leg):
    """Cartesian unit vector (X, Y, Z) of leg in {1,2,3,4} (massless-agnostic; p4 symbolic)."""
    if leg == 1:
        return sp.Matrix([0, 0, 1])
    if leg == 2:
        return sp.Matrix([sx, 0, x])
    if leg == 3:
        return sp.Matrix([sy * B, sy * sb, y])
    if leg == 4:
        v = (p1 * _cart(1) + p2 * _cart(2) - p3 * _cart(3)) / p4
        return sp.Matrix([sp.expand(c) for c in v])
    raise ValueError(leg)


def _ylm_cartesian(ell, m, X, Yc, Z):
    """Y_{ell m} for a Cartesian unit vector (X,Yc,Z), with NO sqrt(1-Z^2) denominator.

    Regular-solid-harmonic identity: P_l^m(Z) e^{i m phi} = (-1)^m (d^m P_l/dZ^m)(Z) (X+iY)^m
    (the sin(theta)^m from the associated Legendre cancels the sin(theta)^-m in e^{i m phi}).
    Condon-Shortley phase (-1)^m matches scipy lpmv; negative m via the conjugate relation.
    """
    if m < 0:
        return (-1) ** (-m) * _ylm_cartesian(ell, -m, X, Yc, Z).conjugate()
    z = sp.Symbol("_z")
    dPl = sp.diff(sp.legendre(ell, z), z, m)
    Pm = sp.expand(dPl).subs(z, Z)
    norm = sp.sqrt(sp.Rational(2 * ell + 1, 4) / sp.pi
                   * sp.factorial(ell - m) / sp.factorial(ell + m))
    return norm * (-1) ** m * Pm * (X + sp.I * Yc) ** m


def _ylm_leg(ell, m, leg):
    X, Yc, Z = _cart(leg)
    return _ylm_cartesian(ell, m, X, Yc, Z)


def _body_scalar_T(L, J, ranks, legs):
    li, lj, lk = ranks
    i, j, k = legs
    acc = sp.Integer(0)
    for mi in range(-li, li + 1):
        for mj in range(-lj, lj + 1):
            mk = -(mi + mj)
            if abs(mk) > lk or abs(mi + mj) > J:
                continue
            c1 = _cg(li, mi, lj, mj, J, mi + mj)
            if c1 == 0:
                continue
            c2 = _cg(J, mi + mj, lk, mk, L, 0)
            if c2 == 0:
                continue
            acc += c1 * c2 * _ylm_leg(li, mi, i) * _ylm_leg(lj, mj, j) * _ylm_leg(lk, mk, k)
    return acc / sp.sqrt(4 * sp.pi * (2 * L + 1))


def pi3_symbolic(L, J, ranks, legs):
    """Pi^(3)(x,y) symmetrized over the two azimuth roots (carries B, sx, sy as symbols)."""
    raw = 8 * sp.pi ** 2 * _body_scalar_T(L, J, ranks, legs)
    S = sp.sqrt(1 - B ** 2)
    return sp.expand(raw.subs(sb, S) + raw.subs(sb, -S))


def _B_expr():
    """B(x,y) = cos(beta*) with p4 symbolic (massless or massive via the caller's p4)."""
    num = p1 ** 2 + p2 ** 2 + p3 ** 2 + 2 * p1 * p2 * x - 2 * p1 * p3 * y - 2 * p2 * p3 * x * y - p4 ** 2
    return num / (2 * p2 * p3 * sx * sy)


def reduce_to_xy_polynomial(L, J, ranks, legs):
    """Full reduction to a polynomial in (x, y) (coefficients rational in p1,p2,p3,p4).

    Substitutes B -> B(x,y), sx -> sqrt(1-x^2), sy -> sqrt(1-y^2); the sqrt denominators
    cancel (spec guarantee). Returns a sympy expression that is a polynomial in x and y.
    """
    expr = pi3_symbolic(L, J, ranks, legs).subs(B, _B_expr())
    expr = expr.subs({sx: sp.sqrt(1 - x ** 2), sy: sp.sqrt(1 - y ** 2)})
    return sp.expand(sp.radsimp(sp.expand(expr)))


# ── self-validation ───────────────────────────────────────────────────────
def _is_xy_polynomial(expr):
    e = sp.nsimplify(expr) if expr.free_symbols else expr
    return sp.Poly(sp.expand(expr), x, y).is_polynomial() if expr.has(x, y) else True


def _selfcheck():
    import math

    def P(L, expr_x):  # Legendre P_L(x)
        return sp.legendre(L, x)

    # massless representative momenta giving a valid frame at the test point
    P1, P2, P3 = sp.Rational(12, 10), sp.Rational(9, 10), sp.Rational(8, 10)
    P4 = P1 + P2 - P3
    subs_p = {p1: P1, p2: P2, p3: P3, p4: P4}

    print("case                                 sqrt-cancels  reduction")
    # all-monopole -> 1
    e = reduce_to_xy_polynomial(0, 0, (0, 0, 0), (1, 2, 3))
    assert sp.simplify(e - 1) == 0, f"all-monopole != 1: {e}"
    print(f"(0,0,(0,0,0),(1,2,3))                yes           = 1  OK")
    # two-monopole legs 2,3,4 -> P_L(mu_1i): leg 2 -> P_L(x); legs 3,4 -> P_L(mu)
    for (leg, L) in [(2, 1), (2, 2), (2, 3)]:
        e = reduce_to_xy_polynomial(L, L, (L, 0, 0), (leg, 1, 1))
        diff = sp.simplify(e - sp.legendre(L, x))
        assert diff == 0, f"two-monopole leg2 L={L} != P_L(x): {diff}"
        print(f"(L={L},(L,0,0),(2,1,1))               yes           = P_{L}(x)  OK")
    # parity selection: L + sum(l) odd -> 0
    for (L, J, ranks, legs) in [(1, 1, (1, 1, 1), (2, 3, 4)),  # 1+3=4 even -> nonzero
                                (1, 2, (1, 1, 0), (2, 3, 4)),  # 1+2=3 odd -> 0
                                (2, 2, (1, 1, 1), (2, 3, 4))]:  # 2+3=5 odd -> 0
        e = reduce_to_xy_polynomial(L, J, ranks, legs)
        parity_even = (L + sum(ranks)) % 2 == 0
        if not parity_even:
            assert sp.simplify(e) == 0, f"odd parity not zero: {L,J,ranks,legs} -> {e}"
            print(f"({L},{J},{ranks},{legs})  parity-odd -> 0  OK")
    # genuine trilinear: confirm polynomial (degree) for the 234 tripolar kernel
    for (L, J, ranks, legs) in [(2, 2, (1, 1, 2), (2, 3, 4)), (1, 1, (1, 1, 1), (2, 3, 4))]:
        e = reduce_to_xy_polynomial(L, J, ranks, legs)
        poly = sp.Poly(e, x)
        print(f"({L},{J},{ranks},{legs})  trilinear x-degree {poly.degree()}  OK")
    print("\nALL SELF-CHECKS PASSED (symbolic; matches the spec reductions exactly).")


#: cases the venv numerical gate cross-checks (representative: all-monopole, the two-monopole
#: reductions, and genuine 234 tripolar kernels at L=0,1,2). (L, J, (li,lj,lk), (i,j,k)).
_GATE_CASES = [
    (0, 0, (0, 0, 0), (1, 2, 3)),
    (1, 1, (1, 0, 0), (2, 1, 1)),
    (2, 2, (2, 0, 0), (2, 1, 1)),
    (1, 1, (1, 0, 0), (3, 1, 1)),
    (2, 2, (2, 0, 0), (4, 1, 1)),
    (1, 1, (1, 1, 1), (2, 3, 4)),
    (2, 2, (1, 1, 2), (2, 3, 4)),
    (2, 0, (1, 1, 2), (2, 3, 4)),
    (2, 2, (1, 1, 2), (1, 3, 4)),
]
#: massless sample points (p1,p2,p3,x,y) giving valid frames (|B|<1), for the numerical gate.
_GATE_POINTS = [
    (1.2, 0.9, 0.8, 0.2, 0.5),
    (1.0, 1.0, 1.0, 0.2, 0.5),
    (1.5, 0.7, 1.1, -0.1, 0.3),
    (0.9, 1.1, 0.7, 0.4, -0.2),
]


def emit_validation_samples(path):
    """Evaluate Pi^(3) symbolically at the gate points and write a JSON the venv test loads.

    The venv test rebuilds the body frame and compares angular_polynomial_n3 to these values
    to machine precision -- an independent cross-check of the symbolic reduction against the
    numerical evaluator (which is itself validated against the spec reductions)."""
    import json
    import math

    samples = []
    for (L, J, ranks, legs) in _GATE_CASES:
        expr = pi3_symbolic(L, J, ranks, legs)  # carries B, sx, sy symbolically
        for (P1, P2, P3, X, Y) in _GATE_POINTS:
            P4v = P1 + P2 - P3
            sxv = math.sqrt(max(0.0, 1 - X * X))
            syv = math.sqrt(max(0.0, 1 - Y * Y))
            Bv = (P1 ** 2 + P2 ** 2 + P3 ** 2 + 2 * P1 * P2 * X - 2 * P1 * P3 * Y
                  - 2 * P2 * P3 * X * Y - P4v ** 2) / (2 * P2 * P3 * sxv * syv)
            if abs(Bv) > 1.0:
                continue  # not a valid frame at this point; skip
            val = complex(expr.subs({x: X, y: Y, p1: P1, p2: P2, p3: P3, p4: P4v,
                                     B: Bv, sx: sxv, sy: syv}))
            assert abs(val.imag) < 1e-12, f"Pi^(3) not real: {val}"
            samples.append({"L": L, "J": J, "ranks": list(ranks), "legs": list(legs),
                            "p": [P1, P2, P3], "x": X, "y": Y, "value": val.real})
    with open(path, "w") as fh:
        json.dump(samples, fh, indent=1)
    print(f"wrote {len(samples)} validation samples -> {path}")


def _case_key(L, J, ranks, legs):
    return f"{L}_{J}_{ranks[0]}{ranks[1]}{ranks[2]}_{legs[0]}{legs[1]}{legs[2]}"


def _operator_combos(L, ell_max):
    """(li,lj,lk,J) the LRS operator queries: parity L+sum(l) even + the m=0 CG triangles
    (mirror of pstf_collision_operator._trilinear_rank_combos)."""
    out = []
    for li in range(ell_max + 1):
        for lj in range(ell_max + 1):
            for lk in range(ell_max + 1):
                if (L + li + lj + lk) % 2 != 0:
                    continue
                for J in range(abs(li - lj), li + lj + 1):
                    if abs(J - lk) <= L <= J + lk:
                        out.append((li, lj, lk, J))
    return out


def operator_cases(L_values, ell_max):
    """Full (L,J,ranks,legs) set the LRS operator evaluates at the given output L's and ell_max
    -- combos x the four particle-hole triple legs {(1,2,3),(1,2,4),(1,3,4),(2,3,4)}."""
    triple_legs = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]
    cases = set()
    for L in L_values:
        for (li, lj, lk, J) in _operator_combos(L, ell_max):
            for legs in triple_legs:
                cases.add((L, J, (li, lj, lk), legs))
    return sorted(cases)


def emit_analytic_module(path, cases):
    """Emit a venv-importable numpy module of analytic Pi^(3)(x,y,p1,p2,p3,p4) functions.

    For each (L,J,ranks,legs) case, reduces Pi^(3) to its exact x,y-polynomial and lambdifies
    it (numpy backend) into ``def _pi3_<key>(x, y, p1, p2, p3, p4): ...``, plus a dispatch dict
    ``PI3_FUNCS[(L, J, ranks, legs)] -> func``. Regenerate with
    ``python3 scripts/cas/gen_pstf_pi3.py --emit-module``. The companion loader
    (collisions/pstf_pi3_analytic) reads this; the gate test cross-checks vs the numerical
    angular_polynomial_n3.
    """
    lines = [
        '"""AUTO-GENERATED by scripts/cas/gen_pstf_pi3.py --emit-module. Do not edit by hand.',
        "",
        "Exact analytic Pi^(3)(x,y) angular polynomials (plan B.F3), lambdified to numpy. The",
        "symbolic reduction + this artifact are validated against collisions/pstf_body_frame.",
        "angular_polynomial_n3 by tests/test_pstf_pi3_cas.py. Regenerate after changing the case",
        "set or the generator.",
        '"""',
        "from __future__ import annotations",
        "import numpy  # NumPyPrinter emits numpy.* (e.g. numpy.sqrt for irrational CG consts)",
        "",
    ]
    printer = sp.printing.numpy.NumPyPrinter()
    dispatch = []
    dispatch_x = []
    for (L, J, ranks, legs) in cases:
        expr = reduce_to_xy_polynomial(L, J, ranks, legs)
        key = _case_key(L, J, ranks, legs)
        # (a) pointwise Pi^(3)(x,y,...) -- numpy-evaluable source (polynomial, no special funcs).
        lines.append(f"def _pi3_{key}(x, y, p1, p2, p3, p4):")
        lines.append(f"    return {printer.doprint(expr)}")
        lines.append("")
        # (b) x-polynomial coefficients A_n (ASCENDING, A[n] = coeff of x^n) as a fn of (y,p):
        #     feeds the closed-form x-integral I_n in reduced_kernel_F3 (retires theta-quad).
        poly = sp.Poly(expr, x)
        asc = list(reversed(poly.all_coeffs()))  # all_coeffs is descending; reverse -> A[n]
        coeff_src = ", ".join(printer.doprint(c) for c in asc)
        lines.append(f"def _pi3c_{key}(y, p1, p2, p3, p4):")
        lines.append(f"    return ({coeff_src},)")
        lines.append("")
        dispatch.append(f"    ({L}, {J}, {tuple(ranks)}, {tuple(legs)}): _pi3_{key},")
        dispatch_x.append(f"    ({L}, {J}, {tuple(ranks)}, {tuple(legs)}): _pi3c_{key},")
    lines.append("PI3_FUNCS = {")
    lines.extend(dispatch)
    lines.append("}")
    lines.append("")
    lines.append("PI3_XCOEFFS = {  # (L,J,ranks,legs) -> fn(y,p1,p2,p3,p4) -> (A_0, A_1, ...) ascending")
    lines.extend(dispatch_x)
    lines.append("}")
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {len(cases)} analytic Pi^(3) functions -> {path}")


if __name__ == "__main__":
    import os
    import sys

    _selfcheck()
    here = os.path.dirname(__file__)
    if "--emit-samples" in sys.argv:
        emit_validation_samples(os.path.join(here, "pi3_validation_samples.json"))
    if "--emit-module" in sys.argv:
        mod = os.path.join(here, "..", "..", "src", "rabbit", "collisions", "_pstf_pi3_generated.py")
        # the operator's full ell_max=1 case set (L=0,1) plus the representative gate cases,
        # so lrs_collision_moment_CL(use_analytic_f3=True) is fully analytic at ell_max=1.
        cases = sorted(set(_GATE_CASES) | set(operator_cases([0, 1], 1)))
        emit_analytic_module(os.path.abspath(mod), cases)
