"""Acceptance battery A1-A10 + anchors for the C-R6 orbit-chart lab.

Exit codes: 0 all-pass, 10 scientific FAIL (first non-PASS preserved),
20 mechanical ERROR. See EXPERIMENT_CONTRACT.json.
"""

import json
import sys
from fractions import Fraction

import fixtures as F
import mechanism as M


class CheckFailure(Exception):
    pass


AGREE = 1e-11
FLOOR = 2.0 ** -40


def require(cond, check_id, message):
    if not cond:
        raise CheckFailure("%s: %s" % (check_id, message))


def agree(fval, rval, check_id, label):
    r = float(rval)
    denom = max(abs(r), FLOOR)
    require(abs(fval - r) / denom <= AGREE, check_id,
            "agreement %s: %r vs %r" % (label, fval, r))


def G(x1, x2, x3, x4):
    return (1 - x1) * (1 - x2) * x3 * x4 - x1 * x2 * (1 - x3) * (1 - x4)


def rat_nodal(coeffs):
    return F.nodal_values(coeffs, M.RAT_NODES)


def grid_nodal(coeffs, grid):
    return tuple(tuple(float(v) for v in row)
                 for row in F.nodal_values(coeffs, grid[0]))


def entries(c):
    for a in range(len(M.SPECIES)):
        for m in range(M.NMODES):
            yield a, m, c[a][m]


def check_anchor():
    half, one = Fraction(1, 2), Fraction(1)
    require(M.kernel(half, Fraction(3, 2), Fraction(1, 4)) == Fraction(27, 256),
            "ANCHOR", "kernel(1/2,3/2,1/4)")
    require(M.kernel(one, one, half) == Fraction(1, 4), "ANCHOR", "kernel(1,1,1/2)")
    fx = [(0, half, Fraction(15, 32)), (1, Fraction(3, 2), Fraction(29, 64)),
          (2, Fraction(5, 2), Fraction(21, 64)), (3, Fraction(7, 2), Fraction(39, 128))]
    for s, y, want in fx:
        require(F.occupancy(F.F_ASYM[s], y) == want, "ANCHOR",
                "fixture species %d at %s" % (s, y))
    require(M.basis_at(Fraction(3))[2] == Fraction(-1, 8), "ANCHOR", "b_2(3)")
    require(M.basis_at(Fraction(1))[3] == Fraction(7, 16), "ANCHOR", "b_3(1)")
    return {"anchors": "all hand-derived anchors hold"}


def check_a1():
    w1 = G(Fraction(1, 2), Fraction(1, 2), Fraction(2, 3), Fraction(2, 3))
    w2 = G(Fraction(1, 4), Fraction(1, 4), Fraction(1, 3), Fraction(1, 3))
    w3 = -G(Fraction(1, 3), Fraction(1, 3), Fraction(1, 4), Fraction(1, 4))
    require(w1 == Fraction(1, 12), "A1", "G witness 1: %s" % w1)
    require(w2 == Fraction(5, 144), "A1", "G witness 2: %s" % w2)
    require(w3 == Fraction(5, 144), "A1", "G witness 3: %s" % w3)
    require(w2 != -w1, "A1", "5/144 must differ from -1/12")
    return {"D(f;z)": str(w1), "D(Pf;z)": str(w2), "-D(f;Rz)": str(w3)}


def check_a2():
    nodal = rat_nodal(F.F_ASYM)
    pnodal = M.permute_nodal(nodal)
    resid = []
    cp = M.evaluate_member(pnodal, M.MEMBER_P, M.RAT_GRID)
    pc = M.permute_rows(M.evaluate_member(nodal, M.MEMBER_P, M.RAT_GRID))
    for a, m, v in entries(cp):
        resid.append(v - pc[a][m])
    require(any(r != 0 for r in resid), "A2",
            "single-member residual must be exactly nonzero")
    fg = M.float_mirror_grid()
    fn = grid_nodal(F.F_ASYM, fg)
    fpn = M.permute_nodal(fn)
    fcp = M.evaluate_member(fpn, M.MEMBER_P, fg)
    fpc = M.permute_rows(M.evaluate_member(fn, M.MEMBER_P, fg))
    fresid = [fcp[a][m] - fpc[a][m] for a, m, _ in entries(fcp)]
    for fr, rr in zip(fresid, resid):
        if rr == 0:
            require(abs(fr) <= FLOOR, "A2", "zero-residual float entry: %r" % fr)
        else:
            agree(fr, rr, "A2", "residual entry")
    mx = max(resid, key=abs)
    return {"max_abs_residual_rat": str(mx), "residual_nonzero": True}


def check_a3():
    out = {}
    nodal = rat_nodal(F.F_ASYM)
    lhs = M.closed_action(M.permute_nodal(nodal), M.RAT_GRID)
    rhs = M.permute_rows(M.closed_action(nodal, M.RAT_GRID))
    for a, m, v in entries(lhs):
        require(v == rhs[a][m], "A3", "rational entry (%d,%d)" % (a, m))
    out["rational_exact"] = True
    for name, grid in (("mirror", M.float_mirror_grid()), ("gl8", M.gl8_grid())):
        fn = grid_nodal(F.F_ASYM, grid)
        flhs = M.closed_action(M.permute_nodal(fn), grid)
        frhs = M.permute_rows(M.closed_action(fn, grid))
        for a, m, v in entries(flhs):
            require(v == frhs[a][m], "A3",
                    "%s bitwise entry (%d,%d): %r vs %r" % (name, a, m, v, frhs[a][m]))
        out[name + "_bitwise"] = True
    return out


def check_a4():
    cat = {M.MEMBER_P: Fraction(1)}
    r1 = M.reynolds(cat)
    require(r1 == {M.MEMBER_P: Fraction(1, 2), M.MEMBER_M: Fraction(1, 2)},
            "A4", "Reynolds of a single member must be the half-half orbit")
    require(M.reynolds(r1) == r1, "A4", "Reynolds idempotence")
    require(2 * M.QUOTIENT * M.QUOTIENT == M.QUOTIENT, "A4", "2q^2 == q")
    for q in (Fraction(1), Fraction(1, 4)):
        bad = M.reynolds(cat, q)
        require(M.reynolds(bad, q) != bad, "A4", "q=%s must fail idempotence" % q)
    return {"quotient": str(M.QUOTIENT), "idempotent": True}


def check_a5():
    nodal = rat_nodal(F.F_ASYM)
    s_asym = M.s_anti(M.closed_action(nodal, M.RAT_GRID))
    require(s_asym != 0, "A5", "asymmetric antisymmetric response must be nonzero")
    s_sym = M.s_anti(M.closed_action(rat_nodal(F.F_SYM), M.RAT_GRID))
    require(s_sym == 0, "A5", "symmetric fixture response must vanish exactly")
    fg = M.float_mirror_grid()
    sf = M.s_anti(M.closed_action(grid_nodal(F.F_ASYM, fg), fg))
    agree(sf, s_asym, "A5", "mirror s_anti")
    g8 = M.gl8_grid()
    s8_sym = M.s_anti(M.closed_action(grid_nodal(F.F_SYM, g8), g8))
    require(s8_sym == 0.0, "A5", "gl8 symmetric response must be bitwise zero")
    s8 = M.s_anti(M.closed_action(grid_nodal(F.F_ASYM, g8), g8))
    require(s8 != 0.0, "A5", "gl8 asymmetric response must be nonzero")
    return {"s_anti_rat": str(s_asym), "s_anti_mirror": repr(sf), "s_anti_gl8": repr(s8)}


def check_a6():
    nodal = rat_nodal(F.F_ASYM)
    trace = []
    c = M.closed_action(nodal, M.RAT_GRID, trace=trace)
    nsum = sum(M.number(c, a) for a in range(4))
    esum = sum(M.energy(c, a) for a in range(4))
    require(nsum == 0, "A6", "exact number null: %s" % nsum)
    require(esum == 0, "A6", "exact energy null: %s" % esum)
    lane0 = M.lanes_for()[0]
    strict = False
    for ev in trace:
        f1, f2, f3, f4 = ev["f"]
        gain = (1 - f1) * (1 - f2) * f3 * f4
        loss = f1 * f2 * (1 - f3) * (1 - f4)
        cert = lane0 * ev["drive"] * (gain - loss)
        require(cert >= 0, "A6", "entropy certificate violated at %s" % (ev["event"],))
        if cert > 0:
            strict = True
    require(strict, "A6", "entropy production must be strictly positive somewhere")
    g8 = M.gl8_grid()
    c8 = M.closed_action(grid_nodal(F.F_ASYM, g8), g8)
    n8 = (M.number(c8, 0) + M.number(c8, 2)) + (M.number(c8, 1) + M.number(c8, 3))
    require(n8 == 0.0, "A6", "float pair-summed number must be bitwise zero")
    e8 = sum(M.energy(c8, a) for a in range(4))
    scale = sum(abs(M.energy(c8, a)) for a in range(4)) + FLOOR
    require(abs(e8) <= AGREE * scale, "A6", "float energy null bound: %r" % e8)
    return {"events_traced": len(trace), "number_null": "exact", "energy_null": "exact",
            "entropy_strict": True, "gl8_number": repr(n8), "gl8_energy": repr(e8)}


def check_a7():
    dn = F.dual_nodal_asym(M.RAT_NODES)
    closed = M.closed_action(dn, M.RAT_GRID)
    cp = M.evaluate_member(dn, M.MEMBER_P, M.RAT_GRID)
    cm = M.evaluate_member(dn, M.MEMBER_M, M.RAT_GRID)
    q = M.QUOTIENT
    diff_seen = False
    minus_seen = False
    for a, m, v in entries(closed):
        both = q * (cp[a][m].b + cm[a][m].b)
        require(v.b == both, "A7", "JVP two-contribution identity (%d,%d)" % (a, m))
        if v.b != q * cp[a][m].b:
            diff_seen = True
        if cm[a][m].b != 0:
            minus_seen = True
    require(minus_seen, "A7", "M_MINUS JVP contribution must be nonzero")
    require(diff_seen, "A7", "dropped-member JVP must differ")
    sample = closed[2][0].b
    return {"jvp_two_contribution": True, "jvp_sample_tau0": str(sample)}


def check_a9():
    for name, grid, nodal in (
        ("rat", M.RAT_GRID, rat_nodal(F.F_ASYM)),
        ("mirror", M.float_mirror_grid(), None),
        ("gl8", M.gl8_grid(), None),
    ):
        if nodal is None:
            nodal = grid_nodal(F.F_ASYM, grid)
        a = M.closed_action(nodal, grid, form="gml_pppp_mm")
        b = M.closed_action(nodal, grid, form="lmg_mmpp")
        for aa, mm, v in entries(a):
            require(v == b[aa][mm], "A9",
                    "%s equivalent-form entry (%d,%d)" % (name, aa, mm))
    return {"equivalent_form_bitwise": True}


def check_a10_scramble():
    nodal = rat_nodal(F.F_ASYM)
    base = M.closed_action(nodal, M.RAT_GRID)
    scr = M.closed_action(nodal, M.RAT_GRID, scramble=True)
    for a, m, v in entries(base):
        require(v == scr[a][m], "A10-scramble", "rational entry (%d,%d)" % (a, m))
    g8 = M.gl8_grid()
    fn = grid_nodal(F.F_ASYM, g8)
    fbase = M.closed_action(fn, g8)
    fscr = M.closed_action(fn, g8, scramble=True)
    for a, m, v in entries(fbase):
        require(v == fscr[a][m], "A10-scramble",
                "gl8 bitwise entry (%d,%d): %r vs %r" % (a, m, v, fscr[a][m]))
    return {"scramble_bitwise": True}


def check_a10_sweep():
    g8 = M.gl8_grid()
    cases = 4096
    for n in range(cases):
        coeffs = F.sweep_coeffs(n)
        fn = grid_nodal(coeffs, g8)
        c = M.closed_action(fn, g8)
        pc = M.permute_rows(c)
        cpf = M.closed_action(M.permute_nodal(fn), g8)
        for a, m, v in entries(cpf):
            require(v == pc[a][m], "A10-sweep",
                    "case %d bitwise entry (%d,%d)" % (n, a, m))
        nsum = (M.number(c, 0) + M.number(c, 2)) + (M.number(c, 1) + M.number(c, 3))
        require(nsum == 0.0, "A10-sweep", "case %d number null: %r" % (n, nsum))
    return {"sweep_cases": cases, "sweep_pass": True}


CHECKS = (
    ("ANCHOR", check_anchor),
    ("A1", check_a1),
    ("A2", check_a2),
    ("A3", check_a3),
    ("A4", check_a4),
    ("A5", check_a5),
    ("A6", check_a6),
    ("A7", check_a7),
    ("A9", check_a9),
    ("A10-scramble", check_a10_scramble),
    ("A10-sweep", check_a10_sweep),
)


def main(argv):
    args = list(argv[1:])
    single = None
    out_path = None
    while args:
        arg = args.pop(0)
        if arg == "--check":
            single = args.pop(0)
        elif arg == "--out":
            out_path = args.pop(0)
        elif arg == "--full":
            pass
        else:
            print("ERROR unknown argument %r" % arg)
            return 20
    table = dict(CHECKS)
    if single is not None:
        if single not in table:
            print("ERROR unknown check %r" % single)
            return 20
        todo = ((single, table[single]),)
    else:
        todo = CHECKS
    results = {"contract_id": "EC-OWNERA-R6-ORBIT-CHART-2026-07-27", "checks": {}}
    for check_id, fn in todo:
        try:
            detail = fn()
        except CheckFailure as exc:
            print("FAIL %s" % exc)
            return 10
        except Exception as exc:  # mechanical error, not a scientific verdict
            print("ERROR %s: %r" % (check_id, exc))
            return 20
        results["checks"][check_id] = detail
        print("PASS %s" % check_id)
    if out_path is not None:
        with open(out_path, "w") as fh:
            json.dump(results, fh, sort_keys=True, indent=1)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
