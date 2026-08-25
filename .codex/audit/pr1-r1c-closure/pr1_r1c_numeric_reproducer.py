#!/usr/bin/env python3
"""Independent focused numerical falsifier for RABBIT PR #1 @ 50d3bc5.

The binary64 section mirrors the current local Pauli-edge control flow for the
three exact fixtures below. Decimal arithmetic independently solves the exact-
real affine-state PairSource residual. Standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
import json
import math
import sys

getcontext().prec = 110
EPS = sys.float_info.epsilon
MIN_SUB = math.ulp(0.0)
TOL = 128.0 * EPS


@dataclass(frozen=True)
class Edge:
    pair: bool
    m1: float
    m2: float
    gain: float
    loss: float

    def factors(self, f1: float, f2: float):
        if self.pair:
            return ((1.0-f1, 1.0-f2), (f1, f2))
        return ((1.0-f1, f2), (f1, 1.0-f2))

    @staticmethod
    def product(c: float, fs: tuple[float, float]):
        if c == 0.0 or fs[0] == 0.0 or fs[1] == 0.0:
            return ("zero", 0.0, 0.0)
        x = c * fs[0]
        if not math.isfinite(x) or x == 0.0:
            return ("unresolved", math.nan, math.inf)
        y = x * fs[1]
        if not math.isfinite(y) or y == 0.0:
            return ("unresolved", math.nan, math.inf)
        return ("value", y, 8.0*EPS*abs(y) + 4.0*MIN_SUB)

    def flux(self, f1: float, f2: float):
        gf, lf = self.factors(f1, f2)
        gk, gv, ge = self.product(self.gain, gf)
        lk, lv, le = self.product(self.loss, lf)
        if gk == lk == "zero":
            return (0.0, 0.0, 0.0, "exact-zero")
        if gk == "unresolved" or lk == "unresolved":
            return (math.nan, 0.0, math.inf, "unresolved")
        if gk == "zero": gv = ge = 0.0
        if lk == "zero": lv = le = 0.0
        net = gv - lv
        err = ge + le + 2.0*EPS*(abs(gv)+abs(lv)) + 2.0*MIN_SUB
        traffic = abs(gv)+abs(lv)+ge+le
        return (net, traffic, err, "resolved" if abs(net) > err else "unresolved")

    def occupations(self, initial, xi):
        f1 = initial[0] + xi/self.m1
        f2 = initial[1] + (xi/self.m2 if self.pair else -xi/self.m2)
        if not (0.0 <= f1 <= 1.0 and 0.0 <= f2 <= 1.0):
            raise ValueError("outside box")
        return f1, f2

    def derivative(self, f1, f2):
        if self.pair:
            d1 = -self.gain*(1.0-f2) - self.loss*f2
            d2 = -self.gain*(1.0-f1) - self.loss*f1
            return d1/self.m1 + d2/self.m2
        d1 = -self.gain*f2 - self.loss*(1.0-f2)
        d2 = self.gain*(1.0-f1) + self.loss*f1
        return d1/self.m1 - d2/self.m2

    def bounds(self, initial):
        f1, f2 = initial
        down = self.m1*f1
        up = self.m1*(1.0-f1)
        if self.pair:
            return (-min(down, self.m2*f2), min(up, self.m2*(1.0-f2)))
        return (-min(down, self.m2*(1.0-f2)), min(up, self.m2*f2))

    def residual(self, h, initial, xi):
        f1, f2 = self.occupations(initial, xi)
        net, traffic, err, resolution = self.flux(f1, f2)
        value = xi - h*net
        derivative = 1.0 - h*self.derivative(f1, f2)
        scale = max(abs(xi), h*traffic, MIN_SUB)
        root_error = abs(value) + h*err
        occ_error = root_error/min(self.m1, self.m2)
        return value, derivative, scale, root_error, occ_error, err, resolution

    def solve_current_head(self, h, initial, cap=96):
        net, _, _, resolution = self.flux(*initial)
        if resolution == "exact-zero": return {"outcome":"EXACT_STATIONARY"}
        if resolution == "unresolved": return {"outcome":"UNRESOLVED_FLUX"}
        lo0, hi0 = self.bounds(initial)
        lo = math.nextafter(lo0, math.inf) if lo0 < 0.0 else lo0
        hi = math.nextafter(hi0, -math.inf) if hi0 > 0.0 else hi0
        xi = 0.0
        previous = None
        repeated = 0
        for iteration in range(1, cap+1):
            value, deriv, scale, rooterr, occerr, err, res = self.residual(h, initial, xi)
            if res == "resolved" and abs(value) <= TOL*scale + MIN_SUB and occerr <= TOL:
                return {"outcome":"SOLVED_CURRENT", "extent":xi, "iterations":iteration,
                        "root_error":rooterr, "occupation_error":occerr,
                        "repeated_extent_iterations":repeated}
            is_lo = value + h*err <= 0.0
            is_hi = value - h*err >= 0.0
            if is_lo: lo = xi
            elif is_hi: hi = xi
            mid = 0.5*(lo+hi)
            mv, _, ms, mr, mo, _, mres = self.residual(h, initial, mid)
            if abs(hi-lo)/min(self.m1,self.m2) <= TOL and mres == "resolved" \
               and abs(mv) <= TOL*ms + MIN_SUB and mo <= TOL:
                return {"outcome":"SOLVED_MIDPOINT", "extent":mid, "iterations":iteration,
                        "root_error":mr, "occupation_error":mo,
                        "repeated_extent_iterations":repeated}
            newton = xi - value/deriv
            nxt = mid if (not is_lo and not is_hi) else (newton if lo < newton < hi else mid)
            if previous is not None and nxt.hex() == previous: repeated += 1
            previous = nxt.hex()
            xi = nxt
        outcome = "UNCERTAIN_PHYSICAL_BRACKET" if abs(hi-lo)/min(self.m1,self.m2) <= TOL else "ITERATION_LIMIT"
        return {"outcome":outcome, "extent":xi, "iterations":cap,
                "repeated_extent_iterations":repeated}


def exact_pair_root(edge: Edge, h: float, initial):
    A, B = Decimal.from_float(edge.gain), Decimal.from_float(edge.loss)
    m1, m2 = Decimal.from_float(edge.m1), Decimal.from_float(edge.m2)
    f10, f20, hd = Decimal.from_float(initial[0]), Decimal.from_float(initial[1]), Decimal.from_float(h)
    def r(x):
        f1, f2 = f10+x/m1, f20+x/m2
        return x - hd*(A*(1-f1)*(1-f2) - B*f1*f2)
    lo0, hi0 = edge.bounds(initial)
    lo, hi = Decimal.from_float(lo0), Decimal.from_float(hi0)
    assert r(lo) <= 0 <= r(hi)
    for _ in range(450):
        mid = (lo+hi)/2
        if r(mid) <= 0: lo = mid
        else: hi = mid
    return (lo+hi)/2


def main():
    p0_edge = Edge(True, 2.0**8, 2.0**-36, 2.0**26, 2.0**-14)
    p0_h = 2.0**-36
    p0_initial = (1.0-2.0**-40, 1.0-2.0**-6)
    p0 = p0_edge.solve_current_head(p0_h, p0_initial)
    root = exact_pair_root(p0_edge, p0_h, p0_initial)
    assert p0["outcome"] == "SOLVED_CURRENT", p0
    actual_extent_error = abs(Decimal.from_float(p0["extent"]) - root)
    actual_occ_error = actual_extent_error/Decimal.from_float(min(p0_edge.m1,p0_edge.m2))
    ratio = actual_occ_error/Decimal.from_float(TOL)
    assert ratio > 100, ratio

    spin_edge = Edge(False, 2.0**-30, 2.0**-30, 2.0**-20, 2.0**-20)
    spin = spin_edge.solve_current_head(1.0, (1.0/8.0,1.0/4.0))
    assert spin["outcome"] == "UNCERTAIN_PHYSICAL_BRACKET", spin
    assert spin["repeated_extent_iterations"] >= 4, spin

    eq = Edge(False,1.0,1.0,1.0,1.0).solve_current_head(0.25,(0.5,0.5))
    assert eq["outcome"] == "UNRESOLVED_FLUX", eq

    out = {
      "schema":"rabbit-pr1-r1c-independent-numeric-falsifier/v1",
      "audited_head":"50d3bc5b8093bc33e9311f94505c5ee0711ce51b",
      "p0_false_solved":{
        "verdict":"P0_FALSE_SOLVED",
        "iterations":p0["iterations"],
        "returned_extent":p0["extent"],
        "returned_extent_hex":p0["extent"].hex(),
        "exact_real_root_decimal":str(root),
        "nearest_binary64_root_hex":float(root).hex(),
        "reported_root_error_abs":p0["root_error"],
        "reported_occupation_error_abs":p0["occupation_error"],
        "actual_occupation_error":str(actual_occ_error),
        "actual_error_over_128eps":str(ratio)
      },
      "fallback_stagnation":dict(spin, verdict="P1_STAGNATED_INTERVAL",
                                  extent_hex=spin["extent"].hex()),
      "equilibrium_hard_abort":dict(eq, verdict="P1_EQUILIBRIUM_ABORT")
    }
    print(json.dumps(out, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
