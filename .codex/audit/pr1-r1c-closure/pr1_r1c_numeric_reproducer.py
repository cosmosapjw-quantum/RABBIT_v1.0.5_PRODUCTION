#!/usr/bin/env python3
"""Baseline audit model and independent exact-real reference for RABBIT PR #1.

Evidence classes are deliberately separated:

* ``solve_audit_model`` mirrors the audited binary64 control flow.  It is an
  audit-authored model, not an implementation oracle and not final-SHA proof.
* ``exact_pair_root`` solves the exact-real affine-state residual represented by
  the frozen binary64 inputs using Decimal arithmetic.  This is the independent
  numerical oracle for the frozen roots and golden bits.

Use this program only before implementation to check that the released audit
package is internally consistent.  Final verification must run Rust regressions
at the exact final commit instead of treating this frozen model as evidence that
Rust changed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, getcontext
import json
import math
import struct
import sys
from typing import Any

getcontext().prec = 120
EPS = sys.float_info.epsilon
MIN_SUB = math.ulp(0.0)
TOL = 128.0 * EPS
KNOWN_BAD_BITS = 0xBCCE_FF08_07BF_A264
PRIMARY_GOLDEN_BITS = 0xBCCE_FF07_E8A3_8D5C


def float_bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0]


@dataclass(frozen=True)
class Edge:
    pair: bool
    m1: float
    m2: float
    gain: float
    loss: float

    def factors(self, f1: float, f2: float) -> tuple[tuple[float, float], tuple[float, float]]:
        if self.pair:
            return ((1.0 - f1, 1.0 - f2), (f1, f2))
        return ((1.0 - f1, f2), (f1, 1.0 - f2))

    @staticmethod
    def product(coefficient: float, factors: tuple[float, float]) -> tuple[str, float, float]:
        if coefficient == 0.0 or factors[0] == 0.0 or factors[1] == 0.0:
            return ("zero", 0.0, 0.0)
        first = coefficient * factors[0]
        if not math.isfinite(first) or first == 0.0:
            return ("unresolved", math.nan, math.inf)
        value = first * factors[1]
        if not math.isfinite(value) or value == 0.0:
            return ("unresolved", math.nan, math.inf)
        return ("value", value, 8.0 * EPS * abs(value) + 4.0 * MIN_SUB)

    def flux_model(self, f1: float, f2: float) -> tuple[float, float, float, str]:
        gain_factors, loss_factors = self.factors(f1, f2)
        gain_kind, gain_value, gain_error = self.product(self.gain, gain_factors)
        loss_kind, loss_value, loss_error = self.product(self.loss, loss_factors)
        if gain_kind == loss_kind == "zero":
            return (0.0, 0.0, 0.0, "exact-zero")
        if gain_kind == "unresolved" or loss_kind == "unresolved":
            return (math.nan, 0.0, math.inf, "unresolved")
        if gain_kind == "zero":
            gain_value = gain_error = 0.0
        if loss_kind == "zero":
            loss_value = loss_error = 0.0
        net = gain_value - loss_value
        error = (
            gain_error
            + loss_error
            + 2.0 * EPS * (abs(gain_value) + abs(loss_value))
            + 2.0 * MIN_SUB
        )
        traffic = abs(gain_value) + abs(loss_value) + gain_error + loss_error
        resolution = "resolved" if abs(net) > error else "unresolved"
        return (net, traffic, error, resolution)

    def occupations_model(self, initial: tuple[float, float], extent: float) -> tuple[float, float]:
        f1 = initial[0] + extent / self.m1
        f2 = initial[1] + (extent / self.m2 if self.pair else -extent / self.m2)
        if not (0.0 <= f1 <= 1.0 and 0.0 <= f2 <= 1.0):
            raise ValueError("outside Pauli box")
        return (f1, f2)

    def derivative_model(self, f1: float, f2: float) -> float:
        if self.pair:
            derivative_first = -self.gain * (1.0 - f2) - self.loss * f2
            derivative_second = -self.gain * (1.0 - f1) - self.loss * f1
            return derivative_first / self.m1 + derivative_second / self.m2
        derivative_first = -self.gain * f2 - self.loss * (1.0 - f2)
        derivative_second = self.gain * (1.0 - f1) + self.loss * f1
        return derivative_first / self.m1 - derivative_second / self.m2

    def bounds(self, initial: tuple[float, float]) -> tuple[float, float]:
        f1, f2 = initial
        first_down = self.m1 * f1
        first_up = self.m1 * (1.0 - f1)
        if self.pair:
            return (-min(first_down, self.m2 * f2), min(first_up, self.m2 * (1.0 - f2)))
        return (-min(first_down, self.m2 * (1.0 - f2)), min(first_up, self.m2 * f2))

    def residual_model(
        self, h: float, initial: tuple[float, float], extent: float
    ) -> tuple[float, float, float, float, float, float, str]:
        f1, f2 = self.occupations_model(initial, extent)
        net, traffic, error, resolution = self.flux_model(f1, f2)
        value = extent - h * net
        derivative = 1.0 - h * self.derivative_model(f1, f2)
        scale = max(abs(extent), h * traffic, MIN_SUB)
        root_error = abs(value) + h * error
        occupation_error = root_error / min(self.m1, self.m2)
        return (value, derivative, scale, root_error, occupation_error, error, resolution)

    def solve_audit_model(
        self, h: float, initial: tuple[float, float], cap: int = 96
    ) -> dict[str, Any]:
        _, _, _, initial_resolution = self.flux_model(*initial)
        if initial_resolution == "exact-zero":
            return {"outcome": "EXACT_STATIONARY"}
        if initial_resolution == "unresolved":
            return {"outcome": "UNRESOLVED_FLUX"}
        lower_capacity, upper_capacity = self.bounds(initial)
        lower = math.nextafter(lower_capacity, math.inf) if lower_capacity < 0.0 else lower_capacity
        upper = math.nextafter(upper_capacity, -math.inf) if upper_capacity > 0.0 else upper_capacity
        extent = 0.0
        previous_state: tuple[str, str, str] | None = None
        repeated = 0
        for iteration in range(1, cap + 1):
            value, derivative, scale, root_error, occupation_error, error, resolution = self.residual_model(
                h, initial, extent
            )
            if resolution == "resolved" and abs(value) <= TOL * scale + MIN_SUB and occupation_error <= TOL:
                return {
                    "outcome": "SOLVED_CURRENT",
                    "extent": extent,
                    "iterations": iteration,
                    "root_error": root_error,
                    "occupation_error": occupation_error,
                    "repeated_extent_iterations": repeated,
                }
            current_is_lower = value + h * error <= 0.0
            current_is_upper = value - h * error >= 0.0
            if current_is_lower:
                lower = extent
            elif current_is_upper:
                upper = extent
            midpoint = 0.5 * (lower + upper)
            mid_value, _, mid_scale, mid_root, mid_occ, _, mid_resolution = self.residual_model(
                h, initial, midpoint
            )
            if (
                abs(upper - lower) / min(self.m1, self.m2) <= TOL
                and mid_resolution == "resolved"
                and abs(mid_value) <= TOL * mid_scale + MIN_SUB
                and mid_occ <= TOL
            ):
                return {
                    "outcome": "SOLVED_MIDPOINT",
                    "extent": midpoint,
                    "iterations": iteration,
                    "root_error": mid_root,
                    "occupation_error": mid_occ,
                    "repeated_extent_iterations": repeated,
                }
            newton = extent - value / derivative
            if not current_is_lower and not current_is_upper:
                next_extent = midpoint
            elif lower < newton < upper and math.isfinite(newton):
                next_extent = newton
            else:
                next_extent = midpoint
            state = (next_extent.hex(), lower.hex(), upper.hex())
            if state == previous_state:
                repeated += 1
            previous_state = state
            extent = next_extent
        outcome = (
            "UNCERTAIN_PHYSICAL_BRACKET"
            if abs(upper - lower) / min(self.m1, self.m2) <= TOL
            else "ITERATION_LIMIT"
        )
        return {
            "outcome": outcome,
            "extent": extent,
            "iterations": cap,
            "repeated_extent_iterations": repeated,
        }


def exact_pair_root(edge: Edge, h: float, initial: tuple[float, float]) -> Decimal:
    if not edge.pair:
        raise ValueError("exact_pair_root is defined for PairSource fixtures")
    gain = Decimal.from_float(edge.gain)
    loss = Decimal.from_float(edge.loss)
    m1 = Decimal.from_float(edge.m1)
    m2 = Decimal.from_float(edge.m2)
    f10 = Decimal.from_float(initial[0])
    f20 = Decimal.from_float(initial[1])
    hd = Decimal.from_float(h)

    def residual(extent: Decimal) -> Decimal:
        f1 = f10 + extent / m1
        f2 = f20 + extent / m2
        return extent - hd * (gain * (1 - f1) * (1 - f2) - loss * f1 * f2)

    lower_float, upper_float = edge.bounds(initial)
    lower = Decimal.from_float(lower_float)
    upper = Decimal.from_float(upper_float)
    if not residual(lower) <= 0 <= residual(upper):
        raise AssertionError("physical root is not bracketed")
    for _ in range(500):
        midpoint = (lower + upper) / 2
        if residual(midpoint) <= 0:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


def golden_entry(edge: Edge, h: float, initial: tuple[float, float]) -> dict[str, Any]:
    root = exact_pair_root(edge, h, initial)
    nearest = float(root)
    return {
        "h_hex": h.hex(),
        "root_decimal": str(root),
        "nearest_binary64_hex": nearest.hex(),
        "nearest_binary64_bits": f"0x{float_bits(nearest):016x}",
    }


def baseline_report() -> dict[str, Any]:
    primary_edge = Edge(True, 2.0**8, 2.0**-36, 2.0**26, 2.0**-14)
    primary_h = 2.0**-36
    primary_initial = (1.0 - 2.0**-40, 1.0 - 2.0**-6)
    primary_model = primary_edge.solve_audit_model(primary_h, primary_initial)
    primary_root = exact_pair_root(primary_edge, primary_h, primary_initial)
    if primary_model["outcome"] != "SOLVED_CURRENT":
        raise AssertionError(primary_model)
    if float_bits(primary_model["extent"]) != KNOWN_BAD_BITS:
        raise AssertionError("audit model no longer produces the released known-bad bits")
    if float_bits(float(primary_root)) != PRIMARY_GOLDEN_BITS:
        raise AssertionError("independent exact-real reference no longer produces the released golden bits")
    actual_extent_error = abs(Decimal.from_float(primary_model["extent"]) - primary_root)
    actual_occupation_error = actual_extent_error / Decimal.from_float(min(primary_edge.m1, primary_edge.m2))
    ratio = actual_occupation_error / Decimal.from_float(TOL)
    if ratio <= 100:
        raise AssertionError(ratio)

    stagnation_edge = Edge(False, 2.0**-30, 2.0**-30, 2.0**-20, 2.0**-20)
    stagnation = stagnation_edge.solve_audit_model(1.0, (1.0 / 8.0, 1.0 / 4.0))
    if stagnation["outcome"] != "UNCERTAIN_PHYSICAL_BRACKET":
        raise AssertionError(stagnation)
    if stagnation["repeated_extent_iterations"] < 4:
        raise AssertionError(stagnation)

    equilibrium = Edge(False, 1.0, 1.0, 1.0, 1.0).solve_audit_model(0.25, (0.5, 0.5))
    if equilibrium["outcome"] != "UNRESOLVED_FLUX":
        raise AssertionError(equilibrium)

    ladder_edge = Edge(True, 2.0, 5.0, 13.0, 17.0)
    ladder_initial = (0.23, 0.79)
    ladder = [golden_entry(ladder_edge, 2.0**-exponent, ladder_initial) for exponent in (8, 14, 20, 30)]

    return {
        "schema": "rabbit-pr1-r1c-baseline-audit-model/v2",
        "audited_head": "50d3bc5b8093bc33e9311f94505c5ee0711ce51b",
        "evidence_classification": {
            "binary64_control_flow": "audit-authored mirror; not implementation oracle",
            "decimal_roots": "independent exact-real reference for frozen fixtures",
            "final_sha_use": "forbidden as proof that Rust was corrected",
        },
        "primary_p0": {
            "verdict": "P0_FALSE_SOLVED",
            "model_outcome": primary_model["outcome"],
            "iterations": primary_model["iterations"],
            "returned_extent_hex": primary_model["extent"].hex(),
            "returned_extent_bits": f"0x{float_bits(primary_model['extent']):016x}",
            "exact_real_root": str(primary_root),
            "nearest_root_bits": f"0x{float_bits(float(primary_root)):016x}",
            "reported_occupation_error": primary_model["occupation_error"],
            "actual_occupation_error": str(actual_occupation_error),
            "actual_error_over_128eps": str(ratio),
        },
        "fallback": dict(stagnation, verdict="P1_STAGNATED_INTERVAL"),
        "equilibrium": dict(equilibrium, verdict="P1_EQUILIBRIUM_ABORT"),
        "local_pair_edge_golden_ladder": ladder,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-audit-model",
        action="store_true",
        help="emit the frozen audit mirror and independent exact-real golden data",
    )
    args = parser.parse_args()
    if not args.baseline_audit_model:
        parser.error("this program is baseline-only; pass --baseline-audit-model")
    print(json.dumps(baseline_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
