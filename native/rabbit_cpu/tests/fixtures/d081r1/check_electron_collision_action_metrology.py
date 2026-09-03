#!/usr/bin/env python3
"""Check D-081R1D3 electron metrology semantic identity across CI hosts.

The committed JSON remains the canonical byte-level fixture.  A freshly
recomputed candidate may differ in the last floating-point bits because NumPy
reductions dispatch to host-dependent SIMD kernels.  This checker preserves
exact structural identity and admits only a fixed, cancellation-aware
binary64 envelope that is much tighter than the Rust parity gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SEMANTIC_BUDGET_ULPS = 4096.0


def bits_float(bits: str) -> float:
    digits = bits.removeprefix("0x")
    return float(np.asarray([int(digits, 16)], dtype=np.uint64).view(np.float64)[0])


def decode_array(encoded: dict[str, object]) -> np.ndarray:
    shape = tuple(int(value) for value in encoded["shape"])
    raw = np.asarray(
        [int(str(value).removeprefix("0x"), 16) for value in encoded["bits"]],
        dtype=np.uint64,
    )
    return np.ascontiguousarray(raw.view(np.float64).reshape(shape))


def maximum_encoded_array(encoded: dict[str, object]) -> float:
    return float(np.max(np.abs(decode_array(encoded)), initial=0.0))


class SemanticAudit:
    def __init__(self) -> None:
        self.worst_ratio = 0.0
        self.worst_label = "none"
        self.worst_difference = 0.0
        self.worst_allowed = 0.0

    def _record(self, label: str, difference: float, allowed: float) -> None:
        ratio = difference / max(allowed, np.finfo(np.float64).tiny)
        if ratio > self.worst_ratio:
            self.worst_ratio = ratio
            self.worst_label = label
            self.worst_difference = difference
            self.worst_allowed = allowed
        if ratio > 1.0:
            raise AssertionError(
                f"semantic mismatch at {label}: difference={difference:.17e}, "
                f"allowed={allowed:.17e}, ratio={ratio:.17e}"
            )

    def scalar(self, actual: float, expected: float, scale: float, label: str) -> None:
        if not np.isfinite(actual) or not np.isfinite(expected):
            raise AssertionError(f"nonfinite scalar at {label}")
        allowed = (
            SEMANTIC_BUDGET_ULPS
            * np.finfo(np.float64).eps
            * max(abs(scale), np.finfo(np.float64).tiny)
        )
        self._record(label, abs(actual - expected), allowed)

    def encoded_scalar(
        self,
        actual: str,
        expected: str,
        scale: float,
        label: str,
    ) -> None:
        self.scalar(bits_float(actual), bits_float(expected), scale, label)

    def array(
        self,
        actual: dict[str, object],
        expected: dict[str, object],
        scale: float,
        label: str,
    ) -> None:
        if actual["shape"] != expected["shape"]:
            raise AssertionError(f"shape mismatch at {label}")
        left = decode_array(actual)
        right = decode_array(expected)
        difference = float(np.max(np.abs(left - right), initial=0.0))
        allowed = (
            SEMANTIC_BUDGET_ULPS
            * np.finfo(np.float64).eps
            * max(abs(scale), np.finfo(np.float64).tiny)
        )
        self._record(label, difference, allowed)


def exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"exact mismatch at {label}: {actual!r} != {expected!r}")


def named_cases(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    cases = payload["cases"]
    if not isinstance(cases, list):
        raise AssertionError("cases must be a list")
    result = {str(case["name"]): case for case in cases}
    if len(result) != len(cases):
        raise AssertionError("duplicate case name")
    return result


def audit(candidate: dict[str, object], canonical: dict[str, object]) -> SemanticAudit:
    exact_top = (
        "schema",
        "private_comparator_git_blob",
        "full_collision_fixture_git_blob",
        "order",
        "y_max_bits",
        "electron_mass_bits",
        "elastic_event_count",
        "pair_event_count",
        "electron_event_count",
        "component_boundary",
        "category_native_reconstruction_budget_ulps",
    )
    for key in exact_top:
        exact(candidate[key], canonical[key], key)

    canonical_cases = named_cases(canonical)
    candidate_cases = named_cases(candidate)
    exact(set(candidate_cases), set(canonical_cases), "case names")

    category_scale = max(
        maximum_encoded_array(encoded)
        for case in canonical_cases.values()
        for encoded in case["category_arrays"].values()
    )
    bath_scale = max(
        abs(bits_float(str(value)))
        for case in canonical_cases.values()
        for value in case["bath_energy_by_family"].values()
    )
    moment_scale = max(
        abs(bits_float(str(value)))
        for case in canonical_cases.values()
        for value in case["moments"].values()
    )
    diagnostic_scales: dict[str, float] = {}
    diagnostic_keys = set().union(
        *(set(case["diagnostics"]) for case in canonical_cases.values())
    )
    for key in diagnostic_keys:
        scale = max(
            abs(bits_float(str(case["diagnostics"][key])))
            for case in canonical_cases.values()
        )
        if "residual" in key:
            scale = max(scale, 1.0)
        elif "category_native_reconstruction" in key:
            scale = max(scale, category_scale)
        diagnostic_scales[key] = scale

    correction_scale = max(
        abs(bits_float(str(case["largest_matrix_roundoff_correction_bits"])))
        for case in canonical_cases.values()
    )

    result = SemanticAudit()
    for name in sorted(canonical_cases):
        expected = canonical_cases[name]
        actual = candidate_cases[name]
        for key in (
            "name",
            "temperature_cm_bits",
            "temperature_gamma_bits",
            "electron_mass_bits",
            "family_order",
            "whole_reaction_domain_rejections",
            "elastic_domain_rejections",
            "pair_domain_rejections",
            "combined_domain_rejections",
            "self_domain_rejections_by_difference",
            "matrix_roundoff_corrections",
        ):
            exact(actual[key], expected[key], f"{name}.{key}")

        result.encoded_scalar(
            str(actual["largest_matrix_roundoff_correction_bits"]),
            str(expected["largest_matrix_roundoff_correction_bits"]),
            correction_scale,
            f"{name}.largest_matrix_roundoff_correction",
        )

        actual_arrays = actual["category_arrays"]
        expected_arrays = expected["category_arrays"]
        exact(set(actual_arrays), set(expected_arrays), f"{name}.category array keys")
        for key in sorted(expected_arrays):
            result.array(
                actual_arrays[key],
                expected_arrays[key],
                category_scale,
                f"{name}.category_arrays.{key}",
            )

        actual_bath = actual["bath_energy_by_family"]
        expected_bath = expected["bath_energy_by_family"]
        exact(set(actual_bath), set(expected_bath), f"{name}.bath family keys")
        for key in sorted(expected_bath):
            result.encoded_scalar(
                str(actual_bath[key]),
                str(expected_bath[key]),
                bath_scale,
                f"{name}.bath_energy_by_family.{key}",
            )

        actual_diagnostics = actual["diagnostics"]
        expected_diagnostics = expected["diagnostics"]
        exact(
            set(actual_diagnostics),
            set(expected_diagnostics),
            f"{name}.diagnostic keys",
        )
        for key in sorted(expected_diagnostics):
            result.encoded_scalar(
                str(actual_diagnostics[key]),
                str(expected_diagnostics[key]),
                diagnostic_scales[key],
                f"{name}.diagnostics.{key}",
            )

        actual_moments = actual["moments"]
        expected_moments = expected["moments"]
        exact(set(actual_moments), set(expected_moments), f"{name}.moment keys")
        for key in sorted(expected_moments):
            result.encoded_scalar(
                str(actual_moments[key]),
                str(expected_moments[key]),
                moment_scale,
                f"{name}.moments.{key}",
            )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("canonical", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    canonical = json.loads(args.canonical.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = audit(candidate, canonical)
    print(
        json.dumps(
            {
                "schema": "rabbit.d081r1d3.cross_runner_semantic_audit.v1",
                "budget_ulps": SEMANTIC_BUDGET_ULPS,
                "worst_label": result.worst_label,
                "worst_ratio": result.worst_ratio,
                "worst_difference": result.worst_difference,
                "worst_allowed": result.worst_allowed,
                "verdict": "PASS",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
