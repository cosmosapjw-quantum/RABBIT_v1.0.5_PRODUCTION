#!/usr/bin/env python3
"""Generate the D-081R1C scalar-kernel fixture from the frozen Python oracle.

The resulting JSON contains binary64 bit patterns rather than decimal values.
It is intentionally narrow: selected Pauli factors, self/electron weak matrices,
and one event-measure point.  It does not claim full collision-action or RHS
parity.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
SOURCE = REPOSITORY_ROOT / "src/rabbit/decoupling/_independent_noqke.py"
OUTPUT = Path(__file__).with_name("collision_kernel_case.json")
EXPECTED_SOURCE_BLOB = "de44feee0aa484abe26976c7dc34c579643005b5"


def git_blob(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(path.relative_to(REPOSITORY_ROOT))],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()


def load_oracle():
    actual_blob = git_blob(SOURCE)
    if actual_blob != EXPECTED_SOURCE_BLOB:
        raise SystemExit(
            f"private comparator blob mismatch: {actual_blob} != {EXPECTED_SOURCE_BLOB}"
        )
    spec = importlib.util.spec_from_file_location("d081r1_frozen_noqke", SOURCE)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load frozen private comparator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bit_string(value: float) -> str:
    scalar = np.asarray(value, dtype=np.float64)
    if scalar.shape != () or not np.isfinite(scalar):
        raise ValueError(f"expected one finite binary64 scalar, got {value!r}")
    return f"0x{int(scalar.view(np.uint64)):016x}"


def bit_array(values: np.ndarray | list[float]) -> list[str]:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("fixture array contains nonfinite values")
    return [f"0x{int(bits):016x}" for bits in array.view(np.uint64).ravel()]


def main() -> None:
    oracle = load_oracle()
    config = oracle.IndependentCollisionConfig()
    electron_mass = oracle.M_ELECTRON_MEV

    support = np.asarray([True, True, False, True], dtype=bool)
    p2 = np.asarray([0.4, 1.2, 2.0, 3.0], dtype=np.float64)
    e2 = np.hypot(p2, electron_mass)
    zeros = np.zeros_like(p2)
    batch = oracle._KinematicBatch(
        support=support,
        p2=p2,
        e2=e2,
        e3=np.asarray([0.7, 1.8, 2.4, 3.3], dtype=np.float64),
        e4=np.asarray([0.9, 1.4, 2.2, 2.8], dtype=np.float64),
        p3_magnitude=np.asarray([0.7, 1.8, 2.4, 3.3], dtype=np.float64),
        p4_magnitude=np.asarray([0.75, 1.3, 2.0, 2.65], dtype=np.float64),
        phase_space=np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float64),
        quadrature_weight=np.asarray([0.3, 0.5, 0.7, 0.9], dtype=np.float64),
        d12=np.asarray([1.25, 3.5, 2.0, 5.0], dtype=np.float64),
        d13=np.asarray([1.1, 2.0, 1.0, 2.6], dtype=np.float64),
        d14=np.asarray([1.8, 2.7, 1.0, 4.2], dtype=np.float64),
        d23=np.asarray([2.2, 1.4, 1.0, 3.1], dtype=np.float64),
        d24=np.asarray([1.7, 3.3, 1.0, 1.9], dtype=np.float64),
        d34=np.asarray([2.0, 4.0, 1.0, 1.5], dtype=np.float64),
    )

    pauli_inputs = [
        ("detailed_balance_zero", [0.0, 0.0, 0.0, 0.0]),
        ("generic_asymmetric", [-1.2, 0.4, 1.1, -0.3]),
        ("opposite_tails", [20.0, -18.0, 19.0, -17.5]),
        ("dilute_tails", [-40.0, -30.0, -35.0, -34.0]),
        ("near_detailed_balance", [2.0, -1.0, 0.3, 0.700000000001]),
    ]
    relative_tolerance = 2.0e-13
    pauli_cases = []
    for name, logits in pauli_inputs:
        expected = float(oracle._stable_pauli_gain_minus_loss(*logits))
        pauli_cases.append(
            {
                "name": name,
                "logits_bits": bit_array(logits),
                "expected_bits": bit_string(expected),
                "relative_tolerance_bits": bit_string(relative_tolerance),
            }
        )

    self_cases = []
    for kernel, coefficient in (("K_s", 16.0), ("K_t", 64.0)):
        event = oracle.IndependentSelfEvent(
            ("nu_e", "nu_mu", "nu_e", "nu_mu"),
            "fixture",
            kernel,
            coefficient,
        )
        values, corrections, largest = oracle._self_matrix(event, batch, config)
        if corrections != 0 or largest != 0.0:
            raise SystemExit("selected self-matrix fixture unexpectedly required projection")
        self_cases.append(
            {
                "kernel": kernel,
                "coefficient_bits": bit_string(coefficient),
                "expected_value_bits": bit_array(values),
                "corrections": corrections,
                "largest_correction_bits": bit_string(largest),
            }
        )

    electron_specs = [
        ("nu_e", "elastic_minus"),
        ("antinu_e", "elastic_minus"),
        ("nu_mu", "elastic_plus"),
        ("antinu_tau", "elastic_plus"),
        ("nu_e", "pair"),
        ("nu_mu", "pair"),
    ]
    electron_cases = []
    for target, category in electron_specs:
        values, corrections, largest = oracle._electron_matrix(
            target, category, batch, electron_mass, config
        )
        if corrections != 0 or largest != 0.0:
            raise SystemExit(
                f"selected electron fixture {target}/{category} unexpectedly required projection"
            )
        electron_cases.append(
            {
                "target": target,
                "category": category,
                "expected_value_bits": bit_array(values),
                "corrections": corrections,
                "largest_correction_bits": bit_string(largest),
            }
        )

    measure_index = 1
    measure_domain = np.zeros_like(support)
    measure_domain[measure_index] = True
    p1 = 2.75
    outer_weight = 0.125
    measure = oracle._event_measure(batch, p1, outer_weight, measure_domain)
    if measure.shape != (1,):
        raise SystemExit("event-measure fixture did not select exactly one point")

    document = {
        "schema": "rabbit.d081r1.kernel_primitives.v1",
        "classification": "FROZEN_PYTHON_SELECTED_KERNEL_PRIMITIVES",
        "claim_ceiling": (
            "selected scalar Pauli, weak-matrix and event-measure parity only; "
            "no full collision action, RHS, Jacobian, trajectory or performance claim"
        ),
        "private_comparator_git_blob": EXPECTED_SOURCE_BLOB,
        "d080f_head": "901a62350b19cf43c17dffe45e96e8b94e4c7ca1",
        "arxiv_sources": ["2008.01074", "1605.09383", "1506.05266", "2012.02726"],
        "constants": {
            "g_f_bits": bit_string(oracle.G_F_MEV_MINUS_2),
            "sin2_theta_w_bits": bit_string(oracle.SIN2_THETA_W),
            "electron_mass_bits": bit_string(electron_mass),
            "matrix_roundoff_ulps_bits": bit_string(config.matrix_roundoff_ulps),
        },
        "self_event_count": len(oracle.independent_self_events()),
        "electron_event_count": len(oracle.independent_electron_events()),
        "invariants": {
            "support": support.tolist(),
            "d12_bits": bit_array(batch.d12),
            "d13_bits": bit_array(batch.d13),
            "d14_bits": bit_array(batch.d14),
            "d23_bits": bit_array(batch.d23),
            "d24_bits": bit_array(batch.d24),
            "d34_bits": bit_array(batch.d34),
        },
        "pauli_cases": pauli_cases,
        "self_matrix_cases": self_cases,
        "electron_matrix_cases": electron_cases,
        "event_measure_case": {
            "p1_bits": bit_string(p1),
            "p2_bits": bit_string(batch.p2[measure_index]),
            "e2_bits": bit_string(batch.e2[measure_index]),
            "phase_space_bits": bit_string(batch.phase_space[measure_index]),
            "quadrature_weight_bits": bit_string(batch.quadrature_weight[measure_index]),
            "outer_weight_bits": bit_string(outer_weight),
            "expected_bits": bit_string(measure[0]),
        },
    }
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
