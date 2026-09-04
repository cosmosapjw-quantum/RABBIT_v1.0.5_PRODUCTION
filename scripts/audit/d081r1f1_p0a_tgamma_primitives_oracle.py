#!/usr/bin/env python3
"""Deterministic D-081R1F1-P0A oracle for moving quadrature and QED-off EOS tangents."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import scipy

from scripts.audit._d080_tgamma_primitives import (
    electron_half_line_tgamma_tangent,
    electromagnetic_eos_tgamma_tangent,
)

EXPECTED_D080A_BLOB = "c585d5865fd68a90a04a76ab540b8437fba8cfce"
EXPECTED_COMPARATOR_BLOB = "de44feee0aa484abe26976c7dc34c579643005b5"


def bits(value: float) -> str:
    return f"{struct.unpack('>Q', struct.pack('>d', float(value)))[0]:016x}"


def bit_array(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "shape": list(array.shape),
        "bits": [bits(value) for value in array.ravel(order="C")],
    }


def build_payload() -> dict[str, object]:
    temperature = 2.05
    momentum, weights, d_momentum, d_weights = electron_half_line_tgamma_tangent(
        48, temperature
    )
    eos_cases: list[dict[str, str]] = []
    for case_temperature in (0.5, 2.05, 10.0):
        tangent = electromagnetic_eos_tgamma_tangent(case_temperature)
        eos_cases.append(
            {
                "temperature_bits": bits(case_temperature),
                "rho_bits": bits(tangent.base.rho),
                "pressure_bits": bits(tangent.base.pressure),
                "d_rho_bits": bits(tangent.d_rho),
                "d_pressure_bits": bits(tangent.d_pressure),
                "d2_rho_bits": bits(tangent.d2_rho),
            }
        )
    return {
        "schema": "rabbit.d081r1f1.p0a_tgamma_primitives_oracle.v1",
        "classification": "FROZEN_PYTHON_D080A_PRIMITIVE_ORACLE",
        "d080a_blob": EXPECTED_D080A_BLOB,
        "comparator_blob": EXPECTED_COMPARATOR_BLOB,
        "runtime": {
            "python_numpy": np.__version__,
            "python_scipy": scipy.__version__,
        },
        "half_line": {
            "order": 48,
            "temperature_bits": bits(temperature),
            "momentum": bit_array(momentum),
            "weights": bit_array(weights),
            "d_momentum_dt": bit_array(d_momentum),
            "d_weights_dt": bit_array(d_weights),
        },
        "eos": eos_cases,
        "claim_ceiling": (
            "moving electron half-line rule and QED-off electromagnetic EOS tangents only; "
            "no finite-mass collision kinematics, collision JVP, packed-RHS JVP, retained state, solver, or trajectory"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload()
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    args.output.write_bytes(encoded)
    print(f"wrote={args.output}")
    print(f"sha256={hashlib.sha256(encoded).hexdigest()}")


if __name__ == "__main__":
    main()
