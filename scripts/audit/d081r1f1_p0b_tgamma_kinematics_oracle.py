#!/usr/bin/env python3
"""Deterministic frozen D-080A oracle for the D-081R1F1 P0B repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import scipy

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d080_tgamma_primitives import (
    evaluate_elastic_tgamma_kinematic_tangent,
    modal_basis_derivative,
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
    grid = ind.build_independent_grid(order=8, y_max=8.0)
    query = np.asarray([0.25, 1.25, 3.75, 7.25], dtype=np.float64)
    basis_derivative = modal_basis_derivative(grid, query)

    p1 = 2.0
    temperature = 2.05
    electron_mass = ind.M_ELECTRON_MEV
    tangent = evaluate_elastic_tgamma_kinematic_tangent(
        p1=p1,
        temperature_gamma_mev=temperature,
        electron_mass_mev=electron_mass,
        config=ind.IndependentCollisionConfig(),
    )
    base = tangent.base
    arrays = {
        "base_p2": bit_array(base.p2),
        "base_e2": bit_array(base.e2),
        "base_e3": bit_array(base.e3),
        "base_e4": bit_array(base.e4),
        "base_p3_magnitude": bit_array(base.p3_magnitude),
        "base_p4_magnitude": bit_array(base.p4_magnitude),
        "base_phase_space": bit_array(base.phase_space),
        "base_quadrature_weight": bit_array(base.quadrature_weight),
        "base_d12": bit_array(base.d12),
        "base_d13": bit_array(base.d13),
        "base_d14": bit_array(base.d14),
        "base_d23": bit_array(base.d23),
        "base_d24": bit_array(base.d24),
        "base_d34": bit_array(base.d34),
        "d_p2": bit_array(tangent.d_p2),
        "d_e2": bit_array(tangent.d_e2),
        "d_e3": bit_array(tangent.d_e3),
        "d_e4": bit_array(tangent.d_e4),
        "d_p3_magnitude": bit_array(tangent.d_p3_magnitude),
        "d_p4_magnitude": bit_array(tangent.d_p4_magnitude),
        "d_phase_space": bit_array(tangent.d_phase_space),
        "d_quadrature_weight": bit_array(tangent.d_quadrature_weight),
        "d_d12": bit_array(tangent.d_d12),
        "d_d13": bit_array(tangent.d_d13),
        "d_d14": bit_array(tangent.d_d14),
        "d_d23": bit_array(tangent.d_d23),
        "d_d24": bit_array(tangent.d_d24),
        "d_d34": bit_array(tangent.d_d34),
    }
    return {
        "schema": "rabbit.d081r1f1.p0b_tgamma_kinematics_oracle.v1",
        "classification": "FROZEN_PYTHON_D080A_P0B_KINEMATIC_ORACLE",
        "d080a_blob": EXPECTED_D080A_BLOB,
        "comparator_blob": EXPECTED_COMPARATOR_BLOB,
        "runtime": {
            "python_numpy": np.__version__,
            "python_scipy": scipy.__version__,
        },
        "basis": {
            "order": 8,
            "y_max_bits": bits(8.0),
            "query": bit_array(query),
            "derivative": bit_array(basis_derivative),
        },
        "kinematics": {
            "p1_bits": bits(p1),
            "temperature_bits": bits(temperature),
            "electron_mass_bits": bits(electron_mass),
            "support": np.asarray(tangent.support, dtype=bool).ravel(order="C").tolist(),
            **arrays,
            "minimum_support_margin_relative_bits": bits(
                tangent.minimum_support_margin
            ),
            "minimum_supported_lambda_margin_relative_bits": bits(
                tangent.minimum_lambda_margin
            ),
        },
        "claim_ceiling": (
            "D-080A mapped-basis and fixed-support finite-mass elastic T_gamma "
            "kinematic oracle only; no collision JVP, packed-RHS JVP, retained "
            "state, solver, or trajectory"
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
