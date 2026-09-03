#!/usr/bin/env python3
"""Freeze the NumPy 2.4.4 GL12/GL48 rules used by the F10 collision lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
OUTPUT = Path(__file__).with_name("f10_inner_quadrature_case.json")
COMPARATOR = ROOT / "src/rabbit/decoupling/_independent_noqke.py"
EXPECTED_COMPARATOR_BLOB = "de44feee0aa484abe26976c7dc34c579643005b5"
EXPECTED_NUMPY = "2.4.4"
TEMPERATURE_PROBE_MEV = 2.05


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def git_blob(path: Path) -> str:
    return run("git", "hash-object", str(path.relative_to(ROOT)))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float_bits(value: float) -> str:
    scalar = np.asarray(value, dtype=np.float64)
    return f"{int(scalar.view(np.uint64)):016x}"


def encode_array(values: object) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if not np.all(np.isfinite(array)):
        raise SystemExit("inner quadrature fixture contains non-finite values")
    return {
        "shape": list(array.shape),
        "bits": [f"{item:016x}" for item in array.view(np.uint64).ravel().tolist()],
    }


def rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        nodes.shape != (order,)
        or weights.shape != (order,)
        or not np.all(np.diff(nodes) > 0.0)
        or not np.all(weights > 0.0)
    ):
        raise SystemExit(f"invalid NumPy GL{order} rule")
    return nodes, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if np.__version__ != EXPECTED_NUMPY:
        raise SystemExit(f"NumPy mismatch: {np.__version__} != {EXPECTED_NUMPY}")
    comparator_blob = git_blob(COMPARATOR)
    if comparator_blob != EXPECTED_COMPARATOR_BLOB:
        raise SystemExit(
            f"comparator mismatch: {comparator_blob} != {EXPECTED_COMPARATOR_BLOB}"
        )

    gl12_nodes, gl12_weights = rule(12)
    gl48_nodes, gl48_weights = rule(48)
    unit = 0.5 * (gl48_nodes + 1.0)
    one_minus = 1.0 - unit
    electron_p2 = TEMPERATURE_PROBE_MEV * unit / one_minus
    electron_weights = 0.5 * gl48_weights * TEMPERATURE_PROBE_MEV / one_minus**2

    payload = {
        "schema": "rabbit.d081r1e.f10_inner_quadrature.v1",
        "classification": "FROZEN_NUMPY_FINITE_DIMENSIONAL_OPERATOR",
        "claim_ceiling": (
            "F10 GL12 angular and GL48 electron-radial binary64 identity only; "
            "no collision-action, packed-RHS, Jacobian, solver or trajectory claim"
        ),
        "numpy_version": np.__version__,
        "python_comparator_git_blob": comparator_blob,
        "generator_git_blob": git_blob(Path(__file__).resolve()),
        "temperature_probe_bits": float_bits(TEMPERATURE_PROBE_MEV),
        "gl12_nodes": encode_array(gl12_nodes),
        "gl12_weights": encode_array(gl12_weights),
        "gl48_nodes": encode_array(gl48_nodes),
        "gl48_weights": encode_array(gl48_weights),
        "electron_p2": encode_array(electron_p2),
        "electron_weights": encode_array(electron_weights),
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
