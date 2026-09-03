#!/usr/bin/env python3
"""Generate a host-local D-081R1E self/electron modal diagnostic fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rabbit.decoupling import _independent_noqke as oracle  # noqa: E402

EXPECTED_COMPARATOR_BLOB = "de44feee0aa484abe26976c7dc34c579643005b5"
EXPECTED_RETAINED_SHA256 = (
    "c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380"
)
EXPECTED_NUMPY = "2.4.4"
EXPECTED_SCIPY = "1.17.1"
ORDER = 60
Y_MAX = 30.0
T_START = 10.0


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def git_blob(path: Path) -> str:
    return run("git", "hash-object", str(path.relative_to(ROOT)))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float_bits(value: float) -> str:
    return f"{np.asarray(value, dtype=np.float64).view(np.uint64).item():016x}"


def encode_array(values: object) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if not np.all(np.isfinite(array)):
        raise SystemExit("diagnostic fixture contains non-finite values")
    return {
        "shape": list(array.shape),
        "bits": [f"{item:016x}" for item in array.view(np.uint64).ravel().tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if np.__version__ != EXPECTED_NUMPY or scipy.__version__ != EXPECTED_SCIPY:
        raise SystemExit(
            f"package identity mismatch: NumPy {np.__version__}, SciPy {scipy.__version__}"
        )
    comparator = ROOT / "src/rabbit/decoupling/_independent_noqke.py"
    comparator_blob = git_blob(comparator)
    if comparator_blob != EXPECTED_COMPARATOR_BLOB:
        raise SystemExit(
            f"comparator mismatch: {comparator_blob} != {EXPECTED_COMPARATOR_BLOB}"
        )

    retained = args.retained.resolve()
    if sha256(retained) != EXPECTED_RETAINED_SHA256:
        raise SystemExit("retained state SHA-256 mismatch")
    with np.load(retained, allow_pickle=False) as archive:
        ln_a = float(np.asarray(archive["t"], dtype=np.float64).reshape(-1)[0])
        packed = np.asarray(archive["y"], dtype=np.float64).reshape(-1)
        retained_h = float(np.asarray(archive["h"], dtype=np.float64).reshape(-1)[0])

    if packed.shape != (182,) or not np.all(np.isfinite(packed)):
        raise SystemExit("invalid retained packed state")
    grid = oracle.build_independent_grid(order=ORDER, y_max=Y_MAX)
    pair_cloglog = packed[: 3 * ORDER].reshape(3, ORDER)
    temperature_gamma = float(packed[3 * ORDER])
    temperature_cm = T_START * float(np.exp(-ln_a))
    chain = oracle.cloglog_chain_factor(pair_cloglog)
    action = oracle.evaluate_independent_collision_action(
        grid=grid,
        pair_cloglog=pair_cloglog,
        temperature_cm_mev=temperature_cm,
        temperature_gamma_mev=temperature_gamma,
        config=oracle.IndependentCollisionConfig(),
    )
    thermo = oracle.independent_thermodynamics(
        grid=grid,
        pair_cloglog=pair_cloglog,
        temperature_cm_mev=temperature_cm,
        temperature_gamma_mev=temperature_gamma,
    )

    self_modal = np.asarray(action.modal_self_interaction, dtype=np.float64)
    electron_modal = np.asarray(action.modal_electron, dtype=np.float64)
    total_modal = np.asarray(action.modal_total, dtype=np.float64)
    self_native = np.asarray(action.self_interaction, dtype=np.float64)
    electron_native = np.asarray(action.electron, dtype=np.float64)
    total_native = np.asarray(action.total, dtype=np.float64)
    if not np.array_equal(total_modal, self_modal + electron_modal):
        raise SystemExit("Python modal component identity changed")
    if not np.array_equal(total_native, self_native + electron_native):
        raise SystemExit("Python native component identity changed")

    pair_rate = 0.5 * np.stack(
        (
            total_native[0] + total_native[1],
            total_native[2] + total_native[3],
            total_native[4] + total_native[5],
        )
    )
    spectral_rhs = pair_rate / (float(thermo.hubble_mev) * chain)

    arrays = {
        "packed_state": packed,
        "self_modal": self_modal,
        "electron_modal": electron_modal,
        "total_modal": total_modal,
        "self_native": self_native,
        "electron_native": electron_native,
        "total_native": total_native,
        "pair_rate": pair_rate,
        "spectral_rhs": spectral_rhs,
    }
    payload = {
        "schema": "rabbit.d081r1e.retained_component_diagnostic.v1",
        "classification": "DIAGNOSTIC_ONLY_NO_ADMISSION_MUTATION",
        "python_comparator_git_blob": comparator_blob,
        "generator_git_blob": git_blob(Path(__file__).resolve()),
        "retained_sha256": EXPECTED_RETAINED_SHA256,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "order": ORDER,
        "y_max_bits": float_bits(Y_MAX),
        "ln_a_bits": float_bits(ln_a),
        "temperature_cm_bits": float_bits(temperature_cm),
        "temperature_gamma_bits": float_bits(temperature_gamma),
        "retained_h_bits": float_bits(retained_h),
        "hubble_mev_bits": float_bits(float(thermo.hubble_mev)),
        "arrays": {name: encode_array(value) for name, value in arrays.items()},
        "absolute_envelopes": {
            name: encode_array(np.abs(value)) for name, value in arrays.items()
        },
        "claim_ceiling": (
            "host-local self/electron/total reduction diagnostic only; no new "
            "collision, packed-RHS, Jacobian, solver, trajectory or gate claim"
        ),
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
