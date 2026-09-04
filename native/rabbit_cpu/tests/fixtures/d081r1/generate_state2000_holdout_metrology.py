#!/usr/bin/env python3
"""Generate the prospectively gated D-081R1E state-2000 holdout fixture."""

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

EXPECTED_PREDECESSOR = "22240bba2f4c4c02ec2eedd4f131a8fffd3be5e2"
EXPECTED_COMPARATOR_BLOB = "de44feee0aa484abe26976c7dc34c579643005b5"
EXPECTED_TRAJECTORY_CORE_BLOB = "465a73f0ce40f7149bebdc2d67103f388e2344d9"
EXPECTED_CARGO_LOCK_BLOB = "a1b5035da5c20712d1a2a4ab077da255ff94a014"
EXPECTED_STATE_SHA256 = (
    "780ad7c1388caec23f02012781717d43ffb85d96d4d501c40c504939e7c9a44d"
)
EXPECTED_STATE_BLOB = "cfb17344ae166c01c2e5bcb14acae0d968e49477"
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
        raise SystemExit("holdout fixture contains non-finite values")
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
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", EXPECTED_PREDECESSOR, "HEAD"],
        cwd=ROOT,
    )

    comparator = ROOT / "src/rabbit/decoupling/_independent_noqke.py"
    trajectory_core = ROOT / "scripts/audit/_trajectory_core.py"
    cargo_lock = ROOT / "native/rabbit_cpu/Cargo.lock"
    identities = {
        "python_comparator_git_blob": git_blob(comparator),
        "trajectory_core_git_blob": git_blob(trajectory_core),
        "cargo_lock_git_blob": git_blob(cargo_lock),
    }
    expected_identities = {
        "python_comparator_git_blob": EXPECTED_COMPARATOR_BLOB,
        "trajectory_core_git_blob": EXPECTED_TRAJECTORY_CORE_BLOB,
        "cargo_lock_git_blob": EXPECTED_CARGO_LOCK_BLOB,
    }
    if identities != expected_identities:
        raise SystemExit(f"authority mismatch: {identities} != {expected_identities}")

    retained = args.retained.resolve()
    if not retained.is_file() or sha256(retained) != EXPECTED_STATE_SHA256:
        raise SystemExit("state-2000 SHA-256 mismatch")
    with np.load(retained, allow_pickle=False) as archive:
        keys = sorted(archive.files)
        if keys != ["h", "order", "raw", "t", "y"]:
            raise SystemExit(f"unexpected state-2000 keys: {keys}")
        ln_a = float(np.asarray(archive["t"], dtype=np.float64).reshape(-1)[0])
        packed = np.asarray(archive["y"], dtype=np.float64).reshape(-1)
        retained_h = float(np.asarray(archive["h"], dtype=np.float64).reshape(-1)[0])
        retained_order = np.asarray(archive["order"], dtype=np.float64).reshape(-1)
        retained_raw = np.asarray(archive["raw"], dtype=np.float64).reshape(-1)
    if packed.shape != (182,) or not np.all(np.isfinite(packed)):
        raise SystemExit("invalid state-2000 packed state")
    if not np.isfinite(retained_h) or retained_h == 0.0:
        raise SystemExit("invalid state-2000 retained step")

    grid = oracle.build_independent_grid(order=ORDER, y_max=Y_MAX)
    pair_cloglog = packed[: 3 * ORDER].reshape(3, ORDER)
    temperature_gamma = float(packed[3 * ORDER])
    elapsed_time = float(packed[3 * ORDER + 1])
    temperature_cm = T_START * float(np.exp(-ln_a))
    occupation = oracle.cloglog_to_occupation(pair_cloglog)
    chain = oracle.cloglog_chain_factor(pair_cloglog)
    if not (
        np.all((occupation > 0.0) & (occupation < 1.0))
        and np.all(np.isfinite(chain))
        and np.all(chain > 0.0)
    ):
        raise SystemExit("state-2000 left the strict cloglog chart")

    config = oracle.IndependentCollisionConfig()
    action = oracle.evaluate_independent_collision_action(
        grid=grid,
        pair_cloglog=pair_cloglog,
        temperature_cm_mev=temperature_cm,
        temperature_gamma_mev=temperature_gamma,
        config=config,
    )
    thermo = oracle.independent_thermodynamics(
        grid=grid,
        pair_cloglog=pair_cloglog,
        temperature_cm_mev=temperature_cm,
        temperature_gamma_mev=temperature_gamma,
    )
    eos = oracle.electromagnetic_eos_adaptive(temperature_gamma)

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
    hubble = float(thermo.hubble_mev)
    spectral_rhs = pair_rate / (hubble * chain)
    q_nu = float(action.diagnostics["event_neutrino_energy_transfer"])
    q_em = float(action.electron_bath_energy_transfer)
    transfer_scale = max(abs(q_nu) + abs(q_em), np.finfo(np.float64).tiny)
    first_law = abs(q_nu + q_em) / transfer_scale
    temperature_rhs = (
        -3.0 * (float(eos.rho) + float(eos.pressure)) + q_em / hubble
    ) / float(eos.drho_dtemperature)
    elapsed_rhs = 1.0 / hubble
    packed_rhs = np.concatenate(
        (spectral_rhs.reshape(-1), [temperature_rhs, elapsed_rhs])
    )
    if packed_rhs.shape != (182,) or not np.all(np.isfinite(packed_rhs)):
        raise SystemExit("invalid holdout packed RHS")

    arrays = {
        "packed_state": packed,
        "grid_nodes": grid.nodes,
        "grid_weights": grid.weights,
        "occupation": occupation,
        "cloglog_chain": chain,
        "self_modal": self_modal,
        "electron_modal": electron_modal,
        "total_modal": total_modal,
        "self_native": self_native,
        "electron_native": electron_native,
        "total_native": total_native,
        "pair_rate": pair_rate,
        "spectral_rhs": spectral_rhs,
        "packed_rhs": packed_rhs,
    }
    payload = {
        "schema": "rabbit.d081r1e.state2000_holdout_metrology.v1",
        "classification": "UNSEEN_HOLDOUT_AFTER_PROSPECTIVE_CONTRACT",
        "contract_path": "docs/audit/D081R1E_HOLDOUT_METROLOGY_CONTRACT_2026-09-04.md",
        "contract_commit": "d98d725c6e252180a4108fb572e97b6a90c00887",
        "holdout_authority_commit": "d22923fd9211a66a4c385c5752b372a879168ea9",
        "historical_source_commit": "78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b",
        "state_git_blob": EXPECTED_STATE_BLOB,
        "state_sha256": EXPECTED_STATE_SHA256,
        **identities,
        "generator_git_blob": git_blob(Path(__file__).resolve()),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "order": ORDER,
        "y_max_bits": float_bits(Y_MAX),
        "ln_a_bits": float_bits(ln_a),
        "temperature_cm_bits": float_bits(temperature_cm),
        "temperature_gamma_bits": float_bits(temperature_gamma),
        "elapsed_time_bits": float_bits(elapsed_time),
        "retained_h_bits": float_bits(retained_h),
        "retained_order": encode_array(retained_order),
        "retained_raw": encode_array(retained_raw),
        "hubble_mev_bits": float_bits(hubble),
        "q_nu_bits": float_bits(q_nu),
        "q_em_bits": float_bits(q_em),
        "first_law_residual_bits": float_bits(first_law),
        "temperature_rhs_bits": float_bits(temperature_rhs),
        "elapsed_rhs_bits": float_bits(elapsed_rhs),
        "support_and_roundoff": {
            "whole_reaction_domain_rejections": int(
                action.whole_reaction_domain_rejections
            ),
            "matrix_roundoff_corrections": int(action.matrix_roundoff_corrections),
            "largest_matrix_roundoff_correction_bits": float_bits(
                action.largest_matrix_roundoff_correction
            ),
        },
        "arrays": {name: encode_array(value) for name, value in arrays.items()},
        "absolute_envelopes": {
            name: encode_array(np.abs(value)) for name, value in arrays.items()
        },
        "prospective_caps": {
            "self_modal_global_relative": 1.0e-7,
            "electron_modal_global_relative": 1.0e-7,
            "total_modal_global_relative": 1.0e-7,
            "maximum_step_impact": 1.0e-3,
            "first_law_residual": 5.0e-13,
        },
        "claim_ceiling": (
            "unseen state-2000 static metrology only; no production threshold "
            "mutation, Jacobian, solver, trajectory, endpoint, N_eff or F10 gate claim"
        ),
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
