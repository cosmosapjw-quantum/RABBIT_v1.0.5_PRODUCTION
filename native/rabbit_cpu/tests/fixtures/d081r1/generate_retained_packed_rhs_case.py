#!/usr/bin/env python3
"""Generate the deterministic D-081R1E retained order-60 packed-RHS oracle."""

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

D4_HEAD = "002086662bf2e553c78f4b247868cb1fd9e43f21"
D4_TREE = "d01ae7c0d3d9fbe8ce9513d054b835d3596f1de2"
EXPECTED_COMPARATOR_BLOB = "de44feee0aa484abe26976c7dc34c579643005b5"
EXPECTED_TRAJECTORY_CORE_BLOB = "465a73f0ce40f7149bebdc2d67103f388e2344d9"
EXPECTED_CARGO_LOCK_BLOB = "a1b5035da5c20712d1a2a4ab077da255ff94a014"
RETAINED_SOURCE_COMMIT = "78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b"
RETAINED_SOURCE_PATH = (
    ".agent-harness/runs/run-20260805-f10-v3-campaign/"
    "v3a_r2/domain/state_1200.npz"
)
EXPECTED_RETAINED_SHA256 = (
    "c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380"
)
ORDER = 60
Y_MAX = 30.0
T_START = 10.0
EXPECTED_NUMPY = "2.4.4"
EXPECTED_SCIPY = "1.17.1"
OUTPUT = Path(__file__).with_name("retained_packed_rhs_case.json")


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def git_blob(path: Path) -> str:
    return run("git", "hash-object", str(path.relative_to(ROOT)))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float_bits(value: float) -> str:
    return f"{np.float64(value).view(np.uint64).item():016x}"


def encode_array(values: object) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return {
        "shape": list(array.shape),
        "bits": [f"{item:016x}" for item in array.view(np.uint64).ravel().tolist()],
    }


def encode_float_map(values: dict[str, float]) -> dict[str, str]:
    return {key: float_bits(values[key]) for key in sorted(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if np.__version__ != EXPECTED_NUMPY:
        raise SystemExit(f"NumPy mismatch: {np.__version__} != {EXPECTED_NUMPY}")
    if scipy.__version__ != EXPECTED_SCIPY:
        raise SystemExit(f"SciPy mismatch: {scipy.__version__} != {EXPECTED_SCIPY}")

    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", D4_HEAD, "HEAD"], cwd=ROOT
    )
    if run("git", "rev-parse", f"{D4_HEAD}^{{tree}}") != D4_TREE:
        raise SystemExit("D4 tree identity mismatch")

    comparator = ROOT / "src/rabbit/decoupling/_independent_noqke.py"
    trajectory_core = ROOT / "scripts/audit/_trajectory_core.py"
    cargo_lock = ROOT / "native/rabbit_cpu/Cargo.lock"
    identities = {
        "python_comparator_git_blob": git_blob(comparator),
        "trajectory_core_git_blob": git_blob(trajectory_core),
        "cargo_lock_git_blob": git_blob(cargo_lock),
    }
    expected = {
        "python_comparator_git_blob": EXPECTED_COMPARATOR_BLOB,
        "trajectory_core_git_blob": EXPECTED_TRAJECTORY_CORE_BLOB,
        "cargo_lock_git_blob": EXPECTED_CARGO_LOCK_BLOB,
    }
    if identities != expected:
        raise SystemExit(f"authority identity mismatch: {identities} != {expected}")

    retained = args.retained.resolve()
    if not retained.is_file():
        raise SystemExit(f"retained state missing: {retained}")
    retained_digest = sha256(retained)
    if retained_digest != EXPECTED_RETAINED_SHA256:
        raise SystemExit(
            f"retained SHA-256 mismatch: {retained_digest} != {EXPECTED_RETAINED_SHA256}"
        )

    with np.load(retained, allow_pickle=False) as archive:
        keys = sorted(archive.files)
        required = {"t", "y", "raw", "h", "order"}
        if set(keys) != required:
            raise SystemExit(f"unexpected retained keys: {keys}")
        ln_a = float(np.asarray(archive["t"], dtype=np.float64).reshape(-1)[0])
        packed = np.asarray(archive["y"], dtype=np.float64).reshape(-1)
        retained_raw = np.asarray(archive["raw"], dtype=np.float64).reshape(-1)
        retained_h = np.asarray(archive["h"], dtype=np.float64).reshape(-1)
        retained_order = np.asarray(archive["order"], dtype=np.float64).reshape(-1)

    if packed.shape != (3 * ORDER + 2,):
        raise SystemExit(f"unexpected packed state shape: {packed.shape}")
    if not np.all(np.isfinite(packed)) or not np.isfinite(ln_a):
        raise SystemExit("retained state contains non-finite values")

    grid = oracle.build_independent_grid(order=ORDER, y_max=Y_MAX)
    pair_cloglog = packed[: 3 * ORDER].reshape(3, ORDER)
    temperature_gamma = float(packed[3 * ORDER])
    elapsed_time = float(packed[3 * ORDER + 1])
    temperature_cm = T_START * float(np.exp(-ln_a))
    if not (np.isfinite(temperature_gamma) and temperature_gamma > 0.0):
        raise SystemExit("invalid retained photon temperature")
    if not (np.isfinite(temperature_cm) and temperature_cm > 0.0):
        raise SystemExit("invalid reconstructed comoving temperature")

    occupation = oracle.cloglog_to_occupation(pair_cloglog)
    if not np.all(np.isfinite(occupation)):
        raise SystemExit("non-finite occupation")
    if not (np.all(occupation > 0.0) and np.all(occupation < 1.0)):
        raise SystemExit("occupation left the strict open interval")
    chain = oracle.cloglog_chain_factor(pair_cloglog)
    if not np.all(np.isfinite(chain)) or not np.all(chain > 0.0):
        raise SystemExit("invalid cloglog chain factor")

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

    total_native = np.asarray(action.total, dtype=np.float64)
    total_modal = np.asarray(action.modal_total, dtype=np.float64)
    if total_native.shape != (6, ORDER) or total_modal.shape != (6, ORDER):
        raise SystemExit("unexpected collision-action shape")
    pair_rate = 0.5 * np.stack(
        (
            total_native[0] + total_native[1],
            total_native[2] + total_native[3],
            total_native[4] + total_native[5],
        )
    )

    hubble = float(thermo.hubble_mev)
    if not (np.isfinite(hubble) and hubble > 0.0):
        raise SystemExit("invalid Hubble rate")
    spectral_rhs = pair_rate / (hubble * chain)
    q_em = float(action.electron_bath_energy_transfer)
    q_nu = float(action.diagnostics["event_neutrino_energy_transfer"])
    transfer_scale = max(abs(q_nu) + abs(q_em), np.finfo(np.float64).tiny)
    first_law_residual = abs(q_nu + q_em) / transfer_scale
    if first_law_residual > 5.0e-13:
        raise SystemExit(f"first-law residual too large: {first_law_residual}")

    temperature_rhs = (
        -3.0 * (float(eos.rho) + float(eos.pressure)) + q_em / hubble
    ) / float(eos.drho_dtemperature)
    elapsed_rhs = 1.0 / hubble
    packed_rhs = np.concatenate(
        (spectral_rhs.reshape(-1), [temperature_rhs, elapsed_rhs])
    )
    if packed_rhs.shape != packed.shape or not np.all(np.isfinite(packed_rhs)):
        raise SystemExit("invalid packed RHS")

    phase = np.asarray(grid.weights) * np.asarray(grid.nodes) ** 3
    rho_by_flavour = temperature_cm**4 / np.pi**2 * np.sum(
        occupation * phase[None, :], axis=1, dtype=np.float64
    )
    rho_nu = float(np.sum(rho_by_flavour, dtype=np.float64))
    rho_em = float(eos.rho)
    pressure_em = float(eos.pressure)
    drho_em_dt = float(eos.drho_dtemperature)
    rho_total = rho_nu + rho_em

    payload = {
        "schema": "rabbit.d081r1e.retained_packed_rhs.v1",
        "repository": "cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION",
        "d4_head": D4_HEAD,
        "d4_tree": D4_TREE,
        **identities,
        "generator_git_blob": git_blob(Path(__file__).resolve()),
        "retained_source_commit": RETAINED_SOURCE_COMMIT,
        "retained_source_path": RETAINED_SOURCE_PATH,
        "retained_sha256": retained_digest,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "order": ORDER,
        "y_max_bits": float_bits(Y_MAX),
        "t_start_bits": float_bits(T_START),
        "ln_a_bits": float_bits(ln_a),
        "temperature_cm_bits": float_bits(temperature_cm),
        "temperature_gamma_bits": float_bits(temperature_gamma),
        "elapsed_time_bits": float_bits(elapsed_time),
        "retained_archive_keys": keys,
        "retained_raw": encode_array(retained_raw),
        "retained_h": encode_array(retained_h),
        "retained_order_history": encode_array(retained_order),
        "packed_state": encode_array(packed),
        "grid_nodes": encode_array(grid.nodes),
        "grid_weights": encode_array(grid.weights),
        "occupation": encode_array(occupation),
        "cloglog_chain_factor": encode_array(chain),
        "combined_action_modal": encode_array(total_modal),
        "combined_action_native": encode_array(total_native),
        "pair_collision_rate": encode_array(pair_rate),
        "rho_neutrino_by_flavour": encode_array(rho_by_flavour),
        "rho_neutrino_total_bits": float_bits(rho_nu),
        "rho_electromagnetic_bits": float_bits(rho_em),
        "pressure_electromagnetic_bits": float_bits(pressure_em),
        "drho_electromagnetic_dt_bits": float_bits(drho_em_dt),
        "rho_total_bits": float_bits(rho_total),
        "hubble_mev_bits": float_bits(hubble),
        "q_nu_bits": float_bits(q_nu),
        "q_em_bits": float_bits(q_em),
        "first_law_residual_bits": float_bits(first_law_residual),
        "spectral_rhs": encode_array(spectral_rhs),
        "temperature_rhs_bits": float_bits(temperature_rhs),
        "elapsed_rhs_bits": float_bits(elapsed_rhs),
        "packed_rhs": encode_array(packed_rhs),
        "support_and_roundoff_metrology": {
            "whole_reaction_domain_rejections": int(
                action.whole_reaction_domain_rejections
            ),
            "matrix_roundoff_corrections": int(action.matrix_roundoff_corrections),
            "largest_matrix_roundoff_correction_bits": float_bits(
                action.largest_matrix_roundoff_correction
            ),
        },
        "diagnostics": encode_float_map(
            {key: float(value) for key, value in dict(action.diagnostics).items()}
        ),
        "absolute_envelopes": {
            "combined_action_native": encode_array(np.abs(total_native)),
            "pair_collision_rate": encode_array(np.abs(pair_rate)),
            "spectral_rhs": encode_array(np.abs(spectral_rhs)),
            "packed_rhs": encode_array(np.abs(packed_rhs)),
        },
        "claim_ceiling": (
            "frozen Python retained order-60 packed-RHS oracle only; no Rust "
            "packed-RHS parity, JVP/Jacobian, PyO3, diffsol, trajectory, "
            "performance, endpoint, N_eff, publication authority or F10 gate movement"
        ),
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"sha256={sha256(output)}")
    print(f"ln_a={ln_a:.17e}")
    print(f"T_cm={temperature_cm:.17e}")
    print(f"T_gamma={temperature_gamma:.17e}")
    print(f"H={hubble:.17e}")
    print(f"first_law={first_law_residual:.17e}")


if __name__ == "__main__":
    main()
