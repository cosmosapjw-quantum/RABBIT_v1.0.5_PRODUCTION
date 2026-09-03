#!/usr/bin/env python3
"""Generate the deterministic D-081R1E0 retained packed-RHS Python oracle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Mapping

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rabbit.decoupling import _independent_noqke as ind  # noqa: E402
from scripts.audit import _trajectory_core as trajectory  # noqa: E402

SCHEMA = "rabbit.d081r1e0.retained_packed_rhs_oracle.v1"
CLAIM_CEILING = "FROZEN_RETAINED_ORDER60_PYTHON_PACKED_RHS_ORACLE_ONLY"
D4_FINAL_HEAD = "002086662bf2e553c78f4b247868cb1fd9e43f21"
COMPARATOR_BLOB = "de44feee0aa484abe26976c7dc34c579643005b5"
TRAJECTORY_CORE_BLOB = "465a73f0ce40f7149bebdc2d67103f388e2344d9"
RETAINED_COMMIT = "78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b"
RETAINED_PATH = (
    ".agent-harness/runs/run-20260805-f10-v3-campaign/"
    "v3a_r2/domain/state_1200.npz"
)
RETAINED_SHA256 = "c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380"
ORDER = 60
Y_MAX = 30.0
STATE_SIZE = 3 * ORDER + 2
EXPANSION_N = 0.16286930247517223
T_START_MEV = 10.0
EXPECTED_PYTHON = "3.12.3"
EXPECTED_NUMPY = "2.4.4"
EXPECTED_SCIPY = "1.17.1"


def git_blob(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(path.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f64_bits(value: float) -> str:
    return f"{np.float64(value).view(np.uint64).item():016x}"


def encode_array(values: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if not np.all(np.isfinite(array)):
        raise RuntimeError("refusing to encode a nonfinite array")
    return {
        "shape": list(array.shape),
        "bits": [f"{item:016x}" for item in array.view(np.uint64).ravel().tolist()],
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def encode_float_map(values: Mapping[str, object]) -> dict[str, str]:
    encoded: dict[str, str] = {}
    for key in sorted(values):
        value = float(values[key])
        if not np.isfinite(value):
            raise RuntimeError(f"nonfinite scalar in {key}")
        encoded[str(key)] = f64_bits(value)
    return encoded


def encode_array_hash_map(values: Mapping[object, np.ndarray]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in sorted(values, key=str):
        array = np.ascontiguousarray(np.asarray(values[key], dtype=np.float64))
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"nonfinite family array {key}")
        result[str(key)] = {
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    return result


def require_environment() -> None:
    actual = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    expected = {
        "python": EXPECTED_PYTHON,
        "numpy": EXPECTED_NUMPY,
        "scipy": EXPECTED_SCIPY,
    }
    if actual != expected:
        raise RuntimeError(f"oracle environment mismatch: {actual} != {expected}")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} is not pinned to one thread")


def make_setup() -> trajectory.Setup:
    grid = ind.build_independent_grid(ORDER, Y_MAX)
    config = ind.IndependentCollisionConfig(
        incoming_polar_order=4,
        final_polar_order=4,
        final_azimuth_order=4,
        electron_radial_order=24,
    )
    return trajectory.Setup(
        grid=grid,
        config=config,
        order=ORDER,
        y_max=Y_MAX,
        t_start=T_START_MEV,
        label="d081r1e0-retained-order60",
    )


def build_payload(retained_state: Path) -> dict[str, object]:
    require_environment()
    comparator = ROOT / "src/rabbit/decoupling/_independent_noqke.py"
    trajectory_core = ROOT / "scripts/audit/_trajectory_core.py"
    generator = Path(__file__).resolve()
    if git_blob(comparator) != COMPARATOR_BLOB:
        raise RuntimeError("frozen comparator Git blob mismatch")
    if git_blob(trajectory_core) != TRAJECTORY_CORE_BLOB:
        raise RuntimeError("frozen trajectory-core Git blob mismatch")
    if sha256(retained_state) != RETAINED_SHA256:
        raise RuntimeError("retained-state SHA-256 mismatch")

    with np.load(retained_state, allow_pickle=False) as archive:
        if "y" not in archive.files:
            raise RuntimeError("retained NPZ does not contain packed member y")
        state = np.asarray(archive["y"], dtype=np.float64)
    if state.shape != (STATE_SIZE,) or not np.all(np.isfinite(state)):
        raise RuntimeError("retained packed state has an invalid layout")

    setup = make_setup()
    spectral_size = 3 * ORDER
    coordinates = np.asarray(state[:spectral_size].reshape(3, ORDER), dtype=np.float64)
    temperature_gamma = float(state[spectral_size])
    elapsed = float(state[-1])
    temperature_cm = T_START_MEV * float(np.exp(-EXPANSION_N))

    occupations = ind.cloglog_to_occupation(coordinates)
    chain = ind.cloglog_chain_factor(coordinates)
    if not (
        np.all(occupations > 0.0)
        and np.all(occupations < 1.0)
        and np.all(chain > 0.0)
    ):
        raise RuntimeError("retained chart left its strict physical domain")

    stats = trajectory.Stats()
    authority_rhs = np.asarray(
        trajectory.make_rhs(
            setup,
            stats,
            trajectory.Deadline(3600.0),
        )(EXPANSION_N, state),
        dtype=np.float64,
    )
    if authority_rhs.shape != (STATE_SIZE,) or stats.evals != 1:
        raise RuntimeError("trajectory-core RHS did not execute exactly once")

    action = ind.evaluate_independent_collision_action(
        grid=setup.grid,
        pair_cloglog=coordinates,
        temperature_cm_mev=temperature_cm,
        temperature_gamma_mev=temperature_gamma,
        config=setup.config,
    )
    thermo = ind.independent_thermodynamics(
        grid=setup.grid,
        pair_cloglog=coordinates,
        temperature_cm_mev=temperature_cm,
        temperature_gamma_mev=temperature_gamma,
    )
    eos = ind.electromagnetic_eos_adaptive(temperature_gamma)

    total = np.asarray(action.total, dtype=np.float64)
    self_native = np.asarray(action.self_interaction, dtype=np.float64)
    electron_native = np.asarray(action.electron, dtype=np.float64)
    modal_total = np.asarray(action.modal_total, dtype=np.float64)
    modal_self = np.asarray(action.modal_self_interaction, dtype=np.float64)
    modal_electron = np.asarray(action.modal_electron, dtype=np.float64)
    expected_shape = (6, ORDER)
    for name, array in {
        "total": total,
        "self_native": self_native,
        "electron_native": electron_native,
        "modal_total": modal_total,
        "modal_self": modal_self,
        "modal_electron": modal_electron,
    }.items():
        if array.shape != expected_shape or not np.all(np.isfinite(array)):
            raise RuntimeError(f"invalid {name} array")

    if not np.array_equal(total.view(np.uint64), (self_native + electron_native).view(np.uint64)):
        raise RuntimeError("native total is not the exact Python component sum")
    if not np.array_equal(modal_total.view(np.uint64), (modal_self + modal_electron).view(np.uint64)):
        raise RuntimeError("modal total is not the exact Python component sum")

    pair_rate = 0.5 * np.stack(
        (total[0] + total[1], total[2] + total[3], total[4] + total[5])
    )
    spectral_rhs = pair_rate / (thermo.hubble_mev * chain)
    q_em = float(action.electron_bath_energy_transfer)
    temperature_rhs = (
        -3.0 * (eos.rho + eos.pressure) + q_em / thermo.hubble_mev
    ) / eos.drho_dtemperature
    elapsed_rhs = 1.0 / thermo.hubble_mev
    reconstructed_rhs = np.concatenate(
        (spectral_rhs.ravel(), [temperature_rhs, elapsed_rhs])
    )
    if not np.array_equal(authority_rhs.view(np.uint64), reconstructed_rhs.view(np.uint64)):
        mismatch = np.flatnonzero(authority_rhs.view(np.uint64) != reconstructed_rhs.view(np.uint64))
        raise RuntimeError(f"packed-RHS reconstruction mismatch at {mismatch[:8].tolist()}")

    diagnostics = {str(key): float(value) for key, value in dict(action.diagnostics).items()}
    q_nu = float(diagnostics["event_neutrino_energy_transfer"])
    first_law = abs(q_nu + q_em) / max(abs(q_nu) + abs(q_em), np.finfo(float).tiny)
    if first_law > 5.0e-13:
        raise RuntimeError(f"first-law residual exceeds contract: {first_law}")

    moments = {
        "self": asdict(
            ind.independent_action_moments(
                grid=setup.grid,
                action=self_native,
                temperature_cm_mev=temperature_cm,
            )
        ),
        "electron": asdict(
            ind.independent_action_moments(
                grid=setup.grid,
                action=electron_native,
                temperature_cm_mev=temperature_cm,
            )
        ),
        "total": asdict(
            ind.independent_action_moments(
                grid=setup.grid,
                action=total,
                temperature_cm_mev=temperature_cm,
            )
        ),
    }

    return {
        "schema": SCHEMA,
        "claim_ceiling": CLAIM_CEILING,
        "authorities": {
            "d081r1d4_final_head": D4_FINAL_HEAD,
            "private_comparator_git_blob": COMPARATOR_BLOB,
            "trajectory_core_git_blob": TRAJECTORY_CORE_BLOB,
            "generator_git_blob": git_blob(generator),
            "retained_state_source_commit": RETAINED_COMMIT,
            "retained_state_path": RETAINED_PATH,
            "retained_state_sha256": RETAINED_SHA256,
        },
        "environment": {
            "python": EXPECTED_PYTHON,
            "numpy": EXPECTED_NUMPY,
            "scipy": EXPECTED_SCIPY,
            "openblas_threads": "1",
            "omp_threads": "1",
            "mkl_threads": "1",
        },
        "configuration": {
            "order": ORDER,
            "y_max_bits": f64_bits(Y_MAX),
            "state_size": STATE_SIZE,
            "expansion_n_bits": f64_bits(EXPANSION_N),
            "temperature_start_mev_bits": f64_bits(T_START_MEV),
            "temperature_cm_mev_bits": f64_bits(temperature_cm),
            "temperature_gamma_mev_bits": f64_bits(temperature_gamma),
            "elapsed_mev_inverse_bits": f64_bits(elapsed),
            "incoming_polar_order": int(setup.config.incoming_polar_order),
            "final_polar_order": int(setup.config.final_polar_order),
            "final_azimuth_order": int(setup.config.final_azimuth_order),
            "electron_radial_order": int(setup.config.electron_radial_order),
        },
        "arrays": {
            "packed_state": encode_array(state),
            "pair_cloglog": encode_array(coordinates),
            "occupation": encode_array(occupations),
            "cloglog_chain": encode_array(chain),
            "grid_nodes": encode_array(setup.grid.nodes),
            "grid_weights": encode_array(setup.grid.weights),
            "self_native": encode_array(self_native),
            "electron_native": encode_array(electron_native),
            "total_native": encode_array(total),
            "self_modal": encode_array(modal_self),
            "electron_modal": encode_array(modal_electron),
            "total_modal": encode_array(modal_total),
            "pair_rate": encode_array(pair_rate),
            "spectral_rhs": encode_array(spectral_rhs),
            "packed_rhs_trajectory_core": encode_array(authority_rhs),
            "packed_rhs_reconstructed": encode_array(reconstructed_rhs),
        },
        "scalars": {
            "temperature_rhs_bits": f64_bits(temperature_rhs),
            "elapsed_rhs_bits": f64_bits(elapsed_rhs),
            "neutrino_energy_transfer_bits": f64_bits(q_nu),
            "electromagnetic_energy_transfer_bits": f64_bits(q_em),
            "first_law_residual_bits": f64_bits(first_law),
            "occupation_min_bits": f64_bits(float(np.min(occupations))),
            "occupation_max_bits": f64_bits(float(np.max(occupations))),
            "chain_min_bits": f64_bits(float(np.min(chain))),
            "chain_max_bits": f64_bits(float(np.max(chain))),
        },
        "thermodynamics": encode_float_map(asdict(thermo)),
        "electromagnetic_eos": encode_float_map(asdict(eos)),
        "action_diagnostics": encode_float_map(diagnostics),
        "moments": {key: encode_float_map(value) for key, value in moments.items()},
        "metrology": {
            "whole_reaction_domain_rejections": int(action.whole_reaction_domain_rejections),
            "matrix_roundoff_corrections": int(action.matrix_roundoff_corrections),
            "largest_matrix_roundoff_correction_bits": f64_bits(
                float(action.largest_matrix_roundoff_correction)
            ),
            "self_row_hashes": encode_array_hash_map(dict(action.self_rows)),
            "electron_family_hashes": encode_array_hash_map(dict(action.electron_families)),
            "rhs_paths_bitwise_identical": True,
            "native_component_sum_bitwise_identical": True,
            "modal_component_sum_bitwise_identical": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained-state", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    retained_state = Path(args.retained_state).resolve()
    output = Path(args.output).resolve()
    payload = build_payload(retained_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
