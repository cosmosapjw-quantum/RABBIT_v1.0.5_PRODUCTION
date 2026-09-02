#!/usr/bin/env python3
"""Execute and audit the retained order-60 D-080F full Jacobian build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_rhs_jvp import evaluate_static_rhs_from_packed_state
from scripts.audit._d080b_tgamma_collision import electron_tgamma_branch_signature
from scripts.audit._d080d_static_jacobian import rhs_block_relative
from scripts.audit._d080f_frozen_full_build import (
    assemble_sealed_static_jacobian,
    classify_full_build_route,
    prepare_and_seal_static_rhs,
    verify_prepared_state_seal,
)

FIXTURE_SHA256 = "c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380"
D080E_PROJECTED_SERIAL_SECONDS = 3019.588435722
D080E_PROJECTED_PREPARED_SECONDS = 425.011892322
WALL_BUDGET_SECONDS = 900.0
CACHE_BUDGET_BYTES = 2 * 1024**3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_memory_bytes() -> dict[str, int | None]:
    result: dict[str, int | None] = {"rss": None, "hwm": None}
    status = Path("/proc/self/status")
    if not status.is_file():
        return result
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            result["rss"] = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            result["hwm"] = int(line.split()[1]) * 1024
    return result


def deterministic_directions(order: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, int(order), dtype=np.float64)
    values = np.stack(
        (
            np.stack(
                (
                    np.cos(1.13 * x),
                    -0.55 * np.sin(0.91 * x + 0.2),
                    0.35 + x * x,
                )
            ),
            np.stack(
                (
                    np.sin(1.77 * x - 0.1),
                    np.exp(-0.8 * (x + 0.15) ** 2),
                    -0.4 * x + np.cos(0.63 * x),
                )
            ),
        )
    )
    norms = np.linalg.norm(values.reshape(values.shape[0], -1), axis=1)
    return np.asarray(values / norms[:, None, None], dtype=np.float64)


def packed(c: np.ndarray, tg: float, elapsed: float) -> np.ndarray:
    return np.concatenate((np.asarray(c, dtype=np.float64).ravel(), [tg, elapsed]))


def centered_directional_ladder(
    *,
    jacobian: np.ndarray,
    grid: ind.IndependentNoQkeGrid,
    config: ind.IndependentCollisionConfig,
    c: np.ndarray,
    tcm: float,
    tg: float,
    elapsed: float,
    spectral_direction: np.ndarray,
    thermal_fraction: float,
    epsilons: list[float],
) -> dict[str, Any]:
    full_direction = np.concatenate(
        (
            spectral_direction.ravel(),
            [float(thermal_fraction) * tg, 0.0],
        )
    )
    analytic = jacobian @ full_direction
    residuals: list[float] = []
    same_branch: list[bool] = []
    base_signature = electron_tgamma_branch_signature(
        grid=grid,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    for epsilon in epsilons:
        plus_tg = tg + epsilon * full_direction[-2]
        minus_tg = tg - epsilon * full_direction[-2]
        same_branch.append(
            electron_tgamma_branch_signature(
                grid=grid,
                temperature_cm_mev=tcm,
                temperature_gamma_mev=plus_tg,
                config=config,
            )
            == base_signature
            and electron_tgamma_branch_signature(
                grid=grid,
                temperature_cm_mev=tcm,
                temperature_gamma_mev=minus_tg,
                config=config,
            )
            == base_signature
        )
        plus = evaluate_static_rhs_from_packed_state(
            grid=grid,
            packed_state=packed(
                c + epsilon * spectral_direction,
                plus_tg,
                elapsed,
            ),
            temperature_cm_mev=tcm,
            config=config,
        )
        minus = evaluate_static_rhs_from_packed_state(
            grid=grid,
            packed_state=packed(
                c - epsilon * spectral_direction,
                minus_tg,
                elapsed,
            ),
            temperature_cm_mev=tcm,
            config=config,
        )
        centered = (plus - minus) / (2.0 * epsilon)
        residuals.append(rhs_block_relative(analytic, centered, grid.order))

    logarithmic_slope = float(
        np.polyfit(np.log(np.asarray(epsilons[:3])), np.log(np.asarray(residuals[:3])), 1)[0]
    )
    return {
        "thermal_fraction": float(thermal_fraction),
        "epsilon_ladder": [float(value) for value in epsilons],
        "block_residuals": [float(value) for value in residuals],
        "best_block_residual": float(min(residuals)),
        "convergence_slope_first_three": logarithmic_slope,
        "all_samples_same_branch": bool(all(same_branch)),
        "direction_norm": float(np.linalg.norm(full_direction)),
    }


def horizontal_bar(path: Path, labels: list[str], values: list[float], title: str, xlabel: str) -> None:
    figure = plt.figure(figsize=(7.8, 4.8), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    axis.barh(labels, values)
    axis.set_xscale("log")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.invert_yaxis()
    axis.grid(True, axis="x", which="both", alpha=0.3)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def line_plot(path: Path, epsilons: list[float], series: dict[str, list[float]]) -> None:
    figure = plt.figure(figsize=(6.8, 4.5), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    for label, values in series.items():
        axis.loglog(epsilons, values, marker="o", label=label)
    axis.set_xlabel(r"dimensionless perturbation $\epsilon$")
    axis.set_ylabel("blockwise relative residual")
    axis.set_title("Order-60 original-RHS directional ladders")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained-state", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    fixture = Path(args.retained_state)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if sha256(fixture) != FIXTURE_SHA256:
        raise RuntimeError("retained state SHA-256 mismatch")
    with np.load(fixture, allow_pickle=False) as archive:
        state = np.asarray(archive["y"], dtype=np.float64)

    order, y_max = 60, 30.0
    state_size = 3 * order + 2
    if state.shape != (state_size,) or not np.all(np.isfinite(state)):
        raise RuntimeError("retained state has an invalid packed layout")
    grid = ind.build_independent_grid(order, y_max)
    config = ind.IndependentCollisionConfig(
        incoming_polar_order=4,
        final_polar_order=4,
        final_azimuth_order=4,
        electron_radial_order=24,
    )
    c = state[: 3 * order].reshape(3, order)
    tg = float(state[3 * order])
    elapsed = float(state[-1])
    expansion = 0.16286930247517223
    tcm = 10.0 * np.exp(-expansion)

    memory_initial = process_memory_bytes()
    preparation_start = time.perf_counter()
    sealed = prepare_and_seal_static_rhs(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
        max_modal_basis_bytes=768 * 1024**2,
    )
    preparation_seconds = float(time.perf_counter() - preparation_start)
    memory_after_seal = process_memory_bytes()
    assert verify_prepared_state_seal(sealed)

    serial_columns = (0, order - 1, order, 2 * order - 1, 2 * order, 3 * order - 1)
    result = assemble_sealed_static_jacobian(
        sealed,
        direction_block_size=12,
        serial_oracle_columns=serial_columns,
    )
    memory_after_build = process_memory_bytes()

    directions = deterministic_directions(order)
    epsilons = [1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5]
    directional = [
        centered_directional_ladder(
            jacobian=result.jacobian,
            grid=grid,
            config=config,
            c=c,
            tcm=tcm,
            tg=tg,
            elapsed=elapsed,
            spectral_direction=direction,
            thermal_fraction=fraction,
            epsilons=epsilons,
        )
        for direction, fraction in zip(directions, (0.015, -0.011))
    ]
    maximum_correctness_residual = max(
        result.maximum_prepared_action_residual,
        result.maximum_serial_oracle_residual,
    )
    route = classify_full_build_route(
        build_seconds=result.build_seconds,
        cache_bytes=sealed.seal.cache_unique_bytes,
        maximum_correctness_residual=maximum_correctness_residual,
        seal_unchanged=result.seal_before_sha256 == result.seal_after_sha256,
        cache_miss_delta=result.cache_miss_delta,
        cache_entry_delta=result.cache_entry_delta,
        full_matrix_measured=True,
        wall_budget_seconds=WALL_BUDGET_SECONDS,
        cache_budget_bytes=CACHE_BUDGET_BYTES,
    )
    if route in {
        "PREPARED_STATE_INTEGRITY_FAILED",
        "SAME_PHYSICS_EQUIVALENCE_FAILED",
        "EXPLICIT_CONSTRUCTION_NOT_YET_ADMISSIBLE",
    }:
        raise RuntimeError(f"D-080F failed closed with route {route}")
    if not all(item["all_samples_same_branch"] for item in directional):
        raise RuntimeError("original-RHS witness crossed a discrete branch")
    if max(item["best_block_residual"] for item in directional) >= 6.0e-3:
        raise RuntimeError("original-RHS witness residual exceeds the frozen gate")

    matrix_path = output / "retained_order60_static_jacobian.npy"
    base_path = output / "retained_order60_base_rhs.npy"
    directions_path = output / "retained_order60_probe_directions.npy"
    np.save(matrix_path, np.asarray(result.jacobian, dtype=np.float64), allow_pickle=False)
    np.save(base_path, np.asarray(result.base_rhs, dtype=np.float64), allow_pickle=False)
    np.save(directions_path, directions, allow_pickle=False)

    manifest_path = output / "prepared_array_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": sealed.seal.schema,
                "fingerprint_sha256": sealed.seal.fingerprint_sha256,
                "array_count": sealed.seal.array_count,
                "unique_array_bytes": sealed.seal.unique_array_bytes,
                "cache_unique_bytes": sealed.seal.cache_unique_bytes,
                "contract": sealed.seal.contract,
                "arrays": sealed.seal.array_manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = {
        "schema": "rabbit.d080f.retained_full_build.v1",
        "classification": "EXECUTED_RETAINED_PRODUCTION_ORDER_JACOBIAN",
        "route_decision": route,
        "claim_ceiling": (
            "single-host measured fixed-state order-60 matrix construction and "
            "explicit-callback candidacy only; no solver, prefix, endpoint, "
            "N_eff, gate, release, or publication claim"
        ),
        "fixture_sha256": FIXTURE_SHA256,
        "order": order,
        "y_max": y_max,
        "state_size": state_size,
        "spectral_columns": 3 * order,
        "matrix_shape": list(result.jacobian.shape),
        "matrix_raw_bytes": int(result.jacobian.nbytes),
        "matrix_content_sha256": result.matrix_sha256,
        "matrix_file_sha256": sha256(matrix_path),
        "elapsed_column_norm": float(np.linalg.norm(result.elapsed_column)),
        "temperature_cm_mev": float(tcm),
        "temperature_gamma_mev": float(tg),
        "preparation_seconds": preparation_seconds,
        "measured_full_build_seconds": result.build_seconds,
        "wall_budget_seconds": WALL_BUDGET_SECONDS,
        "d080e_projected_serial_seconds": D080E_PROJECTED_SERIAL_SECONDS,
        "d080e_projected_prepared_seconds": D080E_PROJECTED_PREPARED_SECONDS,
        "actual_to_prepared_projection_ratio": (
            result.build_seconds / D080E_PROJECTED_PREPARED_SECONDS
        ),
        "serial_projection_to_actual_ratio": (
            D080E_PROJECTED_SERIAL_SECONDS / result.build_seconds
        ),
        "prepared_state_seal": {
            "fingerprint_sha256": sealed.seal.fingerprint_sha256,
            "fingerprint_unchanged": (
                result.seal_before_sha256 == result.seal_after_sha256
            ),
            "array_count": sealed.seal.array_count,
            "all_arrays_readonly": sealed.seal.all_arrays_readonly,
            "unique_array_bytes": sealed.seal.unique_array_bytes,
            "cache_unique_bytes": sealed.seal.cache_unique_bytes,
            "legacy_estimated_cache_bytes": sealed.seal.cache_snapshot[
                "estimated_cache_bytes"
            ],
            "cache_miss_delta": result.cache_miss_delta,
            "cache_entry_delta": result.cache_entry_delta,
        },
        "correctness": {
            "maximum_prepared_action_residual": (
                result.maximum_prepared_action_residual
            ),
            "maximum_serial_basis_column_residual": (
                result.maximum_serial_oracle_residual
            ),
            "maximum_equivalence_residual": maximum_correctness_residual,
            "serial_oracle_columns": list(serial_columns),
            "original_rhs_directional_ladders": directional,
        },
        "memory": {
            "initial": memory_initial,
            "after_seal": memory_after_seal,
            "after_build": memory_after_build,
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
        },
    }
    receipt_path = output / "research_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    horizontal_bar(
        output / "construction_timing.png",
        ["D-080E serial projection", "D-080E prepared projection", "D-080F measured build"],
        [D080E_PROJECTED_SERIAL_SECONDS, D080E_PROJECTED_PREPARED_SECONDS, result.build_seconds],
        "Production-order Jacobian construction time",
        "seconds (single CI host)",
    )
    horizontal_bar(
        output / "memory_accounting.png",
        ["prepared unique arrays", "cache unique arrays", "legacy cache estimate", "dense matrix"],
        [
            sealed.seal.unique_array_bytes,
            sealed.seal.cache_unique_bytes,
            sealed.seal.cache_snapshot["estimated_cache_bytes"],
            result.jacobian.nbytes,
        ],
        "Explicit byte accounting",
        "bytes",
    )
    line_plot(
        output / "original_rhs_directional_ladders.png",
        epsilons,
        {
            "mixed direction 1": directional[0]["block_residuals"],
            "mixed direction 2": directional[1]["block_residuals"],
        },
    )
    horizontal_bar(
        output / "integrity_and_residuals.png",
        ["prepared action", "serial basis columns", "direction 1 best", "direction 2 best"],
        [
            max(result.maximum_prepared_action_residual, np.finfo(float).tiny),
            max(result.maximum_serial_oracle_residual, np.finfo(float).tiny),
            directional[0]["best_block_residual"],
            directional[1]["best_block_residual"],
        ],
        "Integrity and equivalence residuals",
        "dimensionless relative residual",
    )

    summary = f"""# D-080F retained production-order full build

- classification: `EXECUTED_RETAINED_PRODUCTION_ORDER_JACOBIAN`
- route: `{route}`
- matrix: `{state_size} x {state_size}`
- measured full build: `{result.build_seconds:.9f} s`
- D-080E prepared projection: `{D080E_PROJECTED_PREPARED_SECONDS:.9f} s`
- cache fingerprint unchanged: `{result.seal_before_sha256 == result.seal_after_sha256}`
- cache miss delta: `{result.cache_miss_delta}`
- cache entry delta: `{result.cache_entry_delta}`
- unique cache bytes: `{sealed.seal.cache_unique_bytes}`
- maximum prepared-action residual: `{result.maximum_prepared_action_residual:.16e}`
- maximum serial basis-column residual: `{result.maximum_serial_oracle_residual:.16e}`
- mixed original-RHS best residuals: `{directional[0]['best_block_residual']:.16e}`, `{directional[1]['best_block_residual']:.16e}`

This is a measured fixed-state matrix-construction result, not a solver or
trajectory result.  The explicit callback remains only a candidate for a
separately preregistered paired BDF experiment.
"""
    (output / "probe_summary.md").write_text(summary, encoding="utf-8")

    artifact_names = (
        "construction_timing.png",
        "integrity_and_residuals.png",
        "memory_accounting.png",
        "original_rhs_directional_ladders.png",
        "prepared_array_manifest.json",
        "probe_summary.md",
        "research_receipt.json",
        "retained_order60_base_rhs.npy",
        "retained_order60_probe_directions.npy",
        "retained_order60_static_jacobian.npy",
    )
    checksum_lines = [
        f"{sha256(output / name)}  {name}" for name in artifact_names
    ]
    (output / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
