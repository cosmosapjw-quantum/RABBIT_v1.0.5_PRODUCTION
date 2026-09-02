#!/usr/bin/env python3
"""Generate deterministic D-080D square static-Jacobian evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_rhs_jvp import (
    c_only_state_validator,
    evaluate_static_rhs_from_packed_state,
)
from scripts.audit._d080b_tgamma_collision import electron_tgamma_branch_signature
from scripts.audit._d080d_static_jacobian import (
    EXPECTED_COMPARATOR_BLOB_SHA,
    EXPECTED_D079_RHS_BLOB_SHA,
    EXPECTED_D080C_RHS_BLOB_SHA,
    StaticJacobianResult,
    assemble_static_jacobian,
    evaluate_static_rhs_direction_jvp,
    rhs_block_relative,
    static_newton_matrix,
)


@dataclass(frozen=True)
class DirectionalLadder:
    epsilons: list[float]
    block_residuals: list[float]
    spectral_residuals: list[float]
    temperature_residuals: list[float]
    elapsed_residuals: list[float]
    branch_flags: list[bool]
    best_index: int
    best_centered: np.ndarray
    convergence_slope: float


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    framed = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    return hashlib.sha1(framed).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    scale = max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny)
    return float(np.linalg.norm(a - b) / scale)


def scalar_relative(left: float, right: float) -> float:
    scale = max(abs(float(left)), abs(float(right)), np.finfo(float).tiny)
    return float(abs(float(left) - float(right)) / scale)


def block_residuals(
    left: np.ndarray,
    right: np.ndarray,
    order: int,
) -> tuple[float, float, float, float]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    spectral_size = 3 * order
    spectral = relative(a[:spectral_size], b[:spectral_size])
    temperature = scalar_relative(a[-2], b[-2])
    elapsed = scalar_relative(a[-1], b[-1])
    return max(spectral, temperature, elapsed), spectral, temperature, elapsed


def collision_config(*, retained: bool = False) -> ind.IndependentCollisionConfig:
    if retained:
        return ind.IndependentCollisionConfig(
            incoming_polar_order=4,
            final_polar_order=4,
            final_azimuth_order=4,
            electron_radial_order=24,
        )
    return ind.IndependentCollisionConfig(
        incoming_polar_order=2,
        final_polar_order=2,
        final_azimuth_order=4,
        electron_radial_order=8,
    )


def thermal_case() -> tuple[ind.IndependentNoQkeGrid, np.ndarray, float, float]:
    grid = ind.build_independent_grid(8, 8.0)
    logits = np.stack(
        [
            -grid.nodes + 0.04 * np.exp(-grid.nodes / 3.0),
            -grid.nodes - 0.02 * np.exp(-grid.nodes / 4.0),
            -grid.nodes - 0.02 * np.exp(-grid.nodes / 4.0),
        ]
    )
    return grid, ind.pair_logits_to_cloglog(logits), 2.0, 2.05


def equilibrium_case() -> tuple[ind.IndependentNoQkeGrid, np.ndarray, float, float]:
    grid = ind.build_independent_grid(8, 8.0)
    temperature = 2.0
    c = ind.pair_logits_to_cloglog(np.stack([-grid.nodes for _ in range(3)]))
    return grid, c, temperature, temperature


def weak_tail_case() -> tuple[ind.IndependentNoQkeGrid, np.ndarray, float, float]:
    grid = ind.build_independent_grid(8, 10.0)
    logits = np.stack(
        [
            -grid.nodes + 0.012 * np.exp(-grid.nodes / 2.0),
            -grid.nodes - 0.006 * np.exp(-grid.nodes / 2.5),
            -grid.nodes - 0.006 * np.exp(-grid.nodes / 2.5),
        ]
    )
    return grid, ind.pair_logits_to_cloglog(logits), 0.45, 0.50


def packed(c: np.ndarray, tg: float, elapsed: float = 0.0) -> np.ndarray:
    return np.concatenate((np.asarray(c, dtype=np.float64).ravel(), [tg, elapsed]))


def mixed_direction(grid: ind.IndependentNoQkeGrid, tg: float) -> np.ndarray:
    y = grid.nodes
    spectral = np.stack(
        [
            0.11 * np.exp(-y / 4.0) * np.cos(0.7 * y),
            -0.07 * np.exp(-y / 5.0) * np.sin(0.5 * y + 0.2),
            0.05 * np.exp(-y / 6.0) * np.cos(0.3 * y + 0.4),
        ]
    )
    return np.concatenate((spectral.ravel(), [0.025 * tg, 2.0]))


def second_direction(grid: ind.IndependentNoQkeGrid, tg: float) -> np.ndarray:
    y = grid.nodes
    spectral = np.stack(
        [
            -0.035 * np.exp(-y / 7.0),
            0.027 * np.exp(-y / 4.5) * np.cos(y),
            0.019 * np.exp(-y / 5.5) * np.sin(0.4 * y),
        ]
    )
    return np.concatenate((spectral.ravel(), [-0.013 * tg, -5.0]))


def centered_directional_rhs(
    *,
    grid: ind.IndependentNoQkeGrid,
    c: np.ndarray,
    tcm: float,
    tg: float,
    direction: np.ndarray,
    epsilon: float,
    config: ind.IndependentCollisionConfig,
    elapsed: float = 0.0,
) -> tuple[np.ndarray, bool]:
    state = packed(c, tg, elapsed)
    plus_state = state + epsilon * direction
    minus_state = state - epsilon * direction
    if not c_only_state_validator(grid, plus_state):
        raise RuntimeError("positive directional perturbation left the state domain")
    if not c_only_state_validator(grid, minus_state):
        raise RuntimeError("negative directional perturbation left the state domain")

    base_signature = electron_tgamma_branch_signature(
        grid=grid,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    branch_ok = all(
        electron_tgamma_branch_signature(
            grid=grid,
            temperature_cm_mev=tcm,
            temperature_gamma_mev=float(perturbed),
            config=config,
        )
        == base_signature
        for perturbed in (plus_state[-2], minus_state[-2])
    )
    plus = evaluate_static_rhs_from_packed_state(
        grid=grid,
        packed_state=plus_state,
        temperature_cm_mev=tcm,
        config=config,
    )
    minus = evaluate_static_rhs_from_packed_state(
        grid=grid,
        packed_state=minus_state,
        temperature_cm_mev=tcm,
        config=config,
    )
    return (plus - minus) / (2.0 * epsilon), branch_ok


def directional_ladder(
    *,
    grid: ind.IndependentNoQkeGrid,
    c: np.ndarray,
    tcm: float,
    tg: float,
    direction: np.ndarray,
    analytic: np.ndarray,
    epsilons: list[float],
    config: ind.IndependentCollisionConfig,
    elapsed: float = 0.0,
) -> DirectionalLadder:
    total_values: list[float] = []
    spectral_values: list[float] = []
    temperature_values: list[float] = []
    elapsed_values: list[float] = []
    branches: list[bool] = []
    centered_values: list[np.ndarray] = []
    for epsilon in epsilons:
        centered, branch_ok = centered_directional_rhs(
            grid=grid,
            c=c,
            tcm=tcm,
            tg=tg,
            direction=direction,
            epsilon=epsilon,
            config=config,
            elapsed=elapsed,
        )
        total, spectral, temperature, elapsed_residual = block_residuals(
            analytic, centered, grid.order
        )
        total_values.append(total)
        spectral_values.append(spectral)
        temperature_values.append(temperature)
        elapsed_values.append(elapsed_residual)
        branches.append(branch_ok)
        centered_values.append(centered)
    best_index = int(np.argmin(total_values))
    valid = np.asarray(total_values, dtype=np.float64) > 0.0
    if np.count_nonzero(valid) >= 2:
        slope = float(
            np.polyfit(
                np.log(np.asarray(epsilons, dtype=np.float64)[valid]),
                np.log(np.asarray(total_values, dtype=np.float64)[valid]),
                1,
            )[0]
        )
    else:
        slope = float("nan")
    return DirectionalLadder(
        epsilons=epsilons,
        block_residuals=total_values,
        spectral_residuals=spectral_values,
        temperature_residuals=temperature_values,
        elapsed_residuals=elapsed_values,
        branch_flags=branches,
        best_index=best_index,
        best_centered=centered_values[best_index],
        convergence_slope=slope,
    )


def ladder_record(ladder: DirectionalLadder) -> dict[str, Any]:
    return {
        "epsilon_ladder": ladder.epsilons,
        "block_residuals": ladder.block_residuals,
        "spectral_residuals": ladder.spectral_residuals,
        "temperature_residuals": ladder.temperature_residuals,
        "elapsed_residuals": ladder.elapsed_residuals,
        "best_index": ladder.best_index,
        "best_block_residual": min(ladder.block_residuals),
        "all_samples_same_branch": all(ladder.branch_flags),
        "convergence_slope": ladder.convergence_slope,
    }


def matrix_record(result: StaticJacobianResult) -> dict[str, Any]:
    size = result.state_size
    gamma = 1.0e-3
    newton = static_newton_matrix(result.jacobian, gamma)
    active_newton = np.eye(size - 1) - gamma * result.active_jacobian
    return {
        "state_size": size,
        "spectral_size": result.spectral_size,
        "matrix_shape": list(result.jacobian.shape),
        "active_matrix_shape": list(result.active_jacobian.shape),
        "base_reconstruction_residual": result.base_reconstruction_residual,
        "column_assembly_residual": result.column_assembly_residual,
        "elapsed_column_norm": float(np.linalg.norm(result.elapsed_time_column)),
        "elapsed_null_action_norm": float(
            np.linalg.norm(result.jacobian[:, -1])
        ),
        "newton_active_block_residual": relative(
            newton[:-1, :-1], active_newton
        ),
        "newton_final_column_residual": relative(
            newton[:, -1], np.eye(size)[:, -1]
        ),
        "branch_signature": result.branch_signature,
        "rank_statement": (
            "structural nullity at least one from the exact passive elapsed-time "
            "column; no raw-unit numerical condition-number claim"
        ),
    }


def mutations(
    result: StaticJacobianResult,
    direction: np.ndarray,
    witness: np.ndarray,
    order: int,
) -> dict[str, float]:
    transpose = result.jacobian.T.copy()
    swap_flavour = result.jacobian.copy()
    swap_flavour[:, [0, order]] = swap_flavour[:, [order, 0]]
    omit_tgamma = result.jacobian.copy()
    omit_tgamma[:, -2] = 0.0
    flip_tgamma = result.jacobian.copy()
    flip_tgamma[:, -2] *= -1.0
    nonzero_elapsed = result.jacobian.copy()
    nonzero_elapsed[:, -1] = result.tgamma_column
    swap_input = result.jacobian.copy()
    swap_input[:, [-2, -1]] = swap_input[:, [-1, -2]]
    swap_output = result.jacobian.copy()
    swap_output[[-2, -1], :] = swap_output[[-1, -2], :]
    matrices = {
        "transpose": transpose,
        "swap electron/muon column": swap_flavour,
        "omit T_gamma column": omit_tgamma,
        "flip T_gamma column": flip_tgamma,
        "inject elapsed column": nonzero_elapsed,
        "swap T_gamma/elapsed columns": swap_input,
        "swap T_gamma/elapsed rows": swap_output,
    }
    return {
        name: rhs_block_relative(value @ direction, witness, order)
        for name, value in matrices.items()
    }


def line_plot(
    path: Path,
    x: list[float],
    series: dict[str, list[float]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    figure = plt.figure(figsize=(6.8, 4.5), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    markers = ("o", "s", "^", "D", "v")
    for marker, (label, values) in zip(markers, series.items()):
        axis.loglog(x, values, marker=marker, label=label)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def horizontal_bar_plot(
    path: Path,
    values: dict[str, float],
    *,
    xlabel: str,
    title: str,
) -> None:
    figure = plt.figure(figsize=(7.6, 5.0), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    labels = list(values)
    data = list(values.values())
    axis.barh(labels, data)
    axis.set_xscale("log")
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.invert_yaxis()
    axis.grid(True, axis="x", which="both", alpha=0.3)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def structure_plot(path: Path, result: StaticJacobianResult) -> None:
    figure = plt.figure(figsize=(6.4, 5.4), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    structure = np.asarray(result.jacobian != 0.0, dtype=np.float64)
    image = axis.imshow(structure, origin="upper", aspect="auto", interpolation="nearest")
    for boundary in (result.spectral_size, result.spectral_size + 1):
        axis.axvline(boundary - 0.5, linewidth=0.8)
        axis.axhline(boundary - 0.5, linewidth=0.8)
    axis.set_xlabel("input-state column")
    axis.set_ylabel("RHS output row")
    axis.set_title("Exact nonzero pattern of the static Jacobian")
    figure.colorbar(image, ax=axis, label="nonzero indicator")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def normalized_column_profile(path: Path, result: StaticJacobianResult) -> None:
    spectral_size = result.spectral_size
    profiles = {
        "spectral-output norm": np.linalg.norm(
            result.jacobian[:spectral_size, :], axis=0
        ),
        "T_gamma-output magnitude": np.abs(result.jacobian[-2, :]),
        "elapsed-output magnitude": np.abs(result.jacobian[-1, :]),
    }
    figure = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    x = np.arange(result.state_size)
    for label, values in profiles.items():
        scale = max(float(np.max(values)), np.finfo(float).tiny)
        axis.plot(x, values / scale, marker="o", markersize=3, label=label)
    for boundary in (result.spectral_size, result.spectral_size + 1):
        axis.axvline(boundary - 0.5, linewidth=0.8)
    axis.set_xlabel("input-state column")
    axis.set_ylabel("within-output-block normalized magnitude")
    axis.set_title("Dimension-aware Jacobian column profiles")
    axis.set_ylim(-0.03, 1.05)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_summary(path: Path, receipt: dict[str, Any]) -> None:
    thermal = receipt["thermal"]
    weak = receipt["weak_tail"]
    retained = receipt.get("retained_stiff")
    lines = [
        "# D-080D deterministic square static-Jacobian probe",
        "",
        f"- classification: `{receipt['classification']}`",
        f"- explicit thermal matrix: `{thermal['matrix']['matrix_shape']}`",
        f"- thermal direction-1 best residual: `{thermal['direction_1']['best_block_residual']:.16e}`",
        f"- thermal direction-2 best residual: `{thermal['direction_2']['best_block_residual']:.16e}`",
        f"- weak-tail best residual: `{weak['direction']['best_block_residual']:.16e}`",
        f"- exact elapsed-column norm: `{thermal['matrix']['elapsed_column_norm']:.16e}`",
        f"- maximum matrix-vs-direct-JVP residual: `{receipt['maximum_matrix_action_residual']:.16e}`",
        "",
        "Residuals normalize spectral, photon-temperature, and elapsed-output blocks separately.",
        "The full matrix is structurally singular because stored elapsed time is passive;",
        "the associated BDF matrix identity is symbolic/static evidence only, not a solver claim.",
    ]
    if retained is not None:
        lines.extend(
            [
                "",
                f"- retained-state directional residual: `{retained['block_residual']:.16e}`",
                "- retained-state scope: directional only; no order-60 explicit matrix was assembled.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--retained-state")
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    config = collision_config()
    thermal_grid, thermal_c, thermal_tcm, thermal_tg = thermal_case()
    equilibrium_grid, equilibrium_c, equilibrium_tcm, equilibrium_tg = equilibrium_case()
    weak_grid, weak_c, weak_tcm, weak_tg = weak_tail_case()

    thermal_matrix = assemble_static_jacobian(
        grid=thermal_grid,
        pair_cloglog=thermal_c,
        temperature_cm_mev=thermal_tcm,
        temperature_gamma_mev=thermal_tg,
        config=config,
    )
    equilibrium_matrix = assemble_static_jacobian(
        grid=equilibrium_grid,
        pair_cloglog=equilibrium_c,
        temperature_cm_mev=equilibrium_tcm,
        temperature_gamma_mev=equilibrium_tg,
        config=config,
    )
    weak_matrix = assemble_static_jacobian(
        grid=weak_grid,
        pair_cloglog=weak_c,
        temperature_cm_mev=weak_tcm,
        temperature_gamma_mev=weak_tg,
        config=config,
    )

    d1 = mixed_direction(thermal_grid, thermal_tg)
    d2 = second_direction(thermal_grid, thermal_tg)
    direct_d1 = evaluate_static_rhs_direction_jvp(
        grid=thermal_grid,
        pair_cloglog=thermal_c,
        full_direction=d1,
        temperature_cm_mev=thermal_tcm,
        temperature_gamma_mev=thermal_tg,
        config=config,
    )
    direct_d2 = evaluate_static_rhs_direction_jvp(
        grid=thermal_grid,
        pair_cloglog=thermal_c,
        full_direction=d2,
        temperature_cm_mev=thermal_tcm,
        temperature_gamma_mev=thermal_tg,
        config=config,
    )
    matrix_action_residuals = {
        "direction_1": rhs_block_relative(
            thermal_matrix.jacobian @ d1, direct_d1.jvp, thermal_grid.order
        ),
        "direction_2": rhs_block_relative(
            thermal_matrix.jacobian @ d2, direct_d2.jvp, thermal_grid.order
        ),
    }

    thermal_eps = [3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4]
    thermal_ladder_1 = directional_ladder(
        grid=thermal_grid,
        c=thermal_c,
        tcm=thermal_tcm,
        tg=thermal_tg,
        direction=d1,
        analytic=thermal_matrix.jacobian @ d1,
        epsilons=thermal_eps,
        config=config,
        elapsed=11.0,
    )
    thermal_ladder_2 = directional_ladder(
        grid=thermal_grid,
        c=thermal_c,
        tcm=thermal_tcm,
        tg=thermal_tg,
        direction=d2,
        analytic=thermal_matrix.jacobian @ d2,
        epsilons=thermal_eps,
        config=config,
        elapsed=-7.0,
    )

    equilibrium_direction = second_direction(equilibrium_grid, equilibrium_tg)
    equilibrium_direct = evaluate_static_rhs_direction_jvp(
        grid=equilibrium_grid,
        pair_cloglog=equilibrium_c,
        full_direction=equilibrium_direction,
        temperature_cm_mev=equilibrium_tcm,
        temperature_gamma_mev=equilibrium_tg,
        config=config,
    )
    equilibrium_ladder = directional_ladder(
        grid=equilibrium_grid,
        c=equilibrium_c,
        tcm=equilibrium_tcm,
        tg=equilibrium_tg,
        direction=equilibrium_direction,
        analytic=equilibrium_matrix.jacobian @ equilibrium_direction,
        epsilons=[1.0e-3, 3.0e-4, 1.0e-4],
        config=config,
    )
    equilibrium_action_residual = rhs_block_relative(
        equilibrium_matrix.jacobian @ equilibrium_direction,
        equilibrium_direct.jvp,
        equilibrium_grid.order,
    )

    weak_direction = second_direction(weak_grid, weak_tg)
    weak_direct = evaluate_static_rhs_direction_jvp(
        grid=weak_grid,
        pair_cloglog=weak_c,
        full_direction=weak_direction,
        temperature_cm_mev=weak_tcm,
        temperature_gamma_mev=weak_tg,
        config=config,
    )
    weak_ladder = directional_ladder(
        grid=weak_grid,
        c=weak_c,
        tcm=weak_tcm,
        tg=weak_tg,
        direction=weak_direction,
        analytic=weak_matrix.jacobian @ weak_direction,
        epsilons=[1.0e-3, 3.0e-4, 1.0e-4],
        config=config,
    )
    weak_action_residual = rhs_block_relative(
        weak_matrix.jacobian @ weak_direction,
        weak_direct.jvp,
        weak_grid.order,
    )

    mutation_residuals = mutations(
        thermal_matrix,
        d1,
        thermal_ladder_1.best_centered,
        thermal_grid.order,
    )

    receipt: dict[str, Any] = {
        "schema": "rabbit.d080d.square_static_jacobian.v1",
        "classification": "EXPLICIT_SQUARE_STATIC_JACOBIAN",
        "comparator_blob_sha": git_blob_sha(Path(ind.__file__)),
        "expected_comparator_blob_sha": EXPECTED_COMPARATOR_BLOB_SHA,
        "expected_d079_rhs_blob_sha": EXPECTED_D079_RHS_BLOB_SHA,
        "expected_d080c_rhs_blob_sha": EXPECTED_D080C_RHS_BLOB_SHA,
        "state_ordering": list(thermal_matrix.layout.state_labels),
        "state_dimensions": list(thermal_matrix.layout.state_dimensions),
        "rhs_dimensions": list(thermal_matrix.layout.rhs_dimensions),
        "metric_contract": (
            "maximum of independently normalized spectral, photon-temperature, "
            "and elapsed-output residuals; heterogeneous units are never mixed"
        ),
        "thermal": {
            "order": thermal_grid.order,
            "y_max": thermal_grid.y_max,
            "temperature_cm_mev": thermal_tcm,
            "temperature_gamma_mev": thermal_tg,
            "matrix": matrix_record(thermal_matrix),
            "direction_1": ladder_record(thermal_ladder_1),
            "direction_2": ladder_record(thermal_ladder_2),
        },
        "equilibrium": {
            "order": equilibrium_grid.order,
            "matrix": matrix_record(equilibrium_matrix),
            "matrix_action_residual": equilibrium_action_residual,
            "direction": ladder_record(equilibrium_ladder),
            "dQ_nu_dTgamma": (
                equilibrium_matrix.tgamma.collision.neutrino_energy_transfer
            ),
            "dQ_em_dTgamma": (
                equilibrium_matrix.tgamma.collision.electron_bath_energy_transfer
            ),
            "first_law_tangent_residual": (
                equilibrium_matrix.tgamma.collision.first_law_tangent_residual
            ),
        },
        "weak_tail": {
            "status": "controlled static probe; not retained trajectory evidence",
            "order": weak_grid.order,
            "y_max": weak_grid.y_max,
            "temperature_cm_mev": weak_tcm,
            "temperature_gamma_mev": weak_tg,
            "matrix": matrix_record(weak_matrix),
            "matrix_action_residual": weak_action_residual,
            "direction": ladder_record(weak_ladder),
        },
        "matrix_action_residuals": matrix_action_residuals,
        "maximum_matrix_action_residual": max(
            *matrix_action_residuals.values(),
            equilibrium_action_residual,
            weak_action_residual,
        ),
        "mutation_block_residuals": mutation_residuals,
        "symbolic_structure": {
            "jacobian_form": "[[J_active,0],[accumulator_row,0]]",
            "right_null_vector": "elapsed-time basis vector",
            "newton_identity": (
                "det(I-gamma J)=det(I-gamma J_active); verified separately "
                "with stateless Wolfram Language"
            ),
        },
        "claim_ceiling": (
            "full explicit fixed-support static square Jacobian at the tested "
            "order-8 thermal/equilibrium/weak regimes, plus an optional retained "
            "order-60 directional discriminator; no support-crossing derivative, "
            "integrator, BDF/JFNK/Newton behaviour, trajectory completion, speedup, "
            "endpoint, N_eff, gate, release, or publication claim"
        ),
    }

    if args.retained_state:
        retained_path = Path(args.retained_state)
        with np.load(retained_path, allow_pickle=False) as archive:
            state = np.asarray(archive["y"], dtype=np.float64)
        retained_grid = ind.build_independent_grid(60, 30.0)
        retained_config = collision_config(retained=True)
        retained_c = state[:180].reshape(3, 60)
        retained_tg = float(state[180])
        retained_tcm = 10.0 * np.exp(-0.16286930247517223)
        retained_direction = second_direction(retained_grid, retained_tg)
        retained_direct = evaluate_static_rhs_direction_jvp(
            grid=retained_grid,
            pair_cloglog=retained_c,
            full_direction=retained_direction,
            temperature_cm_mev=retained_tcm,
            temperature_gamma_mev=retained_tg,
            config=retained_config,
        )
        retained_centered, retained_branch = centered_directional_rhs(
            grid=retained_grid,
            c=retained_c,
            tcm=retained_tcm,
            tg=retained_tg,
            direction=retained_direction,
            epsilon=2.0e-4,
            config=retained_config,
            elapsed=float(state[-1]),
        )
        total, spectral, temperature, elapsed = block_residuals(
            retained_direct.jvp,
            retained_centered,
            retained_grid.order,
        )
        receipt["retained_stiff"] = {
            "scope": "combined directional operator only; explicit order-60 matrix not assembled",
            "fixture_sha256": sha256(retained_path),
            "order": 60,
            "y_max": 30.0,
            "epsilon": 2.0e-4,
            "same_tgamma_branch": retained_branch,
            "block_residual": total,
            "spectral_residual": spectral,
            "temperature_residual": temperature,
            "elapsed_residual": elapsed,
        }

    if receipt["comparator_blob_sha"] != EXPECTED_COMPARATOR_BLOB_SHA:
        raise RuntimeError("comparator Git-blob identity changed")

    np.save(output / "thermal_static_jacobian.npy", thermal_matrix.jacobian)
    np.save(output / "thermal_base_rhs.npy", thermal_matrix.base_rhs)
    np.save(output / "thermal_probe_directions.npy", np.stack((d1, d2)))
    (output / "research_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(output / "probe_summary.md", receipt)

    line_plot(
        output / "thermal_directional_residual_ladders.png",
        thermal_eps,
        {
            "mixed direction 1": thermal_ladder_1.block_residuals,
            "mixed direction 2": thermal_ladder_2.block_residuals,
        },
        xlabel=r"directional step $\epsilon$",
        ylabel="dimension-aware RHS residual",
        title="Full square-Jacobian directional ladders",
    )
    line_plot(
        output / "thermal_block_residuals.png",
        thermal_eps,
        {
            "spectral": thermal_ladder_1.spectral_residuals,
            "T_gamma": thermal_ladder_1.temperature_residuals,
            "elapsed output": thermal_ladder_1.elapsed_residuals,
        },
        xlabel=r"directional step $\epsilon$",
        ylabel="separately normalized residual",
        title="Direction 1: heterogeneous output blocks",
    )
    line_plot(
        output / "regime_directional_residuals.png",
        [1.0e-3, 3.0e-4, 1.0e-4],
        {
            "equilibrium": equilibrium_ladder.block_residuals,
            "weak-tail": weak_ladder.block_residuals,
        },
        xlabel=r"directional step $\epsilon$",
        ylabel="dimension-aware RHS residual",
        title="Equilibrium and weak-collision static regimes",
    )
    horizontal_bar_plot(
        output / "mutation_kills.png",
        mutation_residuals,
        xlabel="residual against original-RHS witness",
        title="Adversarial square-Jacobian mutations",
    )
    structure_plot(output / "jacobian_structure.png", thermal_matrix)
    normalized_column_profile(output / "column_profiles.png", thermal_matrix)

    files = sorted(path for path in output.iterdir() if path.is_file())
    checksums = "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    (output / "SHA256SUMS").write_text(checksums, encoding="utf-8")


if __name__ == "__main__":
    main()
