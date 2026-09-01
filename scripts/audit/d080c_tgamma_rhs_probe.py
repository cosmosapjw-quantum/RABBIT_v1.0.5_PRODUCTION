#!/usr/bin/env python3
"""Generate deterministic D-080C full-RHS-column evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_rhs_jvp import evaluate_static_rhs_from_packed_state
from scripts.audit._d080b_tgamma_collision import electron_tgamma_branch_signature
from scripts.audit._d080c_tgamma_rhs import (
    EXPECTED_COMPARATOR_BLOB_SHA,
    TgammaRhsColumnResult,
    evaluate_tgamma_rhs_column,
)


@dataclass(frozen=True)
class LadderResult:
    epsilons: list[float]
    block_residuals: list[float]
    spectral_residuals: list[float]
    temperature_residuals: list[float]
    elapsed_residuals: list[float]
    branch_flags: list[bool]
    best_centered: np.ndarray
    best_index: int


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
    size = 3 * order + 2
    if a.shape != (size,) or b.shape != (size,):
        raise ValueError("packed RHS vectors have an invalid shape")
    spectral = relative(a[: 3 * order], b[: 3 * order])
    temperature = scalar_relative(a[-2], b[-2])
    elapsed = scalar_relative(a[-1], b[-1])
    return max(spectral, temperature, elapsed), spectral, temperature, elapsed


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    framed = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    return hashlib.sha1(framed).hexdigest()


def config() -> ind.IndependentCollisionConfig:
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


def centered_rhs(
    *,
    grid: ind.IndependentNoQkeGrid,
    pair_cloglog: np.ndarray,
    tcm: float,
    tg: float,
    epsilon: float,
    collision_config: ind.IndependentCollisionConfig,
) -> np.ndarray:
    plus = evaluate_static_rhs_from_packed_state(
        grid=grid,
        packed_state=packed(pair_cloglog, tg + epsilon),
        temperature_cm_mev=tcm,
        config=collision_config,
    )
    minus = evaluate_static_rhs_from_packed_state(
        grid=grid,
        packed_state=packed(pair_cloglog, tg - epsilon),
        temperature_cm_mev=tcm,
        config=collision_config,
    )
    return (plus - minus) / (2.0 * epsilon)


def evaluate_ladder(
    *,
    grid: ind.IndependentNoQkeGrid,
    pair_cloglog: np.ndarray,
    tcm: float,
    tg: float,
    epsilons: list[float],
    collision_config: ind.IndependentCollisionConfig,
    analytic: TgammaRhsColumnResult,
) -> LadderResult:
    block_values: list[float] = []
    spectral_values: list[float] = []
    temperature_values: list[float] = []
    elapsed_values: list[float] = []
    branch_flags: list[bool] = []
    centered_values: list[np.ndarray] = []

    for epsilon in epsilons:
        plus_signature = electron_tgamma_branch_signature(
            grid=grid,
            temperature_cm_mev=tcm,
            temperature_gamma_mev=tg + epsilon,
            config=collision_config,
        )
        minus_signature = electron_tgamma_branch_signature(
            grid=grid,
            temperature_cm_mev=tcm,
            temperature_gamma_mev=tg - epsilon,
            config=collision_config,
        )
        branch_flags.append(
            plus_signature == analytic.branch_signature
            and minus_signature == analytic.branch_signature
        )
        centered = centered_rhs(
            grid=grid,
            pair_cloglog=pair_cloglog,
            tcm=tcm,
            tg=tg,
            epsilon=epsilon,
            collision_config=collision_config,
        )
        block, spectral, temperature, elapsed = block_residuals(
            analytic.tgamma_column,
            centered,
            grid.order,
        )
        centered_values.append(centered)
        block_values.append(block)
        spectral_values.append(spectral)
        temperature_values.append(temperature)
        elapsed_values.append(elapsed)

    best_index = int(np.argmin(block_values))
    return LadderResult(
        epsilons=epsilons,
        block_residuals=block_values,
        spectral_residuals=spectral_values,
        temperature_residuals=temperature_values,
        elapsed_residuals=elapsed_values,
        branch_flags=branch_flags,
        best_centered=centered_values[best_index],
        best_index=best_index,
    )


def line_plot(
    path: Path,
    x: list[float],
    series: dict[str, list[float]],
    *,
    ylabel: str,
    title: str,
) -> None:
    figure = plt.figure(figsize=(6.6, 4.4), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    markers = ("o", "s", "^", "D")
    for marker, (label, values) in zip(markers, series.items()):
        axis.loglog(x, values, marker=marker, label=label)
    axis.set_xlabel(r"$\epsilon_{T_\gamma}$ [MeV]")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def bar_plot(
    path: Path,
    values: dict[str, float],
    *,
    ylabel: str,
    title: str,
) -> None:
    figure = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(list(values), list(values.values()))
    axis.set_yscale("log")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def component_ratios(result: TgammaRhsColumnResult, order: int) -> dict[str, float]:
    spectral_scale = max(
        np.linalg.norm(result.tgamma_column[: 3 * order]),
        np.finfo(float).tiny,
    )
    temperature_scale = max(abs(float(result.tgamma_column[-2])), np.finfo(float).tiny)
    elapsed_scale = max(abs(float(result.tgamma_column[-1])), np.finfo(float).tiny)
    return {
        "spectral collision": float(
            np.linalg.norm(result.spectral_collision_component[: 3 * order])
            / spectral_scale
        ),
        "spectral Hubble": float(
            np.linalg.norm(result.spectral_hubble_component[: 3 * order])
            / spectral_scale
        ),
        "T expansion/EOS": abs(float(result.temperature_expansion_component[-2]))
        / temperature_scale,
        "T collision": abs(float(result.temperature_collision_component[-2]))
        / temperature_scale,
        "T Hubble": abs(float(result.temperature_hubble_component[-2]))
        / temperature_scale,
        "T heat capacity": abs(float(result.heat_capacity_component[-2]))
        / temperature_scale,
        "time Hubble": abs(float(result.time_hubble_component[-1])) / elapsed_scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    collision_config = config()
    thermal_grid, thermal_c, thermal_tcm, thermal_tg = thermal_case()
    weak_grid, weak_c, weak_tcm, weak_tg = weak_tail_case()
    thermal = evaluate_tgamma_rhs_column(
        grid=thermal_grid,
        pair_cloglog=thermal_c,
        temperature_cm_mev=thermal_tcm,
        temperature_gamma_mev=thermal_tg,
        config=collision_config,
    )
    weak = evaluate_tgamma_rhs_column(
        grid=weak_grid,
        pair_cloglog=weak_c,
        temperature_cm_mev=weak_tcm,
        temperature_gamma_mev=weak_tg,
        config=collision_config,
    )
    equilibrium_c = ind.pair_logits_to_cloglog(
        np.stack([-thermal_grid.nodes for _ in range(3)])
    )
    equilibrium = evaluate_tgamma_rhs_column(
        grid=thermal_grid,
        pair_cloglog=equilibrium_c,
        temperature_cm_mev=thermal_tcm,
        temperature_gamma_mev=thermal_tcm,
        config=collision_config,
    )

    thermal_ladder = evaluate_ladder(
        grid=thermal_grid,
        pair_cloglog=thermal_c,
        tcm=thermal_tcm,
        tg=thermal_tg,
        epsilons=[1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4],
        collision_config=collision_config,
        analytic=thermal,
    )
    weak_ladder = evaluate_ladder(
        grid=weak_grid,
        pair_cloglog=weak_c,
        tcm=weak_tcm,
        tg=weak_tg,
        epsilons=[1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5],
        collision_config=collision_config,
        analytic=weak,
    )

    mutations = {
        "omit collision": thermal.tgamma_column - thermal.collision_component,
        "omit Hubble": thermal.tgamma_column - thermal.hubble_component,
        "omit heat capacity": thermal.tgamma_column - thermal.heat_capacity_component,
        "flip Q_em,T": (
            thermal.tgamma_column - 2.0 * thermal.temperature_collision_component
        ),
    }
    mutation_residuals = {
        name: block_residuals(
            value,
            thermal_ladder.best_centered,
            thermal_grid.order,
        )[0]
        for name, value in mutations.items()
    }
    ratios = component_ratios(thermal, thermal_grid.order)
    comparator_sha = git_blob_sha(Path(ind.__file__))
    if comparator_sha != EXPECTED_COMPARATOR_BLOB_SHA:
        raise RuntimeError("comparator Git-blob identity changed")

    receipt = {
        "schema": "rabbit.d080c.static_tgamma_rhs.v1",
        "classification": "FULL_STATIC_TGAMMA_RHS_COLUMN",
        "comparator_blob_sha": comparator_sha,
        "expected_comparator_blob_sha": EXPECTED_COMPARATOR_BLOB_SHA,
        "metric_contract": (
            "maximum of separately dimensionless spectral, photon-temperature, "
            "and elapsed-output residuals; heterogeneous units are never mixed"
        ),
        "thermal_case": {
            "order": thermal_grid.order,
            "y_max": thermal_grid.y_max,
            "temperature_cm_mev": thermal_tcm,
            "temperature_gamma_mev": thermal_tg,
            "epsilon_ladder_mev": thermal_ladder.epsilons,
            "block_residuals": thermal_ladder.block_residuals,
            "spectral_residuals": thermal_ladder.spectral_residuals,
            "temperature_residuals": thermal_ladder.temperature_residuals,
            "elapsed_residuals": thermal_ladder.elapsed_residuals,
            "best_index": thermal_ladder.best_index,
            "best_block_residual": min(thermal_ladder.block_residuals),
            "all_samples_same_branch": all(thermal_ladder.branch_flags),
            "branch_signature": thermal.branch_signature,
            "base_reconstruction_residual": thermal.base_reconstruction_residual,
            "component_sum_residual": thermal.component_sum_residual,
            "hubble_log_tangent_per_mev": thermal.delta_hubble_over_hubble,
        },
        "manufactured_weak_tail_case": {
            "status": "controlled static probe; not retained trajectory evidence",
            "order": weak_grid.order,
            "y_max": weak_grid.y_max,
            "temperature_cm_mev": weak_tcm,
            "temperature_gamma_mev": weak_tg,
            "epsilon_ladder_mev": weak_ladder.epsilons,
            "block_residuals": weak_ladder.block_residuals,
            "spectral_residuals": weak_ladder.spectral_residuals,
            "temperature_residuals": weak_ladder.temperature_residuals,
            "elapsed_residuals": weak_ladder.elapsed_residuals,
            "best_index": weak_ladder.best_index,
            "best_block_residual": min(weak_ladder.block_residuals),
            "all_samples_same_branch": all(weak_ladder.branch_flags),
            "branch_signature": weak.branch_signature,
        },
        "equilibrium": {
            "neutrino_energy_tangent": equilibrium.collision.neutrino_energy_transfer,
            "electromagnetic_energy_tangent": (
                equilibrium.collision.electron_bath_energy_transfer
            ),
            "first_law_tangent_residual": (
                equilibrium.collision.first_law_tangent_residual
            ),
            "hubble_log_tangent_per_mev": equilibrium.delta_hubble_over_hubble,
            "elapsed_output_tangent": float(equilibrium.tgamma_column[-1]),
        },
        "elapsed_time_input_column_norm": float(
            np.linalg.norm(thermal.elapsed_time_input_column)
        ),
        "component_ratios_within_native_output_block": ratios,
        "mutation_block_residuals": mutation_residuals,
        "claim_ceiling": (
            "full static fixed-support original-RHS T_gamma input column only; "
            "no square Jacobian, integrator, trajectory, endpoint, performance, "
            "N_eff, gate, release, or publication claim"
        ),
    }
    (output / "research_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    line_plot(
        output / "rhs_residual_ladders.png",
        thermal_ladder.epsilons,
        {
            "thermal split": thermal_ladder.block_residuals,
            "quadratic guide": [
                thermal_ladder.block_residuals[0]
                * (epsilon / thermal_ladder.epsilons[0]) ** 2
                for epsilon in thermal_ladder.epsilons
            ],
        },
        ylabel="block-scaled tangent residual",
        title=r"Original packed-RHS $T_\gamma$ derivative ladder",
    )
    line_plot(
        output / "rowwise_residual_ladder.png",
        thermal_ladder.epsilons,
        {
            "spectral rows": thermal_ladder.spectral_residuals,
            "photon-temperature row": thermal_ladder.temperature_residuals,
            "elapsed-time output row": thermal_ladder.elapsed_residuals,
        },
        ylabel="dimensionless row/block residual",
        title="Dimension-aware thermal-split residuals",
    )
    line_plot(
        output / "weak_tail_residual_ladder.png",
        weak_ladder.epsilons,
        {
            "manufactured weak tail": weak_ladder.block_residuals,
            "quadratic guide": [
                weak_ladder.block_residuals[0]
                * (epsilon / weak_ladder.epsilons[0]) ** 2
                for epsilon in weak_ladder.epsilons
            ],
        },
        ylabel="block-scaled tangent residual",
        title="Manufactured low-temperature weak-collision probe",
    )
    bar_plot(
        output / "component_ratios.png",
        ratios,
        ylabel="absolute component / native-block total",
        title="Load-bearing D-080C RHS-column components",
    )
    bar_plot(
        output / "mutation_kills.png",
        mutation_residuals,
        ylabel="block-scaled residual against original RHS",
        title="Adversarial omission/sign mutations",
    )

    summary = f"""# D-080C deterministic static probe

- classification: `FULL_STATIC_TGAMMA_RHS_COLUMN`
- comparator blob: `{comparator_sha}`
- thermal best block residual: `{min(thermal_ladder.block_residuals):.16e}`
- weak-tail best block residual: `{min(weak_ladder.block_residuals):.16e}`
- thermal same branch: `{all(thermal_ladder.branch_flags)}`
- weak-tail same branch: `{all(weak_ladder.branch_flags)}`
- equilibrium dQ_nu/dT_gamma: `{equilibrium.collision.neutrino_energy_transfer:.16e}`
- equilibrium dQ_em/dT_gamma: `{equilibrium.collision.electron_bath_energy_transfer:.16e}`
- elapsed-time input column norm: `{np.linalg.norm(thermal.elapsed_time_input_column):.16e}`

The residual metric is blockwise and dimension-aware.  It does not combine the
MeV^-1 spectral rows, dimensionless photon-temperature row, and MeV^-2 elapsed
output row in one dimensional Euclidean norm.

The manufactured weak-tail state is a controlled static probe, not retained
trajectory evidence.  The claim remains limited to the fixed-support static
original-RHS input column.
"""
    (output / "probe_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
