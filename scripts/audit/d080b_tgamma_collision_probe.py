#!/usr/bin/env python3
"""Generate deterministic D-080B collision-column evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d080b_tgamma_collision import (
    EXPECTED_COMPARATOR_BLOB_SHA,
    electron_tgamma_branch_signature,
    evaluate_tgamma_collision_action_jvp,
)


def relative(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    scale = max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny)
    return float(np.linalg.norm(a - b) / scale)


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    framed = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    return hashlib.sha1(framed).hexdigest()


def setup_case() -> tuple[
    ind.IndependentNoQkeGrid,
    ind.IndependentCollisionConfig,
    np.ndarray,
]:
    grid = ind.build_independent_grid(8, 8.0)
    config = ind.IndependentCollisionConfig(
        incoming_polar_order=2,
        final_polar_order=2,
        final_azimuth_order=4,
        electron_radial_order=8,
    )
    logits = np.stack(
        [
            -grid.nodes + 0.04 * np.exp(-grid.nodes / 3.0),
            -grid.nodes - 0.02 * np.exp(-grid.nodes / 4.0),
            -grid.nodes - 0.02 * np.exp(-grid.nodes / 4.0),
        ]
    )
    return grid, config, ind.pair_logits_to_cloglog(logits)


def centered_action(
    *,
    grid: ind.IndependentNoQkeGrid,
    config: ind.IndependentCollisionConfig,
    pair_cloglog: np.ndarray,
    tcm: float,
    tg: float,
    epsilon: float,
) -> tuple[np.ndarray, float, float]:
    plus = ind.evaluate_independent_collision_action(
        grid=grid,
        pair_cloglog=pair_cloglog,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg + epsilon,
        config=config,
    )
    minus = ind.evaluate_independent_collision_action(
        grid=grid,
        pair_cloglog=pair_cloglog,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg - epsilon,
        config=config,
    )
    action = (
        np.asarray(plus.total, dtype=np.float64)
        - np.asarray(minus.total, dtype=np.float64)
    ) / (2.0 * epsilon)
    qnu = (
        float(plus.diagnostics["event_neutrino_energy_transfer"])
        - float(minus.diagnostics["event_neutrino_energy_transfer"])
    ) / (2.0 * epsilon)
    qem = (
        float(plus.electron_bath_energy_transfer)
        - float(minus.electron_bath_energy_transfer)
    ) / (2.0 * epsilon)
    return action, qnu, qem


def line_plot(
    path: Path,
    x: list[float],
    series: dict[str, list[float]],
    *,
    ylabel: str,
    title: str,
) -> None:
    figure = plt.figure(figsize=(6.4, 4.2), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    for label, values in series.items():
        axis.loglog(x, values, marker="o", label=label)
    axis.set_xlabel(r"$\epsilon_{T_\gamma}$ [MeV]")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def energy_plot(
    path: Path,
    epsilons: list[float],
    analytic_qnu: float,
    analytic_qem: float,
    centered_qnu: list[float],
    centered_qem: list[float],
) -> None:
    figure = plt.figure(figsize=(6.4, 4.2), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    axis.semilogx(epsilons, centered_qnu, marker="o", label="centered dQ_nu/dT")
    axis.semilogx(epsilons, centered_qem, marker="s", label="centered dQ_em/dT")
    axis.axhline(analytic_qnu, linestyle="--", label="analytic dQ_nu/dT")
    axis.axhline(analytic_qem, linestyle=":", label="analytic dQ_em/dT")
    axis.set_xlabel(r"$\epsilon_{T_\gamma}$ [MeV]")
    axis.set_ylabel("energy-transfer tangent")
    axis.set_title("Differentiated neutrino/electromagnetic exchange")
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
    figure = plt.figure(figsize=(6.8, 4.2), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(list(values), list(values.values()))
    axis.set_yscale("log")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=22)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    grid, config, thermal_c = setup_case()
    tcm, tg = 2.0, 2.05
    thermal = evaluate_tgamma_collision_action_jvp(
        grid=grid,
        pair_cloglog=thermal_c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    equilibrium_c = ind.pair_logits_to_cloglog(
        np.stack([-grid.nodes for _ in range(3)])
    )
    equilibrium = evaluate_tgamma_collision_action_jvp(
        grid=grid,
        pair_cloglog=equilibrium_c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tcm,
        config=config,
    )

    epsilons = [1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4]
    action_residuals: list[float] = []
    energy_residuals: list[float] = []
    centered_qnu: list[float] = []
    centered_qem: list[float] = []
    branch_flags: list[bool] = []
    best_centered: np.ndarray | None = None
    best_residual = float("inf")

    for epsilon in epsilons:
        plus_signature = electron_tgamma_branch_signature(
            grid=grid,
            temperature_cm_mev=tcm,
            temperature_gamma_mev=tg + epsilon,
            config=config,
        )
        minus_signature = electron_tgamma_branch_signature(
            grid=grid,
            temperature_cm_mev=tcm,
            temperature_gamma_mev=tg - epsilon,
            config=config,
        )
        branch_flags.append(
            plus_signature == thermal.branch_signature
            and minus_signature == thermal.branch_signature
        )
        centered, qnu, qem = centered_action(
            grid=grid,
            config=config,
            pair_cloglog=thermal_c,
            tcm=tcm,
            tg=tg,
            epsilon=epsilon,
        )
        action_residual = relative(thermal.total, centered)
        energy_residual = relative(
            np.array([
                thermal.neutrino_energy_transfer,
                thermal.electron_bath_energy_transfer,
            ]),
            np.array([qnu, qem]),
        )
        action_residuals.append(action_residual)
        energy_residuals.append(energy_residual)
        centered_qnu.append(qnu)
        centered_qem.append(qem)
        if action_residual < best_residual:
            best_residual = action_residual
            best_centered = centered

    if best_centered is None:
        raise RuntimeError("no centered witness was generated")

    mutations = {
        "flip-pauli": thermal.total - 2.0 * thermal.pauli,
        "omit-measure": thermal.total - thermal.measure,
        "omit-matrix": thermal.total - thermal.matrix,
        "omit-projection": thermal.total - thermal.projection,
        "omit-elastic": thermal.total - thermal.elastic,
        "omit-pair": thermal.total - thermal.pair,
    }
    mutation_residuals = {
        name: relative(value, best_centered)
        for name, value in mutations.items()
    }
    total_norm = max(np.linalg.norm(thermal.total), np.finfo(float).tiny)
    component_norms = {
        "measure": float(np.linalg.norm(thermal.measure) / total_norm),
        "matrix": float(np.linalg.norm(thermal.matrix) / total_norm),
        "Pauli": float(np.linalg.norm(thermal.pauli) / total_norm),
        "projection": float(np.linalg.norm(thermal.projection) / total_norm),
        "elastic": float(np.linalg.norm(thermal.elastic) / total_norm),
        "pair": float(np.linalg.norm(thermal.pair) / total_norm),
    }

    comparator_sha = git_blob_sha(Path(ind.__file__))
    receipt = {
        "schema": "rabbit.d080b.static_tgamma_collision.v1",
        "classification": "FULL_STATIC_TGAMMA_COLLISION_COLUMN",
        "comparator_blob_sha": comparator_sha,
        "expected_comparator_blob_sha": EXPECTED_COMPARATOR_BLOB_SHA,
        "case": {
            "order": grid.order,
            "y_max": grid.y_max,
            "temperature_cm_mev": tcm,
            "temperature_gamma_mev": tg,
        },
        "epsilon_ladder_mev": epsilons,
        "thermal_action_residuals": action_residuals,
        "thermal_energy_residuals": energy_residuals,
        "best_thermal_action_residual": min(action_residuals),
        "best_thermal_energy_residual": min(energy_residuals),
        "all_ladder_samples_same_branch": all(branch_flags),
        "thermal_branch_signature": thermal.branch_signature,
        "minimum_support_margin": thermal.minimum_support_margin,
        "minimum_lambda_margin": thermal.minimum_lambda_margin,
        "base_reconstruction_residual": thermal.base_reconstruction_residual,
        "component_sum_residual": thermal.component_sum_residual,
        "equilibrium_base_norm": float(np.linalg.norm(equilibrium.base.total)),
        "equilibrium_neutrino_energy_tangent": equilibrium.neutrino_energy_transfer,
        "equilibrium_electromagnetic_energy_tangent": equilibrium.electron_bath_energy_transfer,
        "maximum_first_law_tangent_residual": max(
            thermal.first_law_tangent_residual,
            equilibrium.first_law_tangent_residual,
        ),
        "thermal_cp_residual": thermal.charge_conjugation_residual,
        "thermal_mu_tau_residual": thermal.mu_tau_residual,
        "component_norms_over_total": component_norms,
        "mutation_residuals": mutation_residuals,
        "energy_tangent_by_component": {
            key: list(value)
            for key, value in thermal.energy_tangent_by_component.items()
        },
        "claim_ceiling": (
            "static fixed-support T_gamma collision-action column only; "
            "no full RHS, square Jacobian, integrator, trajectory, endpoint, "
            "performance, or gate claim"
        ),
    }
    if comparator_sha != EXPECTED_COMPARATOR_BLOB_SHA:
        raise RuntimeError("comparator Git-blob identity changed")

    (output / "research_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    line_plot(
        output / "thermal_residual_ladder.png",
        epsilons,
        {
            "collision action": action_residuals,
            "energy ledger": energy_residuals,
        },
        ylabel="scaled tangent residual",
        title=r"Full static $T_\gamma$ collision-column ladder",
    )
    energy_plot(
        output / "energy_transfer_ladder.png",
        epsilons,
        thermal.neutrino_energy_transfer,
        thermal.electron_bath_energy_transfer,
        centered_qnu,
        centered_qem,
    )
    bar_plot(
        output / "component_norms.png",
        component_norms,
        ylabel="component norm / total norm",
        title="Load-bearing analytic collision-column components",
    )
    bar_plot(
        output / "mutation_kills.png",
        mutation_residuals,
        ylabel="scaled residual against original comparator",
        title="D-080B collision-column mutation kills",
    )

    summary = f"""# D-080B full static T-gamma collision column

- classification: `{receipt['classification']}`
- comparator Git blob: `{comparator_sha}`
- best thermal action residual: `{min(action_residuals):.9e}`
- best thermal energy residual: `{min(energy_residuals):.9e}`
- all ladder samples same branch: `{all(branch_flags)}`
- equilibrium dQ_nu/dT_gamma: `{equilibrium.neutrino_energy_transfer:.9e}`
- equilibrium dQ_em/dT_gamma: `{equilibrium.electron_bath_energy_transfer:.9e}`
- maximum differentiated first-law residual: `{receipt['maximum_first_law_tangent_residual']:.9e}`
- minimum support margin: `{thermal.minimum_support_margin:.9e}`
- minimum Kallen margin: `{thermal.minimum_lambda_margin:.9e}`
- base reconstruction residual: `{thermal.base_reconstruction_residual:.9e}`
- component-sum residual: `{thermal.component_sum_residual:.9e}`

This result differentiates the frozen static electron collision action with
respect to `T_gamma`, including moving quadrature, kinematics, weak matrix
elements, Pauli blocking, moving output interpolation, pair annihilation,
flavour routing, and the neutrino/electromagnetic energy ledgers.  It does not
construct the full RHS column or call an integrator.
"""
    (output / "probe_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
