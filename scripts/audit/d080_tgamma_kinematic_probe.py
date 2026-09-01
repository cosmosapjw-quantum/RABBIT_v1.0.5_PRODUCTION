#!/usr/bin/env python3
"""Generate deterministic D-080A moving-kinematics/EOS evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d080_tgamma_primitives import (
    EXPECTED_COMPARATOR_BLOB_SHA,
    electromagnetic_eos_tgamma_tangent,
    evaluate_elastic_tgamma_kinematic_tangent,
)


def relative(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    scale = max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny)
    return float(np.linalg.norm(a - b) / scale)


def batch_vector(batch: object) -> np.ndarray:
    names = (
        "p2", "e2", "e3", "e4", "p3_magnitude", "p4_magnitude",
        "phase_space", "quadrature_weight", "d12", "d13", "d14",
        "d23", "d24", "d34",
    )
    return np.concatenate([
        np.asarray(getattr(batch, name), dtype=np.float64).ravel()
        for name in names
    ])


def tangent_vector(result: object) -> np.ndarray:
    names = (
        "d_p2", "d_e2", "d_e3", "d_e4", "d_p3_magnitude",
        "d_p4_magnitude", "d_phase_space", "d_quadrature_weight",
        "d_d12", "d_d13", "d_d14", "d_d23", "d_d24", "d_d34",
    )
    return np.concatenate([
        np.asarray(getattr(result, name), dtype=np.float64).ravel()
        for name in names
    ])


def primal_batch(
    *,
    p1: float,
    temperature: float,
    mass: float,
    config: ind.IndependentCollisionConfig,
) -> object:
    nodes, weights = ind._electron_half_line_rule(
        config.electron_radial_order, temperature
    )
    return ind._two_body_kinematics(
        p1=p1,
        p2_nodes=nodes,
        p2_weights=weights,
        mass2=mass,
        mass3=0.0,
        mass4=mass,
        config=config,
    )


def comparator_blob_sha() -> str:
    """Return the Git blob object ID, not the raw-file SHA-1."""

    payload = Path(ind.__file__).read_bytes()
    framed = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    return hashlib.sha1(framed).hexdigest()


def save_line_plot(
    path: Path,
    x: list[float],
    y: list[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    figure = plt.figure(figsize=(6.4, 4.2), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    axis.loglog(x, y, marker="o")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.3)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_bar_plot(
    path: Path,
    labels: list[str],
    values: list[float],
    *,
    ylabel: str,
    title: str,
) -> None:
    figure = plt.figure(figsize=(6.4, 4.2), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(labels, values)
    axis.set_yscale("log")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=20)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    arguments = parser.parse_args()
    output = Path(arguments.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    temperature = 2.05
    mass = ind.M_ELECTRON_MEV
    p1 = 3.2
    config = ind.IndependentCollisionConfig(
        incoming_polar_order=2,
        final_polar_order=2,
        final_azimuth_order=4,
        electron_radial_order=8,
    )
    epsilons = [1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4]

    tangent = evaluate_elastic_tgamma_kinematic_tangent(
        p1=p1,
        temperature_gamma_mev=temperature,
        electron_mass_mev=mass,
        config=config,
    )
    analytic = tangent_vector(tangent)
    kinematic_residuals: list[float] = []
    same_branch: list[bool] = []
    centered_p2: np.ndarray | None = None
    centered_e2: np.ndarray | None = None
    centered_weight: np.ndarray | None = None
    best_residual = float("inf")

    for epsilon in epsilons:
        plus = primal_batch(
            p1=p1,
            temperature=temperature + epsilon,
            mass=mass,
            config=config,
        )
        minus = primal_batch(
            p1=p1,
            temperature=temperature - epsilon,
            mass=mass,
            config=config,
        )
        branch_ok = bool(
            np.array_equal(plus.support, tangent.support)
            and np.array_equal(minus.support, tangent.support)
        )
        same_branch.append(branch_ok)
        if not branch_ok:
            kinematic_residuals.append(float("nan"))
            continue
        centered = (batch_vector(plus) - batch_vector(minus)) / (2.0 * epsilon)
        residual = relative(analytic, centered)
        kinematic_residuals.append(residual)
        if residual < best_residual:
            best_residual = residual
            centered_p2 = (plus.p2 - minus.p2) / (2.0 * epsilon)
            centered_e2 = (plus.e2 - minus.e2) / (2.0 * epsilon)
            centered_weight = (
                plus.quadrature_weight - minus.quadrature_weight
            ) / (2.0 * epsilon)

    eos = electromagnetic_eos_tgamma_tangent(temperature, mass)
    eos_analytic = np.array([eos.d_rho, eos.d_pressure, eos.d2_rho])
    eos_residuals: list[float] = []
    for epsilon in epsilons:
        plus = ind.electromagnetic_eos_adaptive(temperature + epsilon, mass)
        minus = ind.electromagnetic_eos_adaptive(temperature - epsilon, mass)
        centered = np.array([
            (plus.rho - minus.rho) / (2.0 * epsilon),
            (plus.pressure - minus.pressure) / (2.0 * epsilon),
            (
                plus.drho_dtemperature - minus.drho_dtemperature
            ) / (2.0 * epsilon),
        ])
        eos_residuals.append(relative(eos_analytic, centered))

    if centered_p2 is None or centered_e2 is None or centered_weight is None:
        raise RuntimeError("no same-branch centered witness was available")
    mutation_residuals = {
        "freeze-p2": relative(np.zeros_like(tangent.d_p2), centered_p2),
        "flip-e2-sign": relative(-tangent.d_e2, centered_e2),
        "omit-weight-scale": relative(
            np.zeros_like(tangent.d_quadrature_weight), centered_weight
        ),
    }

    finite_kinematic = [value for value in kinematic_residuals if np.isfinite(value)]
    receipt = {
        "schema": "rabbit-d080a-tgamma-kinematic/v1",
        "classification": "TGAMMA_KINEMATIC_TANGENT_ONLY",
        "comparator_blob_sha": comparator_blob_sha(),
        "expected_comparator_blob_sha": EXPECTED_COMPARATOR_BLOB_SHA,
        "temperature_gamma_mev": temperature,
        "electron_mass_mev": mass,
        "target_neutrino_momentum_mev": p1,
        "epsilons_mev": epsilons,
        "kinematic_residuals": kinematic_residuals,
        "eos_residuals": eos_residuals,
        "best_kinematic_residual": min(finite_kinematic),
        "best_eos_residual": min(eos_residuals),
        "all_centered_samples_same_branch": all(same_branch),
        "support_margin": tangent.minimum_support_margin,
        "lambda_margin": tangent.minimum_lambda_margin,
        "mutation_residuals": mutation_residuals,
        "claim_ceiling": (
            "smooth T_gamma quadrature/kinematic/EOS tangent only; "
            "no collision-column, RHS-column, solver, trajectory, endpoint, or gate claim"
        ),
    }
    if receipt["comparator_blob_sha"] != EXPECTED_COMPARATOR_BLOB_SHA:
        raise RuntimeError("comparator Git blob identity changed")

    (output / "research_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_line_plot(
        output / "kinematic_residual_ladder.png",
        epsilons,
        kinematic_residuals,
        xlabel=r"$\epsilon$ [MeV]",
        ylabel="scaled directional residual",
        title=r"$T_\gamma$ moving-kinematics derivative ladder",
    )
    save_line_plot(
        output / "eos_residual_ladder.png",
        epsilons,
        eos_residuals,
        xlabel=r"$\epsilon$ [MeV]",
        ylabel="scaled directional residual",
        title=r"Electromagnetic EOS $T_\gamma$ derivative ladder",
    )
    save_bar_plot(
        output / "support_margin.png",
        ["support", "Kallen"],
        [tangent.minimum_support_margin, tangent.minimum_lambda_margin],
        ylabel="normalized minimum margin",
        title="Distance from discrete kinematic boundaries",
    )
    save_bar_plot(
        output / "mutation_kills.png",
        list(mutation_residuals),
        list(mutation_residuals.values()),
        ylabel="scaled residual",
        title="Load-bearing derivative mutations",
    )

    summary = f"""# D-080A T-gamma kinematic tangent probe

- classification: `{receipt['classification']}`
- comparator Git blob: `{receipt['comparator_blob_sha']}`
- best kinematic residual: `{receipt['best_kinematic_residual']:.9e}`
- best EOS residual: `{receipt['best_eos_residual']:.9e}`
- all centered samples same branch: `{receipt['all_centered_samples_same_branch']}`
- support margin: `{receipt['support_margin']:.9e}`
- Kallen margin: `{receipt['lambda_margin']:.9e}`
- freeze-p2 mutation: `{mutation_residuals['freeze-p2']:.9e}`
- flip-e2-sign mutation: `{mutation_residuals['flip-e2-sign']:.9e}`
- omit-weight-scale mutation: `{mutation_residuals['omit-weight-scale']:.9e}`

This probe certifies only the smooth incoming-electron quadrature, elastic
kinematics, mapped output coordinates, matrix-element dot products, and QED-off
EOS temperature tangents. It does not assemble the collision or full RHS
`T_gamma` column.
"""
    (output / "probe_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
