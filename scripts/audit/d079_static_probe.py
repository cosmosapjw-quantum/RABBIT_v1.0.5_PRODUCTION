#!/usr/bin/env python3
"""Generate deterministic D-079 static physical derivative evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_collision_jvp import evaluate_collision_action_jvp
from scripts.audit._d079_rhs_jvp import (
    evaluate_c_only_rhs_jvp,
    evaluate_static_rhs_from_packed_state,
)
from scripts.audit._d079_tangent_primitives import EXPECTED_COMPARATOR_BLOB_SHA


def relative(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(np.linalg.norm(left), np.linalg.norm(right), np.finfo(float).tiny)
    return float(np.linalg.norm(left - right) / scale)


def build_case():
    order, y_max = 8, 8.0
    grid = ind.build_independent_grid(order, y_max)
    config = ind.IndependentCollisionConfig(
        incoming_polar_order=2,
        final_polar_order=2,
        final_azimuth_order=4,
        electron_radial_order=8,
    )
    logits = np.stack([
        -grid.nodes + 0.04 * np.exp(-grid.nodes / 3.0),
        -grid.nodes - 0.02 * np.exp(-grid.nodes / 4.0),
        -grid.nodes - 0.02 * np.exp(-grid.nodes / 4.0),
    ])
    c = ind.pair_logits_to_cloglog(logits)
    x = np.linspace(-1.0, 1.0, order)
    direction = np.stack((0.3 + x, -0.2 + x**2, -0.2 + x**2))
    direction /= np.linalg.norm(direction)
    return grid, config, c, direction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    grid, config, c, direction = build_case()
    tcm, tg = 2.0, 2.05
    collision = evaluate_collision_action_jvp(
        grid=grid, pair_cloglog=c, direction_cloglog=direction,
        temperature_cm_mev=tcm, temperature_gamma_mev=tg, config=config,
    )
    rhs = evaluate_c_only_rhs_jvp(
        grid=grid, pair_cloglog=c, direction_cloglog=direction,
        temperature_cm_mev=tcm, temperature_gamma_mev=tg, config=config,
    )
    packed = np.concatenate((c.ravel(), [tg, 0.0]))
    epsilons = np.asarray((1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5))
    collision_residuals = []
    rhs_residuals = []
    centered_collision = []

    for epsilon in epsilons:
        plus_action = ind.evaluate_independent_collision_action(
            grid=grid, pair_cloglog=c + epsilon * direction,
            temperature_cm_mev=tcm, temperature_gamma_mev=tg, config=config,
        ).total
        minus_action = ind.evaluate_independent_collision_action(
            grid=grid, pair_cloglog=c - epsilon * direction,
            temperature_cm_mev=tcm, temperature_gamma_mev=tg, config=config,
        ).total
        centered_c = (plus_action - minus_action) / (2.0 * epsilon)
        centered_collision.append(centered_c)
        collision_residuals.append(relative(collision.total, centered_c))

        plus_rhs = evaluate_static_rhs_from_packed_state(
            grid=grid, packed_state=packed + epsilon * rhs.full_direction,
            temperature_cm_mev=tcm, config=config,
        )
        minus_rhs = evaluate_static_rhs_from_packed_state(
            grid=grid, packed_state=packed - epsilon * rhs.full_direction,
            temperature_cm_mev=tcm, config=config,
        )
        rhs_residuals.append(relative(rhs.jvp, (plus_rhs-minus_rhs)/(2.0*epsilon)))

    collision_residuals = np.asarray(collision_residuals)
    rhs_residuals = np.asarray(rhs_residuals)
    best_index = int(np.argmin(collision_residuals))
    oracle = centered_collision[best_index]
    swapped = collision.total.copy()
    swapped[[0, 2]] = swapped[[2, 0]]
    mutations = {
        "correct": relative(collision.total, oracle),
        "sign": relative(-collision.total, oracle),
        "scale-1.01": relative(1.01 * collision.total, oracle),
        "flavour-swap": relative(swapped, oracle),
        "omit-electron": relative(collision.self_interaction, oracle),
    }

    number_scale = max(
        float(np.sum(
            np.abs(collision.self_interaction)
            * (grid.weights * grid.nodes**2)[None, :]
        )),
        np.finfo(float).tiny,
    )
    energy_scale = max(
        float(np.sum(
            np.abs(collision.self_interaction)
            * (grid.weights * grid.nodes**3)[None, :]
        )),
        np.finfo(float).tiny,
    )
    invariants = {
        "self-number": abs(collision.self_number_moment) / number_scale,
        "self-energy": abs(collision.self_energy_moment) / energy_scale,
        "first-law": collision.first_law_tangent_residual,
        "CP": collision.charge_conjugation_residual,
        "mu-tau": collision.mu_tau_residual,
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.loglog(epsilons, collision_residuals, marker="o", label="collision action")
    ax.loglog(epsilons, rhs_residuals, marker="s", label="full static RHS")
    ax.set_xlabel("centered-difference step epsilon")
    ax.set_ylabel("relative directional residual")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "residual_ladder.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    names = list(invariants)
    values = np.maximum([invariants[name] for name in names], np.finfo(float).tiny)
    ax.bar(names, values)
    ax.set_yscale("log")
    ax.set_ylabel("normalized tangent residual")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "invariant_tangent_residuals.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    names = list(mutations)
    values = np.maximum([mutations[name] for name in names], np.finfo(float).tiny)
    ax.bar(names, values)
    ax.set_yscale("log")
    ax.set_ylabel("residual against best centered witness")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "mutation_kills.png", dpi=180)
    plt.close(fig)

    receipt = {
        "schema": "rabbit.d079.static_c_only_jvp.v1",
        "classification": "STATIC_C_ONLY_PHYSICAL_DERIVATIVE",
        "comparator_blob_sha": EXPECTED_COMPARATOR_BLOB_SHA,
        "case": {
            "order": grid.order,
            "y_max": grid.y_max,
            "temperature_cm_mev": tcm,
            "temperature_gamma_mev": tg,
            "direction_norm": float(np.linalg.norm(direction)),
        },
        "epsilon_ladder": epsilons.tolist(),
        "collision_residuals": collision_residuals.tolist(),
        "rhs_residuals": rhs_residuals.tolist(),
        "best_collision_residual": float(np.min(collision_residuals)),
        "best_rhs_residual": float(np.min(rhs_residuals)),
        "invariant_residuals": invariants,
        "mutation_residuals": mutations,
        "explicit_non_authority": [
            "no temperature or expansion-direction columns",
            "no integrator call",
            "no trajectory or endpoint",
            "no wall-time projection",
            "no F10 gate movement",
        ],
    }
    (out / "research_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = [
        "# D-079 static c-only JVP summary",
        "",
        "This is a static derivative receipt, not a trajectory or gate result.",
        "",
        f"- best collision residual: `{receipt['best_collision_residual']:.8e}`",
        f"- best full-RHS residual: `{receipt['best_rhs_residual']:.8e}`",
        f"- first-law tangent residual: `{invariants['first-law']:.8e}`",
        f"- sign mutation residual: `{mutations['sign']:.8e}`",
        f"- 1% scale mutation residual: `{mutations['scale-1.01']:.8e}`",
    ]
    (out / "probe_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({
        "classification": receipt["classification"],
        "best_collision_residual": receipt["best_collision_residual"],
        "best_rhs_residual": receipt["best_rhs_residual"],
        "mutation_residuals": mutations,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
