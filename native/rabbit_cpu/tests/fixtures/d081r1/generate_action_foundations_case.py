#!/usr/bin/env python3
"""Generate the deterministic D-081R1D1 grid/spectral/kinematic fixture."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rabbit.decoupling import _independent_noqke as oracle  # noqa: E402

COMPARATOR = ROOT / "src/rabbit/decoupling/_independent_noqke.py"
OUTPUT = Path(__file__).with_name("action_foundations_case.json")
EXPECTED_COMPARATOR_BLOB = "de44feee0aa484abe26976c7dc34c579643005b5"
ORDER = 8
Y_MAX = 8.0
T_CM = 2.0
T_GAMMA = 2.05


def git_blob(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(path.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float_bits(value: float) -> str:
    return f"{np.float64(value).view(np.uint64).item():016x}"


def encode_array(values: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return {
        "shape": list(array.shape),
        "bits": [f"{item:016x}" for item in array.view(np.uint64).ravel().tolist()],
    }


def selected_indices(support: np.ndarray) -> list[int]:
    mask = np.asarray(support, dtype=bool).ravel()
    selected: list[int] = []
    for pool in (np.flatnonzero(mask), np.flatnonzero(~mask)):
        if pool.size:
            for location in (0, pool.size // 2, pool.size - 1):
                index = int(pool[location])
                if index not in selected:
                    selected.append(index)
    for index in (0, mask.size // 3, 2 * mask.size // 3, mask.size - 1):
        if index not in selected:
            selected.append(index)
    return selected


def encode_kinematic_batch(name: str, batch: object) -> dict[str, object]:
    support = np.asarray(batch.support, dtype=bool)
    fields = (
        "p2",
        "e2",
        "e3",
        "e4",
        "p3_magnitude",
        "p4_magnitude",
        "phase_space",
        "quadrature_weight",
        "d12",
        "d13",
        "d14",
        "d23",
        "d24",
        "d34",
    )
    flattened = {
        field: np.asarray(getattr(batch, field), dtype=np.float64).ravel()
        for field in fields
    }
    samples = []
    for index in selected_indices(support):
        samples.append(
            {
                "flat_index": index,
                "support": bool(support.ravel()[index]),
                "values": {
                    field: float_bits(flattened[field][index]) for field in fields
                },
            }
        )
    return {
        "name": name,
        "shape": list(support.shape),
        "support_count": int(np.count_nonzero(support)),
        "samples": samples,
    }


def main() -> None:
    blob = git_blob(COMPARATOR)
    if blob != EXPECTED_COMPARATOR_BLOB:
        raise SystemExit(f"comparator mismatch: {blob} != {EXPECTED_COMPARATOR_BLOB}")

    grid = oracle.build_independent_grid(order=ORDER, y_max=Y_MAX)
    equilibrium_logits = np.stack([-grid.nodes, -grid.nodes, -grid.nodes])
    profile = 0.02 * (1.0 - 2.0 * grid.nodes / Y_MAX)
    pair_logits = np.stack(
        [-grid.nodes, -grid.nodes + profile, -grid.nodes - profile]
    )
    pair_cloglog = oracle.pair_logits_to_cloglog(pair_logits)

    query = np.asarray([0.0, 0.125, 0.75, 2.0, 4.5, 7.875, 8.0], dtype=np.float64)
    modal_basis = grid.modal_basis(query)
    modal_coefficients = np.stack(
        [grid.modal_coefficients(row) for row in pair_logits]
    )
    interpolation = modal_basis @ modal_coefficients.T

    rates = np.stack(
        [
            np.linspace(-0.25, 0.30, query.size, dtype=np.float64),
            np.cos(0.7 * query),
        ]
    )
    modal_product = oracle._modal_product(rates, query, grid)
    modal_seed = np.stack(
        [
            np.linspace(-1.0, 1.0, ORDER, dtype=np.float64),
            np.cos(np.arange(ORDER, dtype=np.float64)),
        ]
    ) * 1.0e-20
    native_action = oracle._native_action(grid, modal_seed, T_CM)

    config = oracle.IndependentCollisionConfig()
    incoming_mu, incoming_w, final_mu, final_w, azimuth, azimuth_w = (
        oracle._angular_rule(
            config.incoming_polar_order,
            config.final_polar_order,
            config.final_azimuth_order,
        )
    )
    electron_p2, electron_w = oracle._electron_half_line_rule(
        config.electron_radial_order, T_GAMMA
    )
    p1 = T_CM * float(grid.nodes[3])
    neutrino_p2 = T_CM * grid.nodes
    neutrino_w = T_CM * grid.weights

    batches = [
        encode_kinematic_batch(
            "self",
            oracle._two_body_kinematics(
                p1=p1,
                p2_nodes=neutrino_p2,
                p2_weights=neutrino_w,
                mass2=0.0,
                mass3=0.0,
                mass4=0.0,
                config=config,
            ),
        ),
        encode_kinematic_batch(
            "elastic",
            oracle._two_body_kinematics(
                p1=p1,
                p2_nodes=electron_p2,
                p2_weights=electron_w,
                mass2=oracle.M_ELECTRON_MEV,
                mass3=0.0,
                mass4=oracle.M_ELECTRON_MEV,
                config=config,
            ),
        ),
        encode_kinematic_batch(
            "pair",
            oracle._two_body_kinematics(
                p1=p1,
                p2_nodes=neutrino_p2,
                p2_weights=neutrino_w,
                mass2=0.0,
                mass3=oracle.M_ELECTRON_MEV,
                mass4=oracle.M_ELECTRON_MEV,
                config=config,
            ),
        ),
    ]

    payload = {
        "schema": "rabbit.d081r1.action_foundations.v1",
        "private_comparator_git_blob": blob,
        "generator_git_blob": git_blob(Path(__file__).resolve()),
        "order": ORDER,
        "y_max_bits": float_bits(Y_MAX),
        "temperature_cm_bits": float_bits(T_CM),
        "temperature_gamma_bits": float_bits(T_GAMMA),
        "electron_mass_bits": float_bits(oracle.M_ELECTRON_MEV),
        "grid": {
            "nodes": encode_array(grid.nodes),
            "weights": encode_array(grid.weights),
        },
        "chart": {
            "pair_logits": encode_array(pair_logits),
            "pair_cloglog": encode_array(pair_cloglog),
            "equilibrium_logits": encode_array(equilibrium_logits),
        },
        "spectral": {
            "query": encode_array(query),
            "modal_basis": encode_array(modal_basis),
            "modal_coefficients": encode_array(modal_coefficients),
            "interpolation": encode_array(interpolation.T),
            "rates": encode_array(rates),
            "modal_product": encode_array(modal_product),
            "modal_seed": encode_array(modal_seed),
            "native_action": encode_array(native_action),
        },
        "rules": {
            "incoming_mu": encode_array(incoming_mu),
            "incoming_weights": encode_array(incoming_w),
            "final_mu": encode_array(final_mu),
            "final_weights": encode_array(final_w),
            "azimuth": encode_array(azimuth),
            "azimuth_weights": encode_array(azimuth_w),
            "electron_p2": encode_array(electron_p2),
            "electron_weights": encode_array(electron_w),
        },
        "kinematics": {
            "p1_bits": float_bits(p1),
            "config": {
                "incoming_polar_order": config.incoming_polar_order,
                "final_polar_order": config.final_polar_order,
                "final_azimuth_order": config.final_azimuth_order,
                "electron_radial_order": config.electron_radial_order,
            },
            "batches": batches,
        },
        "claim_ceiling": (
            "grid/chart/spectral/selected-kinematic foundations only; no full "
            "Rust collision action, RHS, JVP, Jacobian, solver, trajectory, "
            "performance, endpoint, N_eff or F10 gate movement"
        ),
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT}")
    print(f"sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
