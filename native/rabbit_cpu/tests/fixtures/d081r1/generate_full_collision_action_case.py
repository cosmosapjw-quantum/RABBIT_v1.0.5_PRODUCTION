#!/usr/bin/env python3
"""Generate the deterministic D-081R1D full-action Python oracle fixture."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rabbit.decoupling import _independent_noqke as oracle  # noqa: E402

COMPARATOR = ROOT / "src/rabbit/decoupling/_independent_noqke.py"
OUTPUT = Path(__file__).with_name("full_collision_action_case.json")
EXPECTED_COMPARATOR_BLOB = "de44feee0aa484abe26976c7dc34c579643005b5"
ORDER = 8
Y_MAX = 8.0


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


def encode_float_map(values: dict[str, float] | object) -> dict[str, str]:
    mapping = dict(values)
    return {key: float_bits(float(mapping[key])) for key in sorted(mapping)}


def encode_array_map(values: dict[object, np.ndarray] | object) -> dict[str, dict[str, object]]:
    mapping = dict(values)
    return {str(key): encode_array(mapping[key]) for key in sorted(mapping, key=str)}


def action_moments(
    grid: oracle.IndependentNoQkeGrid,
    action: np.ndarray,
    temperature_cm: float,
) -> dict[str, str]:
    return encode_float_map(
        asdict(
            oracle.independent_action_moments(
                grid=grid,
                action=action,
                temperature_cm_mev=temperature_cm,
            )
        )
    )


def build_case(
    *,
    name: str,
    pair_logits: np.ndarray,
    temperature_cm: float,
    temperature_gamma: float,
    grid: oracle.IndependentNoQkeGrid,
) -> dict[str, object]:
    pair_cloglog = oracle.pair_logits_to_cloglog(pair_logits)
    result = oracle.evaluate_independent_collision_action(
        grid=grid,
        pair_cloglog=pair_cloglog,
        temperature_cm_mev=temperature_cm,
        temperature_gamma_mev=temperature_gamma,
    )

    if result.species_order != oracle.SPECIES:
        raise AssertionError("species order changed")
    if result.total.shape != (6, ORDER):
        raise AssertionError("unexpected native action shape")
    if result.modal_total.shape != (6, ORDER):
        raise AssertionError("unexpected modal action shape")
    if sorted(result.self_rows) != list(range(1, 10)):
        raise AssertionError("self row decomposition changed")
    if len(result.electron_families) != 15:
        raise AssertionError("electron-family decomposition changed")
    if not all(np.isfinite(float(value)) for value in result.diagnostics.values()):
        raise AssertionError("nonfinite diagnostic")

    first_law = float(result.diagnostics["first_law_residual"])
    if first_law > 5.0e-13:
        raise AssertionError(f"first-law residual too large in {name}: {first_law}")
    if float(result.diagnostics["charge_conjugation_residual"]) > 5.0e-12:
        raise AssertionError(f"CP residual too large in {name}")

    if name == "equilibrium":
        if float(np.max(np.abs(result.total))) > 1.0e-18:
            raise AssertionError("equilibrium action is not numerically null")
        if float(result.diagnostics["mu_tau_residual"]) > 5.0e-12:
            raise AssertionError("equilibrium mu/tau symmetry failed")
    elif name == "thermal_split":
        qnu = float(result.diagnostics["event_neutrino_energy_transfer"])
        qem = float(result.electron_bath_energy_transfer)
        if not (qnu > 0.0 and qem < 0.0):
            raise AssertionError("thermal restoring energy-transfer sign failed")
    elif name == "mu_tau_split":
        if float(result.diagnostics["mu_tau_residual"]) <= 1.0e-8:
            raise AssertionError("mu/tau antisymmetric response was erased")

    arrays = {
        "self_native": result.self_interaction,
        "electron_native": result.electron,
        "total_native": result.total,
        "self_modal": result.modal_self_interaction,
        "electron_modal": result.modal_electron,
        "total_modal": result.modal_total,
    }
    return {
        "name": name,
        "temperature_cm_bits": float_bits(temperature_cm),
        "temperature_gamma_bits": float_bits(temperature_gamma),
        "pair_logits": encode_array(pair_logits),
        "pair_cloglog": encode_array(pair_cloglog),
        "arrays": {key: encode_array(value) for key, value in arrays.items()},
        "absolute_envelopes": {
            key: encode_array(np.abs(value)) for key, value in arrays.items()
        },
        "self_rows": encode_array_map(result.self_rows),
        "electron_families": encode_array_map(result.electron_families),
        "electron_bath_energy_transfer_bits": float_bits(
            result.electron_bath_energy_transfer
        ),
        "electron_bath_energy_by_family": encode_float_map(
            dict(result.electron_bath_energy_by_family)
        ),
        "whole_reaction_domain_rejections": int(
            result.whole_reaction_domain_rejections
        ),
        "matrix_roundoff_corrections": int(result.matrix_roundoff_corrections),
        "largest_matrix_roundoff_correction_bits": float_bits(
            result.largest_matrix_roundoff_correction
        ),
        "diagnostics": encode_float_map(dict(result.diagnostics)),
        "moments": {
            "self": action_moments(grid, result.self_interaction, temperature_cm),
            "electron": action_moments(grid, result.electron, temperature_cm),
            "total": action_moments(grid, result.total, temperature_cm),
        },
    }


def main() -> None:
    blob = git_blob(COMPARATOR)
    if blob != EXPECTED_COMPARATOR_BLOB:
        raise SystemExit(
            f"frozen comparator mismatch: {blob} != {EXPECTED_COMPARATOR_BLOB}"
        )

    grid = oracle.build_independent_grid(order=ORDER, y_max=Y_MAX)
    equilibrium_logits = np.stack([-grid.nodes, -grid.nodes, -grid.nodes])
    profile = 0.02 * (1.0 - 2.0 * grid.nodes / Y_MAX)
    mu_tau_logits = np.stack(
        [-grid.nodes, -grid.nodes + profile, -grid.nodes - profile]
    )

    payload = {
        "schema": "rabbit.d081r1.full_collision_action.v1",
        "private_comparator_git_blob": blob,
        "generator_git_blob": git_blob(Path(__file__).resolve()),
        "order": ORDER,
        "y_max_bits": float_bits(Y_MAX),
        "species_order": list(oracle.SPECIES),
        "self_event_count": len(oracle.independent_self_events()),
        "electron_event_count": len(oracle.independent_electron_events()),
        "collision_config": {
            "incoming_polar_order": 12,
            "final_polar_order": 12,
            "final_azimuth_order": 4,
            "electron_radial_order": 48,
            "matrix_roundoff_ulps_bits": float_bits(1024.0),
        },
        "grid_nodes": encode_array(grid.nodes),
        "grid_weights": encode_array(grid.weights),
        "cases": [
            build_case(
                name="equilibrium",
                pair_logits=equilibrium_logits,
                temperature_cm=2.0,
                temperature_gamma=2.0,
                grid=grid,
            ),
            build_case(
                name="thermal_split",
                pair_logits=equilibrium_logits,
                temperature_cm=2.0,
                temperature_gamma=2.05,
                grid=grid,
            ),
            build_case(
                name="mu_tau_split",
                pair_logits=mu_tau_logits,
                temperature_cm=2.0,
                temperature_gamma=2.0,
                grid=grid,
            ),
        ],
        "claim_ceiling": (
            "frozen Python order-8 full-action oracle only; no Rust full-action "
            "parity, retained order-60 result, RHS/J/Jv, solver, performance, "
            "endpoint, N_eff or F10 gate movement"
        ),
    }

    if payload["self_event_count"] != 27 or payload["electron_event_count"] != 15:
        raise AssertionError("global event catalogue changed")

    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT}")
    print(f"sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
