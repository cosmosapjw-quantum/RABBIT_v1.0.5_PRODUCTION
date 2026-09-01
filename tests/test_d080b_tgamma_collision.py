from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d080b_tgamma_collision import (
    electron_tgamma_branch_signature,
    evaluate_tgamma_collision_action_jvp,
)


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    scale = max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny)
    return float(np.linalg.norm(a - b) / scale)


def _config(*, electron_radial_order: int = 8) -> ind.IndependentCollisionConfig:
    return ind.IndependentCollisionConfig(
        incoming_polar_order=2,
        final_polar_order=2,
        final_azimuth_order=4,
        electron_radial_order=electron_radial_order,
    )


def _thermal_split_case() -> tuple[
    ind.IndependentNoQkeGrid,
    ind.IndependentCollisionConfig,
    np.ndarray,
]:
    grid = ind.build_independent_grid(8, 8.0)
    logits = np.stack(
        [
            -grid.nodes + 0.04 * np.exp(-grid.nodes / 3.0),
            -grid.nodes - 0.02 * np.exp(-grid.nodes / 4.0),
            -grid.nodes - 0.02 * np.exp(-grid.nodes / 4.0),
        ]
    )
    return grid, _config(), ind.pair_logits_to_cloglog(logits)


def _centered_action(
    *,
    grid: ind.IndependentNoQkeGrid,
    pair_cloglog: np.ndarray,
    temperature_cm: float,
    temperature_gamma: float,
    epsilon: float,
    config: ind.IndependentCollisionConfig,
) -> tuple[np.ndarray, float, float]:
    plus = ind.evaluate_independent_collision_action(
        grid=grid,
        pair_cloglog=pair_cloglog,
        temperature_cm_mev=temperature_cm,
        temperature_gamma_mev=temperature_gamma + epsilon,
        config=config,
    )
    minus = ind.evaluate_independent_collision_action(
        grid=grid,
        pair_cloglog=pair_cloglog,
        temperature_cm_mev=temperature_cm,
        temperature_gamma_mev=temperature_gamma - epsilon,
        config=config,
    )
    total = (np.asarray(plus.total) - np.asarray(minus.total)) / (2.0 * epsilon)
    qnu = (
        float(plus.diagnostics["event_neutrino_energy_transfer"])
        - float(minus.diagnostics["event_neutrino_energy_transfer"])
    ) / (2.0 * epsilon)
    qem = (
        float(plus.electron_bath_energy_transfer)
        - float(minus.electron_bath_energy_transfer)
    ) / (2.0 * epsilon)
    return total, qnu, qem


def test_equilibrium_tgamma_column_is_restoring_and_first_law_closed() -> None:
    grid = ind.build_independent_grid(8, 8.0)
    config = _config()
    temperature = 2.0
    equilibrium = ind.pair_logits_to_cloglog(
        np.stack([-grid.nodes for _ in range(3)])
    )
    result = evaluate_tgamma_collision_action_jvp(
        grid=grid,
        pair_cloglog=equilibrium,
        temperature_cm_mev=temperature,
        temperature_gamma_mev=temperature,
        config=config,
    )

    tangent_scale = max(np.linalg.norm(result.total), np.finfo(float).tiny)
    assert np.linalg.norm(result.base.total) < 2.0e-7 * tangent_scale
    assert result.neutrino_energy_transfer > 0.0
    assert result.electron_bath_energy_transfer < 0.0
    assert result.first_law_tangent_residual < 2.0e-10
    assert result.charge_conjugation_residual < 2.0e-10
    assert result.mu_tau_residual < 2.0e-10
    assert np.linalg.norm(result.measure) < 2.0e-7 * tangent_scale
    assert np.linalg.norm(result.matrix) < 2.0e-7 * tangent_scale
    assert np.linalg.norm(result.projection) < 2.0e-7 * tangent_scale


def test_full_tgamma_collision_column_matches_same_branch_ladder() -> None:
    grid, config, c = _thermal_split_case()
    tcm, tg = 2.0, 2.05
    result = evaluate_tgamma_collision_action_jvp(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    assert result.base_reconstruction_residual < 2.0e-12
    assert result.component_sum_residual < 2.0e-13
    assert result.first_law_tangent_residual < 2.0e-10
    assert result.minimum_support_margin > 0.0
    assert result.minimum_lambda_margin > 0.0

    action_residuals: list[float] = []
    energy_residuals: list[float] = []
    best_centered: np.ndarray | None = None
    best = float("inf")
    for epsilon in (3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4):
        assert electron_tgamma_branch_signature(
            grid=grid,
            temperature_cm_mev=tcm,
            temperature_gamma_mev=tg + epsilon,
            config=config,
        ) == result.branch_signature
        assert electron_tgamma_branch_signature(
            grid=grid,
            temperature_cm_mev=tcm,
            temperature_gamma_mev=tg - epsilon,
            config=config,
        ) == result.branch_signature
        centered, qnu, qem = _centered_action(
            grid=grid,
            pair_cloglog=c,
            temperature_cm=tcm,
            temperature_gamma=tg,
            epsilon=epsilon,
            config=config,
        )
        residual = _relative(result.total, centered)
        action_residuals.append(residual)
        energy_residuals.append(
            _relative(
                np.array([
                    result.neutrino_energy_transfer,
                    result.electron_bath_energy_transfer,
                ]),
                np.array([qnu, qem]),
            )
        )
        if residual < best:
            best = residual
            best_centered = centered

    assert min(action_residuals) < 3.0e-5
    assert min(energy_residuals) < 3.0e-5
    assert best_centered is not None

    mutations = {
        "flip-pauli": result.total - 2.0 * result.pauli,
        "omit-measure": result.total - result.measure,
        "omit-matrix": result.total - result.matrix,
        "omit-moving-projection": result.total - result.projection,
        "omit-elastic": result.total - result.elastic,
        "omit-pair": result.total - result.pair,
    }
    for name, mutant in mutations.items():
        assert _relative(mutant, best_centered) > 1.0e-6, name


@pytest.mark.slow
def test_retained_stiff_state_has_admissible_local_tgamma_column() -> None:
    path_text = os.environ.get("D080B_STATE_1200")
    if not path_text:
        pytest.skip("retained state is supplied only by the sealed D-080B workflow")
    with np.load(Path(path_text), allow_pickle=False) as archive:
        state = np.asarray(archive["y"], dtype=np.float64)

    order, y_max = 60, 30.0
    grid = ind.build_independent_grid(order, y_max)
    config = ind.IndependentCollisionConfig(
        incoming_polar_order=4,
        final_polar_order=4,
        final_azimuth_order=4,
        electron_radial_order=24,
    )
    c = state[: 3 * order].reshape(3, order)
    tg = float(state[3 * order])
    expansion = 0.16286930247517223
    tcm = 10.0 * np.exp(-expansion)
    result = evaluate_tgamma_collision_action_jvp(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    epsilon = 3.0e-6 * max(tg, 1.0)
    assert electron_tgamma_branch_signature(
        grid=grid,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg + epsilon,
        config=config,
    ) == result.branch_signature
    assert electron_tgamma_branch_signature(
        grid=grid,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg - epsilon,
        config=config,
    ) == result.branch_signature
    centered, qnu, qem = _centered_action(
        grid=grid,
        pair_cloglog=c,
        temperature_cm=tcm,
        temperature_gamma=tg,
        epsilon=epsilon,
        config=config,
    )
    assert _relative(result.total, centered) < 2.0e-3
    assert _relative(
        np.array([result.neutrino_energy_transfer, result.electron_bath_energy_transfer]),
        np.array([qnu, qem]),
    ) < 2.0e-3
    assert result.first_law_tangent_residual < 2.0e-8
