from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_rhs_jvp import evaluate_static_rhs_from_packed_state
from scripts.audit._d080b_tgamma_collision import electron_tgamma_branch_signature


def _api() -> ModuleType:
    try:
        return importlib.import_module("scripts.audit._d080c_tgamma_rhs")
    except ModuleNotFoundError:
        pytest.fail("D-080C implementation module is not present")


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    scale = max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny)
    return float(np.linalg.norm(a - b) / scale)


def _scalar_relative(left: float, right: float) -> float:
    scale = max(abs(float(left)), abs(float(right)), np.finfo(float).tiny)
    return abs(float(left) - float(right)) / scale


def _block_relative(left: np.ndarray, right: np.ndarray, order: int) -> float:
    """Compare spectral, temperature, and time rows without mixing units."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    size = 3 * order + 2
    if a.shape != (size,) or b.shape != (size,):
        raise ValueError("packed RHS vectors have an invalid shape")
    return max(
        _relative(a[: 3 * order], b[: 3 * order]),
        _scalar_relative(a[-2], b[-2]),
        _scalar_relative(a[-1], b[-1]),
    )


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


def _packed(c: np.ndarray, temperature_gamma: float, elapsed: float = 0.0) -> np.ndarray:
    return np.concatenate((np.asarray(c, dtype=np.float64).ravel(), [temperature_gamma, elapsed]))


def _centered_rhs(
    *,
    grid: ind.IndependentNoQkeGrid,
    pair_cloglog: np.ndarray,
    temperature_cm: float,
    temperature_gamma: float,
    epsilon: float,
    config: ind.IndependentCollisionConfig,
    elapsed: float = 0.0,
) -> np.ndarray:
    plus = evaluate_static_rhs_from_packed_state(
        grid=grid,
        packed_state=_packed(pair_cloglog, temperature_gamma + epsilon, elapsed),
        temperature_cm_mev=temperature_cm,
        config=config,
    )
    minus = evaluate_static_rhs_from_packed_state(
        grid=grid,
        packed_state=_packed(pair_cloglog, temperature_gamma - epsilon, elapsed),
        temperature_cm_mev=temperature_cm,
        config=config,
    )
    return (plus - minus) / (2.0 * epsilon)


def test_full_tgamma_rhs_column_matches_original_packed_rhs_ladder() -> None:
    api = _api()
    grid, config, c = _thermal_split_case()
    tcm, tg = 2.0, 2.05
    result = api.evaluate_tgamma_rhs_column(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    original = evaluate_static_rhs_from_packed_state(
        grid=grid,
        packed_state=_packed(c, tg, 17.0),
        temperature_cm_mev=tcm,
        config=config,
    )
    assert _block_relative(result.base_rhs, original, grid.order) < 5.0e-14
    assert result.base_reconstruction_residual < 5.0e-14
    assert result.component_sum_residual < 5.0e-13
    assert np.array_equal(result.elapsed_time_input_column, np.zeros_like(result.base_rhs))
    assert result.delta_hubble_over_hubble > 0.0
    assert result.tgamma_column[-1] < 0.0

    residuals: list[float] = []
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
        centered = _centered_rhs(
            grid=grid,
            pair_cloglog=c,
            temperature_cm=tcm,
            temperature_gamma=tg,
            epsilon=epsilon,
            config=config,
        )
        residual = _block_relative(result.tgamma_column, centered, grid.order)
        residuals.append(residual)
        if residual < best:
            best = residual
            best_centered = centered

    assert min(residuals) < 5.0e-5
    assert best_centered is not None
    swapped_output_rows = result.tgamma_column.copy()
    swapped_output_rows[-2], swapped_output_rows[-1] = (
        result.tgamma_column[-1],
        result.tgamma_column[-2],
    )
    mutations = {
        "omit-collision": result.tgamma_column - result.collision_component,
        "omit-all-hubble-feedback": result.tgamma_column - result.hubble_component,
        "omit-spectral-hubble-feedback": (
            result.tgamma_column - result.spectral_hubble_component
        ),
        "omit-temperature-hubble-feedback": (
            result.tgamma_column - result.temperature_hubble_component
        ),
        "omit-time-hubble-feedback": (
            result.tgamma_column - result.time_hubble_component
        ),
        "omit-heat-capacity-derivative": (
            result.tgamma_column - result.heat_capacity_component
        ),
        "flip-electromagnetic-transfer": (
            result.tgamma_column - 2.0 * result.temperature_collision_component
        ),
        "swap-temperature-time-output-rows": swapped_output_rows,
    }
    for name, mutant in mutations.items():
        assert _block_relative(mutant, best_centered, grid.order) > max(
            1.0e-6, 20.0 * best
        ), name


def test_elapsed_time_input_column_is_exactly_structural_zero() -> None:
    api = _api()
    grid, config, c = _thermal_split_case()
    tcm, tg = 2.0, 2.05
    first = evaluate_static_rhs_from_packed_state(
        grid=grid,
        packed_state=_packed(c, tg, -1.0e40),
        temperature_cm_mev=tcm,
        config=config,
    )
    second = evaluate_static_rhs_from_packed_state(
        grid=grid,
        packed_state=_packed(c, tg, 1.0e40),
        temperature_cm_mev=tcm,
        config=config,
    )
    result = api.evaluate_tgamma_rhs_column(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    assert np.array_equal(first, second)
    assert np.array_equal(result.elapsed_time_input_column, np.zeros_like(first))


def test_equilibrium_tgamma_rhs_column_retains_restoring_collision_sign() -> None:
    api = _api()
    grid = ind.build_independent_grid(8, 8.0)
    config = _config()
    temperature = 2.0
    c = ind.pair_logits_to_cloglog(np.stack([-grid.nodes for _ in range(3)]))
    result = api.evaluate_tgamma_rhs_column(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=temperature,
        temperature_gamma_mev=temperature,
        config=config,
    )
    assert result.collision.neutrino_energy_transfer > 0.0
    assert result.collision.electron_bath_energy_transfer < 0.0
    assert result.collision.first_law_tangent_residual < 2.0e-10
    assert result.delta_hubble_over_hubble > 0.0
    assert result.tgamma_column[-1] < 0.0
    assert np.all(np.isfinite(result.tgamma_column))


def test_manufactured_weak_collision_tail_matches_original_rhs() -> None:
    api = _api()
    grid = ind.build_independent_grid(8, 10.0)
    config = _config()
    logits = np.stack(
        [
            -grid.nodes + 0.012 * np.exp(-grid.nodes / 2.0),
            -grid.nodes - 0.006 * np.exp(-grid.nodes / 2.5),
            -grid.nodes - 0.006 * np.exp(-grid.nodes / 2.5),
        ]
    )
    c = ind.pair_logits_to_cloglog(logits)
    tcm, tg = 0.45, 0.50
    result = api.evaluate_tgamma_rhs_column(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    epsilon = 2.0e-5
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
    centered = _centered_rhs(
        grid=grid,
        pair_cloglog=c,
        temperature_cm=tcm,
        temperature_gamma=tg,
        epsilon=epsilon,
        config=config,
    )
    assert _block_relative(result.tgamma_column, centered, grid.order) < 3.0e-4


@pytest.mark.slow
def test_retained_stiff_state_has_admissible_full_tgamma_rhs_column() -> None:
    api = _api()
    path_text = os.environ.get("D080C_STATE_1200")
    if not path_text:
        pytest.skip("retained state is supplied only by the sealed D-080C workflow")
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
    result = api.evaluate_tgamma_rhs_column(
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
    centered = _centered_rhs(
        grid=grid,
        pair_cloglog=c,
        temperature_cm=tcm,
        temperature_gamma=tg,
        epsilon=epsilon,
        config=config,
        elapsed=float(state[-1]),
    )
    assert _block_relative(result.tgamma_column, centered, grid.order) < 3.0e-3
    assert result.collision.first_law_tangent_residual < 2.0e-8
    assert np.array_equal(
        result.elapsed_time_input_column, np.zeros_like(result.tgamma_column)
    )
