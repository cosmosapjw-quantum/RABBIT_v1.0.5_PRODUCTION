from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_rhs_jvp import (
    c_only_state_validator,
    evaluate_static_rhs_from_packed_state,
)
from scripts.audit._d080b_tgamma_collision import electron_tgamma_branch_signature


def _api() -> ModuleType:
    try:
        return importlib.import_module("scripts.audit._d080d_static_jacobian")
    except ModuleNotFoundError:
        pytest.fail("D-080D implementation module is not present")


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    scale = max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny)
    return float(np.linalg.norm(a - b) / scale)


def _scalar_relative(left: float, right: float) -> float:
    scale = max(abs(float(left)), abs(float(right)), np.finfo(float).tiny)
    return float(abs(float(left) - float(right)) / scale)


def _block_relative(left: np.ndarray, right: np.ndarray, order: int) -> float:
    """Compare heterogeneous RHS blocks only after separate normalization."""

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


def _thermal_case() -> tuple[
    ind.IndependentNoQkeGrid,
    ind.IndependentCollisionConfig,
    np.ndarray,
    float,
    float,
]:
    grid = ind.build_independent_grid(8, 8.0)
    logits = np.stack(
        [
            -grid.nodes + 0.04 * np.exp(-grid.nodes / 3.0),
            -grid.nodes - 0.02 * np.exp(-grid.nodes / 4.0),
            -grid.nodes - 0.02 * np.exp(-grid.nodes / 4.0),
        ]
    )
    return grid, _config(), ind.pair_logits_to_cloglog(logits), 2.0, 2.05


def _weak_tail_case() -> tuple[
    ind.IndependentNoQkeGrid,
    ind.IndependentCollisionConfig,
    np.ndarray,
    float,
    float,
]:
    grid = ind.build_independent_grid(8, 10.0)
    logits = np.stack(
        [
            -grid.nodes + 0.012 * np.exp(-grid.nodes / 2.0),
            -grid.nodes - 0.006 * np.exp(-grid.nodes / 2.5),
            -grid.nodes - 0.006 * np.exp(-grid.nodes / 2.5),
        ]
    )
    return grid, _config(), ind.pair_logits_to_cloglog(logits), 0.45, 0.50


def _packed(c: np.ndarray, temperature_gamma: float, elapsed: float = 0.0) -> np.ndarray:
    return np.concatenate(
        (np.asarray(c, dtype=np.float64).ravel(), [temperature_gamma, elapsed])
    )


def _mixed_direction(
    grid: ind.IndependentNoQkeGrid,
    temperature_gamma: float,
) -> np.ndarray:
    y = grid.nodes
    spectral = np.stack(
        [
            0.11 * np.exp(-y / 4.0) * np.cos(0.7 * y),
            -0.07 * np.exp(-y / 5.0) * np.sin(0.5 * y + 0.2),
            0.05 * np.exp(-y / 6.0) * np.cos(0.3 * y + 0.4),
        ]
    )
    return np.concatenate((spectral.ravel(), [0.025 * temperature_gamma, 2.0]))


def _second_direction(
    grid: ind.IndependentNoQkeGrid,
    temperature_gamma: float,
) -> np.ndarray:
    y = grid.nodes
    spectral = np.stack(
        [
            -0.035 * np.exp(-y / 7.0),
            0.027 * np.exp(-y / 4.5) * np.cos(y),
            0.019 * np.exp(-y / 5.5) * np.sin(0.4 * y),
        ]
    )
    return np.concatenate((spectral.ravel(), [-0.013 * temperature_gamma, -5.0]))


def _centered_directional_rhs(
    *,
    grid: ind.IndependentNoQkeGrid,
    pair_cloglog: np.ndarray,
    temperature_cm: float,
    temperature_gamma: float,
    direction: np.ndarray,
    epsilon: float,
    config: ind.IndependentCollisionConfig,
    elapsed: float = 0.0,
) -> np.ndarray:
    state = _packed(pair_cloglog, temperature_gamma, elapsed)
    vector = np.asarray(direction, dtype=np.float64)
    if state.shape != vector.shape:
        raise ValueError("state/direction shape mismatch")
    plus_state = state + epsilon * vector
    minus_state = state - epsilon * vector
    assert c_only_state_validator(grid, plus_state)
    assert c_only_state_validator(grid, minus_state)
    if vector[-2] != 0.0:
        base_signature = electron_tgamma_branch_signature(
            grid=grid,
            temperature_cm_mev=temperature_cm,
            temperature_gamma_mev=temperature_gamma,
            config=config,
        )
        for perturbed in (plus_state[-2], minus_state[-2]):
            assert electron_tgamma_branch_signature(
                grid=grid,
                temperature_cm_mev=temperature_cm,
                temperature_gamma_mev=float(perturbed),
                config=config,
            ) == base_signature
    plus = evaluate_static_rhs_from_packed_state(
        grid=grid,
        packed_state=plus_state,
        temperature_cm_mev=temperature_cm,
        config=config,
    )
    minus = evaluate_static_rhs_from_packed_state(
        grid=grid,
        packed_state=minus_state,
        temperature_cm_mev=temperature_cm,
        config=config,
    )
    return (plus - minus) / (2.0 * epsilon)


@pytest.fixture(scope="module")
def thermal_square() -> tuple[ModuleType, object, object, np.ndarray, float, float]:
    api = _api()
    grid, config, c, tcm, tg = _thermal_case()
    result = api.assemble_static_jacobian(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    return api, result, grid, c, tcm, tg


def test_square_layout_base_rhs_and_passive_elapsed_column(
    thermal_square: tuple[ModuleType, object, object, np.ndarray, float, float],
) -> None:
    _api_module, result, grid, c, tcm, tg = thermal_square
    size = 3 * grid.order + 2
    original = evaluate_static_rhs_from_packed_state(
        grid=grid,
        packed_state=_packed(c, tg, 17.0),
        temperature_cm_mev=tcm,
        config=_config(),
    )

    assert result.state_size == size
    assert result.spectral_size == 3 * grid.order
    assert result.jacobian.shape == (size, size)
    assert result.active_jacobian.shape == (size - 1, size - 1)
    assert _block_relative(result.base_rhs, original, grid.order) < 5.0e-14
    assert result.base_reconstruction_residual < 5.0e-14
    assert result.column_assembly_residual < 5.0e-14
    assert np.array_equal(result.jacobian[:, -2], result.tgamma_column)
    assert np.array_equal(result.jacobian[:, -1], np.zeros(size))
    assert np.array_equal(result.elapsed_time_column, np.zeros(size))

    elapsed_basis = np.zeros(size)
    elapsed_basis[-1] = 1.0
    assert np.array_equal(result.jacobian @ elapsed_basis, np.zeros(size))


def test_matrix_action_matches_independent_exact_directional_jvp(
    thermal_square: tuple[ModuleType, object, object, np.ndarray, float, float],
) -> None:
    api, result, grid, c, tcm, tg = thermal_square
    directions = (
        _mixed_direction(grid, tg),
        _second_direction(grid, tg),
    )
    for direction in directions:
        direct = api.evaluate_static_rhs_direction_jvp(
            grid=grid,
            pair_cloglog=c,
            full_direction=direction,
            temperature_cm_mev=tcm,
            temperature_gamma_mev=tg,
            config=_config(),
        )
        matrix_action = result.jacobian @ direction
        assert _block_relative(matrix_action, direct.jvp, grid.order) < 8.0e-13
        assert _block_relative(direct.base_rhs, result.base_rhs, grid.order) < 5.0e-14
        assert direct.elapsed_direction == float(direction[-1])
        assert np.array_equal(direct.elapsed_component, np.zeros_like(direct.jvp))


def test_full_square_matrix_matches_original_rhs_directional_ladders(
    thermal_square: tuple[ModuleType, object, object, np.ndarray, float, float],
) -> None:
    _api_module, result, grid, c, tcm, tg = thermal_square
    for direction in (_mixed_direction(grid, tg), _second_direction(grid, tg)):
        analytic = result.jacobian @ direction
        residuals = []
        for epsilon in (3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4):
            centered = _centered_directional_rhs(
                grid=grid,
                pair_cloglog=c,
                temperature_cm=tcm,
                temperature_gamma=tg,
                direction=direction,
                epsilon=epsilon,
                config=_config(),
                elapsed=11.0,
            )
            residuals.append(_block_relative(analytic, centered, grid.order))
        assert min(residuals) < 8.0e-5
        assert residuals[-1] < residuals[0] / 100.0


def test_matrix_mutations_are_killed_by_original_rhs_witness(
    thermal_square: tuple[ModuleType, object, object, np.ndarray, float, float],
) -> None:
    _api_module, result, grid, c, tcm, tg = thermal_square
    direction = _mixed_direction(grid, tg)
    centered_values = [
        _centered_directional_rhs(
            grid=grid,
            pair_cloglog=c,
            temperature_cm=tcm,
            temperature_gamma=tg,
            direction=direction,
            epsilon=epsilon,
            config=_config(),
        )
        for epsilon in (1.0e-3, 3.0e-4, 1.0e-4)
    ]
    correct_residuals = [
        _block_relative(result.jacobian @ direction, centered, grid.order)
        for centered in centered_values
    ]
    best_index = int(np.argmin(correct_residuals))
    witness = centered_values[best_index]
    best = correct_residuals[best_index]

    transpose = result.jacobian.T.copy()
    swap_flavour_columns = result.jacobian.copy()
    swap_flavour_columns[:, [0, grid.order]] = swap_flavour_columns[
        :, [grid.order, 0]
    ]
    omit_tgamma = result.jacobian.copy()
    omit_tgamma[:, -2] = 0.0
    flip_tgamma = result.jacobian.copy()
    flip_tgamma[:, -2] *= -1.0
    nonzero_elapsed = result.jacobian.copy()
    nonzero_elapsed[:, -1] = result.tgamma_column
    swap_tgamma_elapsed = result.jacobian.copy()
    swap_tgamma_elapsed[:, [-2, -1]] = swap_tgamma_elapsed[:, [-1, -2]]
    swap_output_rows = result.jacobian.copy()
    swap_output_rows[[-2, -1], :] = swap_output_rows[[-1, -2], :]

    mutants = {
        "transpose": transpose,
        "swap-electron-muon-column": swap_flavour_columns,
        "omit-Tgamma-column": omit_tgamma,
        "flip-Tgamma-column": flip_tgamma,
        "inject-nonzero-elapsed-column": nonzero_elapsed,
        "swap-Tgamma-elapsed-columns": swap_tgamma_elapsed,
        "swap-Tgamma-elapsed-output-rows": swap_output_rows,
    }
    for name, mutant in mutants.items():
        residual = _block_relative(mutant @ direction, witness, grid.order)
        assert residual > max(1.0e-5, 30.0 * best), name


def test_equilibrium_and_weak_tail_combined_directional_operator() -> None:
    api = _api()
    cases = []

    equilibrium_grid = ind.build_independent_grid(8, 8.0)
    equilibrium_t = 2.0
    equilibrium_c = ind.pair_logits_to_cloglog(
        np.stack([-equilibrium_grid.nodes for _ in range(3)])
    )
    cases.append(
        (
            equilibrium_grid,
            _config(),
            equilibrium_c,
            equilibrium_t,
            equilibrium_t,
            4.0e-4,
        )
    )
    cases.append((*_weak_tail_case(), 4.0e-4))

    for grid, config, c, tcm, tg, threshold in cases:
        direction = _second_direction(grid, tg)
        direct = api.evaluate_static_rhs_direction_jvp(
            grid=grid,
            pair_cloglog=c,
            full_direction=direction,
            temperature_cm_mev=tcm,
            temperature_gamma_mev=tg,
            config=config,
        )
        residuals = []
        for epsilon in (1.0e-3, 3.0e-4, 1.0e-4):
            centered = _centered_directional_rhs(
                grid=grid,
                pair_cloglog=c,
                temperature_cm=tcm,
                temperature_gamma=tg,
                direction=direction,
                epsilon=epsilon,
                config=config,
            )
            residuals.append(_block_relative(direct.jvp, centered, grid.order))
        assert min(residuals) < threshold
        assert residuals[-1] < residuals[0] / 30.0


@pytest.mark.slow
def test_retained_stiff_state_combined_directional_operator() -> None:
    api = _api()
    path_text = os.environ.get("D080D_STATE_1200")
    if not path_text:
        pytest.skip("retained state is supplied only by the sealed D-080D workflow")
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
    direction = _second_direction(grid, tg)
    direct = api.evaluate_static_rhs_direction_jvp(
        grid=grid,
        pair_cloglog=c,
        full_direction=direction,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    centered = _centered_directional_rhs(
        grid=grid,
        pair_cloglog=c,
        temperature_cm=tcm,
        temperature_gamma=tg,
        direction=direction,
        epsilon=2.0e-4,
        config=config,
        elapsed=float(state[-1]),
    )
    assert _block_relative(direct.jvp, centered, grid.order) < 6.0e-3
    assert np.array_equal(direct.elapsed_component, np.zeros_like(direct.jvp))
