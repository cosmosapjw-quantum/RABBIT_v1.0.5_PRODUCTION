from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_rhs_jvp import evaluate_c_only_rhs_jvp
from scripts.audit._d080d_static_jacobian import assemble_static_jacobian, rhs_block_relative


def _api() -> ModuleType:
    try:
        return importlib.import_module("scripts.audit._d080e_prepared_jvp")
    except ModuleNotFoundError:
        pytest.fail("D-080E implementation module is not present")


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


def _directions(order: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, order)
    directions = np.stack(
        [
            np.stack((np.cos(1.3 * x), np.sin(0.7 * x), x)),
            np.stack((x * x - 0.2, np.cos(0.4 + x), -np.sin(1.1 * x))),
            np.stack((np.exp(-x * x), x - x.mean(), np.cos(2.0 * x))),
        ]
    )
    norms = np.linalg.norm(directions.reshape(directions.shape[0], -1), axis=1)
    return directions / norms[:, None, None]


def test_prepared_multi_direction_jvp_matches_frozen_serial_oracle() -> None:
    api = _api()
    grid, config, c, tcm, tg = _thermal_case()
    directions = _directions(grid.order)
    original_kinematics = ind._two_body_kinematics
    original_self_matrix = ind._self_matrix
    original_electron_matrix = ind._electron_matrix

    prepared = api.prepare_static_rhs_reuse(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    before = prepared.cache.snapshot()
    batched = api.evaluate_prepared_c_only_rhs_jvps(prepared, directions)
    after = prepared.cache.snapshot()

    assert batched.shape == (directions.shape[0], 3 * grid.order + 2)
    for direction, candidate in zip(directions, batched):
        reference = evaluate_c_only_rhs_jvp(
            grid=grid,
            pair_cloglog=c,
            direction_cloglog=direction,
            temperature_cm_mev=tcm,
            temperature_gamma_mev=tg,
            config=config,
        )
        assert rhs_block_relative(candidate, reference.jvp, grid.order) < 2.0e-12

    assert after["kinematic_requests"] > before["kinematic_requests"]
    assert after["kinematic_misses"] == before["kinematic_misses"]
    assert after["self_matrix_misses"] == before["self_matrix_misses"]
    assert after["electron_matrix_misses"] == before["electron_matrix_misses"]
    assert after["modal_basis_misses"] == before["modal_basis_misses"]
    assert after["kinematic_hits"] > before["kinematic_hits"]
    assert after["matrix_hits"] > before["matrix_hits"]
    assert after["modal_basis_hits"] > before["modal_basis_hits"]

    assert ind._two_body_kinematics is original_kinematics
    assert ind._self_matrix is original_self_matrix
    assert ind._electron_matrix is original_electron_matrix


def test_prepared_order8_square_matrix_matches_d080d_explicit_matrix() -> None:
    api = _api()
    grid, config, c, tcm, tg = _thermal_case()
    reference = assemble_static_jacobian(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    candidate = api.assemble_prepared_static_jacobian(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
        direction_block_size=4,
    )
    assert candidate.jacobian.shape == reference.jacobian.shape == (26, 26)
    assert np.array_equal(candidate.jacobian[:, -1], np.zeros(26))
    assert rhs_block_relative(candidate.base_rhs, reference.base_rhs, grid.order) < 5.0e-14
    scale = max(np.linalg.norm(reference.jacobian), np.finfo(float).tiny)
    assert np.linalg.norm(candidate.jacobian - reference.jacobian) / scale < 3.0e-12
    assert candidate.maximum_serial_oracle_residual < 3.0e-12
    assert candidate.cache_snapshot["kinematic_hits"] > 0
    assert candidate.cache_snapshot["matrix_hits"] > 0
    assert candidate.cache_snapshot["modal_basis_hits"] > 0


def test_cache_policy_mutations_preserve_physics_but_reduce_reuse() -> None:
    api = _api()
    grid, config, c, tcm, tg = _thermal_case()
    direction = _directions(grid.order)[0]
    full = api.prepare_static_rhs_reuse(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
        policy=api.FixedStateReusePolicy(),
    )
    no_modal = api.prepare_static_rhs_reuse(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
        policy=api.FixedStateReusePolicy(cache_modal_basis=False),
    )
    no_kinematics = api.prepare_static_rhs_reuse(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
        policy=api.FixedStateReusePolicy(
            cache_kinematics=False,
            cache_matrices=False,
        ),
    )
    full_value = api.evaluate_prepared_c_only_rhs_jvp(full, direction).jvp
    no_modal_value = api.evaluate_prepared_c_only_rhs_jvp(no_modal, direction).jvp
    no_kinematics_value = api.evaluate_prepared_c_only_rhs_jvp(no_kinematics, direction).jvp
    assert rhs_block_relative(full_value, no_modal_value, grid.order) < 2.0e-12
    assert rhs_block_relative(full_value, no_kinematics_value, grid.order) < 2.0e-12
    assert full.cache.snapshot()["modal_basis_hits"] > 0
    assert no_modal.cache.snapshot()["modal_basis_hits"] == 0
    assert full.cache.snapshot()["kinematic_hits"] > 0
    assert no_kinematics.cache.snapshot()["kinematic_hits"] == 0
    assert no_kinematics.cache.snapshot()["matrix_hits"] == 0

    with pytest.raises(ValueError, match="matrix caching requires"):
        api.FixedStateReusePolicy(
            cache_kinematics=False,
            cache_matrices=True,
        )


def test_invalid_direction_batch_and_nested_cache_fail_closed() -> None:
    api = _api()
    grid, config, c, tcm, tg = _thermal_case()
    prepared = api.prepare_static_rhs_reuse(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    with pytest.raises(ValueError):
        api.evaluate_prepared_c_only_rhs_jvps(
            prepared,
            np.zeros((2, 3, grid.order + 1)),
        )
    with prepared.cache.patch():
        with pytest.raises(api.D080EReuseError):
            with prepared.cache.patch():
                pass


@pytest.mark.slow
def test_retained_order60_prepared_direction_matches_serial_oracle() -> None:
    api = _api()
    path_text = os.environ.get("D080E_STATE_1200")
    if not path_text:
        pytest.skip("retained state is supplied only by the sealed D-080E workflow")
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
    direction = _directions(order)[0]

    reference = evaluate_c_only_rhs_jvp(
        grid=grid,
        pair_cloglog=c,
        direction_cloglog=direction,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    prepared = api.prepare_static_rhs_reuse(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
        policy=api.FixedStateReusePolicy(max_modal_basis_bytes=768 * 1024**2),
    )
    candidate = api.evaluate_prepared_c_only_rhs_jvp(prepared, direction)
    assert rhs_block_relative(candidate.jvp, reference.jvp, order) < 5.0e-11
    assert candidate.first_law_tangent_residual < 2.0e-8
    stats = prepared.cache.snapshot()
    assert stats["kinematic_hits"] > 0
    assert stats["matrix_hits"] > 0
    assert stats["estimated_cache_bytes"] > 0
