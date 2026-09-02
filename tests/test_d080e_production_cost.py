from __future__ import annotations

import importlib
from types import ModuleType

import numpy as np
import pytest

from rabbit.decoupling import _independent_noqke as ind


def _api() -> ModuleType:
    try:
        return importlib.import_module("scripts.audit._d080e_production_cost")
    except ModuleNotFoundError:
        pytest.fail("D-080E production-cost module is not present")


def _small_case() -> tuple[ind.IndependentNoQkeGrid, ind.IndependentCollisionConfig, np.ndarray]:
    grid = ind.build_independent_grid(4, 6.0)
    config = ind.IndependentCollisionConfig(
        incoming_polar_order=2,
        final_polar_order=2,
        final_azimuth_order=4,
        electron_radial_order=6,
    )
    logits = np.stack(
        [
            -grid.nodes + 0.02 * np.exp(-grid.nodes / 3.0),
            -grid.nodes - 0.01 * np.exp(-grid.nodes / 4.0),
            -grid.nodes - 0.01 * np.exp(-grid.nodes / 4.0),
        ]
    )
    return grid, config, ind.pair_logits_to_cloglog(logits)


def test_direction_family_is_deterministic_normalized_and_distinct() -> None:
    api = _api()
    grid, _config, _state = _small_case()
    first = api.deterministic_spectral_directions(grid, 4)
    second = api.deterministic_spectral_directions(grid, 4)

    assert first.shape == (4, 3, grid.order)
    assert np.array_equal(first, second)
    np.testing.assert_allclose(
        np.linalg.norm(first.reshape(4, -1), axis=1),
        np.ones(4),
        rtol=0.0,
        atol=2.0e-15,
    )
    gram = first.reshape(4, -1) @ first.reshape(4, -1).T
    assert np.max(np.abs(gram - np.diag(np.diag(gram)))) < 0.95


def test_serial_profile_reuses_one_state_contract_and_returns_finite_timings() -> None:
    api = _api()
    grid, config, state = _small_case()
    profile = api.profile_serial_spectral_jvps(
        grid=grid,
        pair_cloglog=state,
        temperature_cm_mev=2.0,
        temperature_gamma_mev=2.03,
        config=config,
        direction_count=2,
        warmup_count=0,
    )

    assert profile.direction_count == 2
    assert profile.state_size == 3 * grid.order + 2
    assert profile.spectral_size == 3 * grid.order
    assert profile.jvp_seconds.shape == (2,)
    assert np.all(np.isfinite(profile.jvp_seconds))
    assert np.all(profile.jvp_seconds > 0.0)
    assert profile.maximum_base_rhs_residual < 5.0e-13
    assert profile.maximum_first_law_tangent_residual < 2.0e-9
    assert profile.maximum_matrix_action_self_consistency < 5.0e-14


def test_cost_model_and_storage_are_dimensionally_explicit() -> None:
    api = _api()
    estimate = api.estimate_explicit_jacobian_cost(
        order=60,
        jvp_seconds=np.array([2.0, 3.0, 4.0]),
        primal_seconds=1.25,
        tgamma_seconds=0.5,
    )

    assert estimate.state_size == 182
    assert estimate.spectral_columns == 180
    assert estimate.matrix_storage_bytes == 182 * 182 * 8
    assert estimate.median_jvp_seconds == 3.0
    assert estimate.projected_serial_seconds == 180 * 3.0 + 0.5
    assert estimate.projected_base_cached_seconds == 1.25 + 180 * (3.0 - 1.25) + 0.5
    assert estimate.projected_base_cached_seconds < estimate.projected_serial_seconds
    assert estimate.reusable_base_fraction_lower_bound == pytest.approx(1.25 / 3.0)


def test_route_decision_fails_closed_without_production_order_build_evidence() -> None:
    api = _api()
    decision = api.classify_construction_route(
        projected_serial_seconds=4.0 * 3600.0,
        projected_base_cached_seconds=2.5 * 3600.0,
        full_matrix_measured=False,
        batched_kernel_measured=False,
        wall_budget_seconds=1800.0,
    )
    assert decision == "EXPLICIT_CONSTRUCTION_NOT_YET_ADMISSIBLE"

    admitted = api.classify_construction_route(
        projected_serial_seconds=1200.0,
        projected_base_cached_seconds=800.0,
        full_matrix_measured=True,
        batched_kernel_measured=True,
        wall_budget_seconds=1800.0,
    )
    assert admitted == "EXPLICIT_CALLBACK_CANDIDATE"
