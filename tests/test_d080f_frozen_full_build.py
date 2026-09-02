from __future__ import annotations

import importlib
from types import ModuleType

import numpy as np
import pytest

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_rhs_jvp import evaluate_c_only_rhs_jvp
from scripts.audit._d080d_static_jacobian import (
    assemble_static_jacobian,
    rhs_block_relative,
)
from scripts.audit._d080e_prepared_jvp import evaluate_prepared_c_only_rhs_jvp


def _api() -> ModuleType:
    try:
        return importlib.import_module("scripts.audit._d080f_frozen_full_build")
    except ModuleNotFoundError:
        pytest.fail("D-080F frozen full-build module is not present")


def _case(order: int = 4) -> tuple[
    ind.IndependentNoQkeGrid,
    ind.IndependentCollisionConfig,
    np.ndarray,
    float,
    float,
]:
    grid = ind.build_independent_grid(order, 6.0)
    config = ind.IndependentCollisionConfig(
        incoming_polar_order=2,
        final_polar_order=2,
        final_azimuth_order=4,
        electron_radial_order=6,
    )
    logits = np.stack(
        [
            -grid.nodes + 0.025 * np.exp(-grid.nodes / 3.0),
            -grid.nodes - 0.0125 * np.exp(-grid.nodes / 4.0),
            -grid.nodes - 0.0125 * np.exp(-grid.nodes / 4.0),
        ]
    )
    return grid, config, ind.pair_logits_to_cloglog(logits), 2.0, 2.04


def _direction(order: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, order)
    direction = np.stack(
        (
            np.cos(0.8 * x),
            -0.6 * np.sin(1.1 * x + 0.2),
            0.3 + x * x,
        )
    )
    return direction / np.linalg.norm(direction)


def test_prepare_and_seal_is_content_addressed_and_read_only() -> None:
    api = _api()
    grid, config, c, tcm, tg = _case()
    sealed = api.prepare_and_seal_static_rhs(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
        max_modal_basis_bytes=64 * 1024**2,
    )

    assert sealed.seal.schema == "rabbit.d080f.prepared_state_seal.v1"
    assert len(sealed.seal.fingerprint_sha256) == 64
    assert sealed.seal.array_count > 0
    assert sealed.seal.unique_array_bytes > 0
    assert sealed.seal.cache_unique_bytes > 0
    assert sealed.seal.all_arrays_readonly is True
    assert api.verify_prepared_state_seal(sealed) is True

    with pytest.raises(ValueError):
        sealed.prepared.pair_cloglog[0, 0] += 1.0

    changed = c.copy()
    changed[0, 0] += 1.0e-7
    other = api.prepare_and_seal_static_rhs(
        grid=grid,
        pair_cloglog=changed,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
        max_modal_basis_bytes=64 * 1024**2,
    )
    assert other.seal.fingerprint_sha256 != sealed.seal.fingerprint_sha256


def test_seal_survives_repeated_prepared_jvp_without_cache_growth() -> None:
    api = _api()
    grid, config, c, tcm, tg = _case()
    sealed = api.prepare_and_seal_static_rhs(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
        max_modal_basis_bytes=64 * 1024**2,
    )
    before = sealed.prepared.cache.snapshot()
    candidate = evaluate_prepared_c_only_rhs_jvp(
        sealed.prepared,
        _direction(grid.order),
    )
    after = sealed.prepared.cache.snapshot()

    assert np.all(np.isfinite(candidate.jvp))
    assert api.verify_prepared_state_seal(sealed) is True
    for key in (
        "kinematic_misses",
        "self_matrix_misses",
        "electron_matrix_misses",
        "modal_basis_misses",
        "kinematic_entries",
        "self_matrix_entries",
        "electron_matrix_entries",
        "modal_basis_entries",
        "estimated_cache_bytes",
    ):
        assert after[key] == before[key], key


def test_sealed_small_matrix_matches_d080d_and_serial_oracle() -> None:
    api = _api()
    grid, config, c, tcm, tg = _case()
    reference = assemble_static_jacobian(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    sealed = api.prepare_and_seal_static_rhs(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
        max_modal_basis_bytes=64 * 1024**2,
    )
    result = api.assemble_sealed_static_jacobian(
        sealed,
        direction_block_size=4,
        serial_oracle_columns=(0, grid.order, 2 * grid.order),
    )

    size = 3 * grid.order + 2
    assert result.jacobian.shape == (size, size)
    assert np.array_equal(result.jacobian[:, -1], np.zeros(size))
    assert result.seal_before_sha256 == result.seal_after_sha256
    assert result.maximum_serial_oracle_residual < 3.0e-12
    assert result.maximum_prepared_action_residual < 3.0e-12
    assert result.cache_miss_delta == 0
    assert result.cache_entry_delta == 0
    assert len(result.matrix_sha256) == 64

    scale = max(np.linalg.norm(reference.jacobian), np.finfo(float).tiny)
    assert np.linalg.norm(result.jacobian - reference.jacobian) / scale < 3.0e-12

    direction = _direction(grid.order)
    serial = evaluate_c_only_rhs_jvp(
        grid=grid,
        pair_cloglog=c,
        direction_cloglog=direction,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
    )
    full_direction = np.concatenate((direction.ravel(), [0.0, 0.0]))
    assert rhs_block_relative(
        result.jacobian @ full_direction,
        serial.jvp,
        grid.order,
    ) < 3.0e-12


def test_route_decision_requires_measured_build_and_immutable_cache() -> None:
    api = _api()
    assert api.classify_full_build_route(
        build_seconds=420.0,
        cache_bytes=300_000_000,
        maximum_correctness_residual=1.0e-12,
        seal_unchanged=True,
        cache_miss_delta=0,
        cache_entry_delta=0,
        full_matrix_measured=True,
        wall_budget_seconds=900.0,
        cache_budget_bytes=2 * 1024**3,
    ) == "EXPLICIT_CALLBACK_CANDIDATE"

    assert api.classify_full_build_route(
        build_seconds=420.0,
        cache_bytes=300_000_000,
        maximum_correctness_residual=1.0e-12,
        seal_unchanged=False,
        cache_miss_delta=0,
        cache_entry_delta=0,
        full_matrix_measured=True,
        wall_budget_seconds=900.0,
        cache_budget_bytes=2 * 1024**3,
    ) == "PREPARED_STATE_INTEGRITY_FAILED"

    assert api.classify_full_build_route(
        build_seconds=1200.0,
        cache_bytes=300_000_000,
        maximum_correctness_residual=1.0e-12,
        seal_unchanged=True,
        cache_miss_delta=0,
        cache_entry_delta=0,
        full_matrix_measured=True,
        wall_budget_seconds=900.0,
        cache_budget_bytes=2 * 1024**3,
    ) == "MATRIX_FREE_OR_SPLIT_CANDIDATE"
