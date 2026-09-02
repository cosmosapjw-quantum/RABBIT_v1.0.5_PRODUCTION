from __future__ import annotations

import importlib

import numpy as np
import pytest


def _api():
    try:
        return importlib.import_module("scripts.audit._d080f_action_metric")
    except ModuleNotFoundError:
        pytest.fail("D-080F cancellation-aware action metric is not implemented")


def test_contribution_scale_does_not_false_fail_near_cancellation() -> None:
    api = _api()
    order = 1
    size = 3 * order + 2
    jacobian = np.zeros((size, size), dtype=np.float64)
    jacobian[0, 0] = 1.0e8
    jacobian[0, 1] = -1.0e8
    jacobian[1, 0] = 2.0
    jacobian[1, 1] = 3.0
    direction = np.array([1.0, 1.0 - 1.0e-12, 0.0, 0.0, 0.0])
    exact_action = jacobian @ direction
    witness = exact_action.copy()
    witness[0] += 1.0e-8

    ordinary = abs(exact_action[0] - witness[0]) / max(
        abs(exact_action[0]), abs(witness[0])
    )
    report = api.matrix_action_block_residual(
        jacobian=jacobian,
        direction=direction,
        reference_action=witness,
        order=order,
    )

    assert ordinary > 1.0e-5
    assert report.spectral < 1.0e-12
    assert report.maximum == report.spectral
    assert report.spectral_contribution_scale >= 1.9e8


def test_basis_direction_reduces_to_column_relative_scale() -> None:
    api = _api()
    order = 1
    size = 3 * order + 2
    jacobian = np.zeros((size, size), dtype=np.float64)
    jacobian[:3, 1] = np.array([2.0, -3.0, 6.0])
    direction = np.zeros(size, dtype=np.float64)
    direction[1] = 1.0
    candidate = jacobian @ direction
    witness = candidate.copy()
    witness[:3] *= 1.0 + 1.0e-8

    expected = np.linalg.norm(candidate[:3] - witness[:3]) / max(
        np.linalg.norm(candidate[:3]), np.linalg.norm(witness[:3])
    )
    report = api.matrix_action_block_residual(
        jacobian=jacobian,
        direction=direction,
        reference_action=witness,
        order=order,
    )

    assert report.spectral == pytest.approx(expected, rel=2.0e-15)
    assert report.spectral_contribution_scale == pytest.approx(
        np.linalg.norm(jacobian[:3, 1]), rel=2.0e-15
    )


def test_column_mutation_remains_visible() -> None:
    api = _api()
    order = 1
    size = 3 * order + 2
    correct = np.zeros((size, size), dtype=np.float64)
    correct[:3, 0] = np.array([1.0, 2.0, -4.0])
    correct[:3, 1] = np.array([3.0, -1.0, 0.5])
    direction = np.array([0.7, -0.4, 0.0, 0.0, 0.0])
    witness = correct @ direction
    mutated = correct.copy()
    mutated[:3, 0] *= 1.01

    report = api.matrix_action_block_residual(
        jacobian=mutated,
        direction=direction,
        reference_action=witness,
        order=order,
    )

    assert report.spectral > 1.0e-3
    assert report.maximum > 1.0e-3


def test_shape_and_nonfinite_inputs_fail_closed() -> None:
    api = _api()
    with pytest.raises(ValueError):
        api.matrix_action_block_residual(
            jacobian=np.eye(4),
            direction=np.ones(4),
            reference_action=np.ones(4),
            order=1,
        )
    bad = np.eye(5)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        api.matrix_action_block_residual(
            jacobian=bad,
            direction=np.ones(5),
            reference_action=np.ones(5),
            order=1,
        )
