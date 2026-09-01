from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPTS = ROOT / "scripts" / "audit"
if str(AUDIT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(AUDIT_SCRIPTS))

from _d078_logit_linearization import (  # noqa: E402
    LinearizationKinkError,
    logit_chain_terms,
    push_forward_auxiliary_column,
    push_forward_occupation_jacobian,
    push_forward_occupation_jvp,
)
from _d078_tangent_certificate import (  # noqa: E402
    TangentStatus,
    certify_directional_derivative,
)


def _logit(occupation: np.ndarray) -> np.ndarray:
    return np.log(occupation) - np.log1p(-occupation)


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    positive = logit >= 0.0
    out = np.empty_like(logit, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exp_value = np.exp(logit[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def test_unfloored_push_forward_matches_central_difference() -> None:
    occupation = np.array([0.2, 0.4, 0.7])
    matrix = np.array(
        [
            [-1.4, 0.3, -0.2],
            [0.5, -0.9, 0.4],
            [-0.1, 0.2, -0.7],
        ]
    )
    offset = np.array([0.07, -0.03, 0.05])
    physical_rhs = matrix @ occupation + offset
    result = push_forward_occupation_jacobian(
        occupation,
        physical_rhs,
        matrix,
        floor=1.0e-12,
    )

    base_logit = _logit(occupation)

    def transformed_rhs(logit: np.ndarray) -> np.ndarray:
        f = _sigmoid(logit)
        chain = np.maximum(f * (1.0 - f), 1.0e-12)
        return (matrix @ f + offset) / chain

    step = 2.0e-6
    finite_difference = np.empty_like(result.jacobian)
    for column in range(occupation.size):
        direction = np.zeros_like(base_logit)
        direction[column] = step
        finite_difference[:, column] = (
            transformed_rhs(base_logit + direction)
            - transformed_rhs(base_logit - direction)
        ) / (2.0 * step)

    np.testing.assert_allclose(
        result.jacobian,
        finite_difference,
        rtol=3.0e-7,
        atol=3.0e-9,
    )
    assert not np.any(result.terms.floor_active)
    assert not np.any(result.terms.kink)


def test_equilibrium_push_forward_is_similarity_transform() -> None:
    occupation = np.array([0.15, 0.35, 0.65, 0.85])
    occupation_jacobian = np.array(
        [
            [-2.0, 0.2, 0.0, 0.1],
            [0.4, -1.3, 0.3, 0.0],
            [0.0, -0.2, -0.8, 0.5],
            [0.1, 0.0, 0.2, -1.1],
        ]
    )
    result = push_forward_occupation_jacobian(
        occupation,
        np.zeros(occupation.size),
        occupation_jacobian,
    )
    chain = occupation * (1.0 - occupation)
    expected = (occupation_jacobian * chain[np.newaxis, :]) / chain[:, np.newaxis]
    np.testing.assert_allclose(result.jacobian, expected, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        np.sort_complex(np.linalg.eigvals(result.jacobian)),
        np.sort_complex(np.linalg.eigvals(occupation_jacobian)),
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_array_equal(result.correction_diagonal, np.zeros(4))


def test_clamped_row_uses_raw_input_chain_and_zero_floor_derivative() -> None:
    occupation = np.array([1.0e-15, 0.3])
    physical_rhs = np.array([2.0e-5, -0.04])
    occupation_jacobian = np.array([[-3.0, 0.7], [0.2, -1.1]])
    floor = 1.0e-12
    result = push_forward_occupation_jacobian(
        occupation,
        physical_rhs,
        occupation_jacobian,
        floor=floor,
    )
    raw = occupation * (1.0 - occupation)
    effective = np.maximum(raw, floor)
    derivative = np.array([0.0, raw[1] * (1.0 - 2.0 * occupation[1])])
    expected = (occupation_jacobian * raw[np.newaxis, :]) / effective[:, np.newaxis]
    expected[np.diag_indices(2)] -= physical_rhs * derivative / effective**2
    np.testing.assert_allclose(result.jacobian, expected, rtol=0.0, atol=2.0e-15)
    assert result.terms.floor_active.tolist() == [True, False]
    assert result.terms.derivative_wrt_logit[0] == 0.0
    assert result.correction_diagonal[0] == 0.0


def test_exact_floor_kink_is_refused() -> None:
    occupation = np.array([0.5])
    terms = logit_chain_terms(occupation, floor=0.25)
    assert terms.kink.tolist() == [True]
    with pytest.raises(LinearizationKinkError):
        push_forward_occupation_jacobian(
            occupation,
            np.array([0.0]),
            np.array([[-1.0]]),
            floor=0.25,
        )


def test_jvp_matches_matrix_push_forward() -> None:
    occupation = np.array([0.12, 0.31, 0.72])
    physical_rhs = np.array([0.03, -0.07, 0.02])
    occupation_jacobian = np.array(
        [[-1.1, 0.2, 0.4], [0.3, -0.8, 0.1], [-0.2, 0.5, -1.4]]
    )
    direction = np.array([0.6, -0.4, 0.2])
    raw_chain = occupation * (1.0 - occupation)
    occupation_jvp = occupation_jacobian @ (raw_chain * direction)

    matrix_result = push_forward_occupation_jacobian(
        occupation,
        physical_rhs,
        occupation_jacobian,
    )
    jvp_result = push_forward_occupation_jvp(
        occupation,
        physical_rhs,
        occupation_jvp,
        direction,
    )
    np.testing.assert_allclose(
        jvp_result.jvp,
        matrix_result.jacobian @ direction,
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_auxiliary_column_divides_by_effective_chain_once() -> None:
    occupation = np.array([1.0e-15, 0.2, 0.6])
    partial_rhs = np.array([0.4, -0.3, 0.2])
    result = push_forward_auxiliary_column(occupation, partial_rhs, floor=1.0e-12)
    expected = partial_rhs / np.maximum(occupation * (1.0 - occupation), 1.0e-12)
    np.testing.assert_allclose(result.column, expected, rtol=0.0, atol=0.0)


def _manufactured_rhs(state: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.sin(state[0]) + state[1] ** 2,
            state[0] * state[1] + np.exp(0.2 * state[0]),
        ]
    )


def _manufactured_jvp(state: np.ndarray, direction: np.ndarray) -> np.ndarray:
    jacobian = np.array(
        [
            [np.cos(state[0]), 2.0 * state[1]],
            [state[1] + 0.2 * np.exp(0.2 * state[0]), state[0]],
        ]
    )
    return jacobian @ direction


def test_directional_certificate_accepts_correct_same_branch_jvp() -> None:
    state = np.array([0.4, -0.3])
    direction = np.array([0.7, 0.2])
    certificate = certify_directional_derivative(
        _manufactured_rhs,
        state,
        direction,
        _manufactured_jvp(state, direction),
        epsilons=[1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5],
        rtol=2.0e-6,
        atol=1.0e-11,
        min_valid_samples=4,
        required_consecutive_passes=2,
    )
    assert certificate.status is TangentStatus.CERTIFIED
    assert certificate.best_normalized_residual < 2.0e-6
    assert certificate.max_consecutive_passes >= 2


@pytest.mark.parametrize("mutation", ["sign", "scale"])
def test_directional_certificate_kills_sign_and_scale_mutations(mutation: str) -> None:
    state = np.array([0.4, -0.3])
    direction = np.array([0.7, 0.2])
    exact = _manufactured_jvp(state, direction)
    mutated = -exact if mutation == "sign" else 1.01 * exact
    certificate = certify_directional_derivative(
        _manufactured_rhs,
        state,
        direction,
        mutated,
        epsilons=[1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5],
        rtol=2.0e-6,
        atol=1.0e-11,
        min_valid_samples=4,
        required_consecutive_passes=2,
    )
    assert certificate.status is TangentStatus.UNRESOLVED
    assert certificate.max_consecutive_passes == 0


def test_directional_certificate_types_branch_crossing() -> None:
    certificate = certify_directional_derivative(
        lambda state: state.copy(),
        np.array([0.0]),
        np.array([1.0]),
        np.array([1.0]),
        epsilons=[1.0e-1, 1.0e-2, 1.0e-3],
        branch_signature=lambda state: bool(state[0] >= 0.0),
        min_valid_samples=2,
    )
    assert certificate.status is TangentStatus.BRANCH_CROSSING
    assert certificate.valid_samples == 0


def test_directional_certificate_types_strict_domain_exit() -> None:
    certificate = certify_directional_derivative(
        lambda state: state**2,
        np.array([0.1]),
        np.array([1.0]),
        np.array([0.2]),
        epsilons=[0.3, 0.2],
        state_validator=lambda state: bool(np.all(state > 0.0)),
        min_valid_samples=1,
        required_consecutive_passes=1,
    )
    assert certificate.status is TangentStatus.OUT_OF_DOMAIN
    assert certificate.valid_samples == 0


def test_directional_certificate_is_transactional() -> None:
    state = np.array([0.4, -0.3])
    direction = np.array([0.7, 0.2])
    state_before = state.copy()
    direction_before = direction.copy()

    def hostile_rhs(candidate: np.ndarray) -> np.ndarray:
        value = _manufactured_rhs(candidate)
        candidate.fill(123.0)
        return value

    certificate = certify_directional_derivative(
        hostile_rhs,
        state,
        direction,
        _manufactured_jvp(state, direction),
        epsilons=[1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5],
        rtol=2.0e-6,
        min_valid_samples=3,
    )
    assert certificate.status is TangentStatus.CERTIFIED
    np.testing.assert_array_equal(state, state_before)
    np.testing.assert_array_equal(direction, direction_before)
