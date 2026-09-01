from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_collision_jvp import evaluate_collision_action_jvp
from scripts.audit._d079_rhs_jvp import (
    c_only_state_validator,
    evaluate_c_only_rhs_jvp,
    evaluate_static_rhs_from_packed_state,
)
from scripts.audit._d079_tangent_primitives import (
    cloglog_chart_tangent,
    pauli_gain_minus_loss_jvp,
)


def _setup(order: int = 8, y_max: float = 8.0):
    grid = ind.build_independent_grid(order, y_max)
    config = ind.IndependentCollisionConfig(
        incoming_polar_order=2,
        final_polar_order=2,
        final_azimuth_order=4,
        electron_radial_order=8,
    )
    logits = np.stack([
        -grid.nodes + 0.04 * np.exp(-grid.nodes / 3.0),
        -grid.nodes - 0.02 * np.exp(-grid.nodes / 4.0),
        -grid.nodes - 0.02 * np.exp(-grid.nodes / 4.0),
    ])
    c = ind.pair_logits_to_cloglog(logits)
    x = np.linspace(-1.0, 1.0, order)
    direction = np.stack((0.3 + x, -0.2 + x**2, -0.2 + x**2))
    direction /= np.linalg.norm(direction)
    return grid, config, c, direction


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(np.linalg.norm(left), np.linalg.norm(right), np.finfo(float).tiny)
    return float(np.linalg.norm(left - right) / scale)


def test_cloglog_chart_tangent_matches_centered_difference() -> None:
    c = np.array([[-4.0, -1.0, 0.2, 1.5]])
    v = np.array([[0.4, -0.2, 0.1, 0.3]])
    df, du, dlogq = cloglog_chart_tangent(c, v)
    epsilon = 1.0e-6
    f_plus = ind.cloglog_to_occupation(c + epsilon * v)
    f_minus = ind.cloglog_to_occupation(c - epsilon * v)
    u_plus = ind._native_pair_logits(np.repeat(c + epsilon * v, 3, axis=0))[0]
    u_minus = ind._native_pair_logits(np.repeat(c - epsilon * v, 3, axis=0))[0]
    q_plus = ind.cloglog_chain_factor(c + epsilon * v)
    q_minus = ind.cloglog_chain_factor(c - epsilon * v)
    assert np.allclose(df, (f_plus - f_minus) / (2.0 * epsilon), rtol=2e-9, atol=1e-12)
    assert np.allclose(du[0], (u_plus - u_minus) / (2.0 * epsilon), rtol=2e-9, atol=1e-11)
    assert np.allclose(dlogq, (np.log(q_plus) - np.log(q_minus)) / (2.0 * epsilon), rtol=2e-9, atol=1e-11)


def test_pauli_tangent_matches_centered_difference_and_detailed_balance() -> None:
    u = np.array([0.3, -0.7, 0.2, -0.1])
    du = np.array([0.4, -0.2, 0.1, 0.7])
    analytic = pauli_gain_minus_loss_jvp(*u, *du)
    epsilon = 1.0e-6
    plus = ind._stable_pauli_gain_minus_loss(*(u + epsilon * du))
    minus = ind._stable_pauli_gain_minus_loss(*(u - epsilon * du))
    assert np.allclose(analytic, (plus - minus) / (2.0 * epsilon), rtol=2e-9, atol=1e-13)

    u1, u2, u3 = 0.2, -0.4, 0.7
    u4 = u1 + u2 - u3
    tangent = pauli_gain_minus_loss_jvp(u1, u2, u3, u4, *du)
    gain = (1-expit(u1)) * (1-expit(u2)) * expit(u3) * expit(u4)
    expected = gain * (du[2] + du[3] - du[0] - du[1])
    assert np.allclose(tangent, expected, rtol=2e-14, atol=2e-16)


def test_full_collision_jvp_matches_same_path_centered_difference() -> None:
    grid, config, c, direction = _setup()
    result = evaluate_collision_action_jvp(
        grid=grid, pair_cloglog=c, direction_cloglog=direction,
        temperature_cm_mev=2.0, temperature_gamma_mev=2.05, config=config,
    )
    residuals = []
    for epsilon in (3e-4, 1e-4, 3e-5):
        plus = ind.evaluate_independent_collision_action(
            grid=grid, pair_cloglog=c + epsilon * direction,
            temperature_cm_mev=2.0, temperature_gamma_mev=2.05, config=config,
        ).total
        minus = ind.evaluate_independent_collision_action(
            grid=grid, pair_cloglog=c - epsilon * direction,
            temperature_cm_mev=2.0, temperature_gamma_mev=2.05, config=config,
        ).total
        residuals.append(_relative(result.total, (plus - minus) / (2.0 * epsilon)))
    assert min(residuals) < 2.0e-6
    assert result.first_law_tangent_residual < 2.0e-11
    assert abs(result.self_number_moment) < 2.0e-10 * max(np.linalg.norm(result.self_interaction), 1.0)
    assert abs(result.self_energy_moment) < 2.0e-10 * max(np.linalg.norm(result.self_interaction), 1.0)
    assert result.charge_conjugation_residual < 2.0e-12
    assert result.mu_tau_residual < 2.0e-12


def test_full_static_rhs_jvp_and_mutation_kills() -> None:
    grid, config, c, direction = _setup()
    result = evaluate_c_only_rhs_jvp(
        grid=grid, pair_cloglog=c, direction_cloglog=direction,
        temperature_cm_mev=2.0, temperature_gamma_mev=2.05, config=config,
    )
    packed = np.concatenate((c.ravel(), [2.05, 0.0]))
    assert c_only_state_validator(grid, packed)
    residuals = []
    centered_best = None
    for epsilon in (3e-4, 1e-4, 3e-5):
        centered = (
            evaluate_static_rhs_from_packed_state(
                grid=grid, packed_state=packed + epsilon * result.full_direction,
                temperature_cm_mev=2.0, config=config,
            )
            - evaluate_static_rhs_from_packed_state(
                grid=grid, packed_state=packed - epsilon * result.full_direction,
                temperature_cm_mev=2.0, config=config,
            )
        ) / (2.0 * epsilon)
        residual = _relative(result.jvp, centered)
        residuals.append(residual)
        if centered_best is None or residual == min(residuals):
            centered_best = centered
    assert min(residuals) < 3.0e-6
    assert centered_best is not None
    assert _relative(-result.jvp, centered_best) > 1.0
    assert _relative(1.01 * result.jvp, centered_best) > 5.0e-3

    swapped = result.collision.total.copy()
    swapped[[0, 2]] = swapped[[2, 0]]
    assert _relative(swapped, result.collision.total) > 1.0e-3
    assert _relative(result.collision.self_interaction, result.collision.total) > 1.0e-3


@pytest.mark.slow
def test_retained_r4_region_state_has_admissible_local_c_direction() -> None:
    path_text = os.environ.get("D079_STATE_1200")
    if not path_text:
        pytest.skip("retained state is supplied only by the sealed D-079 workflow")
    path = Path(path_text)
    with np.load(path, allow_pickle=False) as archive:
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
    phase = np.linspace(0.0, np.pi, order)
    direction = np.stack((np.cos(phase), np.sin(phase), np.sin(phase)))
    direction /= np.linalg.norm(direction)
    result = evaluate_c_only_rhs_jvp(
        grid=grid, pair_cloglog=c, direction_cloglog=direction,
        temperature_cm_mev=tcm, temperature_gamma_mev=tg, config=config,
    )
    packed = np.concatenate((c.ravel(), [tg, float(state[-1])]))
    epsilon = 3.0e-6
    assert c_only_state_validator(grid, packed + epsilon * result.full_direction)
    assert c_only_state_validator(grid, packed - epsilon * result.full_direction)
    centered = (
        evaluate_static_rhs_from_packed_state(
            grid=grid, packed_state=packed + epsilon * result.full_direction,
            temperature_cm_mev=tcm, config=config,
        )
        - evaluate_static_rhs_from_packed_state(
            grid=grid, packed_state=packed - epsilon * result.full_direction,
            temperature_cm_mev=tcm, config=config,
        )
    ) / (2.0 * epsilon)
    assert _relative(result.jvp, centered) < 2.0e-4
    assert result.collision.first_law_tangent_residual < 2.0e-9
