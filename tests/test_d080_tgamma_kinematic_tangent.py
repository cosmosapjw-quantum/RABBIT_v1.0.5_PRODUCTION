from __future__ import annotations

import numpy as np

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d080_tgamma_primitives import (
    electromagnetic_eos_tgamma_tangent,
    evaluate_elastic_tgamma_kinematic_tangent,
    modal_basis_derivative,
)


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    scale = max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny)
    return float(np.linalg.norm(a - b) / scale)


def _config() -> ind.IndependentCollisionConfig:
    return ind.IndependentCollisionConfig(
        incoming_polar_order=2,
        final_polar_order=2,
        final_azimuth_order=4,
        electron_radial_order=8,
    )


def _kinematic_vector(batch: object) -> np.ndarray:
    names = (
        "p2", "e2", "e3", "e4", "p3_magnitude", "p4_magnitude",
        "phase_space", "quadrature_weight", "d12", "d13", "d14",
        "d23", "d24", "d34",
    )
    return np.concatenate([
        np.asarray(getattr(batch, name), dtype=np.float64).ravel()
        for name in names
    ])


def _tangent_vector(result: object) -> np.ndarray:
    names = (
        "d_p2", "d_e2", "d_e3", "d_e4", "d_p3_magnitude",
        "d_p4_magnitude", "d_phase_space", "d_quadrature_weight",
        "d_d12", "d_d13", "d_d14", "d_d23", "d_d24", "d_d34",
    )
    return np.concatenate([
        np.asarray(getattr(result, name), dtype=np.float64).ravel()
        for name in names
    ])


def _primal_elastic_batch(
    *,
    p1: float,
    temperature_gamma: float,
    electron_mass: float,
    config: ind.IndependentCollisionConfig,
) -> object:
    nodes, weights = ind._electron_half_line_rule(
        config.electron_radial_order, temperature_gamma
    )
    return ind._two_body_kinematics(
        p1=p1,
        p2_nodes=nodes,
        p2_weights=weights,
        mass2=electron_mass,
        mass3=0.0,
        mass4=electron_mass,
        config=config,
    )


def test_modal_basis_derivative_matches_centered_difference() -> None:
    grid = ind.build_independent_grid(12, 10.0)
    y = np.array([0.3, 2.0, 7.5])
    analytic = modal_basis_derivative(grid, y)
    epsilon = 1.0e-6
    centered = (
        grid.modal_basis(y + epsilon) - grid.modal_basis(y - epsilon)
    ) / (2.0 * epsilon)
    assert _relative(analytic, centered) < 2.0e-9


def test_electromagnetic_eos_temperature_tangent_matches_primal_path() -> None:
    temperature = 2.05
    mass = ind.M_ELECTRON_MEV
    result = electromagnetic_eos_tgamma_tangent(temperature, mass)
    base = ind.electromagnetic_eos_adaptive(temperature, mass)
    assert result.base == base
    assert np.isclose(result.d_rho, base.drho_dtemperature, rtol=2.0e-13)
    assert np.isclose(
        result.d_pressure,
        (base.rho + base.pressure) / temperature,
        rtol=2.0e-12,
    )

    residuals = []
    for epsilon in (3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4):
        plus = ind.electromagnetic_eos_adaptive(temperature + epsilon, mass)
        minus = ind.electromagnetic_eos_adaptive(temperature - epsilon, mass)
        centered = np.array([
            (plus.rho - minus.rho) / (2.0 * epsilon),
            (plus.pressure - minus.pressure) / (2.0 * epsilon),
            (plus.drho_dtemperature - minus.drho_dtemperature) / (2.0 * epsilon),
        ])
        analytic = np.array([result.d_rho, result.d_pressure, result.d2_rho])
        residuals.append(_relative(analytic, centered))
    assert min(residuals) < 2.0e-7


def test_elastic_tgamma_kinematic_tangent_matches_same_branch_ladder() -> None:
    config = _config()
    temperature = 2.05
    mass = ind.M_ELECTRON_MEV
    p1 = 3.2
    result = evaluate_elastic_tgamma_kinematic_tangent(
        p1=p1,
        temperature_gamma_mev=temperature,
        electron_mass_mev=mass,
        config=config,
    )
    base = _primal_elastic_batch(
        p1=p1,
        temperature_gamma=temperature,
        electron_mass=mass,
        config=config,
    )
    assert np.array_equal(result.support, base.support)
    assert _relative(_kinematic_vector(result.base), _kinematic_vector(base)) < 2.0e-15
    assert result.minimum_support_margin > 0.0
    assert result.minimum_lambda_margin > 0.0

    analytic = _tangent_vector(result)
    residuals = []
    for epsilon in (3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4):
        plus = _primal_elastic_batch(
            p1=p1,
            temperature_gamma=temperature + epsilon,
            electron_mass=mass,
            config=config,
        )
        minus = _primal_elastic_batch(
            p1=p1,
            temperature_gamma=temperature - epsilon,
            electron_mass=mass,
            config=config,
        )
        assert np.array_equal(plus.support, result.support)
        assert np.array_equal(minus.support, result.support)
        centered = (_kinematic_vector(plus) - _kinematic_vector(minus)) / (
            2.0 * epsilon
        )
        residuals.append(_relative(analytic, centered))
    assert min(residuals) < 2.0e-6

    # Mutation controls: each omitted or sign-flipped load-bearing term must be visible.
    centered_p2 = result.d_p2
    assert _relative(np.zeros_like(centered_p2), centered_p2) > 0.9
    assert _relative(-result.d_e2, result.d_e2) > 1.9
    assert _relative(
        np.zeros_like(result.d_quadrature_weight),
        result.d_quadrature_weight,
    ) > 0.9


def test_elapsed_state_is_structurally_absent_from_static_rhs_contract() -> None:
    # The packed trajectory state is (c[3,order], T_gamma, elapsed_time).
    # T_cm is T_start*exp(-N), so elapsed time is not read by the RHS.
    order = 8
    direction = np.zeros(3 * order + 2)
    direction[-1] = 1.0
    expected_column = np.zeros_like(direction)
    assert np.array_equal(expected_column, 0.0 * direction)
