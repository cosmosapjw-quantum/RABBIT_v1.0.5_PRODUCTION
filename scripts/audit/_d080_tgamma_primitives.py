"""D-080A temperature-column primitives for the private comparator.

This module differentiates only smooth, fixed-branch pieces needed by the
``T_gamma`` input column:

* the temperature-scaled incoming-electron half-line rule;
* the comparator's relativistic elastic two-body kinematics;
* mapped-Legendre interpolation locations; and
* the QED-off electromagnetic equation of state.

No collision assembly, trajectory, solver Jacobian, or gate movement is
implied.  A support change is a non-differentiable discrete event and must be
classified separately rather than hidden inside a finite-difference tolerance.

Units are the comparator's natural-unit MeV convention.  Momenta, energies,
and temperatures have dimension MeV; derivatives with respect to ``T_gamma``
therefore remove one power of MeV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import legder, legval
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.special import expit

from rabbit.decoupling import _independent_noqke as ind

FloatArray = NDArray[np.float64]
EXPECTED_COMPARATOR_BLOB_SHA = "de44feee0aa484abe26976c7dc34c579643005b5"


class D080TgammaLinearizationError(RuntimeError):
    """Fail-closed error for an inadmissible temperature tangent."""


@dataclass(frozen=True)
class ElectromagneticEosTgammaTangent:
    """QED-off electromagnetic EOS and its first temperature tangent."""

    base: ind.IndependentElectromagneticEos
    d_rho: float
    d_pressure: float
    d2_rho: float


@dataclass(frozen=True)
class TgammaKinematicTangent:
    """Elastic ``nu+e -> nu+e`` kinematics and its ``T_gamma`` tangent."""

    base: ind._KinematicBatch
    support: NDArray[np.bool_]
    d_p2: FloatArray
    d_e2: FloatArray
    d_e3: FloatArray
    d_e4: FloatArray
    d_p3_magnitude: FloatArray
    d_p4_magnitude: FloatArray
    d_phase_space: FloatArray
    d_quadrature_weight: FloatArray
    d_d12: FloatArray
    d_d13: FloatArray
    d_d14: FloatArray
    d_d23: FloatArray
    d_d24: FloatArray
    d_d34: FloatArray
    minimum_support_margin: float
    minimum_lambda_margin: float


def _finite(name: str, value: ArrayLike) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise D080TgammaLinearizationError(f"{name} contains NaN/Inf")
    return result


def modal_basis_derivative(
    grid: ind.IndependentNoQkeGrid,
    y: ArrayLike,
) -> FloatArray:
    """Return ``d/dy`` of the comparator's mapped orthonormal basis."""

    query = np.asarray(y, dtype=np.float64)
    if (
        not np.all(np.isfinite(query))
        or np.any(query < 0.0)
        or np.any(query > grid.y_max)
    ):
        raise D080TgammaLinearizationError("basis derivative query outside [0,y_max]")
    x = 2.0 * query / grid.y_max - 1.0
    result = np.zeros(query.shape + (grid.order,), dtype=np.float64)
    scales = np.sqrt((2.0 * np.arange(grid.order) + 1.0) / grid.y_max)
    for degree in range(1, grid.order):
        coefficients = np.zeros(degree + 1, dtype=np.float64)
        coefficients[-1] = 1.0
        result[..., degree] = (
            2.0
            * scales[degree]
            * legval(x, legder(coefficients))
            / grid.y_max
        )
    return _finite("mapped basis derivative", result)


def electron_half_line_tgamma_tangent(
    order: int,
    temperature_gamma_mev: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return ``(p,w,dp/dT,dw/dT)`` for the exact half-line rule.

    The comparator maps fixed Gauss--Legendre nodes by
    ``p=T*r/(1-r)`` and ``dp_weight=T*w_r/(1-r)^2``.  Both therefore
    scale linearly with ``T_gamma``.
    """

    temperature = float(temperature_gamma_mev)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("T_gamma must be finite and positive")
    momentum, weights = ind._electron_half_line_rule(int(order), temperature)
    return (
        _finite("electron momentum", momentum),
        _finite("electron quadrature weights", weights),
        _finite("electron momentum tangent", momentum / temperature),
        _finite("electron weight tangent", weights / temperature),
    )


def electromagnetic_eos_tgamma_tangent(
    temperature_gamma_mev: float,
    electron_mass_mev: float = ind.M_ELECTRON_MEV,
) -> ElectromagneticEosTgammaTangent:
    """Differentiate the comparator's QED-off photon plus e-/e+ EOS.

    ``d_rho`` is the primal heat-capacity path.  At zero chemical potential,
    ``dP/dT=(rho+P)/T``.  ``d2_rho`` is obtained by differentiating the
    comparator's dimensionless integral for ``d_rho/dT`` at fixed integration
    coordinate.
    """

    temperature = float(temperature_gamma_mev)
    mass = float(electron_mass_mev)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("T_gamma must be finite and positive")
    if not np.isfinite(mass) or mass < 0.0:
        raise ValueError("electron mass must be finite and nonnegative")

    base = ind.electromagnetic_eos_adaptive(temperature, mass)
    ratio = mass / temperature

    def second_derivative_integrand(x: float) -> float:
        energy = float(np.hypot(x, ratio))
        occupation = float(expit(-energy))
        blocking = occupation * (1.0 - occupation)
        return (
            x * x
            * blocking
            * (
                3.0 * energy * energy
                + ratio * ratio * (-2.0 + energy * (1.0 - 2.0 * occupation))
            )
        )

    options = {"epsabs": 2.0e-12, "epsrel": 2.0e-11, "limit": 300}
    electron_integral = quad(
        second_derivative_integrand, 0.0, np.inf, **options
    )[0]
    d2_photon = 4.0 * ind.PI**2 * temperature**2 / 5.0
    d2_electron = 2.0 * temperature**2 * electron_integral / ind.PI**2
    result = ElectromagneticEosTgammaTangent(
        base=base,
        d_rho=float(base.drho_dtemperature),
        d_pressure=float((base.rho + base.pressure) / temperature),
        d2_rho=float(d2_photon + d2_electron),
    )
    if not all(np.isfinite(value) for value in (
        result.d_rho, result.d_pressure, result.d2_rho
    )):
        raise D080TgammaLinearizationError("nonfinite electromagnetic EOS tangent")
    return result


def _norm_tangent(vector: FloatArray, tangent: FloatArray) -> FloatArray:
    norm = np.linalg.norm(vector, axis=-1)
    numerator = np.sum(vector * tangent, axis=-1)
    result = np.zeros_like(norm)
    nonzero = norm > 64.0 * np.finfo(np.float64).tiny
    result[nonzero] = numerator[nonzero] / norm[nonzero]
    return _finite("norm tangent", result)


def _dot_tangent(
    energy_a: ArrayLike,
    d_energy_a: ArrayLike,
    vector_a: FloatArray,
    d_vector_a: FloatArray,
    energy_b: ArrayLike,
    d_energy_b: ArrayLike,
    vector_b: FloatArray,
    d_vector_b: FloatArray,
) -> FloatArray:
    return _finite(
        "Minkowski-dot tangent",
        np.asarray(d_energy_a) * np.asarray(energy_b)
        + np.asarray(energy_a) * np.asarray(d_energy_b)
        - np.sum(d_vector_a * vector_b + vector_a * d_vector_b, axis=-1),
    )


def evaluate_elastic_tgamma_kinematic_tangent(
    *,
    p1: float,
    temperature_gamma_mev: float,
    electron_mass_mev: float = ind.M_ELECTRON_MEV,
    config: ind.IndependentCollisionConfig = ind.IndependentCollisionConfig(),
) -> TgammaKinematicTangent:
    """Differentiate the exact elastic electron kinematic batch.

    The independent target-neutrino momentum ``p1`` is fixed.  Incoming
    electron nodes and weights move with ``T_gamma``.  The returned derivative
    follows the comparator's exact smooth branch, including boosted final
    momenta, phase space, quadrature weights, and every matrix-element dot
    product.  The discrete support mask itself is not differentiated.
    """

    target_momentum = float(p1)
    temperature = float(temperature_gamma_mev)
    mass = float(electron_mass_mev)
    if not np.isfinite(target_momentum) or target_momentum <= 0.0:
        raise ValueError("p1 must be finite and positive")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("T_gamma must be finite and positive")
    if not np.isfinite(mass) or mass < 0.0:
        raise ValueError("electron mass must be finite and nonnegative")

    p2_nodes, p2_weights, dp2_nodes, dp2_weights = (
        electron_half_line_tgamma_tangent(
            config.electron_radial_order, temperature
        )
    )
    base_batch = ind._two_body_kinematics(
        p1=target_momentum,
        p2_nodes=p2_nodes,
        p2_weights=p2_weights,
        mass2=mass,
        mass3=0.0,
        mass4=mass,
        config=config,
    )
    (
        incoming_mu,
        incoming_weights,
        final_mu,
        final_weights,
        azimuth,
        azimuth_weights,
    ) = ind._angular_rule(
        config.incoming_polar_order,
        config.final_polar_order,
        config.final_azimuth_order,
    )
    p2, mu12, mu_star, phi = np.meshgrid(
        p2_nodes, incoming_mu, final_mu, azimuth, indexing="ij"
    )
    w2, w12, wstar, wphi = np.meshgrid(
        p2_weights,
        incoming_weights,
        final_weights,
        azimuth_weights,
        indexing="ij",
    )
    dp2 = np.broadcast_to(dp2_nodes[:, None, None, None], p2.shape)
    dw2 = np.broadcast_to(dp2_weights[:, None, None, None], p2.shape)

    e1 = target_momentum
    e2 = np.hypot(p2, mass)
    de2 = p2 * dp2 / e2
    sin12 = np.sqrt(np.maximum(0.0, 1.0 - np.square(mu12)))

    p1_vector = np.zeros(p2.shape + (3,), dtype=np.float64)
    p1_vector[..., 2] = target_momentum
    dp1_vector = np.zeros_like(p1_vector)
    p2_vector = np.zeros_like(p1_vector)
    p2_vector[..., 0] = p2 * sin12
    p2_vector[..., 2] = p2 * mu12
    dp2_vector = np.zeros_like(p1_vector)
    dp2_vector[..., 0] = dp2 * sin12
    dp2_vector[..., 2] = dp2 * mu12

    total_energy = e1 + e2
    d_total_energy = de2
    total_vector = p1_vector + p2_vector
    d_total_vector = dp2_vector
    total_magnitude = np.linalg.norm(total_vector, axis=-1)
    d_total_magnitude = _norm_tangent(total_vector, d_total_vector)

    s_raw = np.square(total_energy) - np.square(total_magnitude)
    d_s_raw = (
        2.0 * total_energy * d_total_energy
        - 2.0 * total_magnitude * d_total_magnitude
    )
    s = np.maximum(0.0, s_raw)
    d_s = np.where(s_raw > 0.0, d_s_raw, 0.0)
    threshold_squared = mass * mass
    support = (s > 0.0) & (s > threshold_squared)
    if not np.any(support):
        raise D080TgammaLinearizationError("elastic batch has no physical support")

    sqrt_s = np.sqrt(np.where(support, s, 1.0))
    d_sqrt_s = np.where(support, d_s / (2.0 * sqrt_s), 0.0)
    lambda_value = ind._kallen(s, 0.0, mass * mass)
    d_lambda = 2.0 * (s - mass * mass) * d_s
    lambda_nonnegative = np.maximum(lambda_value, 0.0)
    d_lambda_nonnegative = np.where(lambda_value > 0.0, d_lambda, 0.0)
    sqrt_lambda = np.sqrt(lambda_nonnegative)
    d_sqrt_lambda = np.zeros_like(sqrt_lambda)
    positive_lambda = lambda_nonnegative > 0.0
    d_sqrt_lambda[positive_lambda] = (
        d_lambda_nonnegative[positive_lambda]
        / (2.0 * sqrt_lambda[positive_lambda])
    )

    k_star = np.where(support, sqrt_lambda / (2.0 * sqrt_s), 0.0)
    d_k_star = np.where(
        support,
        d_sqrt_lambda / (2.0 * sqrt_s)
        - sqrt_lambda * d_sqrt_s / (2.0 * np.square(sqrt_s)),
        0.0,
    )
    numerator_e3 = s - mass * mass
    e3_star = np.where(support, numerator_e3 / (2.0 * sqrt_s), 0.0)
    d_e3_star = np.where(
        support,
        d_s / (2.0 * sqrt_s)
        - numerator_e3 * d_sqrt_s / (2.0 * np.square(sqrt_s)),
        0.0,
    )
    beta = np.where(support, total_magnitude / total_energy, 0.0)
    d_beta = np.where(
        support,
        d_total_magnitude / total_energy
        - total_magnitude * d_total_energy / np.square(total_energy),
        0.0,
    )
    gamma = np.where(support, total_energy / sqrt_s, 1.0)
    d_gamma = np.where(
        support,
        d_total_energy / sqrt_s
        - total_energy * d_sqrt_s / np.square(sqrt_s),
        0.0,
    )

    parallel = np.zeros_like(total_vector)
    d_parallel = np.zeros_like(total_vector)
    nonzero_total = total_magnitude > 64.0 * np.finfo(np.float64).tiny
    parallel[nonzero_total] = (
        total_vector[nonzero_total] / total_magnitude[nonzero_total, None]
    )
    d_parallel[nonzero_total] = (
        d_total_vector[nonzero_total] / total_magnitude[nonzero_total, None]
        - total_vector[nonzero_total]
        * d_total_magnitude[nonzero_total, None]
        / np.square(total_magnitude[nonzero_total, None])
    )
    parallel[..., 2] = np.where(nonzero_total, parallel[..., 2], 1.0)
    d_parallel[..., 2] = np.where(nonzero_total, d_parallel[..., 2], 0.0)

    transverse_x = np.zeros_like(parallel)
    transverse_x[..., 0] = parallel[..., 2]
    transverse_x[..., 2] = -parallel[..., 0]
    d_transverse_x = np.zeros_like(parallel)
    d_transverse_x[..., 0] = d_parallel[..., 2]
    d_transverse_x[..., 2] = -d_parallel[..., 0]
    transverse_y = np.zeros_like(parallel)
    transverse_y[..., 1] = 1.0

    sin_star = np.sqrt(np.maximum(0.0, 1.0 - np.square(mu_star)))
    coefficient_x = k_star * sin_star * np.cos(phi)
    coefficient_y = k_star * sin_star * np.sin(phi)
    d_coefficient_x = d_k_star * sin_star * np.cos(phi)
    d_coefficient_y = d_k_star * sin_star * np.sin(phi)
    transverse = (
        coefficient_x[..., None] * transverse_x
        + coefficient_y[..., None] * transverse_y
    )
    d_transverse = (
        d_coefficient_x[..., None] * transverse_x
        + coefficient_x[..., None] * d_transverse_x
        + d_coefficient_y[..., None] * transverse_y
    )

    bracket = k_star * mu_star + beta * e3_star
    d_bracket = d_k_star * mu_star + d_beta * e3_star + beta * d_e3_star
    p3_parallel = gamma * bracket
    d_p3_parallel = d_gamma * bracket + gamma * d_bracket
    p3_vector = transverse + p3_parallel[..., None] * parallel
    d_p3_vector = (
        d_transverse
        + d_p3_parallel[..., None] * parallel
        + p3_parallel[..., None] * d_parallel
    )
    e3 = gamma * (e3_star + beta * k_star * mu_star)
    d_e3 = (
        d_gamma * (e3_star + beta * k_star * mu_star)
        + gamma
        * (
            d_e3_star
            + d_beta * k_star * mu_star
            + beta * d_k_star * mu_star
        )
    )
    p4_vector = total_vector - p3_vector
    d_p4_vector = d_total_vector - d_p3_vector
    e4 = total_energy - e3
    d_e4 = d_total_energy - d_e3
    p3_magnitude = np.linalg.norm(p3_vector, axis=-1)
    p4_magnitude = np.linalg.norm(p4_vector, axis=-1)
    d_p3_magnitude = _norm_tangent(p3_vector, d_p3_vector)
    d_p4_magnitude = _norm_tangent(p4_vector, d_p4_vector)

    quadrature_weight = w2 * w12 * wstar * wphi
    d_quadrature_weight = dw2 * w12 * wstar * wphi
    phase_space = np.where(support, k_star / sqrt_s, 0.0)
    d_phase_space = np.where(
        support,
        d_k_star / sqrt_s - k_star * d_sqrt_s / np.square(sqrt_s),
        0.0,
    )

    d_d12 = _dot_tangent(
        e1, 0.0, p1_vector, dp1_vector,
        e2, de2, p2_vector, dp2_vector,
    )
    d_d13 = _dot_tangent(
        e1, 0.0, p1_vector, dp1_vector,
        e3, d_e3, p3_vector, d_p3_vector,
    )
    d_d14 = _dot_tangent(
        e1, 0.0, p1_vector, dp1_vector,
        e4, d_e4, p4_vector, d_p4_vector,
    )
    d_d23 = _dot_tangent(
        e2, de2, p2_vector, dp2_vector,
        e3, d_e3, p3_vector, d_p3_vector,
    )
    d_d24 = _dot_tangent(
        e2, de2, p2_vector, dp2_vector,
        e4, d_e4, p4_vector, d_p4_vector,
    )
    d_d34 = _dot_tangent(
        e3, d_e3, p3_vector, d_p3_vector,
        e4, d_e4, p4_vector, d_p4_vector,
    )

    support_scale = max(
        float(np.max(np.abs(s), initial=0.0)), threshold_squared,
        np.finfo(np.float64).tiny,
    )
    minimum_support_margin = float(
        np.min(np.abs(s - threshold_squared)) / support_scale
    )
    lambda_scale = np.maximum(np.square(s[support]), np.finfo(np.float64).tiny)
    minimum_lambda_margin = float(np.min(lambda_value[support] / lambda_scale))
    if minimum_support_margin <= 0.0 or minimum_lambda_margin <= 0.0:
        raise D080TgammaLinearizationError("kinematic state lies on a support boundary")

    result = TgammaKinematicTangent(
        base=base_batch,
        support=np.asarray(support, dtype=bool),
        d_p2=_finite("d_p2", dp2),
        d_e2=_finite("d_e2", de2),
        d_e3=_finite("d_e3", d_e3),
        d_e4=_finite("d_e4", d_e4),
        d_p3_magnitude=_finite("d_p3_magnitude", d_p3_magnitude),
        d_p4_magnitude=_finite("d_p4_magnitude", d_p4_magnitude),
        d_phase_space=_finite("d_phase_space", d_phase_space),
        d_quadrature_weight=_finite(
            "d_quadrature_weight", d_quadrature_weight
        ),
        d_d12=_finite("d_d12", d_d12),
        d_d13=_finite("d_d13", d_d13),
        d_d14=_finite("d_d14", d_d14),
        d_d23=_finite("d_d23", d_d23),
        d_d24=_finite("d_d24", d_d24),
        d_d34=_finite("d_d34", d_d34),
        minimum_support_margin=minimum_support_margin,
        minimum_lambda_margin=minimum_lambda_margin,
    )
    return result
