"""Full static ``T_gamma`` input column of the frozen comparator RHS.

This module composes the previously admitted fixed-support collision-action
column with the exact thermodynamic quotient rules used by the original packed
RHS.  The differentiated state is

    Y = (c_e, c_mu, c_tau, T_gamma, elapsed_time),

while ``T_cm`` and the independent variable ``N`` are held fixed.  No
trajectory, integrator, Newton method, or square Jacobian is constructed here.

The comparator uses natural units with ``hbar = c = k_B = 1``.  Temperatures,
particle energies, and ``H`` have MeV units.  Consequently the spectral rows of
``dF/dT_gamma`` have MeV^-1 units, the photon-temperature row is dimensionless,
and the elapsed-time output row has MeV^-2 units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_rhs_jvp import evaluate_static_rhs_from_packed_state
from scripts.audit._d079_tangent_primitives import matrix, safe_relative
from scripts.audit._d080_tgamma_primitives import (
    D080TgammaLinearizationError,
    ElectromagneticEosTgammaTangent,
    electromagnetic_eos_tgamma_tangent,
)
from scripts.audit._d080b_tgamma_collision import (
    TgammaCollisionJvpResult,
    evaluate_tgamma_collision_action_jvp,
)

FloatArray = NDArray[np.float64]
EXPECTED_COMPARATOR_BLOB_SHA = "de44feee0aa484abe26976c7dc34c579643005b5"


@dataclass(frozen=True)
class TgammaRhsColumnResult:
    """Primal packed RHS, its ``T_gamma`` column, and atomic decomposition."""

    base_rhs: FloatArray
    tgamma_column: FloatArray
    elapsed_time_input_column: FloatArray
    branch_signature: str
    delta_hubble_over_hubble: float
    collision: TgammaCollisionJvpResult
    eos: ElectromagneticEosTgammaTangent
    spectral_collision_component: FloatArray
    spectral_hubble_component: FloatArray
    temperature_expansion_component: FloatArray
    temperature_collision_component: FloatArray
    temperature_hubble_component: FloatArray
    heat_capacity_component: FloatArray
    time_hubble_component: FloatArray
    collision_component: FloatArray
    hubble_component: FloatArray
    base_reconstruction_residual: float
    component_sum_residual: float


def _finite_vector(name: str, value: ArrayLike, size: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise D080TgammaLinearizationError(
            f"{name} must be a finite vector of length {size}"
        )
    return result


def _full_component(
    *,
    spectral: ArrayLike | None,
    temperature: float = 0.0,
    elapsed: float = 0.0,
    order: int,
) -> FloatArray:
    size = 3 * order + 2
    result = np.zeros(size, dtype=np.float64)
    if spectral is not None:
        spectral_array = np.asarray(spectral, dtype=np.float64)
        if spectral_array.shape != (3, order):
            raise D080TgammaLinearizationError(
                "spectral RHS component has an invalid shape"
            )
        result[: 3 * order] = spectral_array.ravel()
    result[-2] = float(temperature)
    result[-1] = float(elapsed)
    return _finite_vector("RHS component", result, size)


def evaluate_tgamma_rhs_column(
    *,
    grid: ind.IndependentNoQkeGrid,
    pair_cloglog: ArrayLike,
    temperature_cm_mev: float,
    temperature_gamma_mev: float,
    config: ind.IndependentCollisionConfig = ind.IndependentCollisionConfig(),
    electron_mass_mev: float = ind.M_ELECTRON_MEV,
) -> TgammaRhsColumnResult:
    """Differentiate the original static packed RHS with respect to ``T_gamma``.

    The derivative is valid only on the fixed support/matrix-correction branch
    reported by the D-080B collision result.  A branch change is a discrete
    event and is deliberately not hidden inside an ordinary derivative.
    """

    c = matrix("pair_cloglog", pair_cloglog, (3, grid.order))
    tcm = float(temperature_cm_mev)
    tg = float(temperature_gamma_mev)
    mass = float(electron_mass_mev)
    if not all(np.isfinite(value) for value in (tcm, tg, mass)):
        raise ValueError("thermodynamic inputs must be finite")
    if min(tcm, tg) <= 0.0 or mass < 0.0:
        raise ValueError("temperatures must be positive and mass nonnegative")

    collision = evaluate_tgamma_collision_action_jvp(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
        electron_mass_mev=mass,
    )
    thermo = ind.independent_thermodynamics(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        electron_mass_mev=mass,
    )
    eos = electromagnetic_eos_tgamma_tangent(tg, mass)
    chain = ind.cloglog_chain_factor(c)
    hubble = float(thermo.hubble_mev)
    if hubble <= 0.0 or eos.d_rho <= 0.0:
        raise D080TgammaLinearizationError("nonpositive Hubble rate or heat capacity")

    base_total = np.asarray(collision.base.total, dtype=np.float64)
    tangent_total = np.asarray(collision.total, dtype=np.float64)
    if base_total.shape != (6, grid.order) or tangent_total.shape != base_total.shape:
        raise D080TgammaLinearizationError("invalid collision-action shape")
    base_pair = 0.5 * np.stack(
        (
            base_total[0] + base_total[1],
            base_total[2] + base_total[3],
            base_total[4] + base_total[5],
        )
    )
    tangent_pair = 0.5 * np.stack(
        (
            tangent_total[0] + tangent_total[1],
            tangent_total[2] + tangent_total[3],
            tangent_total[4] + tangent_total[5],
        )
    )

    # At fixed c and T_cm, only the electromagnetic energy density changes H.
    # H^2 = (8 pi G/3) rho_total implies H_T/H = rho_T/(2 rho_total).
    hubble_log_tangent = 0.5 * eos.d_rho / thermo.energy_density_total
    if not np.isfinite(hubble_log_tangent) or hubble_log_tangent <= 0.0:
        raise D080TgammaLinearizationError("invalid Hubble logarithmic tangent")

    base_c_rate = base_pair / (hubble * chain)
    spectral_collision = tangent_pair / (hubble * chain)
    spectral_hubble = -base_c_rate * hubble_log_tangent

    base_qem = float(collision.base.electron_bath_energy_transfer)
    tangent_qem = float(collision.electron_bath_energy_transfer)
    heat_capacity = eos.d_rho
    base_temperature_numerator = (
        -3.0 * (eos.base.rho + eos.base.pressure) + base_qem / hubble
    )
    base_temperature_rate = base_temperature_numerator / heat_capacity

    # N_gamma = -3(rho_em+p_em)+Q_em/H.
    # Differentiate numerator and denominator separately to expose mutations.
    temperature_expansion = (
        -3.0 * (eos.d_rho + eos.d_pressure) / heat_capacity
    )
    temperature_collision = tangent_qem / (hubble * heat_capacity)
    temperature_hubble = (
        -(base_qem / hubble) * hubble_log_tangent / heat_capacity
    )
    temperature_heat_capacity = (
        -base_temperature_rate * eos.d2_rho / heat_capacity
    )

    base_time_rate = 1.0 / hubble
    time_hubble = -base_time_rate * hubble_log_tangent

    spectral_collision_component = _full_component(
        spectral=spectral_collision, order=grid.order
    )
    spectral_hubble_component = _full_component(
        spectral=spectral_hubble, order=grid.order
    )
    temperature_expansion_component = _full_component(
        spectral=None, temperature=temperature_expansion, order=grid.order
    )
    temperature_collision_component = _full_component(
        spectral=None, temperature=temperature_collision, order=grid.order
    )
    temperature_hubble_component = _full_component(
        spectral=None, temperature=temperature_hubble, order=grid.order
    )
    heat_capacity_component = _full_component(
        spectral=None, temperature=temperature_heat_capacity, order=grid.order
    )
    time_hubble_component = _full_component(
        spectral=None, elapsed=time_hubble, order=grid.order
    )

    atomic_components = (
        spectral_collision_component,
        spectral_hubble_component,
        temperature_expansion_component,
        temperature_collision_component,
        temperature_hubble_component,
        heat_capacity_component,
        time_hubble_component,
    )
    tgamma_column = np.sum(np.stack(atomic_components), axis=0, dtype=np.float64)
    collision_component = (
        spectral_collision_component + temperature_collision_component
    )
    hubble_component = (
        spectral_hubble_component
        + temperature_hubble_component
        + time_hubble_component
    )
    elapsed_time_input_column = np.zeros_like(tgamma_column)

    base_rhs = np.concatenate(
        (base_c_rate.ravel(), [base_temperature_rate, base_time_rate])
    )
    packed = np.concatenate((c.ravel(), [tg, 0.0]))
    original_base_rhs = evaluate_static_rhs_from_packed_state(
        grid=grid,
        packed_state=packed,
        temperature_cm_mev=tcm,
        config=config,
        electron_mass_mev=mass,
    )
    size = 3 * grid.order + 2
    base_rhs = _finite_vector("base RHS", base_rhs, size)
    tgamma_column = _finite_vector("T_gamma RHS column", tgamma_column, size)
    elapsed_time_input_column = _finite_vector(
        "elapsed-time input column", elapsed_time_input_column, size
    )
    component_sum = np.sum(np.stack(atomic_components), axis=0, dtype=np.float64)
    base_reconstruction_residual = safe_relative(base_rhs, original_base_rhs)
    component_sum_residual = safe_relative(tgamma_column, component_sum)
    if not all(
        np.isfinite(value)
        for value in (
            base_reconstruction_residual,
            component_sum_residual,
            hubble_log_tangent,
        )
    ):
        raise D080TgammaLinearizationError("nonfinite RHS-column diagnostic")

    return TgammaRhsColumnResult(
        base_rhs=base_rhs,
        tgamma_column=tgamma_column,
        elapsed_time_input_column=elapsed_time_input_column,
        branch_signature=collision.branch_signature,
        delta_hubble_over_hubble=float(hubble_log_tangent),
        collision=collision,
        eos=eos,
        spectral_collision_component=spectral_collision_component,
        spectral_hubble_component=spectral_hubble_component,
        temperature_expansion_component=temperature_expansion_component,
        temperature_collision_component=temperature_collision_component,
        temperature_hubble_component=temperature_hubble_component,
        heat_capacity_component=heat_capacity_component,
        time_hubble_component=time_hubble_component,
        collision_component=collision_component,
        hubble_component=hubble_component,
        base_reconstruction_residual=float(base_reconstruction_residual),
        component_sum_residual=float(component_sum_residual),
    )
