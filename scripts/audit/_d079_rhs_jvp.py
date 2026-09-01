"""C-only push-forward of the frozen independent trajectory RHS.

The input direction changes only the three native cloglog spectra. T_cm,
T_gamma, elapsed time, and N are fixed. The induced Hubble, photon-temperature
rate, and elapsed-time rate tangents are included. This is therefore a
rectangular spectral-column package, not a full-state Jacobian.

Units follow the comparator's natural-unit MeV convention: c and f are
dimensionless; H and collision actions have MeV (= inverse-time) scale;
dc/dN is dimensionless, dT_gamma/dN has MeV, and dt/dN has MeV^-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_collision_jvp import (
    CollisionJvpResult,
    evaluate_collision_action_jvp,
)
from scripts.audit._d079_tangent_primitives import (
    D079LinearizationError,
    cloglog_chart_tangent,
    matrix,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RhsJvpResult:
    base_rhs: FloatArray
    jvp: FloatArray
    full_direction: FloatArray
    delta_rho_neutrino: float
    delta_hubble_over_hubble: float
    collision: CollisionJvpResult


def _base_rhs(
    *,
    grid: ind.IndependentNoQkeGrid,
    pair_cloglog: FloatArray,
    temperature_cm: float,
    temperature_gamma: float,
    config: ind.IndependentCollisionConfig,
    electron_mass: float,
) -> FloatArray:
    action = ind.evaluate_independent_collision_action(
        grid=grid, pair_cloglog=pair_cloglog,
        temperature_cm_mev=temperature_cm,
        temperature_gamma_mev=temperature_gamma,
        config=config, electron_mass_mev=electron_mass,
    )
    thermo = ind.independent_thermodynamics(
        grid=grid, pair_cloglog=pair_cloglog,
        temperature_cm_mev=temperature_cm,
        temperature_gamma_mev=temperature_gamma,
        electron_mass_mev=electron_mass,
    )
    total = np.asarray(action.total, dtype=np.float64)
    pair_rate = 0.5 * np.stack((
        total[0] + total[1], total[2] + total[3], total[4] + total[5]
    ))
    chain = ind.cloglog_chain_factor(pair_cloglog)
    c_rate = pair_rate / (thermo.hubble_mev * chain)
    eos = ind.electromagnetic_eos_adaptive(temperature_gamma, electron_mass)
    temperature_rate = (
        -3.0 * (eos.rho + eos.pressure)
        + action.electron_bath_energy_transfer / thermo.hubble_mev
    ) / eos.drho_dtemperature
    result = np.concatenate((
        c_rate.ravel(), [temperature_rate, 1.0 / thermo.hubble_mev]
    ))
    if not np.all(np.isfinite(result)):
        raise D079LinearizationError("nonfinite base static RHS")
    return np.asarray(result, dtype=np.float64)


def evaluate_c_only_rhs_jvp(
    *,
    grid: ind.IndependentNoQkeGrid,
    pair_cloglog: ArrayLike,
    direction_cloglog: ArrayLike,
    temperature_cm_mev: float,
    temperature_gamma_mev: float,
    config: ind.IndependentCollisionConfig = ind.IndependentCollisionConfig(),
    electron_mass_mev: float = ind.M_ELECTRON_MEV,
) -> RhsJvpResult:
    """Return the static full RHS and its exact spectral c-direction JVP."""

    c = matrix("pair_cloglog", pair_cloglog, (3, grid.order))
    v = matrix("direction_cloglog", direction_cloglog, c.shape)
    tcm, tg, mass = map(float, (
        temperature_cm_mev, temperature_gamma_mev, electron_mass_mev
    ))
    collision = evaluate_collision_action_jvp(
        grid=grid, pair_cloglog=c, direction_cloglog=v,
        temperature_cm_mev=tcm, temperature_gamma_mev=tg,
        config=config, electron_mass_mev=mass,
    )
    thermo = ind.independent_thermodynamics(
        grid=grid, pair_cloglog=c,
        temperature_cm_mev=tcm, temperature_gamma_mev=tg,
        electron_mass_mev=mass,
    )
    chain = ind.cloglog_chain_factor(c)
    df, _du, dlog_chain = cloglog_chart_tangent(c, v)
    energy_weights = grid.weights * np.power(grid.nodes, 3)
    delta_rho_neutrino = (
        2.0 * tcm**4
        * np.sum(df * energy_weights[None, :], dtype=np.float64)
        / ind.TWO_PI_SQUARED
    )
    delta_h_over_h = 0.5 * delta_rho_neutrino / thermo.energy_density_total

    base_total = np.asarray(collision.base.total, dtype=np.float64)
    base_pair = 0.5 * np.stack((
        base_total[0] + base_total[1],
        base_total[2] + base_total[3],
        base_total[4] + base_total[5],
    ))
    tangent_total = np.asarray(collision.total, dtype=np.float64)
    tangent_pair = 0.5 * np.stack((
        tangent_total[0] + tangent_total[1],
        tangent_total[2] + tangent_total[3],
        tangent_total[4] + tangent_total[5],
    ))
    hubble = thermo.hubble_mev
    base_c_rate = base_pair / (hubble * chain)
    tangent_c_rate = tangent_pair / (hubble * chain) - base_c_rate * (
        delta_h_over_h + dlog_chain
    )

    eos = ind.electromagnetic_eos_adaptive(tg, mass)
    base_qem = float(collision.base.electron_bath_energy_transfer)
    base_tg_rate = (
        -3.0 * (eos.rho + eos.pressure) + base_qem / hubble
    ) / eos.drho_dtemperature
    tangent_tg_rate = (
        collision.electron_bath_energy_transfer / hubble
        - (base_qem / hubble) * delta_h_over_h
    ) / eos.drho_dtemperature
    base_time_rate = 1.0 / hubble
    tangent_time_rate = -base_time_rate * delta_h_over_h

    base_rhs = np.concatenate((
        base_c_rate.ravel(), [base_tg_rate, base_time_rate]
    ))
    jvp = np.concatenate((
        tangent_c_rate.ravel(), [tangent_tg_rate, tangent_time_rate]
    ))
    full_direction = np.concatenate((v.ravel(), [0.0, 0.0]))
    if not all(np.all(np.isfinite(x)) for x in (base_rhs, jvp, full_direction)):
        raise D079LinearizationError("nonfinite static RHS/JVP")
    return RhsJvpResult(
        base_rhs=np.asarray(base_rhs, dtype=np.float64),
        jvp=np.asarray(jvp, dtype=np.float64),
        full_direction=np.asarray(full_direction, dtype=np.float64),
        delta_rho_neutrino=float(delta_rho_neutrino),
        delta_hubble_over_hubble=float(delta_h_over_h),
        collision=collision,
    )


def evaluate_static_rhs_from_packed_state(
    *,
    grid: ind.IndependentNoQkeGrid,
    packed_state: ArrayLike,
    temperature_cm_mev: float,
    config: ind.IndependentCollisionConfig = ind.IndependentCollisionConfig(),
    electron_mass_mev: float = ind.M_ELECTRON_MEV,
) -> FloatArray:
    """Evaluate the original static RHS for packed ``(c,T_gamma,t)`` bytes."""

    state = np.asarray(packed_state, dtype=np.float64)
    size = 3 * grid.order + 2
    if state.shape != (size,) or not np.all(np.isfinite(state)):
        raise ValueError(f"packed state must be a finite vector of length {size}")
    c = state[: 3 * grid.order].reshape(3, grid.order)
    tg = float(state[-2])
    if tg <= 0.0:
        raise ValueError("packed photon temperature must be positive")
    return _base_rhs(
        grid=grid, pair_cloglog=c,
        temperature_cm=float(temperature_cm_mev), temperature_gamma=tg,
        config=config, electron_mass=float(electron_mass_mev),
    )


def c_only_state_validator(
    grid: ind.IndependentNoQkeGrid,
    packed_state: ArrayLike,
) -> bool:
    """Strict state-domain validator for centered-difference certificates."""

    try:
        state = np.asarray(packed_state, dtype=np.float64)
        if (
            state.shape != (3 * grid.order + 2,)
            or not np.all(np.isfinite(state))
            or state[-2] <= 0.0
        ):
            return False
        occupation = ind.cloglog_to_occupation(
            state[: 3 * grid.order].reshape(3, grid.order)
        )
    except Exception:
        return False
    return bool(np.all(occupation > 0.0) and np.all(occupation < 1.0))
