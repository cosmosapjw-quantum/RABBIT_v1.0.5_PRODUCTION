from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rabbit.collisions.kernels import A_TOTAL_NUE, A_TOTAL_NUX
from rabbit.collisions.projected_operator import collision_rate_per_efold
from rabbit.decoupling.grid import DecouplingGrid
from rabbit.decoupling.moments import (
    _M_ELECTRON_MEV,
    fermi_dirac_comoving,
    state_comoving_temperature_scale,
    state_energy_density_per_efold,
)
from rabbit.thermo.nudec_tables import energy_transfer_rate_nue, energy_transfer_rate_nux


_MEV_TO_S = 1.519267447e21
_Q_THERMAL_MEAN = 3.151374371


@dataclass(frozen=True)
class DiagonalCollisionResult:
    species: str
    df_dN: np.ndarray
    energy_gain_per_efold: float
    target_energy_gain_per_efold: float
    gamma_over_H_thermal: float
    source_scale: float


def _target_gain_rate(species: str, *, T_gamma: float, T_nu_e: float, T_nu_x: float) -> float:
    if species in ("nue", "nuebar"):
        return float(energy_transfer_rate_nue(T_gamma, T_nu_e))
    if species == "nux":
        return float(energy_transfer_rate_nux(T_gamma, T_nu_x))
    raise ValueError(f"Unknown species: {species}")


def _thermal_temperature(species: str, *, T_nu_e: float, T_nu_x: float) -> tuple[float, float]:
    if species in ("nue", "nuebar"):
        return float(T_nu_e), float(A_TOTAL_NUE)
    if species == "nux":
        return float(T_nu_x), float(A_TOTAL_NUX)
    raise ValueError(f"Unknown species: {species}")


def evaluate_diagonal_collision_rhs(
    *,
    species: str,
    f_state: np.ndarray,
    grid: DecouplingGrid,
    T_gamma: float,
    T_nu_e: float,
    T_nu_x: float,
    H_MeV: float,
    a_scale: float,
) -> DiagonalCollisionResult:
    """Species-resolved diagonal collision closure on a comoving momentum grid.

    This is a production-bounded skeleton for the future isotropic decoupling
    backbone. The operator is diagonal in species, momentum-local, and
    calibrated so that its energy moment reproduces the existing decoupling
    source rate for the corresponding species.
    """
    f = np.clip(np.asarray(f_state, dtype=np.float64), 0.0, 1.0)
    if T_gamma <= 0.0 or H_MeV <= 1.0e-100:
        return DiagonalCollisionResult(
            species=species,
            df_dN=np.zeros_like(f),
            energy_gain_per_efold=0.0,
            target_energy_gain_per_efold=0.0,
            gamma_over_H_thermal=0.0,
            source_scale=0.0,
        )

    T_bank, coupling = _thermal_temperature(species, T_nu_e=T_nu_e, T_nu_x=T_nu_x)
    H_inv_sec = float(H_MeV * _MEV_TO_S)
    gamma_over_H = float(collision_rate_per_efold(T_gamma, T_bank, H_inv_sec, coupling))

    z_gamma = float(T_gamma * a_scale / _M_ELECTRON_MEV)
    z_bank = max(float(T_bank * a_scale / _M_ELECTRON_MEV), 1.0e-30)
    q_over_T = grid.nodes / z_bank
    shape = np.clip(q_over_T / _Q_THERMAL_MEAN, 0.0, 8.0)
    f_target = fermi_dirac_comoving(grid.nodes, z_gamma)

    raw_df = gamma_over_H * shape * (f_target - f)
    raw_gain = state_energy_density_per_efold(raw_df, grid, a_scale)
    target_gain = float(_target_gain_rate(species, T_gamma=T_gamma, T_nu_e=T_nu_e, T_nu_x=T_nu_x) / H_MeV)

    if abs(raw_gain) < 1.0e-80:
        scale = 1.0 if abs(target_gain) < 1.0e-80 else 0.0
    else:
        scale = float(np.clip(target_gain / raw_gain, 0.0, 100.0))

    df_dN = scale * raw_df
    achieved_gain = state_energy_density_per_efold(df_dN, grid, a_scale)
    return DiagonalCollisionResult(
        species=species,
        df_dN=np.asarray(df_dN, dtype=np.float64),
        energy_gain_per_efold=float(achieved_gain),
        target_energy_gain_per_efold=float(target_gain),
        gamma_over_H_thermal=float(gamma_over_H),
        source_scale=float(scale),
    )
