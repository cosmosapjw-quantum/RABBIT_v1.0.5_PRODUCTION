from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rabbit.collisions.kernels import A_TOTAL_NUE, A_TOTAL_NUX
from rabbit.collisions.projected_operator import collision_rate_per_efold
from rabbit.collisions.species import (
    BANK_DEGENERACY,
    Species,
    as_species,
    bank_energy_transfer_rate,
    bank_temperature,
)
from rabbit.thermo.incomplete_decoupling import compute_energy_exchange_rate
from rabbit.transport.teff_collision_bridge import (
    gather_monopole,
    gather_ray_temperatures,
    scatter_collision_to_rays,
)


_MEV_TO_S = 1.519267447e21
_Q_THERMAL_MEAN = 3.151374371
_FD_CLIP = 500.0
_SPECTRAL_DAMPING_RELAX = 0.0


@dataclass
class SpeciesBoltzmannBridgeResult:
    delta_I: np.ndarray
    delta_rho_nu: float
    C_monopole: np.ndarray
    theta_per_ray: np.ndarray
    f_monopole: np.ndarray
    gamma_over_H_thermal: float
    target_plasma_energy_exchange: float
    achieved_plasma_energy_exchange: float
    source_scale: float
    collision_closure_mode: str = "species_resolved_momentum_calibrated_v1"


def _fermi_dirac(q_nodes: np.ndarray) -> np.ndarray:
    q = np.asarray(q_nodes, dtype=np.float64)
    return 1.0 / (np.exp(np.minimum(q, _FD_CLIP)) + 1.0)


def _species_total_coupling(species: Species) -> float:
    if species in (Species.NUE, Species.NUEBAR):
        return float(A_TOTAL_NUE)
    return float(A_TOTAL_NUX)


def _target_distribution_from_plasma(
    q_nodes: np.ndarray,
    *,
    T_bank: float,
    T_gamma: float,
) -> np.ndarray:
    q = np.asarray(q_nodes, dtype=np.float64)
    if T_gamma <= 0.0 or T_bank <= 0.0:
        return _fermi_dirac(q)
    # q is the thermal momentum variable for the bank temperature T_bank.
    # A FD distribution at T_gamma therefore appears as f(q * T_bank / T_gamma).
    ratio = float(T_bank / max(T_gamma, 1.0e-30))
    return 1.0 / (np.exp(np.minimum(q * ratio, _FD_CLIP)) + 1.0)


def _momentum_rate_shape(q_nodes: np.ndarray) -> np.ndarray:
    q = np.asarray(q_nodes, dtype=np.float64)
    return np.clip(q / _Q_THERMAL_MEAN, 0.0, 8.0)


def _energy_mode_basis(q_nodes: np.ndarray, gamma_q: np.ndarray, f_eq: np.ndarray) -> np.ndarray:
    q = np.asarray(q_nodes, dtype=np.float64)
    gamma = np.asarray(gamma_q, dtype=np.float64)
    one_minus_f = np.maximum(1.0 - np.asarray(f_eq, dtype=np.float64), 1.0e-30)
    # Linearized thermal shift mode: δf/f0 ~ q (1-f0).
    return gamma * q * one_minus_f


def _source_scale(
    source_term: np.ndarray,
    *,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    T_bank: float,
    target_plasma_energy_exchange: float,
) -> float:
    raw_plasma = float(
        compute_energy_exchange_rate(
            source_term,
            np.asarray(q_nodes, dtype=np.float64),
            np.asarray(q_weights, dtype=np.float64),
            float(T_bank),
        )
    )

    if abs(raw_plasma) < 1.0e-80:
        return 1.0 if abs(target_plasma_energy_exchange) < 1.0e-80 else 0.0

    scale = float(target_plasma_energy_exchange / raw_plasma)
    if not np.isfinite(scale):
        return 0.0
    return float(np.clip(scale, 0.0, 100.0))


def evaluate_species_monopole_collision(
    *,
    species,
    f_monopole: np.ndarray,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    T_gamma: float,
    T_nu_e: float,
    T_nu_x: float,
    H_inv_sec: float,
) -> tuple[np.ndarray, float, float, float]:
    """Species-resolved, momentum-weighted collision monopole.

    This is a production-bounded upgrade over the legacy species-tagged bridge:
    - species-resolved ν_e / ν̄_e / ν_x target temperatures
    - momentum-dependent collision rate Γ(q)/H ∝ q
    - source normalization calibrated to the coded incomplete-decoupling dQ target
    - explicit damping of the non-equilibrium monopole perturbation

    Returns
    -------
    (C_monopole, gamma_over_H_thermal, target_plasma_dQ_dN, source_scale)
    """
    sp = as_species(species)
    q = np.asarray(q_nodes, dtype=np.float64)
    f_mono = np.clip(np.asarray(f_monopole, dtype=np.float64), 0.0, 1.0)
    f_eq = _fermi_dirac(q)
    w = np.asarray(q_weights, dtype=np.float64)

    if H_inv_sec <= 0.0 or T_gamma <= 0.0:
        return np.zeros_like(q), 0.0, 0.0, 0.0

    T_bank = bank_temperature(sp, T_nu_e, T_nu_x)
    gamma_over_H = collision_rate_per_efold(
        float(T_gamma),
        float(T_bank),
        float(H_inv_sec),
        _species_total_coupling(sp),
    )
    gamma_q = gamma_over_H * _momentum_rate_shape(q)

    psi_actual = (f_mono - f_eq) / np.maximum(f_eq, 1.0e-30)
    f_target = _target_distribution_from_plasma(q, T_bank=T_bank, T_gamma=T_gamma)
    psi_target = (f_target - f_eq) / np.maximum(f_eq, 1.0e-30)

    energy_basis = _energy_mode_basis(q, gamma_q, f_eq)
    source_raw = gamma_q * psi_target
    damping_raw = -gamma_q * psi_actual

    damping_plasma = float(
        compute_energy_exchange_rate(
            damping_raw,
            q,
            w,
            float(T_bank),
        )
    )
    basis_plasma = float(
        compute_energy_exchange_rate(
            energy_basis,
            q,
            w,
            float(T_bank),
        )
    )
    if abs(basis_plasma) > 1.0e-80:
        damping_raw = damping_raw - (damping_plasma / basis_plasma) * energy_basis

    H_MeV = float(H_inv_sec) / _MEV_TO_S
    nu_gain_dt_bank = float(bank_energy_transfer_rate(sp, T_gamma, T_nu_e, T_nu_x))
    nu_gain_dt_per_state = nu_gain_dt_bank / max(float(BANK_DEGENERACY[sp]), 1.0)
    target_plasma_dQ_dN = -nu_gain_dt_per_state / max(H_MeV, 1.0e-80)

    source_scale = _source_scale(
        source_raw,
        q_nodes=q,
        q_weights=w,
        T_bank=T_bank,
        target_plasma_energy_exchange=target_plasma_dQ_dN,
    )

    C_mono = source_scale * source_raw + _SPECTRAL_DAMPING_RELAX * damping_raw
    return C_mono, float(gamma_over_H), float(target_plasma_dQ_dN), float(source_scale)


def apply_species_boltzmann_collision(
    *,
    species,
    I: np.ndarray,
    J: np.ndarray,
    w0: np.ndarray,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    T_gamma: float,
    T_nu_e: float,
    T_nu_x: float,
    H: float,
    record_debug: bool = False,
) -> SpeciesBoltzmannBridgeResult:
    del record_debug
    theta = gather_ray_temperatures(I, J, w0)
    f_mono = gather_monopole(I, J, w0, q_nodes, theta=theta)

    C_mono, gamma_over_H, target_plasma_dQ_dN, source_scale = evaluate_species_monopole_collision(
        species=species,
        f_monopole=f_mono,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=T_gamma,
        T_nu_e=T_nu_e,
        T_nu_x=T_nu_x,
        H_inv_sec=H,
    )

    delta_I = scatter_collision_to_rays(
        C_mono,
        I,
        J,
        w0,
        q_nodes,
        q_weights,
        H,
    )

    sp = as_species(species)
    T_bank = bank_temperature(sp, T_nu_e, T_nu_x)
    plasma_dQ_dN = float(
        compute_energy_exchange_rate(
            C_mono,
            np.asarray(q_nodes, dtype=np.float64),
            np.asarray(q_weights, dtype=np.float64),
            float(T_bank),
        )
    )

    return SpeciesBoltzmannBridgeResult(
        delta_I=np.asarray(delta_I, dtype=np.float64),
        delta_rho_nu=float(-plasma_dQ_dN),
        C_monopole=np.asarray(C_mono, dtype=np.float64),
        theta_per_ray=np.asarray(theta, dtype=np.float64),
        f_monopole=np.asarray(f_mono, dtype=np.float64),
        gamma_over_H_thermal=float(gamma_over_H),
        target_plasma_energy_exchange=float(target_plasma_dQ_dN),
        achieved_plasma_energy_exchange=float(plasma_dQ_dN),
        source_scale=float(source_scale),
    )
