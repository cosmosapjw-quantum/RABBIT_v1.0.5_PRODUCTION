"""Bounded collision-core preflight for the private full Boltzmann shell.

This module does not widen the public runtime surface. It reuses the existing
host-side isotropic collision backbone to answer one narrow question:

Can the full phase-space driver's explicit transport state be compressed onto a
small species-resolved moment core that yields a finite, nontrivial Jacobian?

Current scope:
- species-resolved isotropic monopole core only
- host-side NumPy evaluation only
- reuses the existing ν-e / pair backbone via
  ``rabbit.transport.species_boltzmann_bridge.evaluate_species_monopole_collision``
- intended for bounded PR-T3B preflight and audit work
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rabbit.collisions.nu_e_scattering import nu_e_scattering_operator, nu_x_scattering_operator
from rabbit.collisions.pair_processes import nu_e_pair_operator, nu_x_pair_operator
from rabbit.collisions.species import Species, bank_temperature
from rabbit.thermo.nudec_coupled import coupled_3T_rhs_from_collision_moments
from rabbit.thermo.incomplete_decoupling import compute_energy_exchange_rate
from rabbit.transport.species_boltzmann_bridge import (
    _MEV_TO_S,
    _energy_mode_basis,
    _fermi_dirac,
    _momentum_rate_shape,
    _source_scale,
    _species_total_coupling,
    _target_distribution_from_plasma,
)
from rabbit.collisions.projected_operator import collision_rate_per_efold
from rabbit.collisions.species import BANK_DEGENERACY, bank_energy_transfer_rate


_BANK_SPECIES = (Species.NUE, Species.NUEBAR, Species.NUX)
_SUPPORTED_CLOSURE_MODES = {"source_only", "spectral_relaxation", "projected_physical", "direct_kernel"}


@dataclass(frozen=True)
class CollisionMomentPreflightResult:
    """Host-side isotropic collision-core evaluation on bank monopoles."""

    C_nue: np.ndarray
    C_nuebar: np.ndarray
    C_nux: np.ndarray
    gamma_over_H: np.ndarray
    target_plasma_energy_exchange: np.ndarray
    achieved_plasma_energy_exchange: np.ndarray
    source_scale: np.ndarray
    closure_mode: str
    closure_strength: float

    def stack_outputs(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.C_nue, dtype=np.float64),
                np.asarray(self.C_nuebar, dtype=np.float64),
                np.asarray(self.C_nux, dtype=np.float64),
                np.asarray(self.achieved_plasma_energy_exchange, dtype=np.float64),
            ]
        )


@dataclass(frozen=True)
class LiftedTransportCollisionPreflightResult:
    """Explicit transport-state collision lift plus factorized Jacobian."""

    state_rhs: np.ndarray
    transport_rhs: np.ndarray
    bank_result: CollisionMomentPreflightResult
    factors: "LiftedCollisionJacobianFactors"
    gather_map: np.ndarray
    scatter_map: np.ndarray


@dataclass(frozen=True)
class LiftedCollisionJacobianFactors:
    """Host-side factorized Jacobian representation J = U @ C @ V."""

    base_jacobian: np.ndarray
    left_factor: np.ndarray
    core_matrix: np.ndarray
    right_factor: np.ndarray


def stack_bank_monopoles(
    f_nue: np.ndarray,
    f_nuebar: np.ndarray,
    f_nux: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(f_nue, dtype=np.float64),
            np.asarray(f_nuebar, dtype=np.float64),
            np.asarray(f_nux, dtype=np.float64),
        ]
    )


def stack_bank_monopoles_with_active_scalars(
    f_nue: np.ndarray,
    f_nuebar: np.ndarray,
    f_nux: np.ndarray,
    *,
    T_gamma: float,
    T_nu_e: float,
    T_nu_x: float,
    H_inv_sec: float,
) -> np.ndarray:
    """Pack bank monopoles with the active thermodynamic scalars used by the core."""
    return np.concatenate(
        [
            stack_bank_monopoles(f_nue, f_nuebar, f_nux),
            np.asarray([T_gamma, T_nu_e, T_nu_x, H_inv_sec], dtype=np.float64),
        ]
    )


def _resolve_closure_mode(
    closure_mode: str | None,
    spectral_damping_relax: float,
) -> tuple[str, float]:
    if closure_mode is None:
        strength = float(spectral_damping_relax)
        if abs(strength) <= 0.0:
            return "source_only", 0.0
        return "spectral_relaxation", strength

    mode = str(closure_mode).strip().lower()
    if mode not in _SUPPORTED_CLOSURE_MODES:
        raise ValueError(
            f"Unsupported closure_mode={closure_mode!r}; expected one of "
            f"{sorted(_SUPPORTED_CLOSURE_MODES)}."
        )
    if mode == "source_only":
        return mode, 0.0
    if mode == "projected_physical":
        return mode, 0.0
    if mode == "direct_kernel":
        return mode, 0.0

    strength = float(spectral_damping_relax)
    if abs(strength) <= 0.0:
        strength = 1.0
    return mode, strength


def _evaluate_projected_physical_species_core(
    *,
    f_mono: np.ndarray,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    T_gamma: float,
    T_bank: float,
    H_inv_sec: float,
    coupling: float,
) -> tuple[np.ndarray, float, float]:
    """State-dependent projected physical source + damping on one bank.

    This mirrors the existing ``PhysicalCollisionOperator`` source/damping
    structure but acts directly on the bank distribution ``f(q)`` rather than
    on the perturbation ``Psi = (f-f_eq)/f_eq``.  The returned collision RHS is
    therefore immediately usable on the explicit-shell transport state.
    """
    q = np.asarray(q_nodes, dtype=np.float64)
    w = np.asarray(q_weights, dtype=np.float64)
    f_eq = _fermi_dirac(q)
    f_clip = np.clip(np.asarray(f_mono, dtype=np.float64), 0.0, 1.0)
    if float(H_inv_sec) <= 0.0 or float(T_gamma) <= 0.0:
        return np.zeros_like(q), 0.0, 1.0

    gamma_over_H = collision_rate_per_efold(
        float(T_gamma),
        float(T_bank),
        float(H_inv_sec),
        float(coupling),
    )
    psi_actual = (f_clip - f_eq) / np.maximum(f_eq, 1.0e-30)
    psi_target = ((float(T_gamma) - float(T_bank)) / max(float(T_gamma), 1.0e-30)) * q * (1.0 - f_eq)
    c_psi = gamma_over_H * (psi_target - psi_actual)
    c_mono = f_eq * c_psi

    return np.asarray(c_mono, dtype=np.float64), float(gamma_over_H), 1.0


def _evaluate_direct_kernel_core(
    *,
    f_nue: np.ndarray,
    f_nuebar: np.ndarray,
    f_nux: np.ndarray,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    T_gamma: float,
    T_nu_e: float,
    T_nu_x: float,
    H_inv_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the existing ν-e and pair kernels on explicitly remapped bank grids.

    The direct operators are written in the thermal variable ``y = p/T``.  The
    bank state here lives on the shell variable associated with the bank
    temperature.  To evaluate the kernels consistently, we map the bank grid to
    the plasma-temperature thermal variable via ``y = q * T_bank / T_gamma`` and
    interpret the tabulated bank occupancies as the same physical momenta.
    """
    q = np.asarray(q_nodes, dtype=np.float64)
    w = np.asarray(q_weights, dtype=np.float64)
    if float(H_inv_sec) <= 0.0 or float(T_gamma) <= 0.0:
        z = np.zeros_like(q, dtype=np.float64)
        z3 = np.zeros((3,), dtype=np.float64)
        return z, z, z, z3, z3, z3

    H_MeV = float(H_inv_sec) / _MEV_TO_S
    rate_to_efold = 1.0 / max(H_MeV, 1.0e-80)

    f_nue = np.clip(np.asarray(f_nue, dtype=np.float64), 0.0, 1.0)
    f_nuebar = np.clip(np.asarray(f_nuebar, dtype=np.float64), 0.0, 1.0)
    f_nux = np.clip(np.asarray(f_nux, dtype=np.float64), 0.0, 1.0)

    y_e = q * float(T_nu_e) / max(float(T_gamma), 1.0e-30)
    y_x = q * float(T_nu_x) / max(float(T_gamma), 1.0e-30)

    scat_e = nu_e_scattering_operator()
    scat_x = nu_x_scattering_operator()
    pair_e = nu_e_pair_operator()
    pair_x = nu_x_pair_operator()

    c_nue_s = np.asarray(scat_e.evaluate(f_nue, y_e, float(T_gamma)), dtype=np.float64)
    c_nuebar_s = np.asarray(scat_e.evaluate(f_nuebar, y_e, float(T_gamma)), dtype=np.float64)
    c_nux_s = np.asarray(scat_x.evaluate(f_nux, y_x, float(T_gamma)), dtype=np.float64)

    c_nue_p = np.asarray(pair_e.evaluate(f_nue, f_nuebar, y_e, float(T_gamma)), dtype=np.float64)
    c_nuebar_p = np.asarray(pair_e.evaluate(f_nuebar, f_nue, y_e, float(T_gamma)), dtype=np.float64)
    c_nux_p = np.asarray(pair_x.evaluate(f_nux, f_nux, y_x, float(T_gamma)), dtype=np.float64)

    c_nue = (c_nue_s + c_nue_p) * rate_to_efold
    c_nuebar = (c_nuebar_s + c_nuebar_p) * rate_to_efold
    c_nux = (c_nux_s + c_nux_p) * rate_to_efold

    gamma_over_H = np.array(
        [
            scat_e.total_rate_all_channels(float(T_gamma)) / max(H_MeV, 1.0e-80),
            scat_e.total_rate_all_channels(float(T_gamma)) / max(H_MeV, 1.0e-80),
            scat_x.total_rate_all_channels(float(T_gamma)) / max(H_MeV, 1.0e-80),
        ],
        dtype=np.float64,
    )
    achieved = np.array(
        [
            compute_energy_exchange_rate(c_nue, q, w, float(T_nu_e)),
            compute_energy_exchange_rate(c_nuebar, q, w, float(T_nu_e)),
            compute_energy_exchange_rate(c_nux, q, w, float(T_nu_x)),
        ],
        dtype=np.float64,
    )
    source_scale = np.ones((3,), dtype=np.float64)
    return c_nue, c_nuebar, c_nux, gamma_over_H, achieved, source_scale


def unstack_bank_monopoles(state: np.ndarray, N_q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = np.asarray(state, dtype=np.float64).reshape(-1)
    n_q = int(N_q)
    if flat.size != 3 * n_q:
        raise ValueError(f"Expected flat bank-monopole state of size {3 * n_q}, got {flat.size}.")
    return flat[:n_q], flat[n_q : 2 * n_q], flat[2 * n_q :]


def unstack_bank_monopoles_with_active_scalars(
    state: np.ndarray,
    N_q: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """Unpack `[bank_state, T_gamma, T_nu_e, T_nu_x, H_inv_sec]`."""
    flat = np.asarray(state, dtype=np.float64).reshape(-1)
    n_q = int(N_q)
    expected = 3 * n_q + 4
    if flat.size != expected:
        raise ValueError(f"Expected augmented state of size {expected}, got {flat.size}.")
    f_nue, f_nuebar, f_nux = unstack_bank_monopoles(flat[: 3 * n_q], n_q)
    return (
        f_nue,
        f_nuebar,
        f_nux,
        float(flat[3 * n_q + 0]),
        float(flat[3 * n_q + 1]),
        float(flat[3 * n_q + 2]),
        float(flat[3 * n_q + 3]),
    )


def extract_bank_monopoles_from_transport_rays(
    f_species_rays: np.ndarray,
    jacobian_weights: np.ndarray,
    w0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project explicit `(species, ray, q)` transport state onto bank monopoles.

    The current isotropic backbone uses three banks:
    - ν_e
    - ν̄_e
    - ν_x bank representing both heavy-flavor helicity-conjugate states
    """
    f = np.asarray(f_species_rays, dtype=np.float64)
    jac = np.asarray(jacobian_weights, dtype=np.float64)
    weights = 0.5 * np.asarray(w0, dtype=np.float64) * jac

    if f.ndim != 3 or f.shape[0] != 4:
        raise ValueError("Expected transport state shaped (4, N_mu, N_q).")

    f_nue = np.tensordot(weights, f[0], axes=((0,), (0,)))
    f_nuebar = np.tensordot(weights, f[1], axes=((0,), (0,)))
    f_nux = 0.5 * (
        np.tensordot(weights, f[2], axes=((0,), (0,)))
        + np.tensordot(weights, f[3], axes=((0,), (0,)))
    )
    return (
        np.asarray(f_nue, dtype=np.float64),
        np.asarray(f_nuebar, dtype=np.float64),
        np.asarray(f_nux, dtype=np.float64),
    )


def evaluate_collision_moment_core(
    *,
    f_nue: np.ndarray,
    f_nuebar: np.ndarray,
    f_nux: np.ndarray,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    T_gamma: float,
    T_nu_e: float,
    T_nu_x: float,
    H_inv_sec: float,
    closure_mode: str | None = None,
    spectral_damping_relax: float = 0.0,
) -> CollisionMomentPreflightResult:
    """Evaluate the bounded isotropic collision backbone on bank monopoles."""
    q_nodes_arr = np.asarray(q_nodes, dtype=np.float64)
    q_weights_arr = np.asarray(q_weights, dtype=np.float64)
    closure_mode_resolved, closure_strength = _resolve_closure_mode(
        closure_mode,
        spectral_damping_relax,
    )
    inputs = {
        Species.NUE: np.asarray(f_nue, dtype=np.float64),
        Species.NUEBAR: np.asarray(f_nuebar, dtype=np.float64),
        Species.NUX: np.asarray(f_nux, dtype=np.float64),
    }

    c_list = []
    gamma_list = []
    target_list = []
    achieved_list = []
    scale_list = []

    for species in _BANK_SPECIES:
        f_mono = np.clip(inputs[species], 0.0, 1.0)
        if float(H_inv_sec) <= 0.0 or float(T_gamma) <= 0.0:
            C_mono = np.zeros_like(q_nodes_arr, dtype=np.float64)
            gamma_over_H = 0.0
            target_plasma_dQ_dN = 0.0
            source_scale = 0.0
        else:
            T_bank = bank_temperature(species, float(T_nu_e), float(T_nu_x))
            if closure_mode_resolved == "projected_physical":
                C_mono, gamma_over_H, source_scale = _evaluate_projected_physical_species_core(
                    f_mono=f_mono,
                    q_nodes=q_nodes_arr,
                    q_weights=q_weights_arr,
                    T_gamma=float(T_gamma),
                    T_bank=float(T_bank),
                    H_inv_sec=float(H_inv_sec),
                    coupling=_species_total_coupling(species),
                )
                H_MeV = float(H_inv_sec) / _MEV_TO_S
                nu_gain_dt_bank = float(
                    bank_energy_transfer_rate(species, float(T_gamma), float(T_nu_e), float(T_nu_x))
                )
                nu_gain_dt_per_state = nu_gain_dt_bank / max(float(BANK_DEGENERACY[species]), 1.0)
                target_plasma_dQ_dN = -nu_gain_dt_per_state / max(H_MeV, 1.0e-80)
            else:
                f_eq = _fermi_dirac(q_nodes_arr)
                gamma_over_H = collision_rate_per_efold(
                    float(T_gamma),
                    float(T_bank),
                    float(H_inv_sec),
                    _species_total_coupling(species),
                )
                gamma_q = gamma_over_H * _momentum_rate_shape(q_nodes_arr)
                psi_actual = (f_mono - f_eq) / np.maximum(f_eq, 1.0e-30)
                f_target = _target_distribution_from_plasma(
                    q_nodes_arr,
                    T_bank=float(T_bank),
                    T_gamma=float(T_gamma),
                )
                psi_target = (f_target - f_eq) / np.maximum(f_eq, 1.0e-30)
                energy_basis = _energy_mode_basis(q_nodes_arr, gamma_q, f_eq)
                source_raw = gamma_q * psi_target
                damping_raw = -gamma_q * psi_actual

                damping_plasma = float(
                    compute_energy_exchange_rate(
                        damping_raw,
                        q_nodes_arr,
                        q_weights_arr,
                        float(T_bank),
                    )
                )
                basis_plasma = float(
                    compute_energy_exchange_rate(
                        energy_basis,
                        q_nodes_arr,
                        q_weights_arr,
                        float(T_bank),
                    )
                )
                if abs(basis_plasma) > 1.0e-80:
                    damping_raw = damping_raw - (damping_plasma / basis_plasma) * energy_basis

                H_MeV = float(H_inv_sec) / _MEV_TO_S
                nu_gain_dt_bank = float(
                    bank_energy_transfer_rate(species, float(T_gamma), float(T_nu_e), float(T_nu_x))
                )
                nu_gain_dt_per_state = nu_gain_dt_bank / max(float(BANK_DEGENERACY[species]), 1.0)
                target_plasma_dQ_dN = -nu_gain_dt_per_state / max(H_MeV, 1.0e-80)
                source_scale = _source_scale(
                    source_raw,
                    q_nodes=q_nodes_arr,
                    q_weights=q_weights_arr,
                    T_bank=float(T_bank),
                    target_plasma_energy_exchange=float(target_plasma_dQ_dN),
                )
                C_mono = source_scale * source_raw + float(closure_strength) * damping_raw

        T_bank = bank_temperature(species, float(T_nu_e), float(T_nu_x))
        achieved = compute_energy_exchange_rate(
            np.asarray(C_mono, dtype=np.float64),
            q_nodes_arr,
            q_weights_arr,
            float(T_bank),
        )
        c_list.append(np.asarray(C_mono, dtype=np.float64))
        gamma_list.append(float(gamma_over_H))
        target_list.append(float(target_plasma_dQ_dN))
        achieved_list.append(float(achieved))
        scale_list.append(float(source_scale))

    if closure_mode_resolved == "direct_kernel":
        c_nue, c_nuebar, c_nux, gamma_direct, achieved_direct, scale_direct = _evaluate_direct_kernel_core(
            f_nue=inputs[Species.NUE],
            f_nuebar=inputs[Species.NUEBAR],
            f_nux=inputs[Species.NUX],
            q_nodes=q_nodes_arr,
            q_weights=q_weights_arr,
            T_gamma=float(T_gamma),
            T_nu_e=float(T_nu_e),
            T_nu_x=float(T_nu_x),
            H_inv_sec=float(H_inv_sec),
        )
        h_mev = float(H_inv_sec) / _MEV_TO_S
        target_direct = np.asarray(
            [
                -bank_energy_transfer_rate(Species.NUE, float(T_gamma), float(T_nu_e), float(T_nu_x))
                / max(float(BANK_DEGENERACY[Species.NUE]), 1.0)
                / max(h_mev, 1.0e-80),
                -bank_energy_transfer_rate(Species.NUEBAR, float(T_gamma), float(T_nu_e), float(T_nu_x))
                / max(float(BANK_DEGENERACY[Species.NUEBAR]), 1.0)
                / max(h_mev, 1.0e-80),
                -bank_energy_transfer_rate(Species.NUX, float(T_gamma), float(T_nu_e), float(T_nu_x))
                / max(float(BANK_DEGENERACY[Species.NUX]), 1.0)
                / max(h_mev, 1.0e-80),
            ],
            dtype=np.float64,
        )
        return CollisionMomentPreflightResult(
            C_nue=np.asarray(c_nue, dtype=np.float64),
            C_nuebar=np.asarray(c_nuebar, dtype=np.float64),
            C_nux=np.asarray(c_nux, dtype=np.float64),
            gamma_over_H=np.asarray(gamma_direct, dtype=np.float64),
            target_plasma_energy_exchange=target_direct,
            achieved_plasma_energy_exchange=np.asarray(achieved_direct, dtype=np.float64),
            source_scale=np.asarray(scale_direct, dtype=np.float64),
            closure_mode=closure_mode_resolved,
            closure_strength=float(closure_strength),
        )

    return CollisionMomentPreflightResult(
        C_nue=c_list[0],
        C_nuebar=c_list[1],
        C_nux=c_list[2],
        gamma_over_H=np.asarray(gamma_list, dtype=np.float64),
        target_plasma_energy_exchange=np.asarray(target_list, dtype=np.float64),
        achieved_plasma_energy_exchange=np.asarray(achieved_list, dtype=np.float64),
        source_scale=np.asarray(scale_list, dtype=np.float64),
        closure_mode=closure_mode_resolved,
        closure_strength=float(closure_strength),
    )


def evaluate_collision_moment_core_from_state(
    state: np.ndarray,
    *,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    T_gamma: float,
    T_nu_e: float,
    T_nu_x: float,
    H_inv_sec: float,
    closure_mode: str | None = None,
    spectral_damping_relax: float = 0.0,
) -> CollisionMomentPreflightResult:
    f_nue, f_nuebar, f_nux = unstack_bank_monopoles(state, len(q_nodes))
    return evaluate_collision_moment_core(
        f_nue=f_nue,
        f_nuebar=f_nuebar,
        f_nux=f_nux,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=T_gamma,
        T_nu_e=T_nu_e,
        T_nu_x=T_nu_x,
        H_inv_sec=H_inv_sec,
        closure_mode=closure_mode,
        spectral_damping_relax=spectral_damping_relax,
    )


def evaluate_collision_moment_core_from_augmented_state(
    state: np.ndarray,
    *,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    closure_mode: str | None = None,
    spectral_damping_relax: float = 0.0,
) -> CollisionMomentPreflightResult:
    """Evaluate the bank collision core from `[bank_state, T_gamma, T_nu_e, T_nu_x, H]`."""
    f_nue, f_nuebar, f_nux, T_gamma, T_nu_e, T_nu_x, H_inv_sec = unstack_bank_monopoles_with_active_scalars(
        state,
        len(q_nodes),
    )
    return evaluate_collision_moment_core(
        f_nue=f_nue,
        f_nuebar=f_nuebar,
        f_nux=f_nux,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=T_gamma,
        T_nu_e=T_nu_e,
        T_nu_x=T_nu_x,
        H_inv_sec=H_inv_sec,
        closure_mode=closure_mode,
        spectral_damping_relax=spectral_damping_relax,
    )


def evaluate_collision_bank_and_3t_sources_from_state(
    state: np.ndarray,
    *,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    T_gamma: float,
    T_nu_e: float,
    T_nu_x: float,
    H_inv_sec: float,
    closure_mode: str,
    spectral_damping_relax: float = 0.0,
) -> np.ndarray:
    """Combined bank collision RHS plus tier-2 thermo source vector.

    Output vector:
    - stacked collision monopoles `[C_nue, C_nuebar, C_nux]`
    - thermo source rows `[dT_gamma, dT_nu_e, dT_nu_x]`
    """
    result = evaluate_collision_moment_core_from_state(
        state,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=T_gamma,
        T_nu_e=T_nu_e,
        T_nu_x=T_nu_x,
        H_inv_sec=H_inv_sec,
        closure_mode=closure_mode,
        spectral_damping_relax=spectral_damping_relax,
    )
    dQ_nue_pair_N = -(
        result.achieved_plasma_energy_exchange[0] + result.achieved_plasma_energy_exchange[1]
    )
    dQ_nux_bank_N = -4.0 * result.achieved_plasma_energy_exchange[2]
    dT_gamma, dT_nu_e, dT_nu_x = coupled_3T_rhs_from_collision_moments(
        float(T_gamma),
        float(T_nu_e),
        float(T_nu_x),
        dQ_nue_pair_N=np.asarray(dQ_nue_pair_N, dtype=np.float64),
        dQ_nux_bank_N=np.asarray(dQ_nux_bank_N, dtype=np.float64),
    )
    thermo = np.asarray([dT_gamma, dT_nu_e, dT_nu_x], dtype=np.float64)
    return np.concatenate([result.stack_outputs()[: 3 * int(len(q_nodes))], thermo])


def evaluate_collision_bank_and_3t_sources_from_augmented_state(
    state: np.ndarray,
    *,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    closure_mode: str,
    spectral_damping_relax: float = 0.0,
) -> np.ndarray:
    """Combined bank collision RHS plus 3T sources from `[bank_state, T_gamma, T_nu_e, T_nu_x, H]`."""
    f_nue, f_nuebar, f_nux, T_gamma, T_nu_e, T_nu_x, H_inv_sec = unstack_bank_monopoles_with_active_scalars(
        state,
        len(q_nodes),
    )
    return evaluate_collision_bank_and_3t_sources_from_state(
        stack_bank_monopoles(f_nue, f_nuebar, f_nux),
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=T_gamma,
        T_nu_e=T_nu_e,
        T_nu_x=T_nu_x,
        H_inv_sec=H_inv_sec,
        closure_mode=closure_mode,
        spectral_damping_relax=spectral_damping_relax,
    )


def materialize_collision_bank_and_3t_source_jacobian(
    state: np.ndarray,
    *,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    T_gamma: float,
    T_nu_e: float,
    T_nu_x: float,
    H_inv_sec: float,
    fd_eps: float = 1.0e-6,
    closure_mode: str,
    spectral_damping_relax: float = 0.0,
) -> np.ndarray:
    """Finite-difference Jacobian of `[bank_rhs, dT_gamma, dT_nu_e, dT_nu_x]`."""
    x = np.asarray(state, dtype=np.float64).reshape(-1)
    base = evaluate_collision_bank_and_3t_sources_from_state(
        x,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=T_gamma,
        T_nu_e=T_nu_e,
        T_nu_x=T_nu_x,
        H_inv_sec=H_inv_sec,
        closure_mode=closure_mode,
        spectral_damping_relax=spectral_damping_relax,
    )
    jac = np.zeros((base.size, x.size), dtype=np.float64)
    for i in range(x.size):
        step = float(fd_eps * max(1.0, abs(x[i])))
        x_hi = x.copy()
        x_lo = x.copy()
        x_hi[i] += step
        x_lo[i] -= step
        y_hi = evaluate_collision_bank_and_3t_sources_from_state(
            x_hi,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
            closure_mode=closure_mode,
            spectral_damping_relax=spectral_damping_relax,
        )
        y_lo = evaluate_collision_bank_and_3t_sources_from_state(
            x_lo,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
            closure_mode=closure_mode,
            spectral_damping_relax=spectral_damping_relax,
        )
        jac[:, i] = (y_hi - y_lo) / (2.0 * step)
    return jac


def materialize_collision_bank_and_3t_source_augmented_jacobian(
    state: np.ndarray,
    *,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    fd_eps: float = 1.0e-6,
    closure_mode: str,
    spectral_damping_relax: float = 0.0,
) -> np.ndarray:
    """Finite-difference Jacobian wrt `[bank_state, T_gamma, T_nu_e, T_nu_x, H_inv_sec]`."""
    x = np.asarray(state, dtype=np.float64).reshape(-1)
    base = evaluate_collision_bank_and_3t_sources_from_augmented_state(
        x,
        q_nodes=q_nodes,
        q_weights=q_weights,
        closure_mode=closure_mode,
        spectral_damping_relax=spectral_damping_relax,
    )
    jac = np.zeros((base.size, x.size), dtype=np.float64)
    for i in range(x.size):
        step = float(fd_eps * max(1.0, abs(x[i])))
        x_hi = x.copy()
        x_lo = x.copy()
        x_hi[i] += step
        x_lo[i] -= step
        y_hi = evaluate_collision_bank_and_3t_sources_from_augmented_state(
            x_hi,
            q_nodes=q_nodes,
            q_weights=q_weights,
            closure_mode=closure_mode,
            spectral_damping_relax=spectral_damping_relax,
        )
        y_lo = evaluate_collision_bank_and_3t_sources_from_augmented_state(
            x_lo,
            q_nodes=q_nodes,
            q_weights=q_weights,
            closure_mode=closure_mode,
            spectral_damping_relax=spectral_damping_relax,
        )
        jac[:, i] = (y_hi - y_lo) / (2.0 * step)
    return jac


def materialize_collision_moment_core_jacobian(
    state: np.ndarray,
    *,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    T_gamma: float,
    T_nu_e: float,
    T_nu_x: float,
    H_inv_sec: float,
    fd_eps: float = 1.0e-6,
    closure_mode: str | None = None,
    spectral_damping_relax: float = 0.0,
) -> np.ndarray:
    """Finite-difference Jacobian of the host collision-core map.

    Input state:
    - stacked bank monopoles `[f_nue, f_nuebar, f_nux]`

    Output vector:
    - stacked collision monopoles `[C_nue, C_nuebar, C_nux]`
    - achieved plasma energy exchange for each bank
    """
    x = np.asarray(state, dtype=np.float64).reshape(-1)
    base = evaluate_collision_moment_core_from_state(
        x,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=T_gamma,
        T_nu_e=T_nu_e,
        T_nu_x=T_nu_x,
        H_inv_sec=H_inv_sec,
        closure_mode=closure_mode,
        spectral_damping_relax=spectral_damping_relax,
    ).stack_outputs()

    jac = np.zeros((base.size, x.size), dtype=np.float64)
    for i in range(x.size):
        step = float(fd_eps * max(1.0, abs(x[i])))
        x_hi = x.copy()
        x_lo = x.copy()
        x_hi[i] += step
        x_lo[i] -= step
        y_hi = evaluate_collision_moment_core_from_state(
            x_hi,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
            closure_mode=closure_mode,
            spectral_damping_relax=spectral_damping_relax,
        ).stack_outputs()
        y_lo = evaluate_collision_moment_core_from_state(
            x_lo,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
            closure_mode=closure_mode,
            spectral_damping_relax=spectral_damping_relax,
        ).stack_outputs()
        jac[:, i] = (y_hi - y_lo) / (2.0 * step)
    return jac


def materialize_collision_moment_core_augmented_jacobian(
    state: np.ndarray,
    *,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    fd_eps: float = 1.0e-6,
    closure_mode: str | None = None,
    spectral_damping_relax: float = 0.0,
) -> np.ndarray:
    """Finite-difference Jacobian wrt `[bank_state, T_gamma, T_nu_e, T_nu_x, H_inv_sec]`."""
    x = np.asarray(state, dtype=np.float64).reshape(-1)
    base = evaluate_collision_moment_core_from_augmented_state(
        x,
        q_nodes=q_nodes,
        q_weights=q_weights,
        closure_mode=closure_mode,
        spectral_damping_relax=spectral_damping_relax,
    ).stack_outputs()

    jac = np.zeros((base.size, x.size), dtype=np.float64)
    for i in range(x.size):
        step = float(fd_eps * max(1.0, abs(x[i])))
        x_hi = x.copy()
        x_lo = x.copy()
        x_hi[i] += step
        x_lo[i] -= step
        y_hi = evaluate_collision_moment_core_from_augmented_state(
            x_hi,
            q_nodes=q_nodes,
            q_weights=q_weights,
            closure_mode=closure_mode,
            spectral_damping_relax=spectral_damping_relax,
        ).stack_outputs()
        y_lo = evaluate_collision_moment_core_from_augmented_state(
            x_lo,
            q_nodes=q_nodes,
            q_weights=q_weights,
            closure_mode=closure_mode,
            spectral_damping_relax=spectral_damping_relax,
        ).stack_outputs()
        jac[:, i] = (y_hi - y_lo) / (2.0 * step)
    return jac


def build_transport_bank_gather_map(
    *,
    jacobian_weights: np.ndarray,
    w0: np.ndarray,
    N_q: int,
    transport_offset: int = 0,
    state_dim: int | None = None,
) -> np.ndarray:
    """Build a linear gather map from explicit rays to isotropic bank monopoles."""
    jac = np.asarray(jacobian_weights, dtype=np.float64).reshape(-1)
    weights = 0.5 * np.asarray(w0, dtype=np.float64).reshape(-1) * jac
    n_mu = int(jac.size)
    n_q = int(N_q)
    transport_dim = 4 * n_mu * n_q
    offset = int(transport_offset)
    total_dim = transport_dim + offset if state_dim is None else int(state_dim)
    if total_dim < offset + transport_dim:
        raise ValueError(
            f"state_dim={total_dim} is too small for transport slice "
            f"[{offset}, {offset + transport_dim})."
        )

    gather = np.zeros((3 * n_q, total_dim), dtype=np.float64)
    for ray in range(n_mu):
        ray_weight = weights[ray]
        heavy_weight = 0.5 * ray_weight
        for q_idx in range(n_q):
            ray_offset = offset + ray * n_q + q_idx
            gather[q_idx, ray_offset] = ray_weight
            gather[n_q + q_idx, offset + n_mu * n_q + ray * n_q + q_idx] = ray_weight
            gather[2 * n_q + q_idx, offset + 2 * n_mu * n_q + ray * n_q + q_idx] = heavy_weight
            gather[2 * n_q + q_idx, offset + 3 * n_mu * n_q + ray * n_q + q_idx] = heavy_weight
    return gather


def build_transport_bank_scatter_map(
    *,
    N_mu: int,
    N_q: int,
    transport_offset: int = 0,
    state_dim: int | None = None,
) -> np.ndarray:
    """Lift bank monopole collision outputs back onto the explicit ray state."""
    n_mu = int(N_mu)
    n_q = int(N_q)
    transport_dim = 4 * n_mu * n_q
    offset = int(transport_offset)
    total_dim = transport_dim + offset if state_dim is None else int(state_dim)
    if total_dim < offset + transport_dim:
        raise ValueError(
            f"state_dim={total_dim} is too small for transport slice "
            f"[{offset}, {offset + transport_dim})."
        )

    scatter = np.zeros((total_dim, 3 * n_q), dtype=np.float64)
    for ray in range(n_mu):
        for q_idx in range(n_q):
            scatter[offset + ray * n_q + q_idx, q_idx] = 1.0
            scatter[offset + n_mu * n_q + ray * n_q + q_idx, n_q + q_idx] = 1.0
            scatter[offset + 2 * n_mu * n_q + ray * n_q + q_idx, 2 * n_q + q_idx] = 1.0
            scatter[offset + 3 * n_mu * n_q + ray * n_q + q_idx, 2 * n_q + q_idx] = 1.0
    return scatter


def build_lifted_transport_collision_preflight(
    *,
    f_species_rays: np.ndarray,
    jacobian_weights: np.ndarray,
    w0: np.ndarray,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    T_gamma: float,
    T_nu_e: float,
    T_nu_x: float,
    H_inv_sec: float,
    transport_offset: int = 0,
    state_dim: int | None = None,
    fd_eps: float = 1.0e-6,
    closure_mode: str | None = None,
    spectral_damping_relax: float = 0.0,
) -> LiftedTransportCollisionPreflightResult:
    """Lift the bank collision-core preflight onto explicit `(species, ray, q)` state."""
    f = np.asarray(f_species_rays, dtype=np.float64)
    if f.ndim != 3 or f.shape[0] != 4:
        raise ValueError("Expected explicit transport rays shaped (4, N_mu, N_q).")

    n_mu = int(f.shape[1])
    n_q = int(f.shape[2])
    bank_state = stack_bank_monopoles(
        *extract_bank_monopoles_from_transport_rays(
            f,
            jacobian_weights,
            w0,
        )
    )
    bank_result = evaluate_collision_moment_core_from_state(
        bank_state,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=T_gamma,
        T_nu_e=T_nu_e,
        T_nu_x=T_nu_x,
        H_inv_sec=H_inv_sec,
        closure_mode=closure_mode,
        spectral_damping_relax=spectral_damping_relax,
    )
    gather = build_transport_bank_gather_map(
        jacobian_weights=jacobian_weights,
        w0=w0,
        N_q=n_q,
        transport_offset=transport_offset,
        state_dim=state_dim,
    )
    scatter = build_transport_bank_scatter_map(
        N_mu=n_mu,
        N_q=n_q,
        transport_offset=transport_offset,
        state_dim=state_dim,
    )
    core_jac = materialize_collision_moment_core_jacobian(
        bank_state,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=T_gamma,
        T_nu_e=T_nu_e,
        T_nu_x=T_nu_x,
        H_inv_sec=H_inv_sec,
        fd_eps=fd_eps,
        closure_mode=bank_result.closure_mode,
        spectral_damping_relax=bank_result.closure_strength,
    )[: 3 * n_q, :]
    bank_rhs = stack_bank_monopoles(
        bank_result.C_nue,
        bank_result.C_nuebar,
        bank_result.C_nux,
    )
    state_rhs = scatter @ bank_rhs
    factors = LiftedCollisionJacobianFactors(
        base_jacobian=np.zeros((state_rhs.size, state_rhs.size), dtype=np.float64),
        left_factor=scatter,
        core_matrix=np.asarray(core_jac, dtype=np.float64),
        right_factor=gather,
    )
    return LiftedTransportCollisionPreflightResult(
        state_rhs=np.asarray(state_rhs, dtype=np.float64),
        transport_rhs=np.asarray(
            state_rhs[int(transport_offset) : int(transport_offset) + 4 * n_mu * n_q],
            dtype=np.float64,
        ).reshape(f.shape),
        bank_result=bank_result,
        factors=factors,
        gather_map=gather,
        scatter_map=scatter,
    )
