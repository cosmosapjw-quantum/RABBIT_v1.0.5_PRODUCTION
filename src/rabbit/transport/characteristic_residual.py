from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rabbit.collisions.projected_operator import (
    KAPPA_ELL0,
    KAPPA_ELL2,
    collision_rate_per_efold,
)
from rabbit.collisions.kernels import A_TOTAL_NUE, A_TOTAL_NUX
from rabbit.collisions.species import Species, as_species


_SPECTRAL_FACTOR_CACHE: dict[
    tuple[bytes, bytes],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float, float],
] = {}


@dataclass(frozen=True)
class ResidualRelaxationResult:
    delta_I: np.ndarray
    delta_J: np.ndarray
    gamma_over_H: float
    gamma_over_H_base: float
    gamma_over_H_ell0: float
    gamma_over_H_ell2: float
    spectral_factor_ell0: float
    spectral_factor_ell2: float
    mismatch_factor: float
    temperature_mismatch_log: float
    distortion_factor_ell0: float
    distortion_factor_ell2: float
    distortion_rms: float
    preserved_I_mean: float
    target_J_equilibrium: float
    target_energy_weighted_jacobian: float


def _species_bank_temperature(species: Species, *, T_nu_e: float, T_nu_x: float) -> tuple[float, float]:
    if species in (Species.NUE, Species.NUEBAR):
        return float(T_nu_e), float(A_TOTAL_NUE)
    if species is Species.NUX:
        return float(T_nu_x), float(A_TOTAL_NUX)
    raise ValueError(f"Unknown species: {species}")


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    norm = float(np.sum(w))
    if norm <= 1.0e-300:
        return 0.0
    return float(np.sum(w * vals) / norm)


def _spectral_reference_data(q_nodes: np.ndarray, q_weights: np.ndarray):
    q = np.asarray(q_nodes, dtype=np.float64)
    w = np.asarray(q_weights, dtype=np.float64)
    key = (q.tobytes(), w.tobytes())
    cached = _SPECTRAL_FACTOR_CACHE.get(key)
    if cached is not None:
        return cached

    q3 = w * q**3
    q4 = q3 * q
    q5 = q4 * q
    f_fd = 1.0 / (np.exp(np.minimum(q, 500.0)) + 1.0)
    block_fd = f_fd * (1.0 - f_fd)

    m3 = float(np.sum(q3 * f_fd))
    m4 = float(np.sum(q4 * f_fd))
    m5 = float(np.sum(q5 * f_fd))
    ref_ell0 = m4 / max(m3, 1.0e-300)
    ref_ell2 = m5 / max(m4, 1.0e-300)
    ref_block0 = float(np.sum(q4 * block_fd))
    ref_block2 = float(np.sum(q5 * block_fd))
    ref_dist0 = float(np.sum(q4 * f_fd**2))
    ref_dist2 = float(np.sum(q5 * f_fd**2))
    cached = (
        q3,
        q4,
        q5,
        f_fd,
        float(ref_ell0),
        float(ref_ell2),
        ref_block0,
        ref_block2,
        ref_dist0,
        ref_dist2,
    )
    _SPECTRAL_FACTOR_CACHE[key] = cached
    return cached


def spectral_rate_factors(
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    f_background: np.ndarray,
) -> tuple[float, float]:
    """Spectrum-aware rate multipliers for residual relaxation.

    The base Γ/H model already carries the thermal scaling.  We refine it using
    low-order moments of the backbone spectrum on the weak-rate q-grid:

      ell0 factor  ~ (⟨q⟩_{q^3 f} / ⟨q⟩_{FD})
      ell2 factor  ~ (⟨q⟩_{q^4 f} / ⟨q⟩_{FD, higher})

    This keeps the correction cheap, smooth, and tied to actual spectral
    hardening/softening rather than a fixed thermal template.
    """
    f = np.clip(np.asarray(f_background, dtype=np.float64), 0.0, 1.0)
    q3, q4, q5, _, ref_ell0, ref_ell2, ref_block0, ref_block2, _, _ = _spectral_reference_data(
        q_nodes, q_weights
    )
    m3 = float(np.sum(q3 * f))
    m4 = float(np.sum(q4 * f))
    m5 = float(np.sum(q5 * f))
    block = f * (1.0 - f)
    block0 = float(np.sum(q4 * block)) / max(ref_block0, 1.0e-300)
    block2 = float(np.sum(q5 * block)) / max(ref_block2, 1.0e-300)
    hard0 = (m4 / max(m3, 1.0e-300)) / max(ref_ell0, 1.0e-300)
    hard2 = (m5 / max(m4, 1.0e-300)) / max(ref_ell2, 1.0e-300)
    ell0 = hard0 * np.sqrt(max(block0, 0.0))
    ell2 = hard2 * np.sqrt(max(block2, 0.0))
    return (
        float(np.clip(ell0, 0.5, 2.0)),
        float(np.clip(ell2, 0.5, 2.5)),
    )


def temperature_mismatch_factor(
    T_gamma: float,
    T_bank: float,
) -> tuple[float, float]:
    """Smooth collision-rate boost from plasma-neutrino thermal mismatch."""
    ratio = max(float(T_gamma), 1.0e-30) / max(float(T_bank), 1.0e-30)
    mismatch_log = abs(float(np.log(ratio)))
    factor = float(np.clip(1.0 + 0.75 * mismatch_log, 1.0, 2.0))
    return factor, mismatch_log


def distortion_rate_factors(
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    f_background: np.ndarray,
) -> tuple[float, float, float]:
    """Rate multipliers from deviation away from the FD backbone shape."""
    f = np.clip(np.asarray(f_background, dtype=np.float64), 0.0, 1.0)
    _, q4, q5, f_fd, _, _, _, _, ref_dist0, ref_dist2 = _spectral_reference_data(q_nodes, q_weights)
    delta = f - f_fd
    dist0 = np.sqrt(float(np.sum(q4 * delta**2)) / max(ref_dist0, 1.0e-300))
    dist2 = np.sqrt(float(np.sum(q5 * delta**2)) / max(ref_dist2, 1.0e-300))
    distortion_rms = float(np.sqrt(0.5 * (dist0**2 + dist2**2)))
    return (
        float(np.clip(1.0 + 1.25 * dist0, 1.0, 2.0)),
        float(np.clip(1.0 + 1.75 * dist2, 1.0, 2.5)),
        distortion_rms,
    )


def apply_species_residual_relaxation(
    *,
    species,
    I,
    J,
    w0,
    T_gamma: float,
    T_nu_e: float,
    T_nu_x: float,
    H_inv_sec: float,
    relax: float = 1.0,
    q_nodes: np.ndarray | None = None,
    q_weights: np.ndarray | None = None,
    f_background: np.ndarray | None = None,
    gamma_factor_ell0: float | None = None,
    gamma_factor_ell2: float | None = None,
    mismatch_factor: float | None = None,
    distortion_factor_ell0: float | None = None,
    distortion_factor_ell2: float | None = None,
    distortion_rms: float | None = None,
    ell2_collision_mode: str = "calibrated_rta",
) -> ResidualRelaxationResult:
    """Backbone-aware anisotropic residual closure for characteristic rays.

    The isotropic momentum-grid decoupling backbone already carries the full
    monopole heating/cooling and weak-background evolution.  What remains on
    the characteristic side is the anisotropic residual.  We model that by
    relaxing only the traceless angular distortion, with a spectrum-aware
    effective rate derived from the backbone monopole shape, thermal mismatch,
    and overall FD-distortion amplitude:

        dI/dN |_coll = -κ0 (Γ/H) [I - <I>]
        dK/dN |_coll = -κ2 (Γ/H) [K - 1],   K = J exp[-8(I-<I>)]

    where `<I>` is the quadrature-weighted mean.  Subtracting the mean keeps
    the isotropic energy shift on the backbone, rather than double-counting it
    in the characteristic residual closure.
    """
    sp = as_species(species)
    I_arr = np.asarray(I, dtype=np.float64)
    J_arr = np.asarray(J, dtype=np.float64)
    w_arr = np.asarray(w0, dtype=np.float64)

    if H_inv_sec <= 1.0e-100 or T_gamma <= 1.0e-12:
        zeros = np.zeros_like(I_arr)
        return ResidualRelaxationResult(
            delta_I=zeros,
            delta_J=np.zeros_like(J_arr),
            gamma_over_H=0.0,
            gamma_over_H_base=0.0,
            gamma_over_H_ell0=0.0,
            gamma_over_H_ell2=0.0,
            spectral_factor_ell0=1.0,
            spectral_factor_ell2=1.0,
            mismatch_factor=1.0,
            temperature_mismatch_log=0.0,
            distortion_factor_ell0=1.0,
            distortion_factor_ell2=1.0,
            distortion_rms=0.0,
            preserved_I_mean=_weighted_mean(I_arr, w_arr),
            target_J_equilibrium=1.0,
            target_energy_weighted_jacobian=1.0,
        )

    T_bank, a_coupling = _species_bank_temperature(sp, T_nu_e=T_nu_e, T_nu_x=T_nu_x)
    gamma_over_H_base = float(collision_rate_per_efold(T_gamma, T_bank, H_inv_sec, a_coupling))
    if gamma_factor_ell0 is None or gamma_factor_ell2 is None:
        if q_nodes is not None and q_weights is not None and f_background is not None:
            gamma_factor_ell0, gamma_factor_ell2 = spectral_rate_factors(q_nodes, q_weights, f_background)
        else:
            gamma_factor_ell0, gamma_factor_ell2 = 1.0, 1.0
    if mismatch_factor is None:
        mismatch_factor, temperature_mismatch_log = temperature_mismatch_factor(T_gamma, T_bank)
    else:
        mismatch_factor = float(mismatch_factor)
        _, temperature_mismatch_log = temperature_mismatch_factor(T_gamma, T_bank)
    if distortion_factor_ell0 is None or distortion_factor_ell2 is None or distortion_rms is None:
        if q_nodes is not None and q_weights is not None and f_background is not None:
            distortion_factor_ell0, distortion_factor_ell2, distortion_rms = distortion_rate_factors(
                q_nodes, q_weights, f_background
            )
        else:
            distortion_factor_ell0, distortion_factor_ell2, distortion_rms = 1.0, 1.0, 0.0
    gamma_over_H_ell0 = gamma_over_H_base * float(gamma_factor_ell0) * mismatch_factor * float(
        distortion_factor_ell0
    )
    gamma_over_H_ell2 = gamma_over_H_base * float(gamma_factor_ell2) * mismatch_factor * float(
        distortion_factor_ell2
    )
    gamma_over_H = gamma_over_H_ell0

    I_mean = _weighted_mean(I_arr, w_arr)
    I_res = I_arr - I_mean
    delta_I = -float(relax) * KAPPA_ELL0 * gamma_over_H_ell0 * I_res

    # Keep the weighted mean exactly zero after numerical roundoff.
    delta_I_mean = _weighted_mean(delta_I, w_arr)
    delta_I = delta_I - delta_I_mean

    # Relax the energy-weighted Jacobian K = J exp(-8 I_res) rather than J
    # itself. This is the residual quantity that should tend to unity when the
    # isotropic energy content is already carried by the backbone.
    K_res = J_arr * np.exp(-8.0 * I_res)
    # ℓ=2 relaxation strength (BD628 W-complete, F-033 fix).
    #   "calibrated_rta" (DEFAULT): the ad-hoc constant κ₂ = 5/3 — bit-identical to
    #       the pre-BD628 path.
    #   "kernelized_scalar": replace the constant with the legacy table-derived
    #       diagnostic κ₂_eff(T_ν).  The v1 electron-minus numerator and aggregate
    #       denominator are channel-mismatched, so neither its value near 2 MeV nor an
    #       overdamping factor is a validated physical claim.  This stays opt-in.
    #   "kernelized" (full q-resolved kernel): NOT implemented in ray space (needs a
    #       ray→Ψ₂ projection, deferred) — treated as the calibrated constant here.
    if ell2_collision_mode == "kernelized_scalar":
        from rabbit.collisions.kernelized_ell2 import kappa2_eff
        kappa2 = kappa2_eff(T_bank, sp.value, H_inv_sec=H_inv_sec)
    else:
        kappa2 = KAPPA_ELL2
    delta_K = -float(relax) * kappa2 * gamma_over_H_ell2 * (K_res - 1.0)
    delta_J = np.exp(8.0 * I_res) * delta_K + 8.0 * J_arr * delta_I

    return ResidualRelaxationResult(
        delta_I=np.asarray(delta_I, dtype=np.float64),
        delta_J=np.asarray(delta_J, dtype=np.float64),
        gamma_over_H=gamma_over_H,
        gamma_over_H_base=gamma_over_H_base,
        gamma_over_H_ell0=gamma_over_H_ell0,
        gamma_over_H_ell2=gamma_over_H_ell2,
        spectral_factor_ell0=float(gamma_factor_ell0),
        spectral_factor_ell2=float(gamma_factor_ell2),
        mismatch_factor=float(mismatch_factor),
        temperature_mismatch_log=float(temperature_mismatch_log),
        distortion_factor_ell0=float(distortion_factor_ell0),
        distortion_factor_ell2=float(distortion_factor_ell2),
        distortion_rms=float(distortion_rms),
        preserved_I_mean=I_mean,
        target_J_equilibrium=1.0,
        target_energy_weighted_jacobian=1.0,
    )
