"""
rabbit.thermo.incomplete_decoupling — Neutrino decoupling thermodynamics.

Computes dT_γ/dN accounting for the gradual (non-instantaneous) decoupling
of neutrinos from the electromagnetic plasma.

Four tiers, from simplest to most complete:

    Tier 1 — Parametric N_eff = 3.044.  Instantaneous decoupling: the
             standard entropy conservation formula.  Smoke test only.

    Tier 2 — Live ν-e collision + O(e³) QED EOS.  Neutrino distributions
             are evolved with collision terms, and the energy exchange
             dQ_ν/dt feeds back into the plasma temperature evolution.
             Paper I target.  Expected: N_eff ∈ [3.033, 3.036].

    Tier 3 — + diagonal ν-ν scattering.  Thermalizes neutrino spectra
             more efficiently.  Expected: N_eff ∈ [3.042, 3.045].

    Tier 4 — + neutrino oscillations.  Full Standard Model.

The fundamental equation is entropy conservation of the γ+e± plasma
WITH a source term from neutrino energy exchange:

    d/dN [s_plasma T_γ³] = -(1/T_γ) dQ_{ν→plasma}/dN

    ⟹ dT_γ/dN = -T_γ/(1 + T_γ S'/(3S)) × [1 - dQ/(T_γ⁴ (dρ/dT))]

where dQ is the energy transfer rate from neutrinos to the plasma.

At Tier 1, dQ = 0 (instantaneous decoupling).
At Tier 2+, dQ = -∫ ε C[f_ν] d³q / (2π)³ for each species.

Instantaneous decoupling formula is FORBIDDEN in the canonical path
(Tier 2+).  Tier 1 exists only for smoke testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

from rabbit.thermo.eos_photon_electron import (
    rho_plasma, entropy_ratio_S, drho_dT, entropy_density_plasma,
    rho_photon, _M_E, _PI2, _RHO_GAMMA_PREFACTOR,
)


# ═══════════════════════════════════════════════════════════════════════
# §1. Tier enumeration
# ═══════════════════════════════════════════════════════════════════════

class DecouplingTier(IntEnum):
    """Incomplete decoupling tier."""
    PARAMETRIC = 1         # Fixed N_eff, instant decoupling
    LIVE_NU_E = 2          # Live ν-e collision (Paper I target)
    PLUS_NU_NU = 3         # + diagonal ν-ν
    FULL_SM = 4            # + oscillations


# ═══════════════════════════════════════════════════════════════════════
# §2. Entropy ratio derivative (numerical)
# ═══════════════════════════════════════════════════════════════════════

def _dS_dT(T_MeV: float, dT_frac: float = 1e-4) -> float:
    """dS/dT via central finite differences.

    S(T) = s_plasma/s_γ = entropy ratio.
    """
    if T_MeV < 1e-6:
        return 0.0
    T_hi = T_MeV * (1.0 + dT_frac)
    T_lo = T_MeV * (1.0 - dT_frac)
    S_hi = entropy_ratio_S(T_hi)
    S_lo = entropy_ratio_S(T_lo)
    return (S_hi - S_lo) / (T_hi - T_lo)


# ═══════════════════════════════════════════════════════════════════════
# §3. Tier 1: Parametric N_eff (instantaneous decoupling)
# ═══════════════════════════════════════════════════════════════════════

def dT_gamma_dN_tier1(T_gamma: float) -> float:
    """Tier 1: entropy-conserving dT_γ/dN with O(e³) QED EOS.

    d/dN [S(T) T³] = 0  ⟹  dT/dN = -T / (1 + T S'(T) / (3 S(T)))

    No energy exchange with neutrinos.  Instantaneous decoupling.
    Used ONLY for smoke testing.
    """
    if T_gamma < 1e-6:
        return 0.0

    S_val = entropy_ratio_S(T_gamma)
    dSdT = _dS_dT(T_gamma)

    correction = T_gamma * dSdT / (3.0 * S_val)
    return -T_gamma / (1.0 + correction)


# ═══════════════════════════════════════════════════════════════════════
# §4. Tier 2: Live ν-e collision energy exchange
# ═══════════════════════════════════════════════════════════════════════

def dT_gamma_dN_tier2(
    T_gamma: float,
    dQ_nu_to_plasma_per_dN: float = 0.0,
) -> float:
    """Tier 2: dT_γ/dN with live energy exchange from neutrino collisions.

    The entropy of the photon-electron plasma is modified by energy
    transfer from/to the neutrino sector:

        d/dN [s_plasma a³] = (1/T_γ) × dQ_{ν→plasma}/dN

    This gives:

        dT_γ/dN = -T_γ/(1 + T_γ S'/(3S)) + dQ/(a³ T_γ (ds_plasma/dT_γ))

    In practice, we parameterize the correction as an additive term:

        dT_γ/dN = dT_γ/dN|_{no exchange} + δ(dT/dN)|_{exchange}

    where the exchange term depends on dQ_ν_to_plasma / (∂ρ_plasma/∂T).

    Parameters
    ----------
    T_gamma : float
        Photon temperature [MeV].
    dQ_nu_to_plasma_per_dN : float
        Energy transfer rate from neutrinos to plasma per e-fold [MeV⁴].
        Positive means neutrinos heat the plasma (occurs during incomplete
        decoupling as e± annihilation entropy is partially shared).
        Computed from ∫ ε C[f_ν] d³q summed over species.

    Returns
    -------
    float
        dT_γ/dN [MeV].
    """
    if T_gamma < 1e-6:
        return 0.0

    # Base: entropy conservation (no exchange)
    dT_base = dT_gamma_dN_tier1(T_gamma)

    if abs(dQ_nu_to_plasma_per_dN) < 1e-50:
        return dT_base

    # Exchange correction: δ(dT/dN) = dQ / (dρ_plasma/dT_γ)
    # dρ/dT has units [MeV³], dQ has units [MeV⁴], so δT has units [MeV]
    drdT = drho_dT(T_gamma)
    if abs(drdT) < 1e-50:
        return dT_base

    delta_dT = dQ_nu_to_plasma_per_dN / drdT

    return dT_base + delta_dT


# ═══════════════════════════════════════════════════════════════════════
# §5. Energy exchange computation
# ═══════════════════════════════════════════════════════════════════════

def compute_energy_exchange_rate(
    collision_C: np.ndarray,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    T_nu_MeV: float,
) -> float:
    """Compute the energy transfer rate from one neutrino species to the plasma.

    dQ_ν→plasma/dN = -(1/H) × T_ν⁴/(2π²) × ∫ q³ C[f](q) × e^q × w_q dq

    The sign convention: C > 0 means neutrinos gain energy (plasma loses).
    So dQ_ν→plasma = -∫ ε C dq: if neutrinos gain, plasma loses.

    Parameters
    ----------
    collision_C : ndarray, shape (N_q,)
        Collision integral C[f](q_i) at each momentum node [MeV].
    q_nodes : ndarray, shape (N_q,)
        Gauss-Laguerre momentum nodes.
    q_weights : ndarray, shape (N_q,)
        Gauss-Laguerre weights.
    T_nu_MeV : float
        Neutrino temperature [MeV].

    Returns
    -------
    float
        Energy transfer rate [MeV⁴ per unit time].
        Positive = plasma gains energy from this ν species.
    """
    # ∫₀^∞ q³ C(q) dq using Gauss-Laguerre:
    # ∫ f(q) dq = ∫ [f(q) e^q] e^{-q} dq ≈ Σ w_i [f(q_i) e^{q_i}]
    integrand = q_nodes**3 * collision_C * np.exp(q_nodes)
    energy_integral = np.sum(q_weights * integrand)

    # Factor: T_ν⁴ / (2π²) from the phase space measure
    prefactor = T_nu_MeV**4 / (2.0 * _PI2)

    # Negative sign: C > 0 means ν gains → plasma loses
    return -prefactor * energy_integral


# ═══════════════════════════════════════════════════════════════════════
# §6. Unified interface
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DecouplingResult:
    """Result of the decoupling temperature evolution step.

    Attributes
    ----------
    dT_gamma_dN : float
        Temperature derivative [MeV].
    dQ_total : float
        Total energy exchange from all ν species [MeV⁴].
    tier : DecouplingTier
        Which tier was used.
    """
    dT_gamma_dN: float
    dQ_total: float
    tier: DecouplingTier


def compute_dT_gamma_dN_coupled(
    T_gamma: float,
    tier: DecouplingTier = DecouplingTier.PARAMETRIC,
    dQ_nu_to_plasma: float = 0.0,
) -> DecouplingResult:
    """Unified dT_γ/dN computation with tier selection.

    Parameters
    ----------
    T_gamma : float
        Photon temperature [MeV].
    tier : DecouplingTier
        Which tier to use.
    dQ_nu_to_plasma : float
        Total energy exchange from all ν species per e-fold [MeV⁴].
        Only used for Tier 2+.  At Tier 1, this is ignored.

    Returns
    -------
    DecouplingResult
    """
    if tier == DecouplingTier.PARAMETRIC:
        dT = dT_gamma_dN_tier1(T_gamma)
        return DecouplingResult(dT_gamma_dN=dT, dQ_total=0.0, tier=tier)

    elif tier >= DecouplingTier.LIVE_NU_E:
        dT = dT_gamma_dN_tier2(T_gamma, dQ_nu_to_plasma)
        return DecouplingResult(dT_gamma_dN=dT, dQ_total=dQ_nu_to_plasma, tier=tier)


# ═══════════════════════════════════════════════════════════════════════
# §7. Neutrino temperature from entropy conservation (Tier 1 helper)
# ═══════════════════════════════════════════════════════════════════════

#: Asymptotic T_ν/T_γ ratio for instantaneous decoupling.
T_NU_OVER_T_GAMMA_ASYMPTOTIC: float = (4.0 / 11.0) ** (1.0 / 3.0)  # ≈ 0.7138

#: Decoupling temperature for Tier 1 transition.
T_DEC_DEFAULT: float = 2.0  # MeV

# Cached g_{*S}(T_dec)
_G_S_DEC: Optional[float] = None


def T_nu_from_T_gamma_tier1(
    T_gamma: float,
    T_dec: float = T_DEC_DEFAULT,
) -> float:
    """Neutrino temperature from photon temperature (Tier 1: instantaneous).

    T > T_dec: T_ν = T_γ  (equilibrium)
    T < T_dec: T_ν = T_γ × (g_{*S}(T_γ)/g_{*S}(T_dec))^{1/3}

    At late times: T_ν/T_γ → (4/11)^{1/3} ≈ 0.7138.
    """
    global _G_S_DEC
    if _G_S_DEC is None:
        from rabbit.thermo.eos_photon_electron import g_star_s
        _G_S_DEC = g_star_s(T_dec)

    if T_gamma >= T_dec:
        return T_gamma

    from rabbit.thermo.eos_photon_electron import g_star_s
    g_s = g_star_s(T_gamma)
    return T_gamma * (g_s / _G_S_DEC) ** (1.0 / 3.0)


def N_eff_from_T_ratio(T_nu: float, T_gamma: float) -> float:
    """Compute effective N_eff from the T_ν/T_γ ratio.

    N_eff = 3 × (T_ν/T_ν^{std})⁴

    where T_ν^{std} = T_γ × (4/11)^{1/3} is the instantaneous value.
    """
    T_nu_std = T_gamma * T_NU_OVER_T_GAMMA_ASYMPTOTIC
    if T_nu_std < 1e-30:
        return 3.044
    return 3.0 * (T_nu / T_nu_std) ** 4


def N_eff_from_two_T(T_nue: float, T_nux: float, T_gamma: float) -> float:
    """N_eff from separate ν_e and ν_x temperatures.

    N_eff = (T_νe/T_ν^std)⁴ + 2 × (T_νx/T_ν^std)⁴
    """
    T_nu_std = T_gamma * T_NU_OVER_T_GAMMA_ASYMPTOTIC
    if T_nu_std < 1e-30:
        return 3.044
    return (T_nue / T_nu_std)**4 + 2.0 * (T_nux / T_nu_std)**4


# ═══════════════════════════════════════════════════════════════════════
# §8. Tier 2: Three-temperature coupled ODE system
# ═══════════════════════════════════════════════════════════════════════

def tier2_three_T_rhs(T_gamma, T_nue, T_nux, H_MeV):
    """Deprecated Tier-2 three-temperature RHS surface.

    This legacy helper used stale sign and flavour conventions and attempted
    to import a removed ``nudec_tables.energy_transfer_rate`` symbol at call
    time.  The active no-QKE path is the bank-aware 3T closure in
    :mod:`rabbit.thermo.nudec_coupled`.
    """
    raise NotImplementedError(
        "tier2_three_T_rhs is deprecated and intentionally unavailable; "
        "use rabbit.thermo.nudec_coupled.coupled_3T_rhs_from_collision_moments "
        "or coupled_3T_rhs for the active bank-aware 3T closure."
    )
