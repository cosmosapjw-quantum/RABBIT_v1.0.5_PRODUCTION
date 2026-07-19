"""rabbit.collisions.dynamic_collision_core — clean dynamic-collision core (plan Phase 2).

Replaces the ill-conditioned temperature-variable 3T evolution that collapses the
q-Laguerre dynamic-collision path (root cause pinned in
``docs/audit/BD205_qlaguerre_collision_root_cause_2026-07-01.md`` /
``tests/test_qlaguerre_collision_conditioning.py``):

    dT_νx/dN = −T_νx + dQ_nux_bank_N / c_v ,   c_v = 2·dρ_pair/dT ∝ T_νx³

As the heavy neutrino bank decouples (T_νx → 0) the heat capacity → 0, so ANY error
in ``dQ_nux_bank_N`` is amplified by ~1/T_νx³ into a catastrophic temperature kick
(measured −4.1e+06/e-fold at the BD203 T_νx). The fix is to evolve the bank ENERGY
density instead of its temperature:

    dρ_bank/dN = −4·ρ_bank + dQ_bank_N          (exact; ρ_pair ∝ T⁴ ⟹ (dρ/dT)·T = 4ρ)

which is bounded for any T_νx (a small dQ error gives a small dρ error), with T_νx
recovered diagnostically as ρ^{1/4}. This module holds the well-conditioned energy
variables the clean core evolves; the A-mode q-Laguerre distribution + collision
source build on top of it.

No external anchor exists for Bianchi-I BBN with a neutrino collision term (no such
paper); this is an internal conditioning/consistency contract, never external validation.
"""

from __future__ import annotations

import numpy as np

from rabbit.collisions.deterministic_reference import (
    evaluate_nue_scattering_reference,
    evaluate_pair_annihilation_reference,
)
from rabbit.collisions.species import BANK_DEGENERACY, Species
from rabbit.thermo.nudec_coupled import (
    _rho_nu_pair,
    coupled_3T_rhs_from_collision_moments,
    hubble_3T,
)
from rabbit.transport.augmented_pstf_distribution import (
    fermi_dirac_from_logit,
    reconstruct_distribution,
)

#: Equilibrium spectral energy moment ∫q³/(e^q+1)dq = 7π⁴/120 (T_ν diagnostic reference).
_EQ_ENERGY_MOMENT = 7.0 * np.pi**4 / 120.0

#: Energy density of one ν+ν̄ pair at T=1 MeV: ρ_pair(T) = _C_PAIR · T⁴  [MeV⁴].
_C_PAIR = _rho_nu_pair(1.0)

#: Standard-model bank sizes (number of ν+ν̄ pairs): ν_e = 1 pair, ν_x = 2 pairs.
N_PAIRS_NUE = 1
N_PAIRS_NUX = 2


def nu_bank_energy_from_T(T_nu: float, n_pairs: int) -> float:
    """Energy density ρ_bank [MeV⁴] of an ``n_pairs``-pair neutrino bank at temperature T_nu."""
    return float(n_pairs) * _rho_nu_pair(float(T_nu))


def nu_bank_T_from_energy(rho_bank: float, n_pairs: int) -> float:
    """Invert ρ_bank = n_pairs·_C_PAIR·T⁴ → T_nu [MeV]. Guards ρ<0 (numerical) to 0."""
    rho = max(float(rho_bank), 0.0)
    return (rho / (float(n_pairs) * _C_PAIR)) ** 0.25


def dnu_bank_energy_dN(rho_bank: float, dQ_bank_N: float) -> float:
    """Well-conditioned energy RHS for any T⁴ neutrino bank: dρ/dN = −4ρ + dQ.

    No division by a heat capacity that vanishes at decoupling, so a small dQ error
    cannot blow up (contrast the temperature form's 1/T³ amplifier). This is the
    root fix for the q-Laguerre collision collapse (BD205)."""
    return -4.0 * float(rho_bank) + float(dQ_bank_N)


def _laguerre_dq_weights(q_nodes: np.ndarray, q_weights: np.ndarray) -> np.ndarray:
    """Plain quadrature weights for ∫₀^∞ g(q) dq on the q-Laguerre grid: w_i·exp(q_i)
    (removes the built-in exp(−q) so a plain-decaying integrand is integrated). The
    exp is clipped at 500 for overflow safety; the FD tail makes the integrand decay."""
    q = np.asarray(q_nodes, dtype=float)
    return np.asarray(q_weights, dtype=float) * np.exp(np.minimum(q, 500.0))


def spectral_energy_moment(f_q: np.ndarray, q_nodes: np.ndarray, q_weights: np.ndarray) -> float:
    """Dimensionless spectral energy moment ∫₀^∞ q³ f(q) dq (per ν+ν̄ degree of freedom,
    q in units of T). Equilibrium f₀=1/(e^q+1) gives 7π⁴/120."""
    dq = _laguerre_dq_weights(q_nodes, q_weights)
    return float(np.sum(dq * np.asarray(q_nodes, dtype=float) ** 3 * np.asarray(f_q, dtype=float)))


def spectral_number_moment(f_q: np.ndarray, q_nodes: np.ndarray, q_weights: np.ndarray) -> float:
    """Dimensionless spectral number moment ∫₀^∞ q² f(q) dq. Equilibrium gives 3ζ(3)/2."""
    dq = _laguerre_dq_weights(q_nodes, q_weights)
    return float(np.sum(dq * np.asarray(q_nodes, dtype=float) ** 2 * np.asarray(f_q, dtype=float)))


def neutrino_distribution_from_modes(A_modes, q_nodes, basis_matrix):
    """Reconstruct the neutrino distribution f(q, angle) from the augmented logit A-modes.

    Re-exports the surviving logit reconstruction (transport.augmented_pstf_distribution):
    f = 1/(exp(logit)+1), logit = q + Σ_m A_m·B_m(angle). Positivity f∈(0,1) is guaranteed
    by the logit form for ANY finite A — the realizability property the clean core relies on
    (so no separate positivity clamp/guard is needed on the distribution). ``A_modes`` = 0
    recovers the equilibrium Fermi-Dirac spectrum."""
    return reconstruct_distribution(A_modes, q_nodes, basis_matrix)


def neutrino_collision_energy_transfer(
    f_nu,
    f_nu_partner,
    *,
    quadrature,
    T_gamma_MeV: float,
    species: str = "nue",
    electron_chemical_potential_MeV: float = 0.0,
):
    """First-principles energy transfer INTO one neutrino distribution [MeV⁵], from the
    plasma via ν-e scattering (ν e± → ν e±) + pair processes (e⁺e⁻ ↔ ν ν̄).

    Reuses the surviving deterministic-reference collision integrals
    (``collisions/deterministic_reference``): full weak matrix element ``M²``, exact
    detailed-balance statistical factor, fixed Gauss collision quadrature. The energy
    transfer is ``∫ q³ C[f] dq``; it VANISHES at equilibrium (f_nu, e± both Fermi-Dirac at
    ``T_gamma``) by detailed balance, and is POSITIVE (heating) when the neutrinos are colder
    than the plasma. ``f_nu``/``f_nu_partner`` must be sampled on ``quadrature.q_nodes``.

    ``species`` selects the weak couplings (``"nue"`` charged+neutral current, ``"nux"``
    neutral current only). ``T_gamma_MeV`` is the plasma/electron temperature (the ν occupancy
    is carried by ``f_nu`` directly)."""
    scatt = evaluate_nue_scattering_reference(
        f_nu, quadrature=quadrature, T_MeV=T_gamma_MeV, species=species,
        electron_chemical_potential_MeV=electron_chemical_potential_MeV,
    )
    pair = evaluate_pair_annihilation_reference(
        f_nu, f_nu_partner, quadrature=quadrature, T_MeV=T_gamma_MeV, species=species,
        electron_chemical_potential_MeV=electron_chemical_potential_MeV,
    )
    dQ = (
        sum(scatt.energy_transfer_by_species.values())
        + sum(pair.energy_transfer_by_species.values())
    )
    db_residual = max(scatt.detailed_balance_residual, pair.detailed_balance_residual)
    return float(dQ), float(db_residual)


def neutrino_temperature_from_spectrum(f_q, q_nodes, q_weights, T_gamma_MeV: float) -> float:
    """Diagnostic ν temperature from the spectral energy: T_ν = T_γ·(∫q³f / (7π⁴/120))^{1/4}.
    Equals T_γ at equilibrium (f = FD); well-conditioned (a ρ^{1/4}, never a 1/T³ division)."""
    ratio = max(spectral_energy_moment(f_q, q_nodes, q_weights) / _EQ_ENERGY_MOMENT, 0.0)
    return float(T_gamma_MeV) * ratio ** 0.25


def _collision_field(f_nu, f_partner, quadrature, T_gamma_MeV, species, mu):
    """Total first-principles collision field C[f] = df/dt (ν-e scattering + pair) [MeV]."""
    scatt = evaluate_nue_scattering_reference(
        f_nu, quadrature=quadrature, T_MeV=T_gamma_MeV, species=species,
        electron_chemical_potential_MeV=mu)
    pair = evaluate_pair_annihilation_reference(
        f_nu, f_partner, quadrature=quadrature, T_MeV=T_gamma_MeV, species=species,
        electron_chemical_potential_MeV=mu)
    return np.asarray(scatt.C, dtype=float) + np.asarray(pair.C, dtype=float)


def flrw_dynamic_collision_rhs(
    A_nue,
    A_nux,
    T_gamma_MeV: float,
    *,
    quadrature,
    electron_chemical_potential_MeV: float = 0.0,
    fnn_floor: float = 1.0e-12,
):
    """Coupled FLRW dynamic-collision RHS: ``(dA_nue/dN, dA_nux/dN, dT_γ/dN)``.

    This is the clean core's driver kernel. The neutrino state is the per-momentum logit
    distortion ``A`` (f = 1/(exp(q+A)+1), positivity free); the collision is the first-principles
    integral C[f] (PR-C4). In comoving momentum the distribution evolves by collision only:

        df/dN = C[f]/H  ⟹  dA/dN = dlogit/dN = −(C[f]/H) / (f(1−f))

    which is BOUNDED (no 1/T_νx³ amplifier — the energy is carried by the distribution, never a
    temperature divided by a vanishing heat capacity). The photon channel reuses the plasma energy
    balance via ``coupled_3T_rhs_from_collision_moments``. ``A_nue``/``A_nux`` are ``(N_q,)`` on
    ``quadrature.q_nodes``. No lepton asymmetry (ν̄ ≡ ν). Returns numpy arrays + a float."""
    q = np.asarray(quadrature.q_nodes, dtype=float)
    qw = np.asarray(quadrature.q_weights, dtype=float)
    mu = float(electron_chemical_potential_MeV)
    A_nue = np.asarray(A_nue, dtype=float)
    A_nux = np.asarray(A_nux, dtype=float)

    f_nue = fermi_dirac_from_logit(q + A_nue)
    f_nux = fermi_dirac_from_logit(q + A_nux)

    T_nue = neutrino_temperature_from_spectrum(f_nue, q, qw, T_gamma_MeV)
    T_nux = neutrino_temperature_from_spectrum(f_nux, q, qw, T_gamma_MeV)
    H_MeV = float(hubble_3T(T_gamma_MeV, T_nue, T_nux))
    H_safe = max(H_MeV, 1.0e-100)

    C_nue = _collision_field(f_nue, f_nue, quadrature, T_gamma_MeV, "nue", mu)
    C_nux = _collision_field(f_nux, f_nux, quadrature, T_gamma_MeV, "nux", mu)

    # dA/dN = −(C/H)/(f(1−f)); f(1−f)→0 only where C→0 (equilibrium tail) → set 0 there.
    fnn_e = f_nue * (1.0 - f_nue)
    fnn_x = f_nux * (1.0 - f_nux)
    dA_nue = np.where(fnn_e > fnn_floor, -(C_nue / H_safe) / np.maximum(fnn_e, fnn_floor), 0.0)
    dA_nux = np.where(fnn_x > fnn_floor, -(C_nux / H_safe) / np.maximum(fnn_x, fnn_floor), 0.0)

    dQ_nue_pair_N, dQ_nux_bank_N = collision_bank_energy_sources_per_efold(
        f_nue, f_nux, quadrature=quadrature, T_gamma_MeV=T_gamma_MeV, H_MeV=H_safe,
        electron_chemical_potential_MeV=mu)
    dT_gamma = coupled_3T_rhs_from_collision_moments(
        T_gamma_MeV, T_nue, T_nux,
        dQ_nue_pair_N=dQ_nue_pair_N, dQ_nux_bank_N=dQ_nux_bank_N)[0]
    return dA_nue, dA_nux, float(dT_gamma)


def collision_bank_energy_sources_per_efold(
    f_nue,
    f_nux,
    *,
    quadrature,
    T_gamma_MeV: float,
    H_MeV: float,
    f_nuebar=None,
    electron_chemical_potential_MeV: float = 0.0,
):
    """First-principles per-e-fold collision energy sources for the 3T banks:
    ``(dQ_nue_pair_N, dQ_nux_bank_N)`` — the exact inputs ``coupled_3T_energy_rhs`` consumes.

    Convention matches ``nudec_coupled.total_energy_transfer`` /
    ``collisions.species.BANK_DEGENERACY``:
      - ν_e pair = ν_e + ν̄_e (each degeneracy 1);
      - ν_x bank = one effective distribution scaled by ``BANK_DEGENERACY[NUX]`` (= 4:
        2 flavors ν_μ,ν_τ × ν + ν̄).
    Absent a lepton asymmetry, ``f_nuebar`` defaults to ``f_nue`` (and the ν_x pair partner is
    ``f_nux``). Distributions must be sampled on ``quadrature.q_nodes``; ``H_MeV`` converts the
    energy-transfer rate to a per-e-fold source."""
    f_nuebar = f_nue if f_nuebar is None else f_nuebar
    H = max(float(H_MeV), 1.0e-100)

    dQ_nue, _ = neutrino_collision_energy_transfer(
        f_nue, f_nuebar, quadrature=quadrature, T_gamma_MeV=T_gamma_MeV, species="nue",
        electron_chemical_potential_MeV=electron_chemical_potential_MeV)
    dQ_nuebar, _ = neutrino_collision_energy_transfer(
        f_nuebar, f_nue, quadrature=quadrature, T_gamma_MeV=T_gamma_MeV, species="nuebar",
        electron_chemical_potential_MeV=electron_chemical_potential_MeV)
    dQ_nux, _ = neutrino_collision_energy_transfer(
        f_nux, f_nux, quadrature=quadrature, T_gamma_MeV=T_gamma_MeV, species="nux",
        electron_chemical_potential_MeV=electron_chemical_potential_MeV)

    dQ_nue_pair_N = (dQ_nue + dQ_nuebar) / H
    dQ_nux_bank_N = float(BANK_DEGENERACY[Species.NUX]) * dQ_nux / H
    return dQ_nue_pair_N, dQ_nux_bank_N


def coupled_3T_energy_rhs(
    T_gamma: float,
    rho_nue: float,
    rho_nux: float,
    *,
    dQ_nue_pair_N: float,
    dQ_nux_bank_N: float,
    **plasma_kw,
):
    """Well-conditioned 3T RHS sourced by collision moments: ``(dT_γ/dN, dρ_νe/dN, dρ_νx/dN)``.

    The photon channel is unchanged from ``nudec_coupled.coupled_3T_rhs_from_collision_moments``
    (the plasma heat capacity does not vanish, so temperature is fine there); the neutrino
    channels are evolved in ENERGY, curing the 1/T³ ill-conditioning. Away from decoupling this
    is bit-equivalent (via the chain rule) to the temperature form; near decoupling it stays
    bounded where the temperature form diverges. ``plasma_kw`` forwards the plasma overrides
    (``electron_chemical_potential_MeV``, ``qed_correction_model``, ``plasma_drho_dT_MeV3``,
    ``plasma_dT_base_dN``, ``enable_nu_nu_equilibration``)."""
    T_nue = nu_bank_T_from_energy(rho_nue, N_PAIRS_NUE)
    T_nux = nu_bank_T_from_energy(rho_nux, N_PAIRS_NUX)
    dT_gamma, _, _ = coupled_3T_rhs_from_collision_moments(
        T_gamma, T_nue, T_nux,
        dQ_nue_pair_N=dQ_nue_pair_N, dQ_nux_bank_N=dQ_nux_bank_N, **plasma_kw,
    )
    return (
        dT_gamma,
        dnu_bank_energy_dN(rho_nue, dQ_nue_pair_N),
        dnu_bank_energy_dN(rho_nux, dQ_nux_bank_N),
    )
