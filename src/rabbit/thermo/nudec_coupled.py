"""
rabbit.thermo.nudec_coupled — 3-temperature coupled neutrino decoupling ODE.

Convention: _rho_nu_pair(T) = (7/8)(π²/15)T⁴ is ONE ν+ν̄ pair.
We have 3 pairs: 1 ν_e pair + 2 ν_x pairs.

Total: ρ_ν = 1×_rho_pair(T_νe) + 2×_rho_pair(T_νx) → N_eff=3 when equal.
"""
import numpy as np
from rabbit.thermo.eos_photon_electron import (
    drho_dT,
    drho_dT_plasma_with_electron_mu,
    entropy_ratio_S,
    pressure_plasma_with_electron_mu,
    rho_plasma,
    rho_plasma_with_electron_mu,
    _RHO_GAMMA_PREFACTOR,
)
from rabbit.thermo.nudec_tables import total_energy_transfer
from rabbit.validation.truncation_guards import enforce_positive_typeI_omega

_G_N = 1.0 / (1.22089e22)**2  # MeV⁻²
_MEV_TO_S = 1.519267447e21
_NEFF_3T_ASYMPTOTIC_T_STOP_MEV = 1.0e-3
_NEFF_3T_ANNIHILATION_COMPLETE_T_MEV = 1.7e-2
_NEFF_3T_NO_QKE_CLASSICAL_TARGET = 3.034
_NEFF_3T_FLOOR_BAND_MIN = 3.00
_NEFF_3T_FLOOR_BAND_MAX = 3.06


def _rho_nu_pair(T_nu):
    """One ν+ν̄ pair: (7/8)(π²/15)T⁴."""
    return (7.0/8.0) * _RHO_GAMMA_PREFACTOR * T_nu**4


def _drho_nu_pair_dT(T_nu):
    """d/dT of one pair: 4(7/8)(π²/15)T³."""
    return 4.0 * (7.0/8.0) * _RHO_GAMMA_PREFACTOR * T_nu**3


def nu_nu_temperature_equilibration_sources(T_nu_e, T_nu_x, H_MeV):
    """Energy-conserving diagonal ν-ν temperature equilibration per e-fold.

    Returns ``(dQ_νe_pair/dN, dQ_νx_bank/dN)`` for a 3T model with one
    electron-neutrino pair and two heavy-flavour pairs.  The source relaxes
    both neutrino sectors toward the common energy-conserving temperature and
    exchanges no energy with the electromagnetic plasma.
    """
    if H_MeV < 1e-100:
        return (0.0, 0.0)
    from rabbit.thermo.rate_prefactors import gamma_nu_nu_over_H

    T_common_4 = max((T_nu_e**4 + 2.0 * T_nu_x**4) / 3.0, 0.0)
    T_common = T_common_4**0.25
    rho_common = _rho_nu_pair(T_common)
    rho_e = _rho_nu_pair(T_nu_e)

    # Distinguishable ν_e-x scattering against two heavy-flavour pairs.
    gamma_over_H = 2.0 * gamma_nu_nu_over_H(
        T_common,
        H_MeV,
        alpha_eq_beta=False,
    )
    rho_delta = rho_common - rho_e
    if abs(T_nu_e - T_nu_x) <= 1e-14 * max(abs(T_nu_e), abs(T_nu_x), 1.0):
        rho_delta = 0.0
    dQ_nue_pair_N = gamma_over_H * rho_delta
    dQ_nux_bank_N = -dQ_nue_pair_N
    return (dQ_nue_pair_N, dQ_nux_bank_N)


def _validate_electron_mu(chemical_potential_MeV: float) -> float:
    mu = float(chemical_potential_MeV)
    if not np.isfinite(mu):
        raise ValueError("electron_chemical_potential_MeV must be finite.")
    return mu


def _validate_optional_thermo_scalar(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _plasma_temperature_base_rhs(
    T_gamma: float,
    electron_chemical_potential_MeV: float,
    *,
    qed_correction_model: str = "finite_mu_scaled",
) -> float:
    mu = _validate_electron_mu(electron_chemical_potential_MeV)
    if mu == 0.0 and str(qed_correction_model) in {"scaled", "finite_mu_scaled"}:
        S = entropy_ratio_S(T_gamma)
        dT_frac = 1e-4
        dSdT = (
            entropy_ratio_S(T_gamma * (1 + dT_frac))
            - entropy_ratio_S(T_gamma * (1 - dT_frac))
        ) / (2 * dT_frac * T_gamma)
        return -T_gamma / (1.0 + T_gamma * dSdT / (3.0 * S))
    rho_em = rho_plasma_with_electron_mu(
        T_gamma,
        mu,
        qed_correction_model=qed_correction_model,
    )
    pressure_em = pressure_plasma_with_electron_mu(
        T_gamma,
        mu,
        qed_correction_model=qed_correction_model,
    )
    drdT_plasma = drho_dT_plasma_with_electron_mu(
        T_gamma,
        mu,
        qed_correction_model=qed_correction_model,
    )
    return -3.0 * (rho_em + pressure_em) / drdT_plasma if abs(drdT_plasma) > 1e-50 else -T_gamma


def hubble_3T(
    T_gamma,
    T_nu_e,
    T_nu_x,
    Sigma_sq=0.0,
    electron_chemical_potential_MeV: float = 0.0,
    qed_correction_model: str = "finite_mu_scaled",
):
    """Hubble rate [MeV]. 1 ν_e pair + 2 ν_x pairs = 3 pairs total.

    For Bianchi: H² = (8πG/3) × ρ_total / (1 − Σ²)
    where Σ² = Σ₊² + Σ₋² is the Hubble-normalized shear scalar.
    """
    mu = _validate_electron_mu(electron_chemical_potential_MeV)
    rho_em = rho_plasma_with_electron_mu(
        T_gamma,
        mu,
        qed_correction_model=qed_correction_model,
    )
    rho_nu = 1.0 * _rho_nu_pair(T_nu_e) + 2.0 * _rho_nu_pair(T_nu_x)
    Omega = enforce_positive_typeI_omega(Sigma_sq, context="thermo.nudec_coupled.hubble_3T", strict=True)
    return np.sqrt(max((8.0*np.pi*_G_N/3.0)*(rho_em + rho_nu) / Omega, 0.0))


def coupled_3T_rhs(
    T_gamma,
    T_nu_e,
    T_nu_x,
    H_MeV=None,
    enable_nu_nu_equilibration=False,
    electron_chemical_potential_MeV: float = 0.0,
    qed_correction_model: str = "finite_mu_scaled",
    plasma_drho_dT_MeV3: float | None = None,
    plasma_dT_base_dN: float | None = None,
):
    """(dT_γ/dN, dT_νe/dN, dT_νx/dN)."""
    if T_gamma < 1e-6:
        return (0.0, 0.0, 0.0)
    mu = _validate_electron_mu(electron_chemical_potential_MeV)
    if H_MeV is None:
        H_MeV = hubble_3T(
            T_gamma,
            T_nu_e,
            T_nu_x,
            electron_chemical_potential_MeV=mu,
            qed_correction_model=qed_correction_model,
        )
    if H_MeV < 1e-100:
        return (-T_gamma, -T_nu_e, -T_nu_x)
    drho_override = _validate_optional_thermo_scalar(
        plasma_drho_dT_MeV3,
        "plasma_drho_dT_MeV3",
    )
    base_override = _validate_optional_thermo_scalar(
        plasma_dT_base_dN,
        "plasma_dT_base_dN",
    )

    # Energy transfer rates [MeV⁵]:
    # dQ_nue = 2×single_rate (ν_e pair), dQ_nux = 4×single_rate (2 ν_x pairs)
    dQ_nue, dQ_nux = total_energy_transfer(T_gamma, T_nu_e, T_nu_x)
    dQ_nue_N = dQ_nue / H_MeV  # per e-fold
    dQ_nux_N = dQ_nux / H_MeV
    if enable_nu_nu_equilibration:
        dQ_nue_nunu_N, dQ_nux_nunu_N = nu_nu_temperature_equilibration_sources(
            T_nu_e, T_nu_x, H_MeV
        )
        dQ_nue_N += dQ_nue_nunu_N
        dQ_nux_N += dQ_nux_nunu_N
    dQ_total_N = dQ_nue_N + dQ_nux_N

    # ── dT_γ/dN ──
    dT_base = (
        base_override
        if base_override is not None
        else _plasma_temperature_base_rhs(
            T_gamma,
            mu,
            qed_correction_model=qed_correction_model,
        )
    )
    drdT_plasma = (
        drho_override
        if drho_override is not None
        else drho_dT_plasma_with_electron_mu(
            T_gamma,
            mu,
            qed_correction_model=qed_correction_model,
        )
    )
    dT_source = -dQ_total_N / drdT_plasma if abs(drdT_plasma) > 1e-50 else 0.0
    dT_gamma_dN = dT_base + dT_source

    # ── dT_νe/dN: 1 pair ──
    # dQ_nue_N = energy gain of the ν_e pair per e-fold
    # denominator: dρ/dT of 1 pair
    d1 = _drho_nu_pair_dT(T_nu_e)
    dT_nue_dN = -T_nu_e + (dQ_nue_N / d1 if d1 > 1e-50 else 0.0)

    # ── dT_νx/dN: 2 pairs at same T ──
    # dQ_nux_N = total energy gain of 2 pairs per e-fold
    # denominator: dρ/dT of 2 pairs
    d2 = 2.0 * _drho_nu_pair_dT(T_nu_x)
    dT_nux_dN = -T_nu_x + (dQ_nux_N / d2 if d2 > 1e-50 else 0.0)

    return (dT_gamma_dN, dT_nue_dN, dT_nux_dN)


def coupled_3T_rhs_from_collision_moments(
    T_gamma,
    T_nu_e,
    T_nu_x,
    *,
    dQ_nue_pair_N: float,
    dQ_nux_bank_N: float,
    enable_nu_nu_equilibration: bool = False,
    electron_chemical_potential_MeV: float = 0.0,
    qed_correction_model: str = "finite_mu_scaled",
    plasma_drho_dT_MeV3: float | None = None,
    plasma_dT_base_dN: float | None = None,
):
    """3T thermo RHS sourced directly by collision moments.

    Parameters
    ----------
    dQ_nue_pair_N : float
        Energy gained by the ν_e + ν̄_e pair per e-fold.
    dQ_nux_bank_N : float
        Energy gained by the full ν_x bank (4 states) per e-fold.
    """
    if T_gamma < 1e-6:
        return (0.0, 0.0, 0.0)

    mu = _validate_electron_mu(electron_chemical_potential_MeV)
    drho_override = _validate_optional_thermo_scalar(
        plasma_drho_dT_MeV3,
        "plasma_drho_dT_MeV3",
    )
    base_override = _validate_optional_thermo_scalar(
        plasma_dT_base_dN,
        "plasma_dT_base_dN",
    )
    dT_base = (
        base_override
        if base_override is not None
        else _plasma_temperature_base_rhs(
            T_gamma,
            mu,
            qed_correction_model=qed_correction_model,
        )
    )

    drdT_plasma = (
        drho_override
        if drho_override is not None
        else drho_dT_plasma_with_electron_mu(
            T_gamma,
            mu,
            qed_correction_model=qed_correction_model,
        )
    )
    dQ_nue_pair_N = float(dQ_nue_pair_N)
    dQ_nux_bank_N = float(dQ_nux_bank_N)
    if enable_nu_nu_equilibration:
        H_MeV = hubble_3T(
            T_gamma,
            T_nu_e,
            T_nu_x,
            electron_chemical_potential_MeV=mu,
            qed_correction_model=qed_correction_model,
        )
        dQ_nue_nunu_N, dQ_nux_nunu_N = nu_nu_temperature_equilibration_sources(
            T_nu_e, T_nu_x, H_MeV
        )
        dQ_nue_pair_N += dQ_nue_nunu_N
        dQ_nux_bank_N += dQ_nux_nunu_N
    dQ_total_N = dQ_nue_pair_N + dQ_nux_bank_N
    dT_source = -dQ_total_N / drdT_plasma if abs(drdT_plasma) > 1e-50 else 0.0
    dT_gamma_dN = dT_base + dT_source

    d1 = _drho_nu_pair_dT(T_nu_e)
    dT_nue_dN = -T_nu_e + (dQ_nue_pair_N / d1 if d1 > 1e-50 else 0.0)

    d2 = 2.0 * _drho_nu_pair_dT(T_nu_x)
    dT_nux_dN = -T_nu_x + (dQ_nux_bank_N / d2 if d2 > 1e-50 else 0.0)

    return (dT_gamma_dN, dT_nue_dN, dT_nux_dN)


def N_eff_from_3T(T_gamma, T_nu_e, T_nu_x):
    """N_eff = (T_νe/T_std)⁴ + 2(T_νx/T_std)⁴ where T_std = T_γ(4/11)^{1/3}."""
    T_std = T_gamma * (4.0/11.0)**(1.0/3.0)
    if T_std < 1e-30:
        return float("nan")
    re = (T_nu_e / T_std)**4
    rx = (T_nu_x / T_std)**4
    return re + 2.0 * rx


def _neff_3t_band_payload(value: float | None) -> dict[str, object]:
    if value is None or not np.isfinite(float(value)):
        return {
            "N_eff_3T_asymptotic_floor_band_min": float(_NEFF_3T_FLOOR_BAND_MIN),
            "N_eff_3T_asymptotic_floor_band_max": float(_NEFF_3T_FLOOR_BAND_MAX),
            "N_eff_3T_asymptotic_floor_band_passed": None,
            "N_eff_3T_asymptotic_delta_from_expected": None,
            "N_eff_3T_asymptotic_delta_from_band_min": None,
            "N_eff_3T_asymptotic_delta_from_band_max": None,
        }
    scalar = float(value)
    return {
        "N_eff_3T_asymptotic_floor_band_min": float(_NEFF_3T_FLOOR_BAND_MIN),
        "N_eff_3T_asymptotic_floor_band_max": float(_NEFF_3T_FLOOR_BAND_MAX),
        "N_eff_3T_asymptotic_floor_band_passed": bool(
            _NEFF_3T_FLOOR_BAND_MIN <= scalar <= _NEFF_3T_FLOOR_BAND_MAX
        ),
        "N_eff_3T_asymptotic_delta_from_expected": float(
            scalar - _NEFF_3T_NO_QKE_CLASSICAL_TARGET
        ),
        "N_eff_3T_asymptotic_delta_from_band_min": float(
            scalar - _NEFF_3T_FLOOR_BAND_MIN
        ),
        "N_eff_3T_asymptotic_delta_from_band_max": float(
            scalar - _NEFF_3T_FLOOR_BAND_MAX
        ),
    }


def _neff_3t_unavailable_payload(
    reason: str,
    *,
    T_gamma: float | None,
    T_nu_e: float | None,
    T_nu_x: float | None,
    N_eff_in_span: float | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "available": False,
        "unavailable_reason": str(reason),
        "N_eff_3T_in_span": N_eff_in_span,
        "N_eff_3T_asymptotic": None,
        "T_gamma_in_span_MeV": T_gamma,
        "T_nu_e_in_span_MeV": T_nu_e,
        "T_nu_x_in_span_MeV": T_nu_x,
        "T_gamma_asymptotic_MeV": None,
        "T_nu_e_asymptotic_MeV": None,
        "T_nu_x_asymptotic_MeV": None,
        "tail_integrated": False,
        "tail_reached_stop": False,
        "tail_N_elapsed": None,
        "tail_step_count": 0,
        "annihilation_complete": False,
        "N_eff_3T_claim_status": "diagnostic_proxy_not_physical_N_eff",
        "N_eff_3T_asymptotic_readout_scope": "thermo_only_tail_no_network_no_phase2",
        "expected_no_qke_classical_target": float(_NEFF_3T_NO_QKE_CLASSICAL_TARGET),
        "T_stop_MeV": float(_NEFF_3T_ASYMPTOTIC_T_STOP_MEV),
    }
    payload.update(_neff_3t_band_payload(None))
    return payload


def asymptotic_N_eff_3T_payload(
    T_gamma,
    T_nu_e,
    T_nu_x,
    *,
    T_stop_MeV: float = _NEFF_3T_ASYMPTOTIC_T_STOP_MEV,
    enable_nu_nu_equilibration: bool = False,
    electron_chemical_potential_MeV: float = 0.0,
    qed_correction_model: str = "finite_mu_scaled",
    rtol: float = 1.0e-10,
    atol: float = 1.0e-13,
    max_step: float = 0.05,
    max_tail_N: float = 80.0,
) -> dict[str, object]:
    """Continue the 3T thermo-only closure and report an asymptotic N_eff proxy.

    This is a diagnostic readout only: it evolves the background temperature
    closure without the nuclear network, phase-2 corrector, or QKE.  It keeps
    the in-span value so hot truncated rows remain auditable, and uses an
    explicit ``available=false`` payload when a tail cannot be computed.
    """

    try:
        Tg0 = float(T_gamma)
        Te0 = float(T_nu_e)
        Tx0 = float(T_nu_x)
        T_stop = float(T_stop_MeV)
        N_eff_in_span = float(N_eff_from_3T(Tg0, Te0, Tx0))
    except (TypeError, ValueError) as exc:
        return _neff_3t_unavailable_payload(
            f"invalid_temperature_input:{exc}",
            T_gamma=None,
            T_nu_e=None,
            T_nu_x=None,
            N_eff_in_span=None,
        )
    if (
        not np.isfinite(Tg0)
        or not np.isfinite(Te0)
        or not np.isfinite(Tx0)
        or Tg0 <= 0.0
        or Te0 <= 0.0
        or Tx0 <= 0.0
    ):
        return _neff_3t_unavailable_payload(
            "temperatures_must_be_positive_finite",
            T_gamma=Tg0,
            T_nu_e=Te0,
            T_nu_x=Tx0,
            N_eff_in_span=N_eff_in_span if np.isfinite(N_eff_in_span) else None,
        )
    if not np.isfinite(T_stop) or T_stop <= 0.0:
        return _neff_3t_unavailable_payload(
            "T_stop_MeV_must_be_positive_finite",
            T_gamma=Tg0,
            T_nu_e=Te0,
            T_nu_x=Tx0,
            N_eff_in_span=N_eff_in_span,
        )

    if Tg0 <= T_stop:
        payload: dict[str, object] = {
            "available": True,
            "unavailable_reason": None,
            "N_eff_3T_in_span": float(N_eff_in_span),
            "N_eff_3T_asymptotic": float(N_eff_in_span),
            "T_gamma_in_span_MeV": float(Tg0),
            "T_nu_e_in_span_MeV": float(Te0),
            "T_nu_x_in_span_MeV": float(Tx0),
            "T_gamma_asymptotic_MeV": float(Tg0),
            "T_nu_e_asymptotic_MeV": float(Te0),
            "T_nu_x_asymptotic_MeV": float(Tx0),
            "tail_integrated": False,
            "tail_reached_stop": True,
            "tail_N_elapsed": 0.0,
            "tail_step_count": 0,
            "annihilation_complete": bool(
                Tg0 <= _NEFF_3T_ANNIHILATION_COMPLETE_T_MEV
            ),
            "N_eff_3T_claim_status": "diagnostic_proxy_not_physical_N_eff",
            "N_eff_3T_asymptotic_readout_scope": "thermo_only_tail_no_network_no_phase2",
            "expected_no_qke_classical_target": float(_NEFF_3T_NO_QKE_CLASSICAL_TARGET),
            "T_stop_MeV": float(T_stop),
            "solver_success": True,
            "solver_message": "initial_state_already_below_T_stop",
        }
        payload.update(_neff_3t_band_payload(float(N_eff_in_span)))
        return payload

    try:
        from scipy.integrate import solve_ivp
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        return _neff_3t_unavailable_payload(
            f"scipy_solve_ivp_unavailable:{exc}",
            T_gamma=Tg0,
            T_nu_e=Te0,
            T_nu_x=Tx0,
            N_eff_in_span=N_eff_in_span,
        )

    mu = _validate_electron_mu(electron_chemical_potential_MeV)

    def rhs(_N, y):
        H = hubble_3T(
            y[0],
            y[1],
            y[2],
            electron_chemical_potential_MeV=mu,
            qed_correction_model=qed_correction_model,
        )
        return coupled_3T_rhs(
            y[0],
            y[1],
            y[2],
            H_MeV=H,
            enable_nu_nu_equilibration=enable_nu_nu_equilibration,
            electron_chemical_potential_MeV=mu,
            qed_correction_model=qed_correction_model,
        )

    def reached_T_stop(_N, y):
        return float(y[0]) - T_stop

    reached_T_stop.terminal = True
    reached_T_stop.direction = -1
    try:
        sol = solve_ivp(
            rhs,
            (0.0, float(max_tail_N)),
            (Tg0, Te0, Tx0),
            method="LSODA",
            rtol=float(rtol),
            atol=float(atol),
            events=reached_T_stop,
            max_step=float(max_step),
        )
    except Exception as exc:
        return _neff_3t_unavailable_payload(
            f"tail_integration_failed:{exc}",
            T_gamma=Tg0,
            T_nu_e=Te0,
            T_nu_x=Tx0,
            N_eff_in_span=N_eff_in_span,
        )
    if sol.y.size == 0 or sol.y.shape[0] < 3:
        return _neff_3t_unavailable_payload(
            "tail_integration_returned_empty_solution",
            T_gamma=Tg0,
            T_nu_e=Te0,
            T_nu_x=Tx0,
            N_eff_in_span=N_eff_in_span,
        )
    Tg, Te, Tx = (float(sol.y[0, -1]), float(sol.y[1, -1]), float(sol.y[2, -1]))
    if (
        not np.isfinite(Tg)
        or not np.isfinite(Te)
        or not np.isfinite(Tx)
        or Tg <= 0.0
        or Te <= 0.0
        or Tx <= 0.0
    ):
        return _neff_3t_unavailable_payload(
            "tail_integration_returned_nonpositive_or_nonfinite_temperature",
            T_gamma=Tg0,
            T_nu_e=Te0,
            T_nu_x=Tx0,
            N_eff_in_span=N_eff_in_span,
        )
    N_eff_asymptotic = float(N_eff_from_3T(Tg, Te, Tx))
    reached_stop = bool(
        sol.t_events
        and len(sol.t_events) >= 1
        and np.asarray(sol.t_events[0]).size > 0
    )
    if not bool(sol.success) or not reached_stop:
        reason = (
            f"tail_stop_not_reached:{sol.message}"
            if bool(sol.success)
            else f"tail_solver_failed:{sol.message}"
        )
        return _neff_3t_unavailable_payload(
            reason,
            T_gamma=Tg0,
            T_nu_e=Te0,
            T_nu_x=Tx0,
            N_eff_in_span=N_eff_in_span,
        )
    payload = {
        "available": True,
        "unavailable_reason": None,
        "N_eff_3T_in_span": float(N_eff_in_span),
        "N_eff_3T_asymptotic": float(N_eff_asymptotic),
        "T_gamma_in_span_MeV": float(Tg0),
        "T_nu_e_in_span_MeV": float(Te0),
        "T_nu_x_in_span_MeV": float(Tx0),
        "T_gamma_asymptotic_MeV": float(Tg),
        "T_nu_e_asymptotic_MeV": float(Te),
        "T_nu_x_asymptotic_MeV": float(Tx),
        "tail_integrated": True,
        "tail_reached_stop": bool(reached_stop),
        "tail_N_elapsed": float(sol.t[-1]) if len(sol.t) else None,
        "tail_step_count": int(max(len(sol.t) - 1, 0)),
        "annihilation_complete": bool(Tg <= _NEFF_3T_ANNIHILATION_COMPLETE_T_MEV),
        "N_eff_3T_claim_status": "diagnostic_proxy_not_physical_N_eff",
        "N_eff_3T_asymptotic_readout_scope": "thermo_only_tail_no_network_no_phase2",
        "expected_no_qke_classical_target": float(_NEFF_3T_NO_QKE_CLASSICAL_TARGET),
        "T_stop_MeV": float(T_stop),
        "solver_success": bool(sol.success),
        "solver_message": str(sol.message),
    }
    payload.update(_neff_3t_band_payload(N_eff_asymptotic))
    return payload
