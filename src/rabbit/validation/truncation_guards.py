
"""Boundary and truncation guards for publishable solver behaviour.

These helpers make manual truncations and domain clamps explicit.  They are
intentionally plain-Python so they can raise/warn before entering JIT code.
"""
from __future__ import annotations

import warnings

RATE_TABLE_T9_MIN = 1.0e-3
RATE_TABLE_T9_MAX = 10.0
MEV_TO_T9 = 11.6045
RATE_TABLE_T_MEV_MIN = RATE_TABLE_T9_MIN / MEV_TO_T9
RATE_TABLE_T_MEV_MAX = RATE_TABLE_T9_MAX / MEV_TO_T9


def _emit(message: str, *, strict: bool = False, category=RuntimeWarning) -> None:
    if strict:
        raise ValueError(message)
    warnings.warn(message, category, stacklevel=2)


def validate_phase_temperature_order(T_start: float, T_handoff: float, T_end: float, *, context: str) -> None:
    if not (T_start > T_handoff > T_end > 0.0):
        raise ValueError(
            f"{context}: require T_start > T_handoff > T_end > 0. Got "
            f"({T_start}, {T_handoff}, {T_end})."
        )


def validate_rate_table_temperature(T_gamma_MeV: float, *, context: str, strict: bool = False) -> None:
    if T_gamma_MeV < RATE_TABLE_T_MEV_MIN or T_gamma_MeV > RATE_TABLE_T_MEV_MAX:
        _emit(
            f"{context}: T_gamma={T_gamma_MeV:.6g} MeV lies outside the PRIMAT AC2024 "
            f"tabulation window [{RATE_TABLE_T_MEV_MIN:.6g}, {RATE_TABLE_T_MEV_MAX:.6g}] MeV. "
            f"Silent boundary clamping is not allowed for publishable runs.",
            strict=strict,
        )


def validate_manual_network_truncation(n_reactions: int, *, context: str, strict: bool = False) -> None:
    if int(n_reactions) <= 0:
        raise ValueError(f"{context}: n_reactions must be positive, got {n_reactions}.")
    if int(n_reactions) not in (12, 31):
        _emit(
            f"{context}: n_reactions={n_reactions} is a manual network truncation outside the "
            f"documented backbone/full choices (12 or 31). Boundary and convergence status are not locked.",
            strict=strict,
            category=UserWarning,
        )


def validate_min_resolution(name: str, value: int, *, minimum: int, strict: bool = True) -> None:
    if int(value) < int(minimum):
        _emit(
            f"{name}={value} violates the minimum supported resolution {minimum}.",
            strict=strict,
        )


def validate_typeI_initial_budget(Sigma_plus: float, Sigma_minus: float, *, context: str) -> None:
    sigma_sq = float(Sigma_plus) ** 2 + float(Sigma_minus) ** 2
    if sigma_sq >= 1.0:
        raise ValueError(
            f"{context}: Sigma_+^2 + Sigma_-^2 = {sigma_sq:.6g} >= 1, giving non-positive Omega in Type I."
        )


def validate_general_initial_budget(Sigma_sq: float, K: float, cA_sq: float, *, context: str) -> None:
    omega = 1.0 - float(Sigma_sq) - float(K) - float(cA_sq)
    if omega <= 0.0:
        raise ValueError(
            f"{context}: initial Omega={omega:.6g} <= 0 from 1 - Sigma^2 - K - cA^2."
        )

def enforce_positive_typeI_omega(Sigma_sq: float, *, context: str, strict: bool = True, floor: float | None = None) -> float:
    omega = 1.0 - float(Sigma_sq)
    if omega <= 0.0:
        _emit(
            f"{context}: Omega={omega:.6g} <= 0 from 1 - Sigma^2. "
            f"Silent boundary clamping is not allowed for publishable runs.",
            strict=strict,
        )
    return float(omega)


def enforce_positive_general_omega(omega: float, *, context: str, strict: bool = True, floor: float | None = None) -> float:
    omega = float(omega)
    if omega <= 0.0:
        _emit(
            f"{context}: Omega={omega:.6g} <= 0. Silent boundary clamping is not allowed for publishable runs.",
            strict=strict,
        )
    return omega


def warn_underresolved_teff(N_q: int, *, floor: int = 20, context: str) -> None:
    if int(N_q) < int(floor):
        warnings.warn(
            f"{context}: enable_teff=True with N_q={N_q} is below the current conservative observable-level "
            f"floor N_q={floor}. Kernel parity may still hold, but full-BBN convergence is not locked.",
            UserWarning,
            stacklevel=2,
        )


def warn_reduced_ell(name: str, n_ell: int, *, expected: int = 2) -> None:
    if int(n_ell) != int(expected):
        warnings.warn(
            f"{name}: reduced driver is documented at n_ell={expected}; received n_ell={n_ell}. "
            f"Treat this as a manual truncation/extension choice rather than a validated default.",
            UserWarning,
            stacklevel=2,
        )
