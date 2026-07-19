# src/rabbit/validation/anisotropic_thermal_prerun.py
"""Anisotropic neutrino-decoupling thermal prerun.

Claim status (BD598 B-R2): the module name is aspirational. Stage A
(isotropic limit) is IMPLEMENTED; the actual anisotropic integration is
SPECIFIED only (Stage B) and currently raises NotImplementedError. This module
integrates NO anisotropy yet — do not read the name as an implemented
anisotropic prerun.

Stage A: isotropic-limit foundation. For zero shear and zero A-monopole
offset this reproduces the FLRW 3T-closure prerun neutrino temperatures
exactly (same engine, same call) and returns an extended payload shaped for
the phase-2 restart state. Nonzero shear or A-offset is deferred to Stage B
(full quadrupole-coupled PSTF integration) and raises NotImplementedError.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from rabbit.thermo.nudec_coupled import N_eff_from_3T, asymptotic_N_eff_3T_payload

_SHEAR_TOL = 1.0e-15


def _finite_float(value: Any, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return out


def anisotropic_thermal_prerun_to_T0(
    *,
    T_start_MeV: float,
    T_end_MeV: float,
    n_species: int,
    q_count: int,
    sigma_plus0: float = 0.0,
    sigma_minus0: float = 0.0,
    initial_A_monopole_offset: float = 0.0,
) -> dict[str, Any]:
    """Integrate the neutrino-decoupling thermal prerun to the AP65 start.

    Stage A handles only the isotropic limit; nonzero shear or A-offset is
    deferred to Stage B.
    """
    T_start = _finite_float(T_start_MeV, name="T_phase1_start_MeV")
    T_end = _finite_float(T_end_MeV, name="T_phase1_end_MeV")
    if T_start <= 0.0 or T_end <= 0.0:
        raise ValueError("prerun temperatures must be positive.")
    if T_start < T_end:
        raise ValueError("T_phase1_start_MeV must be >= T_phase1_end_MeV.")
    n_species_int = int(n_species)
    q_count_int = int(q_count)
    if n_species_int <= 0 or q_count_int <= 0:
        raise ValueError("n_species and q_count must be positive.")

    sigma_plus = _finite_float(sigma_plus0, name="sigma_plus0")
    sigma_minus = _finite_float(sigma_minus0, name="sigma_minus0")
    a_offset = _finite_float(
        initial_A_monopole_offset, name="initial_A_monopole_offset"
    )
    isotropic = (
        abs(sigma_plus) <= _SHEAR_TOL
        and abs(sigma_minus) <= _SHEAR_TOL
        and abs(a_offset) <= _SHEAR_TOL
    )
    if not isotropic:
        raise NotImplementedError(
            "anisotropic (nonzero shear or A-monopole offset) thermal prerun "
            "integration is implemented in Stage B; Stage A covers the "
            "isotropic limit only."
        )

    prerun = asymptotic_N_eff_3T_payload(
        T_start, T_start, T_start, T_stop_MeV=T_end
    )
    if not bool(prerun.get("available")) or not bool(
        prerun.get("tail_reached_stop")
    ):
        raise RuntimeError(
            "anisotropic thermo prerun did not reach T_gamma0: "
            + str(prerun.get("unavailable_reason") or prerun.get("solver_message"))
        )
    Tg = _finite_float(
        prerun["T_gamma_asymptotic_MeV"], name="prerun.T_gamma_at_T0"
    )
    Te = _finite_float(
        prerun["T_nu_e_asymptotic_MeV"], name="prerun.T_nu_e_at_T0"
    )
    Tx = _finite_float(
        prerun["T_nu_x_asymptotic_MeV"], name="prerun.T_nu_x_at_T0"
    )
    A0 = np.zeros((n_species_int, 3, q_count_int), dtype=float)
    return {
        "available": True,
        "policy": "phase1_thermo_prerun_anisotropic",
        "policy_scope": "anisotropic_isotropic_limit",
        "T_phase1_start_MeV": float(T_start),
        "T_phase1_end_MeV": float(T_end),
        "T_gamma0_effective_MeV": float(Tg),
        "T_nu_e0_effective_MeV": float(Te),
        "T_nu_x0_effective_MeV": float(Tx),
        "T_nu_e0_over_T_gamma0": float(Te / Tg),
        "T_nu_x0_over_T_gamma0": float(Tx / Tg),
        "Sigma_plus0_effective": 0.0,
        "Sigma_minus0_effective": 0.0,
        "initial_A_modes": A0.tolist(),
        "N_eff_3T_at_T0": float(N_eff_from_3T(Tg, Te, Tx)),
        "thermo_tail_payload": dict(prerun),
    }
