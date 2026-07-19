"""Shared weak 2-to-2 total-rate prefactors for thermo/rate helpers."""
from __future__ import annotations

import numpy as np

from rabbit.collisions.kernels import G_F_MEV


WEAK_2TO2_RATE_PREFACTOR_MEV = float((7.0 * np.pi / 12.0) * G_F_MEV**2)


def total_rate_nu_nu_diagonal(T_MeV: float, *, alpha_eq_beta: bool = False) -> float:
    """Leading-order diagonal nu-nu elastic rate in natural units [MeV]."""
    T = float(T_MeV)
    a_alpha_beta = 2.0 if bool(alpha_eq_beta) else 1.0
    return WEAK_2TO2_RATE_PREFACTOR_MEV * T**5 * a_alpha_beta


def gamma_nu_nu_over_H(T_MeV: float, H_MeV: float, *, alpha_eq_beta: bool = False) -> float:
    """Dimensionless diagonal nu-nu equilibration ratio Gamma/H."""
    H = max(float(H_MeV), 1.0e-100)
    return total_rate_nu_nu_diagonal(
        float(T_MeV),
        alpha_eq_beta=bool(alpha_eq_beta),
    ) / H


__all__ = [
    "WEAK_2TO2_RATE_PREFACTOR_MEV",
    "total_rate_nu_nu_diagonal",
    "gamma_nu_nu_over_H",
]
