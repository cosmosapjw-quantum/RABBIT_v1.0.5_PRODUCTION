from __future__ import annotations

import os
import numpy as np

from rabbit.collisions.species import Species, as_species, BANK_DEGENERACY


def gamma_nunu_diagonal(T_nu_e: float, T_nu_x: float, gamma: float | None = None) -> float:
    if gamma is not None:
        return float(gamma)
    pref = float(os.environ.get("RABBIT_NUNU_DIAG_RATE", "0.0"))
    Tbar = 0.5 * (float(T_nu_e) + float(T_nu_x))
    return float(pref * Tbar**5)


def weighted_mean_distribution(f_by_species: dict) -> np.ndarray:
    acc = None
    denom = 0.0
    for sp_key, f in f_by_species.items():
        sp = as_species(sp_key)
        g = BANK_DEGENERACY[sp]
        arr = np.asarray(f, dtype=np.float64)
        if acc is None:
            acc = g * arr
        else:
            acc = acc + g * arr
        denom += g
    return acc / denom


def nunu_diagonal_monopole_collision(
    f_by_species: dict,
    *,
    T_nu_e: float,
    T_nu_x: float,
    gamma: float | None = None,
) -> dict:
    """
    Pointwise energy-conserving diagonal ν-ν redistribution operator.

    Weighted sum over species is exactly zero pointwise if all species use the same Γ.
    """
    G = gamma_nunu_diagonal(T_nu_e, T_nu_x, gamma=gamma)
    fbar = weighted_mean_distribution(f_by_species)

    out = {}
    for sp_key, f in f_by_species.items():
        sp = as_species(sp_key)
        arr = np.asarray(f, dtype=np.float64)
        out[sp.value] = -G * (arr - fbar)
    return out


def energy_conservation_residual(C_by_species: dict, q_nodes, q_weights) -> float:
    q = np.asarray(q_nodes, dtype=np.float64)
    w = np.asarray(q_weights, dtype=np.float64)

    total = np.zeros_like(q, dtype=np.float64)
    for sp_key, C in C_by_species.items():
        sp = as_species(sp_key)
        total = total + BANK_DEGENERACY[sp] * np.asarray(C, dtype=np.float64)

    return float(np.sum(w * q**3 * total))
