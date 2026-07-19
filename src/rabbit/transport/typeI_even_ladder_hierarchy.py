from __future__ import annotations

from typing import Tuple

import numpy as np

from rabbit.config.conventions import N_SPECIES_TRANSPORT
from rabbit.config.grids import MomentumGrid
from rabbit.transport.stageAB_state import AxisymmetricHierarchyState
from rabbit.transport.typeI_stageA_hierarchy import q_d_dq


def active_even_ells(lmax: int) -> Tuple[int, ...]:
    if lmax < 0 or lmax % 2 != 0:
        raise ValueError(f"lmax must be a non-negative even integer, got {lmax}")
    return tuple(range(0, lmax + 2, 2))


def A_coeff(ell: int) -> float:
    return 3.0 * ell * (ell - 1.0) / (2.0 * (2.0 * ell - 3.0) * (2.0 * ell - 1.0))


def B_coeff(ell: int) -> float:
    return ell * (ell + 1.0) / ((2.0 * ell - 1.0) * (2.0 * ell + 3.0))


def C_coeff(ell: int) -> float:
    return 3.0 * (ell + 1.0) * (ell + 2.0) / (2.0 * (2.0 * ell + 3.0) * (2.0 * ell + 5.0))


def Atilde_coeff(ell: int) -> float:
    return ell * (ell - 1.0) * (ell - 2.0) / ((2.0 * ell - 3.0) * (2.0 * ell - 1.0))


def Btilde_coeff(ell: int) -> float:
    return B_coeff(ell)


def Ctilde_coeff(ell: int) -> float:
    return (ell + 1.0) * (ell + 2.0) * (ell + 3.0) / ((2.0 * ell + 3.0) * (2.0 * ell + 5.0))


def compute_hierarchy_rhs_typeI_even_ladder(
    state: AxisymmetricHierarchyState,
    Sigma_H: float,
    collision_terms: np.ndarray | None = None,
) -> np.ndarray:
    """Compute finite even-ell ladder RHS with F_{lmax+2}=0 closure.

    This is the finite truncation family used to compare lmax = 2,4,6,8.
    It is separate from the ordered Stage-A quadrupole closure.
    """
    active = tuple(state.active_ells)
    if active[0] != 0 or any(ell % 2 != 0 for ell in active):
        raise ValueError(f"even ladder requires active even ells starting at 0, got {active}")

    q = state.grid.nodes
    rhs = np.zeros_like(state.data)
    has_f2 = 2 in active
    for s in range(state.n_species):
        if has_f2:
            rhs[s, state.ell_index(0), :] = (Sigma_H / 5.0) * q_d_dq(state.moment(s, 2), q)
        for ell in active[1:]:
            comb = np.zeros_like(q)
            angle = np.zeros_like(q)
            if ell - 2 in active:
                Fm = state.moment(s, ell - 2)
                comb = comb + A_coeff(ell) * Fm
                angle = angle - Atilde_coeff(ell) * Fm
            Fell = state.moment(s, ell)
            comb = comb + B_coeff(ell) * Fell
            angle = angle + Btilde_coeff(ell) * Fell
            if ell + 2 in active:
                Fp = state.moment(s, ell + 2)
                comb = comb + C_coeff(ell) * Fp
                angle = angle + Ctilde_coeff(ell) * Fp
            rhs[s, state.ell_index(ell), :] = Sigma_H * q_d_dq(comb, q) + 1.5 * Sigma_H * angle
    if collision_terms is not None:
        if collision_terms.shape != rhs.shape:
            raise ValueError(f"collision_terms shape {collision_terms.shape} != expected {rhs.shape}")
        rhs = rhs + collision_terms
    return rhs.ravel()


def equilibrium_state(
    grid: MomentumGrid,
    lmax: int,
    n_species: int = N_SPECIES_TRANSPORT,
) -> AxisymmetricHierarchyState:
    return AxisymmetricHierarchyState.from_fd_equilibrium(
        grid=grid,
        active_ells=active_even_ells(lmax),
        n_species=n_species,
    )
