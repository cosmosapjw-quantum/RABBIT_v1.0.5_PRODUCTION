"""
rabbit.transport.typeI_stageB_hierarchy — Stage B ell_max=4 comparison model.

Stage B extends the new axisymmetric Bianchi-I hierarchy to ell = 4.  It is
used as the immediate comparison model for Stage A in the ablation study.
Stage B is still a finite closure: F_6 is set to zero by construction.

Collisionless retained terms:
    dF_0/dN = +(Sigma_H/5) q dF_2/dq
    dF_2/dN = +Sigma_H q d/dq(F_0 + 2/7 F_2 + 2/7 F_4)
               + (3/2) Sigma_H (2/7 F_2 + 20/21 F_4)
    dF_4/dN = +Sigma_H q d/dq(18/35 F_2 + 20/77 F_4)
               + (3/2) Sigma_H (-24/35 F_2 + 20/77 F_4)

with F_6 = 0 closure.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from rabbit.config.conventions import N_SPECIES_TRANSPORT
from rabbit.config.grids import MomentumGrid
from rabbit.transport.stageAB_state import AxisymmetricHierarchyState
from rabbit.transport.typeI_stageA_hierarchy import q_d_dq

STAGE_B_ELLS: Tuple[int, ...] = (0, 2, 4)


def compute_hierarchy_rhs_typeI_stageB(
    state: AxisymmetricHierarchyState,
    Sigma_H: float,
    collision_terms: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Stage-B RHS in N = ln a with F6=0 closure."""
    if tuple(state.active_ells) != STAGE_B_ELLS:
        raise ValueError(
            f"Stage B requires active_ells={STAGE_B_ELLS}, got {state.active_ells}"
        )
    q = state.grid.nodes
    rhs = np.zeros_like(state.data)
    for s in range(state.n_species):
        F0 = state.moment(s, 0)
        F2 = state.moment(s, 2)
        F4 = state.moment(s, 4)

        rhs[s, 0, :] = (Sigma_H / 5.0) * q_d_dq(F2, q)
        rhs[s, 1, :] = (
            Sigma_H * q_d_dq(F0 + (2.0 / 7.0) * F2 + (2.0 / 7.0) * F4, q)
            + 1.5 * Sigma_H * ((2.0 / 7.0) * F2 + (20.0 / 21.0) * F4)
        )
        rhs[s, 2, :] = (
            Sigma_H * q_d_dq((18.0 / 35.0) * F2 + (20.0 / 77.0) * F4, q)
            + 1.5 * Sigma_H * (-(24.0 / 35.0) * F2 + (20.0 / 77.0) * F4)
        )
    if collision_terms is not None:
        if collision_terms.shape != rhs.shape:
            raise ValueError(
                f"collision_terms shape {collision_terms.shape} != expected {rhs.shape}"
            )
        rhs = rhs + collision_terms
    return rhs.ravel()


def equilibrium_state(
    grid: MomentumGrid,
    n_species: int = N_SPECIES_TRANSPORT,
) -> AxisymmetricHierarchyState:
    """Convenience constructor for the FLRW exact-FD manifold."""
    return AxisymmetricHierarchyState.from_fd_equilibrium(
        grid=grid, active_ells=STAGE_B_ELLS, n_species=n_species
    )
