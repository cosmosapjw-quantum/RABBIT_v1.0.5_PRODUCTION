"""
rabbit.transport.typeI_stageA_hierarchy — Stage A ordered quadrupole closure.

Stage A is the leading-order axisymmetric Bianchi-I hierarchy in the new
formalism.  It evolves the full q-resolved moments (F_0, F_2) of the
distribution itself.  This is *not* an exact finite-ell truncation; it is
an ordered closure with

    F_2 = O(epsilon),
    delta F_0 = O(epsilon^2),
    epsilon ~ Sigma_H.

Retained collisionless terms:
    dF_0/dN = +(Sigma_H/5) * q dF_2/dq
    dF_2/dN = + Sigma_H * q dF_0/dq

Higher-order self-couplings proportional to Sigma_H * F_2 are omitted at
this stage by construction.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from rabbit.config.conventions import N_SPECIES_TRANSPORT
from rabbit.config.grids import MomentumGrid
from rabbit.transport.stageAB_state import AxisymmetricHierarchyState

STAGE_A_ELLS: Tuple[int, ...] = (0, 2)


def q_d_dq(values: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return q * d(values)/dq on a nonuniform q grid."""
    edge_order = 2 if q.size >= 3 else 1
    return q * np.gradient(values, q, edge_order=edge_order)


def compute_hierarchy_rhs_typeI_stageA(
    state: AxisymmetricHierarchyState,
    Sigma_H: float,
    collision_terms: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Stage-A RHS in N = ln a.

    Parameters
    ----------
    state
        Axisymmetric state with active ell values (0, 2).
    Sigma_H
        Hubble-normalized shear amplitude.
    collision_terms
        Optional array with same shape as ``state.data``.
    """
    if tuple(state.active_ells) != STAGE_A_ELLS:
        raise ValueError(
            f"Stage A requires active_ells={STAGE_A_ELLS}, got {state.active_ells}"
        )
    q = state.grid.nodes
    rhs = np.zeros_like(state.data)
    for s in range(state.n_species):
        F0 = state.moment(s, 0)
        F2 = state.moment(s, 2)
        rhs[s, 0, :] = (Sigma_H / 5.0) * q_d_dq(F2, q)
        rhs[s, 1, :] = Sigma_H * q_d_dq(F0, q)
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
        grid=grid, active_ells=STAGE_A_ELLS, n_species=n_species
    )
