"""
rabbit.transport.stageAB_state — Axisymmetric Stage A/B hierarchy state.

This module introduces a new q-resolved moment state for the Stage A/B
ablation program.  It is intentionally separate from the legacy Type-I
perturbation state because the new formalism evolves full Legendre/PSTF
moments of the distribution itself rather than relative perturbations.

Conventions
-----------
For axisymmetry we expand the full distribution as

    f(q, mu) = sum_{ell in active_ells} F_ell(q) P_ell(mu),

with active_ells = (0, 2) for Stage A and (0, 2, 4) for Stage B.

FLRW exact-FD manifold:
    F_0(q) = f_FD(q),
    F_{ell>=2}(q) = 0.

This manifold must remain invariant when Sigma_H = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from rabbit.config.conventions import N_SPECIES_TRANSPORT
from rabbit.config.grids import MomentumGrid


def fermi_dirac(q: np.ndarray) -> np.ndarray:
    """Equilibrium Fermi–Dirac distribution in comoving momentum."""
    return 1.0 / (np.exp(q) + 1.0)


@dataclass
class AxisymmetricHierarchyState:
    """Structured q-resolved axisymmetric hierarchy state.

    Parameters
    ----------
    data
        ndarray with shape (n_species, n_ell, N_q).
    grid
        Momentum grid.
    active_ells
        Tuple of active even multipoles.  Stage A uses (0, 2),
        Stage B uses (0, 2, 4).
    """

    data: np.ndarray
    grid: MomentumGrid
    active_ells: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.active_ells[0] != 0:
            raise ValueError("active_ells must start with ell=0")
        if tuple(sorted(self.active_ells)) != tuple(self.active_ells):
            raise ValueError("active_ells must be sorted")
        if any((ell < 0) or (ell % 2 != 0) for ell in self.active_ells):
            raise ValueError("Only non-negative even ell values are supported")
        expected_shape = (self.n_species, self.n_ell, self.grid.N_q)
        if self.data.shape != expected_shape:
            raise ValueError(
                f"data shape {self.data.shape} != expected {expected_shape}"
            )

    @property
    def n_species(self) -> int:
        return self.data.shape[0]

    @property
    def n_ell(self) -> int:
        return self.data.shape[1]

    @property
    def N_q(self) -> int:
        return self.data.shape[2]

    def ell_index(self, ell: int) -> int:
        try:
            return self.active_ells.index(ell)
        except ValueError as exc:
            raise KeyError(f"ell={ell} is not active in {self.active_ells}") from exc

    def moment(self, species_idx: int, ell: int) -> np.ndarray:
        return self.data[species_idx, self.ell_index(ell), :]

    def to_flat(self) -> np.ndarray:
        return self.data.ravel()

    @classmethod
    def from_flat(
        cls,
        flat: np.ndarray,
        grid: MomentumGrid,
        active_ells: Tuple[int, ...],
        n_species: int = N_SPECIES_TRANSPORT,
    ) -> "AxisymmetricHierarchyState":
        expected = n_species * len(active_ells) * grid.N_q
        if flat.size != expected:
            raise ValueError(
                f"flat size {flat.size} != expected {expected} for "
                f"n_species={n_species}, active_ells={active_ells}, N_q={grid.N_q}"
            )
        data = flat.reshape(n_species, len(active_ells), grid.N_q)
        return cls(data=data, grid=grid, active_ells=active_ells)

    @classmethod
    def from_fd_equilibrium(
        cls,
        grid: MomentumGrid,
        active_ells: Tuple[int, ...],
        n_species: int = N_SPECIES_TRANSPORT,
    ) -> "AxisymmetricHierarchyState":
        data = np.zeros((n_species, len(active_ells), grid.N_q), dtype=float)
        data[:, 0, :] = fermi_dirac(grid.nodes)
        return cls(data=data, grid=grid, active_ells=active_ells)

    @classmethod
    def zeros(
        cls,
        grid: MomentumGrid,
        active_ells: Tuple[int, ...],
        n_species: int = N_SPECIES_TRANSPORT,
    ) -> "AxisymmetricHierarchyState":
        data = np.zeros((n_species, len(active_ells), grid.N_q), dtype=float)
        return cls(data=data, grid=grid, active_ells=active_ells)
