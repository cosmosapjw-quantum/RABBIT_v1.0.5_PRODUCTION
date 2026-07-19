"""
rabbit.transport.state — Momentum-resolved hierarchy state container.

Stores the PSTF multipole perturbations Ψ_{s,ℓ}(q_i) for each neutrino
species s, multipole ℓ ∈ {0, 2}, and comoving momentum bin q_i.

Convention:
    Ψ_{s,ℓ}(q) is the RELATIVE perturbation to the equilibrium
    Fermi–Dirac distribution.  The full distribution function is:

        f_s(q, θ) = f₀(q) [1 + Ψ_{s,0}(q) + Ψ_{s,2}(q) P₂(cos θ)]

    where f₀(q) = 1/(e^q + 1).

    Initially (equilibrium): Ψ_{s,0} = 0, Ψ_{s,2} = 0 for all s, q.

    Monopole perturbations arise from collisions (incomplete decoupling).
    Quadrupole perturbations arise from shear coupling and collisions.

Flat layout for ODE solver:
    [Ψ_{0,0}(q₁),...,Ψ_{0,0}(q_N), Ψ_{0,2}(q₁),...,Ψ_{0,2}(q_N),
     Ψ_{1,0}(q₁),...,Ψ_{1,0}(q_N), Ψ_{1,2}(q₁),...,Ψ_{1,2}(q_N),
     ...for each species...]

    Total: n_species × n_ell × N_q.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rabbit.config.grids import MomentumGrid, MultipoleSpec
from rabbit.config.conventions import (
    N_SPECIES_TRANSPORT, DEFAULT_SPECIES, NeutrinoSpecies,
)


# ═══════════════════════════════════════════════════════════════════════
# §1. Equilibrium distribution
# ═══════════════════════════════════════════════════════════════════════

def fermi_dirac(q: np.ndarray) -> np.ndarray:
    """Equilibrium Fermi–Dirac distribution f₀(q) = 1/(e^q + 1).

    In comoving momentum q, this is time-independent for free-streaming
    species (the expansion redshift is absorbed into the coordinate).
    """
    return 1.0 / (np.exp(q) + 1.0)


# ═══════════════════════════════════════════════════════════════════════
# §2. HierarchyState
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HierarchyState:
    """Structured view of the momentum-resolved neutrino hierarchy.

    Parameters
    ----------
    data : ndarray, shape (n_species, n_ell, N_q)
        Perturbation coefficients Ψ_{s,ℓ}(q_i).
        data[s, 0, :] = monopole Ψ_{s,0}(q_i)
        data[s, 1, :] = quadrupole Ψ_{s,2}(q_i)
    grid : MomentumGrid
        Gauss–Laguerre momentum grid.
    multipole : MultipoleSpec
        Active multipole configuration.
    """
    data: np.ndarray       # shape (n_species, n_ell, N_q)
    grid: MomentumGrid
    multipole: MultipoleSpec

    @property
    def n_species(self) -> int:
        return self.data.shape[0]

    @property
    def n_ell(self) -> int:
        return self.data.shape[1]

    @property
    def N_q(self) -> int:
        return self.data.shape[2]

    @property
    def n_dof(self) -> int:
        """Total degrees of freedom (= size of flat vector)."""
        return self.data.size

    # ── Per-species accessors ──────────────────────────────────────────

    def monopole(self, species_idx: int) -> np.ndarray:
        """Monopole perturbation Ψ_{s,0}(q) for species s."""
        return self.data[species_idx, 0, :]

    def quadrupole(self, species_idx: int) -> np.ndarray:
        """Quadrupole⁺ perturbation Ψ_{s,2⁺}(q) for species s."""
        return self.data[species_idx, 1, :]

    def quadrupole_minus(self, species_idx: int) -> np.ndarray:
        """Quadrupole⁻ perturbation Ψ_{s,2⁻}(q) for species s.

        Only available when n_ell >= 3 (generic Type I).
        Returns zeros if n_ell == 2 (LRS mode).
        """
        if self.n_ell >= 3:
            return self.data[species_idx, 2, :]
        return np.zeros(self.N_q)

    # ── Pack / unpack ──────────────────────────────────────────────────

    def to_flat(self) -> np.ndarray:
        """Pack into a 1D array for the ODE solver.

        Layout: species-major, ℓ-middle, q-minor.
        """
        return self.data.ravel()

    @classmethod
    def from_flat(
        cls,
        flat: np.ndarray,
        grid: MomentumGrid,
        multipole: MultipoleSpec,
        n_species: int = N_SPECIES_TRANSPORT,
    ) -> HierarchyState:
        """Unpack from a flat 1D array.

        Parameters
        ----------
        flat : ndarray, shape (n_species * n_ell * N_q,)
        grid : MomentumGrid
        multipole : MultipoleSpec
        n_species : int
        """
        n_ell = multipole.n_ell
        expected = n_species * n_ell * grid.N_q
        if flat.size != expected:
            raise ValueError(
                f"Flat array size {flat.size} != expected "
                f"{n_species}×{n_ell}×{grid.N_q} = {expected}"
            )
        data = flat.reshape(n_species, n_ell, grid.N_q)
        return cls(data=data, grid=grid, multipole=multipole)

    # ── Factory methods ────────────────────────────────────────────────

    @classmethod
    def from_isotropic(
        cls,
        grid: MomentumGrid,
        multipole: MultipoleSpec,
        n_species: int = N_SPECIES_TRANSPORT,
    ) -> HierarchyState:
        """Create equilibrium (isotropic) initial state.

        All perturbations are zero: Ψ_{s,0} = 0, Ψ_{s,2} = 0.
        The full distribution is f₀(q) everywhere.
        """
        n_ell = multipole.n_ell
        data = np.zeros((n_species, n_ell, grid.N_q))
        return cls(data=data, grid=grid, multipole=multipole)

    # ── Roundtrip verification ─────────────────────────────────────────

    def roundtrip_check(self) -> bool:
        """Verify pack → unpack roundtrip is bit-identical."""
        flat = self.to_flat()
        restored = HierarchyState.from_flat(
            flat, self.grid, self.multipole, self.n_species
        )
        return np.array_equal(self.data, restored.data)
