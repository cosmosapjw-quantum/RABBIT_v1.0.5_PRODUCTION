"""
rabbit.transport.projectors — Extract physical observables from hierarchy.

Two projectors extract the quantities needed by other sectors:

1. ``extract_aniso_stress`` → Π₊ for geometry feedback.
   Uses the energy-weighted average of the quadrupole perturbation:

       π̃_s = ∫ q³ f₀(q) Ψ_{s,2}(q) dq / ∫ q³ f₀(q) dq

   Geometry feedback: Π₊ = A × f_ν × <π̃>_species  where A = 6.
   In the current reduced Type-I model the species combination is the
   arithmetic mean, which is exact when all transported species share the
   same background normalization and serves as an identical-species
   approximation otherwise.

   In the collisionless limit, Ψ₂ is q-independent, so π̃_s = Ψ₂
   regardless of the weight function.  With collisions, the full
   q-weighted integral becomes essential.

2. ``extract_monopole_distribution`` → f₀(q)(1 + Ψ₀(q)) for weak rates.
   Returns the total monopole distribution function at each momentum
   node, ready for the weak rate integrand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rabbit.transport.state import HierarchyState, fermi_dirac
from rabbit.config.grids import MomentumGrid
from rabbit.config.conventions import NeutrinoSpecies, DEFAULT_SPECIES


# ═══════════════════════════════════════════════════════════════════════
# §1. Physical constants
# ═══════════════════════════════════════════════════════════════════════

#: Geometry feedback coefficient A = 6.
#: Π₊ = A × f_ν × π̃
#: This produces A·B = 6 × 8/15 = 16/5 = 3.2, giving the correct
#: oscillation frequency ω = √(64 f_ν/5 − 1)/2 ≈ 1.023.
A_FEEDBACK: float = 6.0

#: Standard neutrino energy fraction for N_eff = 3.044.
F_NU_STANDARD: float = 0.40520


# ═══════════════════════════════════════════════════════════════════════
# §2. Anisotropic stress extraction
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AnisotropicStressResult:
    """Result of anisotropic stress extraction.

    Attributes
    ----------
    Pi_plus : float
        Hubble-normalized Π₊ for geometry feedback.
    Pi_minus : float
        Hubble-normalized Π₋ (zero for LRS).
    pi_tilde_per_species : ndarray, shape (n_species,)
        Reduced quadrupole π̃_s per species.
    pi_tilde_mean : float
        Species-averaged reduced quadrupole.
    """
    Pi_plus: float
    Pi_minus: float
    pi_tilde_per_species: np.ndarray
    pi_tilde_mean: float


def extract_aniso_stress(
    state: HierarchyState,
    grid: MomentumGrid,
    f_nu: float = F_NU_STANDARD,
) -> AnisotropicStressResult:
    """Extract Hubble-normalized anisotropic stress from hierarchy.

    Computes π̃_s as the energy-weighted average of Ψ_{s,2}(q):

        π̃_s = Σ_i W_i Ψ_{s,2}(q_i) / Σ_i W_i

    where W_i = w_i^{GL} e^{q_i} q_i³ f₀(q_i) is the Gauss–Laguerre
    quadrature weight for the energy density integral ∫ q³ f₀(q) dq.

    For q-independent Ψ₂ (collisionless): π̃_s = Ψ₂ exactly.
    For q-dependent Ψ₂ (with collisions): proper weighted average.

    The geometry feedback is:
        Π₊ = A × f_ν × <π̃>_species

    where A = 6. The species combination is the arithmetic mean,
    which is exact in the identical-species limit used by the reduced
    Type-I model and should be regarded as that approximation otherwise.

    Parameters
    ----------
    state : HierarchyState
    grid : MomentumGrid
    f_nu : float
        Neutrino energy fraction.

    Returns
    -------
    AnisotropicStressResult
    """
    q = grid.nodes
    w = grid.weights
    f0 = fermi_dirac(q)

    # Energy density quadrature weight: w_i e^{q_i} q_i³ f₀(q_i)
    # The e^q factor undoes the Laguerre weight e^{-q} in the nodes.
    energy_weight = w * np.exp(q) * q**3 * f0
    norm = np.sum(energy_weight)

    n_species = state.n_species
    pi_tilde = np.zeros(n_species)
    pi_tilde_minus = np.zeros(n_species)

    for s in range(n_species):
        psi2 = state.quadrupole(s)
        pi_tilde[s] = np.sum(energy_weight * psi2) / norm
        # Minus polarization (generic Type I)
        psi2m = state.quadrupole_minus(s)
        pi_tilde_minus[s] = np.sum(energy_weight * psi2m) / norm

    pi_tilde_mean = np.mean(pi_tilde)
    pi_tilde_minus_mean = np.mean(pi_tilde_minus)

    Pi_plus = A_FEEDBACK * f_nu * pi_tilde_mean
    Pi_minus = A_FEEDBACK * f_nu * pi_tilde_minus_mean

    return AnisotropicStressResult(
        Pi_plus=Pi_plus,
        Pi_minus=Pi_minus,
        pi_tilde_per_species=pi_tilde,
        pi_tilde_mean=pi_tilde_mean,
    )


# ═══════════════════════════════════════════════════════════════════════
# §3. Monopole distribution extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_monopole_distribution(
    state: HierarchyState,
    grid: MomentumGrid,
    species_idx: int = 0,
) -> np.ndarray:
    """Extract the total monopole distribution for weak rates.

    Returns f₀(q_i) × (1 + Ψ_{s,0}(q_i)) at each momentum node.

    In the collisionless limit, Ψ₀ = 0 and this returns f₀(q_i).
    With collisions, the spectral distortion is included.

    Parameters
    ----------
    state : HierarchyState
    grid : MomentumGrid
    species_idx : int
        Which species to extract (0 = ν_e, 1 = ν̄_e, etc.).

    Returns
    -------
    ndarray, shape (N_q,)
        Total monopole f_s(q_i) at each momentum bin.
    """
    f0 = fermi_dirac(grid.nodes)
    psi0 = state.monopole(species_idx)
    return f0 * (1.0 + psi0)


# ═══════════════════════════════════════════════════════════════════════
# §4. Misner viscosity check
# ═══════════════════════════════════════════════════════════════════════

def misner_viscosity_ratio(
    pi_tilde: float,
    Sigma: float,
    f_nu: float = F_NU_STANDARD,
) -> float:
    """Check consistency with Misner neutrino viscosity π = −2 η_ν σ.

    In the linear regime, the anisotropic stress and shear satisfy:
        π̃ = −(B / decay_rate) × Σ = −(8/15) / (1/2) × Σ = −(16/15) Σ

    This ratio should be |π̃ + (16/15)Σ| / |π̃| ≪ 1 at steady state.

    Parameters
    ----------
    pi_tilde : float
        Reduced quadrupole.
    Sigma : float
        Hubble-normalized shear.
    f_nu : float
        Neutrino energy fraction.

    Returns
    -------
    float
        Fractional deviation from Misner viscosity relation.
        Returns 0 if both π̃ and Σ are zero.
    """
    if abs(pi_tilde) < 1e-30 and abs(Sigma) < 1e-30:
        return 0.0
    # Steady-state: π̃ ≈ −(16/15) Σ (from dΠ/dN = 0 condition)
    expected = -(16.0 / 15.0) * Sigma
    if abs(pi_tilde) < 1e-30:
        return float('inf')
    return abs(pi_tilde - expected) / abs(pi_tilde)
