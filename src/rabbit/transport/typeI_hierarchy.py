"""
rabbit.transport.typeI_hierarchy — Type I PSTF hierarchy RHS.

For Type I (K = 0, ν_m = 0): ℓ_max = 2 is EXACT.  The free-streaming
coupling κ_ℓ^m vanishes identically, so ℓ = 2 → ℓ = 3 cascade is absent.
No closure approximation is needed.

Evolution equations (collisionless):
    dΨ_{s,0}(q)/dN = 0              (monopole: no source, no collisions)
    dΨ_{s,2}(q)/dN = −(8/15) Σ₊    (quadrupole: shear source only)

The shear source is q-INDEPENDENT at Born level (angular orthogonality
of the Boltzmann streaming term in a homogeneous spacetime).  This means
the quadrupole perturbation Ψ₂ grows uniformly across all momentum bins
in the collisionless limit.

When collisions are added (Block 3), the RHS gains q-dependent collision
terms C_{s,ℓ}[Ψ](q), breaking the momentum uniformity.  The q-independent
shear source itself is unchanged.
"""

from __future__ import annotations

import numpy as np

from rabbit.transport.state import HierarchyState
from rabbit.config.grids import MomentumGrid, MultipoleSpec


# ═══════════════════════════════════════════════════════════════════════
# §1. Physical constants
# ═══════════════════════════════════════════════════════════════════════

#: Quadrupole source coefficient B = 8/15 from the angular
#: decomposition of the collisionless Boltzmann equation.
B_QUADRUPOLE_SOURCE: float = 8.0 / 15.0


# ═══════════════════════════════════════════════════════════════════════
# §2. Hierarchy RHS
# ═══════════════════════════════════════════════════════════════════════

def compute_hierarchy_rhs_typeI(
    state: HierarchyState,
    Sigma_plus: float,
    Sigma_minus: float = 0.0,
) -> np.ndarray:
    """Compute dΨ/dN for the Type I momentum-resolved hierarchy.

    For the collisionless case (LRS, n_ell=2):
        dΨ_{s,0}(q_i)/dN = 0           for all s, q_i
        dΨ_{s,2⁺}(q_i)/dN = −B Σ₊     for all s, q_i (q-independent)

    For generic Type I (n_ell=3):
        dΨ_{s,0}(q_i)/dN  = 0
        dΨ_{s,2⁺}(q_i)/dN = −B Σ₊     (couples to + polarization)
        dΨ_{s,2⁻}(q_i)/dN = −B Σ₋     (couples to − polarization)

    The NEGATIVE sign is the restoring force that creates the damped
    σ–π oscillations.

    Parameters
    ----------
    state : HierarchyState
        Current hierarchy state (Ψ values).
    Sigma_plus : float
        Hubble-normalized shear Σ₊ (SIGNED).
    Sigma_minus : float
        Hubble-normalized shear Σ₋.  For LRS: 0.

    Returns
    -------
    ndarray, shape (n_species × n_ell × N_q,)
        Flat derivative vector dΨ/dN, same layout as state.to_flat().
    """
    n_species = state.n_species
    n_ell = state.n_ell
    N_q = state.N_q

    drhs = np.zeros((n_species, n_ell, N_q))

    # Monopole (ℓ=0): d/dN = 0 (collisionless)
    # drhs[:, 0, :] = 0.0  (already zero)

    # Quadrupole⁺ (ℓ=2, + polarization): d/dN = −B × Σ₊
    drhs[:, 1, :] = -B_QUADRUPOLE_SOURCE * Sigma_plus

    # Quadrupole⁻ (ℓ=2, − polarization): d/dN = −B × Σ₋
    # Only present when n_ell >= 3 (generic Type I)
    if n_ell >= 3:
        drhs[:, 2, :] = -B_QUADRUPOLE_SOURCE * Sigma_minus

    return drhs.ravel()
