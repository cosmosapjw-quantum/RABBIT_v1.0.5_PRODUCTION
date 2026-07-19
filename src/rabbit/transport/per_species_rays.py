"""
rabbit.transport.per_species_rays — Legacy per-species characteristic helpers.

MATURITY: LEGACY ADAPTER / GUARD

This module is no longer the production wiring point for the SciPy tier-2+
per-species characteristic path. The current production-coupled path lives in:

- ``rabbit.transport.characteristic_species``
- ``rabbit.transport.species_tagged_bridge``
- ``rabbit.drivers.full_coupled_typeI``

This file remains for two narrower purposes:

- legacy helper state for research scripts
- an honesty guard when older shared/species-identical paths are used

In the collisionless limit, all 6 neutrino species share identical
characteristic rays because CP-symmetric free-streaming preserves the
initial equilibrium distribution. For Tier 2+ with collision-coupled
transport, ν_e and ν̄_e develop distinct monopoles due to the asymmetry
in the charged-current weak interaction couplings (a_e/a_x ≈ 4.68).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class PerSpeciesRayState:
    """Ray state for a single neutrino species."""
    I: np.ndarray       # energy-shift integrals, shape (N_mu,)
    J: np.ndarray       # angular Jacobians, shape (N_mu,)
    S: float            # accumulated shear integral
    species: str        # 'nue', 'nuebar', 'nux'
    coupling_g: float   # weak coupling: g_L^2 + g_R^2

    @classmethod
    def from_shared(cls, I, J, S, species='nue'):
        """Create per-species state from shared collisionless rays."""
        g_L_e = 0.5 + 0.2312     # ν_e: g_L = 1/2 + sin²θ_W
        g_R = 0.2312              # g_R = sin²θ_W
        g_L_x = -0.5 + 0.2312    # ν_μ,τ: g_L = -1/2 + sin²θ_W

        couplings = {
            'nue':    g_L_e**2 + g_R**2,   # ≈ 0.588
            'nuebar': g_L_e**2 + g_R**2,   # same (CP for scattering)
            'nux':    g_L_x**2 + g_R**2,   # ≈ 0.126
        }
        return cls(
            I=I.copy(), J=J.copy(), S=float(S),
            species=species,
            coupling_g=couplings.get(species, couplings['nux']),
        )

    def monopole(self, w0, q_gl):
        """Extract per-species monopole f̃₀(q)."""
        alpha = np.exp(2 * self.I)
        f_mono = np.zeros(len(q_gl))
        for k in range(len(q_gl)):
            qa = q_gl[k] * alpha
            f_vals = 1.0 / (np.exp(np.minimum(qa, 500)) + 1)
            f_mono[k] = 0.5 * np.sum(w0 * self.J * f_vals)
        return f_mono

    @property
    def coupling_ratio(self):
        """Coupling ratio relative to ν_e (≈ 4.68 for ν_x)."""
        g_nue = 0.5**2 + 0.2312**2 + 2*0.5*0.2312 + 0.2312**2
        return g_nue / max(self.coupling_g, 1e-30)


def species_identical_guard(thermo_tier: int, *, strict: bool = False):
    """Check whether species-identical approximation is acceptable.

    Parameters
    ----------
    thermo_tier : int
        1 = tier-1 (approximation safe), 2+ = needs per-species.
    strict : bool
        If True, raise instead of warn.

    Returns
    -------
    bool
        True if species-identical is acceptable.
    """
    import warnings
    if thermo_tier >= 2:
        msg = (
            "Tier 2+ thermo uses f_nuebar = f_nue (species-identical "
            "approximation). Per-species characteristic rays are required "
            "for production-grade incomplete decoupling. "
            "The current production path is wired through "
            "rabbit.transport.characteristic_species + "
            "rabbit.transport.species_tagged_bridge + "
            "rabbit.drivers.full_coupled_typeI."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return False
    return True
