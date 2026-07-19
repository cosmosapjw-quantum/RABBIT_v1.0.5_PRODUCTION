"""
rabbit.results.diagnostics — Diagnostic containers for pipeline monitoring.

Each diagnostic container records per-step or summary information for a
subsystem, enabling post-hoc analysis of convergence, constraint
preservation, and signal decomposition without polluting the main result
schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TransportDiag:
    """Diagnostics from the neutrino transport hierarchy.

    Attributes
    ----------
    N_values : ndarray, shape (n_steps,)
        E-fold values at which diagnostics were recorded.
    pi_total : ndarray, shape (n_steps,)
        Total anisotropic stress π(N) summed over all species.
    pi_per_species : ndarray, shape (n_steps, n_species)
        Per-species anisotropic stress.
    max_quadrupole : ndarray, shape (n_steps,)
        max_q |F_{s,2}(q)| at each step (monitors hierarchy amplitude).
    misner_viscosity_check : ndarray, shape (n_steps,)
        Ratio |π + 2 η_ν σ| / |π|; should be ≪ 1 in the linear regime.
    N_q_used : int
        Number of momentum grid points used.
    ell_max_used : int
        Maximum ℓ used (2 for Type I = exact).
    """
    N_values: np.ndarray = field(default_factory=lambda: np.array([]))
    pi_total: np.ndarray = field(default_factory=lambda: np.array([]))
    pi_per_species: Optional[np.ndarray] = None
    max_quadrupole: np.ndarray = field(default_factory=lambda: np.array([]))
    misner_viscosity_check: np.ndarray = field(default_factory=lambda: np.array([]))
    N_q_used: int = 80
    ell_max_used: int = 2


@dataclass
class WeakDiag:
    """Diagnostics from the live weak rate computation.

    Attributes
    ----------
    N_values : ndarray, shape (n_steps,)
        E-fold values at which diagnostics were recorded.
    lambda_np : ndarray, shape (n_steps,)
        n → p rate [s⁻¹] at each step.
    lambda_pn : ndarray, shape (n_steps,)
        p → n rate [s⁻¹] at each step.
    iso_part : ndarray, shape (n_steps,)
        Isotropic component of the rate (from monopole evaluated at
        equilibrium FD).
    aniso_excess : ndarray, shape (n_steps,)
        Fractional excess due to anisotropy: (λ_live − λ_iso) / λ_iso.
    correction_level : str
        Which correction tier was applied ('born', 'coulomb', etc.).
    """
    N_values: np.ndarray = field(default_factory=lambda: np.array([]))
    lambda_np: np.ndarray = field(default_factory=lambda: np.array([]))
    lambda_pn: np.ndarray = field(default_factory=lambda: np.array([]))
    iso_part: np.ndarray = field(default_factory=lambda: np.array([]))
    aniso_excess: np.ndarray = field(default_factory=lambda: np.array([]))
    correction_level: str = "born"


@dataclass
class ConstraintDiag:
    """Diagnostics for constraint preservation.

    Attributes
    ----------
    N_values : ndarray, shape (n_steps,)
        E-fold values.
    friedmann_residual : ndarray, shape (n_steps,)
        |1 − Ω − Σ²| (should be < 1e-6 for accepted runs).
    mass_conservation : ndarray, shape (n_steps,)
        |Σ_i X_i − 1| (should be < 1e-12 for mass-fraction variables).
    energy_conservation : ndarray, shape (n_steps,)
        Fractional energy non-conservation from collision operator
        (should be < 1e-8).
    """
    N_values: np.ndarray = field(default_factory=lambda: np.array([]))
    friedmann_residual: np.ndarray = field(default_factory=lambda: np.array([]))
    mass_conservation: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_conservation: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def max_friedmann_violation(self) -> float:
        """Worst-case Friedmann constraint violation."""
        if len(self.friedmann_residual) == 0:
            return 0.0
        return float(np.max(np.abs(self.friedmann_residual)))

    @property
    def max_mass_violation(self) -> float:
        """Worst-case mass conservation violation."""
        if len(self.mass_conservation) == 0:
            return 0.0
        return float(np.max(np.abs(self.mass_conservation)))


@dataclass
class SensitivityInfo:
    """Input-parameter sensitivity metadata for a canonical run.

    Computed externally by running the driver at perturbed input values
    and stored here for downstream uncertainty analysis.  The driver
    itself does NOT compute these — they are populated by the caller
    via ``result.sensitivity = compute_sensitivity(...)``.

    Sign conventions (standard BBN):
        dYp_dtaun > 0 : longer neutron lifetime → more ⁴He
        dYp_deta  > 0 : higher baryon density  → slightly more ⁴He
        dDH_deta  < 0 : higher baryon density  → less deuterium
        Neff_measured  : actual N_eff at low T (may differ from input)

    Attributes
    ----------
    dYp_dtaun : float
        ∂Y_p/∂τ_n [per second].  Typical: ~5 × 10⁻⁴ s⁻¹.
    dYp_deta : float
        ∂Y_p/∂η [per unit η].  Typical: ~10⁷.
    dDH_deta : float
        ∂(D/H)/∂η [per unit η].  Typical: ~−3 × 10⁴.
    Neff_measured : float
        N_eff measured at T ≪ m_e from T_ν/T_γ ratio.
    Neff_gap : float
        |N_eff(input) − N_eff(measured)|.
    tau_n_values : ndarray, shape (n_points,)
        τ_n values used for the sensitivity scan.
    Yp_at_tau_n : ndarray, shape (n_points,)
        Y_p values at each τ_n.
    eta_values : ndarray, shape (n_points,)
        η values used for the sensitivity scan.
    DH_at_eta : ndarray, shape (n_points,)
        D/H values at each η.
    """
    dYp_dtaun: float = 0.0
    dYp_deta: float = 0.0
    dDH_deta: float = 0.0
    Neff_measured: float = 0.0
    Neff_gap: float = 0.0
    tau_n_values: np.ndarray = field(default_factory=lambda: np.array([]))
    Yp_at_tau_n: np.ndarray = field(default_factory=lambda: np.array([]))
    eta_values: np.ndarray = field(default_factory=lambda: np.array([]))
    DH_at_eta: np.ndarray = field(default_factory=lambda: np.array([]))
