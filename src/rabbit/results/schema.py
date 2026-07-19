"""
rabbit.results.schema — Canonical result container.

CanonicalResult is the single return type for all rabbit drivers.
It carries the fidelity label, grid specification, primary observables,
and optional diagnostic containers so that downstream analysis never
needs to guess what approximations were used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np

from rabbit.config.fidelity import FidelityLevel
from rabbit.config.grids import MomentumGrid, MultipoleSpec
from rabbit.config.solver_config import SolverConfig
from rabbit.config.conventions import BianchiType
from rabbit.results.diagnostics import (
    TransportDiag,
    WeakDiag,
    ConstraintDiag,
    SensitivityInfo,
)


@dataclass
class Observables:
    """Primary BBN observables.

    Attributes
    ----------
    Yp : float
        Primordial ⁴He mass fraction.
    DH : float
        Deuterium-to-hydrogen ratio D/H.
    N_eff : float
        Effective number of neutrino species at decoupling.
    Xn_freeze : float
        Neutron fraction at weak freeze-out.
    Li7H : float
        ⁷Li/H ratio (⁷Li + ⁷Be after decay).
    Li6H : float
        ⁶Li/H ratio. Only populated when n_reactions ≥ 31 (standard network).
    N_eff_input : Optional[float]
        The parametric N_eff that seeds ρ_ν in the Hubble rate (``config.N_eff``).
        Exposed so the gap between the reported ``N_eff`` and the value driving
        the expansion is inspectable rather than hidden (BD598 A-R1). At Tier 1
        this (≈3.044) is the SM-comparable value; the Tier-1 ``N_eff`` field is a
        convention-inconsistent entropy-ratio diagnostic (baseline 3.0 × the
        parametric T_ν), NOT the SM N_eff — see
        docs/audit/BD598_PR-B2_neff_convention_resolution_2026-06-28.md.
    N_eff_is_derived : bool
        True iff ``N_eff`` was DERIVED from the evolved neutrino sector
        (Tier ≥ 2 ``N_eff_from_3T`` or a decoupling backbone). False at Tier 1,
        where ``N_eff`` is the entropy-ratio DIAGNOSTIC of the parametric T_ν
        (SPECIFIED, not derived). Any derived-N_eff claim must use Tier ≥ 2.
    """
    Yp: float = 0.0
    DH: float = 0.0
    N_eff: float = 0.0
    Xn_freeze: float = 0.0
    Li7H: float = 0.0
    Li6H: float = 0.0
    N_eff_input: Optional[float] = None
    N_eff_is_derived: bool = False


@dataclass
class RunParameters:
    """Input parameters for a canonical run.

    Attributes
    ----------
    Sigma_H_plus : float
        Initial Hubble-normalized shear Σ₊.
    Sigma_H_minus : float
        Initial Hubble-normalized shear Σ₋.
    eta : float
        Baryon-to-photon ratio.
    tau_n : float
        Neutron lifetime [s].
    T_start : float
        Integration start temperature [MeV].
    T_end : float
        Integration end temperature [MeV].
    bianchi_type : BianchiType
        Bianchi classification.
    """
    Sigma_H_plus: float = 0.0
    Sigma_H_minus: float = 0.0
    eta: float = 6.104e-10
    tau_n: float = 878.4
    T_start: float = 10.0
    T_end: float = 0.005
    bianchi_type: BianchiType = BianchiType.TYPE_I
    N_eff: float = 3.044
    f_nu: float = 0.0
    N_q: int = 20


@dataclass
class CanonicalResult:
    """Canonical result container for rabbit runs.

    Every driver must return a CanonicalResult.  The fidelity_level
    field is mandatory and determined by the least-faithful component
    in the pipeline.

    Attributes
    ----------
    fidelity_level : FidelityLevel
        Overall computation fidelity (min of all component levels).
    grid : MomentumGrid
        Momentum grid used.
    multipole : MultipoleSpec
        Multipole specification used.
    solver : SolverConfig
        Solver configuration used.
    params : RunParameters
        Input parameters.
    observables : Observables
        Primary BBN outputs.
    transport_diag : TransportDiag or None
        Transport diagnostics (if recorded).
    weak_diag : WeakDiag or None
        Weak rate diagnostics (if recorded).
    constraint_diag : ConstraintDiag or None
        Constraint preservation diagnostics (if recorded).
    sensitivity : SensitivityInfo or None
        Input-parameter sensitivity metadata (if computed).
        Populated by the caller, not by the driver itself.
    trajectory : dict or None
        Optional: full ODE trajectory {N: y(N)} for post-processing.
    wall_time_s : float
        Wall-clock time in seconds.
    metadata : dict
        Arbitrary additional metadata.
    """
    fidelity_level: FidelityLevel
    grid: MomentumGrid
    multipole: MultipoleSpec
    solver: SolverConfig
    params: RunParameters = field(default_factory=RunParameters)
    observables: Observables = field(default_factory=Observables)
    transport_diag: Optional[TransportDiag] = None
    weak_diag: Optional[WeakDiag] = None
    constraint_diag: Optional[ConstraintDiag] = None
    sensitivity: Optional[SensitivityInfo] = None
    trajectory: Optional[Dict[str, np.ndarray]] = None
    wall_time_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_publishable(self) -> bool:
        """Whether this result meets the minimum fidelity for publication."""
        return self.fidelity_level.is_publishable

    @property
    def fidelity_label(self) -> str:
        """Short label for filenames and tables."""
        return self.fidelity_level.short_label

    def summary(self) -> str:
        """One-line summary for logging."""
        obs = self.observables
        p = self.params
        return (
            f"[{self.fidelity_label}] "
            f"Σ₊={p.Sigma_H_plus:.4f} "
            f"Yp={obs.Yp:.6f} D/H={obs.DH:.3e} "
            f"Neff={obs.N_eff:.4f} "
            f"({self.wall_time_s:.1f}s)"
        )

    def sensitivity_summary(self) -> str:
        """Multi-line sensitivity summary for documentation."""
        if self.sensitivity is None:
            return "  (no sensitivity metadata computed)"
        s = self.sensitivity
        lines = [
            f"  dYp/dτ_n = {s.dYp_dtaun:+.2e} per s",
            f"  dYp/dη   = {s.dYp_deta:+.2e} per unit η",
            f"  d(D/H)/dη = {s.dDH_deta:+.2e} per unit η",
            f"  N_eff(measured) = {s.Neff_measured:.4f}",
            f"  N_eff gap = {s.Neff_gap:.4f}",
        ]
        return "\n".join(lines)
