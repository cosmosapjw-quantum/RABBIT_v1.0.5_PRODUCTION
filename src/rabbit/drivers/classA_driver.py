"""
rabbit.drivers.classA_driver — Unified Class A Bianchi BBN driver.

MATURITY: EXPERIMENTAL
  Import-fixed, correction_level wired, network aligned to v2.
  NOT regression-locked against FLRW gold table.
  Curved transport uses ℓ_max=2 truncation (approximate for K≠0).

Extends the Type I driver to all six Class A Bianchi types
(I, II, VI₀, VII₀, VIII, IX) using the Wainwright–Hsu formalism.

State vector layout:
    y = [Σ₊, Σ₋, N₁, N₂, N₃, Ψ_flat(...), T_γ, [T_νe, T_νx], X₀..X₇]

    Type I:   N₁=N₂=N₃=0 (constant) → equivalent to full_coupled_typeI.py
    Type II:  N₁ evolves, N₂=N₃=0
    Type VII₀: N₁=0, N₂,N₃ evolve
    Type IX:  all three N_i evolve

TRANSPORT APPROXIMATION NOTE:
    The PSTF hierarchy with ℓ_max=2 is EXACT for Type I (K=0).
    For curved types (K≠0), curvature coupling generates ℓ ≥ 3 terms
    in the Boltzmann hierarchy.  The current driver TRUNCATES at ℓ=2,
    which is an approximation.  The error is O(K × Σ) and is small
    when K ≪ 1 (weak curvature).  For the Collins–Stewart attractor
    of Type II (K = 3/49 ≈ 0.06), this approximation error is ~6%.
    Full curved hierarchy (ℓ_max > 2) is deferred to P5-B.

Usage:
    from rabbit.drivers.classA_driver import ClassAConfig, run_classA_bbn
    config = ClassAConfig(
        bianchi_type=BianchiType.TYPE_II,
        Sigma_H_plus=0.1, N1_init=0.3,
    )
    result = run_classA_bbn(config)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

import numpy as np
from scipy.integrate import solve_ivp

from rabbit.config.conventions import BianchiType
from rabbit.config.grids import (
    MomentumGrid, MultipoleSpec,
    DEFAULT_GRID, DEFAULT_MULTIPOLE, DEFAULT_MULTIPOLE_GENERIC,
)
from rabbit.config.solver_config import SolverConfig, PRODUCTION_RADAU_CONFIG
from rabbit.geometry.general_classA import (
    classA_geometry_rhs, compute_Omega, compute_q,
    gauss_curvature_K, friedmann_residual,
    typeI_initial, typeII_initial, typeVII0_initial, typeIX_initial,
)
from rabbit.geometry.constraints import friedmann_residual_classA
from rabbit.transport.state import HierarchyState, fermi_dirac
from rabbit.transport.typeI_hierarchy import compute_hierarchy_rhs_typeI
from rabbit.transport.projectors import extract_aniso_stress, extract_monopole_distribution
from rabbit.thermo.incomplete_decoupling import (
    dT_gamma_dN_tier1, T_nu_from_T_gamma_tier1, rho_plasma,
)
from rabbit.thermo.eos_photon_electron import _RHO_GAMMA_PREFACTOR
from rabbit.validation.truncation_guards import enforce_positive_general_omega
from rabbit.weak.live_rates import compute_live_weak_rates
from rabbit.network.abundances_standard import (
    N_SPECIES, abundance_rhs_phase1, abundance_rhs_phase2,
    phase1_to_phase2, mass_conservation_residual,
    ATOMIC_MASSES, SPECIES_NAMES, N_REACTIONS_FULL,
)
from rabbit.results.schema import (
    CanonicalResult, RunParameters, Observables, ConstraintDiag,
)


# ═══════════════════════════════════════════════════════════════
# §1. Physical constants (same as full_coupled_typeI.py)
# ═══════════════════════════════════════════════════════════════

_G_N = 6.70883e-45   # MeV⁻²
_MEV_TO_S = 1.519267447e21
_TAU_N = 878.4
_ETA = 6.104e-10
_N_EFF = 3.044
_F_NU = 0.40520


# ═══════════════════════════════════════════════════════════════
# §2. State layout
# ═══════════════════════════════════════════════════════════════
# y = [Σ₊, Σ₋, N₁, N₂, N₃, Ψ_flat(...), T_γ, X₀..X₇]
#      0    1   2   3   4    5..           ...  ...

_I_SP = 0    # Σ₊
_I_SM = 1    # Σ₋
_I_N1 = 2    # N₁
_I_N2 = 3    # N₂
_I_N3 = 4    # N₃
_I_HIER = 5  # hierarchy start

def _layout(grid, multipole, tier=1):
    """Compute state vector indices.  Returns (n_transport, i_hier_end, i_tg, i_tne, i_tnx, i_net, n_total)."""
    n_transport = 6 * multipole.n_ell * grid.N_q  # 6 species
    i_hier_end = _I_HIER + n_transport
    i_tg = i_hier_end
    if tier >= 2:
        i_tne = i_tg + 1
        i_tnx = i_tne + 1
        i_net = i_tnx + 1
    else:
        i_tne = i_tg  # unused
        i_tnx = i_tg  # unused
        i_net = i_tg + 1
    n_total = i_net + N_SPECIES
    return n_transport, i_hier_end, i_tg, i_tne, i_tnx, i_net, n_total


# ═══════════════════════════════════════════════════════════════
# §3. Hubble rate with curvature
# ═══════════════════════════════════════════════════════════════

def _hubble_invsec(T_gamma, T_nu, N_eff, Sigma_sq, K):
    """Hubble rate [s⁻¹] including spatial curvature.

    H² = (8πG/3) × ρ_total / Ω
    where Ω = 1 - Σ² - K.
    """
    rho_em = rho_plasma(T_gamma)
    rho_nu = N_eff * (7.0/8.0) * _RHO_GAMMA_PREFACTOR * T_nu**4
    rho_total = rho_em + rho_nu
    Omega = enforce_positive_general_omega(1.0 - Sigma_sq - K, context="classA_driver._hubble_invsec", strict=True)
    H_sq = (8.0 * np.pi * _G_N / 3.0) * rho_total / Omega
    return np.sqrt(max(H_sq, 0.0)) * _MEV_TO_S


# ═══════════════════════════════════════════════════════════════
# §4. Coupled RHS
# ═══════════════════════════════════════════════════════════════

def classA_coupled_rhs(N, y, grid, multipole, phase, tau_n, eta, N_eff, f_nu,
                        tier=1, enable_teff=False, n_reactions=12, correction_level=0):
    """Full coupled RHS for Class A Bianchi types.

    Identical to full_coupled_typeI.coupled_rhs except:
    - Geometry uses classA_geometry_rhs with curvature sources
    - State includes (N₁, N₂, N₃)
    - Hubble rate uses Ω = 1 - Σ² - K
    """
    n_transport, i_hier_end, i_tg, i_tne, i_tnx, i_net, n_total = \
        _layout(grid, multipole, tier=tier)

    Sigma_plus = y[_I_SP]
    Sigma_minus = y[_I_SM]
    N1 = y[_I_N1]
    N2 = y[_I_N2]
    N3 = y[_I_N3]
    hier_flat = y[_I_HIER:i_hier_end]
    T_gamma = y[i_tg]
    X = y[i_net:i_net + N_SPECIES]

    if T_gamma < 1e-6:
        return np.zeros(n_total)

    Sigma_sq = Sigma_plus**2 + Sigma_minus**2
    K = gauss_curvature_K(N1, N2, N3)

    # ── Unpack transport ──
    state = HierarchyState.from_flat(hier_flat, grid, multipole)

    # ── Transport → π → Geometry feedback ──
    stress = extract_aniso_stress(state, grid, f_nu=f_nu)

    # ── Geometry RHS (5-component: Σ₊, Σ₋, N₁, N₂, N₃) ──
    dSp, dSm, dN1, dN2, dN3 = classA_geometry_rhs(
        Sigma_plus, Sigma_minus, N1, N2, N3,
        pi_shear_plus=stress.Pi_plus,
        pi_shear_minus=stress.Pi_minus,
    )

    # ── Transport RHS (ℓ_max=2, same as Type I) ──
    # NOTE: This is EXACT for Type I, APPROXIMATE for curved types.
    dhier = compute_hierarchy_rhs_typeI(state, Sigma_plus, Sigma_minus)

    # ── Thermo + Hubble ──
    if tier >= 2:
        T_nu_e = y[i_tne]
        T_nu_x = y[i_tnx]
        from rabbit.thermo.nudec_coupled import coupled_3T_rhs, hubble_3T
        dTg, dTne, dTnx = coupled_3T_rhs(T_gamma, T_nu_e, T_nu_x)
        # hubble_3T doesn't know about K, so compute manually
        H = _hubble_invsec(T_gamma, T_nu_e, _N_EFF, Sigma_sq, K)
        T_nu_for_rates = T_nu_e
    else:
        dTg = dT_gamma_dN_tier1(T_gamma)
        T_nu = T_nu_from_T_gamma_tier1(T_gamma)
        H = _hubble_invsec(T_gamma, T_nu, N_eff, Sigma_sq, K)
        T_nu_for_rates = T_nu

    # ── Weak rates ──
    f_nue = extract_monopole_distribution(state, grid, species_idx=0)
    f_nuebar = extract_monopole_distribution(state, grid, species_idx=1)

    if enable_teff:
        raise ValueError(
            "Class A enable_teff=True is deprecated legacy and is no longer "
            "supported as a runtime path."
        )

    weak = compute_live_weak_rates(
        f_nue, f_nuebar, grid.nodes,
        T_gamma, T_nu_for_rates, tau_n,
        compute_iso_reference=False,
        correction_level=correction_level)

    lnp = weak.lambda_np
    lpn = weak.lambda_pn

    # ── Network RHS ──
    if phase == 1:
        dX = np.zeros(N_SPECIES)
        dX[0] = abundance_rhs_phase1(X[0], lnp, lpn) / max(H, 1e-100)
        dX[1] = -dX[0]
    else:
        dX = abundance_rhs_phase2(X, T_gamma, eta, lnp, lpn,
                                   n_reactions=n_reactions) / max(H, 1e-100)

    # ── Pack ──
    dy = np.zeros(n_total)
    dy[_I_SP] = dSp
    dy[_I_SM] = dSm
    dy[_I_N1] = dN1
    dy[_I_N2] = dN2
    dy[_I_N3] = dN3
    dy[_I_HIER:i_hier_end] = dhier
    dy[i_tg] = dTg
    if tier >= 2:
        dy[i_tne] = dTne
        dy[i_tnx] = dTnx
    dy[i_net:i_net + N_SPECIES] = dX

    return dy


# ═══════════════════════════════════════════════════════════════
# §5. Configuration and entry point
# ═══════════════════════════════════════════════════════════════

@dataclass
class ClassAConfig:
    """Configuration for Class A Bianchi BBN run.

    Parameters
    ----------
    bianchi_type : BianchiType
        Which Class A type to simulate.
    Sigma_H_plus, Sigma_H_minus : float
        Initial Hubble-normalized shear.
    N1_init, N2_init, N3_init : float
        Initial structure-constant variables.
        Type-specific defaults are applied if left at 0.
    """
    bianchi_type: BianchiType = BianchiType.TYPE_I
    Sigma_H_plus: float = 0.0
    Sigma_H_minus: float = 0.0
    N1_init: float = 0.0
    N2_init: float = 0.0
    N3_init: float = 0.0
    T_start: float = 10.0
    T_handoff: float = 0.08
    T_end: float = 0.005
    tau_n: float = _TAU_N
    eta: float = _ETA
    N_eff: float = _N_EFF
    f_nu: float = _F_NU
    N_q: int = 80
    tier: int = 1
    enable_teff: bool = False
    n_reactions: int = 12  # 12=backbone (Paper I), 31=standard with ⁶Li
    correction_level: int = 0  # 0=Born, 1=+Coulomb, 2=+Sirlin, 3=+FM/WM
    grid: Optional[MomentumGrid] = None
    multipole: Optional[MultipoleSpec] = None
    solver: Optional[SolverConfig] = None

    def __post_init__(self):
        if self.grid is None:
            self.grid = MomentumGrid(N_q=self.N_q)
        if self.multipole is None:
            if abs(self.Sigma_H_minus) > 0.0:
                self.multipole = DEFAULT_MULTIPOLE_GENERIC
            else:
                self.multipole = DEFAULT_MULTIPOLE
        if self.solver is None:
            # BD613: PRODUCTION_CONFIG is now BDF; classA preserves its
            # historical Radau-at-production-tolerances default verbatim.
            self.solver = PRODUCTION_RADAU_CONFIG
        if self.enable_teff:
            raise ValueError(
                "ClassAConfig(enable_teff=True) is deprecated legacy and no "
                "longer supported."
            )

        # Validate type-consistency of structure constants
        self._validate_structure_constants()

    def _validate_structure_constants(self):
        """Check that N_i initial conditions match the declared type."""
        bt = self.bianchi_type
        if bt == BianchiType.TYPE_I:
            if any(abs(x) > 0 for x in [self.N1_init, self.N2_init, self.N3_init]):
                raise ValueError("Type I requires N₁=N₂=N₃=0")
        elif bt == BianchiType.TYPE_II:
            if self.N2_init != 0 or self.N3_init != 0:
                raise ValueError("Type II requires N₂=N₃=0")
        elif bt == BianchiType.TYPE_VI0:
            if self.N1_init != 0:
                raise ValueError("Type VI₀ requires N₁=0, with N₂N₃<0")
        elif bt == BianchiType.TYPE_VII0:
            if self.N1_init != 0:
                raise ValueError("Type VII₀ requires N₁=0")
        # VIII and IX: no constraints on individual N_i

    @property
    def initial_geometry(self) -> Tuple[float, float, float, float, float]:
        """(Σ₊, Σ₋, N₁, N₂, N₃) initial conditions."""
        return (self.Sigma_H_plus, self.Sigma_H_minus,
                self.N1_init, self.N2_init, self.N3_init)

    @property
    def initial_K(self) -> float:
        """Initial Gauss curvature."""
        return gauss_curvature_K(self.N1_init, self.N2_init, self.N3_init)

    @property
    def initial_Omega(self) -> float:
        """Initial Ω = 1 - Σ² - K."""
        return compute_Omega(self.Sigma_H_plus, self.Sigma_H_minus,
                             self.N1_init, self.N2_init, self.N3_init)


def type_label(bt: BianchiType) -> str:
    """Human-readable Bianchi type label."""
    labels = {
        BianchiType.TYPE_I: "Type I",
        BianchiType.TYPE_II: "Type II",
        BianchiType.TYPE_VI0: "Type VI₀",
        BianchiType.TYPE_VII0: "Type VII₀",
        BianchiType.TYPE_VIII: "Type VIII",
        BianchiType.TYPE_IX: "Type IX",
    }
    return labels.get(bt, str(bt))


def run_classA_bbn(config: ClassAConfig = None, **kw) -> CanonicalResult:
    """Run a Class A Bianchi BBN computation.

    This is the unified entry point for all Class A types.
    For Type I, this is functionally identical to run_full_coupled_typeI.

    MATURITY: EXPERIMENTAL for curved types (II, VI₀, VII₀, VIII, IX).
    Transport uses ℓ_max=2 truncation which is approximate for K≠0.
    """
    import time
    import warnings

    if config is None:
        config = ClassAConfig(**kw)

    # ── Truncation error guard for curved types ──
    if hasattr(config, 'bianchi_type') and config.bianchi_type not in (None, 'TYPE_I', 'I'):
        warnings.warn(
            f"Class A driver for {config.bianchi_type}: transport uses "
            f"ℓ_max=2 truncation (APPROXIMATE for curved types). "
            f"Truncation error is O(K × Σ); for Type II Collins-Stewart "
            f"attractor, ~6%. Results are EXPERIMENTAL, not publication-grade. "
            f"Full curved hierarchy (ℓ_max > 2) deferred to P5-B.",
            RuntimeWarning, stacklevel=2,
        )

    grid = config.grid
    multipole = config.multipole
    tier = config.tier
    n_transport, i_hier_end, i_tg, i_tne, i_tnx, i_net, n_total = \
        _layout(grid, multipole, tier=tier)

    # ── Initial conditions ──
    hier0 = HierarchyState.from_isotropic(grid, multipole)
    f_eq = fermi_dirac(grid.nodes)

    T_nu_init = config.T_start
    init_weak = compute_live_weak_rates(
        f_eq, f_eq, grid.nodes,
        config.T_start, T_nu_init, config.tau_n,
        compute_iso_reference=False)
    Xn_eq = init_weak.equilibrium_Xn

    y0 = np.zeros(n_total)
    y0[_I_SP] = config.Sigma_H_plus
    y0[_I_SM] = config.Sigma_H_minus
    y0[_I_N1] = config.N1_init
    y0[_I_N2] = config.N2_init
    y0[_I_N3] = config.N3_init
    y0[_I_HIER:i_hier_end] = hier0.to_flat()
    y0[i_tg] = config.T_start
    if tier >= 2:
        y0[i_tne] = config.T_start
        y0[i_tnx] = config.T_start
    y0[i_net] = Xn_eq
    y0[i_net + 1] = 1.0 - Xn_eq

    # ── Phase 1 ──
    def stop_p1(N, y):
        return y[i_tg] - config.T_handoff
    stop_p1.terminal = True
    stop_p1.direction = -1

    t0 = time.perf_counter()

    sol1 = solve_ivp(
        fun=lambda N, y: classA_coupled_rhs(
            N, y, grid, multipole, 1,
            config.tau_n, config.eta, config.N_eff, config.f_nu,
            tier=tier, enable_teff=config.enable_teff,
            n_reactions=config.n_reactions,
            correction_level=config.correction_level),
        t_span=[0.0, 50.0], y0=y0,
        events=stop_p1,
        **config.solver.to_scipy_kwargs())

    if not sol1.success and len(sol1.t) < 2:
        raise RuntimeError(f"Phase 1 failed: {sol1.message}")

    # ── Handoff ──
    y_handoff = sol1.y[:, -1].copy()
    Xn_ho = y_handoff[i_net]
    X_p2 = phase1_to_phase2(Xn_ho)
    y_handoff[i_net:i_net + N_SPECIES] = X_p2
    N_handoff = sol1.t[-1]

    # ── Phase 2 ──
    def stop_p2(N, y):
        return y[i_tg] - config.T_end
    stop_p2.terminal = True
    stop_p2.direction = -1

    sol2 = solve_ivp(
        fun=lambda N, y: classA_coupled_rhs(
            N, y, grid, multipole, 2,
            config.tau_n, config.eta, config.N_eff, config.f_nu,
            tier=tier, enable_teff=config.enable_teff,
            n_reactions=config.n_reactions,
            correction_level=config.correction_level),
        t_span=[N_handoff, N_handoff + 30.0], y0=y_handoff,
        events=stop_p2,
        **config.solver.to_scipy_kwargs())

    wall_time = time.perf_counter() - t0

    if not sol2.success and len(sol2.t) < 2:
        raise RuntimeError(f"Phase 2 failed: {sol2.message}")

    # ── Extract results ──
    y_final = sol2.y[:, -1]
    X_final = y_final[i_net:i_net + N_SPECIES]
    T_final = y_final[i_tg]

    Yp = float(X_final[5])
    DH = float(X_final[2] / (2.0 * max(X_final[1], 1e-30)))
    # ⁷Li/H: mass fraction → number ratio, includes ⁷Be→⁷Li decay (t½=53d)
    Li7_H = float((X_final[6]/7.0 + X_final[7]/7.0) / max(X_final[1], 1e-30))
    Li6_H = float(max(X_final[8], 0.0) / (6.0 * max(X_final[1], 1e-30))) if N_SPECIES > 8 else 0.0

    mc = mass_conservation_residual(X_final)

    # N_eff measurement
    T_nu_final = T_nu_from_T_gamma_tier1(T_final)
    from rabbit.thermo.incomplete_decoupling import N_eff_from_T_ratio
    N_eff_meas = N_eff_from_T_ratio(T_nu_final, T_final)

    # Friedmann constraint at final time
    Sp_f = y_final[_I_SP]
    Sm_f = y_final[_I_SM]
    N1_f = y_final[_I_N1]
    N2_f = y_final[_I_N2]
    N3_f = y_final[_I_N3]
    K_f = gauss_curvature_K(N1_f, N2_f, N3_f)
    Omega_f = compute_Omega(Sp_f, Sm_f, N1_f, N2_f, N3_f)
    friedmann_res = friedmann_residual(Sp_f, Sm_f, N1_f, N2_f, N3_f, Omega_f)

    # Trajectory
    N_all = np.concatenate([sol1.t, sol2.t])
    Sp_all = np.concatenate([sol1.y[_I_SP], sol2.y[_I_SP]])
    Sm_all = np.concatenate([sol1.y[_I_SM], sol2.y[_I_SM]])
    T_all = np.concatenate([sol1.y[i_tg], sol2.y[i_tg]])
    N1_all = np.concatenate([sol1.y[_I_N1], sol2.y[_I_N1]])

    obs = Observables(Yp=Yp, DH=DH, Li7H=Li7_H, Li6H=Li6_H, N_eff=N_eff_meas)

    return CanonicalResult(
        params=RunParameters(
            Sigma_H_plus=config.Sigma_H_plus,
            Sigma_H_minus=config.Sigma_H_minus,
            tau_n=config.tau_n, eta=config.eta,
            N_eff=config.N_eff, f_nu=config.f_nu,
            N_q=grid.N_q,
            bianchi_type=config.bianchi_type),
        observables=obs,
        constraint_diag=ConstraintDiag(
            mass_conservation=np.array([mc])),
        trajectory={
            'N': N_all, 'Sigma_plus': Sp_all, 'Sigma_minus': Sm_all,
            'T_gamma': T_all, 'N1': N1_all,
        },
        wall_time_s=wall_time,
        metadata={
            'driver': 'classA_driver',
            'bianchi_type': type_label(config.bianchi_type),
            'n_dof': n_total,
            'n_geometry_dof': 5,
            'initial_K': config.initial_K,
            'initial_Omega': config.initial_Omega,
            'final_K': K_f,
            'final_Omega': Omega_f,
            'friedmann_residual': friedmann_res,
            'transport_approx': 'ell_max=2 (exact for Type I, approximate for K≠0)',
            'phase1_steps': len(sol1.t),
            'phase2_steps': len(sol2.t),
            'tier': tier,
            'enable_teff': config.enable_teff,
            'correction_level': config.correction_level,
            'N_eff_measured': float(N_eff_meas),
            'mass_conservation': float(mc),
        })
