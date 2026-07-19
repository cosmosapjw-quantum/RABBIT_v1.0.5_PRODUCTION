"""
rabbit.jax.run_tilted_bbn — Tilted BBN for any Bianchi type.

State: [v, Σ₊, Σ₋, N₁, N₂, N₃, (A), Ψ_flat, T_γ, X_species]

Tilt coupling: v ← Σ (eigenvalue), Σ ← Ψ (stress), H ← v (boost).
The tilt RHS is type-independent (depends on q, Σ only).

Supports: all 6 Class A types + Class B types (via A variable).
Tier-1 thermo, CL0-CL3 weak rates, principal-axis scalar tilt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from rabbit.config.conventions import BianchiSpec, BianchiType
from rabbit.jax.geometry_bianchi_base import (
    gauss_curvature_K, curvature_sources, compute_q, compute_Omega,
    structure_eigenvalues, frame_variable_rhs, is_class_B, get_type_mask_jax,
    get_type_mask,
)
from rabbit.jax.geometry_classB_jax import get_c_factor
from rabbit.jax.tilt_jax import (
    boosted_fd_legendre_moments,
    boosted_fd_monopole,
    tilt_anisotropic_stress_principal_axis,
    tilt_hubble_stress_energy_factor,
    tilt_rhs_principal_axis,
    tilt_hubble_factor,
    tilt_omega_correction,
    tilted_normal_energy_density_factor,
)
from rabbit.jax.rhs_classA import classA_transport_rhs, effective_kappa_from_curvature
from rabbit.jax.transport_ops_jax import extract_aniso_stress_operator
from rabbit.jax.thermo_provider_jax import tier1_T_nu_from_T_gamma_jax, tier1_dT_gamma_dN_jax
from rabbit.jax.thermo_jax import rho_plasma, rho_photon, PI, G_N
from rabbit.jax.weak_jax import compute_born_rates, equilibrium_Xn
from rabbit.jax.weak_corrections_jax import compute_corrected_born_rates
from rabbit.jax.weak_live_jax import (
    compute_live_rates_from_moments_cl3_jax,
    compute_live_rates_from_monopoles_level_specialized_jax,
)
from rabbit.jax.network_jax import (
    load_rate_table, abundance_rhs_phase1_jax, abundance_rhs_phase2_jax,
    phase1_to_phase2_jax, N_SPECIES, validate_rate_table_window_jax,
)
from rabbit.validation.truncation_guards import (
    validate_general_initial_budget, validate_manual_network_truncation,
    validate_min_resolution, validate_phase_temperature_order,
)
from rabbit.jax.solver_jax_rodas5p import jax_rodas5p_solve

_MEV_TO_S = 1.519267447e21
_TAU_N = 878.4
_ETA = 6.104e-10
_N_EFF = 3.044
_F_NU = 0.40520
_MOMENTUM_CONSTRAINT_TOL = 1.0e-10
_MOMENTUM_CLOSURE_NONE = "none"
_MOMENTUM_CLOSURE_ALGEBRAIC_HEAT_FLUX = "algebraic_heat_flux"
_MOMENTUM_CLOSURE_MODES = (
    _MOMENTUM_CLOSURE_NONE,
    _MOMENTUM_CLOSURE_ALGEBRAIC_HEAT_FLUX,
)
_TILT_HUBBLE_LEGACY_GAMMA = "legacy_gamma"
_TILT_HUBBLE_STRESS_ENERGY_T00 = "stress_energy_t00"
_TILT_HUBBLE_CLOSURE_MODES = (
    _TILT_HUBBLE_LEGACY_GAMMA,
    _TILT_HUBBLE_STRESS_ENERGY_T00,
)
_WEAK_BUDGET_BY_CL = {
    0: "born",
    1: "born+coulomb",
    2: "born+coulomb+sirlin",
    3: "born+coulomb+sirlin+finite_mass",
}


def _validate_tilt_axis(axis: int) -> int:
    axis_int = int(axis)
    if axis_int not in (1, 2, 3):
        raise ValueError(f"Tilted BBN requires tilt_axis in {{1, 2, 3}}, got {axis!r}.")
    return axis_int


def _axis_tilt_vector(v: float, axis: int) -> tuple[float, float, float]:
    axis_int = _validate_tilt_axis(axis)
    values = [0.0, 0.0, 0.0]
    values[axis_int - 1] = float(v)
    return (values[0], values[1], values[2])


def _validate_momentum_closure(mode: str) -> str:
    normalized = str(mode).lower()
    if normalized not in _MOMENTUM_CLOSURE_MODES:
        allowed = ", ".join(_MOMENTUM_CLOSURE_MODES)
        raise ValueError(f"Unknown momentum_constraint_closure={mode!r}; expected one of {allowed}.")
    return normalized


def _validate_tilt_hubble_closure(mode: str) -> str:
    normalized = str(mode).lower()
    if normalized not in _TILT_HUBBLE_CLOSURE_MODES:
        allowed = ", ".join(_TILT_HUBBLE_CLOSURE_MODES)
        raise ValueError(f"Unknown tilt_hubble_closure={mode!r}; expected one of {allowed}.")
    return normalized


def _vec_norm_inf(values: tuple[float, float, float]) -> float:
    if any(not np.isfinite(x) for x in values):
        return float("nan")
    return float(max(abs(values[0]), abs(values[1]), abs(values[2])))


def _tilted_h_parameter(btype: str, explicit_h: float | None) -> float | None:
    if btype == "TYPE_III":
        return -1.0
    if btype == "TYPE_VI_M19":
        return -1.0 / 9.0
    if btype in ("TYPE_VIH", "TYPE_VIIH"):
        return float(BianchiSpec.from_type(BianchiType[btype], h=explicit_h).h)
    return None


def _masked_tilted_N(
    type_mask: jnp.ndarray,
    values,
    h_parameter: float | None,
) -> jnp.ndarray:
    masked = type_mask * jnp.asarray(values, dtype=jnp.float64)
    if h_parameter is None:
        return masked
    return jnp.asarray([0.0, masked[1], float(h_parameter) * masked[1]], dtype=jnp.float64)


def _structure_rhs_h_locked(
    lam1: jnp.ndarray,
    lam2: jnp.ndarray,
    lam3: jnp.ndarray,
    N1m: jnp.ndarray,
    N2m: jnp.ndarray,
    N3m: jnp.ndarray,
    h_parameter: float | None,
):
    if h_parameter is None:
        return lam1 * N1m, lam2 * N2m, lam3 * N3m
    dN2 = lam2 * N2m
    return jnp.array(0.0, dtype=jnp.float64), dN2, float(h_parameter) * dN2


def _canonical_bianchi_key(raw: str) -> str:
    """Return the string key used by the tilted geometry registry."""
    key = str(raw).upper()
    if key == "FLRW":
        return "TYPE_I"
    if key.startswith("TYPE_"):
        return key
    return f"TYPE_{key}"


def _validate_tilted_structure_constants(
    bianchi_type: str,
    N1: float,
    N2: float,
    N3: float,
    h: float | None = None,
) -> None:
    """Reject structure constants that are inactive or sign-invalid for a type.

    The RHS masks inactive components before integration, but promotion cells
    should not silently convert a requested geometry into another one.
    """
    key = _canonical_bianchi_key(bianchi_type)
    if key in BianchiType.__members__ and key not in ("TYPE_VIH", "TYPE_VIIH") and h is not None:
        BianchiSpec.from_type(BianchiType[key], h=h)
    mask = tuple(float(v) for v in get_type_mask(key))
    values = (float(N1), float(N2), float(N3))
    inactive = [
        name for name, active, value in zip(("N1", "N2", "N3"), mask, values)
        if active == 0.0 and abs(value) > 0.0
    ]
    if inactive:
        joined = ", ".join(inactive)
        raise ValueError(f"{key} has inactive tilted structure constants: {joined}.")

    n1, n2, n3 = values

    def _nz(x: float) -> bool:
        return abs(x) > 0.0

    if key == "TYPE_VI0":
        if _nz(n2) and _nz(n3) and n2 * n3 >= 0.0:
            raise ValueError("TYPE_VI0 requires N2*N3 < 0 for active tilted data.")
    elif key == "TYPE_VII0":
        if _nz(n2) and _nz(n3) and n2 * n3 < 0.0:
            raise ValueError("TYPE_VII0 requires N2*N3 >= 0 for active tilted data.")
    elif key == "TYPE_III":
        if (not _nz(n2)) or (not _nz(n3)) or n2 * n3 >= 0.0:
            raise ValueError("TYPE_III requires canonical h=-1 active tilted data with N2*N3 < 0.")
        scale = max(abs(n2), abs(n3), 1.0e-300)
        if abs(n2 + n3) / scale > 1.0e-12:
            raise ValueError("TYPE_III requires the canonical h=-1 relation N2=-N3.")
    elif key == "TYPE_VI_M19":
        if (not _nz(n2)) or (not _nz(n3)) or n2 * n3 >= 0.0:
            raise ValueError("TYPE_VI_M19 requires canonical h=-1/9 active tilted data with N2*N3 < 0.")
        scale = max(abs(n2), 1.0e-300)
        if abs(n3 + n2 / 9.0) / scale > 1.0e-12:
            raise ValueError("TYPE_VI_M19 requires the canonical h=-1/9 relation N3=-N2/9.")
    elif key in ("TYPE_VIH", "TYPE_VIIH"):
        spec = BianchiSpec.from_type(BianchiType[key], h=h)
        if (not _nz(n2)) or (not _nz(n3)):
            raise ValueError(f"{key} requires active h-family tilted data with nonzero N2 and N3.")
        scale = max(abs(n2), 1.0e-300)
        if abs(n3 - float(spec.h) * n2) / scale > 1.0e-12:
            raise ValueError(f"{key} requires the h-family relation N3=h*N2.")
    elif key == "TYPE_VIII":
        if _nz(n1) and _nz(n2) and _nz(n3):
            same_sign = (n1 * n2 > 0.0) and (n2 * n3 > 0.0)
            if same_sign:
                raise ValueError("TYPE_VIII requires mixed-sign active N_i.")
    elif key == "TYPE_IX":
        if _nz(n1) and _nz(n2) and _nz(n3):
            same_sign = (n1 * n2 > 0.0) and (n2 * n3 > 0.0)
            if not same_sign:
                raise ValueError("TYPE_IX requires same-sign active N_i.")


@dataclass
class TiltedBBNConfig:
    bianchi_type: str = "TYPE_I"
    v0: float = 0.0               # initial principal-axis tilt amplitude
    tilt_axis: int = 3            # velocity axis in the diagonal WH frame
    Sigma_H_plus: float = 0.0
    Sigma_H_minus: float = 0.0
    N1_init: float = 0.0
    N2_init: float = 0.0
    N3_init: float = 0.0
    h: float | None = None
    A_init: float = 0.0           # Class B frame variable
    N_q: int = 6
    n_ell: int = 2
    correction_level: int = 0
    n_reactions: int = 12
    T_start: float = 10.0
    T_handoff: float = 0.08
    T_end: float = 0.005
    tau_n: float = _TAU_N
    eta: float = _ETA
    gamma: float = 4.0 / 3.0
    # v3.0 Phase C (Plan §2.4): defaults flipped False → True. Stress
    # feedback and weak-rate boost are the physically-honest baseline
    # for any v0 ≠ 0; the v2 kinematic-default path is preserved by
    # explicit opt-out. Regression-locked against the new gold cell
    # ``tilted_full_coupled_baseline``.
    tilt_stress_feedback: bool = True
    momentum_constraint_closure: str = _MOMENTUM_CLOSURE_NONE
    tilt_weak_rate_boost: bool = True
    tilt_cl3_angular_kernel: bool = False
    tilt_hubble_closure: str = _TILT_HUBBLE_LEGACY_GAMMA
    rtol: float = 1e-8
    atol: float = 1e-10

    def __post_init__(self):
        validate_phase_temperature_order(self.T_start, self.T_handoff, self.T_end, context="Tilted BBN temperature schedule")
        validate_min_resolution("N_q", self.N_q, minimum=2, strict=True)
        validate_min_resolution("n_ell", self.n_ell, minimum=2, strict=True)
        if int(self.n_ell) not in (2, 3):
            raise ValueError(
                f"Tilted BBN currently supports n_ell=2 LRS or n_ell=3 "
                f"diagonal non-LRS quadrupole transport; got n_ell={self.n_ell}."
            )
        validate_manual_network_truncation(self.n_reactions, context="Tilted BBN network")
        validate_rate_table_window_jax(self.T_handoff, load_rate_table(self.n_reactions), context="Tilted BBN phase-2 handoff", strict=True)
        validate_rate_table_window_jax(self.T_end, load_rate_table(self.n_reactions), context="Tilted BBN phase-2 end", strict=True)
        if abs(float(self.v0)) >= 1.0:
            raise ValueError(f"Tilted BBN initial data require |v0| < 1, got v0={self.v0}.")
        if self.tilt_cl3_angular_kernel and int(self.correction_level) != 3:
            raise ValueError(
                "tilt_cl3_angular_kernel requires correction_level=3 because "
                "the angular K_l coupling belongs to the finite-mass weak budget."
            )
        if self.tilt_cl3_angular_kernel and not self.tilt_weak_rate_boost:
            raise ValueError(
                "tilt_cl3_angular_kernel requires tilt_weak_rate_boost=True so "
                "a non-isotropic boosted FD moment hierarchy is supplied."
            )
        _validate_tilt_axis(self.tilt_axis)
        _validate_momentum_closure(self.momentum_constraint_closure)
        _validate_tilt_hubble_closure(self.tilt_hubble_closure)
        btype = _canonical_bianchi_key(self.bianchi_type)
        _validate_tilted_structure_constants(btype, self.N1_init, self.N2_init, self.N3_init, self.h)
        is_B = is_class_B(btype)
        type_mask = get_type_mask_jax(btype)
        h_parameter = _tilted_h_parameter(btype, self.h)
        N_init = _masked_tilted_N(type_mask, [self.N1_init, self.N2_init, self.N3_init], h_parameter)
        K_init = float(gauss_curvature_K(N_init[0], N_init[1], N_init[2]))
        sigma_sq = float(self.Sigma_H_plus) ** 2 + float(self.Sigma_H_minus) ** 2
        cA_sq = float(get_c_factor(btype)) * float(self.A_init) ** 2 if is_B else 0.0
        validate_general_initial_budget(sigma_sq, K_init, cA_sq, context="Tilted BBN initial data")


@dataclass
class TiltedBBNResult:
    Yp: float
    DH: float
    Li7H: float = 0.0
    v_final: float = 0.0
    Sigma_final: float = 0.0
    success: bool = True
    metadata: dict = None


def _tilted_state_ok(
    v: float,
    Sp: float,
    Sm: float,
    K: float,
    cA_sq: float,
    gamma: float,
    *,
    cap: float = 1e6,
) -> tuple[bool, str, float, float]:
    """Final physical-domain guard for the scalar tilted BBN cell."""
    if not np.isfinite(v):
        return False, "nonfinite_tilt", float("nan"), float("nan")
    v_sq = float(v) ** 2
    if v_sq >= 1.0:
        return False, "superluminal_tilt", float("nan"), float("nan")
    sigma_sq = float(Sp) ** 2 + float(Sm) ** 2
    if not np.isfinite(K):
        return False, "nonfinite_curvature", float("nan"), float("nan")
    if abs(float(K)) > cap:
        omega_orth = float(compute_Omega(sigma_sq, float(K), float(cA_sq)))
        omega_tilt = float(tilt_omega_correction(
            jnp.asarray(omega_orth),
            jnp.asarray(v_sq),
            gamma=float(gamma),
        ))
        return False, "curvature_runaway", omega_orth, omega_tilt
    omega_orth = float(compute_Omega(sigma_sq, float(K), float(cA_sq)))
    if not np.isfinite(omega_orth):
        return False, "nonfinite_omega", float("nan"), float("nan")
    if omega_orth <= 0.0:
        return False, "omega_nonpositive", omega_orth, float("nan")
    omega_tilt = float(tilt_omega_correction(
        jnp.asarray(omega_orth),
        jnp.asarray(v_sq),
        gamma=float(gamma),
    ))
    if not np.isfinite(omega_tilt):
        return False, "nonfinite_tilted_omega", omega_orth, float("nan")
    if omega_tilt <= 0.0:
        return False, "tilted_omega_nonpositive", omega_orth, omega_tilt
    return True, "ok", omega_orth, omega_tilt


def _scalar_tilt_momentum_residual(
    *,
    btype: str,
    v: float,
    tilt_axis: int,
    Sigma_plus: float,
    Sigma_minus: float,
    N1: float,
    N2: float,
    N3: float,
    A: float,
    gamma: float,
    Omega_tilted: float,
    psi_dipole: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> float:
    """Reference G^0_i residual for a principal-axis tilted driver state."""
    if not np.isfinite(Omega_tilted):
        return float("nan")
    from rabbit.geometry.constraints import momentum_residual

    return float(momentum_residual(
        Sigma_plus=float(Sigma_plus),
        Sigma_minus=float(Sigma_minus),
        N1=float(N1),
        N2=float(N2),
        N3=float(N3),
        A=float(A),
        bianchi_type_str=str(btype),
        v_vec=_axis_tilt_vector(float(v), tilt_axis),
        rho_plus_p_over_3H2=float(gamma) * float(Omega_tilted),
        psi_dipole=psi_dipole,
    ))


def _scalar_tilt_required_dipole(
    *,
    btype: str,
    v: float,
    tilt_axis: int,
    Sigma_plus: float,
    Sigma_minus: float,
    N1: float,
    N2: float,
    N3: float,
    A: float,
    gamma: float,
    Omega_tilted: float,
) -> tuple[float, float, float]:
    """Algebraic heat-flux/dipole required by the momentum constraint."""
    if not np.isfinite(Omega_tilted):
        return (float("nan"), float("nan"), float("nan"))
    from rabbit.geometry.constraints import required_momentum_dipole

    return tuple(float(x) for x in required_momentum_dipole(
        Sigma_plus=float(Sigma_plus),
        Sigma_minus=float(Sigma_minus),
        N1=float(N1),
        N2=float(N2),
        N3=float(N3),
        A=float(A),
        bianchi_type_str=str(btype),
        v_vec=_axis_tilt_vector(float(v), tilt_axis),
        rho_plus_p_over_3H2=float(gamma) * float(Omega_tilted),
    ))


def run_tilted_bbn(config: TiltedBBNConfig) -> TiltedBBNResult:
    """Full BBN (Phase-1 + Phase-2) with scalar tilt for any Bianchi type."""
    # CL0-CL3 supported via live weak rates.  CL3 defaults to scalar
    # finite-mass/WM; the direct K_l angular projection is an explicit opt-in.
    if config.correction_level > 3:
        raise ValueError(
            f"correction_level={config.correction_level} not supported in tilted runner. "
            f"Maximum is CL3 (Born + Coulomb + Sirlin + finite mass)."
        )
    btype = _canonical_bianchi_key(config.bianchi_type)
    is_B = is_class_B(btype)
    tilt_axis = _validate_tilt_axis(config.tilt_axis)
    momentum_closure = _validate_momentum_closure(config.momentum_constraint_closure)
    tilt_hubble_closure = _validate_tilt_hubble_closure(config.tilt_hubble_closure)

    # Type mask
    type_mask = get_type_mask_jax(btype)
    c_factor = get_c_factor(btype) if is_B else 0.0
    h_parameter = _tilted_h_parameter(btype, config.h)

    from numpy.polynomial.laguerre import laggauss
    q_nodes_np, q_weights_np = laggauss(config.N_q)
    q_nodes = jnp.array(q_nodes_np, dtype=jnp.float64)
    q_weights = jnp.array(q_weights_np, dtype=jnp.float64)
    f_nu = jnp.array(_F_NU, dtype=jnp.float64)
    rate_table = load_rate_table(n_reactions=config.n_reactions)

    # Spline matrix for live weak rates (needed for CL>0)
    if config.correction_level > 0:
        from rabbit.jax.weak_live_jax import build_not_a_knot_matrix
        _live_weak_spline = build_not_a_knot_matrix(q_nodes)
    else:
        _live_weak_spline = None

    N_q = config.N_q
    n_ell = config.n_ell
    n_transport = 6 * n_ell * N_q

    # State layout:
    #   [v, Σ₊, Σ₋, N₁, N₂, N₃, (A?), Ψ_flat, T_γ, X_species]
    I_V = 0
    I_SP = 1; I_SM = 2
    I_N1 = 3; I_N2 = 4; I_N3 = 5
    if is_B:
        I_A = 6; I_HIER = 7
    else:
        I_A = None; I_HIER = 6
    i_hier_end = I_HIER + n_transport
    i_tg = i_hier_end
    i_net = i_tg + 1

    # Initial conditions
    T0 = config.T_start
    T_nu0 = T0
    Xn_eq = float(equilibrium_Xn(jnp.array(T0), jnp.array(T_nu0), jnp.array(config.tau_n)))
    N_init = _masked_tilted_N(type_mask, [config.N1_init, config.N2_init, config.N3_init], h_parameter)

    # Phase 1
    n_total_p1 = i_net + 2
    y0 = np.zeros(n_total_p1)
    y0[I_V] = config.v0
    y0[I_SP] = config.Sigma_H_plus
    y0[I_SM] = config.Sigma_H_minus
    y0[I_N1] = float(N_init[0])
    y0[I_N2] = float(N_init[1])
    y0[I_N3] = float(N_init[2])
    if is_B:
        y0[I_A] = config.A_init
    y0[i_tg] = T0
    y0[i_net] = Xn_eq
    y0[i_net + 1] = 1.0 - Xn_eq

    gamma = config.gamma
    f_eq_weak = 1.0 / (jnp.exp(jnp.minimum(q_nodes, 500.0)) + 1.0)

    def weak_rates_for_tilted_state(Tg, T_nu, v):
        base_lnp, base_lpn = compute_corrected_born_rates(
            Tg, T_nu, jnp.array(config.tau_n),
            correction_level=config.correction_level,
            q_nodes=q_nodes,
            spline_matrix=_live_weak_spline,
        )
        if not config.tilt_weak_rate_boost:
            return base_lnp, base_lpn

        if config.tilt_cl3_angular_kernel:
            f0_boost, f1_boost, f2_boost = boosted_fd_legendre_moments(q_nodes, v)
            zero_moment = jnp.zeros_like(f_eq_weak)
            live_lnp_boost, live_lpn_boost, _ = compute_live_rates_from_moments_cl3_jax(
                Tg, T_nu, jnp.array(config.tau_n),
                q_nodes,
                f0_boost, f1_boost, f2_boost,
                f0_boost, f1_boost, f2_boost,
                spline_matrix=_live_weak_spline,
            )
            live_lnp_eq, live_lpn_eq, _ = compute_live_rates_from_moments_cl3_jax(
                Tg, T_nu, jnp.array(config.tau_n),
                q_nodes,
                f_eq_weak, zero_moment, zero_moment,
                f_eq_weak, zero_moment, zero_moment,
                spline_matrix=_live_weak_spline,
            )
        else:
            f_boost = boosted_fd_monopole(q_nodes, v)
            live_lnp_boost, live_lpn_boost, _ = compute_live_rates_from_monopoles_level_specialized_jax(
                Tg, T_nu, jnp.array(config.tau_n),
                q_nodes, f_boost, f_boost,
                correction_level=config.correction_level,
                spline_matrix=_live_weak_spline,
            )
            live_lnp_eq, live_lpn_eq, _ = compute_live_rates_from_monopoles_level_specialized_jax(
                Tg, T_nu, jnp.array(config.tau_n),
                q_nodes, f_eq_weak, f_eq_weak,
                correction_level=config.correction_level,
                spline_matrix=_live_weak_spline,
            )
        lnp_factor = live_lnp_boost / jnp.maximum(live_lnp_eq, 1.0e-300)
        lpn_factor = live_lpn_boost / jnp.maximum(live_lpn_eq, 1.0e-300)
        return base_lnp * lnp_factor, base_lpn * lpn_factor

    def weak_rate_boost_diagnostics(Tg: float, T_nu: float, v: float) -> tuple[float, float, float, float, float]:
        if not config.tilt_weak_rate_boost:
            return 1.0, 1.0, 0.0, 0.0, 0.0
        base_lnp, base_lpn = compute_corrected_born_rates(
            jnp.asarray(Tg), jnp.asarray(T_nu), jnp.array(config.tau_n),
            correction_level=config.correction_level,
            q_nodes=q_nodes,
            spline_matrix=_live_weak_spline,
        )
        boosted_lnp, boosted_lpn = weak_rates_for_tilted_state(
            jnp.asarray(Tg), jnp.asarray(T_nu), jnp.asarray(v)
        )
        if config.tilt_cl3_angular_kernel:
            f_boost, f1_boost, f2_boost = boosted_fd_legendre_moments(q_nodes, jnp.asarray(v))
            f1_absmax = jnp.max(jnp.abs(f1_boost))
            f2_absmax = jnp.max(jnp.abs(f2_boost))
        else:
            f_boost = boosted_fd_monopole(q_nodes, jnp.asarray(v))
            f1_absmax = jnp.array(0.0, dtype=jnp.float64)
            f2_absmax = jnp.array(0.0, dtype=jnp.float64)
        f_delta = jnp.max(jnp.abs(f_boost - f_eq_weak))
        return (
            float(boosted_lnp / jnp.maximum(base_lnp, 1.0e-300)),
            float(boosted_lpn / jnp.maximum(base_lpn, 1.0e-300)),
            float(f_delta),
            float(f1_absmax),
            float(f2_absmax),
        )

    def applied_tilt_hubble_factor(v_sq):
        if tilt_hubble_closure == _TILT_HUBBLE_STRESS_ENERGY_T00:
            return tilt_hubble_stress_energy_factor(v_sq, gamma=gamma)
        return tilt_hubble_factor(v_sq)

    def applied_tilt_fluid_omega(Omega, v_sq):
        if tilt_hubble_closure == _TILT_HUBBLE_STRESS_ENERGY_T00:
            factor = tilted_normal_energy_density_factor(v_sq, gamma=gamma)
            return Omega / jnp.maximum(factor, 1.0e-20)
        return tilt_omega_correction(Omega, v_sq, gamma=gamma)

    def rhs_p1(N, y):
        v = y[I_V]
        Sp = y[I_SP]; Sm = y[I_SM]
        N1 = y[I_N1]; N2 = y[I_N2]; N3 = y[I_N3]
        A = y[I_A] if is_B else jnp.array(0.0)
        psi = y[I_HIER:i_hier_end]
        Tg = y[i_tg]
        X = y[i_net:i_net + 2]

        Nm = _masked_tilted_N(type_mask, [N1, N2, N3], h_parameter)
        N1m, N2m, N3m = Nm[0], Nm[1], Nm[2]
        K = gauss_curvature_K(N1m, N2m, N3m)
        S_plus, S_minus = curvature_sources(N1m, N2m, N3m)

        Sigma_sq = Sp**2 + Sm**2
        v_sq = v**2
        cA_sq = c_factor * A**2 if is_B else jnp.array(0.0)
        q = compute_q(Sigma_sq, K, cA_sq)
        Omega = compute_Omega(Sigma_sq, K, cA_sq)
        Omega_tilted_for_stress = applied_tilt_fluid_omega(Omega, v_sq)
        if config.tilt_stress_feedback:
            pi_tilt_plus, pi_tilt_minus = tilt_anisotropic_stress_principal_axis(
                v_sq, Omega_tilted_for_stress, axis=tilt_axis, gamma=gamma)
        else:
            pi_tilt_plus = jnp.array(0.0, dtype=jnp.float64)
            pi_tilt_minus = jnp.array(0.0, dtype=jnp.float64)

        # Principal-axis tilt RHS, coupling through λ_axis.
        dv = tilt_rhs_principal_axis(v, q, Sp, Sm, axis=tilt_axis, gamma=gamma)

        # Transport
        kappa = effective_kappa_from_curvature(N1m, N2m, N3m, jnp.ones(3))
        dpsi = classA_transport_rhs(Sp, Sm, psi, n_ell=n_ell, n_species=6, kappa=kappa)

        pi_plus, pi_minus = extract_aniso_stress_operator(psi, N_q, n_ell, q_nodes, q_weights, f_nu)

        # Geometry
        damping = -(2.0 - q)
        dSp = damping * Sp - S_plus + pi_plus + pi_tilt_plus
        dSm = damping * Sm - S_minus + pi_minus + pi_tilt_minus
        lam1, lam2, lam3 = structure_eigenvalues(q, Sp, Sm)
        dN1, dN2, dN3 = _structure_rhs_h_locked(
            lam1, lam2, lam3, N1m, N2m, N3m, h_parameter)

        # Thermo + Hubble with the selected tilted expansion closure.
        T_nu = tier1_T_nu_from_T_gamma_jax(Tg)
        dTg = tier1_dT_gamma_dN_jax(Tg)
        rho_em = rho_plasma(Tg)
        rho_nu = _N_EFF * (7.0 / 8.0) * rho_photon(T_nu)
        H_sq = jnp.where(
            Omega > 0.0,
            8.0 * PI / 3.0 * G_N * (rho_em + rho_nu) / Omega,
            jnp.nan,
        )
        H_orth = jnp.where((Omega > 0.0) & (H_sq > 0.0), jnp.sqrt(H_sq), jnp.nan)
        H_tilted = H_orth * applied_tilt_hubble_factor(v_sq)
        H_inv_s = H_tilted * _MEV_TO_S

        lnp, lpn = weak_rates_for_tilted_state(Tg, T_nu, v)
        dXn = abundance_rhs_phase1_jax(X[0], lnp, lpn) / jnp.maximum(H_inv_s, 1e-100)

        dy = jnp.zeros_like(y)
        dy = dy.at[I_V].set(dv)
        dy = dy.at[I_SP].set(dSp)
        dy = dy.at[I_SM].set(dSm)
        dy = dy.at[I_N1].set(dN1)
        dy = dy.at[I_N2].set(dN2)
        dy = dy.at[I_N3].set(dN3)
        if is_B:
            dA = frame_variable_rhs(A, q, Sp)
            dy = dy.at[I_A].set(dA)
        dy = dy.at[I_HIER:i_hier_end].set(dpsi)
        dy = dy.at[i_tg].set(dTg)
        dy = dy.at[i_net].set(dXn)
        dy = dy.at[i_net + 1].set(-dXn)
        return dy

    def event_p1(N, y):
        return y[i_tg] - config.T_handoff

    sol1 = jax_rodas5p_solve(
        rhs_p1, jnp.array(y0), jnp.array([0.0, 50.0]),
        rtol=config.rtol, atol=config.atol, max_steps=20000,
        event_fn=event_p1, event_refine_steps=5)

    y_ho = np.asarray(sol1.y_final)

    # Phase 2
    n_total_p2 = i_tg + 1 + N_SPECIES
    i_net_p2 = i_tg + 1
    y0_p2 = np.zeros(n_total_p2)
    y0_p2[:i_net] = y_ho[:i_net]
    X_p2 = np.array(phase1_to_phase2_jax(jnp.array(y_ho[i_net])))
    y0_p2[i_net_p2:i_net_p2 + N_SPECIES] = X_p2

    def rhs_p2(N, y):
        v = y[I_V]
        Sp = y[I_SP]; Sm = y[I_SM]
        N1 = y[I_N1]; N2 = y[I_N2]; N3 = y[I_N3]
        A = y[I_A] if is_B else jnp.array(0.0)
        psi = y[I_HIER:i_hier_end]
        Tg = y[i_tg]
        X = y[i_net_p2:i_net_p2 + N_SPECIES]

        Nm = _masked_tilted_N(type_mask, [N1, N2, N3], h_parameter)
        N1m, N2m, N3m = Nm[0], Nm[1], Nm[2]
        K = gauss_curvature_K(N1m, N2m, N3m)
        S_plus, S_minus = curvature_sources(N1m, N2m, N3m)

        Sigma_sq = Sp**2 + Sm**2
        v_sq = v**2
        cA_sq = c_factor * A**2 if is_B else jnp.array(0.0)
        q = compute_q(Sigma_sq, K, cA_sq)
        Omega = compute_Omega(Sigma_sq, K, cA_sq)
        Omega_tilted_for_stress = applied_tilt_fluid_omega(Omega, v_sq)
        if config.tilt_stress_feedback:
            pi_tilt_plus, pi_tilt_minus = tilt_anisotropic_stress_principal_axis(
                v_sq, Omega_tilted_for_stress, axis=tilt_axis, gamma=gamma)
        else:
            pi_tilt_plus = jnp.array(0.0, dtype=jnp.float64)
            pi_tilt_minus = jnp.array(0.0, dtype=jnp.float64)

        dv = tilt_rhs_principal_axis(v, q, Sp, Sm, axis=tilt_axis, gamma=gamma)

        kappa = effective_kappa_from_curvature(N1m, N2m, N3m, jnp.ones(3))
        dpsi = classA_transport_rhs(Sp, Sm, psi, n_ell=n_ell, n_species=6, kappa=kappa)
        pi_plus, pi_minus = extract_aniso_stress_operator(psi, N_q, n_ell, q_nodes, q_weights, f_nu)

        damping = -(2.0 - q)
        dSp = damping * Sp - S_plus + pi_plus + pi_tilt_plus
        dSm = damping * Sm - S_minus + pi_minus + pi_tilt_minus
        lam1, lam2, lam3 = structure_eigenvalues(q, Sp, Sm)
        dN1, dN2, dN3 = _structure_rhs_h_locked(
            lam1, lam2, lam3, N1m, N2m, N3m, h_parameter)

        T_nu = tier1_T_nu_from_T_gamma_jax(Tg)
        dTg = tier1_dT_gamma_dN_jax(Tg)
        rho_em = rho_plasma(Tg)
        rho_nu = _N_EFF * (7.0 / 8.0) * rho_photon(T_nu)
        H_sq = jnp.where(
            Omega > 0.0,
            8.0 * PI / 3.0 * G_N * (rho_em + rho_nu) / Omega,
            jnp.nan,
        )
        H_orth = jnp.where((Omega > 0.0) & (H_sq > 0.0), jnp.sqrt(H_sq), jnp.nan)
        H_tilted = H_orth * applied_tilt_hubble_factor(v_sq)
        H_inv_s = H_tilted * _MEV_TO_S

        lnp, lpn = weak_rates_for_tilted_state(Tg, T_nu, v)
        dX = abundance_rhs_phase2_jax(X, Tg, config.eta, lnp, lpn, rate_table) \
             / jnp.maximum(H_inv_s, 1e-100)

        dy = jnp.zeros_like(y)
        dy = dy.at[I_V].set(dv)
        dy = dy.at[I_SP].set(dSp)
        dy = dy.at[I_SM].set(dSm)
        dy = dy.at[I_N1].set(dN1)
        dy = dy.at[I_N2].set(dN2)
        dy = dy.at[I_N3].set(dN3)
        if is_B:
            dA = frame_variable_rhs(A, q, Sp)
            dy = dy.at[I_A].set(dA)
        dy = dy.at[I_HIER:i_hier_end].set(dpsi)
        dy = dy.at[i_tg].set(dTg)
        dy = dy.at[i_net_p2:i_net_p2 + N_SPECIES].set(dX)
        return dy

    def event_p2(N, y):
        return y[i_tg] - config.T_end

    N_p2_start = float(sol1.N_final)
    sol2 = jax_rodas5p_solve(
        rhs_p2, jnp.array(y0_p2), jnp.array([N_p2_start, N_p2_start + 50.0]),
        rtol=config.rtol, atol=config.atol, max_steps=20000,
        event_fn=event_p2, event_refine_steps=5)

    yf = np.asarray(sol2.y_final)
    Xf = yf[i_net_p2:i_net_p2 + N_SPECIES]
    Yp = float(Xf[5])
    DH = float(Xf[2] / (2.0 * max(Xf[1], 1e-30)))
    Li7H = float((Xf[6] / 7.0 + Xf[7] / 7.0) / max(Xf[1], 1e-30))
    N_init_final = _masked_tilted_N(type_mask, [yf[I_N1], yf[I_N2], yf[I_N3]], h_parameter)
    K_init = float(gauss_curvature_K(N_init[0], N_init[1], N_init[2]))
    K_final = float(gauss_curvature_K(N_init_final[0], N_init_final[1], N_init_final[2]))
    sigma_sq_final = float(yf[I_SP]) ** 2 + float(yf[I_SM]) ** 2
    A_final = float(yf[I_A]) if is_B else 0.0
    cA_sq_init = float(c_factor) * float(config.A_init) ** 2 if is_B else 0.0
    cA_sq_final = float(c_factor) * A_final**2 if is_B else 0.0
    sigma_sq_init = float(config.Sigma_H_plus) ** 2 + float(config.Sigma_H_minus) ** 2
    omega_init = float(compute_Omega(
        jnp.asarray(sigma_sq_init),
        jnp.asarray(K_init),
        jnp.asarray(cA_sq_init),
    ))
    v0_sq = float(config.v0) ** 2
    v_final_sq = float(yf[I_V]) ** 2
    omega_tilted_legacy_init = float(tilt_omega_correction(
        jnp.asarray(omega_init),
        jnp.asarray(v0_sq),
        gamma=float(config.gamma),
    ))
    omega_tilted_stress_energy_init = float(
        jnp.asarray(omega_init) / jnp.maximum(
            tilted_normal_energy_density_factor(
                jnp.asarray(v0_sq),
                gamma=float(config.gamma),
            ),
            1.0e-20,
        )
    )
    omega_tilted_init = float(applied_tilt_fluid_omega(
        jnp.asarray(omega_init),
        jnp.asarray(v0_sq),
    ))
    tilt_pi_plus_init, tilt_pi_minus_init = tilt_anisotropic_stress_principal_axis(
        jnp.asarray(v0_sq),
        jnp.asarray(omega_tilted_init),
        axis=tilt_axis,
        gamma=float(config.gamma),
    )
    state_ok, state_reason, omega_final, omega_tilted_legacy_final = _tilted_state_ok(
        float(yf[I_V]),
        float(yf[I_SP]),
        float(yf[I_SM]),
        K_final,
        cA_sq_final,
        float(config.gamma),
    )
    omega_tilted_stress_energy_final = float(
        jnp.asarray(omega_final) / jnp.maximum(
            tilted_normal_energy_density_factor(
                jnp.asarray(v_final_sq),
                gamma=float(config.gamma),
            ),
            1.0e-20,
        )
    )
    omega_tilted_final = float(applied_tilt_fluid_omega(
        jnp.asarray(omega_final),
        jnp.asarray(v_final_sq),
    ))
    tilt_pi_plus_final, tilt_pi_minus_final = tilt_anisotropic_stress_principal_axis(
        jnp.asarray(v_final_sq),
        jnp.asarray(omega_tilted_final),
        axis=tilt_axis,
        gamma=float(config.gamma),
    )
    q_final = float(compute_q(
        jnp.asarray(sigma_sq_final),
        jnp.asarray(K_final),
        jnp.asarray(cA_sq_final),
    ))
    (
        weak_boost_lnp_factor_init,
        weak_boost_lpn_factor_init,
        weak_boost_delta_init,
        weak_boost_f1_absmax_init,
        weak_boost_f2_absmax_init,
    ) = weak_rate_boost_diagnostics(
        float(T0), float(T_nu0), float(config.v0)
    )
    T_nu_final = float(tier1_T_nu_from_T_gamma_jax(jnp.asarray(yf[i_tg])))
    (
        weak_boost_lnp_factor_final,
        weak_boost_lpn_factor_final,
        weak_boost_delta_final,
        weak_boost_f1_absmax_final,
        weak_boost_f2_absmax_final,
    ) = weak_rate_boost_diagnostics(
        float(yf[i_tg]), T_nu_final, float(yf[I_V])
    )
    legacy_hubble_factor_init = float(tilt_hubble_factor(jnp.asarray(v0_sq)))
    legacy_hubble_factor_final = float(tilt_hubble_factor(jnp.asarray(v_final_sq)))
    stress_energy_hubble_factor_init = float(tilt_hubble_stress_energy_factor(
        jnp.asarray(v0_sq),
        gamma=float(config.gamma),
    ))
    stress_energy_hubble_factor_final = float(tilt_hubble_stress_energy_factor(
        jnp.asarray(v_final_sq),
        gamma=float(config.gamma),
    ))
    applied_hubble_factor_init = float(applied_tilt_hubble_factor(jnp.asarray(v0_sq)))
    applied_hubble_factor_final = float(applied_tilt_hubble_factor(jnp.asarray(v_final_sq)))
    kappa_init = float(effective_kappa_from_curvature(
        N_init[0], N_init[1], N_init[2], type_mask))
    kappa_final = float(effective_kappa_from_curvature(
        N_init_final[0], N_init_final[1], N_init_final[2], type_mask))
    transport_pi_plus_init, transport_pi_minus_init = extract_aniso_stress_operator(
        jnp.asarray(y0[I_HIER:i_hier_end]),
        N_q,
        n_ell,
        q_nodes,
        q_weights,
        f_nu,
    )
    transport_pi_plus_final, transport_pi_minus_final = extract_aniso_stress_operator(
        jnp.asarray(yf[I_HIER:i_hier_end]),
        N_q,
        n_ell,
        q_nodes,
        q_weights,
        f_nu,
    )
    momentum_required_dipole_init = _scalar_tilt_required_dipole(
        btype=btype,
        v=float(config.v0),
        tilt_axis=tilt_axis,
        Sigma_plus=float(config.Sigma_H_plus),
        Sigma_minus=float(config.Sigma_H_minus),
        N1=float(N_init[0]),
        N2=float(N_init[1]),
        N3=float(N_init[2]),
        A=float(config.A_init) if is_B else 0.0,
        gamma=float(config.gamma),
        Omega_tilted=omega_tilted_init,
    )
    momentum_required_dipole_final = _scalar_tilt_required_dipole(
        btype=btype,
        v=float(yf[I_V]),
        tilt_axis=tilt_axis,
        Sigma_plus=float(yf[I_SP]),
        Sigma_minus=float(yf[I_SM]),
        N1=float(N_init_final[0]),
        N2=float(N_init_final[1]),
        N3=float(N_init_final[2]),
        A=A_final,
        gamma=float(config.gamma),
        Omega_tilted=omega_tilted_final,
    )
    momentum_no_dipole_residual_init = _scalar_tilt_momentum_residual(
        btype=btype,
        v=float(config.v0),
        tilt_axis=tilt_axis,
        Sigma_plus=float(config.Sigma_H_plus),
        Sigma_minus=float(config.Sigma_H_minus),
        N1=float(N_init[0]),
        N2=float(N_init[1]),
        N3=float(N_init[2]),
        A=float(config.A_init) if is_B else 0.0,
        gamma=float(config.gamma),
        Omega_tilted=omega_tilted_init,
    )
    momentum_no_dipole_residual_final = _scalar_tilt_momentum_residual(
        btype=btype,
        v=float(yf[I_V]),
        tilt_axis=tilt_axis,
        Sigma_plus=float(yf[I_SP]),
        Sigma_minus=float(yf[I_SM]),
        N1=float(N_init_final[0]),
        N2=float(N_init_final[1]),
        N3=float(N_init_final[2]),
        A=A_final,
        gamma=float(config.gamma),
        Omega_tilted=omega_tilted_final,
    )
    if momentum_closure == _MOMENTUM_CLOSURE_ALGEBRAIC_HEAT_FLUX:
        momentum_psi_init = momentum_required_dipole_init
        momentum_psi_final = momentum_required_dipole_final
    else:
        momentum_psi_init = (0.0, 0.0, 0.0)
        momentum_psi_final = (0.0, 0.0, 0.0)
    momentum_residual_init = _scalar_tilt_momentum_residual(
        btype=btype,
        v=float(config.v0),
        tilt_axis=tilt_axis,
        Sigma_plus=float(config.Sigma_H_plus),
        Sigma_minus=float(config.Sigma_H_minus),
        N1=float(N_init[0]),
        N2=float(N_init[1]),
        N3=float(N_init[2]),
        A=float(config.A_init) if is_B else 0.0,
        gamma=float(config.gamma),
        Omega_tilted=omega_tilted_init,
        psi_dipole=momentum_psi_init,
    )
    momentum_residual_final = _scalar_tilt_momentum_residual(
        btype=btype,
        v=float(yf[I_V]),
        tilt_axis=tilt_axis,
        Sigma_plus=float(yf[I_SP]),
        Sigma_minus=float(yf[I_SM]),
        N1=float(N_init_final[0]),
        N2=float(N_init_final[1]),
        N3=float(N_init_final[2]),
        A=A_final,
        gamma=float(config.gamma),
        Omega_tilted=omega_tilted_final,
        psi_dipole=momentum_psi_final,
    )

    return TiltedBBNResult(
        Yp=(Yp if state_ok else float("nan")),
        DH=(DH if state_ok else float("nan")),
        Li7H=(Li7H if state_ok else float("nan")),
        v_final=float(yf[I_V]),
        Sigma_final=float(yf[I_SP]),
        success=bool(sol1.success and sol2.success and state_ok),
        metadata={
            'backend': 'jax_tilted_bbn',
            'bianchi_type': config.bianchi_type,
            'canonical_bianchi_type': btype,
            'bianchi_class': 'B' if is_B else 'A',
            'c_factor': float(c_factor) if is_B else 0.0,
            'h_parameter': h_parameter,
            'transport_mode': 'tilted_kappa_cascade_lmax2',
            'transport_quadrupole_mode': (
                'diagonal_nonlrs_plus_minus'
                if int(n_ell) >= 3
                else 'lrs_plus'
            ),
            'transport_n_ell': int(n_ell),
            'transport_pi_minus_enabled': bool(int(n_ell) >= 3),
            'transport_pi_plus_init': float(transport_pi_plus_init),
            'transport_pi_plus_final': float(transport_pi_plus_final),
            'transport_pi_minus_init': float(transport_pi_minus_init),
            'transport_pi_minus_final': float(transport_pi_minus_final),
            'transport_nonlrs_honesty': (
                'n_ell=3 diagonal non-LRS quadrupole feedback: Sigma_minus '
                'sources Psi_minus and Pi_minus in the reduced PSTF state; '
                'not a full m-decomposed curved hierarchy or mixed-axis vector transport'
                if int(n_ell) >= 3
                else 'n_ell=2 LRS reduced quadrupole feedback; Pi_minus inactive'
            ),
            'transport_kappa_init': kappa_init,
            'transport_kappa_final': kappa_final,
            'v0': config.v0,
            'tilt_axis': int(tilt_axis),
            'tilt_vector_init': _axis_tilt_vector(float(config.v0), tilt_axis),
            'tilt_vector_final': _axis_tilt_vector(float(yf[I_V]), tilt_axis),
            'v_final': float(yf[I_V]),
            'v_sq_init': v0_sq,
            'v_sq_final': v_final_sq,
            'tilt_hubble_factor_init': applied_hubble_factor_init,
            'tilt_hubble_factor_final': applied_hubble_factor_final,
            'tilt_hubble_closure_mode': tilt_hubble_closure,
            'tilt_hubble_legacy_gamma_factor_init': legacy_hubble_factor_init,
            'tilt_hubble_legacy_gamma_factor_final': legacy_hubble_factor_final,
            'tilt_hubble_stress_energy_factor_init': stress_energy_hubble_factor_init,
            'tilt_hubble_stress_energy_factor_final': stress_energy_hubble_factor_final,
            'tilt_hubble_closure_honesty': (
                'normal-frame T00 closure for a tilted perfect fluid; '
                'does not yet replace the reduced momentum/transport hierarchy'
                if tilt_hubble_closure == _TILT_HUBBLE_STRESS_ENERGY_T00
                else 'legacy gamma-factor Hubble closure preserved for gold compatibility'
            ),
            'Sigma_plus_final': float(yf[I_SP]),
            'Sigma_minus_final': float(yf[I_SM]),
            'N1_final': float(N_init_final[0]),
            'N2_final': float(N_init_final[1]),
            'N3_final': float(N_init_final[2]),
            'curvature_K_init': K_init,
            'curvature_K_final': K_final,
            'frame_cA_sq_init': cA_sq_init,
            'frame_cA_sq_final': cA_sq_final,
            'Omega_init': omega_init,
            'Omega_final': omega_final,
            'Omega_tilted_init': omega_tilted_init,
            'Omega_tilted_final': omega_tilted_final,
            'Omega_tilted_closure_mode': tilt_hubble_closure,
            'Omega_tilted_legacy_gamma_init': omega_tilted_legacy_init,
            'Omega_tilted_legacy_gamma_final': omega_tilted_legacy_final,
            'Omega_tilted_stress_energy_init': omega_tilted_stress_energy_init,
            'Omega_tilted_stress_energy_final': omega_tilted_stress_energy_final,
            'tilt_stress_feedback_enabled': bool(config.tilt_stress_feedback),
            'tilt_perfect_fluid_pi_plus_init': float(tilt_pi_plus_init),
            'tilt_perfect_fluid_pi_minus_init': float(tilt_pi_minus_init),
            'tilt_perfect_fluid_pi_plus_final': float(tilt_pi_plus_final),
            'tilt_perfect_fluid_pi_minus_final': float(tilt_pi_minus_final),
            'tilt_weak_rate_boost_enabled': bool(config.tilt_weak_rate_boost),
            'tilt_weak_rate_boost_mode': (
                (
                    'boosted_fd_l012_angular_kernel_ratio'
                    if config.tilt_cl3_angular_kernel
                    else 'boosted_fd_monopole_ratio'
                )
                if config.tilt_weak_rate_boost
                else 'off_equilibrium_fd_temperature'
            ),
            'tilt_cl3_angular_kernel_enabled': bool(config.tilt_cl3_angular_kernel),
            'tilt_cl3_angular_kernel_mode': (
                'boosted_fd_f0_f1_f2_principal_axis'
                if config.tilt_cl3_angular_kernel
                else 'off_scalar_f0_live_weak'
            ),
            'tilt_weak_rate_boost_lnp_factor_init': weak_boost_lnp_factor_init,
            'tilt_weak_rate_boost_lpn_factor_init': weak_boost_lpn_factor_init,
            'tilt_weak_rate_boost_lnp_factor_final': weak_boost_lnp_factor_final,
            'tilt_weak_rate_boost_lpn_factor_final': weak_boost_lpn_factor_final,
            'tilt_weak_rate_boost_monopole_delta_init': weak_boost_delta_init,
            'tilt_weak_rate_boost_monopole_delta_final': weak_boost_delta_final,
            'tilt_cl3_angular_f1_absmax_init': weak_boost_f1_absmax_init,
            'tilt_cl3_angular_f1_absmax_final': weak_boost_f1_absmax_final,
            'tilt_cl3_angular_f2_absmax_init': weak_boost_f2_absmax_init,
            'tilt_cl3_angular_f2_absmax_final': weak_boost_f2_absmax_final,
            'tilt_weak_rate_boost_n_mu': 16,
            'tilt_weak_rate_boost_honesty': (
                (
                    'Lorentz-boosted FD l=0,1,2 moments coupled to the CL3 '
                    'finite-mass angular K_l weak kernel; principal-axis '
                    'projection only, not dynamic Boltzmann angular transport'
                    if config.tilt_cl3_angular_kernel
                    else (
                        'Lorentz-boosted FD monopole in the plasma frame; '
                        'not a full angle-dependent weak kernel'
                    )
                )
                if config.tilt_weak_rate_boost
                else 'weak rates use equilibrium FD monopole parameterized by T_nu'
            ),
            'q_final': q_final,
            'final_state_ok': bool(state_ok),
            'final_state_reason': state_reason,
            'momentum_constraint_closure_mode': momentum_closure,
            'momentum_constraint_residual_no_dipole_init': momentum_no_dipole_residual_init,
            'momentum_constraint_residual_no_dipole_final': momentum_no_dipole_residual_final,
            'momentum_required_dipole_init': momentum_required_dipole_init,
            'momentum_required_dipole_final': momentum_required_dipole_final,
            'momentum_required_dipole_norm_init': _vec_norm_inf(momentum_required_dipole_init),
            'momentum_required_dipole_norm_final': _vec_norm_inf(momentum_required_dipole_final),
            'momentum_applied_dipole_init': momentum_psi_init,
            'momentum_applied_dipole_final': momentum_psi_final,
            'momentum_constraint_residual_init': momentum_residual_init,
            'momentum_constraint_residual_final': momentum_residual_final,
            'momentum_constraint_tolerance': _MOMENTUM_CONSTRAINT_TOL,
            'momentum_constraint_ok_init': bool(
                np.isfinite(momentum_residual_init)
                and momentum_residual_init <= _MOMENTUM_CONSTRAINT_TOL
            ),
            'momentum_constraint_ok_final': bool(
                np.isfinite(momentum_residual_final)
                and momentum_residual_final <= _MOMENTUM_CONSTRAINT_TOL
            ),
            'momentum_constraint_source': (
                (
                    'scalar_v3_algebraic_heat_flux'
                    if int(tilt_axis) == 3
                    else f'principal_axis_v{int(tilt_axis)}_algebraic_heat_flux'
                )
                if momentum_closure == _MOMENTUM_CLOSURE_ALGEBRAIC_HEAT_FLUX
                else (
                    'scalar_v3_no_dipole_diagnostic'
                    if int(tilt_axis) == 3
                    else f'principal_axis_v{int(tilt_axis)}_no_dipole_diagnostic'
                )
            ),
            'momentum_constraint_enforced': bool(
                momentum_closure == _MOMENTUM_CLOSURE_ALGEBRAIC_HEAT_FLUX
            ),
            'momentum_constraint_dynamical_heat_flux': False,
            'momentum_constraint_closure_honesty': (
                'algebraic G0i heat-flux closure; heat flux is not yet evolved'
                if momentum_closure == _MOMENTUM_CLOSURE_ALGEBRAIC_HEAT_FLUX
                else 'no heat-flux/dipole closure applied'
            ),
            'A_init': config.A_init,
            'A_final': A_final,
            'T_final': float(yf[i_tg]),
            'N_q': N_q,
            'correction_level': config.correction_level,
            'weak_budget_mode': _WEAK_BUDGET_BY_CL.get(
                int(config.correction_level),
                f"unknown_cl{int(config.correction_level)}",
            ),
            'correction_budget_channels': {
                'coulomb': int(config.correction_level) >= 1,
                'radiative_sirlin': int(config.correction_level) >= 2,
                'finite_mass_recoil_wm': int(config.correction_level) >= 3,
                'finite_mass_scalar': int(config.correction_level) >= 3,
                'finite_mass_angular_kernel_coupled': bool(config.tilt_cl3_angular_kernel),
                'sirlin_channels': ('c', 'f') if int(config.correction_level) >= 2 else (),
            },
            'weak_budget_honesty': (
                (
                    'CL3 finite-mass/recoil/weak-magnetism live-weak budget with '
                    'direct l<=2 K_l angular coupling for boosted FD moments'
                    if config.tilt_cl3_angular_kernel
                    else (
                        'CL3 finite-mass/recoil/weak-magnetism scalar live-weak budget; '
                        'direct anisotropic K_{r,l} angular coupling inactive'
                    )
                )
                if int(config.correction_level) >= 3
                else 'CL0-CL2 live weak budget; angular finite-mass kernel inactive'
            ),
        },
    )
