"""
rabbit.jax.driver_classA — JAX Class A Bianchi BBN driver.

Extends the Type-I JAX driver to all six Class A orthogonal Bianchi types
(I, II, VI₀, VII₀, VIII, IX) using the Wainwright–Hsu formalism.

State vector: y = [Σ₊, Σ₋, N₁, N₂, N₃, Ψ_flat(...), T_γ, T_νe, T_νx, X₀..X_n]

Transport: κ-cascade reduced branch (ℓ_max=2).
  - κ=0 (Type I): exact flat streaming
  - κ>0 (curved types): approximate monopole-quadrupole mixing
  - NOT the exact curved PSTF hierarchy

Geometry: exact for all 6 types (see geometry_classA_jax.py).
Weak: CL0-CL3 live monopole (same as Type I driver).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from rabbit.config.conventions import BianchiType
from rabbit.jax.geometry_classA_jax import (
    compute_classA_geometry_rhs_jax,
    compute_curvature_invariants_jax,
    build_type_mask,
)
from rabbit.jax.rhs_classA import classA_transport_rhs, effective_kappa_from_curvature
from rabbit.jax.thermo_provider_jax import tier1_T_nu_from_T_gamma_jax, tier1_dT_gamma_dN_jax
from rabbit.jax.nudec_coupled_jax import hubble_3T_jax, coupled_3T_rhs_jax, N_eff_from_3T_jax
from rabbit.jax.thermo_jax import rho_plasma, rho_photon, PI, G_N
from rabbit.jax.weak_live_jax import (
    compute_live_rates_from_monopoles_level_specialized_jax,
    build_not_a_knot_matrix,
)
from rabbit.jax.transport_ops_jax import (
    extract_aniso_stress_operator,
    extract_monopole_distributions_operator,
    extract_pi_tilde_per_species,
)
from rabbit.jax.teff_correction_jax import apply_teff_correction_to_monopoles
from rabbit.jax.network_jax import (
    load_rate_table, abundance_rhs_phase1_jax, abundance_rhs_phase2_jax,
    phase1_to_phase2_jax, N_SPECIES,
)
from rabbit.jax.solver_jax_rodas5p import jax_rodas5p_solve
from rabbit.jax.weak_jax import compute_born_rates, equilibrium_Xn


# ═══════════════════════════════════════════════════════════════
# §1. Constants
# ═══════════════════════════════════════════════════════════════

_MEV_TO_S = 1.519267447e21
_I_SP = 0   # Σ₊
_I_SM = 1   # Σ₋
_I_N1 = 2   # N₁
_I_N2 = 3   # N₂
_I_N3 = 4   # N₃
_I_HIER = 5  # hierarchy start


# ═══════════════════════════════════════════════════════════════
# §2. Layout
# ═══════════════════════════════════════════════════════════════

def _layout(N_q, n_ell, n_species_net):
    """State vector layout for Class A.

    Returns (n_transport, i_hier_end, i_tg, i_tne, i_tnx, i_net, n_total).
    """
    n_transport = 6 * n_ell * N_q
    i_hier_end = _I_HIER + n_transport
    i_tg = i_hier_end
    i_tne = i_tg + 1
    i_tnx = i_tne + 1
    i_net = i_tnx + 1
    n_total = i_net + n_species_net
    return n_transport, i_hier_end, i_tg, i_tne, i_tnx, i_net, n_total


def _apply_type_mask_to_curvature(type_mask, N1, N2, N3):
    mask = jnp.asarray(type_mask, dtype=jnp.float64)
    N = jnp.asarray([N1, N2, N3], dtype=jnp.float64)
    return mask * N


def _masked_curvature_diagnostics(type_mask, N1, N2, N3):
    N1m, N2m, N3m = _apply_type_mask_to_curvature(type_mask, N1, N2, N3)
    K, _, _ = compute_curvature_invariants_jax(N1m, N2m, N3m)
    return (float(N1m), float(N2m), float(N3m), float(K))


# ═══════════════════════════════════════════════════════════════
# §3. Hubble rate with curvature
# ═══════════════════════════════════════════════════════════════

@jax.jit
def _hubble_classA_jax(T_gamma, T_nu_e, T_nu_x, Sigma_sq, K):
    """Hubble rate [MeV] with Class-A curvature and explicit 3T neutrino energy.

    Important: for tier-2 3T runs we must use the *actual* neutrino energy density,
    not ``N_eff_from_3T_jax`` multiplied by a temperature-dependent reference photon
    density.  The previous implementation mixed the asymptotic ``N_eff`` diagnostic
    definition with the dynamical Hubble source and overcounted ``rho_nu`` at high T
    (where ``T_nu pprox T_gamma`` but ``T_std=(4/11)^{1/3} T_gamma``), which drove
    spuriously early freeze-out and inflated ``Y_p``.
    """
    rho_pl = rho_plasma(T_gamma)
    rho_nu = (7.0 / 8.0) * (rho_photon(T_nu_e) + 2.0 * rho_photon(T_nu_x))
    rho_total = rho_pl + rho_nu
    Omega = 1.0 - Sigma_sq - K
    H_sq = 8.0 * PI / 3.0 * G_N * rho_total / Omega
    return jnp.where((Omega > 0.0) & (H_sq > 0.0), jnp.sqrt(H_sq), jnp.nan)


# ═══════════════════════════════════════════════════════════════
# §4. Coupled RHS (tier-2, live-weak)
# ═══════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=("correction_level", "enable_teff", "n_ell"))
def _classA_rhs_phase1(
    N, y, tau_n, f_nu,
    q_nodes, q_weights, type_mask, live_weak_spline_matrix,
    correction_level, enable_teff=False, n_ell=2,
):
    N_q = q_nodes.shape[0]
    n_transport = 6 * n_ell * N_q
    i_hier_end = _I_HIER + n_transport
    i_tg = i_hier_end
    i_tne = i_tg + 1
    i_tnx = i_tne + 1
    i_net = i_tnx + 1

    Sp = y[_I_SP]
    Sm = y[_I_SM]
    N1 = y[_I_N1]
    N2 = y[_I_N2]
    N3 = y[_I_N3]
    psi_flat = y[_I_HIER:i_hier_end]
    T_gamma = y[i_tg]
    T_nu_e = y[i_tne]
    T_nu_x = y[i_tnx]
    X = y[i_net:i_net + 2]

    Sigma_sq = Sp**2 + Sm**2
    N1m, N2m, N3m = type_mask * jnp.array([N1, N2, N3])
    K, S_plus, S_minus = compute_curvature_invariants_jax(N1m, N2m, N3m)
    q = 1.0 + Sigma_sq - K

    # Geometry RHS
    damping = -(2.0 - q)
    pi_plus, pi_minus = extract_aniso_stress_operator(psi_flat, N_q, n_ell, q_nodes, q_weights, f_nu)
    dSp = damping * Sp - S_plus + pi_plus
    dSm = damping * Sm - S_minus + pi_minus
    dN1 = (q - 4.0 * Sp) * N1m
    dN2 = (q + 2.0 * Sp + 2.0 * jnp.sqrt(3.0) * Sm) * N2m
    dN3 = (q + 2.0 * Sp - 2.0 * jnp.sqrt(3.0) * Sm) * N3m

    # Transport RHS (approximate κ-cascade)
    kappa = effective_kappa_from_curvature(N1m, N2m, N3m, jnp.ones(3))
    dpsi = classA_transport_rhs(Sp, Sm, psi_flat, n_ell=n_ell, n_species=6, kappa=kappa)

    # Thermo
    H_MeV = _hubble_classA_jax(T_gamma, T_nu_e, T_nu_x, Sigma_sq, K)
    H_inv_s = H_MeV * _MEV_TO_S
    dTg, dTne, dTnx = coupled_3T_rhs_jax(T_gamma, T_nu_e, T_nu_x, H_MeV=H_MeV)

    # Weak rates
    f_nue, f_nuebar = extract_monopole_distributions_operator(psi_flat, N_q, n_ell, q_nodes)
    if enable_teff:
        pi_tilde_species = extract_pi_tilde_per_species(psi_flat, N_q, n_ell, q_nodes, q_weights)
        f_nue, f_nuebar = apply_teff_correction_to_monopoles(
            f_nue, f_nuebar, q_nodes, pi_tilde_species[0], pi_tilde_species[1],
            spline_matrix=live_weak_spline_matrix)
    lnp, lpn, _ = compute_live_rates_from_monopoles_level_specialized_jax(
        T_gamma, T_nu_e, tau_n, q_nodes, f_nue, f_nuebar,
        correction_level=correction_level, spline_matrix=live_weak_spline_matrix)

    dXn = abundance_rhs_phase1_jax(X[0], lnp, lpn) / jnp.maximum(H_inv_s, 1e-100)
    dX = jnp.array([dXn, -dXn])

    dy = jnp.zeros_like(y)
    dy = dy.at[_I_SP].set(dSp)
    dy = dy.at[_I_SM].set(dSm)
    dy = dy.at[_I_N1].set(dN1)
    dy = dy.at[_I_N2].set(dN2)
    dy = dy.at[_I_N3].set(dN3)
    dy = dy.at[_I_HIER:i_hier_end].set(dpsi)
    dy = dy.at[i_tg].set(dTg)
    dy = dy.at[i_tne].set(dTne)
    dy = dy.at[i_tnx].set(dTnx)
    dy = dy.at[i_net:i_net + 2].set(dX)
    return dy


@partial(jax.jit, static_argnames=("correction_level", "enable_teff", "n_ell"))
def _classA_rhs_phase2(
    N, y, tau_n, eta, f_nu,
    q_nodes, q_weights, type_mask, rate_table, live_weak_spline_matrix,
    correction_level, enable_teff=False, n_ell=2,
):
    N_q = q_nodes.shape[0]
    n_transport = 6 * n_ell * N_q
    i_hier_end = _I_HIER + n_transport
    i_tg = i_hier_end
    i_tne = i_tg + 1
    i_tnx = i_tne + 1
    i_net = i_tnx + 1

    Sp = y[_I_SP]
    Sm = y[_I_SM]
    N1 = y[_I_N1]
    N2 = y[_I_N2]
    N3 = y[_I_N3]
    psi_flat = y[_I_HIER:i_hier_end]
    T_gamma = y[i_tg]
    T_nu_e = y[i_tne]
    T_nu_x = y[i_tnx]
    X = y[i_net:i_net + N_SPECIES]

    Sigma_sq = Sp**2 + Sm**2
    N1m, N2m, N3m = type_mask * jnp.array([N1, N2, N3])
    K, S_plus, S_minus = compute_curvature_invariants_jax(N1m, N2m, N3m)
    q = 1.0 + Sigma_sq - K

    pi_plus, pi_minus = extract_aniso_stress_operator(psi_flat, N_q, n_ell, q_nodes, q_weights, f_nu)
    dSp = -(2.0 - q) * Sp - S_plus + pi_plus
    dSm = -(2.0 - q) * Sm - S_minus + pi_minus
    dN1 = (q - 4.0 * Sp) * N1m
    dN2 = (q + 2.0 * Sp + 2.0 * jnp.sqrt(3.0) * Sm) * N2m
    dN3 = (q + 2.0 * Sp - 2.0 * jnp.sqrt(3.0) * Sm) * N3m

    kappa = effective_kappa_from_curvature(N1m, N2m, N3m, jnp.ones(3))
    dpsi = classA_transport_rhs(Sp, Sm, psi_flat, n_ell=n_ell, n_species=6, kappa=kappa)

    H_MeV = _hubble_classA_jax(T_gamma, T_nu_e, T_nu_x, Sigma_sq, K)
    H_inv_s = H_MeV * _MEV_TO_S
    dTg, dTne, dTnx = coupled_3T_rhs_jax(T_gamma, T_nu_e, T_nu_x, H_MeV=H_MeV)

    f_nue, f_nuebar = extract_monopole_distributions_operator(psi_flat, N_q, n_ell, q_nodes)
    if enable_teff:
        pi_tilde_species = extract_pi_tilde_per_species(psi_flat, N_q, n_ell, q_nodes, q_weights)
        f_nue, f_nuebar = apply_teff_correction_to_monopoles(
            f_nue, f_nuebar, q_nodes, pi_tilde_species[0], pi_tilde_species[1],
            spline_matrix=live_weak_spline_matrix)
    lnp, lpn, _ = compute_live_rates_from_monopoles_level_specialized_jax(
        T_gamma, T_nu_e, tau_n, q_nodes, f_nue, f_nuebar,
        correction_level=correction_level, spline_matrix=live_weak_spline_matrix)

    dX = abundance_rhs_phase2_jax(X, T_gamma, eta, lnp, lpn, rate_table) / jnp.maximum(H_inv_s, 1e-100)

    dy = jnp.zeros_like(y)
    dy = dy.at[_I_SP].set(dSp)
    dy = dy.at[_I_SM].set(dSm)
    dy = dy.at[_I_N1].set(dN1)
    dy = dy.at[_I_N2].set(dN2)
    dy = dy.at[_I_N3].set(dN3)
    dy = dy.at[_I_HIER:i_hier_end].set(dpsi)
    dy = dy.at[i_tg].set(dTg)
    dy = dy.at[i_tne].set(dTne)
    dy = dy.at[i_tnx].set(dTnx)
    dy = dy.at[i_net:i_net + N_SPECIES].set(dX)
    return dy


# ═══════════════════════════════════════════════════════════════
# §5. Config and entry point
# ═══════════════════════════════════════════════════════════════

@dataclass
class JAXClassAConfig:
    """Configuration for JAX Class A Bianchi BBN."""
    bianchi_type: str = "TYPE_I"
    Sigma_H_plus: float = 0.0
    Sigma_H_minus: float = 0.0
    N1_init: float = 0.0
    N2_init: float = 0.0
    N3_init: float = 0.0
    eta: float = 6.104e-10
    tau_n: float = 878.4
    N_eff: float = 3.044
    f_nu: float = 0.40520
    N_q: int = 20
    n_ell: int = 2
    n_reactions: int = 12
    correction_level: int = 0
    enable_teff: bool = False
    transport_mode: str = "auto"  # "auto" | "characteristic" | "linearized"
    T_start: float = 10.0
    T_handoff: float = 0.08
    T_end: float = 0.005
    rtol: float = 1e-8
    atol: float = 1e-10

    def __post_init__(self):
        if bool(self.enable_teff):
            raise ValueError(
                "JAXClassAConfig(enable_teff=True) is deprecated legacy and no "
                "longer has a public forward-solver path. Use the characteristic "
                "or reduced curved-transport paths with enable_teff=False."
            )
        bt = _resolve_bianchi_type(self.bianchi_type)
        n1, n2, n3 = float(self.N1_init), float(self.N2_init), float(self.N3_init)

        def _nz(x):
            return abs(x) > 0.0

        if bt == BianchiType.TYPE_I:
            if _nz(n1) or _nz(n2) or _nz(n3):
                raise ValueError("Type I requires N₁=N₂=N₃=0")
        elif bt == BianchiType.TYPE_II:
            if _nz(n2) or _nz(n3):
                raise ValueError("Type II requires N₂=N₃=0")
        elif bt == BianchiType.TYPE_VI0:
            if _nz(n1):
                raise ValueError("Type VI₀ requires N₁=0")
            if _nz(n2) and _nz(n3) and n2 * n3 >= 0.0:
                raise ValueError("Type VI₀ requires N₂N₃<0 (opposite-sign active structure constants)")
        elif bt == BianchiType.TYPE_VII0:
            if _nz(n1):
                raise ValueError("Type VII₀ requires N₁=0")
            if _nz(n2) and _nz(n3) and n2 * n3 < 0.0:
                raise ValueError("Type VII₀ requires N₂N₃≥0 (same-sign active structure constants)")
        elif bt == BianchiType.TYPE_VIII:
            if _nz(n1) and _nz(n2) and _nz(n3) and (n1 * n2 > 0.0 and n2 * n3 > 0.0):
                raise ValueError("Type VIII requires mixed-sign Nᵢ (not all same sign)")
        elif bt == BianchiType.TYPE_IX:
            if _nz(n1) and _nz(n2) and _nz(n3):
                same_sign = (n1 * n2 > 0.0) and (n2 * n3 > 0.0)
                if not same_sign:
                    raise ValueError("Type IX requires same-sign Nᵢ for closed-curvature canonical data")

        if int(self.N_q) < 20:
            raise ValueError(
                "JAX Class A live-weak runs are currently not validated for N_q<20. "
                "FLRW reduced-control scans show large non-monotone Y_p drift at low N_q "
                "(e.g. N_q=6 -> Y_p~0.261 vs N_q=20 -> Y_p~0.242).")

        type_mask = build_type_mask(bt)
        _, _, _, K_init = _masked_curvature_diagnostics(type_mask, n1, n2, n3)
        sigma_sq = float(self.Sigma_H_plus) ** 2 + float(self.Sigma_H_minus) ** 2
        omega_init = 1.0 - sigma_sq - float(K_init)
        if not np.isfinite(omega_init) or omega_init <= 0.0:
            raise ValueError(f"Unphysical initial data: Ω_init={omega_init} ≤ 0")


@dataclass
class ClassAResult:
    """Result container for Class A full BBN run (phase 1 + phase 2)."""
    Yp: float
    DH: float
    N_eff: float
    Li7H: float = 0.0
    Li6H: float = 0.0
    bianchi_type: str = "TYPE_I"
    success: bool = True
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ClassAPhase1Result:
    """Result of phase-1-only Class A BBN (2-species weak freeze-out).

    This is the first checkpoint on the path to full Class A BBN.
    It captures Xn at freeze-out and the geometry/curvature state,
    WITHOUT running the nuclear network (phase 2).
    """
    Xn_freeze: float
    T_freeze: float
    bianchi_type: str
    success: bool
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def _resolve_bianchi_type(raw: str) -> BianchiType:
    """Accept both member names ('TYPE_I') and values ('I')."""
    try:
        return BianchiType(raw)
    except ValueError:
        return BianchiType[raw]


def _final_state_is_physical(Sp: float, Sm: float, K: float, *, curvature_cap: float = 1e6) -> tuple[bool, str, float]:
    sigma_sq = float(Sp) ** 2 + float(Sm) ** 2
    omega = 1.0 - sigma_sq - float(K)
    if not np.isfinite(K):
        return False, 'nonfinite_curvature', float('nan')
    if abs(float(K)) > curvature_cap:
        return False, 'curvature_runaway', float(omega)
    if not np.isfinite(omega):
        return False, 'nonfinite_omega', float('nan')
    if omega <= 0.0:
        return False, 'omega_nonpositive', float(omega)
    return True, 'ok', float(omega)


def run_classA_phase1_only(config: JAXClassAConfig) -> ClassAPhase1Result:
    """Run ONLY phase 1 (2-species weak freeze-out) for Class A.

    Returns freeze-out neutron fraction and geometry diagnostics.
    No nuclear network, no Y_p/D/H claims.
    """
    from numpy.polynomial.laguerre import laggauss

    btype = _resolve_bianchi_type(config.bianchi_type)
    type_mask = build_type_mask(btype)

    N_q = config.N_q
    n_ell = config.n_ell
    q_nodes_np, q_weights_np = laggauss(N_q)
    q_nodes = jnp.asarray(q_nodes_np, dtype=jnp.float64)

    live_weak_spline_matrix = build_not_a_knot_matrix(q_nodes)

    T_start = config.T_start
    T_nu_init = T_start
    Xn_eq = float(equilibrium_Xn(
        jnp.asarray(T_start), jnp.asarray(T_nu_init), jnp.asarray(config.tau_n)))

    n_phase1 = 2
    _, i_hier_end, i_tg, i_tne, i_tnx, i_net, n_total = _layout(N_q, n_ell, n_phase1)

    N1_init, N2_init, N3_init, _ = _masked_curvature_diagnostics(type_mask, config.N1_init, config.N2_init, config.N3_init)

    y0 = np.zeros(n_total)
    y0[_I_SP] = config.Sigma_H_plus
    y0[_I_SM] = config.Sigma_H_minus
    y0[_I_N1] = N1_init
    y0[_I_N2] = N2_init
    y0[_I_N3] = N3_init
    y0[i_tg] = T_start
    y0[i_tne] = T_nu_init
    y0[i_tnx] = T_nu_init
    y0[i_net] = Xn_eq
    y0[i_net + 1] = 1.0 - Xn_eq

    def rhs_p1(N, y):
        return _classA_rhs_phase1(
            N, y, config.tau_n, config.f_nu,
            q_nodes, jnp.asarray(q_weights_np, dtype=jnp.float64),
            type_mask, live_weak_spline_matrix,
            config.correction_level, config.enable_teff, config.n_ell)

    def event_p1(N, y):
        return y[i_tg] - config.T_handoff

    try:
        sol1 = jax_rodas5p_solve(
            rhs_p1, jnp.array(y0), jnp.array([0.0, 50.0]),
            rtol=config.rtol, atol=config.atol, max_steps=20000,
            event_fn=event_p1, event_refine_steps=5,
            jacobian_reuse_experimental=True, jacobian_refresh_stride=3)
        y_final = np.asarray(sol1.y_final)
        success = True
    except Exception as e:
        return ClassAPhase1Result(
            Xn_freeze=float('nan'), T_freeze=float('nan'),
            bianchi_type=config.bianchi_type, success=False,
            metadata={'error': str(e)})

    Xn_freeze = float(y_final[i_net])
    T_freeze = float(y_final[i_tg])
    Sp_final = float(y_final[_I_SP])
    Sm_final = float(y_final[_I_SM])
    N1_final = float(y_final[_I_N1])
    N2_final = float(y_final[_I_N2])
    N3_final = float(y_final[_I_N3])

    N1_init_m, N2_init_m, N3_init_m, K_init = _masked_curvature_diagnostics(
        type_mask, config.N1_init, config.N2_init, config.N3_init)
    N1_final_m, N2_final_m, N3_final_m, K_final = _masked_curvature_diagnostics(
        type_mask, N1_final, N2_final, N3_final)
    kappa_init = float(effective_kappa_from_curvature(
        N1_init_m, N2_init_m, N3_init_m, type_mask))
    kappa_final = float(effective_kappa_from_curvature(
        N1_final_m, N2_final_m, N3_final_m, type_mask))
    Sigma_sq_final = Sp_final**2 + Sm_final**2
    state_ok, state_reason, Omega_final = _final_state_is_physical(Sp_final, Sm_final, K_final)
    q_final = 1.0 + Sigma_sq_final - float(K_final)
    success = bool(success and state_ok)

    return ClassAPhase1Result(
        Xn_freeze=Xn_freeze,
        T_freeze=T_freeze,
        bianchi_type=config.bianchi_type,
        success=success,
        metadata={
            'backend': 'jax_classA_driver',
            'phase': 'phase1_only',
            'bianchi_type': config.bianchi_type,
            'transport_mode': 'kappa_cascade_lmax2',
            'transport_kappa_init': kappa_init,
            'transport_kappa_final': kappa_final,
            'transport_exact_typeI': kappa_init == 0.0,
            'curvature_K_init': float(K_init),
            'curvature_K_final': float(K_final),
            'Omega_final': Omega_final,
            'q_final': q_final,
            'final_state_ok': state_ok,
            'final_state_reason': state_reason,
            'Sigma_plus_final': Sp_final,
            'Sigma_minus_final': Sm_final,
            'N1_final': N1_final_m,
            'N2_final': N2_final_m,
            'N3_final': N3_final_m,
            'T_nu_e_final': float(y_final[i_tne]),
            'T_nu_x_final': float(y_final[i_tnx]),
            'correction_level': config.correction_level,
            'N_q': config.N_q,
            'n_ell': config.n_ell,
        },
    )


def run_classA_jax(config: JAXClassAConfig) -> ClassAResult:
    """Run Class A BBN in the JAX backend.

    Type I LRS data are routed through the SciPy characteristic production
    driver for reference-exact fidelity.

    Curved Class A types, including LRS Type II, remain on this driver's
    Wainwright-Hsu geometry path so that N_i, K, and S_± actually evolve.
    Their neutrino transport uses the reduced κ-cascade at ℓ_max=2.
    """
    from numpy.polynomial.laguerre import laggauss

    btype = _resolve_bianchi_type(config.bianchi_type)

    # ── Characteristic routing for flat Type I only ──
    use_char = config.transport_mode in ("characteristic", "auto")
    is_type_I = (btype == BianchiType.TYPE_I)

    if use_char and is_type_I:
        from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig
        from rabbit.config.transport_mode import TransportMode as TM
        type_mask = build_type_mask(btype)
        N1_init_m, N2_init_m, N3_init_m, K_init = _masked_curvature_diagnostics(
            type_mask, config.N1_init, config.N2_init, config.N3_init)
        kappa_init = float(effective_kappa_from_curvature(
            N1_init_m, N2_init_m, N3_init_m, type_mask))
        scipy_cfg = FullCoupledConfig(
            Sigma_H_plus=config.Sigma_H_plus,
            Sigma_H_minus=0.0,
            eta=config.eta, tau_n=config.tau_n,
            N_q=config.N_q, n_reactions=config.n_reactions,
            correction_level=config.correction_level,
            transport_mode=TM.CHARACTERISTIC,
            T_start=config.T_start, T_handoff=config.T_handoff, T_end=config.T_end,
        )
        r = run_full_coupled_typeI(scipy_cfg)
        return ClassAResult(
            Yp=r.observables.Yp, DH=r.observables.DH,
            N_eff=r.observables.N_eff, Li7H=r.observables.Li7H,
            Li6H=r.observables.Li6H,
            bianchi_type=config.bianchi_type, success=True,
            metadata={
                'backend': 'jax_classA_driver',
                'phase': 'full_bbn',
                'bianchi_type': config.bianchi_type,
                'correction_level': config.correction_level,
                'n_reactions': config.n_reactions,
                'enable_teff': config.enable_teff,
                'N_q': config.N_q,
                'n_ell': config.n_ell,
                'transport_mode': 'characteristic',
                'transport_note': 'Exact Type-I characteristic production route',
                'transport_kappa_init': kappa_init,
                'transport_kappa_final': 0.0,
                'transport_exact_typeI': True,
                'transport_exact_lrs': True,
                'curvature_K_init': float(K_init),
                'curvature_K_final': 0.0,
                'Sigma_plus_final': float(r.trajectory['Sigma_plus'][-1]),
                'Sigma_minus_final': 0.0,
                'T_final': float(r.trajectory['T_gamma'][-1]),
                'T_nu_e_final': float('nan'),
                'T_nu_x_final': float('nan'),
                'N1_final': 0.0,
                'N2_final': 0.0,
                'N3_final': 0.0,
                'routed_via': 'scipy_production_driver',
                'fidelity': 'ref_exact',
                'production_authority': 'scipy_characteristic_lrs_classA_reduction',
                'geometry_note': 'Flat Type-I reduction; no curved structure constants are active.',
                'correction_level': config.correction_level,
                'wall_time_s': r.wall_time_s,
            },
        )

    # ── JAX κ-cascade path for non-LRS curved types ──

    btype = _resolve_bianchi_type(config.bianchi_type)
    type_mask = build_type_mask(btype)

    N_q = config.N_q
    n_ell = config.n_ell
    q_nodes_np, q_weights_np = laggauss(N_q)
    q_nodes = jnp.asarray(q_nodes_np, dtype=jnp.float64)
    q_weights = jnp.asarray(q_weights_np, dtype=jnp.float64)

    rate_table = load_rate_table(n_reactions=config.n_reactions)
    live_weak_spline_matrix = build_not_a_knot_matrix(q_nodes)

    # Initial conditions
    T_start = config.T_start
    T_nu_init = T_start
    Xn_eq = float(equilibrium_Xn(
        jnp.asarray(T_start), jnp.asarray(T_nu_init), jnp.asarray(config.tau_n)))

    n_phase1 = 2  # Xn, Xp
    _, i_hier_end, i_tg, i_tne, i_tnx, i_net, n_total_p1 = _layout(N_q, n_ell, n_phase1)

    N1_init, N2_init, N3_init, _ = _masked_curvature_diagnostics(type_mask, config.N1_init, config.N2_init, config.N3_init)

    y0 = np.zeros(n_total_p1)
    y0[_I_SP] = config.Sigma_H_plus
    y0[_I_SM] = config.Sigma_H_minus
    y0[_I_N1] = N1_init
    y0[_I_N2] = N2_init
    y0[_I_N3] = N3_init
    y0[i_tg] = T_start
    y0[i_tne] = T_nu_init
    y0[i_tnx] = T_nu_init
    y0[i_net] = Xn_eq
    y0[i_net + 1] = 1.0 - Xn_eq

    # Phase 1 RHS closure
    def rhs_p1(N, y):
        return _classA_rhs_phase1(
            N, y, config.tau_n, config.f_nu,
            q_nodes, q_weights, type_mask, live_weak_spline_matrix,
            config.correction_level, config.enable_teff, config.n_ell)

    # Temperature event for handoff
    def event_p1(N, y):
        return y[i_tg] - config.T_handoff

    # Phase 1 solve
    sol1 = jax_rodas5p_solve(
        rhs_p1, jnp.array(y0), jnp.array([0.0, 50.0]),
        rtol=config.rtol, atol=config.atol, max_steps=20000,
        event_fn=event_p1, event_refine_steps=5,
        jacobian_reuse_experimental=True, jacobian_refresh_stride=3)

    y_handoff = np.asarray(sol1.y_final)

    # Phase 2 IC
    _, _, i_tg_p2, i_tne_p2, i_tnx_p2, i_net_p2, n_total_p2 = _layout(N_q, n_ell, N_SPECIES)
    y0_p2 = np.zeros(n_total_p2)
    y0_p2[:_I_HIER + 6 * n_ell * N_q] = y_handoff[:_I_HIER + 6 * n_ell * N_q]  # geometry + hierarchy
    y0_p2[i_tg_p2] = y_handoff[i_tg]
    y0_p2[i_tne_p2] = y_handoff[i_tne]
    y0_p2[i_tnx_p2] = y_handoff[i_tnx]
    Xn_handoff = y_handoff[i_net]
    X_phase2_init = np.array(phase1_to_phase2_jax(jnp.asarray(Xn_handoff)))
    y0_p2[i_net_p2:i_net_p2 + N_SPECIES] = X_phase2_init

    def rhs_p2(N, y):
        return _classA_rhs_phase2(
            N, y, config.tau_n, config.eta, config.f_nu,
            q_nodes, q_weights, type_mask, rate_table, live_weak_spline_matrix,
            config.correction_level, config.enable_teff, config.n_ell)

    def event_p2(N, y):
        return y[i_tg_p2] - config.T_end

    N_start_p2 = float(sol1.N_final) if hasattr(sol1, 'N_final') else float(sol1.t_final)

    sol2 = jax_rodas5p_solve(
        rhs_p2, jnp.array(y0_p2), jnp.array([N_start_p2, N_start_p2 + 50.0]),
        rtol=config.rtol, atol=config.atol, max_steps=20000,
        event_fn=event_p2, event_refine_steps=5,
        jacobian_reuse_experimental=True, jacobian_refresh_stride=3)

    y_final = np.asarray(sol2.y_final)
    X_final = y_final[i_net_p2:i_net_p2 + N_SPECIES]
    T_final = y_final[i_tg_p2]

    # Extract observables (convention matches driver_typeI.py)
    # Species: 0=n, 1=p, 2=D, 3=T, 4=He3, 5=He4, 6=Li7, 7=Be7, 8=Li6
    Yp = float(X_final[5])  # He4 mass fraction ≈ Y_p
    DH = float(X_final[2] / (2.0 * max(X_final[1], 1e-30)))  # D/H = (X_D/2)/X_p
    Li7H = float((X_final[6] / 7.0 + X_final[7] / 7.0) / max(X_final[1], 1e-30))
    Li6H = float(max(X_final[8], 0.0) / (6.0 * max(X_final[1], 1e-30))) if N_SPECIES > 8 else 0.0

    # N_eff
    T_nu_e_final = y_final[i_tne_p2]
    T_nu_x_final = y_final[i_tnx_p2]
    N_eff_final = float(N_eff_from_3T_jax(
        jnp.asarray(T_final), jnp.asarray(T_nu_e_final), jnp.asarray(T_nu_x_final)))

    # Compute curvature diagnostics at initial conditions
    N1_init_m, N2_init_m, N3_init_m, K_init = _masked_curvature_diagnostics(
        type_mask, config.N1_init, config.N2_init, config.N3_init)
    # ... and at final state
    N1_final_m, N2_final_m, N3_final_m, K_final = _masked_curvature_diagnostics(
        type_mask, y_final[_I_N1], y_final[_I_N2], y_final[_I_N3])
    kappa_init = float(effective_kappa_from_curvature(
        N1_init_m, N2_init_m, N3_init_m, type_mask))
    kappa_final = float(effective_kappa_from_curvature(
        N1_final_m, N2_final_m, N3_final_m, type_mask))
    sigma_sq_init = float(config.Sigma_H_plus) ** 2 + float(config.Sigma_H_minus) ** 2
    omega_init = 1.0 - sigma_sq_init - float(K_init)
    sigma_sq_final = float(y_final[_I_SP]) ** 2 + float(y_final[_I_SM]) ** 2
    q_final = 1.0 + sigma_sq_final - float(K_final)

    state_ok, state_reason, omega_final = _final_state_is_physical(
        float(y_final[_I_SP]), float(y_final[_I_SM]), K_final)

    return ClassAResult(
        Yp=(Yp if state_ok else float('nan')),
        DH=(DH if state_ok else float('nan')),
        N_eff=(N_eff_final if state_ok else float('nan')),
        Li7H=(Li7H if state_ok else float('nan')),
        Li6H=(Li6H if state_ok else float('nan')),
        bianchi_type=config.bianchi_type,
        success=bool(sol1.success and sol2.success and state_ok),
        metadata={
            'backend': 'jax_classA_driver',
            'phase': 'full_bbn',
            'bianchi_type': config.bianchi_type,
            'correction_level': config.correction_level,
            'n_reactions': config.n_reactions,
            'enable_teff': config.enable_teff,
            'N_q': config.N_q,
            'n_ell': config.n_ell,
            'transport_mode': 'kappa_cascade_lmax2',
            'transport_kappa_init': kappa_init,
            'transport_kappa_final': kappa_final,
            'transport_exact_typeI': kappa_init == 0.0,
            'curvature_K_init': float(K_init),
            'curvature_K_final': float(K_final),
            'T_final': float(T_final),
            'T_nu_e_final': float(T_nu_e_final),
            'T_nu_x_final': float(T_nu_x_final),
            'N_eff': N_eff_final,
            'Sigma_plus_final': float(y_final[_I_SP]),
            'Sigma_minus_final': float(y_final[_I_SM]),
            'N1_final': N1_final_m,
            'N2_final': N2_final_m,
            'N3_final': N3_final_m,
            'constraint_residual_note': 'Friedmann residual not monitored in-flight',
            'Omega_init': float(omega_init),
            'Omega_final': float(omega_final),
            'q_final': float(q_final),
            'final_state_ok': bool(state_ok),
            'final_state_reason': state_reason,
            'production_authority': 'candidate_classA_curved_transport',
        },
    )
