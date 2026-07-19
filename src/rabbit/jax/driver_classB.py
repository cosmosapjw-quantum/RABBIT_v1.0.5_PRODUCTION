"""
rabbit.jax.driver_classB — Class B Bianchi BBN driver (Phase-1 freeze-out).

Extends the BBN solver to Class B types by adding the frame variable A
to the state vector and modifying the Hubble rate.

State: [Σ₊, Σ₋, N₁, N₂, N₃, A, Ψ_flat(...), T_γ, Xn, Xp]

Key difference from Class A:
  - One extra state variable: A (frame variable)
  - Modified Friedmann: Ω = 1 − Σ² − K − cA²
  - Modified q: q = 1 + Σ² − K − cA²
  - dA/dN = (q + 2Σ₊) A
  - Transport: SAME as Class A (A does not enter transport operator)

Currently: tier-1 thermo (single T_ν), Born weak rates, N_q=6 smoke.
Target: Type V (A only) and Type IV (A + N₁) as first validation points.
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
    structure_eigenvalues, frame_variable_rhs, SQRT3,
)
from rabbit.jax.geometry_classB_jax import (
    build_classB_type_mask, get_c_factor,
)
from rabbit.jax.rhs_classA import classA_transport_rhs, effective_kappa_from_curvature
from rabbit.jax.transport_ops_jax import extract_aniso_stress_operator
from rabbit.jax.thermo_provider_jax import tier1_T_nu_from_T_gamma_jax, tier1_dT_gamma_dN_jax
from rabbit.jax.thermo_jax import rho_plasma, rho_photon, PI, G_N
from rabbit.jax.weak_jax import compute_born_rates, equilibrium_Xn
from rabbit.jax.classB_live_weak import compute_classB_cl_rates
from rabbit.jax.network_jax import (
    load_rate_table, abundance_rhs_phase1_jax, abundance_rhs_phase2_jax,
    phase1_to_phase2_jax, N_SPECIES, validate_rate_table_window_jax,
)
from rabbit.validation.truncation_guards import (
    validate_general_initial_budget, validate_manual_network_truncation,
    validate_min_resolution, validate_phase_temperature_order, warn_reduced_ell,
)
from rabbit.jax.solver_jax_rodas5p import jax_rodas5p_solve

_MEV_TO_S = 1.519267447e21
_TAU_N = 878.4
_ETA = 6.104e-10
_N_EFF = 3.044
_F_NU = 0.40520


def _classB_h_parameter(bt: BianchiType, explicit_h: float | None) -> float | None:
    if bt == BianchiType.TYPE_III:
        return -1.0
    if bt == BianchiType.TYPE_VI_M19:
        return -1.0 / 9.0
    if bt in (BianchiType.TYPE_VIH, BianchiType.TYPE_VIIH):
        return float(BianchiSpec.from_type(bt, h=explicit_h).h)
    return None


def _masked_classB_N(
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


@dataclass
class JAXClassBConfig:
    bianchi_type: str = "TYPE_V"
    Sigma_H_plus: float = 0.0
    Sigma_H_minus: float = 0.0
    N1_init: float = 0.0
    N2_init: float = 0.0
    N3_init: float = 0.0
    h: float | None = None
    A_init: float = 0.05
    N_q: int = 6
    n_ell: int = 2
    correction_level: int = 0
    n_reactions: int = 12
    T_start: float = 10.0
    T_handoff: float = 0.08
    T_end: float = 0.005
    tau_n: float = _TAU_N
    eta: float = _ETA
    rtol: float = 1e-8
    atol: float = 1e-10

    def __post_init__(self):
        bt = self.bianchi_type if isinstance(self.bianchi_type, BianchiType) else None
        if bt is None:
            try:
                bt = BianchiType(self.bianchi_type)
            except ValueError:
                bt = BianchiType[self.bianchi_type]
        if bt not in (BianchiType.TYPE_VIH, BianchiType.TYPE_VIIH) and self.h is not None:
            BianchiSpec.from_type(bt, h=self.h)
        n1, n2, n3 = float(self.N1_init), float(self.N2_init), float(self.N3_init)
        if bt == BianchiType.TYPE_V:
            if any(abs(x) > 0.0 for x in (n1, n2, n3)):
                raise ValueError("Type V requires N₁=N₂=N₃=0; A is the only geometric degree of freedom")
        elif bt == BianchiType.TYPE_IV:
            if abs(n2) > 0.0 or abs(n3) > 0.0:
                raise ValueError("Type IV requires N₂=N₃=0; only N₁ is active in the reduced mask implementation")
        elif bt in (BianchiType.TYPE_III, BianchiType.TYPE_VIH, BianchiType.TYPE_VIIH, BianchiType.TYPE_VI_M19):
            if abs(n1) > 0.0:
                raise ValueError("This reduced Class B implementation deactivates N₁ for III/VI_h/VII_h/VI_{-1/9}; use N₂/N₃ instead")
            if abs(n2) == 0.0 and abs(n3) == 0.0:
                raise ValueError("Reduced Class B III/VI_h/VII_h/VI_{-1/9} requires at least one active curvature coordinate among N₂,N₃")
            if bt == BianchiType.TYPE_III:
                if abs(n2) == 0.0 or abs(n3) == 0.0 or n2 * n3 >= 0.0:
                    raise ValueError("Type III requires canonical h=-1 active data with N₂N₃<0")
                scale = max(abs(n2), abs(n3), 1.0e-300)
                if abs(n2 + n3) / scale > 1.0e-12:
                    raise ValueError("Type III requires the canonical h=-1 relation N₂=-N₃")
            if bt == BianchiType.TYPE_VI_M19:
                if abs(n2) == 0.0 or abs(n3) == 0.0 or n2 * n3 >= 0.0:
                    raise ValueError("Type VI_{-1/9} requires canonical h=-1/9 active data with N₂N₃<0")
                scale = max(abs(n2), 1.0e-300)
                if abs(n3 + n2 / 9.0) / scale > 1.0e-12:
                    raise ValueError("Type VI_{-1/9} requires the canonical h=-1/9 relation N₃=-N₂/9")
            if bt in (BianchiType.TYPE_VIH, BianchiType.TYPE_VIIH):
                spec = BianchiSpec.from_type(bt, h=self.h)
                if abs(n2) == 0.0 or abs(n3) == 0.0:
                    raise ValueError(f"{bt.name} requires active h-family data with nonzero N₂ and N₃")
                scale = max(abs(n2), 1.0e-300)
                if abs(n3 - float(spec.h) * n2) / scale > 1.0e-12:
                    raise ValueError(f"{bt.name} requires the h-family relation N₃=h*N₂")

        validate_phase_temperature_order(self.T_start, self.T_handoff, self.T_end, context="JAX Class-B temperature schedule")
        validate_min_resolution("N_q", self.N_q, minimum=2, strict=True)
        validate_min_resolution("n_ell", self.n_ell, minimum=2, strict=True)
        warn_reduced_ell("JAX Class-B reduced PSTF", self.n_ell, expected=2)
        validate_manual_network_truncation(self.n_reactions, context="JAX Class-B network")
        validate_rate_table_window_jax(self.T_handoff, load_rate_table(self.n_reactions), context="JAX Class-B phase-2 handoff", strict=True)
        validate_rate_table_window_jax(self.T_end, load_rate_table(self.n_reactions), context="JAX Class-B phase-2 end", strict=True)
        type_mask = build_classB_type_mask(bt)
        masked = type_mask * jnp.array([self.N1_init, self.N2_init, self.N3_init], dtype=jnp.float64)
        K_init = float(gauss_curvature_K(masked[0], masked[1], masked[2]))
        sigma_sq = float(self.Sigma_H_plus) ** 2 + float(self.Sigma_H_minus) ** 2
        cA_sq = float(get_c_factor(bt)) * float(self.A_init) ** 2
        validate_general_initial_budget(sigma_sq, K_init, cA_sq, context="JAX Class-B initial data")


@dataclass
class JAXClassBResult:
    Yp: float
    DH: float
    Li7H: float = 0.0
    N_eff: float = float('nan')
    Xn_freeze: float = float('nan')
    success: bool = True
    metadata: dict = None


def _classB_state_ok(Sp: float, Sm: float, K: float, cA_sq: float, *, cap: float = 1e6) -> tuple[bool, str, float]:
    sigma_sq = float(Sp) ** 2 + float(Sm) ** 2
    omega = float(compute_Omega(sigma_sq, float(K), float(cA_sq)))
    if not np.isfinite(K):
        return False, 'nonfinite_curvature', float('nan')
    if abs(float(K)) > cap:
        return False, 'curvature_runaway', omega
    if not np.isfinite(omega):
        return False, 'nonfinite_omega', float('nan')
    if omega <= 0.0:
        return False, 'omega_nonpositive', omega
    return True, 'ok', omega


def run_classB_phase1(config: JAXClassBConfig):
    """Run Phase-1 (n↔p freeze-out) for Class B types.

    Phase η-3 lifted CL1/CL2 via ``compute_classB_cl_rates``;
    Phase η-4 lifted CL3 via the same helper. correction_level > 3
    is rejected because the in-tree weak ladder only defines CL0–CL3.
    """
    if config.correction_level not in (0, 1, 2, 3):
        raise ValueError(
            f"correction_level={config.correction_level} not in {{0, 1, 2, 3}}; "
            f"CL > 3 is not defined in the in-tree weak ladder."
        )
    bt = config.bianchi_type if isinstance(config.bianchi_type, BianchiType) else None
    if bt is None:
        # Try direct enum value first, then by name
        try:
            bt = BianchiType(config.bianchi_type)
        except ValueError:
            bt = BianchiType[config.bianchi_type]
    type_mask = build_classB_type_mask(bt)
    c_factor = get_c_factor(bt)
    h_parameter = _classB_h_parameter(bt, config.h)

    from numpy.polynomial.laguerre import laggauss
    q_nodes_np, q_weights_np = laggauss(config.N_q)
    q_nodes = jnp.array(q_nodes_np, dtype=jnp.float64)
    q_weights = jnp.array(q_weights_np, dtype=jnp.float64)
    f_nu = jnp.array(_F_NU, dtype=jnp.float64)

    N_q = config.N_q
    n_ell = config.n_ell
    n_transport = 6 * n_ell * N_q

    # State: [Σ₊, Σ₋, N₁, N₂, N₃, A, Ψ(transport), T_γ, Xn, Xp]
    I_SP, I_SM, I_N1, I_N2, I_N3, I_A = 0, 1, 2, 3, 4, 5
    I_HIER = 6
    i_hier_end = I_HIER + n_transport
    i_tg = i_hier_end
    i_net = i_tg + 1
    n_total = i_net + 2

    # Initial conditions
    T0 = config.T_start
    T_nu0 = T0
    Xn_eq = float(equilibrium_Xn(jnp.array(T0), jnp.array(T_nu0), jnp.array(config.tau_n)))

    N_init = _masked_classB_N(type_mask, [config.N1_init, config.N2_init, config.N3_init], h_parameter)

    y0 = np.zeros(n_total)
    y0[I_SP] = config.Sigma_H_plus
    y0[I_SM] = config.Sigma_H_minus
    y0[I_N1] = float(N_init[0])
    y0[I_N2] = float(N_init[1])
    y0[I_N3] = float(N_init[2])
    y0[I_A] = config.A_init
    y0[i_tg] = T0
    y0[i_net] = Xn_eq
    y0[i_net + 1] = 1.0 - Xn_eq

    def rhs_p1(N, y):
        Sp = y[I_SP]; Sm = y[I_SM]
        N1 = y[I_N1]; N2 = y[I_N2]; N3 = y[I_N3]
        A = y[I_A]
        psi = y[I_HIER:i_hier_end]
        Tg = y[i_tg]
        X = y[i_net:i_net + 2]

        # Masked curvature
        Nm = _masked_classB_N(type_mask, [N1, N2, N3], h_parameter)
        N1m, N2m, N3m = Nm[0], Nm[1], Nm[2]
        K = gauss_curvature_K(N1m, N2m, N3m)
        S_plus, S_minus = curvature_sources(N1m, N2m, N3m)

        Sigma_sq = Sp**2 + Sm**2
        cA_sq = c_factor * A**2
        q = compute_q(Sigma_sq, K, cA_sq)

        # Transport (same as Class A — A does not enter)
        kappa = effective_kappa_from_curvature(N1m, N2m, N3m, jnp.ones(3))
        dpsi = classA_transport_rhs(Sp, Sm, psi, n_ell=n_ell, n_species=6, kappa=kappa)

        # Stress extraction → geometry
        pi_plus, pi_minus = extract_aniso_stress_operator(psi, N_q, n_ell, q_nodes, q_weights, f_nu)

        # Geometry RHS
        damping = -(2.0 - q)
        dSp = damping * Sp - S_plus + pi_plus
        dSm = damping * Sm - S_minus + pi_minus

        lam1, lam2, lam3 = structure_eigenvalues(q, Sp, Sm)
        dN1, dN2, dN3 = _structure_rhs_h_locked(
            lam1, lam2, lam3, N1m, N2m, N3m, h_parameter)

        dA = frame_variable_rhs(A, q, Sp)

        # Thermo + Hubble (modified Friedmann: Ω = 1-Σ²-K-cA²)
        T_nu = tier1_T_nu_from_T_gamma_jax(Tg)
        dTg = tier1_dT_gamma_dN_jax(Tg)
        rho_em = rho_plasma(Tg)
        rho_nu = _N_EFF * (7.0 / 8.0) * rho_photon(T_nu)
        Omega = compute_Omega(Sigma_sq, K, cA_sq)
        H_sq = 8.0 * PI / 3.0 * G_N * (rho_em + rho_nu) / Omega
        H_inv_s = jnp.where((Omega > 0.0) & (H_sq > 0.0), jnp.sqrt(H_sq), jnp.nan) * _MEV_TO_S

        # Weak rates: Born for CL0; live-weak helper for CL1/CL2 (Phase η-3).
        if int(config.correction_level) == 0:
            lnp, lpn = compute_born_rates(Tg, T_nu, jnp.array(config.tau_n))
        else:
            lnp, lpn = compute_classB_cl_rates(
                Tg, T_nu, jnp.array(config.tau_n),
                correction_level=int(config.correction_level),
                N_q=int(config.N_q),
            )
        dXn = abundance_rhs_phase1_jax(X[0], lnp, lpn) / jnp.maximum(H_inv_s, 1e-100)

        dy = jnp.zeros_like(y)
        dy = dy.at[I_SP].set(dSp)
        dy = dy.at[I_SM].set(dSm)
        dy = dy.at[I_N1].set(dN1)
        dy = dy.at[I_N2].set(dN2)
        dy = dy.at[I_N3].set(dN3)
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

    yf = np.asarray(sol1.y_final)
    result = JAXClassBResult(
        Yp=float('nan'), DH=float('nan'),
        Xn_freeze=float(yf[i_net]),
        success=bool(sol1.success),
        metadata={
            'backend': 'jax_classB_driver',
            'phase': 'phase1',
            'bianchi_type': config.bianchi_type,
            'c_factor': c_factor,
            'h_parameter': h_parameter,
            'Sigma_plus_final': float(yf[I_SP]),
            'Sigma_minus_final': float(yf[I_SM]),
            'A_final': float(yf[I_A]),
            'N1_final': float(yf[I_N1]),
            'N2_final': float(yf[I_N2]),
            'N3_final': float(yf[I_N3]),
            'T_freeze': float(yf[i_tg]),
            'N_q': N_q,
            'A_init': config.A_init,
        },
    )
    layout = (I_SP, I_SM, I_N1, I_N2, I_N3, I_A, I_HIER, i_hier_end, i_tg, i_net, n_total)
    return result, sol1, layout


def run_classB_jax(config: JAXClassBConfig) -> JAXClassBResult:
    """Run full BBN (Phase-1 + Phase-2) for Class B types.

    Phase-1: n↔p freeze-out down to T_handoff
    Phase-2: nucleosynthesis down to T_end
    """
    bt = config.bianchi_type if isinstance(config.bianchi_type, BianchiType) else None
    if bt is None:
        try:
            bt = BianchiType(config.bianchi_type)
        except ValueError:
            bt = BianchiType[config.bianchi_type]
    type_mask = build_classB_type_mask(bt)
    c_factor = get_c_factor(bt)
    h_parameter = _classB_h_parameter(bt, config.h)

    from numpy.polynomial.laguerre import laggauss
    q_nodes_np, q_weights_np = laggauss(config.N_q)
    q_nodes = jnp.array(q_nodes_np, dtype=jnp.float64)
    q_weights = jnp.array(q_weights_np, dtype=jnp.float64)
    f_nu = jnp.array(_F_NU, dtype=jnp.float64)
    rate_table = load_rate_table(n_reactions=config.n_reactions)

    N_q = config.N_q
    n_ell = config.n_ell
    n_transport = 6 * n_ell * N_q

    # Phase-1
    p1_result, sol1, layout = run_classB_phase1(config)
    if not p1_result.success:
        return JAXClassBResult(
            Yp=float('nan'), DH=float('nan'), success=False,
            metadata={**p1_result.metadata, 'phase': 'phase1_failed'})

    I_SP, I_SM, I_N1, I_N2, I_N3, I_A, I_HIER, i_hier_end, i_tg, _, _ = layout

    # Phase-2 state: replace 2-species with N_SPECIES
    i_net_p2 = i_tg + 1
    n_total_p2 = i_net_p2 + N_SPECIES

    y_ho = np.asarray(sol1.y_final)
    y0_p2 = np.zeros(n_total_p2)
    y0_p2[:i_net_p2] = y_ho[:i_net_p2]  # geometry + transport + T_γ
    X_p2 = np.array(phase1_to_phase2_jax(jnp.array(y_ho[i_tg + 1])))  # Xn → 9-species
    y0_p2[i_net_p2:i_net_p2 + N_SPECIES] = X_p2

    def rhs_p2(N, y):
        Sp = y[I_SP]; Sm = y[I_SM]
        N1 = y[I_N1]; N2 = y[I_N2]; N3 = y[I_N3]
        A = y[I_A]
        psi = y[I_HIER:i_hier_end]
        Tg = y[i_tg]
        X = y[i_net_p2:i_net_p2 + N_SPECIES]

        Nm = _masked_classB_N(type_mask, [N1, N2, N3], h_parameter)
        N1m, N2m, N3m = Nm[0], Nm[1], Nm[2]
        K = gauss_curvature_K(N1m, N2m, N3m)
        S_plus, S_minus = curvature_sources(N1m, N2m, N3m)

        Sigma_sq = Sp**2 + Sm**2
        cA_sq = c_factor * A**2
        q = compute_q(Sigma_sq, K, cA_sq)

        kappa = effective_kappa_from_curvature(N1m, N2m, N3m, jnp.ones(3))
        dpsi = classA_transport_rhs(Sp, Sm, psi, n_ell=n_ell, n_species=6, kappa=kappa)

        pi_plus, pi_minus = extract_aniso_stress_operator(psi, N_q, n_ell, q_nodes, q_weights, f_nu)

        damping = -(2.0 - q)
        dSp = damping * Sp - S_plus + pi_plus
        dSm = damping * Sm - S_minus + pi_minus

        lam1, lam2, lam3 = structure_eigenvalues(q, Sp, Sm)
        dN1, dN2, dN3 = _structure_rhs_h_locked(
            lam1, lam2, lam3, N1m, N2m, N3m, h_parameter)
        dA = frame_variable_rhs(A, q, Sp)

        T_nu = tier1_T_nu_from_T_gamma_jax(Tg)
        dTg = tier1_dT_gamma_dN_jax(Tg)
        rho_em = rho_plasma(Tg)
        rho_nu = _N_EFF * (7.0 / 8.0) * rho_photon(T_nu)
        Omega = compute_Omega(Sigma_sq, K, cA_sq)
        H_sq = 8.0 * PI / 3.0 * G_N * (rho_em + rho_nu) / Omega
        H_inv_s = jnp.where((Omega > 0.0) & (H_sq > 0.0), jnp.sqrt(H_sq), jnp.nan) * _MEV_TO_S

        # Phase η-3: same CL dispatch as Phase 1.
        if int(config.correction_level) == 0:
            lnp, lpn = compute_born_rates(Tg, T_nu, jnp.array(config.tau_n))
        else:
            lnp, lpn = compute_classB_cl_rates(
                Tg, T_nu, jnp.array(config.tau_n),
                correction_level=int(config.correction_level),
                N_q=int(config.N_q),
            )
        dX = abundance_rhs_phase2_jax(X, Tg, config.eta, lnp, lpn, rate_table) \
             / jnp.maximum(H_inv_s, 1e-100)

        dy = jnp.zeros_like(y)
        dy = dy.at[I_SP].set(dSp)
        dy = dy.at[I_SM].set(dSm)
        dy = dy.at[I_N1].set(dN1)
        dy = dy.at[I_N2].set(dN2)
        dy = dy.at[I_N3].set(dN3)
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

    masked_init = _masked_classB_N(type_mask, [config.N1_init, config.N2_init, config.N3_init], h_parameter)
    K_init = float(gauss_curvature_K(*masked_init))
    masked_final = _masked_classB_N(type_mask, [yf[I_N1], yf[I_N2], yf[I_N3]], h_parameter)
    K_final = float(gauss_curvature_K(*masked_final))
    cA_sq_init = float(c_factor) * float(config.A_init) ** 2
    cA_sq_final = float(c_factor) * float(yf[I_A]) ** 2
    sigma_sq_init = float(config.Sigma_H_plus) ** 2 + float(config.Sigma_H_minus) ** 2
    omega_init = float(compute_Omega(sigma_sq_init, K_init, cA_sq_init))
    state_ok, state_reason, omega_final = _classB_state_ok(
        float(yf[I_SP]), float(yf[I_SM]), K_final, cA_sq_final)
    q_final = float(compute_q(
        float(yf[I_SP]) ** 2 + float(yf[I_SM]) ** 2,
        K_final,
        cA_sq_final,
    ))
    N_eff_final = float(_N_EFF)

    return JAXClassBResult(
        Yp=(Yp if state_ok else float('nan')), DH=(DH if state_ok else float('nan')), Li7H=(Li7H if state_ok else float('nan')),
        N_eff=(N_eff_final if state_ok else float('nan')),
        Xn_freeze=p1_result.Xn_freeze,
        success=bool(sol1.success and sol2.success and state_ok),
        metadata={
            'backend': 'jax_classB_driver',
            'phase': 'full_bbn',
            'bianchi_type': config.bianchi_type,
            'c_factor': c_factor,
            'h_parameter': h_parameter,
            'correction_level': config.correction_level,
            'n_reactions': config.n_reactions,
            'N_q': N_q,
            'n_ell': n_ell,
            'transport_mode': 'kappa_cascade_lmax2',
            'transport_kappa_init': float(effective_kappa_from_curvature(
                *_masked_classB_N(type_mask, [config.N1_init, config.N2_init, config.N3_init], h_parameter),
                jnp.ones(3))),
            'transport_kappa_final': float(effective_kappa_from_curvature(
                *masked_final,
                jnp.ones(3))),
            'curvature_K_init': K_init,
            'curvature_K_final': K_final,
            'frame_cA_sq_init': cA_sq_init,
            'frame_cA_sq_final': cA_sq_final,
            'Sigma_plus_final': float(yf[I_SP]),
            'Sigma_minus_final': float(yf[I_SM]),
            'A_init': config.A_init,
            'N1_init': float(masked_init[0]),
            'N2_init': float(masked_init[1]),
            'N3_init': float(masked_init[2]),
            'A_final': float(yf[I_A]),
            'N1_final': float(yf[I_N1]),
            'N2_final': float(yf[I_N2]),
            'N3_final': float(yf[I_N3]),
            'T_final': float(yf[i_tg]),
            'N_eff': N_eff_final,
            'Omega_init': omega_init,
            'Omega_final': float(omega_final),
            'q_final': q_final,
            'final_state_ok': bool(state_ok),
            'final_state_reason': state_reason,
        },
    )
