#!/usr/bin/env python3
"""Characteristic ray transport for LRS Bianchi Type I — SciPy + JAX.

Exact collisionless Boltzmann solution via characteristic curves:
    f(q, μ, N) = f₀(q exp(2I(N, μ₀)))

State per ray: (I_j, J_j) where
    dI_j/dN = Σ P₂(μ_j(S))
    dJ_j/dN = 3Σ(1-3μ_j²)J_j
    dS/dN   = Σ

μ_j(S) recovered analytically: μ²/(1-μ²) = [μ₀²/(1-μ₀²)] exp(6S).

Stress: Π₊ = f_ν Σ_j w_j J_j P₂(μ_j) exp(-8I_j)
Monopole: f_mono(q) = (1/2) Σ_j w_j J_j f₀(q exp(2I_j))

No ℓ-truncation, no Ψ-linearization, automatic positivity.
"""
import sys, os, time, json, warnings
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import eval_legendre, roots_laguerre
from numpy.polynomial.legendre import leggauss

from rabbit.geometry.typeI import compute_typeI_geometry_rhs
from rabbit.thermo.incomplete_decoupling import dT_gamma_dN_tier1, T_nu_from_T_gamma_tier1
from rabbit.thermo.eos_photon_electron import _RHO_GAMMA_PREFACTOR
from rabbit.weak.live_rates import compute_live_weak_rates
from rabbit.network.abundances_standard import (
    abundance_rhs_phase1, abundance_rhs_phase2, phase1_to_phase2, N_SPECIES)

OUT_DIR = Path(__file__).parent.parent / "audit_outputs"
OUT_DIR.mkdir(exist_ok=True)

_TAU_N = 879.6; _ETA = 6.104e-10; _N_EFF = 3.044; _F_NU = 0.40520
_MEV_TO_S = 1.0 / 6.58212e-22; _G_N = 6.70883e-45; B_Q = 8.0 / 15.0


def _hubble(Tg, Tnu, S2):
    rho = _RHO_GAMMA_PREFACTOR * Tg**4 * (1 + _N_EFF * 7/8 * (Tnu/Tg)**4)
    return np.sqrt(max(8*np.pi*_G_N/3 * rho / max(1-S2, 1e-10), 0))


# ═══════════════════════════════════════════════════════════════════════
# §1. Grids
# ═══════════════════════════════════════════════════════════════════════

def setup_grids(N_mu=12):
    mu0, w0 = leggauss(N_mu)
    mu0 = mu0.astype(np.float64); w0 = w0.astype(np.float64)
    X0 = mu0**2 / np.maximum(1 - mu0**2, 1e-30)  # precompute for forward map
    q_gl, q_wgl = roots_laguerre(20)
    q_gl = q_gl.astype(np.float64); q_wgl = q_wgl.astype(np.float64)
    f0_gl = 1.0 / (np.exp(np.minimum(q_gl, 500)) + 1)
    return mu0, w0, X0, q_gl, q_wgl, f0_gl


# ═══════════════════════════════════════════════════════════════════════
# §2. Characteristic map
# ═══════════════════════════════════════════════════════════════════════

def mu_from_S(X0, S):
    """Forward map: μ_j(S) from precomputed X0_j = μ₀²/(1-μ₀²)."""
    X = X0 * np.exp(6 * S)
    return np.sqrt(np.minimum(X / (1 + X), 1 - 1e-15))  # |μ|, restore sign after


def mu_signed(mu0, X0, S):
    return np.sign(mu0) * mu_from_S(X0, S)


# ═══════════════════════════════════════════════════════════════════════
# §3. Observables
# ═══════════════════════════════════════════════════════════════════════

def char_stress(I, J, mu, w0, f_nu):
    """Π₊ = f_ν × Σ w_j J_j P₂(μ_j) exp(-8I_j)."""
    P2 = eval_legendre(2, mu)
    return f_nu * np.sum(w0 * J * P2 * np.exp(-8 * I))


def char_monopole(I, J, w0, q_gl, f0_gl):
    """f_mono(q) = (1/2) Σ_j w_j J_j f₀(q exp(2I_j))."""
    alpha = np.exp(2 * I)  # (N_mu,)
    f_mono = np.zeros(len(q_gl))
    for k in range(len(q_gl)):
        qa = q_gl[k] * alpha
        f_vals = 1.0 / (np.exp(np.minimum(qa, 500)) + 1)
        f_mono[k] = 0.5 * np.sum(w0 * J * f_vals)
    return f_mono


# ═══════════════════════════════════════════════════════════════════════
# §4. SciPy driver
# ═══════════════════════════════════════════════════════════════════════

def run_scipy(Sigma_H, N_mu=12, mode='characteristic',
              T_start=10.0, T_handoff=0.08, T_end=0.005,
              rtol=1e-8, atol=1e-10):
    """Run BBN with SciPy Radau.

    mode: 'characteristic' or 'linearized'
    """
    mu0, w0, X0, q_gl, q_wgl, f0_gl = setup_grids(N_mu)

    if mode == 'characteristic':
        # State: [Σ₊, I₁..I_N, J₁..J_N, S, T_γ, X_n, X_p]
        n_I = N_mu; n_J = N_mu
        I_SP = 0; I_I = 1; I_J = 1+n_I; I_S = 1+n_I+n_J
        I_TG = I_S+1; I_NET = I_TG+1
        n_transport = 2*N_mu + 1  # I + J + S
    else:
        # State: [Σ₊, Ψ₂, T_γ, X_n, X_p]
        I_SP = 0; I_PSI = 1; I_TG = 2; I_NET = 3
        n_transport = 1

    def rhs(N, y, phase):
        Sp = y[I_SP]
        Tg = y[I_TG] if mode == 'characteristic' else y[I_TG]
        n_net = 2 if phase == 1 else N_SPECIES
        X = y[I_NET:I_NET+n_net]
        if Tg < 1e-6:
            return np.zeros(len(y))
        S2 = Sp**2

        if mode == 'characteristic':
            I_vals = y[I_I:I_I+n_I]
            J_vals = y[I_J:I_J+n_J]
            S_val = y[I_S]
            mu = mu_signed(mu0, X0, S_val)
            P2_mu = eval_legendre(2, mu)

            # Stress
            Pi = char_stress(I_vals, J_vals, mu, w0, _F_NU)

            # Transport RHS
            dI = Sp * P2_mu
            dJ = 3 * Sp * (1 - 3*mu**2) * J_vals
            dS = Sp

            # Weak rates from monopole
            f_mono = char_monopole(I_vals, J_vals, w0, q_gl, f0_gl)
            f_mono = np.clip(f_mono, 0, 1)
        else:
            Psi2 = y[I_PSI]
            Pi = 6 * _F_NU * Psi2
            f_mono = f0_gl.copy()

        # Geometry
        Om = max(0, 1-S2)
        dSp, _ = compute_typeI_geometry_rhs(Sp, 0., Pi, 0., Om)

        # Thermo + Hubble
        dTg = dT_gamma_dN_tier1(Tg)
        Tnu = T_nu_from_T_gamma_tier1(Tg)
        H = _hubble(Tg, Tnu, S2) * _MEV_TO_S

        # Weak rates
        weak = compute_live_weak_rates(f_mono, f_mono, q_gl, Tg, Tnu, _TAU_N,
                                        compute_iso_reference=False, correction_level=0)

        # Network
        if phase == 1:
            dX = np.zeros(n_net)
            dX[0] = abundance_rhs_phase1(X[0], weak.lambda_np, weak.lambda_pn)/max(H,1e-100)
            dX[1] = -dX[0]
        else:
            dX = abundance_rhs_phase2(X, Tg, _ETA, weak.lambda_np, weak.lambda_pn,
                                       n_reactions=12)/max(H,1e-100)

        dy = np.zeros(len(y))
        dy[I_SP] = dSp
        if mode == 'characteristic':
            dy[I_I:I_I+n_I] = dI
            dy[I_J:I_J+n_J] = dJ
            dy[I_S] = dS
        else:
            dy[I_PSI] = -B_Q * Sp
        dy[I_TG] = dTg
        dy[I_NET:I_NET+n_net] = dX
        return dy

    # IC
    Tnu0 = T_nu_from_T_gamma_tier1(T_start)
    w0_wr = compute_live_weak_rates(f0_gl, f0_gl, q_gl, T_start, Tnu0, _TAU_N,
                                     compute_iso_reference=False, correction_level=0)
    Xn_eq = float(w0_wr.lambda_pn / max(w0_wr.lambda_np + w0_wr.lambda_pn, 1e-100))

    n_total_p1 = I_NET + 2
    y0 = np.zeros(n_total_p1)
    y0[I_SP] = Sigma_H
    if mode == 'characteristic':
        y0[I_J:I_J+n_J] = 1.0  # J = 1 initially
    y0[I_TG] = T_start
    y0[I_NET] = Xn_eq; y0[I_NET+1] = 1-Xn_eq

    def ev1(N, y):
        return y[I_TG] - T_handoff
    ev1.terminal = True; ev1.direction = -1

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s1 = solve_ivp(lambda N,y: rhs(N,y,1), [0,50], y0, method='Radau',
                        events=ev1, rtol=rtol, atol=atol, max_step=0.5)

    if s1.status < 0:
        return {'success': False, 'error': f'P1: {s1.message}', 'mode': mode}

    yh = s1.y[:,-1]; Nh = s1.t[-1]

    # Diagnostics at handoff
    diag = {}
    if mode == 'characteristic':
        I_h = yh[I_I:I_I+n_I]; J_h = yh[I_J:I_J+n_J]; S_h = yh[I_S]
        mu_h = mu_signed(mu0, X0, S_h)
        diag['S'] = float(S_h)
        diag['max_I'] = float(np.max(np.abs(I_h)))
        diag['J_range'] = [float(np.min(J_h)), float(np.max(J_h))]
        diag['mu_range'] = [float(np.min(mu_h)), float(np.max(mu_h))]
        f_mono_h = char_monopole(I_h, J_h, w0, q_gl, f0_gl)
        diag['mono_range'] = [float(np.min(f_mono_h)), float(np.max(f_mono_h))]
        diag['delta_f_max'] = float(np.max(np.abs(f_mono_h - f0_gl)))
        Pi_h = char_stress(I_h, J_h, mu_h, w0, _F_NU)
        diag['Pi_handoff'] = float(Pi_h)
    else:
        diag['Psi2'] = float(yh[I_PSI])
        diag['Pi_handoff'] = float(6*_F_NU*yh[I_PSI])

    # Phase 2
    n_total_p2 = I_NET + N_SPECIES
    Xp2 = phase1_to_phase2(yh[I_NET])
    y2 = np.zeros(n_total_p2)
    y2[:I_NET] = yh[:I_NET]
    y2[I_TG] = yh[I_TG]
    y2[I_NET:I_NET+N_SPECIES] = Xp2

    def ev2(N, y):
        return y[I_TG] - T_end
    ev2.terminal = True; ev2.direction = -1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s2 = solve_ivp(lambda N,y: rhs(N,y,2), [Nh,Nh+30], y2, method='Radau',
                        events=ev2, rtol=rtol, atol=atol, max_step=0.5)

    elapsed = time.perf_counter() - t0
    if s2.status < 0:
        return {'success': False, 'error': f'P2: {s2.message}', 'mode': mode}

    yf = s2.y[:,-1]
    Xf = yf[I_NET:I_NET+N_SPECIES]
    return {
        'success': True, 'mode': mode,
        'Yp': float(Xf[5]), 'DH': float(Xf[2]/(2*max(Xf[1],1e-30))),
        'Xn_freeze': float(yh[I_NET]), 'Sigma_final': float(yf[I_SP]),
        'elapsed_s': round(elapsed, 2),
        'steps': len(s1.t) + len(s2.t),
        'transport_dof': n_transport,
        'diagnostics': diag,
        'config': {'Sigma_H': Sigma_H, 'N_mu': N_mu, 'mode': mode, 'backend': 'scipy'},
    }


# ═══════════════════════════════════════════════════════════════════════
# §5. JAX driver
# ═══════════════════════════════════════════════════════════════════════

def run_jax(Sigma_H, N_mu=12, mode='characteristic',
            T_start=10.0, T_handoff=0.08, T_end=0.005,
            rtol=1e-8, atol=1e-10):
    """Run BBN with JAX Rodas5P."""
    import os
    os.environ.setdefault('JAX_ENABLE_X64', '1')
    import jax; jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rabbit.jax.solver_jax_rodas5p import jax_rodas5p_solve
    from rabbit.jax.thermo_provider_jax import (
        tier1_T_nu_from_T_gamma_jax, tier1_dT_gamma_dN_jax)
    from rabbit.jax.weak_live_jax import compute_live_born_rates_from_monopoles

    mu0_np, w0_np, X0_np, q_np, qw_np, f0_np = setup_grids(N_mu)
    mu0 = jnp.array(mu0_np); w0 = jnp.array(w0_np); X0 = jnp.array(X0_np)
    q_gl = jnp.array(q_np); f0_gl = jnp.array(f0_np)
    signs = jnp.sign(mu0)

    def _hubble_j(Tg, Tnu, S2):
        rho = _RHO_GAMMA_PREFACTOR*Tg**4*(1+_N_EFF*7/8*(Tnu/Tg)**4)
        return jnp.sqrt(jnp.maximum(8*jnp.pi*_G_N/3*rho/jnp.maximum(1-S2,1e-10),0.))

    def _mu_from_S_j(S):
        X = X0 * jnp.exp(6*S)
        return signs * jnp.sqrt(jnp.minimum(X/(1+X), 1-1e-15))

    def _P2_j(mu):
        return 0.5*(3*mu**2 - 1)

    def _stress_j(I, J, mu):
        return _F_NU * jnp.sum(w0 * J * _P2_j(mu) * jnp.exp(-8*I))

    def _monopole_j(I, J):
        alpha = jnp.exp(2*I)  # (N_mu,)
        # f_mono(q_k) = (1/2) Σ_j w_j J_j / (exp(q_k α_j) + 1)
        qa = q_gl[:,None] * alpha[None,:]  # (N_q, N_mu)
        f_vals = 1.0 / (jnp.exp(jnp.minimum(qa, 500)) + 1)
        return 0.5 * f_vals @ (w0 * J)

    if mode == 'characteristic':
        # State: [Σ₊, I₁..I_N, J₁..J_N, S, T_γ, X_n, X_p]
        n_I = N_mu; n_J = N_mu
        i_I = 1; i_J = 1+n_I; i_S = 1+n_I+n_J
        i_TG = i_S+1; i_NET = i_TG+1

        def rhs_char(N, y):
            Sp = y[0]; Iv = y[i_I:i_I+n_I]; Jv = y[i_J:i_J+n_J]; Sv = y[i_S]
            Tg = y[i_TG]; Xn = y[i_NET]
            S2 = Sp**2; mu = _mu_from_S_j(Sv); P2mu = _P2_j(mu)
            Pi = _stress_j(Iv, Jv, mu)
            Om = jnp.maximum(0., 1-S2)
            q_dec = 1+S2; dSp = -(2-q_dec)*Sp + Pi
            dI = Sp * P2mu; dJ = 3*Sp*(1-3*mu**2)*Jv; dS = Sp
            dTg = tier1_dT_gamma_dN_jax(Tg); Tnu = tier1_T_nu_from_T_gamma_jax(Tg)
            H = _hubble_j(Tg, Tnu, S2) * _MEV_TO_S
            fm = jnp.clip(_monopole_j(Iv, Jv), 0, 1)
            lnp, lpn = compute_live_born_rates_from_monopoles(
                Tg, Tnu, jnp.float64(_TAU_N), q_gl, fm, fm, None)
            dXn = (-lnp*Xn + lpn*(1-Xn))/jnp.maximum(H, 1e-100)
            return jnp.concatenate([jnp.array([dSp]), dI, dJ, jnp.array([dS, dTg, dXn, -dXn])])

        Tnu0 = tier1_T_nu_from_T_gamma_jax(jnp.float64(T_start))
        lnp0, lpn0 = compute_live_born_rates_from_monopoles(
            jnp.float64(T_start), Tnu0, jnp.float64(_TAU_N), q_gl, f0_gl, f0_gl, None)
        Xn_eq = float(lpn0/jnp.maximum(lnp0+lpn0, 1e-100))

        y0 = jnp.zeros(i_NET+2)
        y0 = y0.at[0].set(Sigma_H)
        y0 = y0.at[i_J:i_J+n_J].set(1.0)
        y0 = y0.at[i_TG].set(T_start)
        y0 = y0.at[i_NET].set(Xn_eq)
        y0 = y0.at[i_NET+1].set(1-Xn_eq)

        rhs_fn = rhs_char
        tg_idx = i_TG
    else:
        # State: [Σ₊, Ψ₂, T_γ, X_n, X_p]
        i_TG = 2; i_NET = 3

        def rhs_lin(N, y):
            Sp = y[0]; Psi2 = y[1]; Tg = y[2]; Xn = y[3]
            S2 = Sp**2; Pi = 6*_F_NU*Psi2
            q_dec = 1+S2; dSp = -(2-q_dec)*Sp + Pi
            dPsi2 = -B_Q*Sp
            dTg = tier1_dT_gamma_dN_jax(Tg); Tnu = tier1_T_nu_from_T_gamma_jax(Tg)
            H = _hubble_j(Tg, Tnu, S2)*_MEV_TO_S
            lnp, lpn = compute_live_born_rates_from_monopoles(
                Tg, Tnu, jnp.float64(_TAU_N), q_gl, f0_gl, f0_gl, None)
            dXn = (-lnp*Xn + lpn*(1-Xn))/jnp.maximum(H,1e-100)
            return jnp.array([dSp, dPsi2, dTg, dXn, -dXn])

        Tnu0 = tier1_T_nu_from_T_gamma_jax(jnp.float64(T_start))
        lnp0, lpn0 = compute_live_born_rates_from_monopoles(
            jnp.float64(T_start), Tnu0, jnp.float64(_TAU_N), q_gl, f0_gl, f0_gl, None)
        Xn_eq = float(lpn0/jnp.maximum(lnp0+lpn0, 1e-100))

        y0 = jnp.array([Sigma_H, 0.0, T_start, Xn_eq, 1-Xn_eq])
        rhs_fn = rhs_lin
        tg_idx = i_TG

    def event_fn(N, y):
        return y[tg_idx] - T_handoff

    t0 = time.perf_counter()
    try:
        result = jax_rodas5p_solve(
            rhs_fn, y0, jnp.array([0.0, 50.0]),
            rtol=rtol, atol=atol, max_steps=5000,
            event_fn=event_fn, event_refine_steps=24)
        elapsed_p1 = time.perf_counter() - t0
        if not result.success:
            return {'success': False, 'error': 'P1: Rodas5P failed',
                    'mode': mode, 'config': {'Sigma_H': Sigma_H, 'backend': 'jax'}}
        yh = np.array(result.y_final)
    except Exception as e:
        return {'success': False, 'error': f'P1: {e}', 'mode': mode,
                'config': {'Sigma_H': Sigma_H, 'backend': 'jax'}}

    # Phase 2 (simplified: use SciPy for phase 2 since JAX event handling is complex)
    # Extract handoff state and run phase 2 with SciPy
    Xn_h = float(yh[i_NET])
    diag = {}
    if mode == 'characteristic':
        I_h = yh[i_I:i_I+N_mu]; J_h = yh[i_J:i_J+N_mu]; S_h = float(yh[i_S])
        mu_h = np.sign(mu0_np) * np.sqrt(np.minimum(X0_np*np.exp(6*S_h)/(1+X0_np*np.exp(6*S_h)), 1-1e-15))
        diag['S'] = S_h; diag['max_I'] = float(np.max(np.abs(I_h)))
        diag['J_range'] = [float(np.min(J_h)), float(np.max(J_h))]
        Pi_h = float(_F_NU * np.sum(w0_np * J_h * eval_legendre(2, mu_h) * np.exp(-8*I_h)))
        diag['Pi_handoff'] = Pi_h
    else:
        diag['Psi2'] = float(yh[1])
        diag['Pi_handoff'] = float(6*_F_NU*yh[1])

    # Phase 2 via SciPy (reuse the scipy RHS)
    r_p2 = _run_phase2_scipy(yh, mode, N_mu, Xn_h, float(yh[tg_idx]), mu0_np, w0_np, X0_np, q_np, f0_np)
    elapsed = time.perf_counter() - t0

    if not r_p2['success']:
        return {'success': False, 'error': r_p2['error'], 'mode': mode}

    return {
        'success': True, 'mode': mode,
        'Yp': r_p2['Yp'], 'DH': r_p2['DH'],
        'Xn_freeze': Xn_h, 'Sigma_final': r_p2['Sigma_final'],
        'elapsed_s': round(elapsed, 2),
        'steps_p1': int(result.n_steps),
        'transport_dof': 2*N_mu+1 if mode=='characteristic' else 1,
        'diagnostics': diag,
        'config': {'Sigma_H': Sigma_H, 'N_mu': N_mu, 'mode': mode, 'backend': 'jax'},
    }


def _run_phase2_scipy(yh, mode, N_mu, Xn_h, Tg_h, mu0, w0, X0, q_gl, f0_gl):
    """Phase 2 helper (SciPy) for JAX driver."""
    n_I = N_mu; n_J = N_mu
    if mode == 'characteristic':
        i_I=1; i_J=1+n_I; i_S=1+n_I+n_J; i_TG=i_S+1; i_NET=i_TG+1
    else:
        i_TG=2; i_NET=3

    Xp2 = phase1_to_phase2(Xn_h)
    n_total = i_NET + N_SPECIES
    y2 = np.zeros(n_total)
    y2[:i_NET] = yh[:i_NET]
    y2[i_TG] = Tg_h
    y2[i_NET:i_NET+N_SPECIES] = Xp2

    def rhs2(N, y):
        Sp = y[0]; Tg = y[i_TG]; S2 = Sp**2; X = y[i_NET:i_NET+N_SPECIES]
        if Tg < 1e-6: return np.zeros(len(y))

        if mode == 'characteristic':
            Iv=y[i_I:i_I+n_I]; Jv=y[i_J:i_J+n_J]; Sv=y[i_S]
            mu=np.sign(mu0)*np.sqrt(np.minimum(X0*np.exp(6*Sv)/(1+X0*np.exp(6*Sv)),1-1e-15))
            Pi = _F_NU*np.sum(w0*Jv*eval_legendre(2,mu)*np.exp(-8*Iv))
            P2mu = eval_legendre(2,mu)
            dI=Sp*P2mu; dJ=3*Sp*(1-3*mu**2)*Jv; dS=Sp
            fm = np.clip(char_monopole(Iv, Jv, w0, q_gl, f0_gl), 0, 1)
        else:
            Psi2=y[1]; Pi=6*_F_NU*Psi2; fm=f0_gl.copy()

        Om=max(0,1-S2)
        dSp,_=compute_typeI_geometry_rhs(Sp,0.,Pi,0.,Om)
        dTg=dT_gamma_dN_tier1(Tg); Tnu=T_nu_from_T_gamma_tier1(Tg)
        H=_hubble(Tg,Tnu,S2)*_MEV_TO_S
        weak=compute_live_weak_rates(fm,fm,q_gl,Tg,Tnu,_TAU_N,compute_iso_reference=False,correction_level=0)
        dX=abundance_rhs_phase2(X,Tg,_ETA,weak.lambda_np,weak.lambda_pn,n_reactions=12)/max(H,1e-100)

        dy=np.zeros(len(y)); dy[0]=dSp
        if mode=='characteristic':
            dy[i_I:i_I+n_I]=dI; dy[i_J:i_J+n_J]=dJ; dy[i_S]=dS
        else:
            dy[1]=-B_Q*Sp
        dy[i_TG]=dTg; dy[i_NET:i_NET+N_SPECIES]=dX
        return dy

    def ev2(N,y): return y[i_TG]-0.005
    ev2.terminal=True; ev2.direction=-1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s2=solve_ivp(rhs2,[0,30],y2,method='Radau',events=ev2,rtol=1e-8,atol=1e-10,max_step=0.5)
    if s2.status<0:
        return {'success':False,'error':f'P2: {s2.message}'}
    yf=s2.y[:,-1]; Xf=yf[i_NET:i_NET+N_SPECIES]
    return {'success':True,'Yp':float(Xf[5]),'DH':float(Xf[2]/(2*max(Xf[1],1e-30))),
            'Sigma_final':float(yf[0])}


# ═══════════════════════════════════════════════════════════════════════
# §6. Ablation study
# ═══════════════════════════════════════════════════════════════════════

def _configure_platform(force_cpu: bool = False):
    """Auto-detect GPU, fall back to CPU. Must be called before any JAX import."""
    import os
    os.environ['JAX_ENABLE_X64'] = '1'
    if force_cpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'
        return 'cpu'
    # Let JAX auto-detect: gpu > tpu > cpu
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        devices = jax.devices()
        platform = devices[0].platform if devices else 'cpu'
        return platform
    except Exception:
        os.environ['JAX_PLATFORMS'] = 'cpu'
        return 'cpu'


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Characteristic ray transport ablation study (SciPy + JAX)")
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if GPU is available')
    parser.add_argument('--scipy-only', action='store_true',
                        help='Run SciPy backend only (skip JAX)')
    parser.add_argument('--jax-only', action='store_true',
                        help='Run JAX backend only (skip SciPy)')
    parser.add_argument('--sigma', type=float, nargs='+', default=[0.0, 0.1, 0.3, 0.5],
                        help='Sigma values to test (default: 0.0 0.1 0.3 0.5)')
    parser.add_argument('--nmu', type=int, default=12,
                        help='Number of ray quadrature points (default: 12)')
    args = parser.parse_args()

    # Platform configuration
    platform = _configure_platform(force_cpu=args.cpu)

    # Backend selection: JAX is default, SciPy is fallback/comparison
    if args.scipy_only:
        backends = ['scipy']
    elif args.jax_only:
        backends = ['jax']
    else:
        backends = ['jax', 'scipy']  # JAX first (default)

    print("="*70)
    print("  Characteristic Ray Transport — Ablation Study")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Platform: {platform}  |  Backends: {', '.join(backends)}  |  N_μ: {args.nmu}")
    print("="*70)

    results = {"generated": datetime.now(timezone.utc).isoformat(),
               "platform": platform, "runs": []}
    SIGMAS = args.sigma

    for sigma in SIGMAS:
        print(f"\n{'='*60}")
        print(f"  Σ = {sigma}")
        print(f"{'='*60}")

        for backend in backends:
            for mode in ['linearized', 'characteristic']:
                tag = f"S{sigma}_{backend}_{mode}"
                print(f"  [{tag}] ", end="", flush=True)

                try:
                    if backend == 'scipy':
                        r = run_scipy(sigma, N_mu=args.nmu, mode=mode)
                    else:
                        r = run_jax(sigma, N_mu=args.nmu, mode=mode)

                    if r['success']:
                        d = r.get('diagnostics', {})
                        extra = ""
                        if mode == 'characteristic':
                            extra = f"  S={d.get('S',0):.3f}  |I|={d.get('max_I',0):.3e}  Pi={d.get('Pi_handoff',0):.3e}"
                        else:
                            extra = f"  Ψ₂={d.get('Psi2',0):.4e}  Pi={d.get('Pi_handoff',0):.3e}"
                        print(f"Yp={r['Yp']:.10f}  DOF={r['transport_dof']}{extra}  ({r['elapsed_s']}s)")
                    else:
                        print(f"FAILED: {r.get('error', '?')}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    r = {'success': False, 'error': str(e), 'mode': mode,
                         'config': {'Sigma_H': sigma, 'backend': backend}}

                results['runs'].append(r)

    # ── Analysis ──
    print(f"\n{'='*70}")
    print("  COMPARISON: ΔYp(characteristic - linearized)")
    print(f"{'='*70}")
    print(f"  {'Σ':>5s}  {'Backend':>7s}  {'ΔYp':>14s}  {'ΔYp/ΔYp_aniso':>14s}")

    # Find FLRW baseline
    r_flrw = next((r for r in results['runs']
                    if r.get('config',{}).get('Sigma_H')==0.0
                    and r.get('mode')=='linearized' and r.get('success')), None)
    Yp_flrw = r_flrw['Yp'] if r_flrw else 0

    for sigma in SIGMAS:
        for backend in backends:
            rl = next((r for r in results['runs']
                       if r.get('config',{}).get('Sigma_H')==sigma
                       and r.get('config',{}).get('backend')==backend
                       and r.get('mode')=='linearized' and r.get('success')), None)
            rc = next((r for r in results['runs']
                       if r.get('config',{}).get('Sigma_H')==sigma
                       and r.get('config',{}).get('backend')==backend
                       and r.get('mode')=='characteristic' and r.get('success')), None)
            if rl and rc:
                dYp = rc['Yp'] - rl['Yp']
                dYp_aniso = rl['Yp'] - Yp_flrw
                rel = dYp/dYp_aniso if abs(dYp_aniso) > 1e-10 else 0
                print(f"  {sigma:5.1f}  {backend:>7s}  {dYp:+14.6e}  {rel:+14.4f}")
            else:
                status = "char FAIL" if not rc else ("lin FAIL" if not rl else "?")
                print(f"  {sigma:5.1f}  {backend:>7s}  {status}")

    # Write
    json_path = OUT_DIR / "CHARACTERISTIC_ABLATION_RESULTS.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON → {json_path}")


if __name__ == "__main__":
    main()
