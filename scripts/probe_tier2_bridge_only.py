#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import roots_laguerre

from rabbit.drivers import full_coupled_typeI as mod
from rabbit.config.transport_mode import TransportMode
from rabbit.transport.teff_collision_bridge import apply_gather_scatter_collision
from rabbit.thermo.incomplete_decoupling import (
    compute_energy_exchange_rate,
    entropy_density_plasma,
    entropy_ratio_S,
)
from rabbit.thermo.nudec_coupled import hubble_3T, total_energy_transfer


def _jsonify_numeric(x):
    if np.isscalar(x):
        return float(x)
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    return [float(v) for v in arr.ravel()]


def _sum_numeric(x):
    if np.isscalar(x):
        return float(x)
    arr = np.asarray(x, dtype=np.float64)
    return float(np.sum(arr))


def _finite_fraction(x):
    arr = np.asarray(x, dtype=np.float64)
    return float(np.mean(np.isfinite(arr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--cl", type=int, default=0)
    ap.add_argument("--tier", type=int, default=2)
    ap.add_argument("--nq", type=int, default=20)
    ap.add_argument("--nmu", type=int, default=12)
    ap.add_argument("--reactions", type=int, default=12)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    cfg = mod.FullCoupledConfig(
        Sigma_H_plus=args.sigma,
        Sigma_H_minus=0.0,
        correction_level=args.cl,
        tier=args.tier,
        enable_collisions=False,   # 중요: ODE에는 collision feedback 안 넣음
        N_q=args.nq,
        N_mu=args.nmu,
        n_reactions=args.reactions,
        enable_teff=False,
        transport_mode=TransportMode.CHARACTERISTIC,
    )

    (n_transport, i_hier_end, i_tg, i_tne, i_tnx, i_net, n_total,
     i_I, i_J, i_S) = mod._layout_characteristic(cfg.N_mu, tier=cfg.tier)

    mu0, w0, X0, signs = mod.setup_ray_grid(cfg.N_mu)
    q_gl_np, q_wgl_np = roots_laguerre(cfg.N_q)
    q_gl = q_gl_np.astype(np.float64)
    q_wt = q_wgl_np.astype(np.float64)
    f_eq = 1.0 / (np.exp(np.minimum(q_gl, 500.0)) + 1.0)

    char_cache = {
        "mu0": mu0,
        "w0": w0,
        "X0": X0,
        "signs": signs,
        "q_gl": q_gl,
        "q_wt": q_wt,
    }

    init_weak = mod.compute_live_weak_rates(
        f_eq, f_eq, q_gl,
        cfg.T_start, cfg.T_start, cfg.tau_n,
        compute_iso_reference=False,
    )
    Xn_eq = init_weak.equilibrium_Xn

    y0 = np.zeros(n_total)
    y0[mod._I_SP] = cfg.Sigma_H_plus
    y0[mod._I_SM] = cfg.Sigma_H_minus
    y0[i_J:i_J + cfg.N_mu] = 1.0
    y0[i_tg] = cfg.T_start
    if cfg.tier >= 2:
        y0[i_tne] = cfg.T_start
        y0[i_tnx] = cfg.T_start
    y0[i_net] = Xn_eq
    y0[i_net + 1] = 1.0 - Xn_eq

    rhs_kw = dict(
        grid=cfg.grid,
        multipole=cfg.multipole,
        tau_n=cfg.tau_n,
        eta=cfg.eta,
        N_eff=cfg.N_eff,
        f_nu=cfg.f_nu,
        tier=cfg.tier,
        enable_teff=cfg.enable_teff,
        enable_collisions=False,   # 중요
        n_reactions=cfg.n_reactions,
        correction_level=cfg.correction_level,
        transport_mode=TransportMode.CHARACTERISTIC,
        N_mu=cfg.N_mu,
        _char_cache=char_cache,
    )

    def stop_p1(N, y):
        return y[i_tg] - cfg.T_handoff
    stop_p1.terminal = True
    stop_p1.direction = -1

    sol1 = solve_ivp(
        fun=lambda N, y: mod.coupled_rhs(N, y, phase=1, **rhs_kw),
        t_span=[0.0, 50.0],
        y0=y0,
        events=stop_p1,
        **cfg.solver.to_scipy_kwargs(),
    )
    if (not sol1.success) and len(sol1.t) < 2:
        raise RuntimeError(sol1.message)

    y = sol1.y[:, -1].copy()
    I_h = y[i_I:i_I + cfg.N_mu]
    J_h = y[i_J:i_J + cfg.N_mu]

    T_gamma = float(y[i_tg])
    T_nu_e = float(y[i_tne])
    T_nu_x = float(y[i_tnx])
    Sigma_sq = float(y[mod._I_SP]**2 + y[mod._I_SM]**2)
    H_inv_sec = float(hubble_3T(T_gamma, T_nu_e, T_nu_x, Sigma_sq=Sigma_sq) * mod._MEV_TO_S)

    gs = apply_gather_scatter_collision(
        I_h, J_h, w0, q_gl, q_wt,
        T_gamma, T_nu_e, H_inv_sec
    )

    rho_ref_terms = q_wt * np.exp(np.minimum(q_gl, 80.0)) * q_gl**3 * np.asarray(gs.f_monopole, dtype=np.float64)
    ex_rate = compute_energy_exchange_rate(gs.C_monopole, q_gl, q_wt, T_nu_e)

    tet = total_energy_transfer(T_gamma, T_nu_e, T_nu_x)

    out = {
        "config": {
            "Sigma_H": args.sigma,
            "correction_level": args.cl,
            "tier": args.tier,
            "N_q": args.nq,
            "N_mu": args.nmu,
            "n_reactions": args.reactions,
        },
        "handoff_state": {
            "N": float(sol1.t[-1]),
            "T_gamma": T_gamma,
            "T_nu_e": T_nu_e,
            "T_nu_x": T_nu_x,
            "sigma_plus": float(y[mod._I_SP]),
            "sigma_minus": float(y[mod._I_SM]),
        },
        "bridge": {
            "tangency_D2": float(gs.tangency_D2),
            "delta_rho_nu": float(gs.delta_rho_nu),
            "max_abs_delta_I": float(np.max(np.abs(gs.delta_I))),
            "max_abs_C_monopole": float(np.max(np.abs(gs.C_monopole))),
            "energy_exchange_rate": float(ex_rate),
            "total_energy_transfer_raw": _jsonify_numeric(tet),
            "total_energy_transfer_sum": _sum_numeric(tet),
        },
        "thermo": {
            "entropy_density_plasma": float(entropy_density_plasma(T_gamma)),
            "entropy_ratio_S": float(entropy_ratio_S(T_gamma)),
        },
        "numerics": {
            "q_max": float(np.max(q_gl)),
            "exp_qmax_capped": float(np.exp(min(float(np.max(q_gl)), 80.0))),
            "rho_ref_term_max": float(np.max(rho_ref_terms)),
            "rho_ref_term_finite_fraction": _finite_fraction(rho_ref_terms),
            "f_monopole_min": float(np.min(gs.f_monopole)),
            "f_monopole_max": float(np.max(gs.f_monopole)),
            "C_monopole_finite_fraction": _finite_fraction(gs.C_monopole),
            "delta_I_finite_fraction": _finite_fraction(gs.delta_I),
        },
    }

    print(json.dumps(out, indent=2, sort_keys=True))
    if args.out:
        pp = Path(args.out)
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"[saved] {pp}")


if __name__ == "__main__":
    main()
