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


def monopole_moments(q, w, f):
    q = np.asarray(q, dtype=float)
    w = np.asarray(w, dtype=float)
    f = np.asarray(f, dtype=float)
    return {
        "m0": float(np.sum(w * f)),
        "m1": float(np.sum(w * q * f)),
        "m2": float(np.sum(w * q**2 * f)),
        "m3": float(np.sum(w * q**3 * f)),
        "f_min": float(np.min(f)),
        "f_max": float(np.max(f)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--cl", type=int, default=0)
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--collisions", action="store_true")
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
        enable_collisions=bool(args.collisions),
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
        enable_collisions=cfg.enable_collisions,
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
    S_h = float(y[i_S])

    mu_h = mod.mu_current(X0, signs, S_h)
    Pi_plus_h = mod.char_extract_stress(I_h, J_h, mu_h, w0, cfg.f_nu)
    f_nue_h = mod.char_extract_monopole(I_h, J_h, w0, q_gl)
    f_nuebar_h = f_nue_h.copy()

    if cfg.tier >= 2:
        T_nu_h = float(y[i_tne])
    else:
        T_nu_h = float(mod.T_nu_from_T_gamma_tier1(y[i_tg]))

    weak_h = mod.compute_live_weak_rates(
        f_nue_h, f_nuebar_h, q_gl,
        float(y[i_tg]), T_nu_h, cfg.tau_n,
        compute_iso_reference=False,
        correction_level=cfg.correction_level,
    )

    out = {
        "config": {
            "Sigma_H": args.sigma,
            "correction_level": args.cl,
            "tier": args.tier,
            "collisions": bool(args.collisions),
            "N_q": args.nq,
            "N_mu": args.nmu,
            "n_reactions": args.reactions,
        },
        "phase1_handoff_N": float(sol1.t[-1]),
        "phase1_handoff_T": float(y[i_tg]),
        "phase1_handoff_Xn": float(y[i_net]),
        "phase1_handoff_sigma_plus": float(y[mod._I_SP]),
        "phase1_handoff_sigma_minus": float(y[mod._I_SM]),
        "phase1_handoff_pi_plus": float(Pi_plus_h),
        "phase1_handoff_lambda_np": float(weak_h.lambda_np),
        "phase1_handoff_lambda_pn": float(weak_h.lambda_pn),
        "phase1_handoff_I0": float(getattr(weak_h, "I0", np.nan)),
        "phase1_handoff_monopole_probe": monopole_moments(q_gl, q_wt, f_nue_h),
    }

    print(json.dumps(out, indent=2, sort_keys=True))
    if args.out:
        pp = Path(args.out)
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"[saved] {pp}")


if __name__ == "__main__":
    main()
