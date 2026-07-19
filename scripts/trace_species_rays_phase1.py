#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import roots_laguerre

from rabbit.drivers import full_coupled_typeI as mod
from rabbit.config.transport_mode import TransportMode
from rabbit.transport.teff_collision_bridge import apply_gather_scatter_collision
from rabbit.transport.species_tagged_bridge import apply_species_tagged_bridge
from rabbit.thermo.nudec_coupled import coupled_3T_rhs, hubble_3T


SPECIES = ("nue", "nuebar", "nux")


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


def species_energy_weights(f_nu_total: float, T_nu_e: float, T_nu_x: float):
    # ν_e, \barν_e, and one effective ν_x bank representing 4 states
    raw = {
        "nue": 1.0 * T_nu_e**4,
        "nuebar": 1.0 * T_nu_e**4,
        "nux": 4.0 * T_nu_x**4,
    }
    norm = sum(raw.values())
    return {k: f_nu_total * raw[k] / norm for k in raw}


def layout_species(N_mu: int):
    idx = {}
    p = 0
    idx["sp"] = p; p += 1
    idx["sm"] = p; p += 1
    idx["S"] = p; p += 1

    for sp in SPECIES:
        idx[f"I_{sp}"] = slice(p, p + N_mu); p += N_mu
        idx[f"J_{sp}"] = slice(p, p + N_mu); p += N_mu

    idx["Tg"] = p; p += 1
    idx["Tne"] = p; p += 1
    idx["Tnx"] = p; p += 1
    idx["Xn"] = p; p += 1
    idx["Xp"] = p; p += 1
    idx["n_total"] = p
    return idx


def get_species_temperature(sp: str, T_nu_e: float, T_nu_x: float) -> float:
    if sp in ("nue", "nuebar"):
        return float(T_nu_e)
    return float(T_nu_x)


def rhs_phase1_species(
    N,
    y,
    *,
    cfg,
    idx,
    q_gl,
    q_wt,
    mu0,
    w0,
    X0,
    signs,
):
    dydN = np.zeros_like(y)

    Sigma_plus = float(y[idx["sp"]])
    Sigma_minus = float(y[idx["sm"]])
    S_val = float(y[idx["S"]])

    T_gamma = float(y[idx["Tg"]])
    T_nu_e = float(y[idx["Tne"]])
    T_nu_x = float(y[idx["Tnx"]])
    Xn = float(y[idx["Xn"]])

    if T_gamma < 1e-6:
        return dydN

    Sigma_sq = Sigma_plus**2 + Sigma_minus**2
    mu = mod.mu_current(X0, signs, S_val)

    # species-split anisotropic stress
    fweights = species_energy_weights(cfg.f_nu, T_nu_e, T_nu_x)
    Pi_plus_total = 0.0
    for sp in SPECIES:
        I_sp = y[idx[f"I_{sp}"]]
        J_sp = y[idx[f"J_{sp}"]]
        Pi_plus_total += mod.char_extract_stress(I_sp, J_sp, mu, w0, fweights[sp])

    Omega = mod.enforce_positive_typeI_omega(
        Sigma_sq,
        context="trace_species_rays_phase1",
        strict=True,
        floor=0.0,
    )
    dSp, dSm = mod.compute_typeI_geometry_rhs(
        Sigma_plus, Sigma_minus, Pi_plus_total, 0.0, Omega
    )
    dydN[idx["sp"]] = dSp
    dydN[idx["sm"]] = dSm

    # one shared ray-geometry variable S
    dS_shared = None

    # Hubble / thermo
    dTg, dTne, dTnx = coupled_3T_rhs(T_gamma, T_nu_e, T_nu_x)
    H = float(hubble_3T(T_gamma, T_nu_e, T_nu_x, Sigma_sq=Sigma_sq) * mod._MEV_TO_S)

    dydN[idx["Tg"]] = dTg
    dydN[idx["Tne"]] = dTne
    dydN[idx["Tnx"]] = dTnx

    # species ray transport + optional collisions
    for sp in SPECIES:
        I_sp = y[idx[f"I_{sp}"]]
        J_sp = y[idx[f"J_{sp}"]]

        dI, dJ, dS = mod.characteristic_transport_rhs(Sigma_plus, I_sp, J_sp, mu)
        if dS_shared is None:
            dS_shared = float(dS)

        if cfg.enable_collisions and cfg.tier >= 2:
            gs = apply_species_tagged_bridge(
                species=sp,
                I=I_sp,
                J=J_sp,
                w0=w0,
                q_nodes=q_gl,
                q_weights=q_wt,
                T_gamma=T_gamma,
                T_nu_e=T_nu_e,
                T_nu_x=T_nu_x,
                H=H,
            )
            dI = dI + gs.delta_I

        dydN[idx[f"I_{sp}"]] = dI
        dydN[idx[f"J_{sp}"]] = dJ

    dydN[idx["S"]] = 0.0 if dS_shared is None else dS_shared

    # weak rates now use ν_e and \barν_e separately
    f_nue = mod.char_extract_monopole(
        y[idx["I_nue"]], y[idx["J_nue"]], w0, q_gl
    )
    f_nuebar = mod.char_extract_monopole(
        y[idx["I_nuebar"]], y[idx["J_nuebar"]], w0, q_gl
    )

    weak = mod.compute_live_weak_rates(
        f_nue, f_nuebar, q_gl,
        T_gamma, T_nu_e, cfg.tau_n,
        compute_iso_reference=False,
        correction_level=cfg.correction_level,
    )

    dXn = (weak.lambda_pn * (1.0 - Xn) - weak.lambda_np * Xn) / H
    dydN[idx["Xn"]] = dXn
    dydN[idx["Xp"]] = -dXn

    return dydN


def run_species_phase1(cfg):
    assert cfg.transport_mode == TransportMode.CHARACTERISTIC
    assert cfg.tier >= 2

    idx = layout_species(cfg.N_mu)

    mu0, w0, X0, signs = mod.setup_ray_grid(cfg.N_mu)
    q_gl_np, q_wgl_np = roots_laguerre(cfg.N_q)
    q_gl = q_gl_np.astype(np.float64)
    q_wt = q_wgl_np.astype(np.float64)

    f_eq = 1.0 / (np.exp(np.minimum(q_gl, 500.0)) + 1.0)
    init_weak = mod.compute_live_weak_rates(
        f_eq, f_eq, q_gl,
        cfg.T_start, cfg.T_start, cfg.tau_n,
        compute_iso_reference=False,
    )
    Xn_eq = float(init_weak.equilibrium_Xn)

    y0 = np.zeros(idx["n_total"], dtype=np.float64)
    y0[idx["sp"]] = cfg.Sigma_H_plus
    y0[idx["sm"]] = cfg.Sigma_H_minus
    y0[idx["S"]] = 0.0

    for sp in SPECIES:
        y0[idx[f"J_{sp}"]] = 1.0

    y0[idx["Tg"]] = cfg.T_start
    y0[idx["Tne"]] = cfg.T_start
    y0[idx["Tnx"]] = cfg.T_start
    y0[idx["Xn"]] = Xn_eq
    y0[idx["Xp"]] = 1.0 - Xn_eq

    def stop_p1(N, y):
        return y[idx["Tg"]] - cfg.T_handoff
    stop_p1.terminal = True
    stop_p1.direction = -1

    sol = solve_ivp(
        fun=lambda N, y: rhs_phase1_species(
            N, y,
            cfg=cfg, idx=idx,
            q_gl=q_gl, q_wt=q_wt,
            mu0=mu0, w0=w0, X0=X0, signs=signs,
        ),
        t_span=[0.0, 50.0],
        y0=y0,
        events=stop_p1,
        **cfg.solver.to_scipy_kwargs(),
    )

    if (not sol.success) and len(sol.t) < 2:
        raise RuntimeError(sol.message)

    y = sol.y[:, -1].copy()

    T_gamma = float(y[idx["Tg"]])
    T_nu_e = float(y[idx["Tne"]])
    T_nu_x = float(y[idx["Tnx"]])

    mu = mod.mu_current(X0, signs, float(y[idx["S"]]))
    fweights = species_energy_weights(cfg.f_nu, T_nu_e, T_nu_x)

    pi_species = {}
    mono_species = {}
    bridge_debug = {}
    for sp in SPECIES:
        I_sp = y[idx[f"I_{sp}"]]
        J_sp = y[idx[f"J_{sp}"]]
        pi_species[sp] = float(mod.char_extract_stress(I_sp, J_sp, mu, w0, fweights[sp]))
        mono = mod.char_extract_monopole(I_sp, J_sp, w0, q_gl)
        mono_species[sp] = monopole_moments(q_gl, q_wt, mono)

        if cfg.enable_collisions and cfg.tier >= 2:
            gs_dbg = apply_species_tagged_bridge(
                species=sp,
                I=I_sp,
                J=J_sp,
                w0=w0,
                q_nodes=q_gl,
                q_weights=q_wt,
                T_gamma=T_gamma,
                T_nu_e=T_nu_e,
                T_nu_x=T_nu_x,
                H=float(hubble_3T(T_gamma, T_nu_e, T_nu_x,
                                      Sigma_sq=float(y[idx["sp"]]**2 + y[idx["sm"]]**2)) * mod._MEV_TO_S),
            )
            bridge_debug[sp] = {
                "relax": float(getattr(gs_dbg, "species_tagged_relax", float("nan"))),
                "amp": float(getattr(gs_dbg, "species_tagged_amp", float("nan"))),
                "qdot_shape": float(getattr(gs_dbg, "species_tagged_qdot_shape", float("nan"))),
                "qdot_target": float(getattr(gs_dbg, "species_tagged_qdot_target", float("nan"))),
                "deltaI_norm": float(np.linalg.norm(gs_dbg.delta_I)),
                "C_norm": float(np.linalg.norm(gs_dbg.C_monopole)),
            }

    f_nue = mod.char_extract_monopole(y[idx["I_nue"]], y[idx["J_nue"]], w0, q_gl)
    f_nuebar = mod.char_extract_monopole(y[idx["I_nuebar"]], y[idx["J_nuebar"]], w0, q_gl)

    weak = mod.compute_live_weak_rates(
        f_nue, f_nuebar, q_gl,
        T_gamma, T_nu_e, cfg.tau_n,
        compute_iso_reference=False,
        correction_level=cfg.correction_level,
    )

    out = {
        "config": {
            "Sigma_H": cfg.Sigma_H_plus,
            "tier": cfg.tier,
            "collisions": bool(cfg.enable_collisions),
            "correction_level": cfg.correction_level,
            "N_q": cfg.N_q,
            "N_mu": cfg.N_mu,
            "relax": float(os.environ.get("RABBIT_COLLISION_BRIDGE_RELAX", "1.0")),
        },
        "phase1": {
            "phase1_handoff_N": float(sol.t[-1]),
            "phase1_handoff_T_gamma": T_gamma,
            "phase1_handoff_T_nu_e": T_nu_e,
            "phase1_handoff_T_nu_x": T_nu_x,
            "phase1_handoff_Xn": float(y[idx["Xn"]]),
            "phase1_handoff_sigma_plus": float(y[idx["sp"]]),
            "phase1_handoff_sigma_minus": float(y[idx["sm"]]),
            "phase1_handoff_pi_plus_total": float(sum(pi_species.values())),
            "phase1_handoff_pi_plus_species": pi_species,
            "phase1_handoff_bridge_debug": bridge_debug,
            "phase1_handoff_lambda_np": float(weak.lambda_np),
            "phase1_handoff_lambda_pn": float(weak.lambda_pn),
            "phase1_handoff_I0": float(getattr(weak, "I0", np.nan)),
            "phase1_steps_external": int(len(sol.t)),
            "phase1_handoff_monopole_species": mono_species,
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--relax", type=float, default=1.0)
    ap.add_argument("--cl", type=int, default=0)
    ap.add_argument("--nq", type=int, default=20)
    ap.add_argument("--nmu", type=int, default=12)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    os.environ["RABBIT_COLLISION_BRIDGE_RELAX"] = str(args.relax)

    cfg = mod.FullCoupledConfig(
        Sigma_H_plus=args.sigma,
        Sigma_H_minus=0.0,
        tier=2,
        enable_collisions=True,
        correction_level=args.cl,
        N_q=args.nq,
        N_mu=args.nmu,
        n_reactions=12,
        enable_teff=False,
        transport_mode=TransportMode.CHARACTERISTIC,
    )

    out = run_species_phase1(cfg)

    print(json.dumps(out, indent=2, sort_keys=True))
    if args.out:
        pp = Path(args.out)
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"[saved] {pp}")


if __name__ == "__main__":
    main()
