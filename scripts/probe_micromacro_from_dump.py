#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
from pathlib import Path

import numpy as np
from scipy.special import roots_laguerre

from rabbit.debug.micromacro_probe import analyze_collision_monopole


def resolve_symbol(symbol_name: str, preferred_modules: list[str] | None = None):
    preferred_modules = preferred_modules or []

    for modname in preferred_modules:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(mod, symbol_name):
            return getattr(mod, symbol_name), modname

    import rabbit
    for modinfo in pkgutil.walk_packages(rabbit.__path__, prefix="rabbit."):
        modname = modinfo.name
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(mod, symbol_name):
            return getattr(mod, symbol_name), modname

    raise ImportError(f"Could not resolve symbol '{symbol_name}' under package 'rabbit'.")


def pick(d: dict, *names):
    for n in names:
        if n in d:
            return d[n]
    raise KeyError(f"None of keys {names} found in record.")


def optional_pick(d: dict, *names, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--last-k", type=int, default=3)
    args = ap.parse_args()

    apply_species_tagged_bridge, bridge_mod = resolve_symbol(
        "apply_species_tagged_bridge",
        preferred_modules=[
            "rabbit.transport.species_tagged_bridge",
        ],
    )

    data = json.loads(Path(args.dump).read_text())
    cfg = data["config"]
    accepted = data["accepted_states"]

    N_q = int(cfg["N_q"])
    N_mu = int(cfg["N_mu"])
    tier = int(cfg.get("tier", 1))

    q_nodes_np, q_weights_np = roots_laguerre(N_q)
    q_nodes = q_nodes_np.astype(np.float64)
    q_weights = q_weights_np.astype(np.float64)

    setup_ray_grid, ray_mod = resolve_symbol(
        "setup_ray_grid",
        preferred_modules=[
            "rabbit.transport.characteristic",
            "rabbit.transport.characteristic_utils",
            "rabbit.transport.ray_tools",
            "rabbit.transport.quadrature",
        ],
    )
    _, w0_global, _, _ = setup_ray_grid(N_mu)
    w0_global = np.asarray(w0_global, dtype=np.float64)

    hubble_3T = None
    try:
        hubble_3T, hubble_mod = resolve_symbol(
            "hubble_3T",
            preferred_modules=[
                "rabbit.thermo.nudec_coupled",
            ],
        )
    except Exception:
        hubble_mod = None

    MEV_TO_S = 1.519267447e21

    rows = []
    for rec in accepted:
        I = np.asarray(pick(rec, "I", "I_vals", "I_state"), dtype=np.float64)
        J = np.asarray(pick(rec, "J", "J_vals", "J_state"), dtype=np.float64)

        w0_rec = optional_pick(rec, "w0", "ray_weights", "mu_weights", default=None)
        w0 = np.asarray(w0_rec, dtype=np.float64) if w0_rec is not None else w0_global

        T_gamma = float(pick(rec, "T_gamma", "phase1_handoff_T", "Tg"))
        T_nu_e = float(pick(rec, "T_nu_e", "Tne"))
        T_nu_x = float(pick(rec, "T_nu_x", "Tnx"))

        sigma_plus = float(pick(rec, "sigma_plus", "Sigma_plus", "Sigma_H_plus"))
        Xn = float(pick(rec, "Xn", "X_n"))
        N = float(pick(rec, "N"))

        H_rec = optional_pick(rec, "H", "Hubble", "H_invsec", default=None)
        if H_rec is not None:
            H = float(H_rec)
        else:
            if tier >= 2 and hubble_3T is not None:
                H = float(hubble_3T(T_gamma, T_nu_e, T_nu_x, Sigma_sq=sigma_plus**2) * MEV_TO_S)
            else:
                raise KeyError("H not present in dump and no tier-2 hubble fallback available.")

        species_out = {}
        for sp in ["nue", "nuebar", "nux"]:
            gs = apply_species_tagged_bridge(
                species=sp,
                I=I, J=J, w0=w0,
                q_nodes=q_nodes, q_weights=q_weights,
                T_gamma=T_gamma, T_nu_e=T_nu_e, T_nu_x=T_nu_x,
                H=H,
            )
            mm = analyze_collision_monopole(gs.C_monopole, q_nodes, q_weights)

            species_out[sp] = {
                "raw_qdot": mm.raw_qdot,
                "raw_n2": mm.raw_n2,
                "coeff_T": mm.coeff_T,
                "coeff_mu": mm.coeff_mu,
                "proj_qdot_T": mm.proj_qdot_T,
                "proj_qdot_Tmu": mm.proj_qdot_Tmu,
                "orth_qdot_T": mm.orth_qdot_T,
                "orth_qdot_Tmu": mm.orth_qdot_Tmu,
                "tail_frac_raw_last3": mm.tail_frac_raw_last3,
                "tail_frac_orthT_last3": mm.tail_frac_orthT_last3,
                "tail_frac_orthTmu_last3": mm.tail_frac_orthTmu_last3,
                "signflip_index_raw": mm.signflip_index_raw,
                "signflip_index_orthT": mm.signflip_index_orthT,
                "signflip_index_orthTmu": mm.signflip_index_orthTmu,
                "C_norm": float(np.linalg.norm(np.asarray(gs.C_monopole, dtype=np.float64))),
                "deltaI_norm": float(np.linalg.norm(np.asarray(gs.delta_I, dtype=np.float64))),
            }

        rows.append({
            "label": rec.get("label", ""),
            "N": N,
            "sigma_plus": sigma_plus,
            "Xn": Xn,
            "T_gamma": T_gamma,
            "T_nu_e": T_nu_e,
            "T_nu_x": T_nu_x,
            "species": species_out,
        })

    out = {
        "config": cfg,
        "imports": {
            "bridge_module": bridge_mod,
            "ray_module": ray_mod,
            "hubble_module": hubble_mod,
        },
        "n_states": len(rows),
        "rows": rows,
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "out": str(p.resolve()),
        "n_states": len(rows),
        "bridge_module": bridge_mod,
    }, indent=2))


if __name__ == "__main__":
    main()
