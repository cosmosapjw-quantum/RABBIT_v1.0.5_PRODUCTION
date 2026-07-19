#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp
from scipy.special import roots_laguerre

from rabbit.transport.species_tagged_bridge import apply_species_tagged_bridge
from rabbit.collisions.species import as_species, bank_energy_transfer_rate


def resolve_setup_ray_grid():
    import rabbit.drivers.full_coupled_typeI as fc

    if hasattr(fc, "setup_ray_grid"):
        return fc.setup_ray_grid

    candidates = [
        "rabbit.transport.characteristic",
        "rabbit.transport.characteristic_core",
        "rabbit.transport.characteristic_utils",
        "rabbit.transport.rays",
        "rabbit.transport.ray_utils",
        "rabbit.transport.phase_space",
    ]
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(mod, "setup_ray_grid"):
            return getattr(mod, "setup_ray_grid")

    raise ImportError(
        "Could not resolve setup_ray_grid(). "
        "It is not visible from rabbit.drivers.full_coupled_typeI "
        "and no fallback candidate module worked."
    )


def energy_exchange_rate_np(C_monopole, q_nodes, q_weights):
    q = np.asarray(q_nodes, dtype=np.float64)
    w = np.asarray(q_weights, dtype=np.float64)
    C = np.asarray(C_monopole, dtype=np.float64)
    term = w * np.exp(np.minimum(q, 500.0)) * q**3 * C
    return float(np.sum(term))


def energy_exchange_rate_jax(C_monopole, q_nodes, q_weights):
    q = jnp.asarray(q_nodes, dtype=jnp.float64)
    w = jnp.asarray(q_weights, dtype=jnp.float64)
    C = jnp.asarray(C_monopole, dtype=jnp.float64)
    term = w * jnp.exp(jnp.minimum(q, 500.0)) * q**3 * C
    return float(jnp.sum(term))


def topk_terms(C_monopole, q_nodes, q_weights, k=8):
    q = np.asarray(q_nodes, dtype=np.float64)
    w = np.asarray(q_weights, dtype=np.float64)
    C = np.asarray(C_monopole, dtype=np.float64)
    term = w * np.exp(np.minimum(q, 500.0)) * q**3 * C
    idx = np.argsort(np.abs(term))[::-1][:k]
    out = []
    for i in idx:
        out.append({
            "i": int(i),
            "q": float(q[i]),
            "w": float(w[i]),
            "C": float(C[i]),
            "term": float(term[i]),
        })
    return out


def maybe_hubble_3T(Tg, Tne, Tnx, sigma_plus):
    try:
        from rabbit.thermo.nudec_coupled import hubble_3T
        return float(hubble_3T(Tg, Tne, Tnx, Sigma_sq=float(sigma_plus)**2))
    except Exception:
        return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="accepted-state dump json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.dump).read_text())
    cfg = data["config"]
    states = data["accepted_states"]

    setup_ray_grid = resolve_setup_ray_grid()
    _, w0, _, _ = setup_ray_grid(cfg["N_mu"])

    q_nodes, q_weights = roots_laguerre(cfg["N_q"])
    q_nodes = q_nodes.astype(np.float64)
    q_weights = q_weights.astype(np.float64)

    out_rows = []
    species_list = ["nue", "nuebar", "nux"]

    for st in states:
        I = np.asarray(st["I"], dtype=np.float64)
        J = np.asarray(st["J"], dtype=np.float64)
        Tg = float(st["T_gamma"])
        Tne = float(st["T_nu_e"])
        Tnx = float(st["T_nu_x"])
        sigma_plus = float(st["sigma_plus"])
        H = maybe_hubble_3T(Tg, Tne, Tnx, sigma_plus)

        row = {
            "label": st["label"],
            "N": float(st["N"]),
            "sigma_plus": sigma_plus,
            "sigma_minus": float(st["sigma_minus"]),
            "T_gamma": Tg,
            "T_nu_e": Tne,
            "T_nu_x": Tnx,
            "Xn": float(st["Xn"]),
            "species": {},
        }

        for sp in species_list:
            gs = apply_species_tagged_bridge(
                species=sp,
                I=I, J=J, w0=w0,
                q_nodes=q_nodes, q_weights=q_weights,
                T_gamma=Tg, T_nu_e=Tne, T_nu_x=Tnx,
                H=H,
            )

            Tsp = Tne if sp in ("nue", "nuebar") else Tnx
            qdot_np = energy_exchange_rate_np(gs.C_monopole, q_nodes, q_weights)
            qdot_jax = energy_exchange_rate_jax(gs.C_monopole, q_nodes, q_weights)
            qdot_target = float(bank_energy_transfer_rate(as_species(sp), Tg, Tne, Tnx))

            row["species"][sp] = {
                "qdot_shape_np": qdot_np,
                "qdot_shape_jax": qdot_jax,
                "dqdot_jax_minus_np": qdot_jax - qdot_np,
                "qdot_target": qdot_target,
                "C_norm": float(np.linalg.norm(np.asarray(gs.C_monopole, dtype=np.float64))),
                "deltaI_norm": float(np.linalg.norm(np.asarray(gs.delta_I, dtype=np.float64))),
                "topk_terms": topk_terms(gs.C_monopole, q_nodes, q_weights, k=8),
            }

            if hasattr(gs, "debug"):
                row["species"][sp]["debug"] = gs.debug

        out_rows.append(row)

    out = {
        "config": cfg,
        "n_states": len(out_rows),
        "jax_devices": [str(d) for d in jax.devices()],
        "rows": out_rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "out": str(out_path.resolve()),
        "n_states": len(out_rows),
        "jax_devices": [str(d) for d in jax.devices()],
    }, indent=2))


if __name__ == "__main__":
    main()
