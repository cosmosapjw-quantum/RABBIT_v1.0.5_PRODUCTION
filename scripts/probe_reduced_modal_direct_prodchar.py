#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.special import roots_laguerre

from rabbit.transport.characteristic_rays import setup_ray_grid
from rabbit.thermo.nudec_coupled import hubble_3T
from rabbit.transport.species_tagged_bridge import apply_species_tagged_bridge
from rabbit.transport.reduced_modal_bridge import apply_reduced_modal_species_bridge


def load_dump(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def select_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("accepted_states", "rows", "states", "history"):
        if key in data and isinstance(data[key], list):
            return data[key]
    raise RuntimeError(f"No accepted-state list found. top-level keys={sorted(data.keys())}")


def pick_state(rows: List[Dict[str, Any]], which: str) -> Tuple[int, Dict[str, Any]]:
    n = len(rows)
    if which == "first":
        return 0, rows[0]
    if which == "mid":
        i = n // 2
        return i, rows[i]
    if which == "last":
        return n - 1, rows[-1]
    i = int(which)
    if i < 0:
        i = n + i
    if i < 0 or i >= n:
        raise IndexError(f"state index out of range: {i} / {n}")
    return i, rows[i]


def arr1(x: Any, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise RuntimeError(f"{name} is not 1D: shape={a.shape}")
    if not np.all(np.isfinite(a)):
        bad = np.where(~np.isfinite(a))[0][:10].tolist()
        raise RuntimeError(f"{name} contains non-finite values at {bad}")
    return a


def summarize_array(a: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(a, dtype=np.float64)
    return {
        "shape": list(a.shape),
        "finite_fraction": float(np.isfinite(a).sum()) / float(a.size),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "norm": float(np.linalg.norm(np.nan_to_num(a))),
    }


def gather_debug(obj: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # reduced-modal side
    for dbg_name in ("bridge_debug", "reduced_modal_debug"):
        if hasattr(obj, dbg_name):
            dbg = getattr(obj, dbg_name)
            if isinstance(dbg, dict):
                clean = {}
                for k, v in dbg.items():
                    if isinstance(v, (np.floating, np.integer)):
                        clean[k] = float(v)
                    elif isinstance(v, (float, int, str, bool)) or v is None:
                        clean[k] = v
                    elif isinstance(v, (list, tuple)):
                        vv = []
                        for x in v:
                            if isinstance(x, (np.floating, np.integer)):
                                vv.append(float(x))
                            else:
                                vv.append(x)
                        clean[k] = vv
                    else:
                        clean[k] = repr(v)
                out[dbg_name] = clean

    # species-tagged side
    for name in (
        "species_tagged_relax",
        "species_tagged_alpha",
        "species_tagged_amp",
        "species_tagged_qdot_shape",
        "species_tagged_qdot_target",
    ):
        if hasattr(obj, name):
            v = getattr(obj, name)
            try:
                out[name] = float(v)
            except Exception:
                out[name] = repr(v)

    return out


def summarize_result(obj: Any, store_full_vectors: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "type": type(obj).__name__,
        "module": type(obj).__module__,
    }

    for key in ("tangency_D2", "delta_rho_nu"):
        if hasattr(obj, key):
            v = getattr(obj, key)
            try:
                out[key] = float(v)
            except Exception:
                out[key] = repr(v)

    for key in ("delta_I", "C_monopole", "f_monopole", "theta_per_ray"):
        if hasattr(obj, key):
            arr = np.asarray(getattr(obj, key), dtype=np.float64)
            out[key] = summarize_array(arr)
            if store_full_vectors:
                out[key]["values"] = [float(x) for x in arr.tolist()]

    dbg = gather_debug(obj)
    if dbg:
        out["debug"] = dbg

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--state", default="last", help="first|mid|last|int")
    ap.add_argument("--shared-rank", type=int, default=2)
    ap.add_argument("--species-rank", type=int, default=1)
    ap.add_argument("--include-fd2", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--store-full-vectors", action="store_true")
    args = ap.parse_args()

    data = load_dump(Path(args.dump))
    rows = select_rows(data)
    idx, row = pick_state(rows, args.state)

    cfg = data.get("config", {})
    if "N_q" not in cfg or "N_mu" not in cfg:
        raise RuntimeError(f"Dump config missing N_q/N_mu. config keys={sorted(cfg.keys())}")

    N_q = int(cfg["N_q"])
    N_mu = int(cfg["N_mu"])

    I = arr1(row["I"], "I")
    J = arr1(row["J"], "J")
    if I.size != N_mu or J.size != N_mu:
        raise RuntimeError(
            f"I/J length mismatch with config.N_mu: len(I)={I.size}, len(J)={J.size}, N_mu={N_mu}"
        )

    mu0, w0, X0, signs = setup_ray_grid(N_mu)
    q_nodes, q_weights = roots_laguerre(N_q)

    mu0 = arr1(mu0, "mu0")
    w0 = arr1(w0, "w0")
    X0 = arr1(X0, "X0")
    q_nodes = arr1(q_nodes, "q_nodes")
    q_weights = arr1(q_weights, "q_weights")

    sigma_plus = float(row["sigma_plus"])
    sigma_minus = float(row.get("sigma_minus", 0.0))
    T_gamma = float(row["T_gamma"])
    T_nu_e = float(row["T_nu_e"])
    T_nu_x = float(row["T_nu_x"])
    H = float(hubble_3T(T_gamma, T_nu_e, T_nu_x, Sigma_sq=sigma_plus**2 + sigma_minus**2))

    species_out: Dict[str, Any] = {}
    for sp in ("nue", "nux"):
        baseline = apply_species_tagged_bridge(
            species=sp,
            I=I,
            J=J,
            w0=w0,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H=H,
        )

        reduced = apply_reduced_modal_species_bridge(
            species=sp,
            I=I,
            J=J,
            w0=w0,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H=H,
            bank_path=str(Path(args.bank).resolve()),
            n_shared=int(args.shared_rank),
            n_species=int(args.species_rank),
            include_fd2=bool(args.include_fd2),
        )

        species_out[sp] = {
            "baseline": summarize_result(baseline, store_full_vectors=args.store_full_vectors),
            "reduced": summarize_result(reduced, store_full_vectors=args.store_full_vectors),
        }

    out = {
        "dump": str(Path(args.dump).resolve()),
        "bank": str(Path(args.bank).resolve()),
        "state_index": idx,
        "state_selector": args.state,
        "config": {
            "N_q": N_q,
            "N_mu": N_mu,
            "shared_rank": int(args.shared_rank),
            "species_rank": int(args.species_rank),
            "include_fd2": bool(args.include_fd2),
        },
        "resolved_from": {
            "ray": {
                "module": "rabbit.transport.characteristic_rays",
                "attr": "setup_ray_grid",
            },
            "q": {
                "module": "scipy.special",
                "attr": "roots_laguerre",
            },
        },
        "state": {
            "sigma_plus": sigma_plus,
            "sigma_minus": sigma_minus,
            "T_gamma": T_gamma,
            "T_nu_e": T_nu_e,
            "T_nu_x": T_nu_x,
            "H": H,
            "I_summary": summarize_array(I),
            "J_summary": summarize_array(J),
            "mu0_summary": summarize_array(mu0),
            "w0_summary": summarize_array(w0),
            "X0_summary": summarize_array(X0),
            "q_nodes_summary": summarize_array(q_nodes),
            "q_weights_summary": summarize_array(q_weights),
        },
        "species": species_out,
    }

    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "out": str(pp.resolve()),
        "state_index": idx,
        "N_q": N_q,
        "N_mu": N_mu,
        "resolved_from": out["resolved_from"],
    }, indent=2))


if __name__ == "__main__":
    main()
