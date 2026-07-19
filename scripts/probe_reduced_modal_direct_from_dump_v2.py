#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from rabbit.transport.species_tagged_bridge import apply_species_tagged_bridge
from rabbit.transport.reduced_modal_bridge import apply_reduced_modal_species_bridge
from rabbit.thermo.nudec_coupled import hubble_3T

def walk(obj: Any, prefix: str = "root") -> Iterable[Tuple[str, Any]]:
    yield prefix, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{prefix}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{prefix}[{i}]")

def maybe_array(x: Any):
    if isinstance(x, np.ndarray):
        arr = np.asarray(x, dtype=np.float64)
        return arr if arr.ndim == 1 and arr.size > 0 else None
    if isinstance(x, (list, tuple)) and len(x) > 0:
        try:
            arr = np.asarray(x, dtype=np.float64)
        except Exception:
            return None
        return arr if arr.ndim == 1 and arr.size > 0 else None
    return None

def load_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    for k in ("accepted_states", "rows", "states", "accepted", "history"):
        if k in data and isinstance(data[k], list):
            return data[k]
    raise RuntimeError("No accepted state list found.")

def choose_row(rows: List[Dict[str, Any]], which: str):
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
    return i, rows[i]

def get_row_arrays(row):
    I = np.asarray(row["I"], dtype=np.float64)
    J = np.asarray(row["J"], dtype=np.float64)
    return I, J

def score_path(path: str, include=(), exclude=()):
    p = path.lower()
    s = 0.0
    for x in include:
        if x in p:
            s += 10.0
    for x in exclude:
        if x in p:
            s -= 100.0
    s -= 1e-4 * len(p)
    return s

def best_array(obj: Any, target_len: int, include=(), exclude=()):
    cand = []
    for path, val in walk(obj):
        arr = maybe_array(val)
        if arr is None or arr.size != target_len:
            continue
        cand.append((score_path(path, include, exclude), path, arr))
    if not cand:
        return None, None, []
    cand.sort(key=lambda x: x[0], reverse=True)
    return cand[0][1], np.asarray(cand[0][2], dtype=np.float64), [
        {"score": c[0], "path": c[1]} for c in cand[:12]
    ]

def finite_summary(arr):
    a = np.asarray(arr, dtype=np.float64)
    return {
        "shape": list(a.shape),
        "finite_fraction": float(np.isfinite(a).sum()) / float(a.size),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "norm": float(np.linalg.norm(np.nan_to_num(a))),
    }

def summarize_result(obj):
    d = getattr(obj, "__dict__", {})
    out = {
        "type": type(obj).__name__,
        "module": type(obj).__module__,
    }
    for key in ("tangency_D2", "delta_rho_nu"):
        if key in d:
            try:
                out[key] = float(d[key])
            except Exception:
                out[key] = repr(d[key])
    for key in ("delta_I", "C_monopole", "f_monopole", "theta_per_ray"):
        if key in d:
            out[key] = finite_summary(d[key])
    if "debug" in d:
        out["debug"] = d["debug"]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--state", default="last")
    ap.add_argument("--shared-rank", type=int, default=2)
    ap.add_argument("--species-rank", type=int, default=1)
    ap.add_argument("--include-fd2", action="store_true", default=False)
    ap.add_argument("--nq", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.dump).read_text())
    rows = load_rows(data)
    idx, row = choose_row(rows, args.state)

    I, J = get_row_arrays(row)
    N_mu = int(I.size)

    # N_q: prefer explicit arg, else config, else best non-I/J candidate length > N_mu
    cfg = data.get("config", {})
    N_q = args.nq or cfg.get("N_q") or cfg.get("nq")
    if N_q is None:
        lengths = {}
        for path, val in walk(data):
            arr = maybe_array(val)
            if arr is None:
                continue
            lengths[arr.size] = lengths.get(arr.size, 0) + 1
        bigger = [L for L in lengths if L > N_mu]
        if not bigger:
            raise RuntimeError("Could not infer N_q; pass --nq explicitly.")
        N_q = min(bigger)
    N_q = int(N_q)

    q_path, q_nodes, q_cands = best_array(
        data, N_q,
        include=("q_gl", "q_nodes", "momentum", ".q", "qgrid"),
        exclude=(".i", ".j", "theta", "sigma", "accepted_states", "ray", "w0", "weight"),
    )
    qw_path, q_weights, qw_cands = best_array(
        data, N_q,
        include=("q_wt", "q_weights", "momentum_weights", "weights", "wt"),
        exclude=(".i", ".j", "theta", "sigma", "accepted_states", "ray", "w0"),
    )
    w0_path, w0, w0_cands = best_array(
        data, N_mu,
        include=("w0", "mu_weight", "ray_weight"),
        exclude=(".i", ".j", "q_", "qgl", "q_nodes", "momentum"),
    )

    if q_nodes is None or q_weights is None or w0 is None:
        raise RuntimeError(json.dumps({
            "message": "Failed to locate q_nodes/q_weights/w0 in dump.",
            "N_q": N_q,
            "N_mu": N_mu,
            "q_candidates": q_cands,
            "qw_candidates": qw_cands,
            "w0_candidates": w0_cands,
        }, indent=2))

    T_gamma = float(row["T_gamma"])
    T_nu_e = float(row["T_nu_e"])
    T_nu_x = float(row["T_nu_x"])
    sigma_plus = float(row["sigma_plus"])
    sigma_minus = float(row.get("sigma_minus", 0.0))
    H = float(hubble_3T(T_gamma, T_nu_e, T_nu_x, Sigma_sq=sigma_plus**2 + sigma_minus**2))

    out_species = {}
    for sp in ("nue", "nux"):
        base = apply_species_tagged_bridge(
            species=sp,
            I=I, J=J, w0=w0,
            q_nodes=q_nodes, q_weights=q_weights,
            T_gamma=T_gamma, T_nu_e=T_nu_e, T_nu_x=T_nu_x, H=H,
        )
        red = apply_reduced_modal_species_bridge(
            species=sp,
            I=I, J=J, w0=w0,
            q_nodes=q_nodes, q_weights=q_weights,
            T_gamma=T_gamma, T_nu_e=T_nu_e, T_nu_x=T_nu_x, H=H,
            bank_path=str(Path(args.bank).resolve()),
            n_shared=args.shared_rank,
            n_species=args.species_rank,
            include_fd2=bool(args.include_fd2),
        )
        out_species[sp] = {
            "baseline": summarize_result(base),
            "reduced": summarize_result(red),
        }

    out = {
        "dump": str(Path(args.dump).resolve()),
        "bank": str(Path(args.bank).resolve()),
        "state_index": idx,
        "config": {
            "N_q": N_q,
            "N_mu": N_mu,
            "shared_rank": args.shared_rank,
            "species_rank": args.species_rank,
            "include_fd2": bool(args.include_fd2),
        },
        "resolved_paths": {
            "q_nodes": q_path,
            "q_weights": qw_path,
            "w0": w0_path,
            "q_candidates": q_cands,
            "qw_candidates": qw_cands,
            "w0_candidates": w0_cands,
        },
        "state": {
            "T_gamma": T_gamma,
            "T_nu_e": T_nu_e,
            "T_nu_x": T_nu_x,
            "sigma_plus": sigma_plus,
            "sigma_minus": sigma_minus,
            "H": H,
        },
        "species": out_species,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()
