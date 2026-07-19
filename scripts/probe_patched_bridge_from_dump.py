#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, importlib.util, inspect, json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from rabbit.thermo.nudec_coupled import hubble_3T

def load_wrapper(path: Path):
    spec = importlib.util.spec_from_file_location("reduced_modal_wrapper_probe", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def walk(obj: Any, prefix: str = "root") -> Iterable[Tuple[str, Any]]:
    yield prefix, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{prefix}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{prefix}[{i}]")

def maybe_array(x: Any) -> Optional[np.ndarray]:
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

def find_scalar(obj: Any, *keys: str) -> Optional[float]:
    keys_l = [k.lower() for k in keys]
    for path, val in walk(obj):
        tail = path.lower().split(".")[-1]
        if any(k in tail for k in keys_l):
            try:
                return float(val)
            except Exception:
                pass
    return None

def find_array(obj: Any, patterns, target_len=None):
    pats = [p.lower() for p in patterns]
    best = None
    best_score = -1e18
    for path, val in walk(obj):
        arr = maybe_array(val)
        if arr is None:
            continue
        if target_len is not None and arr.size != target_len:
            continue
        lpath = path.lower()
        score = sum(10 for p in pats if p in lpath) - 1e-6 * len(lpath)
        if score > best_score:
            best = (path, arr)
            best_score = score
    return best if best is not None else (None, None)

def load_rows(data):
    if isinstance(data, list):
        return data
    for k in ("accepted_states", "rows", "states", "accepted", "history"):
        if isinstance(data, dict) and k in data and isinstance(data[k], list):
            return data[k]
    raise RuntimeError("No accepted_states/rows/states found.")

def choose_row(rows, which):
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

def summarize(obj):
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
            arr = np.asarray(d[key], dtype=np.float64)
            out[key] = {
                "shape": list(arr.shape),
                "finite_fraction": float(np.isfinite(arr).sum()) / float(arr.size),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "norm": float(np.linalg.norm(np.nan_to_num(arr))),
            }
    if "debug" in d:
        out["debug"] = d["debug"]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--state", default="last")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dump = json.loads(Path(args.dump).read_text())
    rows = load_rows(dump)
    idx, row = choose_row(rows, args.state)

    q_path, q_nodes = find_array(dump, ["q_nodes", "q_gl", "momentum_nodes", ".q"])
    qw_path, q_weights = find_array(dump, ["q_weights", "q_wt", "momentum_weights", "weights"], target_len=q_nodes.size if q_nodes is not None else None)
    I_path, I_vals = find_array(row, ["i_vals", ".i", "ray_i", "ivals"])
    J_path, J_vals = find_array(row, ["j_vals", ".j", "ray_j", "jvals"], target_len=I_vals.size if I_vals is not None else None)
    w0_path, w0 = find_array(dump, ["w0", "ray_weights", "mu_weights"], target_len=I_vals.size if I_vals is not None else None)
    if w0 is None:
        w0_path, w0 = find_array(row, ["w0", "ray_weights", "mu_weights"], target_len=I_vals.size if I_vals is not None else None)

    if any(x is None for x in (q_nodes, q_weights, I_vals, J_vals, w0)):
        raise RuntimeError("Could not locate q/qw/I/J/w0 arrays in dump.")

    T_gamma = find_scalar(row, "T_gamma", "tgamma")
    T_nu_e = find_scalar(row, "T_nu_e", "tnue", "t_nue")
    T_nu_x = find_scalar(row, "T_nu_x", "tnux", "t_nux")
    sigma_plus = find_scalar(row, "sigma_plus")
    sigma_minus = find_scalar(row, "sigma_minus") or 0.0
    if any(v is None for v in (T_gamma, T_nu_e, T_nu_x, sigma_plus)):
        raise RuntimeError("Could not locate thermo/shear scalars in dump.")

    H = find_scalar(row, "H", "hubble")
    if H is None:
        H = float(hubble_3T(float(T_gamma), float(T_nu_e), float(T_nu_x),
                            Sigma_sq=float(sigma_plus)**2 + float(sigma_minus)**2))

    stg = importlib.import_module("rabbit.transport.species_tagged_bridge")
    baseline_fn = stg.apply_species_tagged_bridge
    baseline_id = id(baseline_fn)
    baseline_sig = str(inspect.signature(baseline_fn))

    wrapper = load_wrapper(Path(args.wrapper))
    wrapper.install_patch()

    stg2 = importlib.import_module("rabbit.transport.species_tagged_bridge")
    patched_fn = stg2.apply_species_tagged_bridge
    patched_id = id(patched_fn)
    patched_sig = str(inspect.signature(patched_fn))

    kwargs = dict(
        I=np.asarray(I_vals, dtype=np.float64),
        J=np.asarray(J_vals, dtype=np.float64),
        w0=np.asarray(w0, dtype=np.float64),
        q_nodes=np.asarray(q_nodes, dtype=np.float64),
        q_weights=np.asarray(q_weights, dtype=np.float64),
        T_gamma=float(T_gamma),
        T_nu_e=float(T_nu_e),
        T_nu_x=float(T_nu_x),
        H=float(H),
    )

    out_species = {}
    for sp in ("nue", "nux"):
        b = baseline_fn(species=sp, **kwargs)
        try:
            p = patched_fn(species=sp, **kwargs)
            p_sum = summarize(p)
            p_err = None
        except Exception as e:
            p_sum = None
            p_err = repr(e)
        out_species[sp] = {
            "baseline": summarize(b),
            "patched": p_sum,
            "patched_error": p_err,
        }

    out = {
        "wrapper": str(Path(args.wrapper).resolve()),
        "dump": str(Path(args.dump).resolve()),
        "state_index": idx,
        "patch_changed_apply_species_tagged_bridge": baseline_id != patched_id,
        "baseline_id": baseline_id,
        "patched_id": patched_id,
        "baseline_signature": baseline_sig,
        "patched_signature": patched_sig,
        "selected_row": {
            "T_gamma": float(T_gamma),
            "T_nu_e": float(T_nu_e),
            "T_nu_x": float(T_nu_x),
            "sigma_plus": float(sigma_plus),
            "sigma_minus": float(sigma_minus),
            "H": float(H),
            "N_q": int(q_nodes.size),
            "N_mu": int(I_vals.size),
            "paths": {
                "q_nodes": q_path,
                "q_weights": qw_path,
                "I": I_path,
                "J": J_path,
                "w0": w0_path,
            },
        },
        "species": out_species,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()
