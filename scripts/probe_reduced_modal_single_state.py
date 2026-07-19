#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, inspect, json, math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from rabbit.transport.species_tagged_bridge import apply_species_tagged_bridge
from rabbit.thermo.nudec_coupled import hubble_3T

def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("reduced_modal_wrapper_probe", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

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

def walk(obj: Any, prefix: str = "root") -> Iterable[Tuple[str, Any]]:
    yield prefix, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{prefix}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{prefix}[{i}]")

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

def find_array(obj: Any, patterns: List[str], target_len: Optional[int] = None) -> Tuple[Optional[str], Optional[np.ndarray]]:
    pats = [p.lower() for p in patterns]
    best = None
    best_score = -10**9
    for path, val in walk(obj):
        arr = maybe_array(val)
        if arr is None:
            continue
        if target_len is not None and arr.size != target_len:
            continue
        lpath = path.lower()
        score = sum(12 for p in pats if p in lpath) - len(lpath) * 1e-6
        if score > best_score:
            best_score = score
            best = (path, arr)
    if best is None:
        return None, None
    return best

def choose_state(rows: List[Any], which: str):
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

def load_rows(data: Any) -> List[Any]:
    if isinstance(data, list):
        return data
    for k in ("accepted_states", "rows", "states", "accepted", "history"):
        if isinstance(data, dict) and k in data and isinstance(data[k], list):
            return data[k]
    raise RuntimeError("Could not find rows/accepted_states/states in dump JSON.")

def build_reduced_callable(mod, bank_path: Path, shared_rank: int, species_rank: int, include_fd2: bool):
    bank_data = json.loads(bank_path.read_text())
    candidates = []
    for name, obj in inspect.getmembers(mod):
        if not callable(obj):
            continue
        if name.startswith("_"):
            continue
        try:
            sig = inspect.signature(obj)
            params = set(sig.parameters)
        except Exception:
            params = set()
        score = 0
        lname = name.lower()
        for key in ("reduced", "modal", "bridge", "bank", "rank", "species"):
            if key in lname:
                score += 2
        for key in ("bank", "bank_path", "mode_bank", "mode_bank_path", "shared_rank", "species_rank"):
            if key in params:
                score += 3
        candidates.append((score, name, obj, params))
    candidates.sort(reverse=True)

    attempts = []
    for _, name, fn, params in candidates:
        kwargs_bank_variants = [
            {
                "bank": bank_data,
                "shared_rank": shared_rank,
                "species_rank": species_rank,
                "include_fd2": include_fd2,
                "use_fd2": include_fd2,
                "fd2": include_fd2,
            },
            {
                "bank_path": str(bank_path),
                "shared_rank": shared_rank,
                "species_rank": species_rank,
                "include_fd2": include_fd2,
                "use_fd2": include_fd2,
                "fd2": include_fd2,
            },
            {
                "mode_bank": bank_data,
                "shared_rank": shared_rank,
                "species_rank": species_rank,
                "include_fd2": include_fd2,
                "use_fd2": include_fd2,
                "fd2": include_fd2,
            },
            {
                "mode_bank_path": str(bank_path),
                "shared_rank": shared_rank,
                "species_rank": species_rank,
                "include_fd2": include_fd2,
                "use_fd2": include_fd2,
                "fd2": include_fd2,
            },
        ]
        for kw0 in kwargs_bank_variants:
            kwargs = {k: v for k, v in kw0.items() if k in params}
            if not kwargs:
                continue
            try:
                ret = fn(**kwargs)
            except Exception as e:
                attempts.append({"candidate": name, "kwargs": sorted(kwargs), "status": "error", "error": repr(e)})
                continue
            if callable(ret):
                return name, ret, attempts
            attempts.append({"candidate": name, "kwargs": sorted(kwargs), "status": "returned_noncallable", "type": type(ret).__name__})
    raise RuntimeError(f"Could not find reduced bridge factory. Attempts: {attempts[:30]}")

def finite_fraction(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    return float(np.isfinite(arr).sum()) / float(arr.size)

def summarize_result(obj: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"type": type(obj).__name__}
    d = getattr(obj, "__dict__", {})
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
                "finite_fraction": finite_fraction(arr),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "norm": float(np.linalg.norm(np.nan_to_num(arr))),
            }
    debug = d.get("debug", None)
    if debug is not None:
        out["debug"] = debug
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--state", default="last", help="first | mid | last | integer index")
    ap.add_argument("--shared-rank", type=int, default=2)
    ap.add_argument("--species-rank", type=int, default=1)
    ap.add_argument("--include-fd2", action="store_true", default=False)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wrapper_mod = load_module(Path(args.wrapper))
    factory_name, reduced_bridge, attempts = build_reduced_callable(
        wrapper_mod, Path(args.bank), args.shared_rank, args.species_rank, args.include_fd2
    )

    dump = json.loads(Path(args.dump).read_text())
    rows = load_rows(dump)
    idx, row = choose_state(rows, args.state)

    q_path, q_nodes = find_array(dump, ["q_nodes", "q_gl", "momentum_nodes", ".q"])
    if q_nodes is None:
        raise RuntimeError("Could not find q_nodes in dump.")
    qw_path, q_weights = find_array(dump, ["q_weights", "q_wt", "momentum_weights", "weights"], target_len=q_nodes.size)
    if q_weights is None:
        raise RuntimeError("Could not find q_weights in dump.")
    I_path, I_vals = find_array(row, ["i_vals", ".i", "ray_i", "ivals"])
    J_path, J_vals = find_array(row, ["j_vals", ".j", "ray_j", "jvals"], target_len=(I_vals.size if I_vals is not None else None))
    if I_vals is None or J_vals is None:
        raise RuntimeError("Could not find I/J ray arrays in selected row.")
    w0_path, w0 = find_array(dump, ["w0", "ray_weights", "mu_weights"], target_len=I_vals.size)
    if w0 is None:
        w0_path, w0 = find_array(row, ["w0", "ray_weights", "mu_weights"], target_len=I_vals.size)
    if w0 is None:
        raise RuntimeError("Could not find w0/mu_weights in dump.")

    T_gamma = find_scalar(row, "T_gamma", "tgamma")
    T_nu_e = find_scalar(row, "T_nu_e", "tnue", "t_nue")
    T_nu_x = find_scalar(row, "T_nu_x", "tnux", "t_nux")
    sigma_plus = find_scalar(row, "sigma_plus")
    sigma_minus = find_scalar(row, "sigma_minus") or 0.0
    if any(v is None for v in (T_gamma, T_nu_e, T_nu_x, sigma_plus)):
        raise RuntimeError("Could not find T_gamma/T_nu_e/T_nu_x/sigma_plus in row.")
    H = find_scalar(row, "H", "hubble")
    if H is None:
        Sigma_sq = float(sigma_plus)**2 + float(sigma_minus)**2
        H = float(hubble_3T(float(T_gamma), float(T_nu_e), float(T_nu_x), Sigma_sq=Sigma_sq))

    species_out = {}
    for sp in ("nue", "nux"):
        kwargs = dict(
            species=sp,
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
        base = apply_species_tagged_bridge(**kwargs)
        try:
            red = reduced_bridge(**kwargs)
            red_summary = summarize_result(red)
            red_error = None
        except Exception as e:
            red_summary = None
            red_error = repr(e)
        species_out[sp] = {
            "baseline": summarize_result(base),
            "reduced": red_summary,
            "reduced_error": red_error,
        }

    out = {
        "wrapper": str(Path(args.wrapper).resolve()),
        "bank": str(Path(args.bank).resolve()),
        "dump": str(Path(args.dump).resolve()),
        "state_index": idx,
        "factory_name": factory_name,
        "factory_attempts_head": attempts[:30],
        "selected_row_summary": {
            "T_gamma": float(T_gamma),
            "T_nu_e": float(T_nu_e),
            "T_nu_x": float(T_nu_x),
            "sigma_plus": float(sigma_plus),
            "sigma_minus": float(sigma_minus),
            "H": float(H),
            "paths": {
                "q_nodes": q_path,
                "q_weights": qw_path,
                "w0": w0_path,
                "I": I_path,
                "J": J_path,
            },
            "sizes": {
                "N_q": int(q_nodes.size),
                "N_mu": int(I_vals.size),
            },
        },
        "species": species_out,
    }

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()
