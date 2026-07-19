#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
import numpy as np

Q_ALIASES = ["q_nodes", "q", "q_gl", "momentum_nodes"]
W_ALIASES = ["q_weights", "q_wt", "q_wgl", "momentum_weights"]
F_ALIASES = [
    "f_monopole", "f_profile", "f", "f_vals", "distribution",
    "monopole_f", "f_species", "profile_f"
]
RAW_ALIASES = [
    "raw_integrand", "raw_source", "raw_collision_profile",
    "raw_profile", "C_raw", "collision_raw", "raw_C"
]

def pick(d, *names, default=None):
    if not isinstance(d, dict):
        return default
    for n in names:
        if n in d:
            return d[n]
    return default

def to_arr(x):
    if x is None:
        return None
    try:
        a = np.asarray(x, dtype=np.float64)
        if a.ndim == 1 and a.size > 0:
            return a
    except Exception:
        pass
    return None

def find_arr(d, aliases):
    for k in aliases:
        a = to_arr(pick(d, k, default=None))
        if a is not None:
            return a, k
    return None, None

def parse_name(path: Path):
    stem = path.stem
    m = re.match(r"(.+?)_(nuebar|nue|nux)_(onset|peak_raw|peak_orth)$", stem)
    if m:
        return {"case": m.group(1), "species": m.group(2), "anchor": m.group(3)}
    return {"case": stem, "species": "unknown", "anchor": "unknown"}

def load_profiles(paths):
    out = []
    files = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.glob("*.json")))
        elif pp.is_file():
            files.append(pp)
    for fp in files:
        meta = parse_name(fp)
        d = json.loads(fp.read_text())
        q, _ = find_arr(d, Q_ALIASES)
        w, _ = find_arr(d, W_ALIASES)
        f, _ = find_arr(d, F_ALIASES)
        raw, _ = find_arr(d, RAW_ALIASES)
        if q is None or w is None or f is None or raw is None:
            print(f"[skip] {fp.name}")
            continue
        out.append({
            "path": str(fp),
            "meta": meta,
            "q": q, "w": w, "f": f, "raw": raw,
            "raw_qdot": float(pick(d, "raw_qdot", default=np.nan)),
            "proj_qdot": float(pick(d, "proj_qdot", default=np.nan)),
            "orth_qdot": float(pick(d, "orth_qdot", default=np.nan)),
        })
    return out

def fit_fd(q, f, w):
    eps = 1e-14
    ff = np.clip(f, eps, 1.0 - eps)
    y = np.log(1.0 / ff - 1.0)
    A = np.column_stack([q, np.ones_like(q)])
    W = np.sqrt(np.clip(w, 0.0, None))
    beta, *_ = np.linalg.lstsq(A * W[:, None], y * W, rcond=None)
    a, b = beta
    if not np.isfinite(a) or abs(a) < 1e-12:
        a = 1.0
    T = 1.0 / a
    mu = -b / a
    if not np.isfinite(T) or abs(T) < 1e-12:
        T = 1.0
    if not np.isfinite(mu):
        mu = 0.0
    ffd = 1.0 / (np.exp(np.clip((q - mu) / T, -200, 200)) + 1.0)
    return float(T), float(mu), ffd

def fd_basis(q, ffd, T, mu):
    chi = ffd * (1.0 - ffd)
    b_mu = chi / max(abs(T), 1e-12)
    b_T = ((q - mu) / max(abs(T), 1e-12)**2) * chi
    return [b_mu, b_T]

def weighted_orthonormalize(vecs, w, tol=1e-14):
    basis = []
    for v in vecs:
        vv = np.asarray(v, dtype=np.float64).copy()
        for e in basis:
            vv -= np.sum(w * vv * e) * e
        n = math.sqrt(max(np.sum(w * vv * vv), 0.0))
        if np.isfinite(n) and n > tol:
            basis.append(vv / n)
    return basis

def project(v, basis, w):
    p = np.zeros_like(v)
    for e in basis:
        p += np.sum(w * v * e) * e
    return p

def metrics(q, w, raw, proj):
    nr = math.sqrt(max(np.sum(w * raw * raw), 0.0))
    npj = math.sqrt(max(np.sum(w * proj * proj), 0.0))
    dot = float(np.sum(w * raw * proj))
    cos = float(dot / (nr * npj)) if nr > 0 and npj > 0 else float("nan")
    frac = float(npj / nr) if nr > 0 else float("nan")
    resid = raw - proj
    nres = math.sqrt(max(np.sum(w * resid * resid), 0.0))
    mraw = float(np.sum(w * q**3 * raw))
    mproj = float(np.sum(w * q**3 * proj))
    sign_match = None
    if mraw != 0 and mproj != 0:
        sign_match = bool(np.sign(mraw) == np.sign(mproj))
    return {
        "cosine": cos,
        "proj_norm_over_raw_norm": frac,
        "resid_norm_over_raw_norm": float(nres / nr) if nr > 0 else float("nan"),
        "surrogate_raw_moment": mraw,
        "surrogate_proj_moment": mproj,
        "surrogate_sign_match": sign_match,
    }


def sanitize_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [sanitize_jsonable(v) for v in obj.tolist()]
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bank = json.loads(Path(args.bank).read_text())
    shared_modes = [np.asarray(m["mode"], dtype=np.float64) for m in bank.get("shared_modes", [])]
    species_modes = {
        sp: [np.asarray(m["mode"], dtype=np.float64) for m in ms]
        for sp, ms in bank.get("species_modes", {}).items()
    }

    recs = load_profiles(args.inputs)
    rows = []
    for r in recs:
        q, w, f, raw = r["q"], r["w"], r["f"], r["raw"]
        T, mu, ffd = fit_fd(q, f, w)
        fd2 = fd_basis(q, ffd, T, mu)
        sp = r["meta"]["species"]
        spm = species_modes.get(sp, [])

        candidates = {
            "fd2": fd2,
            "fd2_plus_shared1": fd2 + shared_modes[:1],
            "fd2_plus_shared2": fd2 + shared_modes[:2],
            "fd2_plus_shared2_species1": fd2 + shared_modes[:2] + spm[:1],
            "fd2_plus_species1": fd2 + spm[:1],
            "fd2_plus_species2": fd2 + spm[:2],
        }

        crow = {
            "path": r["path"],
            "case": r["meta"]["case"],
            "species": sp,
            "anchor": r["meta"]["anchor"],
            "raw_qdot_reported": r["raw_qdot"],
            "proj_qdot_reported_current": r["proj_qdot"],
            "orth_qdot_reported_current": r["orth_qdot"],
            "metrics": {},
        }

        for name, vecs in candidates.items():
            basis = weighted_orthonormalize(vecs, w)
            proj = project(raw, basis, w)
            crow["metrics"][name] = metrics(q, w, raw, proj)

        rows.append(crow)

    out = {"rows": rows}
    out = sanitize_jsonable(out)
    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "out": str(pp.resolve()),
        "n_rows": len(rows),
        "candidates": sorted(list(rows[0]["metrics"].keys())) if rows else [],
    }, indent=2))

if __name__ == "__main__":
    main()
