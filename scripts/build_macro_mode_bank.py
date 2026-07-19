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
    toks = stem.split("_")
    return {"case": stem, "species": "unknown", "anchor": "unknown"}

def load_profiles(paths, anchors=None):
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
        if anchors and meta["anchor"] not in anchors:
            continue
        d = json.loads(fp.read_text())
        q, qk = find_arr(d, Q_ALIASES)
        w, wk = find_arr(d, W_ALIASES)
        f, fk = find_arr(d, F_ALIASES)
        raw, rk = find_arr(d, RAW_ALIASES)
        if q is None or w is None or f is None:
            print(f"[skip] {fp.name}: missing q/w/f")
            continue
        if raw is None:
            print(f"[skip] {fp.name}: missing raw source profile")
            continue
        if not (len(q) == len(w) == len(f) == len(raw)):
            print(f"[skip] {fp.name}: inconsistent lengths")
            continue
        out.append({
            "path": str(fp),
            "meta": meta,
            "q": q,
            "w": w,
            "f": f,
            "raw": raw,
            "raw_qdot": float(pick(d, "raw_qdot", default=np.nan)),
            "proj_qdot": float(pick(d, "proj_qdot", default=np.nan)),
            "orth_qdot": float(pick(d, "orth_qdot", default=np.nan)),
            "keys": {"q": qk, "w": wk, "f": fk, "raw": rk},
        })
    return out

def fit_fd(q, f, w):
    eps = 1e-14
    ff = np.clip(f, eps, 1.0 - eps)
    y = np.log(1.0 / ff - 1.0)
    A = np.column_stack([q, np.ones_like(q)])
    W = np.sqrt(np.clip(w, 0.0, None))
    Aw = A * W[:, None]
    yw = y * W
    beta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
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

def residual_after_fd(rec):
    q, w, f = rec["q"], rec["w"], rec["f"]
    T, mu, ffd = fit_fd(q, f, w)
    basis2 = weighted_orthonormalize(fd_basis(q, ffd, T, mu), w)
    r = f.copy()
    for e in basis2:
        r -= np.sum(w * r * e) * e
    n = math.sqrt(max(np.sum(w * r * r), 0.0))
    if n > 0:
        r /= n
    return {"T": T, "mu": mu, "ffd": ffd, "resid": r}

def svd_modes(rows, n_modes=2):
    if not rows:
        return []
    q = rows[0]["q"]
    w = rows[0]["w"]
    sw = np.sqrt(w)
    M = []
    for rec in rows:
        rr = residual_after_fd(rec)["resid"]
        M.append(sw * rr)
    M = np.vstack(M)
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    modes = []
    for i in range(min(n_modes, VT.shape[0])):
        v = VT[i] / np.where(sw > 0, sw, 1.0)
        basis = weighted_orthonormalize([v], w)
        if basis:
            modes.append({
                "index": i,
                "singular_value": float(S[i]),
                "mode": basis[0].tolist(),
            })
    return modes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--anchors", nargs="*", default=["onset", "peak_raw", "peak_orth"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-modes", type=int, default=2)
    args = ap.parse_args()

    recs = load_profiles(args.inputs, anchors=set(args.anchors))
    if not recs:
        raise SystemExit("[FAIL] no usable qprofile jsons found")

    q = recs[0]["q"]
    w = recs[0]["w"]
    for r in recs[1:]:
        if len(r["q"]) != len(q) or np.max(np.abs(r["q"] - q)) > 0:
            raise SystemExit("[FAIL] q grids are inconsistent across inputs")

    shared_modes = svd_modes(recs, n_modes=args.n_modes)

    species_modes = {}
    for sp in sorted(set(r["meta"]["species"] for r in recs)):
        rr = [r for r in recs if r["meta"]["species"] == sp]
        species_modes[sp] = svd_modes(rr, n_modes=args.n_modes)

    out = {
        "anchors": args.anchors,
        "n_profiles": len(recs),
        "q_nodes": q.tolist(),
        "q_weights": w.tolist(),
        "shared_modes": shared_modes,
        "species_modes": species_modes,
        "profiles": [
            {
                "path": r["path"],
                "case": r["meta"]["case"],
                "species": r["meta"]["species"],
                "anchor": r["meta"]["anchor"],
                "raw_qdot": r["raw_qdot"],
                "proj_qdot": r["proj_qdot"],
                "orth_qdot": r["orth_qdot"],
            } for r in recs
        ],
    }
    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "out": str(pp.resolve()),
        "n_profiles": len(recs),
        "shared_modes": len(shared_modes),
        "species_keys": list(species_modes.keys()),
    }, indent=2))

if __name__ == "__main__":
    main()
