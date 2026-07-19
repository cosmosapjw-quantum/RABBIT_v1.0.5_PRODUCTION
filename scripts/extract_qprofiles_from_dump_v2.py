#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_laguerre

from rabbit.transport.species_tagged_bridge import apply_species_tagged_bridge
from rabbit.thermo.nudec_coupled import hubble_3T

try:
    from rabbit.drivers.full_coupled_typeI import setup_ray_grid
except Exception:
    setup_ray_grid = None

# Fallback if private constant import fails
_MEV_TO_S = 1.519267447e21


def jload(path: str | Path):
    return json.loads(Path(path).read_text())


def maybe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def maybe_array(x):
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float64)
    return arr


def pick(obj, *names, default=None):
    if not isinstance(obj, dict):
        return default
    for n in names:
        if n in obj and obj[n] is not None:
            return obj[n]
    return default


def find_rows(data: dict):
    for key in ("accepted_states", "rows", "states", "accepted", "records", "data"):
        v = data.get(key)
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            return v
    raise KeyError("Could not find accepted-state rows in dump.")


def get_global_array(data: dict, row: dict, *names):
    for obj in (row, data):
        val = pick(obj, *names, default=None)
        if val is not None:
            arr = maybe_array(val)
            if arr is not None:
                return arr
    raise KeyError(f"None of keys {names} found in row/global dump.")


def infer_nmu_from_row(row: dict):
    for key in ("N_mu", "nmu", "n_mu"):
        v = pick(row, key, default=None)
        if v is not None:
            return int(v)
    for key in ("I", "I_vals", "ray_I", "J", "J_vals", "ray_J"):
        v = pick(row, key, default=None)
        if isinstance(v, list) and len(v) > 0:
            return int(len(v))
    return None


def infer_nq_from_row_or_dump(data: dict, row: dict):
    for obj in (row, data):
        for key in ("N_q", "nq", "n_q"):
            v = pick(obj, key, default=None)
            if v is not None:
                return int(v)
        cfg = pick(obj, "config", default=None)
        if isinstance(cfg, dict):
            for key in ("N_q", "nq", "n_q"):
                v = pick(cfg, key, default=None)
                if v is not None:
                    return int(v)
    return 20


def reconstruct_q_quadrature(data: dict, row: dict):
    nq = infer_nq_from_row_or_dump(data, row)
    q, w = roots_laguerre(nq)
    return np.asarray(q, dtype=np.float64), np.asarray(w, dtype=np.float64), nq


def reconstruct_ray_weights(data: dict, row: dict):
    nmu = infer_nmu_from_row(row)
    if nmu is None:
        raise RuntimeError("Could not infer N_mu from dump row.")
    if setup_ray_grid is not None:
        mu0, w0, X0, signs = setup_ray_grid(nmu)
        return (
            np.asarray(mu0, dtype=np.float64),
            np.asarray(w0, dtype=np.float64),
            np.asarray(X0, dtype=np.float64),
            np.asarray(signs, dtype=np.float64),
            nmu,
        )
    # conservative fallback: equal weights on [-1,1]
    mu0 = np.linspace(-1.0, 1.0, nmu)
    w0 = np.full(nmu, 2.0 / nmu, dtype=np.float64)
    X0 = np.zeros(nmu, dtype=np.float64)
    signs = np.sign(mu0)
    signs[signs == 0.0] = 1.0
    return mu0, w0, X0, signs, nmu


def get_scalar(data: dict, row: dict, *names, default=np.nan):
    for obj in (row, data):
        val = pick(obj, *names, default=None)
        if val is not None:
            return maybe_float(val, default=default)
    return default


def find_case_rows(micro: dict):
    for key in ("rows", "states", "records", "data"):
        if key in micro and isinstance(micro[key], list):
            return micro[key]
    if isinstance(micro, list):
        return micro
    raise KeyError("Could not find row list in micromacro json.")


def build_idx_lookup(rows):
    out = {}
    for i, r in enumerate(rows):
        label = r.get("label", f"accepted_{i}")
        out[label] = i
        out[f"accepted_{i}"] = i
        if "idx" in r and r["idx"] is not None:
            out[str(r["idx"])] = int(r["idx"])
    return out


def infer_row_index(summary_entry: dict, lookup: dict):
    if summary_entry is None:
        return None
    idx = summary_entry.get("idx", None)
    if idx is not None:
        return int(idx)
    label = summary_entry.get("label", None)
    if label in lookup:
        return lookup[label]
    return None


def weighted_projection(C, f, q, w):
    # Debug-only tangent basis around FD-like manifold.
    # q is dimensionless momentum variable already used by code.
    g = np.clip(f * (1.0 - f), 0.0, np.inf)
    b_mu = g
    b_T = q * g

    W = w * q**3
    W = np.clip(W, 0.0, np.inf)

    X = np.column_stack([b_T * np.sqrt(W), b_mu * np.sqrt(W)])
    y = C * np.sqrt(W)

    gram = X.T @ X
    cond = np.linalg.cond(gram) if np.all(np.isfinite(gram)) else np.inf

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    aT, aMu = float(coef[0]), float(coef[1])

    C_proj = aT * b_T + aMu * b_mu
    C_orth = C - C_proj

    raw_int = W * C
    proj_int = W * C_proj
    orth_int = W * C_orth

    return {
        "aT": aT,
        "aMu": aMu,
        "cond_gram": float(cond),
        "basis_T": b_T,
        "basis_mu": b_mu,
        "C_proj": C_proj,
        "C_orth": C_orth,
        "raw_integrand": raw_int,
        "proj_integrand": proj_int,
        "orth_integrand": orth_int,
        "raw_qdot": float(np.sum(raw_int)),
        "proj_qdot": float(np.sum(proj_int)),
        "orth_qdot": float(np.sum(orth_int)),
    }


def cumulative(arr):
    return np.cumsum(np.asarray(arr, dtype=np.float64))


def signflips(arr):
    arr = np.asarray(arr, dtype=np.float64)
    s = np.sign(arr)
    out = []
    for i in range(1, len(s)):
        if s[i] == 0 or s[i - 1] == 0:
            continue
        if s[i] != s[i - 1]:
            out.append(i)
    return out


def plot_profile(case_key, species, anchor, payload, outpng: Path):
    q = np.asarray(payload["q_nodes"])
    raw = np.asarray(payload["raw_integrand"])
    proj = np.asarray(payload["proj_integrand"])
    orth = np.asarray(payload["orth_integrand"])
    C = np.asarray(payload["C_monopole"])
    Cproj = np.asarray(payload["C_proj"])
    Corth = np.asarray(payload["C_orth"])
    f = np.asarray(payload["f_monopole"])

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)

    axes[0].plot(q, C, label="C(q)")
    axes[0].plot(q, Cproj, label="proj(q)")
    axes[0].plot(q, Corth, label="orth(q)")
    axes[0].set_ylabel("collision shape")
    axes[0].set_title(f"{case_key} | {species} | {anchor}")
    axes[0].legend()

    axes[1].plot(q, raw, label="raw")
    axes[1].plot(q, proj, label="proj")
    axes[1].plot(q, orth, label="orth")
    axes[1].set_ylabel(r"$w q^3 C$")
    axes[1].legend()

    axes[2].plot(q, cumulative(raw), label="cum raw")
    axes[2].plot(q, cumulative(proj), label="cum proj")
    axes[2].plot(q, cumulative(orth), label="cum orth")
    axes[2].set_xlabel("q")
    axes[2].set_ylabel("cumulative")
    axes[2].legend()

    fig.savefig(outpng, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
    axes[0].plot(q, f, label="f(q)")
    axes[0].set_ylabel("f")
    axes[0].legend()

    axes[1].plot(q, np.asarray(payload["basis_T"]), label="basis_T")
    axes[1].plot(q, np.asarray(payload["basis_mu"]), label="basis_mu")
    axes[1].set_xlabel("q")
    axes[1].set_ylabel("basis")
    axes[1].legend()

    fig.savefig(outpng.with_name(outpng.stem + "_basis.png"), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--micro", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--case-key", required=True)
    ap.add_argument("--anchors", nargs="+", default=["onset", "peak_raw", "peak_orth"])
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    dump = jload(args.dump)
    micro = jload(args.micro)
    summary = jload(args.summary)

    rows_dump = find_rows(dump)
    rows_micro = find_case_rows(micro)
    lookup = build_idx_lookup(rows_micro)

    case = summary[args.case_key]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for species in ("nue", "nux"):
        for anchor in args.anchors:
            sentry = case.get(species, {}).get(anchor, None)
            idx = infer_row_index(sentry, lookup)
            if idx is None:
                print(f"[skip] {args.case_key} {species} {anchor}: no idx")
                continue

            row = rows_dump[idx]

            I = get_global_array(dump, row, "I", "I_vals", "ray_I")
            J = get_global_array(dump, row, "J", "J_vals", "ray_J")

            try:
                q = get_global_array(dump, row, "q_nodes", "q_gl", "q", "momentum_nodes")
                qw = get_global_array(dump, row, "q_weights", "q_wt", "q_wgl", "momentum_weights")
            except KeyError:
                q, qw, nq = reconstruct_q_quadrature(dump, row)

            try:
                w0 = get_global_array(dump, row, "w0", "ray_weights", "mu_weights")
            except KeyError:
                _mu0, w0, _X0, _signs, nmu = reconstruct_ray_weights(dump, row)

            Tg = get_scalar(dump, row, "T_gamma", "tg", "phase1_handoff_T_gamma")
            Tne = get_scalar(dump, row, "T_nu_e", "tne", "phase1_handoff_T_nu_e")
            Tnx = get_scalar(dump, row, "T_nu_x", "tnx", "phase1_handoff_T_nu_x")
            sp = get_scalar(dump, row, "sigma_plus", "Sigma_H_plus", "phase1_handoff_sigma_plus", default=0.0)
            sm = get_scalar(dump, row, "sigma_minus", "Sigma_H_minus", "phase1_handoff_sigma_minus", default=0.0)

            H = get_scalar(dump, row, "H", "hubble", default=np.nan)
            if not np.isfinite(H):
                H = float(hubble_3T(Tg, Tne, Tnx, Sigma_sq=sp**2 + sm**2)) * _MEV_TO_S

            gs = apply_species_tagged_bridge(
                species=species,
                I=I,
                J=J,
                w0=w0,
                q_nodes=q,
                q_weights=qw,
                T_gamma=Tg,
                T_nu_e=Tne,
                T_nu_x=Tnx,
                H=H,
            )

            C = np.asarray(gs.C_monopole, dtype=np.float64)
            f = np.asarray(gs.f_monopole, dtype=np.float64)
            dI = np.asarray(gs.delta_I, dtype=np.float64)

            proj = weighted_projection(C=C, f=f, q=q, w=qw)

            payload = {
                "case_key": args.case_key,
                "species": species,
                "anchor": anchor,
                "idx": int(idx),
                "label": rows_micro[idx].get("label", f"accepted_{idx}"),
                "N": maybe_float(rows_micro[idx].get("N", row.get("N"))),
                "sigma_plus": sp,
                "sigma_minus": sm,
                "T_gamma": Tg,
                "T_nu_e": Tne,
                "T_nu_x": Tnx,
                "q_nodes": q.tolist(),
                "q_weights": qw.tolist(),
                "N_q_inferred": int(len(q)),
                "N_mu_inferred": int(len(w0)),
                "C_monopole": C.tolist(),
                "delta_I": dI.tolist(),
                "f_monopole": f.tolist(),
                "basis_T": proj["basis_T"].tolist(),
                "basis_mu": proj["basis_mu"].tolist(),
                "C_proj": proj["C_proj"].tolist(),
                "C_orth": proj["C_orth"].tolist(),
                "raw_integrand": proj["raw_integrand"].tolist(),
                "proj_integrand": proj["proj_integrand"].tolist(),
                "orth_integrand": proj["orth_integrand"].tolist(),
                "raw_qdot": proj["raw_qdot"],
                "proj_qdot": proj["proj_qdot"],
                "orth_qdot": proj["orth_qdot"],
                "aT": proj["aT"],
                "aMu": proj["aMu"],
                "cond_gram": proj["cond_gram"],
                "signflips_raw_integrand": signflips(proj["raw_integrand"]),
                "signflips_proj_integrand": signflips(proj["proj_integrand"]),
                "signflips_orth_integrand": signflips(proj["orth_integrand"]),
            }

            outjson = outdir / f"{args.case_key}_{species}_{anchor}.json"
            outpng = outdir / f"{args.case_key}_{species}_{anchor}.png"
            outjson.write_text(json.dumps(payload, indent=2))
            plot_profile(args.case_key, species, anchor, payload, outpng)
            print(f"[saved] {outjson}")
            print(f"[saved] {outpng}")

if __name__ == "__main__":
    main()
