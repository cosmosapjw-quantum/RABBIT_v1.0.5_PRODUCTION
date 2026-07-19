#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def cluster_label(sigma_plus, state_index, onset_max_state_index):
    if sigma_plus < 0:
        return "neg"
    return "pos_onset" if state_index <= onset_max_state_index else "pos_bulk"

def load_entries(manifests, onset_max_state_index):
    rows = []
    for mp in manifests:
        files = json.loads(Path(mp).read_text())["files"]
        for fp in files:
            d = json.loads(Path(fp).read_text())
            sigma_plus = float(d["state"]["sigma_plus"])
            state_index = int(d["state_index"])
            cluster = cluster_label(sigma_plus, state_index, onset_max_state_index)
            for sp in ("nue", "nux"):
                vec = np.asarray(d["species"][sp]["baseline"]["C_monopole"]["values"], dtype=np.float64)
                rows.append({
                    "file": fp,
                    "species": sp,
                    "cluster": cluster,
                    "state_index": state_index,
                    "sigma_plus": sigma_plus,
                    "vec": vec,
                })
    return rows

def build_basis(train_vecs, rank):
    M = np.stack(train_vecs, axis=0)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(rank, Vt.shape[0], M.shape[0], M.shape[1])
    return Vt[:r], S[:r]

def cos_resid(v, basis):
    v = np.asarray(v, dtype=np.float64)
    B = np.asarray(basis, dtype=np.float64)
    coeff = B @ v
    proj = coeff @ B
    nv = np.linalg.norm(v)
    npj = np.linalg.norm(proj)
    cos = 0.0 if nv == 0.0 or npj == 0.0 else float(np.dot(v, proj) / (nv * npj))
    resid = 0.0 if nv == 0.0 else float(np.linalg.norm(v - proj) / nv)
    return cos, resid

def tail_last5_share(v):
    a = np.abs(np.asarray(v, dtype=np.float64))
    den = float(a.sum())
    return 0.0 if den == 0.0 else float(a[-5:].sum() / den)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--rank", type=int, default=3)
    ap.add_argument("--onset-max-state-index", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    entries = load_entries(args.manifest, args.onset_max_state_index)
    out = []

    for i, row in enumerate(entries):
        train = [
            r["vec"] for j, r in enumerate(entries)
            if j != i and r["species"] == row["species"] and r["cluster"] == row["cluster"]
        ]
        if len(train) == 0:
            out.append({
                "file": row["file"],
                "species": row["species"],
                "cluster": row["cluster"],
                "state_index": row["state_index"],
                "sigma_plus": row["sigma_plus"],
                "train_n": 0,
                "rank_used": 0,
                "status": "insufficient_train",
                "tail_last5_share": tail_last5_share(row["vec"]),
            })
            continue

        basis, svals = build_basis(train, args.rank)
        cos, resid = cos_resid(row["vec"], basis)
        out.append({
            "file": row["file"],
            "species": row["species"],
            "cluster": row["cluster"],
            "state_index": row["state_index"],
            "sigma_plus": row["sigma_plus"],
            "train_n": len(train),
            "rank_used": int(basis.shape[0]),
            "status": "ok",
            "cos": cos,
            "resid": resid,
            "tail_last5_share": tail_last5_share(row["vec"]),
            "leading_svals": [float(x) for x in svals[:min(5, len(svals))]],
        })

    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "n_rows": len(out),
        "onset_max_state_index": args.onset_max_state_index,
        "out": str(pp.resolve())
    }, indent=2))

if __name__ == "__main__":
    main()
