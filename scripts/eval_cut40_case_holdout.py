#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from rabbit.debug.modal_contract import cluster_label, ONSET_MAX_STATE_INDEX, research_allows_cluster_bank

def load_manifest(mp):
    return json.loads(Path(mp).read_text())["files"]

def build_basis(vectors, rank):
    M = np.stack(vectors, axis=0)
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

def collect(files, onset_cut):
    rows = []
    for fp in files:
        d = json.loads(Path(fp).read_text())
        sigma_plus = float(d["state"]["sigma_plus"])
        state_index = int(d["state_index"])
        cluster = cluster_label(
            sigma_plus=sigma_plus,
            state_index=state_index,
            onset_max_state_index=onset_cut,
        )
        if not research_allows_cluster_bank(cluster=cluster):
            continue
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--rank", type=int, default=3)
    ap.add_argument("--onset-max-state-index", type=int, default=ONSET_MAX_STATE_INDEX)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train_rows = collect(load_manifest(args.train_manifest), args.onset_max_state_index)
    test_rows  = collect(load_manifest(args.test_manifest),  args.onset_max_state_index)

    bank = {}
    for cluster in ("pos_onset", "pos_bulk"):
        for sp in ("nue", "nux"):
            vecs = [r["vec"] for r in train_rows if r["cluster"] == cluster and r["species"] == sp]
            if len(vecs) >= 2:
                basis, svals = build_basis(vecs, args.rank)
                bank[(cluster, sp)] = (basis, svals, len(vecs))

    out_rows = []
    for r in test_rows:
        key = (r["cluster"], r["species"])
        if key not in bank:
            out_rows.append({
                "file": r["file"],
                "species": r["species"],
                "cluster": r["cluster"],
                "state_index": r["state_index"],
                "status": "insufficient_train",
            })
            continue
        basis, svals, ntrain = bank[key]
        cos, resid = cos_resid(r["vec"], basis)
        out_rows.append({
            "file": r["file"],
            "species": r["species"],
            "cluster": r["cluster"],
            "state_index": r["state_index"],
            "status": "ok",
            "train_n": ntrain,
            "cos": cos,
            "resid": resid,
        })

    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(out_rows, indent=2))
    print(json.dumps({"out": str(pp.resolve()), "n_rows": len(out_rows)}, indent=2))

if __name__ == "__main__":
    main()
