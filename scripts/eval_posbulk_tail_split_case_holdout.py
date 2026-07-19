#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ONSET_MAX_STATE_INDEX = 40

def cluster_label(sigma_plus, state_index, bulk_tail_share=None, bulk_cut=None):
    if sigma_plus < 0.0:
        return "neg"
    if state_index <= ONSET_MAX_STATE_INDEX:
        return "pos_onset"
    if bulk_cut is None or bulk_tail_share is None:
        return "pos_bulk"
    return "pos_bulk_lowtail" if bulk_tail_share <= bulk_cut else "pos_bulk_hightail"

def load_rows(manifest_path, bulk_cut=None):
    files = json.loads(Path(manifest_path).read_text())["files"]
    rows = []
    for fp in files:
        d = json.loads(Path(fp).read_text())
        sigma_plus = float(d["state"]["sigma_plus"])
        state_index = int(d["state_index"])
        for sp in ("nue", "nux"):
            vec = np.asarray(d["species"][sp]["baseline"]["C_monopole"]["values"], dtype=np.float64)
            tail = float(np.sum(np.abs(vec[-5:])) / np.sum(np.abs(vec)))
            cl = cluster_label(sigma_plus, state_index, tail, bulk_cut)
            if cl == "neg":
                continue
            rows.append({
                "file": fp,
                "species": sp,
                "cluster": cl,
                "state_index": state_index,
                "sigma_plus": sigma_plus,
                "tail_last5_share": tail,
                "vec": vec,
            })
    return rows

def build_basis(vectors, rank):
    M = np.stack(vectors, axis=0)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(rank, Vt.shape[0], M.shape[0], M.shape[1])
    return Vt[:r]

def cos_resid(v, basis):
    coeff = basis @ v
    proj = coeff @ basis
    nv = np.linalg.norm(v)
    npj = np.linalg.norm(proj)
    cos = 0.0 if nv == 0.0 or npj == 0.0 else float(np.dot(v, proj) / (nv * npj))
    resid = 0.0 if nv == 0.0 else float(np.linalg.norm(v - proj) / nv)
    return cos, resid

def run_direction(train_manifest, test_manifest, bulk_cut, rank):
    train = load_rows(train_manifest, bulk_cut=bulk_cut)
    test  = load_rows(test_manifest,  bulk_cut=bulk_cut)

    bank = {}
    for cl in ("pos_onset", "pos_bulk_lowtail", "pos_bulk_hightail"):
        for sp in ("nue", "nux"):
            vecs = [r["vec"] for r in train if r["cluster"] == cl and r["species"] == sp]
            if len(vecs) >= 2:
                bank[(cl, sp)] = build_basis(vecs, rank)

    out = []
    for r in test:
        key = (r["cluster"], r["species"])
        if key not in bank:
            out.append({
                "species": r["species"],
                "cluster": r["cluster"],
                "state_index": r["state_index"],
                "status": "insufficient_train",
            })
            continue
        cos, resid = cos_resid(r["vec"], bank[key])
        out.append({
            "species": r["species"],
            "cluster": r["cluster"],
            "state_index": r["state_index"],
            "status": "ok",
            "cos": cos,
            "resid": resid,
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--cuts-json", required=True)
    ap.add_argument("--rank", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cands = json.loads(Path(args.cuts_json).read_text())
    summary = []

    for cand in cands:
        cut = float(cand["cut"])
        rows = run_direction(args.train_manifest, args.test_manifest, bulk_cut=cut, rank=args.rank)
        ok = [r for r in rows if r["status"] == "ok"]
        if not ok:
            continue
        summary.append({
            "cut": cut,
            "n_ok": len(ok),
            "worst_resid": max(float(r["resid"]) for r in ok),
            "worst_cos": min(float(r["cos"]) for r in ok),
            "rows": rows,
        })

    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "n_cuts_scored": len(summary),
        "out": str(pp.resolve()),
        "best": sorted(
            [{"cut": x["cut"], "worst_resid": x["worst_resid"], "worst_cos": x["worst_cos"], "n_ok": x["n_ok"]} for x in summary],
            key=lambda z: (z["worst_resid"], -z["worst_cos"])
        )[:5],
    }, indent=2))

if __name__ == "__main__":
    main()
