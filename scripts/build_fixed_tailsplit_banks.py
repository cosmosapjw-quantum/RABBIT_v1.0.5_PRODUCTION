#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from rabbit.debug.posbulk_tailsplit_contract import (
    classify_cluster,
    research_bank_allows,
    shadow_only,
    CONTRACT_VERSION,
    POSBULK_TAIL_CUT_V1,
)

def build_basis(vectors, rank):
    M = np.stack(vectors, axis=0)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(rank, Vt.shape[0], M.shape[0], M.shape[1])
    return Vt[:r], S[:r]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--rank", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    grouped = {}
    shadow_counts = {}

    for mp in args.manifest:
        files = json.loads(Path(mp).read_text())["files"]
        for fp in files:
            d = json.loads(Path(fp).read_text())
            sigma_plus = float(d["state"]["sigma_plus"])
            state_index = int(d["state_index"])
            for sp in ("nue", "nux"):
                vec = np.asarray(d["species"][sp]["baseline"]["C_monopole"]["values"], dtype=np.float64)
                tail = float(np.sum(np.abs(vec[-5:])) / np.sum(np.abs(vec)))
                cl = classify_cluster(
                    sigma_plus=sigma_plus,
                    state_index=state_index,
                    tail_last5_share=tail,
                )
                if research_bank_allows(cl):
                    grouped.setdefault((cl, sp), []).append((fp, tail, vec))
                elif shadow_only(cl):
                    shadow_counts[(cl, sp)] = shadow_counts.get((cl, sp), 0) + 1

    out = {
        "contract_version": CONTRACT_VERSION,
        "tail_cut": POSBULK_TAIL_CUT_V1,
        "rank": args.rank,
        "banks": {},
        "shadow_only_counts": {},
    }

    for (cl, sp), rows in sorted(grouped.items()):
        vecs = [x[2] for x in rows]
        basis, svals = build_basis(vecs, args.rank)
        out["banks"][f"{cl}:{sp}"] = {
            "n_samples": len(rows),
            "tail_min": min(float(x[1]) for x in rows),
            "tail_max": max(float(x[1]) for x in rows),
            "basis": [[float(y) for y in row] for row in basis.tolist()],
            "singular_values": [float(x) for x in svals.tolist()],
            "files": [x[0] for x in rows],
        }

    for (cl, sp), n in sorted(shadow_counts.items()):
        out["shadow_only_counts"][f"{cl}:{sp}"] = n

    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "out": str(pp.resolve()),
        "bank_keys": sorted(out["banks"].keys()),
        "shadow_only_counts": out["shadow_only_counts"],
    }, indent=2))

if __name__ == "__main__":
    main()
