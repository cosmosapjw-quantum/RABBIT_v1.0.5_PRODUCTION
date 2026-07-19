#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from rabbit.debug.modal_contract import cluster_label, ONSET_MAX_STATE_INDEX, research_allows_cluster_bank

def load_files(manifests):
    files = []
    for mp in manifests:
        files.extend(json.loads(Path(mp).read_text())["files"])
    return files

def build_basis(vectors, rank):
    M = np.stack(vectors, axis=0)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(rank, Vt.shape[0], M.shape[0], M.shape[1])
    return Vt[:r], S[:r]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--rank", type=int, default=3)
    ap.add_argument("--onset-max-state-index", type=int, default=ONSET_MAX_STATE_INDEX)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    grouped = {}
    for fp in load_files(args.manifest):
        d = json.loads(Path(fp).read_text())
        sigma_plus = float(d["state"]["sigma_plus"])
        state_index = int(d["state_index"])
        cluster = cluster_label(
            sigma_plus=sigma_plus,
            state_index=state_index,
            onset_max_state_index=args.onset_max_state_index,
        )
        if not research_allows_cluster_bank(cluster=cluster):
            continue
        for sp in ("nue", "nux"):
            vec = np.asarray(d["species"][sp]["baseline"]["C_monopole"]["values"], dtype=np.float64)
            grouped.setdefault((cluster, sp), []).append((fp, vec))

    bank = {
        "policy": {
            "onset_max_state_index": args.onset_max_state_index,
            "rank": args.rank,
            "clusters": ["pos_onset", "pos_bulk"],
            "authoritative_path": "raw_characteristic",
            "mode": "research_only",
        },
        "banks": {}
    }

    for (cluster, sp), rows in sorted(grouped.items()):
        files = [x[0] for x in rows]
        vecs = [x[1] for x in rows]
        basis, svals = build_basis(vecs, args.rank)
        bank["banks"][f"{cluster}:{sp}"] = {
            "n_samples": len(vecs),
            "basis": [[float(y) for y in x] for x in basis.tolist()],
            "singular_values": [float(x) for x in svals.tolist()],
            "files": files,
        }

    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(bank, indent=2))
    print(json.dumps({
        "out": str(pp.resolve()),
        "keys": sorted(bank["banks"].keys())
    }, indent=2))

if __name__ == "__main__":
    main()
