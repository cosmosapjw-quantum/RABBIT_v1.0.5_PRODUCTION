#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def sgn(x):
    return "pos" if x >= 0 else "neg"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--rank", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for mp in args.manifest:
        files = json.loads(Path(mp).read_text())["files"]
        for fp in files:
            d = json.loads(Path(fp).read_text())
            sig = float(d["state"]["sigma_plus"])
            br = sgn(sig)
            for sp in ("nue","nux"):
                vec = np.asarray(d["species"][sp]["baseline"]["C_monopole"]["values"], dtype=np.float64)
                rows.append((sp, br, vec, fp, sig))

    bank = {}
    for sp in ("nue","nux"):
        for br in ("pos","neg"):
            mats = [v for s,b,v,_,_ in rows if s == sp and b == br]
            srcs = [f for s,b,_,f,_ in rows if s == sp and b == br]
            if len(mats) < 2:
                continue
            M = np.stack(mats, axis=0)
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            r = min(args.rank, Vt.shape[0])
            bank[f"{sp}:{br}"] = {
                "n_samples": len(mats),
                "rank": r,
                "sources": srcs,
                "singular_values": [float(x) for x in S.tolist()],
                "basis": [[float(x) for x in row] for row in Vt[:r].tolist()],
            }

    out = {"bank": bank}
    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(out, indent=2))
    print(json.dumps({"keys": sorted(bank.keys()), "out": str(pp.resolve())}, indent=2))

if __name__ == "__main__":
    main()
