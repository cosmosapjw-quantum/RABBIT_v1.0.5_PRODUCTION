#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def sgn(x):
    return "pos" if x >= 0 else "neg"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bank = {}
    rows = []

    for fp in args.inputs:
        d = json.loads(Path(fp).read_text())
        sig = float(d["state"]["sigma_plus"])
        br = sgn(sig)
        for sp in ("nue","nux"):
            vec = np.asarray(d["species"][sp]["baseline"]["C_monopole"]["values"], dtype=np.float64)
            rows.append((sp, br, vec, fp, sig))

    for sp in ("nue","nux"):
        for br in ("pos","neg"):
            mats = [v for s,b,v,_,_ in rows if s == sp and b == br]
            if not mats:
                continue
            M = np.stack(mats, axis=0)
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            r = min(args.rank, Vt.shape[0])
            bank[f"{sp}:{br}"] = {
                "rank": r,
                "singular_values": [float(x) for x in S.tolist()],
                "basis": [[float(x) for x in row] for row in Vt[:r].tolist()],
            }

    out = {"bank": bank, "n_inputs": len(args.inputs)}
    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(out, indent=2))
    print(json.dumps({"keys": sorted(bank.keys()), "out": str(pp.resolve())}, indent=2))

if __name__ == "__main__":
    main()
