#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def cos_resid(v, basis):
    v = np.asarray(v, dtype=np.float64)
    B = np.asarray(basis, dtype=np.float64)
    coeff = B @ v
    proj = coeff @ B
    nv = np.linalg.norm(v)
    npj = np.linalg.norm(proj)
    cos = 0.0 if nv == 0 or npj == 0 else float(np.dot(v, proj) / (nv * npj))
    resid = 0.0 if nv == 0 else float(np.linalg.norm(v - proj) / nv)
    return cos, resid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bank = json.loads(Path(args.bank).read_text())["bank"]
    rows = []
    for fp in args.inputs:
        d = json.loads(Path(fp).read_text())
        sig = float(d["state"]["sigma_plus"])
        br = "pos" if sig >= 0 else "neg"
        for sp in ("nue","nux"):
            key = f"{sp}:{br}"
            if key not in bank:
                continue
            v = d["species"][sp]["baseline"]["C_monopole"]["values"]
            cos, resid = cos_resid(v, bank[key]["basis"])
            rows.append({
                "file": str(fp),
                "species": sp,
                "branch": br,
                "cos": cos,
                "resid": resid,
            })

    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
