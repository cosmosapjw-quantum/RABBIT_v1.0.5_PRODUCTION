#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-loo", required=True)
    ap.add_argument("--bank", required=True)
    args = ap.parse_args()

    rows = json.loads(Path(args.cluster_loo).read_text())
    bank = json.loads(Path(args.bank).read_text())

    keys = sorted(bank["banks"].keys())
    print("bank keys =", keys)

    if any(k.startswith("neg:") for k in keys):
        print("[FAIL] negative bank should not exist", file=sys.stderr)
        sys.exit(1)

    pos_rows = [r for r in rows if r["status"] == "ok" and r["cluster"] in ("pos_onset", "pos_bulk")]
    if not pos_rows:
        print("[FAIL] no positive clustered rows found", file=sys.stderr)
        sys.exit(1)

    worst_resid = max(float(r["resid"]) for r in pos_rows)
    worst_cos = min(float(r["cos"]) for r in pos_rows)

    print("worst_positive_cluster_resid =", worst_resid)
    print("worst_positive_cluster_cos   =", worst_cos)

    if worst_resid > 0.10:
        print("[FAIL] cut40 positive cluster contract failed on resid", file=sys.stderr)
        sys.exit(1)
    if worst_cos < 0.99:
        print("[FAIL] cut40 positive cluster contract failed on cos", file=sys.stderr)
        sys.exit(1)

    print("[OK] cut40 positive contract passed")

if __name__ == "__main__":
    main()
