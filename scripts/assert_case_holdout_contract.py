#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

def summarize(rows, species, cluster):
    rr = [r for r in rows if r["status"] == "ok" and r["species"] == species and r["cluster"] == cluster]
    if not rr:
        return None
    return {
        "n": len(rr),
        "worst_cos": min(float(r["cos"]) for r in rr),
        "worst_resid": max(float(r["resid"]) for r in rr),
        "mean_cos": sum(float(r["cos"]) for r in rr) / len(rr),
        "mean_resid": sum(float(r["resid"]) for r in rr) / len(rr),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    args = ap.parse_args()

    rows = []
    for fp in args.json:
        rows.extend(json.loads(Path(fp).read_text()))

    targets = {}
    for sp in ("nue", "nux"):
        for cl in ("pos_onset", "pos_bulk"):
            s = summarize(rows, sp, cl)
            if s is None:
                print(f"[FAIL] missing block for {(sp, cl)}", file=sys.stderr)
                sys.exit(1)
            targets[(sp, cl)] = s
            print((sp, cl), s)

    # Strong contract for onset
    for sp in ("nue", "nux"):
        s = targets[(sp, "pos_onset")]
        if s["worst_resid"] > 0.05 or s["worst_cos"] < 0.998:
            print(f"[FAIL] pos_onset contract failed for {sp}", file=sys.stderr)
            sys.exit(1)

    # Provisional contract for bulk
    for sp in ("nue", "nux"):
        s = targets[(sp, "pos_bulk")]
        if s["worst_resid"] > 0.20 or s["worst_cos"] < 0.98:
            print(f"[FAIL] pos_bulk provisional contract failed for {sp}", file=sys.stderr)
            sys.exit(1)

    print("[OK] case-holdout contract passed")
