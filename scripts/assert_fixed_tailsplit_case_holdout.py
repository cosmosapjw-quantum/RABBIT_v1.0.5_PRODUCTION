#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

TARGET_CUT = 0.8245908567511172

def load_best(fp):
    rows = json.loads(Path(fp).read_text())
    row = None
    for r in rows:
        if abs(float(r["cut"]) - TARGET_CUT) < 1e-12:
            row = r
            break
    if row is None:
        raise RuntimeError(f"target cut {TARGET_CUT} not found in {fp}")
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs=2, required=True)
    args = ap.parse_args()

    ok = True
    for fp in args.json:
        row = load_best(fp)
        rr = row["rows"]

        onset = [r for r in rr if r["status"] == "ok" and r["cluster"] == "pos_onset"]
        high  = [r for r in rr if r["status"] == "ok" and r["cluster"] == "pos_bulk_hightail"]
        low_bad = [r for r in rr if r["cluster"] == "pos_bulk_lowtail"]

        onset_worst_resid = max(float(r["resid"]) for r in onset) if onset else math.inf
        onset_worst_cos = min(float(r["cos"]) for r in onset) if onset else -math.inf
        high_worst_resid = max(float(r["resid"]) for r in high) if high else math.inf
        high_worst_cos = min(float(r["cos"]) for r in high) if high else -math.inf

        print("\n==", fp, "==")
        print("onset worst_resid =", onset_worst_resid, "worst_cos =", onset_worst_cos, "n_ok =", len(onset))
        print("high  worst_resid =", high_worst_resid, "worst_cos =", high_worst_cos, "n_ok =", len(high))
        print("lowtail rows =", len(low_bad), "statuses =", [r["status"] for r in low_bad])

        if onset_worst_resid > 0.05 or onset_worst_cos < 0.998:
            ok = False
        if high and (high_worst_resid > 0.05 or high_worst_cos < 0.998):
            ok = False
        if any(r["status"] == "ok" for r in low_bad):
            # lowtail is not supposed to be promoted yet
            ok = False

    if not ok:
        print("[FAIL] fixed tailsplit contract failed", file=sys.stderr)
        sys.exit(1)
    print("\n[OK] fixed tailsplit case-holdout contract passed")
