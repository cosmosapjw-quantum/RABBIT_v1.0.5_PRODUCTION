#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

BASELINES = {
    "train05_test07": {
        "nue_pos_bulk_worst_resid": 0.19005912890677362,
        "nux_pos_bulk_worst_resid": 0.18848634556123536,
    },
    "train07_test05": {
        "nue_pos_bulk_worst_resid": 0.0012578015498275464,
        "nux_pos_bulk_worst_resid": 0.0010971217783316714,
    },
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs=2, required=True)
    args = ap.parse_args()

    names = ["train05_test07", "train07_test05"]
    improved_any = False

    for name, fp in zip(names, args.json):
        rows = json.loads(Path(fp).read_text())
        rows = sorted(rows, key=lambda x: (x["worst_resid"], -x["worst_cos"]))
        if not rows:
            print(f"[FAIL] no rows in {fp}", file=sys.stderr)
            sys.exit(1)
        best = rows[0]
        print(name, "best_cut=", best["cut"], "worst_resid=", best["worst_resid"], "worst_cos=", best["worst_cos"])
        if name == "train05_test07" and best["worst_resid"] < 0.19005912890677362:
            improved_any = True

    if not improved_any:
        print("[FAIL] no meaningful improvement over unsplit pos_bulk in the hard direction", file=sys.stderr)
        sys.exit(1)

    print("[OK] pos_bulk split gives at least directional improvement")
