#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

TARGET_CUT = 0.8245908567511172

def pick_cut(fp):
    rows = json.loads(Path(fp).read_text())
    for r in rows:
        if abs(float(r["cut"]) - TARGET_CUT) < 1e-12:
            return r
    raise RuntimeError(f"target cut {TARGET_CUT} not found in {fp}")

def main():
    cand = json.loads(Path("audit_outputs/pr20/posbulk_tail_split_candidates.json").read_text())
    a = pick_cut("audit_outputs/pr21/train05_test07_tailcuts.json")
    b = pick_cut("audit_outputs/pr21/train07_test05_tailcuts.json")

    print("=== fixed split ===")
    print("cut =", TARGET_CUT)

    print("\n=== candidate summary ===")
    chosen = [x for x in cand if abs(float(x["cut"]) - TARGET_CUT) < 1e-12][0]
    print("n_low =", chosen["n_low"], "n_high =", chosen["n_high"])
    print("low_worst_resid =", chosen["low_worst_resid"])
    print("high_worst_resid =", chosen["high_worst_resid"])

    print("\n=== holdout train05->07 ===")
    print("worst_resid =", a["worst_resid"], "worst_cos =", a["worst_cos"], "n_ok =", a["n_ok"])

    print("\n=== holdout train07->05 ===")
    print("worst_resid =", b["worst_resid"], "worst_cos =", b["worst_cos"], "n_ok =", b["n_ok"])

    print("\nVERDICT")
    print("production = raw_only")
    print("research = pos_onset + pos_bulk_hightail validated")
    print("pos_bulk_lowtail = undercovered_shadow_only")
    print("negative = disabled")
