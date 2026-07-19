#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

def main():
    gate = json.loads(Path("audit_outputs/pr_smoke/pr6_gate_mid.json").read_text())
    cluster_rows = json.loads(Path("audit_outputs/loo_cluster/cluster_loo_cut40.json").read_text())
    branch_rows = json.loads(Path("audit_outputs/loo/branch_loo_rank3.json").read_text())

    print("=== gate ===")
    for sp in ("nue", "nux"):
        dbg = gate["species"][sp]["reduced"]["debug"]["bridge_debug"]
        print(sp, dbg["reduced_modal_status"], dbg["authoritative_path"])

    print("\n=== branch verdict ===")
    ok = [r for r in branch_rows if r["status"] == "ok"]
    neg = [r for r in ok if r["branch"] == "neg"]
    pos = [r for r in ok if r["branch"] == "pos"]
    print("worst_neg_resid =", max(r["resid"] for r in neg))
    print("worst_pos_resid =", max(r["resid"] for r in pos))

    print("\n=== cut40 clustered positive ===")
    posc = [r for r in cluster_rows if r["status"] == "ok" and r["cluster"] in ("pos_onset", "pos_bulk")]
    print("worst_pos_cluster_resid =", max(r["resid"] for r in posc))
    print("worst_pos_cluster_cos   =", min(r["cos"] for r in posc))

    print("\nVERDICT: production=raw_only, reduced=offline_only, research=cut40_positive_only, negative=disabled")

if __name__ == "__main__":
    main()
