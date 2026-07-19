#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-loo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = json.loads(Path(args.cluster_loo).read_text())
    bulk = [r for r in rows if r.get("status") == "ok" and r.get("cluster") == "pos_bulk"]

    vals = sorted(float(r["tail_last5_share"]) for r in bulk)
    mids = []
    for i in range(len(vals) - 1):
        mids.append(0.5 * (vals[i] + vals[i + 1]))

    out = []
    for cut in mids:
        low = [r for r in bulk if float(r["tail_last5_share"]) <= cut]
        high = [r for r in bulk if float(r["tail_last5_share"]) > cut]
        if not low or not high:
            continue
        out.append({
            "cut": cut,
            "n_low": len(low),
            "n_high": len(high),
            "low_state_indices": sorted(int(r["state_index"]) for r in low),
            "high_state_indices": sorted(int(r["state_index"]) for r in high),
            "low_tail_minmax": [
                min(float(r["tail_last5_share"]) for r in low),
                max(float(r["tail_last5_share"]) for r in low),
            ],
            "high_tail_minmax": [
                min(float(r["tail_last5_share"]) for r in high),
                max(float(r["tail_last5_share"]) for r in high),
            ],
            "low_worst_resid": max(float(r["resid"]) for r in low),
            "high_worst_resid": max(float(r["resid"]) for r in high),
        })

    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(out, indent=2))

    print(json.dumps({
        "n_pos_bulk_rows": len(bulk),
        "n_candidates": len(out),
        "out": str(pp.resolve()),
        "best_candidates_by_maxblock_resid": sorted(
            [
                {
                    "cut": x["cut"],
                    "score": max(x["low_worst_resid"], x["high_worst_resid"]),
                    "n_low": x["n_low"],
                    "n_high": x["n_high"],
                }
                for x in out
            ],
            key=lambda z: (z["score"], abs(z["n_low"] - z["n_high"]))
        )[:5],
    }, indent=2))

if __name__ == "__main__":
    main()
