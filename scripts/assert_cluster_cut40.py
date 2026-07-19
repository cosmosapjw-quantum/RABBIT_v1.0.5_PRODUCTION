#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

def metrics(path: Path):
    rows = json.loads(path.read_text())
    rows = [r for r in rows if r["status"] == "ok" and str(r.get("cluster","")).startswith("pos")]
    if not rows:
        return {"worst_resid": 1e99, "mean_resid": 1e99, "worst_cos": -1e99}
    resid = [float(r["resid"]) for r in rows]
    cos = [float(r["cos"]) for r in rows]
    return {
        "worst_resid": max(resid),
        "mean_resid": sum(resid)/len(resid),
        "worst_cos": min(cos),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="audit_outputs/loo_cluster")
    args = ap.parse_args()

    base = Path(args.dir)
    cand = sorted(base.glob("cluster_loo_cut*.json"))
    out = {}
    for p in cand:
        cut = int(p.stem.split("cut")[-1])
        out[cut] = metrics(p)

    for cut in sorted(out):
        print(cut, out[cut])

    best = min(out.items(), key=lambda kv: (kv[1]["worst_resid"], kv[1]["mean_resid"], -kv[1]["worst_cos"]))[0]
    if best != 40:
        print(f"[FAIL] best cut is {best}, expected 40", file=sys.stderr)
        sys.exit(1)
    if out[40]["worst_resid"] > 0.10:
        print(f"[FAIL] cut40 worst positive resid too large: {out[40]['worst_resid']}", file=sys.stderr)
        sys.exit(1)
    print("[OK] cut40 selected as best positive split")

if __name__ == "__main__":
    main()
