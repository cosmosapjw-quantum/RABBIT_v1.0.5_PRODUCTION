#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {}
    for jp in args.json:
        data = json.loads(Path(jp).read_text())
        rows = data["rows"]
        stats = {}
        for sp in ("nue","nux"):
            vals = [abs(r["species"][sp]["qdot_shape_np"]) for r in rows]
            tails = []
            for r in rows:
                tt = r["species"][sp].get("topk_terms", [])
                denom = sum(abs(x["term"]) for x in tt) or 1.0
                numer = sum(abs(x["term"]) for x in tt[:3])
                tails.append(numer / denom)
            stats[sp] = {
                "max_abs_qdot_shape_np": max(vals),
                "median_abs_qdot_shape_np": sorted(vals)[len(vals)//2],
                "median_top3_share_of_top8": sorted(tails)[len(tails)//2],
            }
        out[Path(jp).stem] = stats

    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("[saved]", pp)

if __name__ == "__main__":
    main()
