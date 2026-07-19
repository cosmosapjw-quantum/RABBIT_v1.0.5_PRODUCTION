#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path

def walk(x, path="root", bad=None, stats=None):
    if bad is None:
        bad = []
    if stats is None:
        stats = {"n_float": 0, "n_bad": 0}
    if isinstance(x, dict):
        for k, v in x.items():
            walk(v, f"{path}.{k}", bad, stats)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            walk(v, f"{path}[{i}]", bad, stats)
    elif isinstance(x, float):
        stats["n_float"] += 1
        if math.isnan(x) or math.isinf(x):
            stats["n_bad"] += 1
            bad.append({"path": path, "value": repr(x)})
    return bad, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = Path(args.json_path)
    data = json.loads(p.read_text())
    bad, stats = walk(data)

    out = {
        "json_path": str(p.resolve()),
        "n_float": stats["n_float"],
        "n_bad": stats["n_bad"],
        "bad_examples": bad[:200],
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()
