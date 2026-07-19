#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, inspect, json
from pathlib import Path

def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("reduced_modal_wrapper_probe", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mod = load_module(Path(args.wrapper))
    rows = []
    for name, obj in inspect.getmembers(mod):
        if callable(obj) and getattr(obj, "__module__", None) == mod.__name__:
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "<signature unavailable>"
            lname = name.lower()
            score = 0
            for key in ("reduced", "modal", "bridge", "bank", "rank", "species"):
                if key in lname:
                    score += 1
            rows.append({
                "name": name,
                "signature": sig,
                "score": score,
            })
    rows.sort(key=lambda r: (-r["score"], r["name"]))
    out = {
        "wrapper": str(Path(args.wrapper).resolve()),
        "module_name": mod.__name__,
        "candidate_callables": rows,
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()
