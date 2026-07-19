#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

def walk(obj: Any, prefix: str = "root") -> Iterable[Tuple[str, Any]]:
    yield prefix, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{prefix}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{prefix}[{i}]")

def maybe_array(x: Any):
    if isinstance(x, np.ndarray):
        arr = np.asarray(x, dtype=np.float64)
        return arr if arr.ndim == 1 and arr.size > 0 else None
    if isinstance(x, (list, tuple)) and len(x) > 0:
        try:
            arr = np.asarray(x, dtype=np.float64)
        except Exception:
            return None
        return arr if arr.ndim == 1 and arr.size > 0 else None
    return None

def preview(arr, n=4):
    a = np.asarray(arr, dtype=np.float64)
    head = a[:n].tolist()
    tail = a[-n:].tolist() if a.size > n else []
    return {"head": head, "tail": tail}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.dump).read_text())

    vectors = []
    scalars = {}
    for path, val in walk(data):
        arr = maybe_array(val)
        if arr is not None:
            vectors.append({
                "path": path,
                "len": int(arr.size),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "preview": preview(arr),
            })
            continue
        if isinstance(val, (int, float, str, bool)) and path.count(".") <= 3:
            scalars[path] = val

    vectors.sort(key=lambda r: (r["len"], r["path"]))
    by_len = {}
    for r in vectors:
        by_len.setdefault(str(r["len"]), []).append(r)

    out = {
        "dump": str(Path(args.dump).resolve()),
        "top_scalars": scalars,
        "n_vectors": len(vectors),
        "vector_lengths": {k: len(v) for k, v in by_len.items()},
        "vectors_by_length": by_len,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "dump": out["dump"],
        "n_vectors": out["n_vectors"],
        "vector_lengths": out["vector_lengths"],
    }, indent=2))
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()
