#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def collect_vectors(paths):
    out = {"nue": [], "nux": []}
    meta = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        spv = d["state"]["sigma_plus"]
        meta.append({"file": str(p), "sigma_plus": spv})
        for sp in ("nue","nux"):
            # reduced가 아니라 baseline C를 bank input으로 쓴다.
            # reduced bank가 raw geometry를 못 잡는지 먼저 보기 위함.
            # C_monopole summary밖에 없으면 실패하니 json 구조 확인용으로 guard.
            pass
    return out, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # 현재 direct_prodchar json에는 full vector가 없고 summary만 있다.
    # 그래서 이 PR은 '실패를 명시적으로 문서화'하는 offline scaffold 역할도 한다.
    out = {
        "status": "scaffold_only",
        "reason": "direct_prodchar json currently stores summaries, not full C_monopole vectors",
        "next_required_patch": "store full baseline/reduced C_monopole vectors for offline SVD bank build",
        "inputs": [str(Path(x)) for x in args.inputs],
    }

    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("[saved]", pp)

if __name__ == "__main__":
    main()
