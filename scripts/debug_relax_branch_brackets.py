#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "audit_outputs" / "branch_debug" / "relax_branch_scan_summary.json"

def find_brackets(seq, key_sign):
    out = []
    for a, b in zip(seq[:-1], seq[1:]):
        sa = a.get(key_sign)
        sb = b.get(key_sign)
        if sa is None or sb is None:
            continue
        if sa == 0 or sb == 0:
            out.append({
                "from_relax": a["relax"],
                "to_relax": b["relax"],
                "kind": "touches_zero",
                "from_sign": sa,
                "to_sign": sb,
            })
        elif sa != sb:
            out.append({
                "from_relax": a["relax"],
                "to_relax": b["relax"],
                "kind": "sign_flip",
                "from_sign": sa,
                "to_sign": sb,
            })
    return out

def main():
    data = json.loads(INFILE.read_text())
    out = {}

    for sigma, modes in data.items():
        out[sigma] = {}
        for mode, seq in modes.items():
            out[sigma][mode] = {
                "sigma_plus_crossings": find_brackets(seq, "sigma_sign"),
                "pi_plus_crossings": find_brackets(seq, "pi_sign"),
            }

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
