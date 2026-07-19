#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str):
    return json.loads(Path(path).read_text())


def pick_phase1(d):
    return d.get("phase1", d)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare baseline vs reduced-modal phase1 outputs.")
    p.add_argument("baseline_json")
    p.add_argument("reduced_json")
    args = p.parse_args()

    a = pick_phase1(load_json(args.baseline_json))
    b = pick_phase1(load_json(args.reduced_json))

    keys = [
        "phase1_handoff_Xn",
        "phase1_handoff_sigma_plus",
        "phase1_handoff_pi_plus_total",
        "phase1_handoff_lambda_np",
        "phase1_handoff_lambda_pn",
        "phase1_handoff_T_nu_e",
        "phase1_handoff_T_nu_x",
    ]

    out = {"baseline": args.baseline_json, "reduced": args.reduced_json, "delta": {}}
    for k in keys:
        if k in a and k in b:
            out["delta"][k] = float(b[k]) - float(a[k])

    dbg = b.get("phase1_handoff_bridge_debug", None)
    if dbg is not None:
        out["reduced_bridge_debug"] = dbg

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
