#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from rabbit.debug.research_bank_contract import cluster_label, ONSET_MAX_STATE_INDEX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.loads(Path(args.dump).read_text())
    sigma_plus = float(d["state"]["sigma_plus"])
    state_index = int(d["state_index"])
    cl = cluster_label(
        sigma_plus=sigma_plus,
        state_index=state_index,
        onset_max_state_index=ONSET_MAX_STATE_INDEX,
    )

    out = {
        "state_index": state_index,
        "sigma_plus": sigma_plus,
        "cluster_label": cl,
        "authoritative_path": "raw_characteristic",
        "reduced_modal_mode": "offline_only",
        "notes": "shadow diagnostic only",
    }

    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
