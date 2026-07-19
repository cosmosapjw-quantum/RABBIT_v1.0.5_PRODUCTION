#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from rabbit.debug.posbulk_tailsplit_contract import classify_cluster, POSBULK_TAIL_CUT_V1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.loads(Path(args.dump).read_text())
    sigma_plus = float(d["state"]["sigma_plus"])
    state_index = int(d["state_index"])

    sp_out = {}
    for sp in ("nue", "nux"):
        vec = np.asarray(d["species"][sp]["baseline"]["C_monopole"]["values"], dtype=np.float64)
        tail = float(np.sum(np.abs(vec[-5:])) / np.sum(np.abs(vec)))
        cl = classify_cluster(
            sigma_plus=sigma_plus,
            state_index=state_index,
            tail_last5_share=tail,
        )
        sp_out[sp] = {
            "tail_last5_share": tail,
            "cluster": cl,
            "authoritative_path": "raw_characteristic",
            "research_bank_status": (
                "undercovered_shadow_only" if cl == "pos_bulk_lowtail"
                else "validated_case_holdout" if cl in ("pos_onset", "pos_bulk_hightail")
                else "disabled"
            ),
        }

    out = {
        "state_index": state_index,
        "sigma_plus": sigma_plus,
        "tail_cut": POSBULK_TAIL_CUT_V1,
        "species": sp_out,
    }

    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
