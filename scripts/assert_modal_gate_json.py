#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

def get_dbg(node):
    dbg = node.get("debug", {})
    if isinstance(dbg.get("bridge_debug"), dict):
        return dbg["bridge_debug"]
    if isinstance(dbg.get("reduced_modal_debug"), dict):
        return dbg["reduced_modal_debug"]
    return dbg if isinstance(dbg, dict) else {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--expect-status", default="rejected_to_raw")
    ap.add_argument("--expect-authority", default="raw_characteristic")
    args = ap.parse_args()

    d = json.loads(Path(args.json_path).read_text())
    ok = True
    for sp in ("nue", "nux"):
        dbg = get_dbg(d["species"][sp]["reduced"])
        st = dbg.get("reduced_modal_status")
        au = dbg.get("authoritative_path")
        print(f"[{sp}] status={st} authority={au} cos={dbg.get('cos')} resid={dbg.get('resid')}")
        if st != args.expect_status or au != args.expect_authority:
            ok = False
    if not ok:
        print("[FAIL] gate assertion failed", file=sys.stderr)
        sys.exit(1)
    print("[OK] gate assertion passed")

if __name__ == "__main__":
    main()
