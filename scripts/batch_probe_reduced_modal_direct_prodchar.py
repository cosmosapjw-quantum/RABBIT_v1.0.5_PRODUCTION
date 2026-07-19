#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--states", nargs="+", required=True, help="e.g. first mid last 0 50 100 -1")
    ap.add_argument("--shared-rank", type=int, default=2)
    ap.add_argument("--species-rank", type=int, default=1)
    ap.add_argument("--include-fd2", action="store_true")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for st in args.states:
        out = outdir / f"state_{str(st).replace('-', 'm')}.json"
        cmd = [
            sys.executable, "-u", "scripts/probe_reduced_modal_direct_prodchar.py",
            "--dump", args.dump,
            "--bank", args.bank,
            "--state", st,
            "--shared-rank", str(args.shared_rank),
            "--species-rank", str(args.species_rank),
            "--store-full-vectors",
            "--out", str(out),
        ]
        if args.include_fd2:
            cmd.append("--include-fd2")
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)
        manifest.append(str(out))

    mp = outdir / "manifest.json"
    mp.write_text(json.dumps({"files": manifest}, indent=2))
    print("[saved]", mp)

if __name__ == "__main__":
    main()
