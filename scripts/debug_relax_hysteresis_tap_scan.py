#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def parse_grid(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def run_one(target: str, sigma: float, relax: float, nq: int, nmu: int, cl: int, outdir: Path):
    out = outdir / f"{target}_sigma{sigma:.3f}_relax{relax:.3f}.json"
    cmd = [
        PYTHON, "-u", str(ROOT / "scripts" / "debug_phase1_runtime_tap.py"),
        "--target", target,
        "--sigma", str(sigma),
        "--relax", str(relax),
        "--cl", str(cl),
        "--nq", str(nq),
        "--nmu", str(nmu),
        "--out", str(out),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    data = json.loads(out.read_text())

    phase = data.get("phase1_trace", [])
    last = phase[-1] if phase else {}
    return {
        "relax": relax,
        "sigma_plus": last.get("sigma_plus"),
        "pi_plus": last.get("pi_plus"),
        "Xn": last.get("Xn"),
        "T_gamma": last.get("T_gamma"),
        "shared_bridge_calls": data["counters"].get("shared_bridge_calls", 0),
        "species_bridge_calls": data["counters"].get("species_bridge_calls", 0),
        "transport_calls": data["counters"].get("transport_calls", 0),
        "tap_json": str(out),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["shared", "species"], required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--grid", type=str, required=True)
    ap.add_argument("--cl", type=int, default=0)
    ap.add_argument("--nq", type=int, default=20)
    ap.add_argument("--nmu", type=int, default=12)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    grid = parse_grid(args.grid)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "target": args.target,
        "sigma": args.sigma,
        "grid_forward": [],
        "grid_reverse": [],
    }

    fwd_dir = out.parent / (out.stem + "_forward")
    rev_dir = out.parent / (out.stem + "_reverse")
    fwd_dir.mkdir(parents=True, exist_ok=True)
    rev_dir.mkdir(parents=True, exist_ok=True)

    for r in grid:
        summary["grid_forward"].append(run_one(args.target, args.sigma, r, args.nq, args.nmu, args.cl, fwd_dir))
    for r in list(reversed(grid)):
        summary["grid_reverse"].append(run_one(args.target, args.sigma, r, args.nq, args.nmu, args.cl, rev_dir))

    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
