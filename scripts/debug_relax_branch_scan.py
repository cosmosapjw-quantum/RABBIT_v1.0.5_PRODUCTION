#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "audit_outputs" / "branch_debug"
OUTDIR.mkdir(parents=True, exist_ok=True)

SHARED_SCRIPT = ROOT / "scripts" / "trace_scipy_typeI_phase1_external.py"
SPECIES_SCRIPT = ROOT / "scripts" / "trace_species_rays_phase1.py"

# control + problematic regions
RELAX_GRID = {
    0.3: [0.00, 0.02, 0.05, 0.10, 0.20, 0.40, 1.00],
    0.5: [0.00, 0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.30, 0.40, 0.60, 1.00],
    0.7: [0.00, 0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.30, 0.40, 0.60, 1.00],
}

def run_shared(sigma: float, relax: float):
    out = OUTDIR / f"tmp_shared_sigma{sigma:.3f}_relax{relax:.3f}.json"
    env = os.environ.copy()
    env["RABBIT_COLLISION_BRIDGE_RELAX"] = str(relax)

    cmd = [
        PYTHON, "-u", str(SHARED_SCRIPT),
        "--sigma", str(sigma),
        "--cl", "0",
        "--nq", "20",
        "--reactions", "12",
        "--out", str(out),
    ]
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)
    data = json.loads(out.read_text())
    ch = data["characteristic"]

    return {
        "Xn": ch.get("phase1_handoff_Xn"),
        "sigma_plus": ch.get("phase1_handoff_sigma_plus"),
        "pi_plus": ch.get("phase1_handoff_pi_plus_total", ch.get("phase1_handoff_pi_plus")),
        "lambda_np": ch.get("phase1_handoff_lambda_np"),
        "lambda_pn": ch.get("phase1_handoff_lambda_pn"),
        "T_gamma": ch.get("phase1_handoff_T"),
        "N": ch.get("phase1_handoff_N"),
    }

def run_species(sigma: float, relax: float):
    out = OUTDIR / f"tmp_species_sigma{sigma:.3f}_relax{relax:.3f}.json"
    cmd = [
        PYTHON, "-u", str(SPECIES_SCRIPT),
        "--sigma", str(sigma),
        "--relax", str(relax),
        "--cl", "0",
        "--nq", "20",
        "--nmu", "12",
        "--out", str(out),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    data = json.loads(out.read_text())
    ph = data["phase1"]

    return {
        "Xn": ph.get("phase1_handoff_Xn"),
        "sigma_plus": ph.get("phase1_handoff_sigma_plus"),
        "pi_plus": ph.get("phase1_handoff_pi_plus_total"),
        "lambda_np": ph.get("phase1_handoff_lambda_np"),
        "lambda_pn": ph.get("phase1_handoff_lambda_pn"),
        "T_gamma": ph.get("phase1_handoff_T_gamma"),
        "N": ph.get("phase1_handoff_N"),
        "bridge_debug": ph.get("phase1_handoff_bridge_debug", {}),
    }

def sign(x):
    if x is None:
        return None
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def main():
    rows = []
    summary = {}

    for sigma, relaxes in RELAX_GRID.items():
        sigma_key = f"{sigma:.1f}"
        summary[sigma_key] = {"shared": [], "species": []}

        for relax in relaxes:
            shared = run_shared(sigma, relax)
            species = run_species(sigma, relax)

            rows.append({
                "Sigma_H": sigma,
                "relax": relax,
                "shared": shared,
                "species": species,
            })

            summary[sigma_key]["shared"].append({
                "relax": relax,
                "sigma_plus": shared["sigma_plus"],
                "sigma_sign": sign(shared["sigma_plus"]),
                "pi_plus": shared["pi_plus"],
                "pi_sign": sign(shared["pi_plus"]),
                "Xn": shared["Xn"],
                "lambda_np": shared["lambda_np"],
            })
            summary[sigma_key]["species"].append({
                "relax": relax,
                "sigma_plus": species["sigma_plus"],
                "sigma_sign": sign(species["sigma_plus"]),
                "pi_plus": species["pi_plus"],
                "pi_sign": sign(species["pi_plus"]),
                "Xn": species["Xn"],
                "lambda_np": species["lambda_np"],
            })

    out_rows = OUTDIR / "relax_branch_scan_rows.json"
    out_summary = OUTDIR / "relax_branch_scan_summary.json"
    out_rows.write_text(json.dumps(rows, indent=2))
    out_summary.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"[saved] {out_rows}")
    print(f"[saved] {out_summary}")

if __name__ == "__main__":
    main()
