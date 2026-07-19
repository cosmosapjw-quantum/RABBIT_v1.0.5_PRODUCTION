#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

SIGMAS = [0.3, 0.5, 0.7]
RELAXS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0]

rows = []
for sigma in SIGMAS:
    # tier1 reference
    cmd = [
        "python", "-u", "scripts/trace_tier2_branch_point.py",
        "--sigma", str(sigma),
        "--relax", "0.0",
        "--tier", "1",
        "--cl", "0",
        "--nq", "20",
        "--nmu", "12",
        "--reactions", "12",
        "--label", "tier1_ref",
    ]
    rows.append(json.loads(subprocess.check_output(cmd, text=True)))

    # tier2 no-collision
    cmd = [
        "python", "-u", "scripts/trace_tier2_branch_point.py",
        "--sigma", str(sigma),
        "--relax", "0.0",
        "--tier", "2",
        "--cl", "0",
        "--nq", "20",
        "--nmu", "12",
        "--reactions", "12",
        "--label", "tier2_nocoll",
    ]
    rows.append(json.loads(subprocess.check_output(cmd, text=True)))

    # tier2 collision continuation
    for relax in RELAXS:
        cmd = [
            "python", "-u", "scripts/trace_tier2_branch_point.py",
            "--sigma", str(sigma),
            "--relax", str(relax),
            "--tier", "2",
            "--collisions",
            "--cl", "0",
            "--nq", "20",
            "--nmu", "12",
            "--reactions", "12",
            "--label", "tier2_nue",
        ]
        rows.append(json.loads(subprocess.check_output(cmd, text=True)))

print(json.dumps(rows, indent=2))
Path("audit_outputs/tier2").mkdir(parents=True, exist_ok=True)
Path("audit_outputs/tier2/highshear_branches.json").write_text(json.dumps(rows, indent=2))
