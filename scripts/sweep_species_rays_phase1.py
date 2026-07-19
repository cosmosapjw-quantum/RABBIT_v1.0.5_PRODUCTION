#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

SIGMAS = [0.3, 0.5, 0.7]
RELAXS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0]

rows = []
for sigma in SIGMAS:
    for relax in RELAXS:
        cmd = [
            "python", "-u", "scripts/trace_species_rays_phase1.py",
            "--sigma", str(sigma),
            "--relax", str(relax),
            "--cl", "0",
            "--nq", "20",
            "--nmu", "12",
        ]
        rows.append(json.loads(subprocess.check_output(cmd, text=True)))

print(json.dumps(rows, indent=2))
Path("audit_outputs/species_rays").mkdir(parents=True, exist_ok=True)
Path("audit_outputs/species_rays/species_phase1_sweep.json").write_text(json.dumps(rows, indent=2))
