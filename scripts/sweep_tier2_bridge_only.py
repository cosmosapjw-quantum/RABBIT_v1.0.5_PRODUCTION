#!/usr/bin/env python3
from __future__ import annotations
import json
import subprocess
from pathlib import Path

SIGMAS = [0.0, 0.1, 0.5]
NQS = [20, 24, 32]
NMUS = [8, 12, 16]

rows = []
for sigma in SIGMAS:
    for nq in NQS:
        for nmu in NMUS:
            cmd = [
                "python", "-u", "scripts/probe_tier2_bridge_only.py",
                "--sigma", str(sigma),
                "--cl", "0",
                "--tier", "2",
                "--nq", str(nq),
                "--nmu", str(nmu),
                "--reactions", "12",
            ]
            try:
                out = subprocess.check_output(cmd, text=True, timeout=240)
                rows.append(json.loads(out))
            except Exception as e:
                rows.append({
                    "config": {"Sigma_H": sigma, "N_q": nq, "N_mu": nmu},
                    "error": repr(e),
                })

print(json.dumps(rows, indent=2))
Path("audit_outputs/tier2").mkdir(parents=True, exist_ok=True)
Path("audit_outputs/tier2/bridge_only_sweep.json").write_text(json.dumps(rows, indent=2))
