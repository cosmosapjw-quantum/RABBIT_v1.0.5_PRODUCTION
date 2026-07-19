#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

shared = json.loads(Path("audit_outputs/tier2/highshear_branches.json").read_text())
species = json.loads(Path("audit_outputs/species_rays/species_phase1_sweep.json").read_text())

# shared branch에서 tier2_nue만 뽑음
shared_map = {}
for r in shared:
    if r.get("label") == "tier2_nue" and r["config"]["collisions"]:
        key = (float(r["config"]["Sigma_H"]), float(r["config"]["relax"]))
        shared_map[key] = r

report = defaultdict(list)
for r in species:
    key = (float(r["config"]["Sigma_H"]), float(r["config"]["relax"]))
    if key not in shared_map:
        continue
    s = shared_map[key]
    report[str(key[0])].append({
        "relax": key[1],
        "shared_Xn": float(s["phase1"]["phase1_handoff_Xn"]),
        "species_Xn": float(r["phase1"]["phase1_handoff_Xn"]),
        "dXn_species_minus_shared": float(r["phase1"]["phase1_handoff_Xn"]) - float(s["phase1"]["phase1_handoff_Xn"]),
        "shared_sigma_plus": float(s["phase1"]["phase1_handoff_sigma_plus"]),
        "species_sigma_plus": float(r["phase1"]["phase1_handoff_sigma_plus"]),
        "dSigma_species_minus_shared": float(r["phase1"]["phase1_handoff_sigma_plus"]) - float(s["phase1"]["phase1_handoff_sigma_plus"]),
        "shared_pi_plus": float(s["phase1"]["phase1_handoff_pi_plus"]),
        "species_pi_plus_total": float(r["phase1"]["phase1_handoff_pi_plus_total"]),
        "dPi_species_minus_shared": float(r["phase1"]["phase1_handoff_pi_plus_total"]) - float(s["phase1"]["phase1_handoff_pi_plus"]),
        "shared_lambda_np": float(s["phase1"]["phase1_handoff_lambda_np"]),
        "species_lambda_np": float(r["phase1"]["phase1_handoff_lambda_np"]),
        "dLambda_np_species_minus_shared": float(r["phase1"]["phase1_handoff_lambda_np"]) - float(s["phase1"]["phase1_handoff_lambda_np"]),
    })

for sigma in report:
    report[sigma] = sorted(report[sigma], key=lambda x: x["relax"])

print(json.dumps(report, indent=2))
Path("audit_outputs/species_rays/species_vs_shared_compare.json").write_text(json.dumps(report, indent=2))
