#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

path = Path("audit_outputs/tier2/highshear_branches.json")
rows = json.loads(path.read_text())

group = defaultdict(list)
refs = {}
nocs = {}

for r in rows:
    sigma = float(r["config"]["Sigma_H"])
    label = r.get("label")
    if label == "tier1_ref":
        refs[sigma] = r
    elif label == "tier2_nocoll":
        nocs[sigma] = r
    elif label == "tier2_nue":
        group[sigma].append(r)

report = {}
for sigma, pts in sorted(group.items()):
    pts = sorted(pts, key=lambda x: float(x["config"]["relax"]))
    ref = refs[sigma]
    noc = nocs[sigma]

    entries = []
    prev = None
    for p in pts:
        entry = {
            "relax": float(p["config"]["relax"]),
            "Yp": float(p["final"]["Yp"]),
            "DH": float(p["final"]["DH"]),
            "N_eff": float(p["final"]["N_eff"]),
            "phase1_handoff_Xn": float(p["phase1"]["phase1_handoff_Xn"]),
            "phase1_handoff_sigma_plus": float(p["phase1"]["phase1_handoff_sigma_plus"]),
            "phase1_handoff_pi_plus": float(p["phase1"]["phase1_handoff_pi_plus"]),
            "phase1_handoff_lambda_np": float(p["phase1"]["phase1_handoff_lambda_np"]),
            "phase1_handoff_lambda_pn": float(p["phase1"]["phase1_handoff_lambda_pn"]),
            "dYp_vs_tier1": float(p["final"]["Yp"]) - float(ref["final"]["Yp"]),
            "dYp_vs_tier2_nocoll": float(p["final"]["Yp"]) - float(noc["final"]["Yp"]),
            "dNeff_vs_tier1": float(p["final"]["N_eff"]) - float(ref["final"]["N_eff"]),
        }
        if prev is not None:
            entry["step"] = {
                "dYp_prev": float(p["final"]["Yp"]) - float(prev["final"]["Yp"]),
                "dDH_prev": float(p["final"]["DH"]) - float(prev["final"]["DH"]),
                "dXn_prev": float(p["phase1"]["phase1_handoff_Xn"]) - float(prev["phase1"]["phase1_handoff_Xn"]),
                "dSigma_prev": float(p["phase1"]["phase1_handoff_sigma_plus"]) - float(prev["phase1"]["phase1_handoff_sigma_plus"]),
                "dPi_prev": float(p["phase1"]["phase1_handoff_pi_plus"]) - float(prev["phase1"]["phase1_handoff_pi_plus"]),
            }
            entry["jump_flag"] = (
                abs(entry["step"]["dYp_prev"]) > 1.0e-3
                or abs(entry["step"]["dXn_prev"]) > 5.0e-4
                or abs(entry["step"]["dSigma_prev"]) > 5.0e-3
            )
        else:
            entry["step"] = None
            entry["jump_flag"] = False
        prev = p
        entries.append(entry)

    report[str(sigma)] = {
        "tier1_ref": {
            "Yp": float(ref["final"]["Yp"]),
            "DH": float(ref["final"]["DH"]),
            "N_eff": float(ref["final"]["N_eff"]),
        },
        "tier2_nocoll": {
            "Yp": float(noc["final"]["Yp"]),
            "DH": float(noc["final"]["DH"]),
            "N_eff": float(noc["final"]["N_eff"]),
        },
        "tier2_nue_continuation": entries,
    }

print(json.dumps(report, indent=2))
Path("audit_outputs/tier2/branch_jump_report.json").write_text(json.dumps(report, indent=2))
