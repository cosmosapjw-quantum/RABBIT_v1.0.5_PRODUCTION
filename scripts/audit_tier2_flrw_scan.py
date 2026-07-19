#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig
from rabbit.config.transport_mode import TransportMode

CASES = [
    ("tier1_cl0",  dict(tier=1, enable_collisions=False, correction_level=0)),
    ("tier1_cl2",  dict(tier=1, enable_collisions=False, correction_level=2)),
    ("tier2_nocoll_cl0", dict(tier=2, enable_collisions=False, correction_level=0)),
    ("tier2_nocoll_cl2", dict(tier=2, enable_collisions=False, correction_level=2)),
    ("tier2_nue_cl0", dict(tier=2, enable_collisions=True, correction_level=0)),
    ("tier2_nue_cl2", dict(tier=2, enable_collisions=True, correction_level=2)),
]

rows = []
for label, kw in CASES:
    cfg = FullCoupledConfig(
        Sigma_H_plus=0.0,
        Sigma_H_minus=0.0,
        N_q=20,
        N_mu=12,
        n_reactions=12,
        enable_teff=False,
        transport_mode=TransportMode.CHARACTERISTIC,
        **kw,
    )
    try:
        res = run_full_coupled_typeI(cfg)
        rows.append({
            "label": label,
            "success": True,
            "tier": kw["tier"],
            "enable_collisions": kw["enable_collisions"],
            "correction_level": kw["correction_level"],
            "Yp": float(res.observables.Yp),
            "DH": float(res.observables.DH),
            "N_eff": float(res.observables.N_eff),
            "phase1_steps": int(res.metadata.get("phase1_steps", -1)),
            "phase2_steps": int(res.metadata.get("phase2_steps", -1)),
            "transport_mode": res.metadata.get("transport_mode"),
            "collision_flag": bool(res.metadata.get("enable_collisions", False)),
            "tier_meta": res.metadata.get("tier"),
        })
    except Exception as e:
        rows.append({
            "label": label,
            "success": False,
            "tier": kw["tier"],
            "enable_collisions": kw["enable_collisions"],
            "correction_level": kw["correction_level"],
            "error": repr(e),
        })

print(json.dumps(rows, indent=2))
Path("audit_outputs/tier2").mkdir(parents=True, exist_ok=True)
Path("audit_outputs/tier2/flrw_scan.json").write_text(json.dumps(rows, indent=2))
