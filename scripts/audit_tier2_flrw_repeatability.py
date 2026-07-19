#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig
from rabbit.config.transport_mode import TransportMode

rows = []
for i in range(3):
    cfg = FullCoupledConfig(
        Sigma_H_plus=0.0,
        Sigma_H_minus=0.0,
        tier=2,
        enable_collisions=True,
        correction_level=0,
        N_q=20,
        N_mu=12,
        n_reactions=12,
        enable_teff=False,
        transport_mode=TransportMode.CHARACTERISTIC,
    )
    res = run_full_coupled_typeI(cfg)
    rows.append({
        "run": i + 1,
        "Yp": float(res.observables.Yp),
        "DH": float(res.observables.DH),
        "N_eff": float(res.observables.N_eff),
        "phase1_steps": int(res.metadata.get("phase1_steps", -1)),
        "phase2_steps": int(res.metadata.get("phase2_steps", -1)),
    })

print(json.dumps(rows, indent=2))
Path("audit_outputs/tier2").mkdir(parents=True, exist_ok=True)
Path("audit_outputs/tier2/flrw_repeatability.json").write_text(json.dumps(rows, indent=2))
