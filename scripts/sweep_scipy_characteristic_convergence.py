#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig
from rabbit.config.transport_mode import TransportMode

cases = []
for nmu in [8, 12, 16, 24]:
    cfg = FullCoupledConfig(
        Sigma_H_plus=0.1,
        Sigma_H_minus=0.0,
        correction_level=0,
        N_q=20,
        N_mu=nmu,
        n_reactions=12,
        enable_teff=False,
        transport_mode=TransportMode.CHARACTERISTIC,
        tier=1,
    )
    res = run_full_coupled_typeI(cfg)
    cases.append({
        "N_mu": nmu,
        "Yp": float(res.observables.Yp),
        "DH": float(res.observables.DH),
        "N_eff": float(res.observables.N_eff),
        "phase1_steps": int(res.metadata.get("phase1_steps", -1)),
        "phase2_steps": int(res.metadata.get("phase2_steps", -1)),
        "transport_mode": res.metadata.get("transport_mode"),
    })

print(json.dumps(cases, indent=2))

Path("diagnostic_outputs").mkdir(exist_ok=True)
Path("diagnostic_outputs/scipy_characteristic_convergence_nmu.json").write_text(
    json.dumps(cases, indent=2)
)
print("[saved] diagnostic_outputs/scipy_characteristic_convergence_nmu.json")
