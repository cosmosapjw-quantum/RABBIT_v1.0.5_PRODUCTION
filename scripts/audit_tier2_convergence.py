#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig
from rabbit.config.transport_mode import TransportMode

rows = []
for nq in [20, 24, 32]:
    for nmu in [8, 12, 16]:
        cfg = FullCoupledConfig(
            Sigma_H_plus=0.0,
            Sigma_H_minus=0.0,
            tier=2,
            enable_collisions=True,
            correction_level=0,
            N_q=nq,
            N_mu=nmu,
            n_reactions=12,
            enable_teff=False,
            transport_mode=TransportMode.CHARACTERISTIC,
        )
        try:
            res = run_full_coupled_typeI(cfg)
            rows.append({
                "N_q": nq,
                "N_mu": nmu,
                "Yp": float(res.observables.Yp),
                "DH": float(res.observables.DH),
                "N_eff": float(res.observables.N_eff),
            })
        except Exception as e:
            rows.append({
                "N_q": nq,
                "N_mu": nmu,
                "error": repr(e),
            })

print(json.dumps(rows, indent=2))
Path("audit_outputs/tier2").mkdir(parents=True, exist_ok=True)
Path("audit_outputs/tier2/convergence.json").write_text(json.dumps(rows, indent=2))
