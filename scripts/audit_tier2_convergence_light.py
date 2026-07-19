#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig
from rabbit.config.transport_mode import TransportMode

rows = []

# 1) N_mu sweep at fixed N_q=20
for nmu in [8, 12, 16]:
    cfg = FullCoupledConfig(
        Sigma_H_plus=0.0,
        Sigma_H_minus=0.0,
        tier=2,
        enable_collisions=True,
        correction_level=0,
        N_q=20,
        N_mu=nmu,
        n_reactions=12,
        enable_teff=False,
        transport_mode=TransportMode.CHARACTERISTIC,
    )
    try:
        res = run_full_coupled_typeI(cfg)
        rows.append({
            "scan": "N_mu",
            "N_q": 20,
            "N_mu": nmu,
            "Yp": float(res.observables.Yp),
            "DH": float(res.observables.DH),
            "N_eff": float(res.observables.N_eff),
        })
    except Exception as e:
        rows.append({
            "scan": "N_mu",
            "N_q": 20,
            "N_mu": nmu,
            "error": repr(e),
        })

# 2) N_q sweep at fixed N_mu=12
for nq in [20, 24]:
    cfg = FullCoupledConfig(
        Sigma_H_plus=0.0,
        Sigma_H_minus=0.0,
        tier=2,
        enable_collisions=True,
        correction_level=0,
        N_q=nq,
        N_mu=12,
        n_reactions=12,
        enable_teff=False,
        transport_mode=TransportMode.CHARACTERISTIC,
    )
    try:
        res = run_full_coupled_typeI(cfg)
        rows.append({
            "scan": "N_q",
            "N_q": nq,
            "N_mu": 12,
            "Yp": float(res.observables.Yp),
            "DH": float(res.observables.DH),
            "N_eff": float(res.observables.N_eff),
        })
    except Exception as e:
        rows.append({
            "scan": "N_q",
            "N_q": nq,
            "N_mu": 12,
            "error": repr(e),
        })

print(json.dumps(rows, indent=2))
Path("audit_outputs/tier2").mkdir(parents=True, exist_ok=True)
Path("audit_outputs/tier2/convergence_light.json").write_text(json.dumps(rows, indent=2))
