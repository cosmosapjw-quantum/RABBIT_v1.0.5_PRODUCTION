#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig
from rabbit.config.transport_mode import TransportMode

SIGMAS = [0.0, 0.1, 0.3]
RELAXS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0]

rows = []
for sigma in SIGMAS:
    for relax in RELAXS:
        os.environ["RABBIT_COLLISION_BRIDGE_RELAX"] = str(relax)
        t0 = time.perf_counter()
        cfg = FullCoupledConfig(
            Sigma_H_plus=sigma,
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
        try:
            res = run_full_coupled_typeI(cfg)
            rows.append({
                "Sigma_H": sigma,
                "relax": relax,
                "success": True,
                "elapsed_s": time.perf_counter() - t0,
                "Yp": float(res.observables.Yp),
                "DH": float(res.observables.DH),
                "N_eff": float(res.observables.N_eff),
                "phase1_steps": int(res.metadata.get("phase1_steps", -1)),
                "phase2_steps": int(res.metadata.get("phase2_steps", -1)),
            })
        except Exception as e:
            rows.append({
                "Sigma_H": sigma,
                "relax": relax,
                "success": False,
                "elapsed_s": time.perf_counter() - t0,
                "error": repr(e),
            })

print(json.dumps(rows, indent=2))
Path("audit_outputs/tier2").mkdir(parents=True, exist_ok=True)
Path("audit_outputs/tier2/relax_homotopy.json").write_text(json.dumps(rows, indent=2))
