#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig
from rabbit.config.transport_mode import TransportMode

SIGMAS = [0.0, 0.03, 0.1, 0.3, 0.5, 0.7]

rows = []
for sigma in SIGMAS:
    pack = {"Sigma_H": sigma}

    for label, tier, enable_collisions in [
        ("tier1_ref", 1, False),
        ("tier2_nue", 2, True),
    ]:
        cfg = FullCoupledConfig(
            Sigma_H_plus=sigma,
            Sigma_H_minus=0.0,
            tier=tier,
            enable_collisions=enable_collisions,
            correction_level=0,
            N_q=20,
            N_mu=12,
            n_reactions=12,
            enable_teff=False,
            transport_mode=TransportMode.CHARACTERISTIC,
        )
        try:
            res = run_full_coupled_typeI(cfg)
            pack[label] = {
                "Yp": float(res.observables.Yp),
                "DH": float(res.observables.DH),
                "N_eff": float(res.observables.N_eff),
            }
        except Exception as e:
            pack[label] = {"error": repr(e)}

    if "tier1_ref" in pack and "tier2_nue" in pack and "error" not in pack["tier1_ref"] and "error" not in pack["tier2_nue"]:
        pack["delta_tier2_minus_tier1"] = {
            "dYp": pack["tier2_nue"]["Yp"] - pack["tier1_ref"]["Yp"],
            "dDH": pack["tier2_nue"]["DH"] - pack["tier1_ref"]["DH"],
            "dNeff": pack["tier2_nue"]["N_eff"] - pack["tier1_ref"]["N_eff"],
        }

    rows.append(pack)

print(json.dumps(rows, indent=2))
Path("audit_outputs/tier2").mkdir(parents=True, exist_ok=True)
Path("audit_outputs/tier2/shear_scan.json").write_text(json.dumps(rows, indent=2))
