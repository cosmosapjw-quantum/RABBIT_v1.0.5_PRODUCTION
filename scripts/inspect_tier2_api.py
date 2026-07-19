#!/usr/bin/env python3
from __future__ import annotations
import importlib
import inspect
import json

mods = [
    "rabbit.thermo.incomplete_decoupling",
    "rabbit.thermo.nudec_coupled",
    "rabbit.thermo.eos_photon_electron",
    "rabbit.thermo.qed_eos_exact",
    "rabbit.collisions.kernels",
    "rabbit.collisions.nu_e_scattering",
    "rabbit.collisions.pair_processes",
    "rabbit.collisions.projected_operator",
    "rabbit.transport.teff_collision_bridge",
]

want = (
    "entropy", "energy", "transfer", "collision", "qed", "hubble",
    "temperature", "nue", "nux", "pair", "scatter", "operator", "neff"
)

out = {}
for modname in mods:
    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        out[modname] = {"import_error": str(e)}
        continue

    rows = []
    for name in sorted(dir(mod)):
        if name.startswith("_"):
            continue
        low = name.lower()
        if not any(tok in low for tok in want):
            continue
        obj = getattr(mod, name)
        kind = "other"
        sig = None
        try:
            if inspect.isclass(obj):
                kind = "class"
            elif inspect.isfunction(obj):
                kind = "function"
            elif inspect.ismethod(obj):
                kind = "method"
            sig = str(inspect.signature(obj))
        except Exception:
            pass
        rows.append({"name": name, "kind": kind, "signature": sig})
    out[modname] = rows

print(json.dumps(out, indent=2, sort_keys=True))
