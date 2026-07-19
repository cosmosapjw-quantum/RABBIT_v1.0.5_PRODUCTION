from __future__ import annotations

import numpy as np

from rabbit.config.transport_mode import TransportMode

SPECIES = ("nue", "nuebar", "nux")


def characteristic_species_enabled(*, transport_mode, tier, enable_collisions):
    return (
        transport_mode == TransportMode.CHARACTERISTIC
        and int(tier) >= 2
        and bool(enable_collisions)
    )


def species_energy_weights(f_nu_total: float, T_nu_e: float, T_nu_x: float):
    raw = {
        "nue": 1.0 * T_nu_e**4,
        "nuebar": 1.0 * T_nu_e**4,
        "nux": 4.0 * T_nu_x**4,
    }
    norm = max(sum(raw.values()), 1e-300)
    return {k: float(f_nu_total * raw[k] / norm) for k in raw}


def layout_characteristic_species(N_mu: int, start: int = 2):
    idx = {}
    p = start
    idx["S"] = p
    p += 1
    for sp in SPECIES:
        idx[f"I_{sp}"] = slice(p, p + N_mu)
        p += N_mu
        idx[f"J_{sp}"] = slice(p, p + N_mu)
        p += N_mu
    return idx, p
