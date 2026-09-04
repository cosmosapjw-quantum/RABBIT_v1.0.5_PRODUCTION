#!/usr/bin/env python3
"""Order-eight pair-channel reference from unchanged D-080B; no retained data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import mpmath as mp
import numpy as np
from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d080b_tgamma_collision import (
    evaluate_tgamma_collision_action_jvp, _electron_matrix_raw,
)

PINS = {
    "src/rabbit/decoupling/_independent_noqke.py": "de44feee0aa484abe26976c7dc34c579643005b5",
    "scripts/audit/_d080b_tgamma_collision.py": "78489c43f3046db09d8ba2d96070124ed7b0aa91",
    "scripts/audit/_d080_tgamma_primitives.py": "c585d5865fd68a90a04a76ab540b8437fba8cfce",
}

def bits(x: float) -> str:
    return struct.pack(">d", float(x)).hex()


def array(x: object) -> dict:
    values = np.asarray(x, dtype=np.float64)
    assert np.all(np.isfinite(values))
    return {"shape": list(values.shape), "bits": [bits(v) for v in values.ravel()]}


def pauli_probes() -> list:
    mp.mp.dps = 100
    cases = [([-2., -3., -2., -3.], [0., 0., .5, .75]),
             ([-2., -3., -2.0000000001, -3.], [.1, -.2, .3, .4]),
             ([-40., -30., -20., -10.], [0., 0., 1., 2.]),
             ([40., -30., -20., 10.], [0., 0., 1., 2.]),
             ([-1000., -900., -800., -700.], [0., 0., 1., 2.])]
    result = []
    for u, v in cases:
        um = list(map(mp.mpf, u)); vm = list(map(mp.mpf, v))
        def value(t):
            f = [1 / (1 + mp.exp(-(a + t*b))) for a, b in zip(um, vm)]
            return (1-f[0])*(1-f[1])*f[2]*f[3] - f[0]*f[1]*(1-f[2])*(1-f[3])
        derivative = mp.diff(value, mp.mpf(0))
        result.append({"logits": array(u), "direction": array(v), "derivative": bits(derivative)})
    return result


def build() -> dict:
    for path, expected in PINS.items():
        assert subprocess.check_output(["git", "hash-object", path], text=True).strip() == expected
    grid = ind.build_independent_grid(order=8, y_max=8.)
    config = ind.IndependentCollisionConfig()
    family_names = ["e:pair", "mu:pair", "tau:pair"]
    support, corrected = [], []
    for y1 in grid.nodes:
        batch = ind._two_body_kinematics(p1=2.*float(y1), p2_nodes=2.*grid.nodes,
            p2_weights=2.*grid.weights, mass2=0., mass3=ind.M_ELECTRON_MEV,
            mass4=ind.M_ELECTRON_MEV, config=config)
        support.extend(batch.support.ravel().tolist())
        for event in ind.independent_electron_events()[12:]:
            raw, _ = _electron_matrix_raw(event.target, "pair", batch, ind.M_ELECTRON_MEV)
            corrected.extend((raw < 0.).ravel().tolist())
    cases = []
    for name, tg in [("equilibrium", 2.), ("thermal_split", 2.05), ("mu_tau_split", 2.05)]:
        logits = np.tile(-grid.nodes, (3, 1))
        if name == "mu_tau_split":
            logits[1] += .03 * np.cos(grid.nodes)
            logits[2] -= .03 * np.cos(grid.nodes)
        occupation = 1. / (1. + np.exp(-logits))
        state = np.log(-np.log1p(-occupation))
        result = evaluate_tgamma_collision_action_jvp(grid=grid, pair_cloglog=state,
            temperature_cm_mev=2., temperature_gamma_mev=tg, config=config)
        base_pair = sum((result.base.electron_families[k] for k in family_names), np.zeros((6, 8)))
        qem = sum(result.electron_bath_energy_by_family[k] for k in family_names)
        cases.append({"name": name, "tg": bits(tg), "state": array(state),
            "pair_native": array(result.pair), "base_pair_native": array(base_pair),
            "family_native": [array(result.electron_families[k]) for k in family_names],
            "qem": bits(qem), "support": support, "corrected": corrected})
    return {"scope": "P1_PAIR_ONLY_ORDER8_NO_RETAINED", "d080b_blob": PINS["scripts/audit/_d080b_tgamma_collision.py"],
        "pins": PINS, "cases": cases, "pauli_probes": pauli_probes()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    encoded = (json.dumps(build(), sort_keys=True, indent=2) + "\n").encode()
    args.output.write_bytes(encoded)
    print("pair_oracle_sha256=" + hashlib.sha256(encoded).hexdigest())
