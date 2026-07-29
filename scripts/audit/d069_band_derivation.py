#!/usr/bin/env python3
"""Regenerate the D-069 pre-freeze band derivation (disclosed equilibrium state).

Deterministic and anchor-free: it uses only f_eq = 1/(1+e^y), the two candidate
independent grids, and the frozen Rust rule's nodes and weights. It touches no
trajectory output, which is what makes it admissible as pre-freeze evidence.

Usage: PYTHONPATH=src:scripts/audit venv/bin/python scripts/audit/d069_band_derivation.py \
           --out scripts/audit/d069_band_derivation.json
"""
from __future__ import annotations

import argparse
import json
import os

for _pin in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_pin] = "1"

import numpy as np  # noqa: E402

import _f10c2_anchors as A  # noqa: E402
import _trajectory_core as core  # noqa: E402

POWERS = (2, 3, 4, 5)
EXACT = {2: 1.8030853547393686, 3: 7 * np.pi**4 / 120}  # 1.5*zeta(3), 7pi^4/120


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rust_y = np.asarray([row[0] for row in A.RUST_SPECTRUM])
    rust_w = np.asarray([row[1] for row in A.RUST_SPECTRUM])
    f_rust = 1.0 / (1.0 + np.exp(rust_y))

    out = {
        "purpose": "pre-freeze derivation of the D-069 T8a/T8b bands and "
                   "validation of the T13 analytic tail correction",
        "disclosed_state": "equilibrium f = 1/(1+exp(y)); no trajectory, no "
                           "endpoint, no anchor spectrum used",
        "rust_rule": {
            "nodes": int(rust_y.size),
            "y_max": float(rust_y[-1]),
            "sum_weights": float(rust_w.sum()),
            "vs_exact_M2_rel": float(np.sum(rust_w * rust_y**2 * f_rust) / EXACT[2] - 1.0),
            "vs_exact_M3_rel": float(np.sum(rust_w * rust_y**3 * f_rust) / EXACT[3] - 1.0),
        },
        "setups": {},
    }
    for order, y_max in ((48, 24.0), (60, 30.0)):
        setup = core.build_setup(order, y_max)
        y, w = setup.grid.nodes, setup.grid.weights
        f_ind = 1.0 / (1.0 + np.exp(y))
        rows = {}
        for k in POWERS:
            m_ind = float(np.sum(w * y**k * f_ind))
            m_rust = float(np.sum(rust_w * rust_y**k * f_rust))
            tail = core.equilibrium_tail_fraction(y_max, power=k)
            rows[f"M{k}"] = {
                "independent": m_ind,
                "rust": m_rust,
                "rel_raw": m_ind / m_rust - 1.0,
                "analytic_tail_fraction": tail,
                "rel_after_tail_correction": m_ind / (1.0 - tail) / m_rust - 1.0,
            }
        out["setups"][f"order{order}_ymax{y_max}"] = {
            "first_node": float(y[0]),
            "last_node": float(y[-1]),
            "node_density_per_unit_y": order / y_max,
            "moments": rows,
        }

    energy_weight = rust_w * rust_y**3
    out["subsupport_energy_weight_share_equilibrium"] = float(
        np.sum(energy_weight[:2] * f_rust[:2]) / np.sum(energy_weight * f_rust)
    )
    out["collocation_order_needed_to_cover_first_rust_node"] = float(
        48.0 * np.sqrt(0.014747912970887178 / float(rust_y[0]))
    )
    with open(args.out, "w") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
