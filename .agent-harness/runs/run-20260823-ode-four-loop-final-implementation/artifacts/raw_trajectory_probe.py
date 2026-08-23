from __future__ import annotations

import argparse
import json

import numpy as np

import rabbit.collisions.dynamic_collision_driver as driver
from rabbit.collisions.deterministic_reference import build_fixed_collision_quadrature


parser = argparse.ArgumentParser()
parser.add_argument("--collisions", action="store_true")
parser.add_argument("--n-q", type=int, default=24)
args = parser.parse_args()

captured = {}
real_solve_ivp = driver.solve_ivp


def capture_solve(*solve_args, **solve_kwargs):
    solution = real_solve_ivp(*solve_args, **solve_kwargs)
    captured["solution"] = solution
    return solution


driver.solve_ivp = capture_solve
exception = None
try:
    result = driver.integrate_flrw_decoupling(n_q=args.n_q, collisions=args.collisions)
except Exception as error:  # preserve the exact fail-closed boundary for the audit artifact
    result = None
    exception = {"type": type(error).__name__, "message": str(error)}

solution = captured["solution"]
quad = build_fixed_collision_quadrature(
    N_q=args.n_q,
    N_nue_y2=args.n_q,
    N_nue_y3=args.n_q,
    N_pair_y2=args.n_q,
    N_pair_leg=16,
)
q_nodes = np.asarray(quad.q_nodes, dtype=float)
n = q_nodes.size
states = np.asarray(solution.y)
occupations = states[: 2 * n]
invalid = ~np.isfinite(occupations) | (occupations <= 0.0) | (occupations >= 1.0)
entries = []
for flat_row, sample in zip(*np.nonzero(invalid), strict=True):
    bank = "nue" if flat_row < n else "nux"
    node = int(flat_row % n)
    entries.append(
        {
            "sample_index": int(sample),
            "N_hex": float(solution.t[sample]).hex(),
            "bank": bank,
            "node": node,
            "q_hex": float(q_nodes[node]).hex(),
            "value_hex": float(occupations[flat_row, sample]).hex(),
            "initial_fd_hex": float(1.0 / (np.exp(q_nodes[node]) + 1.0)).hex(),
        }
    )

payload = {
    "configuration": {
        "n_q": args.n_q,
        "collisions": args.collisions,
        "rtol": 1.0e-8,
        "atol": 1.0e-10,
        "max_step": 0.5,
        "T_gamma_stop_MeV": 1.0e-2,
    },
    "solver": {
        "success": bool(solution.success),
        "status": int(solution.status),
        "message": str(solution.message),
        "stored_points": int(solution.t.size),
        "nfev": int(solution.nfev),
        "njev": int(solution.njev),
        "nlu": int(solution.nlu),
        "terminal_event_count": int(np.asarray(solution.t_events[0]).size),
    },
    "raw_occupations": {
        "invalid_count": int(invalid.sum()),
        "samples_with_invalid": np.flatnonzero(invalid.any(axis=0)).tolist(),
        "negative_count": int((occupations < 0.0).sum()),
        "zero_count": int((occupations == 0.0).sum()),
        "at_or_above_one_count": int((occupations >= 1.0).sum()),
        "nonfinite_count": int((~np.isfinite(occupations)).sum()),
        "minimum_hex": float(np.nanmin(occupations)).hex(),
        "maximum_hex": float(np.nanmax(occupations)).hex(),
        "max_clip_excursion_hex": float(
            max(np.nanmax(occupations - 1.0), np.nanmax(-occupations), 0.0)
        ).hex(),
        "entries": entries,
    },
    "integration_exception": exception,
    "returned_result": None if result is None else {
        "N_eff_hex": float(result.N_eff).hex(),
        "N_final_hex": float(result.N_final).hex(),
        "T_gamma_final_hex": float(result.T_gamma_final).hex(),
        "max_clip_excursion_hex": float(result.max_clip_excursion).hex(),
    },
}
print(json.dumps(payload, indent=2, sort_keys=True))
