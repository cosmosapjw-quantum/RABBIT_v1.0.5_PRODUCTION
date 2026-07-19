from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from rabbit.config.grids import MomentumGrid
from rabbit.transport.stageAB_state import AxisymmetricHierarchyState, fermi_dirac
from rabbit.transport.typeI_stageA_hierarchy import STAGE_A_ELLS, compute_hierarchy_rhs_typeI_stageA
from rabbit.transport.typeI_stageB_hierarchy import STAGE_B_ELLS, compute_hierarchy_rhs_typeI_stageB

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "session6_outputs"
OUTDIR.mkdir(exist_ok=True)


def integrate_stage(stage: str, sigma_h: float, n_q: int, n_end: float = 0.5) -> dict:
    grid = MomentumGrid(N_q=n_q)
    quad_weight = grid.weights * np.exp(grid.nodes)
    quad_norm = float(np.sum(quad_weight))

    def q_weighted_l2(values: np.ndarray) -> float:
        return float(np.sqrt(np.sum(quad_weight * values**2) / quad_norm))
    if stage == "A":
        active_ells = STAGE_A_ELLS
        state0 = AxisymmetricHierarchyState.from_fd_equilibrium(grid, active_ells, n_species=1)
        rhs_fn = compute_hierarchy_rhs_typeI_stageA
    elif stage == "B":
        active_ells = STAGE_B_ELLS
        state0 = AxisymmetricHierarchyState.from_fd_equilibrium(grid, active_ells, n_species=1)
        rhs_fn = compute_hierarchy_rhs_typeI_stageB
    else:
        raise ValueError(stage)

    def rhs(_n, y):
        st = AxisymmetricHierarchyState.from_flat(y, grid, active_ells, n_species=1)
        return rhs_fn(st, Sigma_H=sigma_h)

    sol = solve_ivp(rhs, (0.0, n_end), state0.to_flat(), method="RK45", rtol=1e-8, atol=1e-10)
    statef = AxisymmetricHierarchyState.from_flat(sol.y[:, -1], grid, active_ells, n_species=1)
    fd = fermi_dirac(grid.nodes)
    f0 = statef.moment(0, 0)
    f2 = statef.moment(0, 2)
    metrics = {
        "stage": stage,
        "Sigma_H": sigma_h,
        "N_q": n_q,
        "success": bool(sol.success),
        "nfev": int(sol.nfev),
        "n_steps": int(sol.t.size),
        "F0_L2": q_weighted_l2(f0),
        "deltaF0_L2": q_weighted_l2(f0 - fd),
        "F2_L2": q_weighted_l2(f2),
        "F2_min": float(np.min(f2)),
        "F2_max": float(np.max(f2)),
    }
    if stage == "B":
        f4 = statef.moment(0, 4)
        metrics["F4_L2"] = q_weighted_l2(f4)
        metrics["F4_min"] = float(np.min(f4))
        metrics["F4_max"] = float(np.max(f4))
    return metrics


def main() -> None:
    rows = []
    for n_q in (20, 40):
        for sigma_h in (0.0, 0.3, 0.5):
            rows.append(integrate_stage("A", sigma_h, n_q))
            rows.append(integrate_stage("B", sigma_h, n_q))

    by_key = {(r["Sigma_H"], r["N_q"], r["stage"]): r for r in rows}
    comparisons = []
    for n_q in (20, 40):
        for sigma_h in (0.0, 0.3, 0.5):
            a = by_key[(sigma_h, n_q, "A")]
            b = by_key[(sigma_h, n_q, "B")]
            comparisons.append(
                {
                    "Sigma_H": sigma_h,
                    "N_q": n_q,
                    "deltaF0_L2_A": a["deltaF0_L2"],
                    "deltaF0_L2_B": b["deltaF0_L2"],
                    "F2_L2_A": a["F2_L2"],
                    "F2_L2_B": b["F2_L2"],
                    "F4_L2_B": b.get("F4_L2", 0.0),
                    "A_minus_B_deltaF0": a["deltaF0_L2"] - b["deltaF0_L2"],
                    "A_minus_B_F2": a["F2_L2"] - b["F2_L2"],
                }
            )

    payload = {"rows": rows, "comparisons": comparisons}
    (OUTDIR / "stageAB_transport_probe_pr1_2026-04-09.json").write_text(
        json.dumps(payload, indent=2)
    )

    import csv
    with (OUTDIR / "stageAB_transport_probe_pr1_2026-04-09.csv").open("w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
