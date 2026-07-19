from __future__ import annotations

import csv
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scipy.integrate import solve_ivp

from rabbit.config.grids import MomentumGrid
from rabbit.transport.stageAB_state import AxisymmetricHierarchyState
from rabbit.transport.typeI_stageA_hierarchy import STAGE_A_ELLS, compute_hierarchy_rhs_typeI_stageA
from rabbit.transport.typeI_stageB_hierarchy import STAGE_B_ELLS, compute_hierarchy_rhs_typeI_stageB
from rabbit.transport.typeI_stageAB_reconstruct import compute_reconstruction_diagnostics

OUTDIR = ROOT / "session7_outputs"
OUTDIR.mkdir(exist_ok=True)


def integrate_final_state(stage: str, sigma_h: float, n_q: int, n_end: float = 0.5) -> AxisymmetricHierarchyState:
    grid = MomentumGrid(N_q=n_q)
    if stage == "A":
        active_ells = STAGE_A_ELLS
        rhs_fn = compute_hierarchy_rhs_typeI_stageA
    elif stage == "B":
        active_ells = STAGE_B_ELLS
        rhs_fn = compute_hierarchy_rhs_typeI_stageB
    else:
        raise ValueError(stage)
    state0 = AxisymmetricHierarchyState.from_fd_equilibrium(grid, active_ells, n_species=1)

    def rhs(_n, y):
        st = AxisymmetricHierarchyState.from_flat(y, grid, active_ells, n_species=1)
        return rhs_fn(st, Sigma_H=sigma_h)

    sol = solve_ivp(rhs, (0.0, n_end), state0.to_flat(), method="RK45", rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for stage={stage}, sigma={sigma_h}, N_q={n_q}: {sol.message}")
    return AxisymmetricHierarchyState.from_flat(sol.y[:, -1], grid, active_ells, n_species=1)


def main() -> None:
    rows: list[dict] = []
    for stage in ("A", "B"):
        for n_q in (20, 40):
            for sigma_h in (0.0, 0.3, 0.5):
                state = integrate_final_state(stage=stage, sigma_h=sigma_h, n_q=n_q)
                for mode in ("R1", "R2"):
                    diag = compute_reconstruction_diagnostics(state, mode=mode, n_mu=17)
                    row = {
                        "stage": stage,
                        "mode": mode,
                        "Sigma_H": sigma_h,
                        "N_q": n_q,
                        "n_mu": diag.n_mu,
                        "f_min": diag.f_min,
                        "f_max": diag.f_max,
                        "violation_fraction": diag.violation_fraction,
                        "reprojection_l2_abs": diag.reprojection_l2_abs,
                        "reprojection_l2_rel": diag.reprojection_l2_rel,
                    }
                    for ell, val in diag.per_ell_l2_abs.items():
                        row[f"ell{ell}_l2_abs"] = val
                    for ell, val in diag.per_ell_l2_rel.items():
                        row[f"ell{ell}_l2_rel"] = val
                    rows.append(row)

    payload = {"rows": rows}
    (OUTDIR / "stageAB_reconstruction_probe_pr2_2026-04-09.json").write_text(json.dumps(payload, indent=2))
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with (OUTDIR / "stageAB_reconstruction_probe_pr2_2026-04-09.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
