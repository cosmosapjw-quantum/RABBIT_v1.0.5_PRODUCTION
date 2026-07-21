"""BD622 W5 — D-028 native mu-tau covariance discriminator (E4 localization + E5 dual-norm).

Owner-authorized static run. Imports the FROZEN comparator bytes unchanged (read-only) and
re-evaluates the recorded GL48-Y24 split-scale S1 state. Reports:
  - the native mu-tau relative-L-inf residual (the D-028 observable, ~4.67e-10 class);
  - E5 dual-norm: the SAME data measured on the well-conditioned weak content
      (modal coefficients, and measure-weighted m_j*A_j where the 1/y^2 cancels);
  - E4 localization: argmax node of |A_mu - A_tau| and the residual restricted to y >= y_cut.

If native ~1e-10 while modal/measure-weighted ~1e-13, and the argmax sits at the smallest node
with the residual collapsing as small nodes are excluded, the D-028 FAIL is a metrology/1/y^2
conditioning event (F-1/F-2), not a physics covariance defect. Does NOT modify any gate or the
frozen bytes.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from rabbit.decoupling import _independent_noqke as ind  # noqa: E402


def split_state(grid):
    scales = (1.01, 0.995, 0.995)
    return ind.pair_logits_to_cloglog(np.stack([-grid.nodes / s for s in scales]))


def rel_linf(a, b):
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), np.finfo(float).tiny)
    return float(np.max(np.abs(a - b)) / scale)


def main():
    grid = ind.build_independent_grid(48, 24.0)
    config = ind.IndependentCollisionConfig()  # default = full-degree GL48
    action = ind.evaluate_independent_collision_action(
        grid=grid, pair_cloglog=split_state(grid),
        temperature_cm_mev=10.0, temperature_gamma_mev=10.0, config=config,
    )
    y = grid.nodes
    native_resid = action.diagnostics["mu_tau_residual"]

    # native pair actions (with 1/y^2): pair index 1 = mu, 2 = tau
    A_mu = 0.5 * (action.total[2] + action.total[3])
    A_tau = 0.5 * (action.total[4] + action.total[5])
    diff = np.abs(A_mu - A_tau)

    # E5 dual-norm: modal coefficients (weak content, no inversion)
    M_mu = 0.5 * (action.modal_total[2] + action.modal_total[3])
    M_tau = 0.5 * (action.modal_total[4] + action.modal_total[5])
    modal_resid = rel_linf(M_mu, M_tau)
    # measure-weighted native (m_j = w_j y_j^2 -> y^2 cancels the inversion)
    m = grid.weights * np.square(grid.nodes)
    meas_resid = rel_linf(m * A_mu, m * A_tau)

    # E4 localization
    argmax = int(np.argmax(diff))
    subgrid = {}
    for cut in (0.0, float(y[1]), float(y[2]), 0.5, 1.0):
        mask = y >= cut
        sc = max(float(np.max(np.abs(A_mu[mask]))), float(np.max(np.abs(A_tau[mask]))), np.finfo(float).tiny)
        subgrid[f"y>={cut:.4g}"] = float(np.max(diff[mask]) / sc)

    print(f"grid GL48-Y24  y_min={y[0]:.6e}  1/y_min^2={1/y[0]**2:.3e}")
    print(f"[D-028 observable] native mu_tau relative-Linf = {native_resid:.6e}   (cap 1e-10)")
    print(f"[E5 dual-norm] modal-coeff residual          = {modal_resid:.6e}")
    print(f"[E5 dual-norm] measure-weighted (m*A) residual= {meas_resid:.6e}")
    print(f"[E4] argmax node j={argmax}  y[j]={y[argmax]:.6e}  (y_min index 0)")
    print("[E4] subgrid relative-Linf (residual vs lower cut):")
    for k, v in subgrid.items():
        print(f"      {k:<12} {v:.6e}")
    verdict = ("METROLOGY/1-over-y^2 CONDITIONING (F-1/F-2 confirmed)"
               if (modal_resid < 1e-12 and meas_resid < 1e-12 and argmax <= 1)
               else "INCONCLUSIVE / possible real covariance defect")
    print(f"\nVERDICT: {verdict}")

    artifact = {
        "test_id": "W5-D028-E4E5", "grid": "GL48-Y24", "state": "split(1.01,0.995,0.995)",
        "native_mu_tau_relative_linf": native_resid, "native_cap": 1e-10,
        "modal_coeff_residual": modal_resid, "measure_weighted_residual": meas_resid,
        "argmax_node": argmax, "argmax_y": float(y[argmax]), "y_min": float(y[0]),
        "one_over_ymin_sq": float(1 / y[0] ** 2), "subgrid_ladder": subgrid, "verdict": verdict,
    }
    print("\nARTIFACT " + json.dumps(artifact))


if __name__ == "__main__":
    main()
