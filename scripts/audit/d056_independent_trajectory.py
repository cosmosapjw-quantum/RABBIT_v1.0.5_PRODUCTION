"""BD622 D-056 — independent collision-coupled FLRW trajectory (T1..T6).

Formulation, bands, and anchors frozen in
``docs/audit/BD622_D056_independent_trajectory_contract_2026-07-28.md``.

Usage: PYTHONPATH=src python3 scripts/audit/d056_independent_trajectory.py [--out PATH]
"""

from __future__ import annotations

import json
import os
import sys

for _pin in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_pin] = "1"

import hashlib

import numpy as np
from scipy.integrate import solve_ivp

from rabbit.decoupling import _independent_noqke as ind

MODULE_SHA_PREFIX = "760a7c04"
T_START = 10.0
T_GAMMA_END = 0.005
ANCHOR_N = 7.9367214190
ANCHOR_T = 52680.2048
ANCHOR_NEFF = 3.0333103242
RTOL, ATOL = 1e-6, 1e-9

GRID = ind.build_independent_grid(48, 24.0)
CONFIG = ind.IndependentCollisionConfig(
    incoming_polar_order=4, final_polar_order=4, electron_radial_order=24
)
STATS = {"evals": 0, "max_first_law": 0.0, "occ_ok": True}


def unpack(y):
    c = y[:144].reshape(3, GRID.order)
    return c, float(y[144]), float(y[145])


def rhs(n, y):
    c, t_gamma, _t = unpack(y)
    t_cm = T_START * float(np.exp(-n))
    occupations = ind.cloglog_to_occupation(c)
    if not (np.all(occupations > 0.0) and np.all(occupations < 1.0)):
        STATS["occ_ok"] = False
        raise ind.IndependentNoQkeError("occupation left (0, 1)")
    action = ind.evaluate_independent_collision_action(
        grid=GRID, pair_cloglog=c,
        temperature_cm_mev=t_cm, temperature_gamma_mev=t_gamma, config=CONFIG,
    )
    thermo = ind.independent_thermodynamics(
        grid=GRID, pair_cloglog=c,
        temperature_cm_mev=t_cm, temperature_gamma_mev=t_gamma,
    )
    STATS["evals"] += 1
    STATS["max_first_law"] = max(
        STATS["max_first_law"], float(action.diagnostics["first_law_residual"])
    )
    hubble = thermo.hubble_mev
    total = np.asarray(action.total)
    pair_rate = 0.5 * np.stack(
        (total[0] + total[1], total[2] + total[3], total[4] + total[5])
    )
    chain = ind.cloglog_chain_factor(c)
    dc_dn = pair_rate / (hubble * chain)
    eos = ind.electromagnetic_eos_adaptive(t_gamma)
    q_em = action.electron_bath_energy_transfer
    dtg_dn = (-3.0 * (eos.rho + eos.pressure) + q_em / hubble) / eos.drho_dtemperature
    dt_dn = 1.0 / hubble
    if STATS["evals"] % 50 == 1:
        print(f"eval {STATS['evals']}: N={n:.4f} T_cm={t_cm:.5f} "
              f"T_gamma={t_gamma:.5f} ratio={t_gamma / t_cm:.6f}", flush=True)
    return np.concatenate((dc_dn.ravel(), [dtg_dn, dt_dn]))


def terminal(n, y):
    return float(y[144]) - T_GAMMA_END


terminal.terminal = True
terminal.direction = -1


def main(argv):
    out_path = None
    args = list(argv[1:])
    while args:
        arg = args.pop(0)
        if arg == "--out":
            if not args:
                print("ERROR --out requires a path")
                return 20
            out_path = args.pop(0)
        else:
            print(f"ERROR unknown argument {arg!r}")
            return 20

    with open(ind.__file__, "rb") as fh:
        module_sha = hashlib.sha256(fh.read()).hexdigest()
    if not module_sha.startswith(MODULE_SHA_PREFIX):
        print(f"MODULE_SHA_MISMATCH {module_sha}")
        return 30

    if out_path is not None:
        with open(out_path, "w") as fh:
            fh.write("{}\n")

    c0 = ind.pair_logits_to_cloglog(np.stack([-GRID.nodes for _ in range(3)]))
    y0 = np.concatenate((c0.ravel(), [T_START, 0.0]))
    n_max = float(np.log(T_START / T_GAMMA_END) + 2.0)
    try:
        solution = solve_ivp(
            rhs, (0.0, n_max), y0, method="BDF", rtol=RTOL, atol=ATOL,
            events=terminal, dense_output=False,
        )
    except Exception as exc:  # mechanical failure: preserve evidence, exit 20
        failure = {"contract": "BD622_D056_independent_trajectory_contract_2026-07-28",
                   "mechanical_failure": repr(exc), "evals": STATS["evals"],
                   "max_first_law_residual": STATS["max_first_law"],
                   "verdict": "ERROR"}
        text = json.dumps(failure, sort_keys=True, indent=1)
        print(text)
        if out_path is not None:
            with open(out_path, "w") as fh:
                fh.write(text + "\n")
        return 20
    print(f"solver status={solution.status} message={solution.message} "
          f"evals={STATS['evals']}", flush=True)

    reached = solution.status == 1 and len(solution.t_events[0]) == 1
    c_end, t_gamma_end, t_end_mev = unpack(solution.y[:, -1])
    n_end = float(solution.t[-1])
    t_cm_end = T_START * float(np.exp(-n_end))
    thermo_end = ind.independent_thermodynamics(
        grid=GRID, pair_cloglog=c_end,
        temperature_cm_mev=t_cm_end, temperature_gamma_mev=t_gamma_end,
    )
    rho_gamma = np.pi**2 * t_gamma_end**4 / 15.0
    n_eff = (8.0 / 7.0) * (11.0 / 4.0) ** (4.0 / 3.0) * (
        thermo_end.energy_density_neutrino / rho_gamma
    )
    t_end_seconds = t_end_mev / ind.MEV_TO_INVERSE_SECONDS

    occ_end = ind.cloglog_to_occupation(c_end)
    occ_eq = ind.cloglog_to_occupation(c0)
    weights = GRID.weights * np.power(GRID.nodes, 3)
    energy_moments = np.sum(occ_end * weights[None, :], axis=1)
    energy_eq = float(np.sum(occ_eq[0] * weights))
    enhancements = energy_moments / energy_eq - 1.0
    e_enh, mu_enh, tau_enh = (float(v) for v in enhancements)
    heavy_rel_gap = abs(mu_enh - tau_enh) / max(abs(mu_enh), abs(tau_enh), TINY := float(np.finfo(float).tiny))

    checks = {
        "T1_solver": {"ok": bool(reached and STATS["occ_ok"]),
                      "status": int(solution.status), "evals": STATS["evals"],
                      "occupations_in_bounds": STATS["occ_ok"]},
        "T2_neff": {"ok": bool(abs(n_eff - ANCHOR_NEFF) <= 3e-3),
                    "N_eff": float(n_eff), "delta": float(n_eff - ANCHOR_NEFF)},
        "T3_n_end": {"ok": bool(abs(n_end - ANCHOR_N) <= 3e-3),
                     "N_end": n_end, "delta": float(n_end - ANCHOR_N)},
        "T4_t_end": {"ok": bool(abs(t_end_seconds - ANCHOR_T) <= 15.0),
                     "t_end_s": float(t_end_seconds),
                     "delta_s": float(t_end_seconds - ANCHOR_T)},
        "T5_blockwise": {"ok": bool(e_enh > mu_enh > 0.0 and heavy_rel_gap <= 1e-6),
                         "e_enhancement": e_enh, "mu_enhancement": mu_enh,
                         "tau_enhancement": tau_enh,
                         "heavy_relative_gap": float(heavy_rel_gap)},
        "T6_first_law": {"ok": bool(STATS["max_first_law"] <= 1e-8),
                         "max_first_law_residual": STATS["max_first_law"]},
    }
    verdict = "PASS" if all(c["ok"] for c in checks.values()) else "FAIL"
    report = {"contract": "BD622_D056_independent_trajectory_contract_2026-07-28",
              "module_sha256": module_sha,
              "config": {"incoming_polar_order": 4, "final_polar_order": 4,
                         "final_azimuth_order": 4, "electron_radial_order": 24,
                         "rtol": RTOL, "atol": ATOL},
              "endpoint": {"N_end": n_end, "t_end_s": float(t_end_seconds),
                           "N_eff": float(n_eff), "T_gamma_end": t_gamma_end,
                           "T_cm_end": t_cm_end,
                           "gamma_heating_ratio": float(t_gamma_end / t_cm_end)},
              "checks": checks, "verdict": verdict}
    text = json.dumps(report, sort_keys=True, indent=1)
    print(text)
    if out_path is not None:
        with open(out_path, "w") as fh:
            fh.write(text + "\n")
    return 0 if verdict == "PASS" else 10


if __name__ == "__main__":
    sys.exit(main(sys.argv))
