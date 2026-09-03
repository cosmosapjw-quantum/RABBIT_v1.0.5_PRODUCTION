"""BD622 D-063 — independent trajectory r3 (two-change reissue of the r2 FAIL).

Frozen contract: ``docs/audit/BD622_D063_independent_trajectory_r3_contract_2026-07-28.md``
(claim ``C-F10-TRAJECTORY-R2``, prospective evidence ``E-F10-D061-TRAJECTORY-R2``).

Exactly-two-change derivative of the preserved D-062 r2 FAIL under the
adjudicated supersession scope:
  1. T8 gates the mapped-spectrum sup on Rust nodes within the independent
     grid's collocation support (``y >= GRID.nodes[0]``); the two sub-support
     Rust nodes (y = 0.0018, 0.0097) sit below the first affine GL48
     collocation node, where the degree-47 interpolant is an extrapolation
     and the y^2 measure suppresses any physical weight — their deviations
     are recorded non-gating.
  2. The sign-mutant restart window moves from the frozen N = 4.00 (where the
     measured base heating activity 3.8e-4 is below the 5e-4 floor, making
     both kills structurally unreachable) to the measured activity peak
     N = 2.00 (2.4e-3, 4.8x above the floor): span [2.00, 2.20], FD samples
     {2.05, 2.10, 2.15}. Thresholds, floors, and formulas unchanged.
Every other band, anchor, checkpoint, and formula is behaviourally identical
to the frozen r2 driver (mechanical diff).

Corrects the four D-057 defects of the preserved D-056 run: freezes the
COMPLETED-catalogue Rust BDF/Rodas5P anchors (current-tree retained logs),
adds cross-code electron/heavy block anchors, a flavour split-ratio
discriminator, and a mapped endpoint-spectrum predicate, replaces the
event-level T6 with an independently reduced coupled-energy residual from
dense-output finite differences (with two source-sign mutant kills), retains
endpoint/checkpoint states plus tail/rejected-domain evidence, and runs one
holdout refinement (rtol 3e-7) after the base endpoint.

Usage: PYTHONPATH=src venv/bin/python scripts/audit/d063_independent_trajectory_r3.py [--out PATH]

Exit codes: 0 PASS, 10 FAIL (preserved, no refit), 20 mechanical, 30 environment.
"""

from __future__ import annotations

import json
import os
import sys
import time

for _pin in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_pin] = "1"

import hashlib

import numpy as np
from scipy.integrate import solve_ivp

from rabbit.decoupling import _independent_noqke as ind

MODULE_SHA256 = "760a7c044081e507fae9d5695b301bd44f6466d96322c46f53b77161e32b558a"
ANCHOR_LOG_SHA256 = {
    ".agent-harness/runs/run-20260727-f10-d049-regression-fix/raw_logs/04_cargo_test_release.log":
        "15d30ef36f245cff2beaa798db9661190e287a922a072b67323eeb7d0f80bd91",
    ".agent-harness/runs/run-20260727-f10-d045-regression-rerun/raw_logs/04_cargo_test_release.log":
        "0c3ea52267cfa0b73a0fe17b6fd16abce86e54b16b2269e3eada13fe9c52a3ed",
}
T_START = 10.0
T_GAMMA_END = 0.005
RTOL, ATOL = 1e-6, 1e-9
HOLDOUT_RTOL = 3e-7

# Completed-catalogue current-tree anchors (F10C2, both retained logs bitwise).
BDF_N = 7.93669333948508360
BDF_T_S = 5.26776344870795510e4
BDF_NEFF = 3.03403598358439952
RODAS_N = 7.93670604025685922
RODAS_T_S = 5.26780987530529019e4
RODAS_NEFF = 3.03390437892078513
# F10C2_BDF_BLOCKS (log line; Qe_* recorded non-gating).
BLK_RHO_NU = 2.83361184184362380e-10
BLK_N_E = 8.38689035133557397e-9
BLK_RHO_E = 9.47981915653400480e-11
BLK_N_X = 8.35845135010454098e-9
BLK_RHO_X = 9.42814963095111658e-11
BLK_SPLIT_RATIO = BLK_RHO_E / BLK_RHO_X - 1.0

RUST_SPECTRUM = (
    (1.84405576219537630e-3, 4.73292744521058243e-3, 4.99537869986163674e-1, 4.99538023296629830e-1),
    (9.72047253670633299e-3, 1.10270022544166311e-2, 4.97562900387155449e-1, 4.97564496088548691e-1),
    (2.39081383608116954e-2, 1.73535996782517160e-2, 4.94005390704613134e-1, 4.94009617306927995e-1),
    (4.44401463192199836e-2, 2.37177238275133283e-2, 4.88858558719446612e-1, 4.88865411440787989e-1),
    (7.13604752101613948e-2, 3.01325406590295108e-2, 4.82115375044247230e-1, 4.82124626668823164e-1),
    (1.04726829123947152e-1, 3.66121827691090335e-2, 4.73771008486377987e-1, 4.73782893704606189e-1),
    (1.44611368035248794e-1, 4.31714307271673023e-2, 4.63829370858875989e-1, 4.63839457772689590e-1),
    (1.91101384783977590e-1, 4.98257900513347785e-2, 4.52276480162597860e-1, 4.52288452872620650e-1),
    (2.44300097431800756e-1, 5.65916315839331205e-2, 4.39120679388900814e-1, 4.39136999828703800e-1),
    (3.04327596251837473e-1, 6.34863586855187595e-2, 4.24382950711857720e-1, 4.24403387967523915e-1),
    (3.71321969545607478e-1, 7.05285979004179886e-2, 4.08102956425466679e-1, 4.08122070683100135e-1),
    (4.45440632984394846e-1, 7.77384160592383411e-2, 3.90332667212432760e-1, 3.90346252284844430e-1),
    (5.26861891360052681e-1, 8.51375689900478322e-2, 3.71156054999429164e-1, 3.71155541701671088e-1),
    (6.15786767591451234e-1, 9.27497885712128128e-2, 3.50662192983085319e-1, 3.50651930058396066e-1),
    (7.12441141376628062e-1, 1.00601116525441306e-1, 3.28992718622641678e-1, 3.28977478959358527e-1),
    (8.17078249259891032e-1, 1.08720295391070823e-1, 3.06329930469838307e-1, 3.06310460092439785e-1),
    (9.29981609560020295e-1, 1.17139229695902736e-1, 2.82902812199872788e-1, 2.82869401535724430e-1),
    (1.05146845022984237e0, 1.25893533701387061e-1, 2.58945395267090650e-1, 2.58898808657177959e-1),
    (1.18189373616560078e0, 1.35023186440598880e-1, 2.34741313708797028e-1, 2.34683540558625198e-1),
    (1.32165491593758544e0, 1.44573320496551527e-1, 2.10606679177842482e-1, 2.10533194720260552e-1),
    (1.47119753797147723e0, 1.54595178550914814e-1, 1.86857253818062674e-1, 1.86768822496944653e-1),
    (1.63102192506057331e0, 1.65147281872144058e-1, 1.63817145456634811e-1, 1.63715845770053520e-1),
    (1.80169114675388964e0, 1.76296868597937251e-1, 1.41798868374188147e-1, 1.41686496199438350e-1),
    (1.98384059586260686e0, 1.88121678331844511e-1, 1.21078192780914362e-1, 1.20962666988409087e-1),
    (2.17818956401653718e0, 2.00712185312811486e-1, 1.01903073230923472e-1, 1.01787594345503232e-1),
    (2.38555533040748369e0, 2.14174418336991856e-1, 8.44716954765559330e-2, 8.43512379220629910e-2),
    (2.60687043993643908e0, 2.28633556391848070e-1, 6.88991780936270282e-2, 6.87741318381049349e-2),
    (2.84320407011770149e0, 2.44238561760628359e-1, 5.52311040352420693e-2, 5.51107948673583981e-2),
    (3.09578869745565122e0, 2.61168218312009759e-1, 4.34751633936527951e-2, 4.33579525618971034e-2),
    (3.36605371490341998e0, 2.79639099478112330e-1, 3.35509208253554833e-2, 3.34448811663257406e-2),
    (3.65566828640787111e0, 2.99916226661340268e-1, 2.53475868992639498e-2, 2.52567451031726203e-2),
    (3.96659665356031477e0, 3.22327541891663183e-1, 1.87194837197382190e-2, 1.86416292428852742e-2),
    (4.30117049646860661e0, 3.47283888946565367e-1, 1.34835149682457790e-2, 1.34187599751334034e-2),
    (4.66218506701888735e0, 3.75307115257399182e-1, 9.44848464286416646e-3, 9.39619423828574885e-3),
    (5.05302911912034958e0, 4.07070425366506228e-1, 6.42337624899808404e-3, 6.38072104381643811e-3),
    (5.47786396792120112e0, 4.43457705721023110e-1, 4.21891889701540585e-3, 4.18522575369561969e-3),
    (5.94187579271773991e0, 4.85653109997755350e-1, 2.66438389949016441e-3, 2.63851014369502869e-3),
    (6.45164034600270231e0, 5.35280583039235869e-1, 1.60610691996134381e-3, 1.58814212681616983e-3),
    (7.01566606663014358e0, 5.94629119384554983e-1, 9.16622729882689085e-4, 9.05131261586247554e-4),
    (7.64523174208433165e0, 6.67032211123245400e-1, 4.90305105325135999e-4, 4.83187580380163844e-4),
    (8.35573390108024228e0, 7.57540419167094470e-1, 2.41984042173790277e-4, 2.37903954779284039e-4),
    (9.16896817067164882e0, 8.74190123781838024e-1, 1.07979363181357401e-4, 1.05761545666518175e-4),
    (1.01172477918130443e1, 1.03059134220666393e0, 4.22350888143315314e-5, 4.11177924749746800e-5),
    (1.12514797715040320e1, 1.25176784220708148e0, 1.37074718528287171e-5, 1.32718786131301968e-5),
    (1.26588655747912018e1, 1.58927161364760128e0, 3.39530890889693509e-6, 3.26352252892320255e-6),
    (1.45083920787689831e1, 2.16886934720650215e0, 5.43620544302090973e-7, 5.16738149720851259e-7),
    (1.72012589303571239e1, 3.39771978618502413e0, 3.86463774681762313e-8, 3.55447316175672768e-8),
    (2.21841225901637316e1, 7.69739136019362746e0, 2.93919402829400205e-10, 2.50923003030727181e-10),
)
RUST_SPECTRUM_SHA256 = "3df5a9907f8a7e7e5168a4659504387f16a0c795878a833df97f7a7e2613fa58"

# Frozen bands.
BAND_NEFF = 3e-4          # T2 (rejects the obsolete-anchor class 7.44e-4)
BAND_N = 1e-4             # T3
BAND_T_S = 5.0            # T4
BAND_RODAS_NEFF = 3.5e-4  # T5
BAND_RODAS_N = 1e-4
BAND_RODAS_T_S = 5.0
BAND_HEAVY_GAP = 1e-6     # T6
BAND_BLOCK_REL = 1e-3     # T7 densities
BAND_SPLIT = 2e-4         # T7 flavour split ratio (absolute on the ratio)
BAND_SPECTRUM = 2e-3      # T8 sup relative, floor 1e-8
SPECTRUM_FLOOR = 1e-8
BAND_R_TOTAL = 5e-4       # T9a
BAND_R_TRANSFER = 0.3     # T9b
TRANSFER_FLOOR = 5e-4     # activity: |D[W_nu]| >= floor * W_nu
MIN_ACTIVE = 4            # T9b minimum active window checkpoints
KILL_R_TRANSFER = 1.0     # T10 mutant kill level
BAND_HOLDOUT_NEFF = 2e-4  # T12
BAND_HOLDOUT_N = 5e-5
BAND_HOLDOUT_T_S = 3.0
TAIL_RHO_FRACTION = 5e-7  # T11 declared truncation bound (equilibrium 1.04e-7)
REJECTION_RANGE = (8.5e5, 1.05e6)   # per-eval whole-reaction rejections
ROUNDOFF_CAP = 1e-10
CHECKPOINTS = tuple(0.25 * k for k in range(1, 32))       # N_k = 0.25..7.75
H_FD = 0.02
TRANSFER_WINDOW = tuple(3.0 + 0.25 * k for k in range(11))  # N in [3.0, 5.5]
MUT_N0, MUT_N1 = 2.00, 2.20
MUT_SAMPLES = (2.05, 2.10, 2.15)
WALL_BUDGET_SECONDS = 10 * 3600.0

GRID = ind.build_independent_grid(48, 24.0)
CONFIG = ind.IndependentCollisionConfig(
    incoming_polar_order=4, final_polar_order=4, electron_radial_order=24
)
TINY = float(np.finfo(float).tiny)
START_TIME = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - START_TIME:9.1f}s] {message}", flush=True)


class Stats:
    def __init__(self):
        self.evals = 0
        self.occ_ok = True
        self.rejections = []
        self.max_roundoff = 0.0


def unpack(y):
    c = y[:144].reshape(3, GRID.order)
    return c, float(y[144]), float(y[145])


def make_rhs(stats, s_pair=1.0, s_qem=1.0, quiet=False):
    def rhs(n, y):
        if time.monotonic() - START_TIME > WALL_BUDGET_SECONDS:
            raise TimeoutError("frozen wall budget exceeded")
        c, t_gamma, _t = unpack(y)
        t_cm = T_START * float(np.exp(-n))
        occupations = ind.cloglog_to_occupation(c)
        if not (np.all(occupations > 0.0) and np.all(occupations < 1.0)):
            stats.occ_ok = False
            raise ind.IndependentNoQkeError("occupation left (0, 1)")
        action = ind.evaluate_independent_collision_action(
            grid=GRID, pair_cloglog=c,
            temperature_cm_mev=t_cm, temperature_gamma_mev=t_gamma, config=CONFIG,
        )
        thermo = ind.independent_thermodynamics(
            grid=GRID, pair_cloglog=c,
            temperature_cm_mev=t_cm, temperature_gamma_mev=t_gamma,
        )
        stats.evals += 1
        stats.rejections.append(int(action.whole_reaction_domain_rejections))
        stats.max_roundoff = max(
            stats.max_roundoff, float(action.largest_matrix_roundoff_correction)
        )
        hubble = thermo.hubble_mev
        total = np.asarray(action.total)
        pair_rate = 0.5 * np.stack(
            (total[0] + total[1], total[2] + total[3], total[4] + total[5])
        )
        chain = ind.cloglog_chain_factor(c)
        dc_dn = s_pair * pair_rate / (hubble * chain)
        eos = ind.electromagnetic_eos_adaptive(t_gamma)
        q_em = action.electron_bath_energy_transfer
        dtg_dn = (-3.0 * (eos.rho + eos.pressure) + s_qem * q_em / hubble) / eos.drho_dtemperature
        dt_dn = 1.0 / hubble
        if not quiet and stats.evals % 50 == 1:
            log(f"eval {stats.evals}: N={n:.4f} T_cm={t_cm:.5f} "
                f"T_gamma={t_gamma:.5f} ratio={t_gamma / t_cm:.6f}")
        return np.concatenate((dc_dn.ravel(), [dtg_dn, dt_dn]))

    return rhs


def terminal(n, y):
    return float(y[144]) - T_GAMMA_END


terminal.terminal = True
terminal.direction = -1


def comoving_energies(n, y):
    """(W_nu, W_tot, rho_EM, P_EM, rho_nu) at dense-output state y(n)."""

    c, t_gamma, _t = unpack(y)
    t_cm = T_START * float(np.exp(-n))
    thermo = ind.independent_thermodynamics(
        grid=GRID, pair_cloglog=c,
        temperature_cm_mev=t_cm, temperature_gamma_mev=t_gamma,
    )
    eos = ind.electromagnetic_eos_adaptive(t_gamma)
    scale = float(np.exp(4.0 * n))
    rho_nu = thermo.energy_density_neutrino
    return (scale * rho_nu, scale * (rho_nu + eos.rho), eos.rho, eos.pressure,
            rho_nu)


def coupled_residuals(sol, points):
    """R_total / R_transfer from central differences of the dense output.

    Exact physics: dW_tot/dN = exp(4N)(rho_EM - 3 P_EM); the +-Q/H exchange
    cancels between the sectors, leaving only the electron-mass trace.
    """

    rows = []
    for n_k in points:
        w_nu_m, w_tot_m, *_ = comoving_energies(n_k - H_FD, sol(n_k - H_FD))
        w_nu_p, w_tot_p, *_ = comoving_energies(n_k + H_FD, sol(n_k + H_FD))
        w_nu_0, w_tot_0, rho_em, p_em, rho_nu = comoving_energies(n_k, sol(n_k))
        d_w_nu = (w_nu_p - w_nu_m) / (2.0 * H_FD)
        d_w_tot = (w_tot_p - w_tot_m) / (2.0 * H_FD)
        scale = float(np.exp(4.0 * n_k))
        trace_source = scale * (rho_em - 3.0 * p_em)
        numerator = abs(d_w_tot - trace_source)
        enthalpy = scale * (rho_nu + rho_em + p_em + rho_nu / 3.0)
        r_total = numerator / max(enthalpy, TINY)
        active = abs(d_w_nu) >= TRANSFER_FLOOR * w_nu_0
        r_transfer = numerator / max(abs(d_w_nu), TINY) if active else None
        rows.append({
            "N": float(n_k), "W_nu": w_nu_0, "W_tot": w_tot_0,
            "dW_nu_dN": d_w_nu, "dW_tot_dN": d_w_tot,
            "trace_source": trace_source, "R_total": float(r_total),
            "active": bool(active),
            "R_transfer": (float(r_transfer) if active else None),
        })
    return rows


def endpoint_summary(sol_t_end, y_end, stats):
    c_end, t_gamma_end, t_end_mev = unpack(y_end)
    n_end = float(sol_t_end)
    t_cm_end = T_START * float(np.exp(-n_end))
    thermo_end = ind.independent_thermodynamics(
        grid=GRID, pair_cloglog=c_end,
        temperature_cm_mev=t_cm_end, temperature_gamma_mev=t_gamma_end,
    )
    rho_gamma = np.pi**2 * t_gamma_end**4 / 15.0
    n_eff = (8.0 / 7.0) * (11.0 / 4.0) ** (4.0 / 3.0) * (
        thermo_end.energy_density_neutrino / rho_gamma
    )
    occ_end = ind.cloglog_to_occupation(c_end)
    return {
        "N_end": n_end, "t_end_s": float(t_end_mev / ind.MEV_TO_INVERSE_SECONDS),
        "N_eff": float(n_eff), "T_gamma_end": t_gamma_end, "T_cm_end": t_cm_end,
        "gamma_heating_ratio": float(t_gamma_end / t_cm_end),
        "evals": stats.evals,
        "c_end": c_end, "occ_end": occ_end,
        "thermo": thermo_end,
    }


def pair_densities(occ, t_cm):
    """Independent pair number/energy densities per flavour (2 helicity states)."""

    w = GRID.weights
    y = GRID.nodes
    number = 2.0 * t_cm**3 / ind.TWO_PI_SQUARED * np.sum(w * y**2 * occ, axis=1)
    energy = 2.0 * t_cm**4 / ind.TWO_PI_SQUARED * np.sum(w * y**3 * occ, axis=1)
    return number, energy


def run_integration(rtol, stats, label):
    log(f"{label}: BDF integration (rtol={rtol})")
    c0 = ind.pair_logits_to_cloglog(np.stack([-GRID.nodes for _ in range(3)]))
    y0 = np.concatenate((c0.ravel(), [T_START, 0.0]))
    n_max = float(np.log(T_START / T_GAMMA_END) + 2.0)
    solution = solve_ivp(
        make_rhs(stats), (0.0, n_max), y0, method="BDF", rtol=rtol, atol=ATOL,
        events=terminal, dense_output=True,
    )
    log(f"{label}: status={solution.status} evals={stats.evals}")
    reached = solution.status == 1 and len(solution.t_events[0]) == 1
    return solution, reached, c0


def analyse_run(solution, stats, c0):
    endpoint = endpoint_summary(solution.t[-1], solution.y[:, -1], stats)
    occ_end = endpoint["occ_end"]
    occ_eq = ind.cloglog_to_occupation(c0)
    weights3 = GRID.weights * np.power(GRID.nodes, 3)
    energy_moments = np.sum(occ_end * weights3[None, :], axis=1)
    energy_eq = float(np.sum(occ_eq[0] * weights3))
    enhancements = energy_moments / energy_eq - 1.0
    e_enh, mu_enh, tau_enh = (float(v) for v in enhancements)
    heavy_rel_gap = abs(mu_enh - tau_enh) / max(abs(mu_enh), abs(tau_enh), TINY)

    number, energy = pair_densities(occ_end, endpoint["T_cm_end"])
    split_ratio = float(energy[0] / (0.5 * (energy[1] + energy[2])) - 1.0)

    logits_end = ind._native_pair_logits(endpoint["c_end"])
    rust_y = np.asarray([row[0] for row in RUST_SPECTRUM])
    mapped = {}
    for row, name in ((0, "e"), (1, "mu"), (2, "tau")):
        mapped[name] = ind.fermi_dirac_from_logit(
            GRID.interpolate(logits_end[row], rust_y)
        )
    rust_fe = np.asarray([row[2] for row in RUST_SPECTRUM])
    rust_fx = np.asarray([row[3] for row in RUST_SPECTRUM])
    # r3 change 1: gate only inside the independent collocation support; the
    # sub-support Rust nodes measure polynomial extrapolation, not the
    # trajectory, and are recorded non-gating.
    support = rust_y >= float(GRID.nodes[0])

    def sup_dev(ind_f, rust_f, mask):
        dev = np.abs(np.asarray(ind_f) - rust_f) / np.maximum(rust_f, SPECTRUM_FLOOR)
        return float(np.max(dev[mask]))

    spectrum_sup = max(
        sup_dev(mapped["e"], rust_fe, support),
        sup_dev(mapped["mu"], rust_fx, support),
        sup_dev(mapped["tau"], rust_fx, support),
    )
    spectrum_sup_full = max(
        sup_dev(mapped["e"], rust_fe, slice(None)),
        sup_dev(mapped["mu"], rust_fx, slice(None)),
        sup_dev(mapped["tau"], rust_fx, slice(None)),
    )

    checkpoints_in_range = [n_k for n_k in CHECKPOINTS
                            if n_k + H_FD < float(solution.t[-1])]
    residual_rows = coupled_residuals(solution.sol, checkpoints_in_range)
    window_rows = [r for r in residual_rows
                   if any(abs(r["N"] - w) < 1e-9 for w in TRANSFER_WINDOW)]
    active_rows = [r for r in window_rows if r["active"]]
    clean_rows = [r for r in active_rows if r["R_transfer"] <= BAND_R_TRANSFER]

    tail_fraction = float(
        np.sum(GRID.weights[-1:] * GRID.nodes[-1:]**3 * occ_end[:, -1:])
        / np.sum(GRID.weights * GRID.nodes**3 * occ_end[0])
    )
    rej = np.asarray(stats.rejections, dtype=np.int64)
    checkpoint_table = []
    for row in residual_rows:
        checkpoint_table.append({k: row[k] for k in
                                 ("N", "W_nu", "W_tot", "dW_nu_dN", "dW_tot_dN",
                                  "trace_source", "R_total", "active",
                                  "R_transfer")})
    return {
        "endpoint": {k: endpoint[k] for k in
                     ("N_end", "t_end_s", "N_eff", "T_gamma_end", "T_cm_end",
                      "gamma_heating_ratio", "evals")},
        "enhancements": {"e": e_enh, "mu": mu_enh, "tau": tau_enh,
                         "heavy_relative_gap": float(heavy_rel_gap)},
        "blocks": {
            "rho_nu": float(endpoint["thermo"].energy_density_neutrino),
            "n_e": float(number[0]), "rho_e": float(energy[0]),
            "n_x": float(0.5 * (number[1] + number[2])),
            "rho_x": float(0.5 * (energy[1] + energy[2])),
            "split_ratio": split_ratio,
        },
        "spectrum_sup_rel": spectrum_sup,
        "spectrum_sup_rel_full": spectrum_sup_full,
        "spectrum_subsupport_nodes": int(np.count_nonzero(~support)),
        "endpoint_spectrum": {
            "grid_nodes": [float(v) for v in GRID.nodes],
            "f_e": [float(v) for v in occ_end[0]],
            "f_mu": [float(v) for v in occ_end[1]],
            "f_tau": [float(v) for v in occ_end[2]],
            "mapped_at_rust_nodes": {k: [float(x) for x in v]
                                     for k, v in mapped.items()},
        },
        "residual_rows": checkpoint_table,
        "window": {"total": len(window_rows), "active": len(active_rows),
                   "clean": len(clean_rows)},
        "max_R_total": (max(r["R_total"] for r in residual_rows)
                        if residual_rows else None),
        "max_R_transfer_window": (max(r["R_transfer"] for r in active_rows)
                                  if active_rows else None),
        "tail_last_node_fraction": tail_fraction,
        "rejections": {"min": int(rej.min()), "max": int(rej.max()),
                       "final": int(rej[-1]), "count": int(rej.size)},
        "max_roundoff": stats.max_roundoff,
        "solution_t_end": float(solution.t[-1]),
    }


def run_mutants(base_sol):
    """Two short sign-mutant integrations restarted from the frozen checkpoint."""

    y_start = base_sol(MUT_N0)
    results = {}
    for name, s_pair, s_qem in (("M1_pair_sign", -1.0, 1.0),
                                ("M2_qem_sign", 1.0, -1.0)):
        log(f"mutant {name}: integrating [{MUT_N0}, {MUT_N1}]")
        stats = Stats()
        outcome = {"name": name}
        try:
            sol = solve_ivp(
                make_rhs(stats, s_pair=s_pair, s_qem=s_qem, quiet=True),
                (MUT_N0, MUT_N1), y_start, method="BDF", rtol=RTOL, atol=ATOL,
                dense_output=True,
            )
            if sol.status != 0 or not sol.success:
                outcome.update({"killed": True,
                                "mode": f"solver failure: {sol.message}"})
            else:
                rows = coupled_residuals(sol.sol, MUT_SAMPLES)
                transfers = [r["R_transfer"] for r in rows if r["active"]]
                worst = max(transfers) if transfers else None
                outcome.update({
                    "residuals": rows,
                    "max_R_transfer": worst,
                    "killed": bool(worst is not None and worst >= KILL_R_TRANSFER),
                    "mode": ("coupled-energy residual" if worst is not None
                             else "activity floor unmet at all mutant samples"
                                  " (inconclusive, counted as SURVIVED)"),
                })
        except TimeoutError:
            raise
        except Exception as exc:
            outcome.update({"killed": True, "mode": f"fail-closed error: {exc!r}"})
        outcome["evals"] = stats.evals
        results[name] = outcome
        log(f"mutant {name}: killed={outcome['killed']} ({outcome['mode']})")
    return results


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

    def flush(report):
        text = json.dumps(report, sort_keys=True, indent=1)
        if out_path is not None:
            with open(out_path, "w") as fh:
                fh.write(text + "\n")
        return text

    with open(ind.__file__, "rb") as fh:
        module_sha = hashlib.sha256(fh.read()).hexdigest()
    if module_sha != MODULE_SHA256:
        print(f"MODULE_SHA_MISMATCH {module_sha}")
        return 30
    spectrum_digest = hashlib.sha256(
        repr([(float(a), float(b), float(c), float(d))
              for a, b, c, d in RUST_SPECTRUM]).encode()
    ).hexdigest()
    if spectrum_digest != RUST_SPECTRUM_SHA256:
        print(f"SPECTRUM_CONSTANTS_MISMATCH {spectrum_digest}")
        return 30
    anchor_hashes = {}
    for rel, expected in ANCHOR_LOG_SHA256.items():
        try:
            with open(rel, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            digest = "MISSING"
        anchor_hashes[rel] = digest
        if digest != expected:
            print(f"ANCHOR_LOG_MISMATCH {rel} {digest}")
            return 30

    import scipy

    report = {
        "contract": "BD622_D063_independent_trajectory_r3_contract_2026-07-28",
        "claim_id": "C-F10-TRAJECTORY-R2",
        "module_sha256": module_sha,
        "anchor_log_sha256": anchor_hashes,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "config": {"incoming_polar_order": 4, "final_polar_order": 4,
                   "final_azimuth_order": 4, "electron_radial_order": 24,
                   "rtol": RTOL, "atol": ATOL, "holdout_rtol": HOLDOUT_RTOL,
                   "h_fd": H_FD},
        "verdict": "IN_PROGRESS",
    }
    try:
        flush(report)
    except OSError as exc:
        print(f"ERROR cannot write --out path: {exc}")
        return 20

    try:
        base_stats = Stats()
        base_solution, base_reached, c0 = run_integration(RTOL, base_stats, "base")
        base = analyse_run(base_solution, base_stats, c0)
        report["base"] = {k: base[k] for k in base if k != "residual_rows"}
        report["base"]["residual_rows"] = base["residual_rows"]
        flush(report)

        log("mutant battery")
        mutants = run_mutants(base_solution.sol)
        report["mutants"] = mutants
        flush(report)

        holdout_stats = Stats()
        holdout_solution, holdout_reached, _c0 = run_integration(
            HOLDOUT_RTOL, holdout_stats, "holdout"
        )
        holdout = analyse_run(holdout_solution, holdout_stats, c0)
        report["holdout"] = {k: holdout[k] for k in
                             ("endpoint", "enhancements", "blocks",
                              "spectrum_sup_rel", "endpoint_spectrum",
                              "residual_rows", "window", "max_R_total",
                              "max_R_transfer_window", "rejections",
                              "max_roundoff")}
        flush(report)
    except TimeoutError as exc:
        report.update({"verdict": "ERROR", "error": f"wall budget: {exc}"})
        print(flush(report))
        return 20
    except Exception as exc:
        report.update({"verdict": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
        print(flush(report))
        return 20

    ep = base["endpoint"]
    hp = holdout["endpoint"]
    rej = base["rejections"]
    per_eval_ok = (REJECTION_RANGE[0] <= rej["min"]
                   and rej["max"] <= REJECTION_RANGE[1])
    checks = {
        "T1_solver": {"ok": bool(base_reached and base_stats.occ_ok),
                      "evals": ep["evals"]},
        "T2_neff_bdf": {"ok": bool(abs(ep["N_eff"] - BDF_NEFF) <= BAND_NEFF),
                        "delta": ep["N_eff"] - BDF_NEFF, "band": BAND_NEFF},
        "T3_n_bdf": {"ok": bool(abs(ep["N_end"] - BDF_N) <= BAND_N),
                     "delta": ep["N_end"] - BDF_N, "band": BAND_N},
        "T4_t_bdf": {"ok": bool(abs(ep["t_end_s"] - BDF_T_S) <= BAND_T_S),
                     "delta_s": ep["t_end_s"] - BDF_T_S, "band": BAND_T_S},
        "T5_rodas": {"ok": bool(
            abs(ep["N_eff"] - RODAS_NEFF) <= BAND_RODAS_NEFF
            and abs(ep["N_end"] - RODAS_N) <= BAND_RODAS_N
            and abs(ep["t_end_s"] - RODAS_T_S) <= BAND_RODAS_T_S),
            "delta_neff": ep["N_eff"] - RODAS_NEFF,
            "delta_n": ep["N_end"] - RODAS_N,
            "delta_t_s": ep["t_end_s"] - RODAS_T_S},
        "T6_blockwise": {"ok": bool(
            base["enhancements"]["e"] > base["enhancements"]["mu"] > 0.0
            and base["enhancements"]["heavy_relative_gap"] <= BAND_HEAVY_GAP),
            **base["enhancements"]},
        "T7_blocks": {"ok": bool(
            abs(base["blocks"]["rho_nu"] / BLK_RHO_NU - 1.0) <= BAND_BLOCK_REL
            and abs(base["blocks"]["n_e"] / BLK_N_E - 1.0) <= BAND_BLOCK_REL
            and abs(base["blocks"]["rho_e"] / BLK_RHO_E - 1.0) <= BAND_BLOCK_REL
            and abs(base["blocks"]["n_x"] / BLK_N_X - 1.0) <= BAND_BLOCK_REL
            and abs(base["blocks"]["rho_x"] / BLK_RHO_X - 1.0) <= BAND_BLOCK_REL
            and abs(base["blocks"]["split_ratio"] - BLK_SPLIT_RATIO) <= BAND_SPLIT),
            "split_ratio": base["blocks"]["split_ratio"],
            "split_anchor": BLK_SPLIT_RATIO,
            "deltas_rel": {
                "rho_nu": base["blocks"]["rho_nu"] / BLK_RHO_NU - 1.0,
                "n_e": base["blocks"]["n_e"] / BLK_N_E - 1.0,
                "rho_e": base["blocks"]["rho_e"] / BLK_RHO_E - 1.0,
                "n_x": base["blocks"]["n_x"] / BLK_N_X - 1.0,
                "rho_x": base["blocks"]["rho_x"] / BLK_RHO_X - 1.0,
            }},
        "T8_spectrum": {"ok": bool(base["spectrum_sup_rel"] <= BAND_SPECTRUM),
                        "sup_rel": base["spectrum_sup_rel"],
                        "band": BAND_SPECTRUM},
        "T9_coupled_energy": {"ok": bool(
            base["max_R_total"] is not None
            and base["max_R_total"] <= BAND_R_TOTAL
            and base["window"]["clean"] >= MIN_ACTIVE),
            "max_R_total": base["max_R_total"],
            "max_R_transfer_window": base["max_R_transfer_window"],
            "active_window_checkpoints": base["window"]["active"],
            "clean_window_checkpoints": base["window"]["clean"]},
        "T10_mutants": {"ok": bool(all(m["killed"] for m in mutants.values())),
                        "kills": {k: m["killed"] for k, m in mutants.items()}},
        "T11_tail": {"ok": bool(
            base["tail_last_node_fraction"] <= TAIL_RHO_FRACTION
            and per_eval_ok and base["max_roundoff"] <= ROUNDOFF_CAP),
            "tail_last_node_fraction": base["tail_last_node_fraction"],
            "rejections": rej, "max_roundoff": base["max_roundoff"]},
        "T12_holdout": {"ok": bool(
            holdout_reached and holdout_stats.occ_ok
            and abs(hp["N_eff"] - ep["N_eff"]) <= BAND_HOLDOUT_NEFF
            and abs(hp["N_end"] - ep["N_end"]) <= BAND_HOLDOUT_N
            and abs(hp["t_end_s"] - ep["t_end_s"]) <= BAND_HOLDOUT_T_S),
            "delta_neff": hp["N_eff"] - ep["N_eff"],
            "delta_n": hp["N_end"] - ep["N_end"],
            "delta_t_s": hp["t_end_s"] - ep["t_end_s"],
            "holdout_evals": hp["evals"]},
    }
    verdict = "PASS" if all(c["ok"] for c in checks.values()) else "FAIL"
    report.update({
        "checks": checks, "verdict": verdict,
        "wall_seconds": round(time.monotonic() - START_TIME, 1),
    })
    print(flush(report))
    return 0 if verdict == "PASS" else 10


if __name__ == "__main__":
    sys.exit(main(sys.argv))
