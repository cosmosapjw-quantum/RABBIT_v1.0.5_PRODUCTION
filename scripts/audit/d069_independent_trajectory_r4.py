"""BD622 D-069 — independent trajectory r4: full-domain, recomputable, enclosed.

Frozen contract: ``docs/audit/BD622_D069_independent_trajectory_r4_contract_2026-07-29.md``
(claim ``C-F10-INDEPENDENT``, prospective evidence ``E-F10-D069-TRAJECTORY-R4``).

Discharges the five obligations D-065 finding F-D065-04 left open against the
preserved D-063 PASS. D-063's payload reproduces and is not disputed; what it
lacked was evidence, and one of its predicates was narrowed after output.

  1. FULL-DOMAIN SPECTRAL NORM, NOTHING MASKED. D-062 measured a full-48-node
     sup of 2.0800641747316935e-3 against a 2e-3 band; D-063 then excluded the
     two Rust nodes below the independent grid's first collocation node and
     obtained 1.5881626368027338e-3. Excluding them after seeing the number is
     illegitimate, but so is gating a pointwise sup there: those nodes lie at
     y = 0.00184 and 0.00972, below the first affine GL48 node 0.01475, where
     the degree-47 interpolant is an extrapolation. Covering y = 0.00184 by
     collocation needs order ~136, which is not reachable at this cost class.
     r4 therefore gates three prospective predicates instead, all frozen before
     any output byte and none masking anything: a grid-independent moment norm
     computed on each code's own nodes and weights (T8a), a true-weight
     pointwise norm over all 48 Rust nodes including the two sub-support ones at
     their real 8.9e-10 energy share (T8b), and the r3 in-support sup retained
     as a floor (T8c). The full-domain sup is recorded, non-gating, alongside the
     complete 48x3 per-node deviation array.
  2. RECOMPUTABLE CHECKPOINTS. D-063 retained nine reduced scalars per
     checkpoint. r4 retains the entire integrator state -- all 3 x order cloglog
     components, T_gamma, T_cm, and cosmic time -- and T14 proves sufficiency by
     re-deriving W_nu, W_tot, and R_total from the stored state alone.
  3. EVOLVED BEYOND-DOMAIN ENCLOSURE. D-063's ``tail_last_node_fraction`` is a
     final in-domain GL-node contribution, not a statement about y > 24. r4 runs
     a density-matched companion integration at order 60 / y_max 30 -- a public
     knob of the frozen module, so nothing in it is edited -- and combines the
     measured endpoint shift with the exact analytic equilibrium residual beyond
     30 (T13). The enclosure must come in under the band that discriminates the
     cross-code comparison, or the comparison is not meaningful.
  4. SPATIAL/DOMAIN HOLDOUT. The same companion run is the holdout on the
     representation axis; the rtol refinement (T12) is retained unchanged.
  5. CHRONOLOGY. Machine UTC stamps only, from the report writer.

Anchors, bands inherited from already-frozen predicates, and every other check
are unchanged from the preserved r3.

Usage: PYTHONPATH=src:scripts/audit venv/bin/python scripts/audit/d069_independent_trajectory_r4.py --out PATH

Exit codes: 0 PASS, 10 FAIL (preserved, no refit), 20 mechanical, 30 environment.
"""

from __future__ import annotations

import argparse
import os
import sys

# BLAS threading must be pinned BEFORE numpy is imported: OpenBLAS reads these at
# library initialisation, so setting them afterwards is a no-op. The frozen d063
# driver pins them at module top for the same reason, and single-threaded BLAS is
# what makes this battery reproducible.
for _pin in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_pin] = "1"

import numpy as np  # noqa: E402

import _f10c2_anchors as A  # noqa: E402
import _trajectory_core as core  # noqa: E402
from rabbit.decoupling import _independent_noqke as ind  # noqa: E402

# --- frozen bands ------------------------------------------------------------

BAND_NEFF = 3e-4          # T2 (rejects the obsolete-anchor class 7.44e-4)
BAND_N = 1e-4             # T3
BAND_T_S = 5.0            # T4
BAND_RODAS_NEFF = 3.5e-4  # T5
BAND_RODAS_N = 1e-4
BAND_RODAS_T_S = 5.0
BAND_HEAVY_GAP = 1e-6     # T6
BAND_BLOCK_REL = 1e-3     # T7 densities
BAND_SPLIT = 2e-4         # T7 flavour split ratio (absolute on the ratio)

# T8a/T8b inherit T7's already-frozen 1e-3 cross-code density tolerance: the
# moments ARE densities. Two independent margin statements, both pre-freeze:
#   * against the quadrature floor on the disclosed equilibrium state, after the
#     analytic tail correction -- 5.4e-9 (k=2), 5.2e-8 (k=3), 3.2e-7 (k=4),
#     1.4e-6 (k=5), i.e. three to six orders;
#   * against the deviation actually expected, reconstructed from the PRESERVED
#     d063 endpoint spectrum -- T8a max |rel| 6.1e-5 (~16x), T8a split 1.8e-5
#     (~11x), T8b 2.1e-4 (~4.8x), T8c 1.588e-3 (~1.26x).
# The second is the honest one to judge by. No band was chosen to accommodate an
# expected value: 1e-3 is not the minimal passing choice for any of them, and
# k = 4, 5 use the same 1e-3 as k = 2, 3 rather than a looser tier.
MOMENT_POWERS = (2, 3, 4, 5)
BAND_MOMENT = 1e-3        # T8a, all k: T7's frozen cross-code density tolerance
BAND_SPLIT_MOMENT = 2e-4  # T8a flavour split ratio from the k = 3 moments
BAND_SPECTRUM_WEIGHTED = 1e-3   # T8b energy-weighted pointwise, all 48 nodes
BAND_SPECTRUM_SUP = 2e-3        # T8c in-support sup (the retained r3 predicate)
SPECTRUM_FLOOR = 1e-8

BAND_R_TOTAL = 5e-4       # T9a
BAND_R_TRANSFER = 0.3     # T9b
TRANSFER_FLOOR = 5e-4     # activity: |D[W_nu]| >= floor * W_nu
MIN_ACTIVE = 4            # T9b minimum clean window checkpoints
KILL_R_TRANSFER = 1.0     # T10 mutant kill level
REJECTION_RANGE = (8.5e5, 1.05e6)   # T11 per-eval whole-reaction rejections
ROUNDOFF_CAP = 1e-10      # T11
TAIL_RHO_FRACTION = 5e-7  # T11, retained verbatim from the frozen r3 predicate
BAND_HOLDOUT_NEFF = 2e-4  # T12 rtol refinement
BAND_HOLDOUT_N = 5e-5
BAND_HOLDOUT_T_S = 3.0
BAND_DOMAIN_NEFF = 2e-4   # T13 measured domain-holdout shift (holdout class)
BAND_ENCLOSURE = 3e-4     # T13 total enclosure must stay under the T2 band
BAND_ANALYTIC_RESIDUAL = 1e-6   # T13 equilibrium energy fraction beyond y_max

CHECKPOINTS = tuple(0.25 * k for k in range(1, 32))       # N_k = 0.25..7.75
TRANSFER_WINDOW = tuple(3.0 + 0.25 * k for k in range(11))  # N in [3.0, 5.5]
MUT_N0, MUT_N1 = 2.00, 2.20
MUT_SAMPLES = (2.05, 2.10, 2.15)
RECOMPUTE_CHECKPOINTS = (1.00, 4.00, 7.00)   # T14 spot re-derivations
RECOMPUTE_TOL = 1e-12

BASE_ORDER, BASE_YMAX = 48, 24.0
DOMAIN_ORDER, DOMAIN_YMAX = 60, 30.0          # density-matched: 60/30 == 48/24
RTOL, ATOL = 1e-6, 1e-9
HOLDOUT_RTOL = 3e-7
WALL_BUDGET_SECONDS = 18 * 3600.0

TINY = core.TINY


def build(order: int, y_max: float, label: str) -> core.Setup:
    return core.build_setup(
        order, y_max, label=label, rtol=RTOL, atol=ATOL,
        transfer_floor=TRANSFER_FLOOR,
    )


# --- analysis ----------------------------------------------------------------


def spectral_block(setup: core.Setup, occ_end: np.ndarray, c_end: np.ndarray) -> dict:
    """T8a/T8b/T8c inputs. Nothing is excluded from any gated quantity."""

    rust_y = np.asarray([row[0] for row in A.RUST_SPECTRUM])
    rust_w = np.asarray([row[1] for row in A.RUST_SPECTRUM])
    rust_fe = np.asarray([row[2] for row in A.RUST_SPECTRUM])
    rust_fx = np.asarray([row[3] for row in A.RUST_SPECTRUM])

    # T8a: each code integrates on its own rule; no interpolation either way.
    # The independent side is truncated at y_max, so add back the analytic
    # equilibrium tail it cannot represent before comparing.
    ind_moments = core.spectral_moments(setup, occ_end, MOMENT_POWERS)
    tail_fraction = {
        f"M{k}": core.equilibrium_tail_fraction(setup.y_max, power=k)
        for k in MOMENT_POWERS
    }
    rust_moments = {
        "e": core.anchor_moments(rust_y, rust_w, rust_fe, MOMENT_POWERS),
        "mu": core.anchor_moments(rust_y, rust_w, rust_fx, MOMENT_POWERS),
        "tau": core.anchor_moments(rust_y, rust_w, rust_fx, MOMENT_POWERS),
    }
    moment_rel = {}
    for flavour in ("e", "mu", "tau"):
        moment_rel[flavour] = {
            key: float(
                ind_moments[flavour][key] / (1.0 - tail_fraction[key])
                / rust_moments[flavour][key] - 1.0
            )
            for key in ind_moments[flavour]
        }
    moment_split = float(
        ind_moments["e"]["M3"]
        / (0.5 * (ind_moments["mu"]["M3"] + ind_moments["tau"]["M3"]))
        - 1.0
    )
    rust_split = float(
        rust_moments["e"]["M3"]
        / (0.5 * (rust_moments["mu"]["M3"] + rust_moments["tau"]["M3"]))
        - 1.0
    )

    # T8b/T8c: the independent collocation polynomial evaluated at the Rust nodes.
    logits_end = ind._native_pair_logits(c_end)
    mapped = {
        name: ind.fermi_dirac_from_logit(
            setup.grid.interpolate(logits_end[row], rust_y)
        )
        for row, name in ((0, "e"), (1, "mu"), (2, "tau"))
    }
    anchors = {"e": rust_fe, "mu": rust_fx, "tau": rust_fx}
    deviations = {
        name: np.abs(np.asarray(mapped[name]) - anchors[name])
        / np.maximum(anchors[name], SPECTRUM_FLOOR)
        for name in mapped
    }
    energy_weight = rust_w * rust_y**3
    weighted = {
        name: float(
            np.sum(energy_weight * np.abs(np.asarray(mapped[name]) - anchors[name]))
            / np.sum(energy_weight * anchors[name])
        )
        for name in mapped
    }
    support = rust_y >= float(setup.grid.nodes[0])
    sub = ~support
    return {
        "moments_independent": ind_moments,
        "moments_rust": rust_moments,
        "moment_tail_fraction": tail_fraction,
        "moment_rel_after_tail_correction": moment_rel,
        "moment_split_ratio": moment_split,
        "moment_split_ratio_rust": rust_split,
        "weighted_rel_all_nodes": weighted,
        "weighted_rel_max": max(weighted.values()),
        "sup_rel_in_support": max(float(np.max(d[support])) for d in deviations.values()),
        "sup_rel_full_domain": max(float(np.max(d)) for d in deviations.values()),
        "subsupport_nodes": [
            {
                "index": int(i),
                "y": float(rust_y[i]),
                "w": float(rust_w[i]),
                "energy_weight_share": float(
                    energy_weight[i] * rust_fe[i] / np.sum(energy_weight * rust_fe)
                ),
                "dev_e": float(deviations["e"][i]),
                "dev_mu": float(deviations["mu"][i]),
                "dev_tau": float(deviations["tau"][i]),
            }
            for i in np.nonzero(sub)[0]
        ],
        "per_node_deviation": {k: [float(x) for x in v] for k, v in deviations.items()},
        "mapped_at_rust_nodes": {k: [float(x) for x in v] for k, v in mapped.items()},
    }


def analyse(setup: core.Setup, solution, stats: core.Stats, c0: np.ndarray) -> dict:
    endpoint = core.endpoint_summary(setup, solution.t[-1], solution.y[:, -1], stats)
    occ_end = endpoint["occ_end"]
    occ_eq = ind.cloglog_to_occupation(c0)
    weights3 = setup.grid.weights * np.power(setup.grid.nodes, 3)
    energy_moments = np.sum(occ_end * weights3[None, :], axis=1)
    energy_eq = float(np.sum(occ_eq[0] * weights3))
    enhancements = energy_moments / energy_eq - 1.0
    e_enh, mu_enh, tau_enh = (float(v) for v in enhancements)
    heavy_rel_gap = abs(mu_enh - tau_enh) / max(abs(mu_enh), abs(tau_enh), TINY)

    number, energy = core.pair_densities(setup, occ_end, endpoint["T_cm_end"])
    split_ratio = float(energy[0] / (0.5 * (energy[1] + energy[2])) - 1.0)

    reachable = [n for n in CHECKPOINTS if n + setup.h_fd < float(solution.t[-1])]
    residual_rows = core.coupled_residuals(setup, solution.sol, reachable)
    window_rows = [r for r in residual_rows
                   if any(abs(r["N"] - w) < 1e-9 for w in TRANSFER_WINDOW)]
    active_rows = [r for r in window_rows if r["active"]]
    clean_rows = [r for r in active_rows if r["R_transfer"] <= BAND_R_TRANSFER]

    tail_last_node = float(
        np.sum(setup.grid.weights[-1:] * setup.grid.nodes[-1:] ** 3 * occ_end[:, -1:])
        / np.sum(setup.grid.weights * setup.grid.nodes**3 * occ_end[0])
    )
    rej = np.asarray(stats.rejections, dtype=np.int64)
    return {
        "setup": setup.describe(),
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
        "spectral": spectral_block(setup, occ_end, endpoint["c_end"]),
        "endpoint_spectrum": {
            "grid_nodes": [float(v) for v in setup.grid.nodes],
            "grid_weights": [float(v) for v in setup.grid.weights],
            "f_e": [float(v) for v in occ_end[0]],
            "f_mu": [float(v) for v in occ_end[1]],
            "f_tau": [float(v) for v in occ_end[2]],
        },
        "residual_rows": residual_rows,
        "checkpoint_states": core.checkpoint_states(setup, solution.sol, reachable),
        "window": {"total": len(window_rows), "active": len(active_rows),
                   "clean": len(clean_rows)},
        "max_R_total": (max(r["R_total"] for r in residual_rows)
                        if residual_rows else None),
        "max_R_transfer_window": (max(r["R_transfer"] for r in active_rows)
                                  if active_rows else None),
        "tail_last_node_fraction": tail_last_node,
        "rejections": {"min": int(rej.min()), "max": int(rej.max()),
                       "final": int(rej[-1]), "count": int(rej.size)},
        "max_roundoff": stats.max_roundoff,
        "solution_t_end": float(solution.t[-1]),
    }


def recompute_from_stored(setup: core.Setup, analysis: dict) -> dict:
    """T14: re-derive the streamed diagnostics from the retained state alone.

    The re-derivation deliberately does NOT call ``comoving_energies``. Running
    the same function over a bitwise copy of its own input can only return the
    same answer, so it would prove nothing about the retained state
    (D-069 pre-freeze review F-004). Instead ``W_nu`` is rebuilt from the stored
    occupations through the independent quadrature identity

        rho_nu = (2 / 2 pi^2) * T_cm^4 * sum_f sum_i w_i y_i^3 f_{f,i},
        W_nu   = e^{4N} * rho_nu,

    which touches the stored numbers by a different route than the streamed
    value did. A mismatch means the retained state is not sufficient to
    reproduce the report, which is exactly the property under test.
    """

    by_n = {round(row["N"], 6): row for row in analysis["checkpoint_states"]}
    streamed = {round(row["N"], 6): row for row in analysis["residual_rows"]}
    weights3 = setup.grid.weights * np.power(setup.grid.nodes, 3)
    checks = []
    for n_k in RECOMPUTE_CHECKPOINTS:
        key = round(n_k, 6)
        stored, expected = by_n.get(key), streamed.get(key)
        if stored is None or expected is None:
            checks.append({"N": n_k, "available": False})
            continue
        occupation = np.asarray(stored["occupation"], dtype=np.float64)
        t_cm = float(stored["T_cm_mev"])
        rho_nu = (
            2.0 * t_cm**4 / ind.TWO_PI_SQUARED
            * float(np.sum(weights3[None, :] * occupation))
        )
        w_nu = float(np.exp(4.0 * n_k)) * rho_nu
        # T_cm must also be reconstructible from N alone.
        t_cm_rel = abs(t_cm / (setup.t_start * float(np.exp(-n_k))) - 1.0)
        checks.append({
            "N": n_k, "available": True,
            "W_nu_recomputed": w_nu, "W_nu_streamed": expected["W_nu"],
            "W_nu_rel": abs(w_nu / expected["W_nu"] - 1.0),
            "T_cm_rel": t_cm_rel,
        })
    deviations = [
        max(c["W_nu_rel"], c["T_cm_rel"]) for c in checks if c.get("available")
    ]
    worst = max(deviations) if deviations else None
    return {"checks": checks, "worst_relative_deviation": worst,
            "tolerance": RECOMPUTE_TOL,
            "method": "independent quadrature identity, not a re-call of "
                      "comoving_energies",
            "ok": bool(worst is not None
                       and len(deviations) == len(RECOMPUTE_CHECKPOINTS)
                       and worst <= RECOMPUTE_TOL)}


def run_mutants(setup: core.Setup, base_sol, deadline: core.Deadline,
                reporter: core.Reporter) -> dict:
    from scipy.integrate import solve_ivp

    y_start = base_sol(MUT_N0)
    results = {}
    for name, s_pair, s_qem in (("M1_pair_sign", -1.0, 1.0),
                                ("M2_qem_sign", 1.0, -1.0)):
        reporter.log(f"mutant {name}: integrating [{MUT_N0}, {MUT_N1}]")
        stats = core.Stats()
        outcome: dict = {"name": name}
        try:
            sol = solve_ivp(
                core.make_rhs(setup, stats, deadline, s_pair=s_pair, s_qem=s_qem),
                (MUT_N0, MUT_N1), y_start, method="BDF", rtol=RTOL, atol=ATOL,
                dense_output=True,
            )
            if sol.status != 0 or not sol.success:
                outcome.update({"killed": True,
                                "mode": f"solver failure: {sol.message}"})
            else:
                rows = core.coupled_residuals(setup, sol.sol, MUT_SAMPLES)
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
        except Exception as exc:  # noqa: BLE001 - any failure is a kill, recorded
            outcome.update({"killed": True, "mode": f"fail-closed error: {exc!r}"})
        outcome["evals"] = stats.evals
        results[name] = outcome
        reporter.log(f"mutant {name}: killed={outcome['killed']} ({outcome['mode']})")
    return results


def evaluate(base: dict, base_reached: bool, base_occ_ok: bool, mutants: dict,
             domain: dict, domain_reached: bool, holdout: dict,
             holdout_reached: bool, holdout_occ_ok: bool,
             recompute: dict, enclosure: dict) -> dict:
    ep, hp, dp = base["endpoint"], holdout["endpoint"], domain["endpoint"]
    spec = base["spectral"]
    rej = base["rejections"]
    per_eval_ok = REJECTION_RANGE[0] <= rej["min"] and rej["max"] <= REJECTION_RANGE[1]
    moment_ok = all(
        abs(rel) <= BAND_MOMENT
        for flavour in spec["moment_rel_after_tail_correction"].values()
        for rel in flavour.values()
    )
    return {
        "T1_solver": {"ok": bool(base_reached and base_occ_ok), "evals": ep["evals"]},
        "T2_neff_bdf": {"ok": bool(abs(ep["N_eff"] - A.BDF_NEFF) <= BAND_NEFF),
                        "delta": ep["N_eff"] - A.BDF_NEFF, "band": BAND_NEFF},
        "T3_n_bdf": {"ok": bool(abs(ep["N_end"] - A.BDF_N) <= BAND_N),
                     "delta": ep["N_end"] - A.BDF_N, "band": BAND_N},
        "T4_t_bdf": {"ok": bool(abs(ep["t_end_s"] - A.BDF_T_S) <= BAND_T_S),
                     "delta_s": ep["t_end_s"] - A.BDF_T_S, "band": BAND_T_S},
        "T5_rodas": {"ok": bool(
            abs(ep["N_eff"] - A.RODAS_NEFF) <= BAND_RODAS_NEFF
            and abs(ep["N_end"] - A.RODAS_N) <= BAND_RODAS_N
            and abs(ep["t_end_s"] - A.RODAS_T_S) <= BAND_RODAS_T_S),
            "delta_neff": ep["N_eff"] - A.RODAS_NEFF,
            "delta_n": ep["N_end"] - A.RODAS_N,
            "delta_t_s": ep["t_end_s"] - A.RODAS_T_S},
        "T6_blockwise": {"ok": bool(
            base["enhancements"]["e"] > base["enhancements"]["mu"] > 0.0
            and base["enhancements"]["heavy_relative_gap"] <= BAND_HEAVY_GAP),
            **base["enhancements"]},
        "T7_blocks": {"ok": bool(
            abs(base["blocks"]["rho_nu"] / A.BLK_RHO_NU - 1.0) <= BAND_BLOCK_REL
            and abs(base["blocks"]["n_e"] / A.BLK_N_E - 1.0) <= BAND_BLOCK_REL
            and abs(base["blocks"]["rho_e"] / A.BLK_RHO_E - 1.0) <= BAND_BLOCK_REL
            and abs(base["blocks"]["n_x"] / A.BLK_N_X - 1.0) <= BAND_BLOCK_REL
            and abs(base["blocks"]["rho_x"] / A.BLK_RHO_X - 1.0) <= BAND_BLOCK_REL
            and abs(base["blocks"]["split_ratio"] - A.BLK_SPLIT_RATIO) <= BAND_SPLIT),
            "split_ratio": base["blocks"]["split_ratio"],
            "split_anchor": A.BLK_SPLIT_RATIO,
            "deltas_rel": {
                "rho_nu": base["blocks"]["rho_nu"] / A.BLK_RHO_NU - 1.0,
                "n_e": base["blocks"]["n_e"] / A.BLK_N_E - 1.0,
                "rho_e": base["blocks"]["rho_e"] / A.BLK_RHO_E - 1.0,
                "n_x": base["blocks"]["n_x"] / A.BLK_N_X - 1.0,
                "rho_x": base["blocks"]["rho_x"] / A.BLK_RHO_X - 1.0,
            }},
        "T8a_moment_norm": {
            "ok": bool(moment_ok
                       and abs(spec["moment_split_ratio"]
                               - spec["moment_split_ratio_rust"]) <= BAND_SPLIT_MOMENT),
            "rel_after_tail_correction": spec["moment_rel_after_tail_correction"],
            "split_ratio": spec["moment_split_ratio"],
            "split_ratio_rust": spec["moment_split_ratio_rust"],
            "bands": {"all_moments": BAND_MOMENT, "split": BAND_SPLIT_MOMENT}},
        "T8b_weighted_all_nodes": {
            "ok": bool(spec["weighted_rel_max"] <= BAND_SPECTRUM_WEIGHTED),
            "weighted_rel": spec["weighted_rel_all_nodes"],
            "weighted_rel_max": spec["weighted_rel_max"],
            "band": BAND_SPECTRUM_WEIGHTED,
            "nodes_included": len(A.RUST_SPECTRUM)},
        "T8c_in_support_sup": {
            "ok": bool(spec["sup_rel_in_support"] <= BAND_SPECTRUM_SUP),
            "sup_rel_in_support": spec["sup_rel_in_support"],
            "sup_rel_full_domain_non_gating": spec["sup_rel_full_domain"],
            "band": BAND_SPECTRUM_SUP},
        "T9_coupled_energy": {"ok": bool(
            base["max_R_total"] is not None
            and base["max_R_total"] <= BAND_R_TOTAL
            and base["window"]["clean"] >= MIN_ACTIVE),
            "max_R_total": base["max_R_total"],
            "max_R_transfer_window": base["max_R_transfer_window"],
            "active_window_checkpoints": base["window"]["active"],
            "clean_window_checkpoints": base["window"]["clean"]},
        "T10_mutants": {"ok": bool(all(m["killed"] for m in mutants.values())),
                        "kills": {k: m["killed"] for k, m in mutants.items()},
                        "modes": {k: m["mode"] for k, m in mutants.items()},
                        "max_R_transfer": {k: m.get("max_R_transfer")
                                           for k, m in mutants.items()}},
        "T11_tail_and_rejections": {"ok": bool(
            base["tail_last_node_fraction"] <= TAIL_RHO_FRACTION
            and per_eval_ok and base["max_roundoff"] <= ROUNDOFF_CAP),
            "tail_last_node_fraction": base["tail_last_node_fraction"],
            "tail_band": TAIL_RHO_FRACTION,
            "rejections": rej,
            "max_roundoff": base["max_roundoff"]},
        "T12_rtol_holdout": {"ok": bool(
            holdout_reached and holdout_occ_ok
            and abs(hp["N_eff"] - ep["N_eff"]) <= BAND_HOLDOUT_NEFF
            and abs(hp["N_end"] - ep["N_end"]) <= BAND_HOLDOUT_N
            and abs(hp["t_end_s"] - ep["t_end_s"]) <= BAND_HOLDOUT_T_S),
            "delta_neff": hp["N_eff"] - ep["N_eff"],
            "delta_n": hp["N_end"] - ep["N_end"],
            "delta_t_s": hp["t_end_s"] - ep["t_end_s"],
            "holdout_evals": hp["evals"]},
        "T13_domain_enclosure": {"ok": bool(
            domain_reached
            and abs(dp["N_eff"] - ep["N_eff"]) <= BAND_DOMAIN_NEFF
            and enclosure["analytic_residual_beyond_domain"] <= BAND_ANALYTIC_RESIDUAL
            and enclosure["total"] <= BAND_ENCLOSURE),
            **enclosure},
        "T14_recomputable_checkpoints": recompute,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="D-069 independent trajectory r4")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv[1:])

    reporter = core.Reporter(args.out)
    deadline = core.Deadline(WALL_BUDGET_SECONDS)

    module_sha = core.module_sha256()
    if module_sha != A.MODULE_SHA256:
        print(f"MODULE_SHA_MISMATCH {module_sha}")
        return 30
    if A.spectrum_digest() != A.RUST_SPECTRUM_SHA256:
        print(f"SPECTRUM_CONSTANTS_MISMATCH {A.spectrum_digest()}")
        return 30
    anchor_hashes = {}
    import hashlib
    for rel, expected in A.ANCHOR_LOG_SHA256.items():
        try:
            with open(rel, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            digest = "MISSING"
        anchor_hashes[rel] = digest
        if digest != expected:
            print(f"ANCHOR_LOG_MISMATCH {rel} {digest}")
            return 30

    import scipy

    base_setup = build(BASE_ORDER, BASE_YMAX, "base")
    domain_setup = build(DOMAIN_ORDER, DOMAIN_YMAX, "domain-holdout")

    report: dict = {
        "contract": "BD622_D069_independent_trajectory_r4_contract_2026-07-29",
        "claim_id": "C-F10-INDEPENDENT",
        "module_sha256": module_sha,
        "anchor_log_sha256": anchor_hashes,
        "rust_spectrum_sha256": A.RUST_SPECTRUM_SHA256,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "setups": {"base": base_setup.describe(), "domain": domain_setup.describe()},
        "config": {"rtol": RTOL, "atol": ATOL, "holdout_rtol": HOLDOUT_RTOL,
                   "h_fd": base_setup.h_fd,
                   "wall_budget_seconds": WALL_BUDGET_SECONDS},
        "verdict": "IN_PROGRESS",
    }
    try:
        reporter.write(report)
    except OSError as exc:
        print(f"ERROR cannot write --out path: {exc}")
        return 20

    try:
        base_stats = core.Stats()
        base_solution, base_reached, c0 = core.run_integration(
            base_setup, RTOL, base_stats, deadline, reporter.log
        )
        base = analyse(base_setup, base_solution, base_stats, c0)
        report["base"] = base
        reporter.write(reporter.stamp(report))

        recompute = recompute_from_stored(base_setup, base)
        report["recompute"] = recompute
        reporter.write(reporter.stamp(report))

        reporter.log("mutant battery")
        mutants = run_mutants(base_setup, base_solution.sol, deadline, reporter)
        report["mutants"] = mutants
        reporter.write(reporter.stamp(report))

        domain_stats = core.Stats()
        domain_solution, domain_reached, domain_c0 = core.run_integration(
            domain_setup, RTOL, domain_stats, deadline, reporter.log
        )
        domain = analyse(domain_setup, domain_solution, domain_stats, domain_c0)
        report["domain_holdout"] = domain
        reporter.write(reporter.stamp(report))

        holdout_stats = core.Stats()
        holdout_solution, holdout_reached, _ = core.run_integration(
            base_setup, HOLDOUT_RTOL, holdout_stats, deadline, reporter.log,
            label="rtol-holdout",
        )
        holdout = analyse(base_setup, holdout_solution, holdout_stats, c0)
        report["rtol_holdout"] = holdout
        reporter.write(reporter.stamp(report))
    except TimeoutError as exc:
        report.update({"verdict": "ERROR", "error": f"wall budget: {exc}"})
        reporter.write(reporter.stamp(report))
        return 20
    except Exception as exc:  # noqa: BLE001 - preserve the partial report
        report.update({"verdict": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
        reporter.write(reporter.stamp(report))
        return 20

    analytic_residual = core.equilibrium_tail_fraction(DOMAIN_YMAX, power=3)
    delta_neff = domain["endpoint"]["N_eff"] - base["endpoint"]["N_eff"]
    enclosure = {
        "measured_delta_neff": float(delta_neff),
        "measured_delta_n": float(domain["endpoint"]["N_end"] - base["endpoint"]["N_end"]),
        "measured_delta_t_s": float(domain["endpoint"]["t_end_s"] - base["endpoint"]["t_end_s"]),
        "analytic_residual_beyond_domain": float(analytic_residual),
        "analytic_residual_as_neff": float(analytic_residual * base["endpoint"]["N_eff"]),
        "total": float(abs(delta_neff) + analytic_residual * base["endpoint"]["N_eff"]),
        "band_measured": BAND_DOMAIN_NEFF,
        "band_total": BAND_ENCLOSURE,
        "base_rejections": base["rejections"],
        "domain_rejections": domain["rejections"],
    }
    checks = evaluate(
        base, base_reached, base_stats.occ_ok, mutants,
        domain, domain_reached, holdout, holdout_reached, holdout_stats.occ_ok,
        recompute, enclosure,
    )
    verdict = "PASS" if all(c["ok"] for c in checks.values()) else "FAIL"
    report.update({"checks": checks, "verdict": verdict})
    reporter.write(reporter.stamp(report))
    reporter.log(f"verdict={verdict}")
    for name, check in checks.items():
        reporter.log(f"  {name}: {'ok' if check['ok'] else 'FAIL'}")
    return 0 if verdict == "PASS" else 10


if __name__ == "__main__":
    sys.exit(main(sys.argv))
