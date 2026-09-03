"""BD622 D-050 — OWNER-C row-6 closure verification (C1, C3, C4, C5).

Runs against the modified ``_independent_noqke.py``; thresholds frozen in
``docs/audit/BD622_D050_ownerc_row6_closure_contract_2026-07-27.md``.

Usage: PYTHONPATH=src python3 scripts/audit/d050_ownerc_closure_verification.py [--out PATH]
"""

from __future__ import annotations

import json
import os
import sys

for _pin in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_pin, "1")

import numpy as np

from rabbit.decoupling import _independent_noqke as ind

PREDICTED = 3.8807636157303184e-11
MU_TAU_LEGS = {
    ("nu_mu", "antinu_mu", "nu_tau", "antinu_tau"),
    ("nu_tau", "antinu_tau", "nu_mu", "antinu_mu"),
}


def check_c1():
    events = ind.independent_self_events()
    fingerprint = ind.independent_pair_row_fingerprint()
    members = [e for e in events if e.category == "pair_conversion"
               and {leg.split("_", 1)[1] for leg in e.legs} == {"mu", "tau"}]
    legs = {m.legs for m in members}
    ok = (
        len(events) == 25
        and fingerprint == (2, 1, 6, 2, 2, 2, 4, 4, 2)
        and len(members) == 2
        and legs == MU_TAU_LEGS
        and all(m.kernel == "K_t" and m.coefficient == 16.0 for m in members)
    )
    from collections import defaultdict

    agg = defaultdict(float)
    for event in events:
        for species in set(event.legs):
            agg[(event.category, species)] += event.coefficient * event.legs.count(species)
    target = defaultdict(float)
    for row in ind.independent_self_reactions():
        target[(row.category, row.target)] += row.coefficient
    ok = ok and dict(agg) == dict(target)
    return {"ok": bool(ok), "events": len(events), "fingerprint": list(fingerprint),
            "member_count": len(members)}


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

    c1 = check_c1()

    grid = ind.build_independent_grid(48, 24.0)
    config = ind.IndependentCollisionConfig()
    scales = (1.01, 0.995, 0.995)
    split = ind.pair_logits_to_cloglog(np.stack([-grid.nodes / s for s in scales]))
    action = ind.evaluate_independent_collision_action(
        grid=grid, pair_cloglog=split,
        temperature_cm_mev=10.0, temperature_gamma_mev=10.0, config=config,
    )
    resid = float(action.diagnostics["mu_tau_residual"])
    agree = abs(resid - PREDICTED) / PREDICTED
    c3 = {"ok": bool(1e-12 <= resid <= 1e-10 and agree <= 5e-1),
          "mu_tau_residual": resid, "prediction_agreement": agree}

    moments = ind.independent_action_moments(
        grid=grid, action=action.self_interaction, temperature_cm_mev=10.0
    )
    tiny = float(np.finfo(float).tiny)
    number_ratio = abs(moments.signed_number_rate) / max(moments.absolute_number_rate, tiny)
    energy_ratio = abs(moments.signed_energy_rate) / max(moments.absolute_energy_rate, tiny)
    first_law = float(action.diagnostics["first_law_residual"])
    cp = float(action.diagnostics["charge_conjugation_residual"])
    entropy = float(action.diagnostics["entropy_production"])
    c4 = {"ok": bool(number_ratio <= 1e-12 and energy_ratio <= 1e-12
                     and first_law <= 1e-8 and cp <= 1e-10 and entropy >= -1e-24),
          "number_ratio": float(number_ratio), "energy_ratio": float(energy_ratio),
          "first_law_residual": first_law, "charge_conjugation_residual": cp,
          "entropy_production": entropy}

    common = ind.pair_logits_to_cloglog(np.stack([-grid.nodes for _ in range(3)]))
    null_action = ind.evaluate_independent_collision_action(
        grid=grid, pair_cloglog=common,
        temperature_cm_mev=10.0, temperature_gamma_mev=10.0, config=config,
    )
    thermo = ind.independent_thermodynamics(
        grid=grid, pair_cloglog=common,
        temperature_cm_mev=10.0, temperature_gamma_mev=10.0,
    )
    null = ind.independent_action_moments(
        grid=grid, action=null_action.total, temperature_cm_mev=10.0
    )
    h_number = null.absolute_number_rate / (thermo.hubble_mev * thermo.number_density_neutrino)
    h_energy = null.absolute_energy_rate / (thermo.hubble_mev * thermo.energy_density_neutrino)
    c5 = {"ok": bool(h_number <= 1e-10 and h_energy <= 1e-10),
          "h_normalized_number": float(h_number), "h_normalized_energy": float(h_energy)}

    checks = {"C1_catalogue": c1, "C3_covariance": c3, "C4_diagnostics": c4,
              "C5_equilibrium_null": c5}
    verdict = "PASS" if all(c["ok"] for c in checks.values()) else "FAIL"
    report = {"contract": "BD622_D050_ownerc_row6_closure_contract_2026-07-27",
              "checks": checks, "verdict": verdict}
    text = json.dumps(report, sort_keys=True, indent=1)
    print(text)
    if out_path is not None:
        with open(out_path, "w") as fh:
            fh.write(text + "\n")
    return 0 if verdict == "PASS" else 10


if __name__ == "__main__":
    sys.exit(main(sys.argv))
