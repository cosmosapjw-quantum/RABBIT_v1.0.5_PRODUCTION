#!/usr/bin/env python3
"""Regenerate FLRW gold table from current code.

Runs all 6 configurations (CL0/1/2 × 12/31 rxn) and writes
tests/fixtures/flrw_gold_v90rc.json.

Also updates the symlink/copy at flrw_gold_v861.json so existing
tests pick up the new values.
"""
import sys, os, json, time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

def run(n_rxn, cl):
    cfg = FullCoupledConfig(
        Sigma_H_plus=0.0, Sigma_H_minus=0.0,
        N_q=20, correction_level=cl, n_reactions=n_rxn,
        enable_teff=False, tier=1,
    )
    t0 = time.perf_counter()
    r = run_full_coupled_typeI(cfg)
    elapsed = time.perf_counter() - t0
    o = r.observables
    print(f"  CL{cl} {n_rxn}-rxn: Yp={o.Yp:.10f}  DH={o.DH:.6e}  "
          f"Li7H={o.Li7H:.4e}  Neff={o.N_eff:.6f}  ({elapsed:.1f}s)")
    return {
        "n_rxn": n_rxn, "N_q": 20, "rtol": 1e-8, "CL": cl,
        "Yp": round(o.Yp, 10), "DH": round(o.DH, 11),
        "Li7H": round(o.Li7H, 14), "Li6H": round(o.Li6H, 30),
        "N_eff": round(o.N_eff, 6),
    }

def main():
    print("Regenerating FLRW gold table...")
    entries = []
    for n_rxn in [12, 31]:
        for cl in [0, 1, 2]:
            entries.append(run(n_rxn, cl))

    gold = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "code_version": "v9.0-rc+PR16H-next",
        "description": "FLRW baseline gold table. Sigma=0, N_q=20, rtol=1e-8, enable_teff=False.",
        "tolerances": {
            "Yp_abs": 1e-6,
            "DH_rel": 0.005,
            "Li7H_rel": 0.05,
            "N_eff_abs": 0.02,
        },
        "entries": entries,
    }

    # Write new version
    new_path = FIXTURE_DIR / "flrw_gold_v90rc.json"
    with open(new_path, "w") as f:
        json.dump(gold, f, indent=4)
    print(f"\nWritten: {new_path}")

    # Overwrite old fixture so existing tests pass
    old_path = FIXTURE_DIR / "flrw_gold_v861.json"
    with open(old_path, "w") as f:
        json.dump(gold, f, indent=4)
    print(f"Updated: {old_path}")

    # Verification
    print("\nVerification (re-read):")
    with open(old_path) as f:
        check = json.load(f)
    print(f"  Version: {check['code_version']}")
    print(f"  Entries: {len(check['entries'])}")
    for e in check["entries"]:
        print(f"    CL{e['CL']} {e['n_rxn']}-rxn: Yp={e['Yp']}")

if __name__ == "__main__":
    main()
