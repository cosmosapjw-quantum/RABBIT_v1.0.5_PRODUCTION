"""T2: FLRW baseline regression test.

PURPOSE:
  Lock down the FLRW baseline (Σ=0) across the (n_reactions, correction_level)
  parameter space. This is the decisive missing evidence identified by audit:
  "baseline is not robustly demonstrated as a reference."

DESIGN:
  Gold reference table stored as JSON (tests/fixtures/flrw_gold_v861.json).
  Test compares live solver output against the table.
  Three tiers controlled by RABBIT_TEST_TIER env var:
    smoke (default): 1 run (CL0, 12rxn) ~50s
    fast:            3 runs (CL0/1/2, 12rxn) ~4min
    full:            6 runs (CL0/1/2 × 12/31 rxn) ~9min

ACCEPTANCE CRITERIA:
  AC1: All runs succeed (no solver failure).
  AC2: Y_p monotonically increases with CL (Born < Coulomb < Sirlin).
  AC3: N_eff ∈ [2.99, 3.05].
  AC4: Gold table parity within stored tolerances.
  AC5: 12-rxn vs 31-rxn Y_p match to < 1e-6 (Y_p from weak freeze-out).
  AC6: 12-rxn vs 31-rxn D/H match to < 0.01%.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig

TIER = os.environ.get("RABBIT_TEST_TIER", "smoke")
FIXTURE = Path(__file__).parent / "fixtures" / "flrw_gold_v861.json"


def load_gold():
    with open(FIXTURE) as f:
        return json.load(f)


def _run_flrw(n_rxn=12, cl=0):
    """Run a single FLRW baseline point."""
    cfg = FullCoupledConfig(
        Sigma_H_plus=0.0, Sigma_H_minus=0.0,
        N_q=20, n_reactions=n_rxn, correction_level=cl,
        enable_teff=False,
    )
    return run_full_coupled_typeI(cfg)


def _find_gold(gold, n_rxn, cl):
    """Find matching gold entry."""
    for e in gold["entries"]:
        if e["n_rxn"] == n_rxn and e["CL"] == cl:
            return e
    raise ValueError(f"No gold entry for n_rxn={n_rxn}, CL={cl}")


# ═══════════════════════════════════════════════════════════════
# SMOKE TIER — always runs
# ═══════════════════════════════════════════════════════════════

class TestFLRWSmoke:
    """Single CL0/12-rxn run against gold table."""

    def test_flrw_baseline_cl0_succeeds(self):
        """AC1: FLRW CL0 baseline runs without failure."""
        r = _run_flrw(n_rxn=12, cl=0)
        assert r.observables.Yp > 0.2
        assert r.observables.DH > 1e-6

    def test_flrw_baseline_cl0_gold_parity(self):
        """AC4: CL0 matches gold table."""
        gold = load_gold()
        tol = gold["tolerances"]
        g = _find_gold(gold, 12, 0)
        r = _run_flrw(n_rxn=12, cl=0)

        assert abs(r.observables.Yp - g["Yp"]) < tol["Yp_abs"], \
            f"Y_p: {r.observables.Yp} vs gold {g['Yp']}"
        assert abs(r.observables.DH - g["DH"]) / g["DH"] < tol["DH_rel"], \
            f"D/H: {r.observables.DH} vs gold {g['DH']}"
        assert abs(r.observables.N_eff - g["N_eff"]) < tol["N_eff_abs"], \
            f"N_eff: {r.observables.N_eff} vs gold {g['N_eff']}"

    def test_neff_in_range(self):
        """AC3: N_eff within physical range."""
        r = _run_flrw(n_rxn=12, cl=0)
        assert 2.99 < r.observables.N_eff < 3.05


# ═══════════════════════════════════════════════════════════════
# FAST TIER — CL monotonicity + network parity
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(TIER == "smoke", reason="fast tier only")
class TestFLRWMonotonicity:
    """AC2: Y_p increases with correction level."""

    def test_yp_monotone_with_cl(self):
        results = {}
        for cl in [0, 1, 2]:
            results[cl] = _run_flrw(n_rxn=12, cl=cl)

        yp = [results[cl].observables.Yp for cl in [0, 1, 2]]
        assert yp[1] > yp[0], f"CL1 ({yp[1]}) should exceed CL0 ({yp[0]})"
        assert yp[2] > yp[1], f"CL2 ({yp[2]}) should exceed CL1 ({yp[1]})"

    def test_correction_budget_reasonable(self):
        r0 = _run_flrw(n_rxn=12, cl=0)
        r2 = _run_flrw(n_rxn=12, cl=2)
        budget = (r2.observables.Yp - r0.observables.Yp) / r0.observables.Yp
        # Coulomb + Sirlin should add 1-5% to Y_p
        assert 0.01 < budget < 0.05, f"Correction budget {budget:.4f} outside [1%, 5%]"


@pytest.mark.skipif(TIER == "smoke", reason="fast tier only")
class TestFLRWNetworkParity:
    """AC5/AC6: 12-rxn vs 31-rxn parity."""

    def test_yp_network_parity(self):
        """AC5: Y_p identical between 12 and 31 reactions."""
        r12 = _run_flrw(n_rxn=12, cl=0)
        r31 = _run_flrw(n_rxn=31, cl=0)
        delta = abs(r12.observables.Yp - r31.observables.Yp)
        assert delta < 1e-6, f"|ΔY_p| = {delta:.2e} > 1e-6"

    def test_dh_network_parity(self):
        """AC6: D/H matches to < 0.01%."""
        r12 = _run_flrw(n_rxn=12, cl=0)
        r31 = _run_flrw(n_rxn=31, cl=0)
        rel = abs(r12.observables.DH - r31.observables.DH) / r12.observables.DH
        assert rel < 1e-4, f"D/H relative diff = {rel:.2e} > 1e-4"

    def test_li7_network_shift(self):
        """31-rxn Li7 should be ~1-3% lower than 12-rxn (R30 destruction)."""
        r12 = _run_flrw(n_rxn=12, cl=0)
        r31 = _run_flrw(n_rxn=31, cl=0)
        shift = (r31.observables.Li7H - r12.observables.Li7H) / r12.observables.Li7H
        # R30 (7Li+D→4He4He+n) destroys ~1-3% of 7Li
        assert -0.05 < shift < 0.0, f"Li7 shift = {shift:.4f}, expected -0.03 to 0"


# ═══════════════════════════════════════════════════════════════
# FULL TIER — complete gold table verification
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(TIER != "full", reason="full tier only")
class TestFLRWGoldTable:
    """AC4: All 6 gold entries reproduced."""

    @pytest.mark.parametrize("n_rxn,cl", [
        (12, 0), (12, 1), (12, 2),
        (31, 0), (31, 1), (31, 2),
    ])
    def test_gold_entry(self, n_rxn, cl):
        gold = load_gold()
        tol = gold["tolerances"]
        g = _find_gold(gold, n_rxn, cl)
        r = _run_flrw(n_rxn=n_rxn, cl=cl)

        assert abs(r.observables.Yp - g["Yp"]) < tol["Yp_abs"], \
            f"Y_p: {r.observables.Yp:.8f} vs gold {g['Yp']:.8f}"
        assert abs(r.observables.DH - g["DH"]) / g["DH"] < tol["DH_rel"], \
            f"D/H: {r.observables.DH:.6e} vs gold {g['DH']:.6e}"
        if g["Li7H"] > 0:
            assert abs(r.observables.Li7H - g["Li7H"]) / g["Li7H"] < tol["Li7H_rel"], \
                f"Li7H: {r.observables.Li7H:.4e} vs gold {g['Li7H']:.4e}"
        assert abs(r.observables.N_eff - g["N_eff"]) < tol["N_eff_abs"], \
            f"N_eff: {r.observables.N_eff:.4f} vs gold {g['N_eff']:.4f}"
