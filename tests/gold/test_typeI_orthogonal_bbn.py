"""Gold gate: Bianchi Type I orthogonal BBN cell."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI

pytestmark = pytest.mark.gold

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "flrw_gold_v861.json"


def _gold_cl0_12():
    with open(FIXTURE, encoding="utf-8") as handle:
        gold = json.load(handle)
    for entry in gold["entries"]:
        if entry["n_rxn"] == 12 and entry["CL"] == 0:
            return entry
    raise AssertionError("FLRW CL0/12 gold entry not found")


def test_typeI_orthogonal_sigma_zero_recovers_flrw_gold():
    gold = _gold_cl0_12()
    result = run_full_coupled_typeI(
        FullCoupledConfig(
            Sigma_H_plus=0.0,
            Sigma_H_minus=0.0,
            enable_teff=False,
            N_q=20,
            n_reactions=12,
            correction_level=0,
        )
    )

    assert abs(result.observables.Yp - gold["Yp"]) < 1.0e-6
    assert abs(result.observables.DH - gold["DH"]) / gold["DH"] < 5.0e-3


def test_typeI_orthogonal_shear_changes_final_yields():
    baseline = run_full_coupled_typeI(
        FullCoupledConfig(
            Sigma_H_plus=0.0,
            Sigma_H_minus=0.0,
            enable_teff=False,
            N_q=20,
            n_reactions=12,
            correction_level=0,
        )
    )
    sheared = run_full_coupled_typeI(
        FullCoupledConfig(
            Sigma_H_plus=0.1,
            Sigma_H_minus=0.0,
            enable_teff=False,
            N_q=20,
            n_reactions=12,
            correction_level=0,
        )
    )

    assert sheared.observables.Yp > baseline.observables.Yp
    assert sheared.metadata["transport_mode"] == "characteristic"
