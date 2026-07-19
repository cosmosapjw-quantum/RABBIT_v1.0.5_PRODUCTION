"""tests/test_cross_code_drift.py — v3.0 Phase K drift watcher.

Plan §3.3. Periodically re-runs the live NUDEC_BSM cross-code reference
and compares to the recorded baseline fixture. Hard-fails when upstream
drift exceeds the documented budget so an unexpected NUDEC_BSM update
does not silently invalidate v3.0's Mangano-gap claims.

This test is marked ``@pytest.mark.cross_code`` so it only runs when
the operator opts in (``pytest -m cross_code``). The intent is a CI
nightly hook; routine pytest runs ignore it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _has_nudec_bsm() -> bool:
    """Wrap has_nudec_bsm() so environment partial-config doesn't break collection."""
    try:
        from rabbit.external.nudec_bsm import has_nudec_bsm
        return bool(has_nudec_bsm())
    except Exception:
        return False


_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "nudec_bsm_drift_baseline.json"


def _load_fixture():
    with open(_FIXTURE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_drift_fixture_exists_and_is_well_formed():
    """Fixture file is on disk and has the documented schema.

    Always-on smoke; does NOT require NUDEC_BSM to be installed.
    """
    fixture = _load_fixture()
    assert "version" in fixture
    assert "input" in fixture and "output_reference" in fixture
    assert "drift_budget" in fixture
    assert "N_eff" in fixture["output_reference"]
    assert "max_neff_drift" in fixture["drift_budget"]
    assert "max_neff_to_mangano_drift" in fixture["drift_budget"]
    # Sanity: fixture N_eff must be in the published NUDEC_BSM ballpark.
    nudec_neff = float(fixture["output_reference"]["N_eff"])
    assert 3.04 < nudec_neff < 3.05, (
        f"fixture N_eff={nudec_neff} outside published NUDEC_BSM ballpark"
    )


@pytest.mark.cross_code
@pytest.mark.skipif(
    not _has_nudec_bsm(),
    reason=(
        "NUDEC_BSM not reachable. Drift watcher opt-in: clone "
        "https://github.com/MiguelEA/nuDec_BSM and set "
        "RABBIT_NUDEC_BSM_PATH=/path/to/nuDec_BSM."
    ),
)
def test_nudec_bsm_no_upstream_drift():
    """Live NUDEC_BSM N_eff has not drifted from the recorded baseline.

    Plan §3.3 hard-fail conditions:
      - |live N_eff - fixture N_eff| > 1e-3
      - |live N_eff - 3.044|         > 5e-3 (NUDEC_BSM-vs-Mangano)

    Either condition indicates an upstream-NUDEC_BSM update that
    invalidates v3.0's Mangano-gap claim or a regression in the
    rabbit-side wrapper. Both must be investigated before promoting
    further work.
    """
    import contextlib
    import io
    from rabbit.external.nudec_bsm import run_nudec_bsm

    fixture = _load_fixture()
    inp = fixture["input"]
    expected_neff = float(fixture["output_reference"]["N_eff"])
    drift_budget = fixture["drift_budget"]

    # NUDEC_BSM emits stdout chatter; suppress so test output stays clean.
    _trash = io.StringIO()
    with contextlib.redirect_stdout(_trash):
        live = run_nudec_bsm(
            eta=float(inp["eta"]),
            tau_n=float(inp["tau_n"]),
            n_eff_target=float(inp["n_eff_target"]),
        )
    live_neff = float(live.N_eff)

    drift_vs_fixture = abs(live_neff - expected_neff)
    drift_vs_mangano = abs(live_neff - 3.044)

    assert drift_vs_fixture < float(drift_budget["max_neff_drift"]), (
        f"v3.0 §3.3 drift watcher: live NUDEC_BSM N_eff drifted from "
        f"fixture: live={live_neff:.6f}, fixture={expected_neff:.6f}, "
        f"|drift|={drift_vs_fixture:.3e} > budget="
        f"{drift_budget['max_neff_drift']:.3e}. Investigate upstream "
        f"NUDEC_BSM SHA before re-running v3.0 gates."
    )
    assert drift_vs_mangano < float(drift_budget["max_neff_to_mangano_drift"]), (
        f"v3.0 §3.3 drift watcher: live NUDEC_BSM diverged from Mangano "
        f"3.044: live={live_neff:.6f}, |drift|={drift_vs_mangano:.3e} > "
        f"budget={drift_budget['max_neff_to_mangano_drift']:.3e}."
    )
