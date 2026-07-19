"""tests/test_tier1_neff_reporting_consistency.py — BD598 PR-B (A-R1/A-R3).

Honesty layer for the Tier-1 N_eff reporting gap. The default Tier-1 path
reports the entropy-ratio *diagnostic* of the parametric T_ν (~3.011), while the
expansion is seeded by ``config.N_eff`` (3.044). Rather than silently changing a
codebase-wide physical convention (AGENTS.md forbids that), this PR keeps the
reported value unchanged but makes the gap inspectable and correctly labelled:

  * ``Observables.N_eff_input``     — the Hubble-seed N_eff (config.N_eff).
  * ``Observables.N_eff_is_derived``— False at Tier 1 (SPECIFIED diagnostic),
                                      True at Tier ≥ 2 (derived from the evolved
                                      neutrino sector).

The underlying value-convention question (which N_eff is the physically correct
one to report) is escalated, not decided here.
"""

from __future__ import annotations

import pytest

from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig


@pytest.fixture(scope="module")
def flrw_tier1_result():
    cfg = FullCoupledConfig(
        Sigma_H_plus=0.0, Sigma_H_minus=0.0,
        N_q=20, n_reactions=12, correction_level=0,
        N_eff=3.044, enable_teff=False,
    )
    return cfg, run_full_coupled_typeI(cfg)


def test_tier1_reported_neff_is_diagnostic_distinct_from_hubble_seed(flrw_tier1_result):
    """Reported Tier-1 N_eff is the entropy-ratio diagnostic, distinct from the seed."""
    cfg, r = flrw_tier1_result
    # value unchanged (~3.011), within the physical band
    assert 2.99 < r.observables.N_eff < 3.03
    # the gap to the Hubble seed is real and now surfaced, not hidden
    assert abs(r.observables.N_eff - r.observables.N_eff_input) > 1e-3


def test_tier1_exposes_hubble_seed_input(flrw_tier1_result):
    """N_eff_input echoes config.N_eff (the value seeding ρ_ν in the Hubble)."""
    cfg, r = flrw_tier1_result
    assert r.observables.N_eff_input == pytest.approx(cfg.N_eff, abs=1e-9)


def test_tier1_neff_is_specified_not_derived(flrw_tier1_result):
    """Tier-1 N_eff is a parametric diagnostic (SPECIFIED), never DERIVED."""
    cfg, r = flrw_tier1_result
    assert r.observables.N_eff_is_derived is False


@pytest.mark.slow
def test_tier2_neff_is_marked_derived():
    """The other half of the contract: Tier ≥ 2 reports a DERIVED N_eff.

    Tier 2 evolves the neutrino sector (N_eff_from_3T), so the reported value is
    derived, not a parametric echo. Marked slow (full live-neutrino run).
    """
    cfg = FullCoupledConfig(
        Sigma_H_plus=0.0, Sigma_H_minus=0.0,
        N_q=20, n_reactions=12, correction_level=0,
        N_eff=3.044, enable_teff=False, tier=2,
    )
    r = run_full_coupled_typeI(cfg)
    assert r.observables.N_eff_is_derived is True
    assert r.observables.N_eff_input == pytest.approx(cfg.N_eff, abs=1e-9)
