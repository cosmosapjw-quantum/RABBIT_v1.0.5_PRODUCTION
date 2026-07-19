from __future__ import annotations

import pytest

from rabbit.config.solver_config import FAST_CONFIG, REFERENCE_CONFIG
from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI


def _solve(*, solver=None):
    cfg = FullCoupledConfig(
        Sigma_H_plus=0.1,
        N_q=8,
        N_mu=8,
        n_reactions=12,
        correction_level=0,
        tier=1,
        enable_teff=False,
        solver=solver,
    )
    return run_full_coupled_typeI(cfg)


@pytest.mark.production
@pytest.mark.gold
def test_typeI_characteristic_tolerance_envelope_against_reference_solve():
    """Tolerance audit: production and fast solves stay near a tight reference."""
    reference = _solve(solver=REFERENCE_CONFIG)
    production = _solve()
    fast = _solve(solver=FAST_CONFIG)

    assert production.metadata["solver_method_effective"] == "BDF"
    assert reference.metadata["solver_method_effective"] == "Radau"
    assert production.metadata["solver_method_requested"] == "BDF"
    assert reference.metadata["solver_method_requested"] == "Radau"

    assert abs(production.observables.Yp - reference.observables.Yp) < 5.0e-6
    rel_dh_prod = abs(production.observables.DH - reference.observables.DH) / reference.observables.DH
    assert rel_dh_prod < 5.0e-4

    assert abs(fast.observables.Yp - reference.observables.Yp) < 1.0e-4
    rel_dh_fast = abs(fast.observables.DH - reference.observables.DH) / reference.observables.DH
    assert rel_dh_fast < 3.0e-3
