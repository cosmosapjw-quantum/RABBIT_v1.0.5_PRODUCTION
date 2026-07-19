import math

import pytest

from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI


@pytest.mark.parametrize(
    ("sigma_h_plus", "correction_level"),
    [
        (0.00, 2),
        (0.03, 3),
    ],
)
def test_characteristic_tier2_per_species_envelope_lock(sigma_h_plus, correction_level):
    r = run_full_coupled_typeI(
        FullCoupledConfig(
            Sigma_H_plus=sigma_h_plus,
            Sigma_H_minus=0.0,
            tier=2,
            enable_collisions=True,
            correction_level=correction_level,
            enable_teff=False,
            N_q=12,
            N_mu=12,
        )
    )

    assert r.metadata["transport_species_mode"] == "per_species"
    assert r.metadata["species_identical_approx"] is False
    assert r.metadata["production_authority"] == "characteristic_decoupling_backbone_residual_relaxation"
    assert r.metadata["phase1_solver_diagnostics"]["solver_outcome"] == "target_reached"
    assert r.metadata["phase2_solver_diagnostics"]["solver_outcome"] == "target_reached"

    assert math.isfinite(r.observables.Yp)
    assert math.isfinite(r.observables.DH)
    assert r.observables.Yp > 0.0
    assert r.observables.DH > 0.0
