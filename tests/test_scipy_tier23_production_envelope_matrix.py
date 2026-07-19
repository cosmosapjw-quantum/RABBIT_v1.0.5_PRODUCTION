import math

import pytest

from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI


@pytest.mark.parametrize(
    ("sigma_h_plus", "correction_level"),
    [
        (0.00, 2),
        (0.00, 3),
        (0.03, 2),
        (0.05, 3),
    ],
)
def test_scipy_tier23_production_envelope_matrix(sigma_h_plus, correction_level):
    result = run_full_coupled_typeI(
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

    md = result.metadata
    assert md["transport_species_mode"] == "per_species"
    assert md["species_identical_approx"] is False
    assert md["production_authority"] == "characteristic_decoupling_backbone_residual_relaxation"
    assert md["collision_closure_mode"] == "anisotropic_residual_relaxation_v1"
    assert md["thermo_exchange_mode"] == "isotropic_decoupling_backbone_v1"
    assert md["weak_background_mode"] == "isotropic_decoupling_backbone_v1"
    assert md["decoupling_backbone_mode"] == "isotropic_momentum_grid_v1"
    assert md["phase1_solver_diagnostics"]["solver_outcome"] == "target_reached"
    assert md["phase2_solver_diagnostics"]["solver_outcome"] == "target_reached"
    assert md["decoupling_backbone_solver_diagnostics"]["solver_outcome"] == "target_reached"

    if correction_level == 2:
        assert md["weak_budget_mode"] == "born+coulomb+sirlin"
        assert md["correction_budget_channels"]["radiative_sirlin"] is True
        assert md["correction_budget_channels"]["finite_mass_recoil_wm"] is False
    else:
        assert md["weak_budget_mode"] == "born+coulomb+sirlin+finite_mass"
        assert md["correction_budget_channels"]["finite_mass_recoil_wm"] is True

    assert math.isfinite(result.observables.Yp)
    assert math.isfinite(result.observables.DH)
    assert result.observables.Yp > 0.0
    assert result.observables.DH > 0.0
