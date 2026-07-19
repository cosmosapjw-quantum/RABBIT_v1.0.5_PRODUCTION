import math

import pytest

from rabbit.inference.forward_likelihood import canonical_forward_solver


@pytest.mark.parametrize(
    ("sigma_h", "correction_level"),
    [
        (0.00, 2),
        (0.00, 3),
        (0.03, 2),
        (0.05, 3),
    ],
)
def test_scipy_tier23_inference_envelope_matrix(sigma_h, correction_level):
    pred = canonical_forward_solver(
        Sigma_H=sigma_h,
        backend="scipy",
        correction_level=correction_level,
        enable_collisions=True,
        enable_teff=False,
        N_q=12,
    )

    assert pred.success
    assert pred.metadata["transport_mode"] == "characteristic"
    assert pred.metadata["transport_species_mode"] == "per_species"
    assert pred.metadata["species_identical_approx"] is False
    assert pred.metadata["production_authority"] == "characteristic_decoupling_backbone_residual_relaxation"

    if correction_level == 2:
        assert pred.metadata["weak_budget_mode"] == "born+coulomb+sirlin"
        assert pred.metadata["correction_budget_channels"]["radiative_sirlin"] is True
        assert pred.metadata["correction_budget_channels"]["finite_mass_recoil_wm"] is False
    else:
        assert pred.metadata["weak_budget_mode"] == "born+coulomb+sirlin+finite_mass"
        assert pred.metadata["correction_budget_channels"]["finite_mass_recoil_wm"] is True

    assert math.isfinite(pred.Yp)
    assert math.isfinite(pred.DH)
    assert pred.Yp > 0.0
    assert pred.DH > 0.0
