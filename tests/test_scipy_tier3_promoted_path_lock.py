from rabbit.inference.forward_likelihood import canonical_forward_solver


def test_scipy_tier3_promoted_path_lock():
    pred = canonical_forward_solver(
        Sigma_H=0.02,
        backend="scipy",
        correction_level=3,
        enable_collisions=True,
        enable_teff=False,
        N_q=12,
    )

    assert pred.success
    assert pred.metadata["transport_mode"] == "characteristic"
    assert pred.metadata["transport_species_mode"] == "per_species"
    assert pred.metadata["species_identical_approx"] is False
    assert pred.metadata["production_authority"] == "characteristic_decoupling_backbone_residual_relaxation"
    assert pred.metadata["weak_budget_mode"] == "born+coulomb+sirlin+finite_mass"
    assert pred.metadata["correction_budget_channels"]["finite_mass_recoil_wm"] is True
