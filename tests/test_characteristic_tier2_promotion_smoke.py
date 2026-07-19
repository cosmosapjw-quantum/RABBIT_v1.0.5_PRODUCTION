from rabbit.inference.forward_likelihood import canonical_forward_solver


def test_scipy_characteristic_tier2_promoted_smoke():
    pred = canonical_forward_solver(
        Sigma_H=0.03,
        backend="scipy",
        correction_level=3,
        enable_collisions=True,
        enable_teff=False,
        N_q=12,
    )
    assert pred.success
    assert pred.metadata["characteristic_species_mode_requested"] == "auto"
    assert pred.metadata["characteristic_species_mode_resolved"] == "per_species"
    assert pred.metadata["transport_mode"] == "characteristic"
    assert pred.metadata["transport_species_mode"] == "per_species"
    assert pred.metadata["species_identical_approx"] is False
    assert pred.metadata["production_authority"] == "characteristic_decoupling_backbone_residual_relaxation"


def test_scipy_characteristic_tier2_is_not_legacy_shared():
    pred = canonical_forward_solver(
        Sigma_H=0.0,
        backend="scipy",
        correction_level=3,
        enable_collisions=True,
        enable_teff=False,
        N_q=12,
    )
    assert pred.success
    assert pred.metadata["transport_species_mode"] == "per_species"
    assert pred.metadata["species_identical_approx"] is False
