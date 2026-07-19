from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI


def test_tier2_backbone_residual_transport_modulates_handoff_weak_rates():
    cfg = dict(
        tier=2,
        enable_collisions=True,
        correction_level=3,
        enable_teff=False,
        N_q=12,
        N_mu=12,
    )

    flrw = run_full_coupled_typeI(FullCoupledConfig(Sigma_H_plus=0.0, **cfg))
    shear = run_full_coupled_typeI(FullCoupledConfig(Sigma_H_plus=0.03, **cfg))

    assert flrw.metadata["decoupling_backbone_mode"] == "isotropic_momentum_grid_v1"
    assert shear.metadata["decoupling_backbone_mode"] == "isotropic_momentum_grid_v1"

    lam0 = float(flrw.metadata["phase1_handoff_lambda_np"])
    lam1 = float(shear.metadata["phase1_handoff_lambda_np"])
    assert lam0 > 0.0
    assert lam1 > 0.0
    assert abs(lam1 - lam0) > 0.0

    m10 = float(flrw.metadata["phase1_handoff_monopole_probe"]["m1"])
    m11 = float(shear.metadata["phase1_handoff_monopole_probe"]["m1"])
    assert abs(m11 - m10) > 0.0
