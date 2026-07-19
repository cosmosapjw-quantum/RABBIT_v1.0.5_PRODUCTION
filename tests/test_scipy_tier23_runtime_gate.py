import time

from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI


def test_scipy_tier23_runtime_gate():
    cfg = FullCoupledConfig(
        Sigma_H_plus=0.02,
        Sigma_H_minus=0.0,
        tier=2,
        enable_collisions=True,
        correction_level=3,
        enable_teff=False,
        N_q=12,
        N_mu=12,
    )

    t0 = time.perf_counter()
    r = run_full_coupled_typeI(cfg)
    dt = time.perf_counter() - t0

    assert r.metadata["transport_species_mode"] == "per_species"
    assert r.metadata["production_authority"] == "characteristic_decoupling_backbone_residual_relaxation"
    assert r.metadata["decoupling_backbone_mode"] == "isotropic_momentum_grid_v1"
    assert dt < 60.0, f"Tier-2/3 representative SciPy solve regressed: {dt:.3f}s >= 60.0s"
