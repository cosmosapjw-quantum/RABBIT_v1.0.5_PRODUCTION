from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig


def test_scipy_driver_reports_cl2_budget():
    r = run_full_coupled_typeI(
        FullCoupledConfig(Sigma_H_plus=0.0, N_q=6, correction_level=2, enable_teff=False)
    )
    assert r.metadata["weak_budget_mode"] == "born+coulomb+sirlin"
    assert r.metadata["fidelity_breakdown"]["weak"] == "prod_hier"


def test_scipy_driver_reports_cl3_budget():
    r = run_full_coupled_typeI(
        FullCoupledConfig(Sigma_H_plus=0.0, N_q=6, correction_level=3, enable_teff=False)
    )
    assert r.metadata["weak_budget_mode"] == "born+coulomb+sirlin+finite_mass"
    assert r.metadata["correction_budget_channels"]["finite_mass_recoil_wm"] is True
