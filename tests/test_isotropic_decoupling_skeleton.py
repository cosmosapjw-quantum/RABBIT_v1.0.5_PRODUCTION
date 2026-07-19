import numpy as np

from rabbit.decoupling import (
    DecouplingGrid,
    evaluate_diagonal_collision_rhs,
    solve_isotropic_decoupling,
    IsotropicDecouplingConfig,
)
from rabbit.decoupling.moments import fermi_dirac_comoving
from rabbit.thermo.nudec_coupled import hubble_3T


def test_diagonal_collision_quiet_in_equilibrium():
    grid = DecouplingGrid(n_y=21)
    T_gamma = 1.5
    a_scale = 1.0
    z = T_gamma * a_scale / 0.51099895
    f_eq = fermi_dirac_comoving(grid.nodes, z)
    H = hubble_3T(T_gamma, T_gamma, T_gamma)

    out = evaluate_diagonal_collision_rhs(
        species="nue",
        f_state=f_eq,
        grid=grid,
        T_gamma=T_gamma,
        T_nu_e=T_gamma,
        T_nu_x=T_gamma,
        H_MeV=H,
        a_scale=a_scale,
    )

    assert np.max(np.abs(out.df_dN)) < 1e-12
    assert abs(out.energy_gain_per_efold) < 1e-12


def test_diagonal_collision_heating_hierarchy():
    grid = DecouplingGrid(n_y=21)
    T_gamma = 1.5
    T_nu_e = 1.45
    T_nu_x = 1.40
    a_scale = 1.0
    f_e = fermi_dirac_comoving(grid.nodes, T_nu_e * a_scale / 0.51099895)
    f_x = fermi_dirac_comoving(grid.nodes, T_nu_x * a_scale / 0.51099895)
    H = hubble_3T(T_gamma, T_nu_e, T_nu_x)

    nue = evaluate_diagonal_collision_rhs(
        species="nue",
        f_state=f_e,
        grid=grid,
        T_gamma=T_gamma,
        T_nu_e=T_nu_e,
        T_nu_x=T_nu_x,
        H_MeV=H,
        a_scale=a_scale,
    )
    nux = evaluate_diagonal_collision_rhs(
        species="nux",
        f_state=f_x,
        grid=grid,
        T_gamma=T_gamma,
        T_nu_e=T_nu_e,
        T_nu_x=T_nu_x,
        H_MeV=H,
        a_scale=a_scale,
    )

    assert nue.energy_gain_per_efold > 0.0
    assert nux.energy_gain_per_efold > 0.0
    assert nue.energy_gain_per_efold > nux.energy_gain_per_efold
    assert nue.gamma_over_H_thermal > nux.gamma_over_H_thermal


def test_isotropic_decoupling_solver_smoke():
    result = solve_isotropic_decoupling(
        IsotropicDecouplingConfig(
            T_start=3.0,
            T_end=0.20,
            n_y=21,
            max_efolds=6.0,
            rtol=1e-5,
            atol=1e-8,
        )
    )

    assert result.success
    assert result.solver_diagnostics["solver_outcome"] == "target_reached"
    assert np.isfinite(result.T_gamma_final)
    assert np.isfinite(result.T_nu_e_final)
    assert np.isfinite(result.T_nu_x_final)
    assert np.isfinite(result.N_eff_final)
    assert result.T_gamma_final <= 0.2000001
    assert result.T_nu_e_final > 0.0
    assert result.T_nu_x_final > 0.0
    assert result.N_eff_final > 3.0
    assert "f_nue" in result.trajectory
    assert "dT_gamma_dN" in result.trajectory
    assert "dT_nu_e_dN" in result.trajectory
    assert result.trajectory["f_nue"].shape[0] == result.trajectory["N"].shape[0]
