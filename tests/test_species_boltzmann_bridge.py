import numpy as np
from scipy.special import roots_laguerre

from rabbit.drivers import full_coupled_typeI as mod
from rabbit.thermo.nudec_coupled import hubble_3T
from rabbit.transport.species_boltzmann_bridge import apply_species_boltzmann_collision


_MEV_TO_S = 1.519267447e21


def _toy_state(*, equilibrium: bool):
    N_mu = 12
    N_q = 20
    mu0, w0, X0, signs = mod.setup_ray_grid(N_mu)
    q_gl_np, q_wgl_np = roots_laguerre(N_q)
    q = q_gl_np.astype(np.float64)
    w = q_wgl_np.astype(np.float64)

    I = np.zeros(N_mu, dtype=np.float64)
    J = np.ones(N_mu, dtype=np.float64)

    if equilibrium:
        Tg = 0.060
        Tne = 0.060
        Tnx = 0.060
    else:
        Tg = 0.080
        Tne = 0.0590989484
        Tnx = 0.0589753145

    H = float(hubble_3T(Tg, Tne, Tnx) * _MEV_TO_S)
    return w0, q, w, I, J, Tg, Tne, Tnx, H


def test_species_boltzmann_bridge_equilibrium_is_quiet():
    w0, q, w, I, J, Tg, Tne, Tnx, H = _toy_state(equilibrium=True)

    for sp in ("nue", "nuebar", "nux"):
        out = apply_species_boltzmann_collision(
            species=sp,
            I=I,
            J=J,
            w0=w0,
            q_nodes=q,
            q_weights=w,
            T_gamma=Tg,
            T_nu_e=Tne,
            T_nu_x=Tnx,
            H=H,
        )
        assert np.max(np.abs(out.C_monopole)) < 1e-12
        assert np.max(np.abs(out.delta_I)) < 1e-12
        assert abs(out.delta_rho_nu) < 1e-12


def test_species_boltzmann_bridge_resolves_heating_hierarchy():
    w0, q, w, I, J, Tg, Tne, Tnx, H = _toy_state(equilibrium=False)

    nue = apply_species_boltzmann_collision(
        species="nue",
        I=I,
        J=J,
        w0=w0,
        q_nodes=q,
        q_weights=w,
        T_gamma=Tg,
        T_nu_e=Tne,
        T_nu_x=Tnx,
        H=H,
    )
    nux = apply_species_boltzmann_collision(
        species="nux",
        I=I,
        J=J,
        w0=w0,
        q_nodes=q,
        q_weights=w,
        T_gamma=Tg,
        T_nu_e=Tne,
        T_nu_x=Tnx,
        H=H,
    )

    assert nue.delta_rho_nu > 0.0
    assert nux.delta_rho_nu > 0.0
    assert nue.delta_rho_nu > nux.delta_rho_nu
    assert nue.gamma_over_H_thermal > nux.gamma_over_H_thermal
    assert nue.source_scale > 0.0
    assert nux.source_scale > 0.0


def test_species_boltzmann_bridge_keeps_nue_and_nuebar_symmetric():
    w0, q, w, I, J, Tg, Tne, Tnx, H = _toy_state(equilibrium=False)

    nue = apply_species_boltzmann_collision(
        species="nue",
        I=I,
        J=J,
        w0=w0,
        q_nodes=q,
        q_weights=w,
        T_gamma=Tg,
        T_nu_e=Tne,
        T_nu_x=Tnx,
        H=H,
    )
    nuebar = apply_species_boltzmann_collision(
        species="nuebar",
        I=I,
        J=J,
        w0=w0,
        q_nodes=q,
        q_weights=w,
        T_gamma=Tg,
        T_nu_e=Tne,
        T_nu_x=Tnx,
        H=H,
    )

    np.testing.assert_allclose(nue.C_monopole, nuebar.C_monopole, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(nue.delta_I, nuebar.delta_I, rtol=0.0, atol=1e-12)
    assert abs(nue.delta_rho_nu - nuebar.delta_rho_nu) < 1e-12
