import numpy as np
from scipy.special import roots_laguerre

from rabbit.drivers import full_coupled_typeI as mod
from rabbit.transport.species_tagged_bridge import apply_species_tagged_bridge
from rabbit.transport.teff_collision_bridge import apply_gather_scatter_collision
from rabbit.collisions.species import BANK_DEGENERACY, Species
from rabbit.thermo.incomplete_decoupling import compute_energy_exchange_rate


def _toy_state():
    N_mu = 12
    N_q = 20
    mu0, w0, X0, signs = mod.setup_ray_grid(N_mu)
    q_gl_np, q_wgl_np = roots_laguerre(N_q)
    q = q_gl_np.astype(np.float64)
    w = q_wgl_np.astype(np.float64)

    I = np.zeros(N_mu, dtype=np.float64)
    J = np.ones(N_mu, dtype=np.float64)

    Tg = 0.08
    Tne = 0.0590989484
    Tnx = 0.0589753145
    H = 1.0
    return w0, q, w, I, J, Tg, Tne, Tnx, H


def test_nue_and_nux_differ_in_mvp_species_tagged_bridge():
    w0, q, w, I, J, Tg, Tne, Tnx, H = _toy_state()

    gs_nue = apply_species_tagged_bridge(
        species="nue", I=I, J=J, w0=w0, q_nodes=q, q_weights=w,
        T_gamma=Tg, T_nu_e=Tne, T_nu_x=Tnx, H=H,
    )
    gs_nux = apply_species_tagged_bridge(
        species="nux", I=I, J=J, w0=w0, q_nodes=q, q_weights=w,
        T_gamma=Tg, T_nu_e=Tne, T_nu_x=Tnx, H=H,
    )

    qdot_nue = float(compute_energy_exchange_rate(gs_nue.C_monopole, q, w, Tne))
    qdot_nux = float(compute_energy_exchange_rate(gs_nux.C_monopole, q, w, Tnx))

    assert np.isfinite(qdot_nue)
    assert np.isfinite(qdot_nux)
    assert abs(qdot_nue - qdot_nux) > 0.0


def test_nue_and_nuebar_split_in_v4():
    w0, q, w, I, J, Tg, Tne, Tnx, H = _toy_state()

    gs_nue = apply_species_tagged_bridge(
        species="nue", I=I, J=J, w0=w0, q_nodes=q, q_weights=w,
        T_gamma=Tg, T_nu_e=Tne, T_nu_x=Tnx, H=H,
    )
    gs_nuebar = apply_species_tagged_bridge(
        species="nuebar", I=I, J=J, w0=w0, q_nodes=q, q_weights=w,
        T_gamma=Tg, T_nu_e=Tne, T_nu_x=Tnx, H=H,
    )

    assert np.linalg.norm(gs_nue.C_monopole - gs_nuebar.C_monopole) > 0.0
    assert np.linalg.norm(gs_nue.delta_I - gs_nuebar.delta_I) > 0.0


def test_species_tagged_bridge_preserves_relax_dependence(monkeypatch):
    w0, q, w, I, J, Tg, Tne, Tnx, H = _toy_state()

    monkeypatch.setenv("RABBIT_COLLISION_BRIDGE_RELAX", "0.1")
    gs_lo = apply_species_tagged_bridge(
        species="nue", I=I, J=J, w0=w0, q_nodes=q, q_weights=w,
        T_gamma=Tg, T_nu_e=Tne, T_nu_x=Tnx, H=H,
    )

    monkeypatch.setenv("RABBIT_COLLISION_BRIDGE_RELAX", "1.0")
    gs_hi = apply_species_tagged_bridge(
        species="nue", I=I, J=J, w0=w0, q_nodes=q, q_weights=w,
        T_gamma=Tg, T_nu_e=Tne, T_nu_x=Tnx, H=H,
    )

    norm_lo = float(np.linalg.norm(gs_lo.delta_I))
    norm_hi = float(np.linalg.norm(gs_hi.delta_I))

    assert norm_hi > norm_lo
    assert norm_lo > 0.0


def test_species_tagged_bridge_preserves_weighted_total(monkeypatch):
    w0, q, w, I, J, Tg, Tne, Tnx, H = _toy_state()
    monkeypatch.setenv("RABBIT_COLLISION_BRIDGE_RELAX", "1.0")

    base = apply_gather_scatter_collision(
        I, J, w0, q, w,
        Tg, Tne, H
    )

    gs_nue = apply_species_tagged_bridge(
        species="nue", I=I, J=J, w0=w0, q_nodes=q, q_weights=w,
        T_gamma=Tg, T_nu_e=Tne, T_nu_x=Tnx, H=H,
    )
    gs_nuebar = apply_species_tagged_bridge(
        species="nuebar", I=I, J=J, w0=w0, q_nodes=q, q_weights=w,
        T_gamma=Tg, T_nu_e=Tne, T_nu_x=Tnx, H=H,
    )
    gs_nux = apply_species_tagged_bridge(
        species="nux", I=I, J=J, w0=w0, q_nodes=q, q_weights=w,
        T_gamma=Tg, T_nu_e=Tne, T_nu_x=Tnx, H=H,
    )

    g_nue = BANK_DEGENERACY[Species.NUE]
    g_nuebar = BANK_DEGENERACY[Species.NUEBAR]
    g_nux = BANK_DEGENERACY[Species.NUX]
    gtot = float(g_nue + g_nuebar + g_nux)

    C_mean = (
        g_nue * gs_nue.C_monopole
        + g_nuebar * gs_nuebar.C_monopole
        + g_nux * gs_nux.C_monopole
    ) / gtot

    dI_mean = (
        g_nue * gs_nue.delta_I
        + g_nuebar * gs_nuebar.delta_I
        + g_nux * gs_nux.delta_I
    ) / gtot

    np.testing.assert_allclose(C_mean, base.C_monopole, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(dI_mean, base.delta_I, rtol=0.0, atol=1e-14)


def test_v4_preserves_weighted_total_pointwise(monkeypatch):
    w0, q, w, I, J, Tg, Tne, Tnx, H = _toy_state()
    monkeypatch.setenv("RABBIT_COLLISION_BRIDGE_RELAX", "1.0")

    gs_nue = apply_species_tagged_bridge(
        species="nue", I=I, J=J, w0=w0, q_nodes=q, q_weights=w,
        T_gamma=Tg, T_nu_e=Tne, T_nu_x=Tnx, H=H,
    )
    gs_nuebar = apply_species_tagged_bridge(
        species="nuebar", I=I, J=J, w0=w0, q_nodes=q, q_weights=w,
        T_gamma=Tg, T_nu_e=Tne, T_nu_x=Tnx, H=H,
    )
    gs_nux = apply_species_tagged_bridge(
        species="nux", I=I, J=J, w0=w0, q_nodes=q, q_weights=w,
        T_gamma=Tg, T_nu_e=Tne, T_nu_x=Tnx, H=H,
    )

    C_mean = (gs_nue.C_monopole + gs_nuebar.C_monopole + 4.0 * gs_nux.C_monopole) / 6.0
    dI_mean = (gs_nue.delta_I + gs_nuebar.delta_I + 4.0 * gs_nux.delta_I) / 6.0

    base = apply_gather_scatter_collision(I, J, w0, q, w, Tg, Tne, H)
    np.testing.assert_allclose(C_mean, base.C_monopole, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(dI_mean, base.delta_I, rtol=0.0, atol=1e-14)
