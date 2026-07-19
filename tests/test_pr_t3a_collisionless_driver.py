from __future__ import annotations

import os
import numpy as np
import pytest

os.environ.setdefault("RABBIT_JAX_CACHE_DIR", "/tmp/rabbit_jax_cache")

pytest.importorskip("jax", reason="JAX required for PR-T3A tests")
import jax
import jax.numpy as jnp

from rabbit.jax.driver_typeI_full_boltzmann import (
    JAXFullBoltzmannConfig,
    _build_collision_preflight_update_for_state,
    _get_full_boltzmann_rhs,
    _full_layout,
    _initial_transport_state,
    run_full_boltzmann_jax,
)
from rabbit.jax.q_advection_jax import cached_q_advection_operators, semi_lagrangian_q_advect
from rabbit.jax.solver_jax_rodas5p import materialize_low_rank_jacobian


jax.config.update("jax_enable_x64", True)


def _fd_equilibrium(q_nodes: np.ndarray) -> np.ndarray:
    return 1.0 / (np.exp(np.clip(q_nodes, -500.0, 500.0)) + 1.0)


def _fd_jacobian(fn, x: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    x0 = np.asarray(x, dtype=np.float64).reshape(-1)
    y0 = np.asarray(fn(x0), dtype=np.float64).reshape(-1)
    jac = np.zeros((y0.size, x0.size), dtype=np.float64)
    for i in range(x0.size):
        step = float(eps * max(1.0, abs(x0[i])))
        x_hi = x0.copy()
        x_lo = x0.copy()
        x_hi[i] += step
        x_lo[i] -= step
        jac[:, i] = (np.asarray(fn(x_hi), dtype=np.float64) - np.asarray(fn(x_lo), dtype=np.float64)) / (2.0 * step)
    return jac


def test_q_advection_operators_encode_physical_inflow_boundaries():
    n_q = 6
    q_nodes, _ = np.polynomial.laguerre.laggauss(n_q)
    forward, backward, _centered = cached_q_advection_operators(n_q)
    forward = np.asarray(forward)
    backward = np.asarray(backward)

    h_last = q_nodes[-1] - q_nodes[-2]
    expected_last = -q_nodes[-1] / h_last

    np.testing.assert_allclose(backward[0], 0.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(forward[-1, :-1], 0.0, rtol=0.0, atol=1e-14)
    assert forward[-1, -1] == pytest.approx(expected_last, rel=0.0, abs=1e-14)


def test_pchip_oracle_preserves_bounds_and_tracks_exact_fd_shift():
    q_nodes, _ = np.polynomial.laguerre.laggauss(20)
    f0 = _fd_equilibrium(q_nodes)
    dI = 0.25

    advected = np.asarray(
        semi_lagrangian_q_advect(
            jnp.asarray(f0, dtype=jnp.float64),
            jnp.asarray(q_nodes, dtype=jnp.float64),
            jnp.asarray(dI, dtype=jnp.float64),
        )
    )
    exact = _fd_equilibrium(q_nodes * np.exp(-2.0 * dI))

    np.testing.assert_allclose(
        np.asarray(
            semi_lagrangian_q_advect(
                jnp.asarray(f0, dtype=jnp.float64),
                jnp.asarray(q_nodes, dtype=jnp.float64),
                jnp.asarray(0.0, dtype=jnp.float64),
            )
        ),
        f0,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.min(advected) >= -1e-14
    assert np.max(advected) <= 1.0 + 1e-14
    assert np.all(np.diff(advected) <= 1e-12)
    assert np.linalg.norm(advected - exact) / np.linalg.norm(exact) < 1e-2


def test_full_boltzmann_layout_matches_tier1_compact_scope():
    layout_p1 = _full_layout(12, 20, n_network_species=2, thermo_tier=1)
    layout_p2 = _full_layout(12, 20, n_network_species=9, thermo_tier=1)

    assert layout_p1["n_transport"] == 960
    assert layout_p1["n_total"] == 966
    assert layout_p2["n_total"] == 973
    assert layout_p2["transport_shape"] == (4, 12, 20)


@pytest.mark.production
def test_full_boltzmann_factorized_jacobian_matches_dense_ad():
    rhs_fn, jac_fn, layout, *_rest = _get_full_boltzmann_rhs(
        phase=1,
        correction_level=0,
        collision_mode="collisionless",
        collision_preflight_relaxation=1.0,
        thermo_tier=1,
        N_mu=4,
        N_q=6,
        n_network_species=2,
        n_reactions=12,
        tau_n=878.4,
        eta=6.104e-10,
        N_eff=3.044,
        f_nu=0.40520,
    )

    y0 = np.zeros(layout["n_total"], dtype=np.float64)
    y0[layout["i_Sp"]] = 0.05
    y0[layout["i_S"]] = 0.02
    i0 = layout["i_transport"]
    i1 = i0 + layout["n_transport"]
    y0[i0:i1] = _initial_transport_state(4, 6).reshape(-1)
    y0[layout["i_tg"]] = 10.0
    y0[layout["i_net"] : layout["i_net"] + 2] = np.array([0.13, 0.87], dtype=np.float64)

    y = jnp.asarray(y0)
    factors = jac_fn(jnp.asarray(0.0), y)
    J_fact = np.asarray(materialize_low_rank_jacobian(factors))
    J_dense = np.asarray(jax.jacfwd(rhs_fn, argnums=1)(jnp.asarray(0.0), y))

    assert factors.left_factor.shape == (layout["n_total"], 6)
    assert factors.core_matrix.shape == (6, 13)
    assert factors.right_factor.shape == (13, layout["n_total"])
    np.testing.assert_allclose(J_fact, J_dense, rtol=1e-10, atol=1e-10)


def test_collision_preflight_update_embeds_cleanly_in_full_state_layout():
    _rhs_fn, _jac_fn, layout, mu0, w0, x0, signs, _q_nodes, _energy_weight = _get_full_boltzmann_rhs(
        phase=1,
        correction_level=0,
        collision_mode="collisionless",
        collision_preflight_relaxation=1.0,
        thermo_tier=1,
        N_mu=4,
        N_q=6,
        n_network_species=2,
        n_reactions=12,
        tau_n=878.4,
        eta=6.104e-10,
        N_eff=3.044,
        f_nu=0.40520,
    )

    y0 = np.zeros(layout["n_total"], dtype=np.float64)
    y0[layout["i_Sp"]] = 0.05
    y0[layout["i_S"]] = 0.02
    i0 = layout["i_transport"]
    i1 = i0 + layout["n_transport"]
    y0[i0:i1] = _initial_transport_state(4, 6).reshape(-1)
    y0[layout["i_tg"]] = 10.0
    y0[layout["i_net"] : layout["i_net"] + 2] = np.array([0.13, 0.87], dtype=np.float64)

    lifted = _build_collision_preflight_update_for_state(
        y0,
        layout=layout,
        mu0=mu0,
        w0=w0,
        x0=x0,
        signs=signs,
    )

    assert lifted.state_rhs.shape == (layout["n_total"],)
    np.testing.assert_allclose(lifted.state_rhs[:i0], 0.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(lifted.state_rhs[i1:], 0.0, rtol=0.0, atol=1e-14)

    J_fact = np.asarray(materialize_low_rank_jacobian(lifted.factors))
    assert J_fact.shape == (layout["n_total"], layout["n_total"])
    np.testing.assert_allclose(J_fact[:i0], 0.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(J_fact[i1:], 0.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(J_fact[:, :i0], 0.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(J_fact[:, i1:], 0.0, rtol=0.0, atol=1e-14)
    assert np.linalg.norm(J_fact[i0:i1, i0:i1]) > 0.0


def test_full_boltzmann_collision_preflight_factorized_jacobian_matches_dense_ad():
    rhs_fn, jac_fn, layout, *_rest = _get_full_boltzmann_rhs(
        phase=1,
        correction_level=0,
        collision_mode="spectral_relaxation_preflight",
        collision_preflight_relaxation=1.0,
        thermo_tier=1,
        N_mu=4,
        N_q=6,
        n_network_species=2,
        n_reactions=12,
        tau_n=878.4,
        eta=6.104e-10,
        N_eff=3.044,
        f_nu=0.40520,
    )

    y0 = np.zeros(layout["n_total"], dtype=np.float64)
    y0[layout["i_Sp"]] = 0.05
    y0[layout["i_S"]] = 0.02
    i0 = layout["i_transport"]
    i1 = i0 + layout["n_transport"]
    y0[i0:i1] = _initial_transport_state(4, 6).reshape(-1)
    y0[layout["i_tg"]] = 10.0
    y0[layout["i_net"] : layout["i_net"] + 2] = np.array([0.13, 0.87], dtype=np.float64)

    y = jnp.asarray(y0)
    factors = jac_fn(jnp.asarray(0.0), y)
    J_fact = np.asarray(materialize_low_rank_jacobian(factors))
    J_dense = np.asarray(jax.jacfwd(rhs_fn, argnums=1)(jnp.asarray(0.0), y))

    assert factors.left_factor.shape == (layout["n_total"], 24)
    assert factors.core_matrix.shape == (24, 25)
    assert factors.right_factor.shape == (25, layout["n_total"])
    np.testing.assert_allclose(J_fact, J_dense, rtol=5e-9, atol=5e-9)


def test_full_boltzmann_collision_preflight_smoke_marks_private_contract():
    cfg_full = JAXFullBoltzmannConfig(
        Sigma_H_plus=0.05,
        N_mu=4,
        N_q=6,
        correction_level=0,
        n_reactions=12,
        collision_mode="spectral_relaxation_preflight",
        collision_preflight_relaxation=1.0,
        rtol=1e-6,
        atol=1e-8,
        max_steps=256,
        event_refine_steps=12,
    )

    full = run_full_boltzmann_jax(cfg_full)

    assert full.success
    assert full.metadata["collision_scope_contract"] == "spectral_relaxation_preflight_v1"
    assert full.metadata["collision_mode"] == "spectral_relaxation_preflight"
    assert full.metadata["jacobian_payload_contract"] == "collision_preflight_transport_plus_active_low_rank_v1"
    assert full.metadata["jacobian_low_rank_moment_dim"] == 25
    assert full.metadata["species_identical_approx"] is False
    assert full.metadata["species_identical_max_abs_final"] > 1e-10
    assert full.metadata["phase1_jax_solver_diagnostics"]["linear_solver_policy"] == "custom_hook"
    assert full.metadata["phase2_jax_solver_diagnostics"]["linear_solver_policy"] == "custom_hook"


def test_full_boltzmann_private_tier2_collision_preflight_factorized_jacobian_matches_dense_ad():
    rhs_fn, jac_fn, layout, *_rest = _get_full_boltzmann_rhs(
        phase=1,
        correction_level=0,
        collision_mode="spectral_relaxation_preflight",
        collision_preflight_relaxation=1.0,
        thermo_tier=2,
        N_mu=4,
        N_q=6,
        n_network_species=2,
        n_reactions=12,
        tau_n=878.4,
        eta=6.104e-10,
        N_eff=3.044,
        f_nu=0.40520,
    )

    y0 = np.zeros(layout["n_total"], dtype=np.float64)
    y0[layout["i_Sp"]] = 0.05
    y0[layout["i_S"]] = 0.02
    i0 = layout["i_transport"]
    i1 = i0 + layout["n_transport"]
    y0[i0:i1] = _initial_transport_state(4, 6).reshape(-1)
    y0[layout["i_tg"]] = 10.0
    y0[layout["i_tne"]] = 10.0
    y0[layout["i_tnx"]] = 10.0
    y0[layout["i_net"] : layout["i_net"] + 2] = np.array([0.13, 0.87], dtype=np.float64)

    y = jnp.asarray(y0)
    factors = jac_fn(jnp.asarray(0.0), y)
    J_fact = np.asarray(materialize_low_rank_jacobian(factors))
    J_dense = np.asarray(jax.jacfwd(rhs_fn, argnums=1)(jnp.asarray(0.0), y))

    assert factors.left_factor.shape == (layout["n_total"], 26)
    assert factors.core_matrix.shape == (26, 37)
    assert factors.right_factor.shape == (37, layout["n_total"])
    thermo_rows = np.array([layout["i_tg"], layout["i_tne"], layout["i_tnx"]], dtype=np.int64)
    non_thermo_rows = np.setdiff1d(np.arange(layout["n_total"]), thermo_rows)
    np.testing.assert_allclose(
        J_fact[non_thermo_rows],
        J_dense[non_thermo_rows],
        rtol=5e-9,
        atol=5e-9,
    )
    assert np.max(np.abs(J_fact[thermo_rows] - J_dense[thermo_rows])) < 3.0e-3


def test_full_boltzmann_private_tier2_collision_preflight_smoke():
    cfg_full = JAXFullBoltzmannConfig(
        Sigma_H_plus=0.05,
        N_mu=4,
        N_q=6,
        correction_level=0,
        n_reactions=12,
        collision_mode="spectral_relaxation_preflight",
        collision_preflight_relaxation=1.0,
        thermo_tier=2,
        rtol=1e-6,
        atol=1e-8,
        max_steps=256,
        event_refine_steps=12,
    )

    full = run_full_boltzmann_jax(cfg_full)

    assert full.success
    assert full.metadata["collision_scope_contract"] == "spectral_relaxation_preflight_v1"
    assert full.metadata["thermo_scope_contract"] == "private_tier2_collision_moment_3T_v1"
    assert full.metadata["thermo_tier"] == 2
    assert full.metadata["state_dim_phase1"] == 104
    assert full.metadata["state_dim_phase2"] == 111
    assert full.metadata["T_nu_e_final"] > 0.0
    assert full.metadata["T_nu_x_final"] > 0.0
    assert abs(full.metadata["T_nu_e_final"] - full.metadata["T_nu_x_final"]) > 1e-10
    assert full.metadata["jacobian_payload_contract"] == "collision_preflight_transport_plus_active_plus_3T_low_rank_v1"
    assert full.metadata["jacobian_low_rank_moment_dim"] == 37
    assert full.metadata["jacobian_low_rank_apply_dim"] == 33


def test_full_boltzmann_private_tier2_projected_physical_factorized_jacobian_matches_dense_ad():
    rhs_fn, jac_fn, layout, *_rest = _get_full_boltzmann_rhs(
        phase=1,
        correction_level=0,
        collision_mode="projected_physical_preflight",
        collision_preflight_relaxation=1.0,
        thermo_tier=2,
        N_mu=4,
        N_q=6,
        n_network_species=2,
        n_reactions=12,
        tau_n=878.4,
        eta=6.104e-10,
        N_eff=3.044,
        f_nu=0.40520,
    )

    y0 = np.zeros(layout["n_total"], dtype=np.float64)
    y0[layout["i_Sp"]] = 0.05
    y0[layout["i_S"]] = 0.02
    i0 = layout["i_transport"]
    i1 = i0 + layout["n_transport"]
    y0[i0:i1] = _initial_transport_state(4, 6).reshape(-1)
    y0[layout["i_tg"]] = 10.0
    y0[layout["i_tne"]] = 10.0
    y0[layout["i_tnx"]] = 10.0
    y0[layout["i_net"] : layout["i_net"] + 2] = np.array([0.13, 0.87], dtype=np.float64)

    y = jnp.asarray(y0)
    factors = jac_fn(jnp.asarray(0.0), y)
    J_fact = np.asarray(materialize_low_rank_jacobian(factors))
    J_dense = np.asarray(jax.jacfwd(rhs_fn, argnums=1)(jnp.asarray(0.0), y))

    assert factors.left_factor.shape == (layout["n_total"], 26)
    assert factors.core_matrix.shape == (26, 37)
    assert factors.right_factor.shape == (37, layout["n_total"])
    np.testing.assert_allclose(J_fact, J_dense, rtol=5e-9, atol=5e-9)


def test_full_boltzmann_private_tier2_projected_physical_smoke():
    cfg_full = JAXFullBoltzmannConfig(
        Sigma_H_plus=0.05,
        N_mu=4,
        N_q=6,
        correction_level=0,
        n_reactions=12,
        collision_mode="projected_physical_preflight",
        thermo_tier=2,
        rtol=1e-6,
        atol=1e-8,
        max_steps=256,
        event_refine_steps=12,
    )

    full = run_full_boltzmann_jax(cfg_full)

    assert full.success
    assert full.metadata["collision_scope_contract"] == "projected_physical_preflight_v1"
    assert full.metadata["thermo_scope_contract"] == "private_tier2_collision_moment_3T_v1"
    assert full.metadata["thermo_tier"] == 2
    assert full.metadata["state_dim_phase1"] == 104
    assert full.metadata["state_dim_phase2"] == 111
    assert full.metadata["T_nu_e_final"] > 0.0
    assert full.metadata["T_nu_x_final"] > 0.0
    assert abs(full.metadata["T_nu_e_final"] - full.metadata["T_nu_x_final"]) > 1e-10
    assert full.metadata["jacobian_payload_contract"] == "projected_physical_transport_plus_active_plus_3T_low_rank_v1"
    assert full.metadata["jacobian_low_rank_moment_dim"] == 37
    assert full.metadata["jacobian_low_rank_apply_dim"] == 33


def test_full_boltzmann_private_tier2_direct_kernel_rhs_and_jacobian_smoke():
    rhs_fn, jac_fn, layout, *_rest = _get_full_boltzmann_rhs(
        phase=1,
        correction_level=0,
        collision_mode="direct_kernel_preflight",
        collision_preflight_relaxation=1.0,
        thermo_tier=2,
        N_mu=1,
        N_q=4,
        n_network_species=2,
        n_reactions=12,
        tau_n=878.4,
        eta=6.104e-10,
        N_eff=3.044,
        f_nu=0.40520,
    )

    y0 = np.zeros(layout["n_total"], dtype=np.float64)
    y0[layout["i_Sp"]] = 0.05
    y0[layout["i_S"]] = 0.02
    i0 = layout["i_transport"]
    i1 = i0 + layout["n_transport"]
    y0[i0:i1] = _initial_transport_state(1, 4).reshape(-1)
    y0[layout["i_tg"]] = 10.0
    y0[layout["i_tne"]] = 10.0
    y0[layout["i_tnx"]] = 10.0
    y0[layout["i_net"] : layout["i_net"] + 2] = np.array([0.13, 0.87], dtype=np.float64)

    y = jnp.asarray(y0)
    dy = np.asarray(rhs_fn(jnp.asarray(0.0), y), dtype=np.float64)
    factors = jac_fn(jnp.asarray(0.0), y)

    assert dy.shape == (layout["n_total"],)
    assert np.isfinite(dy).all()
    assert factors.left_factor.shape == (layout["n_total"], 23)
    assert factors.core_matrix.shape == (23, 17)
    assert factors.right_factor.shape == (17, layout["n_total"])


def test_full_boltzmann_private_tier2_jax_kernel_rhs_and_jacobian_smoke():
    """PR-T3B-PF Phase 2: wire pure-JAX nu-e + pair kernels through the
    bank-core dispatch.  No host callback; all ops are JIT-compatible."""
    rhs_fn, jac_fn, layout, *_rest = _get_full_boltzmann_rhs(
        phase=1,
        correction_level=0,
        collision_mode="jax_kernel_preflight",
        collision_preflight_relaxation=1.0,
        thermo_tier=2,
        N_mu=4,
        N_q=6,
        n_network_species=2,
        n_reactions=12,
        tau_n=878.4,
        eta=6.104e-10,
        N_eff=3.044,
        f_nu=0.40520,
    )

    y0 = np.zeros(layout["n_total"], dtype=np.float64)
    y0[layout["i_Sp"]] = 0.05
    y0[layout["i_S"]] = 0.02
    i0 = layout["i_transport"]
    i1 = i0 + layout["n_transport"]
    y0[i0:i1] = _initial_transport_state(4, 6).reshape(-1)
    y0[layout["i_tg"]] = 10.0
    y0[layout["i_tne"]] = 10.0
    y0[layout["i_tnx"]] = 10.0
    y0[layout["i_net"] : layout["i_net"] + 2] = np.array([0.13, 0.87], dtype=np.float64)

    y = jnp.asarray(y0)
    dy = np.asarray(rhs_fn(jnp.asarray(0.0), y), dtype=np.float64)
    factors = jac_fn(jnp.asarray(0.0), y)

    assert dy.shape == (layout["n_total"],)
    assert np.isfinite(dy).all()
    # Bank-core path shares the projected_physical / spectral_relaxation
    # Jacobian shape at tier-2:
    #   moment dim = 1 + 6*N_q = 37 at N_q=6
    #   apply dim  = 6 + n_species + 3*N_q = 26 at N_q=6, n_species=2
    assert factors.left_factor.shape == (layout["n_total"], 26)
    assert factors.core_matrix.shape == (26, 37)
    assert factors.right_factor.shape == (37, layout["n_total"])


def test_full_boltzmann_private_tier2_jax_kernel_smoke():
    """End-to-end Rodas5P solve through the new pure-JAX
    ``jax_kernel_preflight`` mode at bounded shear; mirrors the
    spectral_relaxation / projected_physical end-to-end smokes."""
    cfg_full = JAXFullBoltzmannConfig(
        Sigma_H_plus=0.05,
        N_mu=4,
        N_q=6,
        correction_level=0,
        n_reactions=12,
        collision_mode="jax_kernel_preflight",
        thermo_tier=2,
        rtol=1e-6,
        atol=1e-8,
        max_steps=256,
        event_refine_steps=12,
    )

    full = run_full_boltzmann_jax(cfg_full)

    assert full.success
    assert full.metadata["collision_scope_contract"] == "jax_kernel_preflight_v1"
    assert full.metadata["thermo_scope_contract"] == "private_tier2_collision_moment_3T_v1"
    assert full.metadata["thermo_tier"] == 2
    assert full.metadata["state_dim_phase1"] == 104
    assert full.metadata["state_dim_phase2"] == 111
    assert full.metadata["T_nu_e_final"] > 0.0
    assert full.metadata["T_nu_x_final"] > 0.0
    # nu-e + pair coupling drives a heating asymmetry between nue and nux
    # at incomplete decoupling; sanity-check it is non-zero on the
    # bounded smoke trajectory.
    assert abs(full.metadata["T_nu_e_final"] - full.metadata["T_nu_x_final"]) > 1e-10
    assert full.metadata["jacobian_payload_contract"] == "jax_kernel_transport_plus_active_plus_3T_low_rank_v1"
    # moment dim = 1 + 6*N_q = 37 at N_q=6
    # apply dim  = 6 + N_SPECIES + 3*N_q = 6 + 9 + 18 = 33 at phase 2
    assert full.metadata["jacobian_low_rank_moment_dim"] == 37
    assert full.metadata["jacobian_low_rank_apply_dim"] == 33


def test_full_boltzmann_private_tier2_jax_kernel_flrw_neff_baseline():
    """FLRW (Σ=0) end-to-end on ``jax_kernel_preflight`` — measure
    ``N_eff`` and lock the baseline.

    Phase prompt PR-T3B target: ``|N_eff - 3.044| < 0.01`` against
    Mangano 2005.  Current preflight measurement at the bounded
    ``(N_mu=4, N_q=6)`` smoke grid: ``N_eff ≈ 2.993`` with a
    ``~5%`` gap to 3.044.  This is the joint contribution of:

    * coarse grid (under-resolves the y3/y2 collision integrals),
    * PCHIP-vs-scipy-cubic interpolation gap on the pair-process
      ``f_ν̄(y_2)`` evaluation,
    * the placeholder ``(y_1 y_2)^2 + (y_3 y_4)^2`` matrix-element
      coefficient inherited from the SciPy ν-e port rather than a
      Hannestad-Madsen / Mangano-calibrated normalisation.

    The lock is documented as a *baseline* — it ensures the gap
    does not silently widen over future preflight refactors.
    Closing the gap is an explicit PR-T3B canonical follow-up.
    """
    cfg_full = JAXFullBoltzmannConfig(
        Sigma_H_plus=0.0,
        N_mu=4,
        N_q=6,
        correction_level=0,
        n_reactions=12,
        collision_mode="jax_kernel_preflight",
        thermo_tier=2,
        rtol=1e-6,
        atol=1e-8,
        max_steps=256,
        event_refine_steps=12,
    )

    full = run_full_boltzmann_jax(cfg_full)
    assert full.success

    N_eff = float(full.metadata["N_eff_measured"])
    # Baseline measurement at the bounded preflight grid.  Locked at
    # +/- 0.02 around the measured 2.993 to detect regressions while
    # tolerating small numerical drift; tighten when the grid /
    # matrix-element / interpolant gaps close.
    assert 2.97 < N_eff < 3.02, (
        f"FLRW N_eff baseline drifted: {N_eff:.6f} "
        f"(measured 2.993 at preflight grid; canonical Mangano 2005 = 3.044)"
    )
    # Document the gap to Mangano 2005 in the assertion message; the
    # phase-prompt target |N_eff - 3.044| < 0.01 is NOT met by this
    # preflight, by construction.
    gap = abs(N_eff - 3.044)
    assert gap < 0.10, (
        f"FLRW N_eff gap to Mangano 2005 widened beyond preflight tolerance: "
        f"|N_eff - 3.044| = {gap:.4f} (measured baseline ~0.051)"
    )

    # T_nu_e == T_nu_x at strict FLRW (LRS, no anisotropic transport
    # asymmetry); the heating asymmetry test
    # (test_full_boltzmann_private_tier2_jax_kernel_smoke) covers
    # the Σ != 0 case where the asymmetry is non-zero.
    T_nu_e = float(full.metadata["T_nu_e_final"])
    T_nu_x = float(full.metadata["T_nu_x_final"])
    assert abs(T_nu_e - T_nu_x) < 1e-12, (
        f"FLRW LRS symmetry broken: T_nu_e={T_nu_e}, T_nu_x={T_nu_x}"
    )


def test_full_boltzmann_private_tier2_jax_kernel_relaxation_sign_convention():
    """The bank-core preflight currently shares the SciPy ν-e
    convention of treating both the neutrino and electron grids on
    the same dimensionless ``y`` axis (i.e., T_e = T_ν inside the
    kernel).  The full PR-T3B physics check ``dQ_α > 0 when plasma
    hotter than ν`` therefore cannot be exercised on this preflight
    surface — see ``docs/audit/PR-T3B_jax_kernel_runtime.md``.

    What the preflight DOES verify is the *relaxation* sign
    convention: when ``f_α`` is perturbed above its FD baseline
    (over-populated), the collision integral returns ``C_α < 0``
    on average (loss term, restoring towards FD).  Conversely,
    under-populated yields ``C_α > 0`` on average.  This is a
    detailed-balance corollary that holds without any T-rescaling.

    The energy moment of the bank-core output is the cleanest scalar
    diagnostic: ``Σ w_i exp(y_i) y_i^3 C_i`` is positive when the
    distribution is gaining energy and negative when losing.
    """
    from rabbit.jax.driver_typeI_full_boltzmann import (
        _collision_jax_kernel_bank_core_jax,
        _compute_energy_exchange_rate_jax,
    )
    from rabbit.jax.collisions_jax import laguerre_grid

    N_q = 8
    q_nodes_np, q_weights_np = laguerre_grid(N_q)
    q_nodes = jnp.asarray(q_nodes_np)
    q_weights = jnp.asarray(q_weights_np)

    T_common = 4.0  # T_γ = T_ν on the preflight surface
    f_FD_np = 1.0 / (np.exp(np.minimum(q_nodes_np, 500.0)) + 1.0)
    f_high_np = np.clip(f_FD_np * 1.30, 0.0, 1.0)  # over-populated
    f_low_np = np.clip(f_FD_np * 0.70, 0.0, 1.0)   # under-populated

    H_inv_sec = jnp.asarray(1.0e-15 * 1.519267447e21)

    def _energy_exchange(f_state_np):
        bank_state = jnp.concatenate([
            jnp.asarray(f_state_np),
            jnp.asarray(f_state_np),
            jnp.asarray(f_state_np),
        ])
        bank_rhs = _collision_jax_kernel_bank_core_jax(
            bank_state,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=jnp.asarray(T_common),
            T_nu_e=jnp.asarray(T_common),
            T_nu_x=jnp.asarray(T_common),
            H_inv_sec=H_inv_sec,
        )
        # Take the nu_e block of the bank RHS and its energy moment.
        C_nue = bank_rhs[:N_q]
        return float(np.asarray(_compute_energy_exchange_rate_jax(
            C_nue, q_nodes, q_weights, jnp.asarray(T_common)
        )))

    # ``_compute_energy_exchange_rate_jax`` returns
    #   -T_nu^4 / (2 pi^2) * Σ w_i exp(y_i) y_i^3 C_i.
    # When C > 0 (gain) the inner sum is positive and the function
    # returns NEGATIVE; when C < 0 (loss) the function returns
    # POSITIVE.  The sign-flip is by design so the dQ_alpha
    # downstream (in ``_collision_bank_energy_exchange_jax``) carries
    # the standard ``positive = ν gains energy`` convention via an
    # outer minus sign.
    moment_FD = _energy_exchange(f_FD_np)
    moment_high = _energy_exchange(f_high_np)
    moment_low = _energy_exchange(f_low_np)

    # FD baseline is detailed-balance zero (down to the y_4 PCHIP
    # noise floor measured in earlier preflights).
    assert abs(moment_FD) < 1.0e-10, (
        f"FD baseline moment not zero: {moment_FD:.3e}"
    )
    # Over-populated -> kernel produces loss (C < 0) -> moment positive
    # via the function's -1 sign-flip convention.
    assert moment_high > 0.0, (
        f"Over-populated relaxation sign violated: moment {moment_high:.3e}"
    )
    # Under-populated -> kernel produces gain (C > 0) -> moment negative.
    assert moment_low < 0.0, (
        f"Under-populated relaxation sign violated: moment {moment_low:.3e}"
    )


def test_full_boltzmann_private_tier2_jax_kernel_flrw_detailed_balance():
    """At FLRW (Sigma=0) with f_FD initial transport and T_nu = T_gamma,
    the pure-JAX bank-core's collision contribution should vanish to
    numerical precision because (a) all neutrino monopoles equal f_FD,
    (b) the electron sector inside the operators is also at the same
    temperature, so detailed balance kicks in algebraically (paper
    eq 84 statistical factor)."""
    rhs_fn, _jac_fn, layout, *_rest = _get_full_boltzmann_rhs(
        phase=1,
        correction_level=0,
        collision_mode="jax_kernel_preflight",
        collision_preflight_relaxation=1.0,
        thermo_tier=2,
        N_mu=4,
        N_q=6,
        n_network_species=2,
        n_reactions=12,
        tau_n=878.4,
        eta=6.104e-10,
        N_eff=3.044,
        f_nu=0.40520,
    )

    # Use a baseline collisionless RHS to subtract the transport / weak /
    # network terms; the residual is the collision contribution alone.
    base_rhs_fn, *_ = _get_full_boltzmann_rhs(
        phase=1,
        correction_level=0,
        collision_mode="collisionless",
        collision_preflight_relaxation=1.0,
        thermo_tier=2,
        N_mu=4,
        N_q=6,
        n_network_species=2,
        n_reactions=12,
        tau_n=878.4,
        eta=6.104e-10,
        N_eff=3.044,
        f_nu=0.40520,
    )

    y0 = np.zeros(layout["n_total"], dtype=np.float64)
    y0[layout["i_Sp"]] = 0.0  # FLRW
    y0[layout["i_S"]] = 0.0
    i0 = layout["i_transport"]
    i1 = i0 + layout["n_transport"]
    y0[i0:i1] = _initial_transport_state(4, 6).reshape(-1)  # all f_FD
    y0[layout["i_tg"]] = 10.0
    y0[layout["i_tne"]] = 10.0  # T_nu_e == T_gamma
    y0[layout["i_tnx"]] = 10.0  # T_nu_x == T_gamma
    y0[layout["i_net"] : layout["i_net"] + 2] = np.array([0.13, 0.87], dtype=np.float64)

    y = jnp.asarray(y0)
    dy_full = np.asarray(rhs_fn(jnp.asarray(0.0), y), dtype=np.float64)
    dy_base = np.asarray(base_rhs_fn(jnp.asarray(0.0), y), dtype=np.float64)
    transport_offset = int(layout["i_transport"])
    transport_dim = int(layout["n_transport"])
    db_residual = dy_full - dy_base
    transport_db = db_residual[transport_offset : transport_offset + transport_dim]

    # Detailed balance on the transport rays at T_nu = T_gamma at the
    # matched bank-core grid configuration.  The bank-core JAX kernel
    # operators each produce ``|C[f_FD]| <= 1e-30`` absolute (locked in
    # ``test_pr_t3b_jax_operator_parity.py``).  After the bank-core
    # divides by ``H_MeV ~ 1e-20`` and scatters back to all rays the
    # natural noise floor scales to ~``1e-14`` per ray.  The bound is
    # locked at ``1e-10`` to flag any algorithmic regression while
    # reflecting the genuine kernel→df/dN amplification.
    measured = np.max(np.abs(transport_db))
    assert measured < 1e-10, (
        f"jax_kernel_preflight detailed balance failed at FLRW: "
        f"max|Delta_transport| = {measured:.3e}"
    )
