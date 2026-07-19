"""Runtime diagonal nu-nu spectral bank coupling for tier-3 preflight."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from rabbit.jax.collisions_jax import laguerre_grid
from rabbit.jax.nu_nu_scattering_jax import (
    nu_nu_diagonal_collision_jax,
    nu_nu_diagonal_collision_tensorized_jax,
)
from rabbit.jax.driver_typeI_full_boltzmann import (
    _MEV_TO_S,
    _ap_energy_transfer_scale_for_mode,
    _collision_bank_energy_exchange_jax,
    _collision_shape_energy_moment_jax,
    _collision_shape_number_moment_jax,
    _collision_nu_nu_spectral_bank_core_jax,
)
from rabbit.jax.nudec_coupled_jax import (
    hubble_3T_jax,
    nu_nu_temperature_equilibration_sources_jax,
)


jax.config.update("jax_enable_x64", True)


def _fd(q: np.ndarray) -> np.ndarray:
    return 1.0 / (np.exp(np.minimum(q, 500.0)) + 1.0)


@pytest.mark.production
@pytest.mark.parametrize("N_q", [6, 12])
def test_tensorized_nu_nu_kernel_matches_legacy_cubic_path(N_q: int) -> None:
    q_nodes, q_weights = laguerre_grid(N_q)
    q = jnp.asarray(q_nodes)
    w = jnp.asarray(q_weights)
    f_eq = _fd(q_nodes)
    f_alpha = jnp.asarray(
        np.clip(f_eq + 0.015 * f_eq * (1.0 - f_eq) * (q_nodes / np.mean(q_nodes) - 0.5), 0.0, 1.0)
    )
    f_beta = jnp.asarray(
        np.clip(f_eq - 0.010 * f_eq * (1.0 - f_eq) * (q_nodes / np.mean(q_nodes) - 1.2), 0.0, 1.0)
    )
    legacy = nu_nu_diagonal_collision_jax(
        f_alpha,
        f_beta,
        q,
        jnp.asarray(1.7),
        y3_nodes=q,
        y3_weights=w,
        y2_nodes=q,
        y2_weights=w,
    )
    tensorized = nu_nu_diagonal_collision_tensorized_jax(
        f_alpha,
        f_beta,
        q,
        jnp.asarray(1.7),
    )
    assert jnp.allclose(tensorized, legacy, rtol=1.0e-10, atol=1.0e-32)


@pytest.mark.production
def test_tensorized_nu_nu_kernel_is_jit_and_grad_safe() -> None:
    q_nodes, _q_weights = laguerre_grid(6)
    q = jnp.asarray(q_nodes)
    f_eq = jnp.asarray(_fd(q_nodes))

    @jax.jit
    def objective(theta):
        f_alpha = jnp.clip(f_eq + theta[0] * f_eq * (1.0 - f_eq), 0.0, 1.0)
        f_beta = jnp.clip(f_eq + theta[1] * f_eq * (1.0 - f_eq) * q / jnp.mean(q), 0.0, 1.0)
        return jnp.sum(
            nu_nu_diagonal_collision_tensorized_jax(
                f_alpha,
                f_beta,
                q,
                jnp.asarray(1.7),
            )
        )

    theta = jnp.asarray([0.02, -0.015], dtype=jnp.float64)
    value = objective(theta)
    grad = jax.grad(objective)(theta)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(grad)))


def _spectral_source(T_nu_e: float, T_nu_x: float, N_q: int = 6):
    q_nodes, q_weights = laguerre_grid(N_q)
    H_MeV = hubble_3T_jax(
        jnp.asarray(2.0),
        jnp.asarray(T_nu_e),
        jnp.asarray(T_nu_x),
        jnp.asarray(0.0),
    )
    bank = np.concatenate([_fd(q_nodes), _fd(q_nodes), _fd(q_nodes)])
    C = _collision_nu_nu_spectral_bank_core_jax(
        jnp.asarray(bank),
        q_nodes=jnp.asarray(q_nodes),
        q_weights=jnp.asarray(q_weights),
        T_nu_e=jnp.asarray(T_nu_e),
        T_nu_x=jnp.asarray(T_nu_x),
        H_inv_sec=H_MeV * _MEV_TO_S,
    )
    dQ_e, dQ_x = _collision_bank_energy_exchange_jax(
        C,
        q_nodes=jnp.asarray(q_nodes),
        q_weights=jnp.asarray(q_weights),
        T_nu_e=jnp.asarray(T_nu_e),
        T_nu_x=jnp.asarray(T_nu_x),
    )
    return np.asarray(C), float(dQ_e), float(dQ_x), H_MeV


@pytest.mark.production
def test_nu_nu_spectral_bank_zero_for_equal_fd_temperature() -> None:
    C, dQ_e, dQ_x, _H = _spectral_source(1.7, 1.7)
    assert np.max(np.abs(C)) == pytest.approx(0.0, abs=1e-30)
    assert dQ_e == pytest.approx(0.0, abs=1e-30)
    assert dQ_x == pytest.approx(0.0, abs=1e-30)


@pytest.mark.production
@pytest.mark.parametrize(
    ("T_nu_e", "T_nu_x", "electron_sign", "heavy_sign"),
    [
        (1.85, 1.55, -1.0, 1.0),
        (1.55, 1.85, 1.0, -1.0),
    ],
)
def test_nu_nu_spectral_bank_energy_conserved_and_moment_projected(
    T_nu_e: float,
    T_nu_x: float,
    electron_sign: float,
    heavy_sign: float,
) -> None:
    C, dQ_e, dQ_x, H_MeV = _spectral_source(T_nu_e, T_nu_x)
    target_e, target_x = nu_nu_temperature_equilibration_sources_jax(
        jnp.asarray(T_nu_e),
        jnp.asarray(T_nu_x),
        H_MeV,
    )

    assert np.max(np.abs(C)) > 0.0
    assert electron_sign * dQ_e > 0.0
    assert heavy_sign * dQ_x > 0.0
    assert dQ_e + dQ_x == pytest.approx(0.0, abs=1e-10)
    assert dQ_e == pytest.approx(float(target_e), rel=1e-12, abs=1e-12)
    assert dQ_x == pytest.approx(float(target_x), rel=1e-12, abs=1e-12)


@pytest.mark.production
def test_nu_nu_spectral_bank_self_thermalizes_shape_without_changing_moments() -> None:
    q_nodes, q_weights = laguerre_grid(6)
    f_eq = _fd(q_nodes)
    distortion = 0.04 * f_eq * (1.0 - f_eq) * (q_nodes / np.mean(q_nodes) - 1.0)
    f_nue = np.clip(f_eq + distortion, 0.0, 1.0)
    bank = np.concatenate([f_nue, f_eq, f_eq])
    H_MeV = hubble_3T_jax(
        jnp.asarray(2.0),
        jnp.asarray(1.7),
        jnp.asarray(1.7),
        jnp.asarray(0.0),
    )
    C = _collision_nu_nu_spectral_bank_core_jax(
        jnp.asarray(bank),
        q_nodes=jnp.asarray(q_nodes),
        q_weights=jnp.asarray(q_weights),
        T_nu_e=jnp.asarray(1.7),
        T_nu_x=jnp.asarray(1.7),
        H_inv_sec=H_MeV * _MEV_TO_S,
    )
    C_np = np.asarray(C)
    C_nue = C[: len(q_nodes)]
    dQ_e, dQ_x = _collision_bank_energy_exchange_jax(
        C,
        q_nodes=jnp.asarray(q_nodes),
        q_weights=jnp.asarray(q_weights),
        T_nu_e=jnp.asarray(1.7),
        T_nu_x=jnp.asarray(1.7),
    )

    assert np.max(np.abs(C_np[: len(q_nodes)])) > 0.0
    assert np.max(np.abs(C_np[len(q_nodes) :])) == pytest.approx(0.0, abs=1e-24)
    assert float(_collision_shape_number_moment_jax(C_nue, jnp.asarray(q_nodes), jnp.asarray(q_weights))) == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert float(_collision_shape_energy_moment_jax(C_nue, jnp.asarray(q_nodes), jnp.asarray(q_weights))) == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert float(dQ_e) == pytest.approx(0.0, abs=1e-10)
    assert float(dQ_x) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.production
def test_nu_nu_spectral_bank_self_thermalizes_common_shape_distortion() -> None:
    q_nodes, q_weights = laguerre_grid(6)
    f_eq = _fd(q_nodes)
    distortion = 0.03 * f_eq * (1.0 - f_eq) * (q_nodes / np.mean(q_nodes) - 1.0)
    f_distorted = np.clip(f_eq + distortion, 0.0, 1.0)
    bank = np.concatenate([f_distorted, f_distorted, f_distorted])
    H_MeV = hubble_3T_jax(
        jnp.asarray(2.0),
        jnp.asarray(1.7),
        jnp.asarray(1.7),
        jnp.asarray(0.0),
    )
    C = _collision_nu_nu_spectral_bank_core_jax(
        jnp.asarray(bank),
        q_nodes=jnp.asarray(q_nodes),
        q_weights=jnp.asarray(q_weights),
        T_nu_e=jnp.asarray(1.7),
        T_nu_x=jnp.asarray(1.7),
        H_inv_sec=H_MeV * _MEV_TO_S,
    )
    C_np = np.asarray(C)
    n_q = len(q_nodes)

    assert np.max(np.abs(C_np)) > 0.0
    assert np.max(np.abs(C_np[:n_q] - C_np[n_q : 2 * n_q])) == pytest.approx(0.0, abs=1e-24)
    assert np.max(np.abs(C_np[:n_q] - C_np[2 * n_q :])) == pytest.approx(0.0, abs=1e-24)
    for block in (C[:n_q], C[n_q : 2 * n_q], C[2 * n_q :]):
        assert float(
            _collision_shape_number_moment_jax(
                block,
                jnp.asarray(q_nodes),
                jnp.asarray(q_weights),
            )
        ) == pytest.approx(0.0, abs=1e-12)
        assert float(
            _collision_shape_energy_moment_jax(
                block,
                jnp.asarray(q_nodes),
                jnp.asarray(q_weights),
            )
        ) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.production
def test_accuracy_collision_mode_declares_calibrated_energy_transfer_scale() -> None:
    from rabbit.jax.driver_typeI_full_boltzmann import JAXFullBoltzmannConfig

    cfg = JAXFullBoltzmannConfig(
        collision_mode="ap_unified_nu_nu_spectral_accuracy_preflight",
        thermo_tier=2,
        N_mu=4,
        N_q=6,
    )

    assert cfg.collision_mode == "ap_unified_nu_nu_spectral_accuracy_preflight"
    assert _ap_energy_transfer_scale_for_mode(cfg.collision_mode) == pytest.approx(4.0 / 3.0)
    assert _ap_energy_transfer_scale_for_mode("ap_unified_nu_nu_spectral_preflight") == pytest.approx(1.0)


@pytest.mark.slow
def test_ap_unified_nu_nu_spectral_preflight_full_ode_smoke() -> None:
    from rabbit.jax.driver_typeI_full_boltzmann import (
        JAXFullBoltzmannConfig,
        run_full_boltzmann_jax,
    )

    common = dict(
        Sigma_H_plus=0.0,
        N_mu=4,
        N_q=6,
        correction_level=0,
        n_reactions=12,
        thermo_tier=2,
        rtol=1e-6,
        atol=1e-8,
        max_steps=512,
        event_refine_steps=12,
    )
    baseline = run_full_boltzmann_jax(
        JAXFullBoltzmannConfig(collision_mode="ap_unified_preflight", **common)
    )
    spectral = run_full_boltzmann_jax(
        JAXFullBoltzmannConfig(
            collision_mode="ap_unified_nu_nu_spectral_preflight",
            **common,
        )
    )

    assert baseline.success
    assert spectral.success
    assert spectral.metadata["collision_scope_contract"] == (
        "ap_unified_plus_energy_projected_nu_nu_spectral_bank_preflight_v1"
    )
    assert spectral.metadata["nu_nu_spectral_bank_enabled"] is True
    assert spectral.metadata["nu_nu_spectral_impl"] == "tensorized_static_kernel_v1"
    assert spectral.metadata["nu_nu_spectral_self_thermalization_contract"] == (
        "number_energy_neutral_leading_order_shape_damping_v1"
    )
    assert spectral.metadata["nu_nu_equilibration_enabled"] is False

    base_split = abs(baseline.metadata["T_nu_e_final"] - baseline.metadata["T_nu_x_final"])
    spectral_split = abs(spectral.metadata["T_nu_e_final"] - spectral.metadata["T_nu_x_final"])
    assert spectral_split < 0.75 * base_split
    assert spectral.metadata["N_eff_measured"] > baseline.metadata["N_eff_measured"]
    assert spectral.metadata["N_eff_measured"] < 3.044
    assert 0.23 < spectral.Yp < 0.25


@pytest.mark.slow
def test_ap_unified_accuracy_preflight_full_ode_closes_flrw_neff_gap() -> None:
    from rabbit.jax.driver_typeI_full_boltzmann import (
        JAXFullBoltzmannConfig,
        run_full_boltzmann_jax,
    )

    common = dict(
        Sigma_H_plus=0.0,
        N_mu=4,
        N_q=6,
        correction_level=0,
        n_reactions=12,
        thermo_tier=2,
        rtol=1e-6,
        atol=1e-8,
        max_steps=512,
        event_refine_steps=12,
    )
    spectral = run_full_boltzmann_jax(
        JAXFullBoltzmannConfig(
            collision_mode="ap_unified_nu_nu_spectral_preflight",
            **common,
        )
    )
    accuracy = run_full_boltzmann_jax(
        JAXFullBoltzmannConfig(
            collision_mode="ap_unified_nu_nu_spectral_accuracy_preflight",
            **common,
        )
    )

    assert spectral.success
    assert accuracy.success
    assert accuracy.metadata["collision_scope_contract"] == (
        "ap_unified_plus_calibrated_energy_transfer_and_projected_nu_nu_spectral_bank_preflight_v1"
    )
    assert accuracy.metadata["nu_nu_spectral_bank_enabled"] is True
    assert accuracy.metadata["nu_nu_spectral_impl"] == "tensorized_static_kernel_v1"
    assert accuracy.metadata["ap_accuracy_candidate_enabled"] is True
    assert accuracy.metadata["ap_accuracy_candidate_contract"] == (
        "calibrated_no_qke_energy_transfer_scale_plus_spectral_nu_nu_v1"
    )
    assert accuracy.metadata["ap_accuracy_energy_transfer_scale"] == pytest.approx(4.0 / 3.0)
    assert accuracy.metadata["mangano_energy_transfer_C_rate_effective"] == pytest.approx(280.0)
    assert accuracy.metadata["N_eff_measured"] > spectral.metadata["N_eff_measured"]
    assert abs(accuracy.metadata["N_eff_measured"] - 3.0446) < 5.0e-3
    assert 0.23 < accuracy.Yp < 0.25
