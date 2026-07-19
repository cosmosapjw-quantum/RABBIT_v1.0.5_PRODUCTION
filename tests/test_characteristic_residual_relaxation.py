import numpy as np

from rabbit.collisions.kernels import A_TOTAL_NUE, A_TOTAL_NUX
from rabbit.drivers import full_coupled_typeI as mod
from rabbit.transport.characteristic_residual import (
    apply_species_residual_relaxation,
    distortion_rate_factors,
    spectral_rate_factors,
    temperature_mismatch_factor,
)


def _toy_state():
    mu0, w0, X0, signs = mod.setup_ray_grid(12)
    I = np.linspace(-0.02, 0.02, 12, dtype=np.float64)
    J = 1.0 + 0.1 * np.linspace(-1.0, 1.0, 12, dtype=np.float64)
    return w0, I, J


def test_residual_relaxation_quiet_at_isotropy():
    w0, _, _ = _toy_state()
    out = apply_species_residual_relaxation(
        species="nue",
        I=np.zeros_like(w0),
        J=np.ones_like(w0),
        w0=w0,
        T_gamma=0.08,
        T_nu_e=0.079,
        T_nu_x=0.078,
        H_inv_sec=1.0,
    )
    np.testing.assert_allclose(out.delta_I, 0.0, rtol=0.0, atol=1e-18)
    np.testing.assert_allclose(out.delta_J, 0.0, rtol=0.0, atol=1e-18)


def test_residual_relaxation_preserves_weighted_I_mean():
    w0, I, J = _toy_state()
    out = apply_species_residual_relaxation(
        species="nue",
        I=I,
        J=J,
        w0=w0,
        T_gamma=0.08,
        T_nu_e=0.079,
        T_nu_x=0.078,
        H_inv_sec=1.0,
    )
    assert abs(np.sum(w0 * out.delta_I)) < 1.0e-15


def test_residual_relaxation_species_hierarchy():
    w0, I, J = _toy_state()
    nue = apply_species_residual_relaxation(
        species="nue",
        I=I,
        J=J,
        w0=w0,
        T_gamma=0.08,
        T_nu_e=0.079,
        T_nu_x=0.078,
        H_inv_sec=1.0,
    )
    nux = apply_species_residual_relaxation(
        species="nux",
        I=I,
        J=J,
        w0=w0,
        T_gamma=0.08,
        T_nu_e=0.079,
        T_nu_x=0.078,
        H_inv_sec=1.0,
    )
    assert nue.gamma_over_H > nux.gamma_over_H
    assert nue.gamma_over_H / max(nux.gamma_over_H, 1.0e-30) > 0.9 * (A_TOTAL_NUE / A_TOTAL_NUX)


def test_residual_relaxation_damps_energy_weighted_jacobian():
    w0, I, J = _toy_state()
    out = apply_species_residual_relaxation(
        species="nue",
        I=I,
        J=J,
        w0=w0,
        T_gamma=0.08,
        T_nu_e=0.079,
        T_nu_x=0.078,
        H_inv_sec=1.0,
    )

    I_mean = np.sum(w0 * I) / np.sum(w0)
    K0 = J * np.exp(-8.0 * (I - I_mean))

    dt = 1.0e-3
    I1 = I + dt * out.delta_I
    J1 = J + dt * out.delta_J
    I1_mean = np.sum(w0 * I1) / np.sum(w0)
    K1 = J1 * np.exp(-8.0 * (I1 - I1_mean))

    err0 = np.linalg.norm(K0 - 1.0)
    err1 = np.linalg.norm(K1 - 1.0)
    assert err1 < err0


def test_spectral_rate_factors_track_hardening():
    q = np.linspace(0.2, 12.0, 41, dtype=np.float64)
    w = np.ones_like(q)
    f_soft = 1.0 / (np.exp(q / 0.9) + 1.0)
    f_hard = 1.0 / (np.exp(q / 1.1) + 1.0)

    soft0, soft2 = spectral_rate_factors(q, w, f_soft)
    hard0, hard2 = spectral_rate_factors(q, w, f_hard)

    assert hard0 > soft0
    assert hard2 > soft2


def test_spectral_rate_factors_track_blocking_suppression():
    q = np.linspace(0.2, 12.0, 41, dtype=np.float64)
    w = np.ones_like(q)
    f_ref = 1.0 / (np.exp(q) + 1.0)
    f_dilute = 0.6 * f_ref

    ref0, ref2 = spectral_rate_factors(q, w, f_ref)
    dilute0, dilute2 = spectral_rate_factors(q, w, f_dilute)

    assert dilute0 < ref0
    assert dilute2 < ref2


def test_temperature_mismatch_factor_tracks_thermal_split():
    near, near_log = temperature_mismatch_factor(0.08, 0.079)
    far, far_log = temperature_mismatch_factor(0.08, 0.06)

    assert far > near
    assert far_log > near_log


def test_distortion_rate_factors_track_background_deviation():
    q = np.linspace(0.2, 12.0, 41, dtype=np.float64)
    w = np.ones_like(q)
    f_ref = 1.0 / (np.exp(q) + 1.0)
    f_dist = np.clip(f_ref * (1.0 + 0.25 * np.tanh((q - 3.0) / 1.5)), 0.0, 1.0)

    ref0, ref2, ref_rms = distortion_rate_factors(q, w, f_ref)
    dist0, dist2, dist_rms = distortion_rate_factors(q, w, f_dist)

    assert dist0 > ref0
    assert dist2 > ref2
    assert dist_rms > ref_rms


def test_residual_relaxation_uses_spectrum_aware_rate_factors():
    w0, I, J = _toy_state()
    q = np.linspace(0.2, 12.0, 41, dtype=np.float64)
    wq = np.ones_like(q)
    f_soft = 1.0 / (np.exp(q / 0.9) + 1.0)
    f_hard = 1.0 / (np.exp(q / 1.1) + 1.0)

    soft = apply_species_residual_relaxation(
        species="nue",
        I=I,
        J=J,
        w0=w0,
        T_gamma=0.08,
        T_nu_e=0.079,
        T_nu_x=0.078,
        H_inv_sec=1.0,
        q_nodes=q,
        q_weights=wq,
        f_background=f_soft,
    )
    hard = apply_species_residual_relaxation(
        species="nue",
        I=I,
        J=J,
        w0=w0,
        T_gamma=0.08,
        T_nu_e=0.079,
        T_nu_x=0.078,
        H_inv_sec=1.0,
        q_nodes=q,
        q_weights=wq,
        f_background=f_hard,
    )

    assert hard.gamma_over_H_ell0 > soft.gamma_over_H_ell0
    assert hard.gamma_over_H_ell2 > soft.gamma_over_H_ell2


def test_residual_relaxation_uses_mismatch_and_distortion_factors():
    w0, I, J = _toy_state()
    base = apply_species_residual_relaxation(
        species="nue",
        I=I,
        J=J,
        w0=w0,
        T_gamma=0.08,
        T_nu_e=0.079,
        T_nu_x=0.078,
        H_inv_sec=1.0,
        gamma_factor_ell0=1.0,
        gamma_factor_ell2=1.0,
        mismatch_factor=1.0,
        distortion_factor_ell0=1.0,
        distortion_factor_ell2=1.0,
        distortion_rms=0.0,
    )
    boosted = apply_species_residual_relaxation(
        species="nue",
        I=I,
        J=J,
        w0=w0,
        T_gamma=0.08,
        T_nu_e=0.079,
        T_nu_x=0.078,
        H_inv_sec=1.0,
        gamma_factor_ell0=1.0,
        gamma_factor_ell2=1.0,
        mismatch_factor=1.15,
        distortion_factor_ell0=1.10,
        distortion_factor_ell2=1.20,
        distortion_rms=0.12,
    )

    assert boosted.gamma_over_H_ell0 > base.gamma_over_H_ell0
    assert boosted.gamma_over_H_ell2 > base.gamma_over_H_ell2
    assert boosted.distortion_rms > base.distortion_rms
