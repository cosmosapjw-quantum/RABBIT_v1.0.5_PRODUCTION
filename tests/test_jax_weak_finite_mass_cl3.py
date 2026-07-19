"""
test_jax_weak_finite_mass_cl3.py — Parity and integration tests for CL3 JAX port.

Tests:
  1. Scalar correction parity: JAX vs NumPy for all 6 channels
  2. Angular kernel parity: K_{r,0}, K_{r,1}, K_{r,2}
  3. I₀ normalization parity (FM-only and full CL3)
  4. CL3 live weak kernel: level-specialized dispatch
  5. CL3 budget wrapper contract
  6. Physical monotonicity: CL2→CL3 direction check
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
jax.config.update("jax_enable_x64", True)


# ── §1. Scalar correction parity ──

@pytest.mark.parametrize("channel", list("abcdef"))
def test_recoil_parity(channel):
    from rabbit.weak.finite_mass import recoil_scalar_correction
    from rabbit.jax.weak_finite_mass_jax import recoil_scalar_correction_jax

    E_e = np.linspace(1.01, 10.0, 50)
    E_nu = np.linspace(0.5, 8.0, 50)
    ref = recoil_scalar_correction(E_e, E_nu, channel)
    jax_val = np.asarray(recoil_scalar_correction_jax(jnp.asarray(E_e), jnp.asarray(E_nu), channel))
    assert np.max(np.abs(ref - jax_val)) < 1e-15


@pytest.mark.parametrize("channel", list("abcdef"))
def test_weak_magnetism_parity(channel):
    from rabbit.weak.finite_mass import weak_magnetism_scalar_correction
    from rabbit.jax.weak_finite_mass_jax import weak_magnetism_scalar_correction_jax

    E_e = np.linspace(1.01, 10.0, 50)
    E_nu = np.linspace(0.5, 8.0, 50)
    ref = weak_magnetism_scalar_correction(E_e, E_nu, channel)
    jax_val = np.asarray(weak_magnetism_scalar_correction_jax(jnp.asarray(E_e), jnp.asarray(E_nu), channel))
    assert np.max(np.abs(ref - jax_val)) < 1e-15


@pytest.mark.parametrize("channel", list("abcdef"))
def test_combined_fm_parity(channel):
    from rabbit.weak.finite_mass import finite_mass_scalar_correction
    from rabbit.jax.weak_finite_mass_jax import finite_mass_scalar_correction_jax

    E_e = np.linspace(1.01, 10.0, 50)
    E_nu = np.linspace(0.5, 8.0, 50)
    ref = finite_mass_scalar_correction(E_e, E_nu, channel)
    jax_val = np.asarray(finite_mass_scalar_correction_jax(jnp.asarray(E_e), jnp.asarray(E_nu), channel))
    assert np.max(np.abs(ref - jax_val)) < 1e-15


# ── §2. Angular kernel parity ──

@pytest.mark.parametrize("channel", list("abcdef"))
def test_angular_kernel_parity(channel):
    from rabbit.weak.finite_mass import finite_mass_angular_kernel
    from rabbit.jax.weak_finite_mass_jax import finite_mass_angular_kernel_jax

    ref = finite_mass_angular_kernel(3.0, 2.0, channel)
    jax_val = finite_mass_angular_kernel_jax(3.0, 2.0, channel)
    for ell in (0, 1, 2):
        assert abs(ref[ell] - jax_val[ell]) < 1e-14, f"K_{ell} mismatch for channel {channel}"


# ── §3. I₀ normalization parity ──

def test_I0_fm_recoil_parity():
    from rabbit.weak.finite_mass import compute_I0_with_fm
    from rabbit.jax.weak_finite_mass_jax import compute_I0_with_fm_jax
    ref = compute_I0_with_fm(True, False)
    jax_val = compute_I0_with_fm_jax(True, False)
    assert abs(ref - jax_val) < 1e-14


def test_I0_fm_wm_parity():
    from rabbit.weak.finite_mass import compute_I0_with_fm
    from rabbit.jax.weak_finite_mass_jax import compute_I0_with_fm_jax
    ref = compute_I0_with_fm(False, True)
    jax_val = compute_I0_with_fm_jax(False, True)
    assert abs(ref - jax_val) < 1e-14


def test_I0_fm_both_parity():
    from rabbit.weak.finite_mass import compute_I0_with_fm
    from rabbit.jax.weak_finite_mass_jax import compute_I0_with_fm_jax
    ref = compute_I0_with_fm(True, True)
    jax_val = compute_I0_with_fm_jax(True, True)
    assert abs(ref - jax_val) < 1e-14


def test_I0_cl3_physical_range():
    """CL3 I₀ should be between Born and CL2 (FM is net negative)."""
    from rabbit.jax.weak_finite_mass_jax import _I0_CL3
    from rabbit.weak.corrections import compute_I0_corrected

    I0_born = compute_I0_corrected(False, False)
    I0_cl2 = compute_I0_corrected(True, True)
    assert I0_born < float(_I0_CL3) < I0_cl2, (
        f"CL3 I₀ should satisfy Born < CL3 < CL2: "
        f"{I0_born} < {float(_I0_CL3)} < {I0_cl2}"
    )


# ── §4. CL3 live weak kernel ──

def _make_equilibrium_inputs(N_q=12):
    from numpy.polynomial.laguerre import laggauss
    q_np, _ = laggauss(N_q)
    q = jnp.asarray(q_np, dtype=jnp.float64)
    f_eq = jnp.asarray(1.0 / (np.exp(q_np) + 1.0), dtype=jnp.float64)
    return q, f_eq


def test_cl3_kernel_runs():
    from rabbit.jax.weak_live_jax import (
        compute_live_rates_from_monopoles_cl3_jax,
        build_not_a_knot_matrix,
    )
    q, f_eq = _make_equilibrium_inputs()
    spl = build_not_a_knot_matrix(q)
    T_g = jnp.asarray(1.0, dtype=jnp.float64)
    T_n = jnp.asarray(0.8, dtype=jnp.float64)
    tau = jnp.asarray(878.4, dtype=jnp.float64)

    lnp, lpn, I0 = compute_live_rates_from_monopoles_cl3_jax(
        T_g, T_n, tau, q, f_eq, f_eq, spl
    )
    assert float(lnp) > 0
    assert float(lpn) > 0
    assert np.isfinite(float(lnp))
    assert np.isfinite(float(lpn))


def test_cl3_dispatcher_matches_direct():
    from rabbit.jax.weak_live_jax import (
        compute_live_rates_from_monopoles_cl3_jax,
        compute_live_rates_from_monopoles_level_specialized_jax,
        build_not_a_knot_matrix,
    )
    q, f_eq = _make_equilibrium_inputs()
    spl = build_not_a_knot_matrix(q)
    T_g = jnp.asarray(1.0, dtype=jnp.float64)
    T_n = jnp.asarray(0.8, dtype=jnp.float64)
    tau = jnp.asarray(878.4, dtype=jnp.float64)

    lnp_d, lpn_d, I0_d = compute_live_rates_from_monopoles_cl3_jax(
        T_g, T_n, tau, q, f_eq, f_eq, spl
    )
    lnp_s, lpn_s, I0_s = compute_live_rates_from_monopoles_level_specialized_jax(
        T_g, T_n, tau, q, f_eq, f_eq, correction_level=3, spline_matrix=spl
    )
    assert abs(float(lnp_d) - float(lnp_s)) < 1e-15
    assert abs(float(lpn_d) - float(lpn_s)) < 1e-15


def test_cl3_moment_kernel_isotropic_limit_matches_scalar_kernel():
    from rabbit.jax.weak_live_jax import (
        compute_live_rates_from_moments_cl3_jax,
        compute_live_rates_from_monopoles_cl3_jax,
        build_not_a_knot_matrix,
    )
    q, f_eq = _make_equilibrium_inputs()
    spl = build_not_a_knot_matrix(q)
    zeros = jnp.zeros_like(f_eq)
    T_g = jnp.asarray(1.0, dtype=jnp.float64)
    T_n = jnp.asarray(0.8, dtype=jnp.float64)
    tau = jnp.asarray(878.4, dtype=jnp.float64)

    lnp_scalar, lpn_scalar, I0_scalar = compute_live_rates_from_monopoles_cl3_jax(
        T_g, T_n, tau, q, f_eq, f_eq, spl
    )
    lnp_moment, lpn_moment, I0_moment = compute_live_rates_from_moments_cl3_jax(
        T_g, T_n, tau, q,
        f_eq, zeros, zeros,
        f_eq, zeros, zeros,
        spl,
    )

    assert abs(float(lnp_scalar) - float(lnp_moment)) < 1e-15
    assert abs(float(lpn_scalar) - float(lpn_moment)) < 1e-15
    assert abs(float(I0_scalar) - float(I0_moment)) < 1e-15


def test_cl3_moment_kernel_responds_to_l1_l2_inputs():
    from rabbit.jax.weak_live_jax import (
        compute_live_rates_from_moments_cl3_jax,
        build_not_a_knot_matrix,
    )
    q, f_eq = _make_equilibrium_inputs()
    spl = build_not_a_knot_matrix(q)
    zeros = jnp.zeros_like(f_eq)
    f1 = 0.02 * f_eq * (1.0 - f_eq)
    f2 = 0.01 * f_eq * (1.0 - f_eq)
    T_g = jnp.asarray(1.0, dtype=jnp.float64)
    T_n = jnp.asarray(0.8, dtype=jnp.float64)
    tau = jnp.asarray(878.4, dtype=jnp.float64)

    lnp_iso, lpn_iso, _ = compute_live_rates_from_moments_cl3_jax(
        T_g, T_n, tau, q,
        f_eq, zeros, zeros,
        f_eq, zeros, zeros,
        spl,
    )
    lnp_aniso, lpn_aniso, _ = compute_live_rates_from_moments_cl3_jax(
        T_g, T_n, tau, q,
        f_eq, f1, f2,
        f_eq, f1, f2,
        spl,
    )

    assert abs(float(lnp_aniso) - float(lnp_iso)) > 0.0
    assert abs(float(lpn_aniso) - float(lpn_iso)) > 0.0


# ── §5. Budget wrapper ──

def test_cl3_budget_wrapper():
    from rabbit.jax.weak_live_jax import (
        compute_live_rates_with_budget_from_monopoles,
        build_not_a_knot_matrix,
    )
    q, f_eq = _make_equilibrium_inputs()
    spl = build_not_a_knot_matrix(q)
    T_g = jnp.asarray(1.0, dtype=jnp.float64)
    T_n = jnp.asarray(0.8, dtype=jnp.float64)
    tau = jnp.asarray(878.4, dtype=jnp.float64)

    budget = compute_live_rates_with_budget_from_monopoles(
        T_g, T_n, tau, q, f_eq, f_eq, correction_level=3, spline_matrix=spl
    )
    assert budget.correction_level == 3
    assert "cl3" in budget.weak_mode
    assert budget.lambda_np > 0
    assert budget.lambda_pn > 0


# ── §6. Physical monotonicity ──

def test_cl3_rates_lower_than_cl2():
    """FM/WM net negative → CL3 rates should be slightly lower than CL2."""
    from rabbit.jax.weak_live_jax import (
        compute_live_rates_from_monopoles_level_specialized_jax,
        build_not_a_knot_matrix,
    )
    q, f_eq = _make_equilibrium_inputs()
    spl = build_not_a_knot_matrix(q)
    T_g = jnp.asarray(1.0, dtype=jnp.float64)
    T_n = jnp.asarray(0.8, dtype=jnp.float64)
    tau = jnp.asarray(878.4, dtype=jnp.float64)

    lnp2, lpn2, _ = compute_live_rates_from_monopoles_level_specialized_jax(
        T_g, T_n, tau, q, f_eq, f_eq, correction_level=2, spline_matrix=spl
    )
    lnp3, lpn3, _ = compute_live_rates_from_monopoles_level_specialized_jax(
        T_g, T_n, tau, q, f_eq, f_eq, correction_level=3, spline_matrix=spl
    )
    # FM/WM reduces rates
    assert float(lnp3) < float(lnp2), "CL3 λ_np should be < CL2 (FM net negative)"
    assert float(lpn3) < float(lpn2), "CL3 λ_pn should be < CL2 (FM net negative)"

    # But not by more than a few percent
    delta = abs(float(lnp3) - float(lnp2)) / float(lnp2)
    assert delta < 0.02, f"CL2→CL3 change too large: {delta:.4f}"
