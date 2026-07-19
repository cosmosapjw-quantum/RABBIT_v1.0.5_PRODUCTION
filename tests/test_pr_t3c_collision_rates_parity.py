"""PR-T3C-PF #6: element-wise parity for the JAX total-rate
helper (``rabbit.jax.collision_rates_jax``) versus the SciPy
reference ``NuEScatteringOperator.total_rate_all_channels``.

The Mangano / Hannestad-Madsen total rate
``Γ_α(T) = (7π/12) G_F² T⁵ a_α`` is a closed-form scalar that the
canonical PR-T3B asymptotic-preserving (AP-form) collision wrapper
will use to build the equilibration timescale ``Γ/H`` without
re-deriving it from the full 2D collision integral every RHS
call.  This test locks the JAX port at machine precision against
the SciPy reference for both ``nu_e`` (CC+NC) and ``nu_x`` (NC
only) channels across a representative temperature range.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from rabbit.collisions.nu_e_scattering import (
    NuEScatteringOperator,
    nu_e_scattering_operator,
    nu_x_scattering_operator,
)
from rabbit.jax.collision_rates_jax import (
    gamma_over_H_jax,
    total_rate_jax,
    total_rate_nu_e_jax,
    total_rate_nu_x_jax,
)


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("T_MeV", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_total_rate_nu_e_matches_scipy(T_MeV: float) -> None:
    """JAX ν_e total rate matches SciPy at machine precision."""
    op = nu_e_scattering_operator()
    scipy_rate = op.total_rate_all_channels(T_MeV)
    jax_rate = float(np.asarray(total_rate_nu_e_jax(jnp.asarray(T_MeV))))
    rel_err = abs(scipy_rate - jax_rate) / max(abs(scipy_rate), 1e-300)
    assert rel_err < 1e-14, (
        f"nu_e total rate parity failed at T={T_MeV} MeV: "
        f"SciPy {scipy_rate:.6e}, JAX {jax_rate:.6e}, rel {rel_err:.3e}"
    )


@pytest.mark.parametrize("T_MeV", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_total_rate_nu_x_matches_scipy(T_MeV: float) -> None:
    """JAX ν_x total rate matches SciPy at machine precision."""
    op = nu_x_scattering_operator()
    scipy_rate = op.total_rate_all_channels(T_MeV)
    jax_rate = float(np.asarray(total_rate_nu_x_jax(jnp.asarray(T_MeV))))
    rel_err = abs(scipy_rate - jax_rate) / max(abs(scipy_rate), 1e-300)
    assert rel_err < 1e-14, (
        f"nu_x total rate parity failed at T={T_MeV} MeV: "
        f"SciPy {scipy_rate:.6e}, JAX {jax_rate:.6e}, rel {rel_err:.3e}"
    )


def test_total_rate_dispatch() -> None:
    """``total_rate_jax`` dispatches by species name."""
    T = jnp.asarray(1.0)
    np.testing.assert_allclose(
        np.asarray(total_rate_jax(T, "nue")),
        np.asarray(total_rate_nu_e_jax(T)),
    )
    np.testing.assert_allclose(
        np.asarray(total_rate_jax(T, "nux")),
        np.asarray(total_rate_nu_x_jax(T)),
    )
    with pytest.raises(ValueError):
        total_rate_jax(T, "tau_neutrino")


def test_total_rate_t5_scaling() -> None:
    """``Γ ∝ T^5`` to floating-point precision."""
    T1, T2 = 1.0, 2.0
    r1 = float(np.asarray(total_rate_nu_e_jax(jnp.asarray(T1))))
    r2 = float(np.asarray(total_rate_nu_e_jax(jnp.asarray(T2))))
    expected_ratio = (T2 / T1) ** 5
    measured_ratio = r2 / r1
    assert abs(measured_ratio - expected_ratio) / expected_ratio < 1e-14, (
        f"T^5 scaling violated: measured ratio {measured_ratio:.6e}, "
        f"expected {expected_ratio:.6e}"
    )


def test_species_coupling_ratio() -> None:
    """``Γ_e / Γ_x = a_e / a_x ≈ 4.68`` (Notzold-Raffelt)."""
    from rabbit.collisions.kernels import A_TOTAL_NUE, A_TOTAL_NUX

    T = jnp.asarray(1.0)
    r_e = float(np.asarray(total_rate_nu_e_jax(T)))
    r_x = float(np.asarray(total_rate_nu_x_jax(T)))
    expected = A_TOTAL_NUE / A_TOTAL_NUX  # ≈ 4.68
    measured = r_e / r_x
    assert abs(measured - expected) / expected < 1e-14, (
        f"species coupling ratio drifted: measured {measured:.6f}, "
        f"expected {expected:.6f}"
    )


def test_gamma_over_h_dimensional_consistency() -> None:
    """``Γ/H`` is dimensionless; numerical scale matches the
    standard freeze-out condition ``Γ/H ~ 1`` at ``T ~ 1 MeV``
    in the early universe."""
    # Representative early-universe Hubble at T = 1 MeV:
    # H ~ 1.66 sqrt(g_*/10) T^2 / M_pl ~ 1e-22 MeV.
    T = jnp.asarray(1.0)
    H = jnp.asarray(1e-22)
    ratio = float(np.asarray(gamma_over_H_jax(T, H, "nue")))
    # Γ_e(1 MeV) ~ 1.8 * 1e-22 MeV * 2.353 ~ 6e-23 / 1e-22 ~ 0.6.
    # The exact scale is what matters: the order of magnitude
    # should land near unity at the freeze-out temperature.
    assert 1e-2 < ratio < 1e2, (
        f"Γ/H at T=1 MeV out of expected freeze-out range: {ratio:.3e}"
    )


def test_gamma_over_h_jit_compatible() -> None:
    """The helper is JIT-compatible."""
    fn = jax.jit(gamma_over_H_jax, static_argnames=("species",))
    T = jnp.asarray(1.0)
    H = jnp.asarray(1e-22)
    out = float(np.asarray(fn(T, H, "nue")))
    assert np.isfinite(out)


# ---------------------------------------------------------------------------
# Diagonal nu-nu total rate (Mangano / DH-S leading-order)
# ---------------------------------------------------------------------------


def test_total_rate_nu_nu_diagonal_t5_scaling() -> None:
    """``Γ_νν ∝ T⁵``  (Mangano 2005 leading-order form)."""
    from rabbit.jax.collision_rates_jax import total_rate_nu_nu_diagonal_jax

    r1 = float(np.asarray(total_rate_nu_nu_diagonal_jax(jnp.asarray(1.0))))
    r2 = float(np.asarray(total_rate_nu_nu_diagonal_jax(jnp.asarray(2.0))))
    expected_ratio = 32.0  # (2/1)^5
    assert abs(r2 / r1 - expected_ratio) / expected_ratio < 1e-14


def test_total_rate_nu_nu_diagonal_fierz_factor() -> None:
    """``Γ_νν(α=β) / Γ_νν(α≠β) = 2`` (identical-particle Fierz
    factor)."""
    from rabbit.jax.collision_rates_jax import total_rate_nu_nu_diagonal_jax

    T = jnp.asarray(1.0)
    r_eq = float(np.asarray(total_rate_nu_nu_diagonal_jax(T, alpha_eq_beta=True)))
    r_neq = float(np.asarray(total_rate_nu_nu_diagonal_jax(T, alpha_eq_beta=False)))
    assert abs(r_eq / r_neq - 2.0) / 2.0 < 1e-14


def test_total_rate_nu_nu_diagonal_numpy_jax_share_prefactor() -> None:
    """The 3T NumPy thermo source and JAX rate helper use one prefactor."""
    from rabbit.jax.collision_rates_jax import total_rate_nu_nu_diagonal_jax
    from rabbit.thermo.rate_prefactors import total_rate_nu_nu_diagonal

    for T in (0.3, 1.0, 3.0):
        for identical in (False, True):
            np_rate = total_rate_nu_nu_diagonal(T, alpha_eq_beta=identical)
            jax_rate = float(
                np.asarray(
                    total_rate_nu_nu_diagonal_jax(
                        jnp.asarray(T),
                        alpha_eq_beta=identical,
                    )
                )
            )
            assert jax_rate == pytest.approx(np_rate, rel=1e-14, abs=0.0)


def test_gamma_nu_nu_over_h_dimensional_consistency() -> None:
    """At ``T = 1 MeV`` and ``H ~ 10⁻²² MeV`` (radiation era),
    ``Γ_νν / H ~ O(1)`` — the canonical freeze-out scale."""
    from rabbit.jax.collision_rates_jax import gamma_nu_nu_over_H_jax

    ratio = float(
        np.asarray(
            gamma_nu_nu_over_H_jax(jnp.asarray(1.0), jnp.asarray(1e-22), alpha_eq_beta=False)
        )
    )
    assert 1e-2 < ratio < 1e2, (
        f"Γ_νν / H at T=1 MeV out of expected freeze-out range: {ratio:.3e}"
    )


def test_total_rate_nu_nu_diagonal_jit_compatible() -> None:
    """JIT-compatible."""
    from rabbit.jax.collision_rates_jax import total_rate_nu_nu_diagonal_jax

    fn = jax.jit(total_rate_nu_nu_diagonal_jax, static_argnames=("alpha_eq_beta",))
    T = jnp.asarray(1.0)
    out = float(np.asarray(fn(T, alpha_eq_beta=True)))
    assert np.isfinite(out) and out > 0.0
