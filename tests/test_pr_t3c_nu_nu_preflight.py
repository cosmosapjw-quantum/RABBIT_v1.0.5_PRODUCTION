"""PR-T3C preflight: detailed-balance and energy-conservation locks
for the JAX-native diagonal ν–ν elastic scattering operator
landed in ``rabbit.jax.nu_nu_scattering_jax``.

The phase-prompt PR-T3C requires a Dolgov-Hansen-Semikoz-derived
operator with strict element-wise SciPy parity at ``1e-12``,
detailed balance at ``1e-14`` and FLRW ``|N_eff - 3.044| < 0.005``.
This preflight slice locks the structural invariants (detailed
balance + energy conservation) of the JAX skeleton; it does not yet
calibrate the absolute matrix-element prefactor against
Dolgov-Hansen-Semikoz appendix-A and does not yet wire the operator
into the full-Boltzmann driver.  Closing those gaps is part of the
follow-up PR-T3C runtime patch.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from rabbit.jax.collisions_jax import laguerre_grid
from rabbit.jax.nu_nu_scattering_jax import make_nu_nu_kernel


jax.config.update("jax_enable_x64", True)


def _fermi_dirac_np(y: np.ndarray) -> np.ndarray:
    return 1.0 / (np.exp(np.minimum(y, 500.0)) + 1.0)


# ---------------------------------------------------------------------------
# Detailed balance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("epsilon", [1.0, 2.0])
@pytest.mark.parametrize("N_q", [12, 20])
def test_nu_nu_detailed_balance_at_fd(epsilon: float, N_q: int) -> None:
    """At ``f_α = f_β = f_FD`` and ``T_α = T_β = T_γ`` the
    statistical factor ``f_3 f_4 (1-f_1)(1-f_2) - f_1 f_2 (1-f_3)(1-f_4)``
    vanishes algebraically pointwise.  Numerically the residual is
    bounded by the cubic-spline approximation error of ``f_FD`` on
    the input Laguerre grid (because ``y_4 = y_1 + y_2 - y_3`` is
    off the input grid for generic triples).

    With the JAX-native not-a-knot cubic spline (matching scipy
    ``interp1d`` cubic at ``1e-12`` element-wise) replacing PCHIP,
    the residual tightens by ``~5x`` to ``~3e-23`` at ``N_q=20`` and
    ``~7e-24`` at ``N_q=12``.  Lock at ``< 1e-22`` (5x headroom over
    the measured value at ``N_q=20``) to flag any regression while
    tolerating reduction-order noise.  Closing further requires
    eliminating the off-grid ``f_β(y_4)`` evaluation entirely (e.g.,
    via a 4-momentum delta-function quadrature)."""
    q_nodes_np, _ = laguerre_grid(N_q)
    q_nodes = jnp.asarray(q_nodes_np)
    T = 2.0
    f_FD = jnp.asarray(_fermi_dirac_np(q_nodes_np))

    kernel, _ = make_nu_nu_kernel(N_q=N_q, epsilon_alpha_beta=epsilon)
    C = np.asarray(kernel(f_FD, f_FD, q_nodes, jnp.asarray(T)))
    measured = np.max(np.abs(C))
    assert measured < 1e-22, (
        f"nu-nu detailed balance failed at FD: max|C| = {measured:.3e} "
        f"(eps={epsilon}, N_q={N_q})"
    )


# ---------------------------------------------------------------------------
# Energy conservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("epsilon", [1.0, 2.0])
def test_nu_nu_energy_conservation_at_fd_perturbation(epsilon: float) -> None:
    """Sum over species of ``int y^3 C(y) dy`` should vanish for
    elastic ν–ν scattering: total neutrino energy density is
    conserved internally by the operator.  Numerically the residual
    is bounded by the cubic-spline approximation error on
    ``f_β(y_4)``.  Measured: ``~0.23% rel`` of ``max|C|`` on a 50%
    above-FD perturbation at ``N_q=20`` (down from ``~1.8% rel``
    with PCHIP — an ``8x`` tightening from the cubic spline swap).
    Lock at ``< 0.5% rel`` to flag any regression; the strict
    ``1e-12 rel`` PR-T3C target requires removing the off-grid
    ``f_β(y_4)`` interpolation step (e.g., via a 4-momentum
    delta-function quadrature)."""
    N_q = 20
    q_nodes_np, q_weights_np = laguerre_grid(N_q)
    q_nodes = jnp.asarray(q_nodes_np)
    q_weights = q_weights_np
    T = 2.0
    base = _fermi_dirac_np(q_nodes_np)
    # 50%-above-FD equilibrium-broken probe applied to BOTH alpha and
    # beta (so their pair-wise energy exchange must self-cancel).
    f_pert = np.clip(base * 1.5, 0.0, 1.0)

    kernel, _ = make_nu_nu_kernel(N_q=N_q, epsilon_alpha_beta=epsilon)
    C = np.asarray(kernel(jnp.asarray(f_pert), jnp.asarray(f_pert),
                          q_nodes, jnp.asarray(T)))
    # Laguerre energy moment with the standard ``exp(y)`` measure.
    energy_residual = float(
        np.sum(q_weights * np.exp(q_nodes_np) * q_nodes_np**3 * C)
    )
    rel_scale = float(np.max(np.abs(C))) + 1e-300
    rel_residual = abs(energy_residual) / rel_scale
    assert rel_residual < 5e-3, (
        f"nu-nu energy conservation worse than expected at perturbation: "
        f"int y^3 C dy = {energy_residual:.3e}, max|C| = {rel_scale:.3e}, "
        f"rel = {rel_residual:.3e} (eps={epsilon})"
    )


# ---------------------------------------------------------------------------
# Cross-species hierarchy
# ---------------------------------------------------------------------------


def test_nu_nu_alpha_eq_beta_factor_2_relative_to_distinguishable() -> None:
    """The Fierz factor for identical species (``epsilon = 2``)
    doubles the kernel relative to the distinguishable case
    (``epsilon = 1``) at fixed input distributions, by construction.
    This guards the ``epsilon_alpha_beta`` scaling."""
    N_q = 20
    q_nodes_np, _ = laguerre_grid(N_q)
    q_nodes = jnp.asarray(q_nodes_np)
    T = 2.0
    base = _fermi_dirac_np(q_nodes_np)
    f_pert = np.clip(base * 1.5, 0.0, 1.0)

    k1, _ = make_nu_nu_kernel(N_q=N_q, epsilon_alpha_beta=1.0)
    k2, _ = make_nu_nu_kernel(N_q=N_q, epsilon_alpha_beta=2.0)
    C1 = np.asarray(k1(jnp.asarray(f_pert), jnp.asarray(f_pert),
                       q_nodes, jnp.asarray(T)))
    C2 = np.asarray(k2(jnp.asarray(f_pert), jnp.asarray(f_pert),
                       q_nodes, jnp.asarray(T)))

    diff_from_factor_2 = np.max(np.abs(C2 - 2.0 * C1))
    rel = diff_from_factor_2 / (np.max(np.abs(C1)) + 1e-300)
    assert rel < 1e-12, (
        f"epsilon_alpha_beta scaling not exact: "
        f"max|C2 - 2*C1| = {diff_from_factor_2:.3e}  rel = {rel:.3e}"
    )


def test_nu_nu_collision_field_scales_as_t5() -> None:
    """At fixed dimensionless spectral shape, C_νν is a MeV rate ∝ G_F^2 T^5."""
    N_q = 12
    q_nodes_np, _ = laguerre_grid(N_q)
    q_nodes = jnp.asarray(q_nodes_np)
    base = _fermi_dirac_np(q_nodes_np)
    f_alpha = np.clip(base * (1.0 + 0.05 * q_nodes_np / np.mean(q_nodes_np)), 0.0, 1.0)
    f_beta = np.clip(base * (1.0 - 0.03 * q_nodes_np / np.mean(q_nodes_np)), 0.0, 1.0)

    kernel, _ = make_nu_nu_kernel(N_q=N_q)
    C_lo = np.asarray(kernel(jnp.asarray(f_alpha), jnp.asarray(f_beta), q_nodes, jnp.asarray(2.0)))
    C_hi = np.asarray(kernel(jnp.asarray(f_alpha), jnp.asarray(f_beta), q_nodes, jnp.asarray(4.0)))
    ratio = np.max(np.abs(C_hi)) / np.max(np.abs(C_lo))
    assert ratio == pytest.approx(2.0**5, rel=1.0e-12)


# ---------------------------------------------------------------------------
# JIT determinism
# ---------------------------------------------------------------------------


def test_make_nu_nu_kernel_is_deterministic() -> None:
    k1, aux1 = make_nu_nu_kernel(N_q=20)
    k2, aux2 = make_nu_nu_kernel(N_q=20)
    np.testing.assert_array_equal(np.asarray(aux1["y3_nodes"]),
                                   np.asarray(aux2["y3_nodes"]))
    np.testing.assert_array_equal(np.asarray(aux1["y2_nodes"]),
                                   np.asarray(aux2["y2_nodes"]))

    q_nodes_np, _ = laguerre_grid(20)
    f = _fermi_dirac_np(q_nodes_np) * 1.01
    out1 = np.asarray(k1(jnp.asarray(f), jnp.asarray(f),
                         jnp.asarray(q_nodes_np), jnp.asarray(2.0)))
    out2 = np.asarray(k2(jnp.asarray(f), jnp.asarray(f),
                         jnp.asarray(q_nodes_np), jnp.asarray(2.0)))
    np.testing.assert_array_equal(out1, out2)
