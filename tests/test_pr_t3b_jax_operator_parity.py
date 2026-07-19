"""PR-T3B preflight: element-wise SciPy <-> JAX parity for the
``nu-e`` and pair-process operators ported in
``rabbit.jax.collisions_jax``.

The ports mirror the SciPy reference structurally (matrix elements,
statistical factors, quadrature schemes).  Element-wise parity is
verified at the *shared Laguerre grid* configuration where the SciPy
``interp1d`` step is identity at the input nodes; this is the
configuration referenced in the PR-T3B phase prompt.

The pair port additionally evaluates the input distributions
off-node via PCHIP cubic Hermite while the SciPy reference uses a
``scipy.interpolate.interp1d`` natural cubic spline, so its parity
target is documented at a looser tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from rabbit.collisions.kernels import (
    G_L_NUE,
    G_L_NUX,
    G_R_NUE,
    G_R_NUX,
)
from rabbit.collisions.nu_e_scattering import NuEScatteringOperator
from rabbit.collisions.pair_processes import PairProcessOperator
from rabbit.jax.collisions_jax import (
    collision_statistical_factor_jax,
    laguerre_grid,
    make_nu_e_kernel,
    make_pair_kernel,
)


jax.config.update("jax_enable_x64", True)


def _fermi_dirac_np(y: np.ndarray) -> np.ndarray:
    return 1.0 / (np.exp(np.minimum(y, 500.0)) + 1.0)


def test_jax_collision_statistical_factor_is_six_monomial_pauli_polynomial() -> None:
    f1 = jnp.asarray([0.10, 0.35, 0.80])
    f2 = jnp.asarray([0.20, 0.45, 0.70])
    f3 = jnp.asarray([0.30, 0.55, 0.60])
    f4 = jnp.asarray([0.40, 0.65, 0.50])

    expected = f3 * f4 * (1.0 - f1) * (1.0 - f2) - f1 * f2 * (1.0 - f3) * (1.0 - f4)

    np.testing.assert_allclose(
        np.asarray(collision_statistical_factor_jax(f1, f2, f3, f4)),
        np.asarray(expected),
        rtol=0.0,
        atol=1.0e-16,
    )


# ---------------------------------------------------------------------------
# nu-e: element-wise parity at the shared Laguerre grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("species", ["nue", "nux"])
@pytest.mark.parametrize("N_q", [12, 20, 24])
def test_nu_e_jax_matches_scipy_shared_laguerre(species: str, N_q: int) -> None:
    """At ``N_int = N_q`` with ``q_nodes = laggauss(N_q)`` the SciPy
    interp1d is identity at the input nodes, so the JAX port must
    reproduce the SciPy operator to floating-point reduction order."""
    q_nodes_np, _ = laguerre_grid(N_q)
    q_nodes = jnp.asarray(q_nodes_np)

    T_MeV = 2.0  # MeV; arbitrary nonzero
    rng = np.random.default_rng(12345)
    base = _fermi_dirac_np(q_nodes_np)
    perturb = 1.0 + 0.05 * rng.standard_normal(q_nodes_np.shape)
    f_np = np.clip(base * perturb, 0.0, 1.0)
    f_jax = jnp.asarray(f_np)

    G_L = G_L_NUE if species == "nue" else G_L_NUX
    G_R = G_R_NUE if species == "nue" else G_R_NUX
    scipy_op = NuEScatteringOperator(G_L=G_L, G_R=G_R)
    # Force N_int == N_q for parity at the shared grid.
    scipy_op.N_quad_electron = 32
    # Reach into the SciPy implementation to mirror N_int = N_q exactly.
    # The SciPy reference uses ``min(N_q, 24)``; we test below 24, so it
    # already collapses to N_q.
    assert min(N_q, 24) == N_q

    C_scipy = scipy_op._evaluate_vectorized(f_np, q_nodes_np, T_MeV)

    kernel, _ = make_nu_e_kernel(species=species, N_q=N_q, N_int=N_q,
                                 N_quad_electron=32)
    C_jax = np.asarray(kernel(f_jax, q_nodes, jnp.asarray(T_MeV)))

    abs_err = np.max(np.abs(C_scipy - C_jax))
    # Outputs are O(1e-22 .. 1e-24) at T=2 MeV; any element-wise mismatch
    # at this level is dominated by floating-point reduction order.
    assert abs_err < 1e-30, (
        f"NuE shared-Laguerre parity failed: |Delta| = {abs_err:.3e}"
        f"  N_q={N_q} species={species}"
    )


@pytest.mark.parametrize("species", ["nue", "nux"])
def test_nu_e_jax_detailed_balance_at_fd(species: str) -> None:
    """``C[f_FD]`` vanishes to numerical precision when ``T_nu = T_gamma``."""
    N_q = 20
    q_nodes_np, _ = laguerre_grid(N_q)
    q_nodes = jnp.asarray(q_nodes_np)
    T = 1.5
    f_FD = _fermi_dirac_np(q_nodes_np)

    kernel, _ = make_nu_e_kernel(species=species, N_q=N_q, N_int=N_q)
    C = np.asarray(kernel(jnp.asarray(f_FD), q_nodes, jnp.asarray(T)))
    # Detailed balance: gain - loss vanishes identically at f = f_FD.
    assert np.max(np.abs(C)) < 1e-30, (
        f"NuE detailed balance failed for {species}: "
        f"max|C[f_FD]| = {np.max(np.abs(C)):.3e}"
    )


# ---------------------------------------------------------------------------
# Pair: detailed balance + structural parity (PCHIP vs SciPy cubic spline)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("species", ["nue", "nux"])
def test_pair_jax_detailed_balance_at_fd(species: str) -> None:
    """``C_pair[f_FD, f_FD]`` vanishes when ``T_nu = T_gamma`` and the
    ``f_nubar`` evaluation grid matches ``q_nodes`` (no interpolation)."""
    # Use the matched-grid configuration (N_q == N_quad) so that the
    # interpolant of ``f_nubar`` at ``y2_nodes = q_nodes`` is the
    # identity at the input nodes; detailed balance then collapses
    # algebraically (energy conservation y1+y2 = y3+y4 with FD inputs).
    N_q = 24
    q_nodes_np, _ = laguerre_grid(N_q)
    q_nodes = jnp.asarray(q_nodes_np)
    T = 1.5
    f_FD = jnp.asarray(_fermi_dirac_np(q_nodes_np))

    kernel, _ = make_pair_kernel(species=species, N_quad=N_q)
    C = np.asarray(kernel(f_FD, f_FD, q_nodes, jnp.asarray(T)))
    assert np.max(np.abs(C)) < 1e-30, (
        f"Pair detailed balance failed for {species}: "
        f"max|C[f_FD,f_FD]| = {np.max(np.abs(C)):.3e}"
    )


@pytest.mark.parametrize("species", ["nue", "nux"])
def test_pair_jax_matches_scipy_shared_laguerre(species: str) -> None:
    """Pair port matches SciPy element-wise on the shared Laguerre grid.

    With ``N_q == N_quad`` and ``q_nodes = laggauss(N_q)`` the
    ``f_nubar`` interpolation step is identity at the input nodes for
    both the SciPy reference (interp1d cubic) and the JAX port (PCHIP),
    and the y3 Gauss-Legendre quadrature is bit-for-bit shared between
    the two implementations.  Parity is then floating-point reduction
    order.

    A non-trivial physical signal is generated by a 50% perturbation
    above FD on both species, well above the residual interpolation
    noise floor.
    """
    N_q = 24
    q_nodes_np, _ = laguerre_grid(N_q)
    q_nodes = jnp.asarray(q_nodes_np)
    T = 2.0
    base = _fermi_dirac_np(q_nodes_np)
    f_np = np.clip(base * 1.5, 0.0, 1.0)
    f_bar_np = np.clip(base * 1.5, 0.0, 1.0)

    G_L = G_L_NUE if species == "nue" else G_L_NUX
    G_R = G_R_NUE if species == "nue" else G_R_NUX
    scipy_op = PairProcessOperator(G_L=G_L, G_R=G_R, N_quad=N_q)
    C_scipy = scipy_op.evaluate(f_np, f_bar_np, q_nodes_np, T)

    kernel, _ = make_pair_kernel(species=species, N_quad=N_q)
    C_jax = np.asarray(kernel(jnp.asarray(f_np), jnp.asarray(f_bar_np),
                              q_nodes, jnp.asarray(T)))

    abs_err = np.max(np.abs(C_scipy - C_jax))
    rel_scale = np.max(np.abs(C_scipy)) + 1e-300
    rel_err = abs_err / rel_scale
    # On the shared grid both implementations evaluate every quantity at
    # identical points, so element-wise parity should be bounded by
    # floating-point reduction order.
    assert abs_err < 1e-30 or rel_err < 1e-12, (
        f"Pair shared-Laguerre parity failed for {species}: "
        f"|Delta|={abs_err:.3e}  rel={rel_err:.3e}"
    )


@pytest.mark.parametrize("species", ["nue", "nux"])
def test_pair_jax_off_grid_matches_scipy_cubic(species: str) -> None:
    """Off-grid configuration: ``N_q != N_quad`` forces ``f_ν̄(y_2)``
    interpolation.

    With the JAX-native not-a-knot natural cubic spline
    (``rabbit.jax.cubic_spline_jax``) replacing PCHIP cubic Hermite
    inside ``pair_collision_jax``, the JAX port matches the SciPy
    reference (``scipy.interpolate.interp1d(kind='cubic')``)
    element-wise to floating-point reduction order on a 50%-above-FD
    perturbation.  Measured: ``~6.5e-16 rel`` at ``N_q=20``,
    ``N_quad=24``, ``T=2 MeV``.
    """
    N_q = 20
    q_nodes_np, _ = laguerre_grid(N_q)
    q_nodes = jnp.asarray(q_nodes_np)
    T = 2.0
    base = _fermi_dirac_np(q_nodes_np)
    f_np = np.clip(base * 1.5, 0.0, 1.0)
    f_bar_np = np.clip(base * 1.5, 0.0, 1.0)

    G_L = G_L_NUE if species == "nue" else G_L_NUX
    G_R = G_R_NUE if species == "nue" else G_R_NUX
    scipy_op = PairProcessOperator(G_L=G_L, G_R=G_R, N_quad=24)
    C_scipy = scipy_op.evaluate(f_np, f_bar_np, q_nodes_np, T)

    kernel, _ = make_pair_kernel(species=species, N_quad=24)
    C_jax = np.asarray(kernel(jnp.asarray(f_np), jnp.asarray(f_bar_np),
                              q_nodes, jnp.asarray(T)))

    abs_err = np.max(np.abs(C_scipy - C_jax))
    rel_scale = np.max(np.abs(C_scipy)) + 1e-300
    rel_err = abs_err / rel_scale
    # Reduction-order parity is the new target now that the JAX cubic
    # spline matches scipy interp1d cubic at element-wise machine
    # precision; lock at 1e-12 rel to flag any regression while
    # tolerating cross-platform float reduction noise.
    assert rel_err < 1.0e-12, (
        f"Pair JAX-cubic vs scipy off-grid parity widened beyond "
        f"reduction-order for {species}: "
        f"|Delta|={abs_err:.3e}  rel={rel_err:.3e}"
    )


# ---------------------------------------------------------------------------
# JIT determinism / reuse
# ---------------------------------------------------------------------------


def test_make_nu_e_kernel_is_deterministic() -> None:
    """The kernel factory produces deterministic outputs; repeated
    calls share underlying numpy data via the ``lru_cache`` on
    ``laguerre_grid`` and produce element-equal jnp arrays.  (The
    jnp arrays themselves are intentionally NOT identity-shared
    across calls — see ``laguerre_grid_jax`` doctring for the
    trace-context isolation rationale.)"""
    k1, aux1 = make_nu_e_kernel(species="nue", N_q=20, N_int=20)
    k2, aux2 = make_nu_e_kernel(species="nue", N_q=20, N_int=20)
    np.testing.assert_array_equal(np.asarray(aux1["y3_nodes"]),
                                   np.asarray(aux2["y3_nodes"]))
    np.testing.assert_array_equal(np.asarray(aux1["y2_nodes"]),
                                   np.asarray(aux2["y2_nodes"]))

    q_nodes_np, _ = laguerre_grid(20)
    f = _fermi_dirac_np(q_nodes_np) * 1.01
    out1 = np.asarray(k1(jnp.asarray(f), jnp.asarray(q_nodes_np),
                         jnp.asarray(2.0)))
    out2 = np.asarray(k2(jnp.asarray(f), jnp.asarray(q_nodes_np),
                         jnp.asarray(2.0)))
    np.testing.assert_array_equal(out1, out2)


def test_make_pair_kernel_is_deterministic() -> None:
    k1, aux1 = make_pair_kernel(species="nue", N_quad=24)
    k2, aux2 = make_pair_kernel(species="nue", N_quad=24)
    np.testing.assert_array_equal(np.asarray(aux1["y2_nodes"]),
                                   np.asarray(aux2["y2_nodes"]))
    np.testing.assert_array_equal(np.asarray(aux1["leg_nodes"]),
                                   np.asarray(aux2["leg_nodes"]))

    q_nodes_np, _ = laguerre_grid(20)
    f = _fermi_dirac_np(q_nodes_np) * 1.01
    out1 = np.asarray(k1(jnp.asarray(f), jnp.asarray(f),
                         jnp.asarray(q_nodes_np), jnp.asarray(2.0)))
    out2 = np.asarray(k2(jnp.asarray(f), jnp.asarray(f),
                         jnp.asarray(q_nodes_np), jnp.asarray(2.0)))
    np.testing.assert_array_equal(out1, out2)
