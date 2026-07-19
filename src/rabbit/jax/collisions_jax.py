"""JAX-native ports of the SciPy ``NuEScatteringOperator`` and
``PairProcessOperator``.

PR-T3B preflight: this module provides JIT-compatible 1D collision
kernels intended for the full-Boltzmann gather-collide-scatter cycle.
The ports mirror the SciPy reference structure
(``rabbit.collisions.nu_e_scattering._evaluate_vectorized`` and
``rabbit.collisions.pair_processes.evaluate``) including matrix
elements, statistical factors and quadrature schemes.

Element-wise parity vs the SciPy reference:

* ``nu_e_collision_jax`` — at the reference Laguerre grid configuration
  ``N_int = N_q`` with ``q_nodes = laggauss(N_q)``, the SciPy interp1d
  step is identity on the input nodes and the JAX port matches the
  SciPy output to floating-point reduction order (target: ``< 1e-12``
  absolute, often ``< 1e-15``).

* ``pair_collision_jax`` — the SciPy reference integrates ``y3`` on
  ``[0, y1 + y2]`` via Gauss-Legendre, requiring off-node interpolation
  of the input distribution.  The JAX port currently uses the existing
  monotone PCHIP cubic Hermite interpolant (``q_advection_jax``) rather
  than scipy ``interp1d``'s natural cubic spline.  Element-wise parity
  vs SciPy is therefore looser than NuE; the achievable tolerance
  depends on the smoothness of ``f`` and the chosen ``N_q``.  See
  ``tests/test_pr_t3b_jax_operator_parity.py`` for measured numbers.

Both ports preserve the SciPy invariants:

1. Detailed balance: ``C[f_FD(T_nu = T_gamma)] -> 0`` to numerical
   precision.
2. The matrix element + statistical factor structure is bit-for-bit
   identical to the SciPy operators on shared inputs.

The ports do **not** wire collision energy transfer into the 3T
thermodynamic sector — that is delivered separately via
``coupled_3T_rhs_from_collision_moments_jax``
(``rabbit.jax.nudec_coupled_jax``).
"""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss

from rabbit.collisions.kernels import (
    G_F_MEV,
    G_L_NUE,
    G_L_NUX,
    G_R_NUE,
    G_R_NUX,
    HM_REDUCED_COLLISION_DENOMINATOR,
)
from rabbit.jax.cubic_spline_jax import cubic_spline_eval_with_clamp
from rabbit.jax.q_advection_jax import _hermite_interp, _pchip_slopes

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Quadrature builders
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def laguerre_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached numpy Gauss-Laguerre nodes and weights of order ``n``."""
    nodes, weights = laggauss(int(n))
    return nodes.astype(np.float64), weights.astype(np.float64)


@lru_cache(maxsize=16)
def legendre_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached numpy Gauss-Legendre nodes and weights of order ``n``."""
    nodes, weights = leggauss(int(n))
    return nodes.astype(np.float64), weights.astype(np.float64)


def laguerre_grid_jax(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JAX wrapper around ``laguerre_grid``.  Intentionally NOT
    ``lru_cache``-decorated: caching ``jnp.ndarray`` outputs from a
    function that may be called from inside a JIT trace risks
    leaking trace-local DeviceArrays across trace boundaries (the
    same anti-pattern that produced ``UnexpectedTracerError`` in
    the driver's ``_cached_*`` helpers; see
    ``docs/audit/PR-T3B_jax_kernel_runtime.md``).  The underlying
    numpy arrays are still cached via ``laguerre_grid``; this
    wrapper applies ``jnp.asarray`` afresh per call so each
    DeviceArray is tied to the current trace context.
    """
    nodes, weights = laguerre_grid(n)
    return jnp.asarray(nodes), jnp.asarray(weights)


def legendre_grid_jax(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JAX wrapper around ``legendre_grid``.  Same caching
    discipline as ``laguerre_grid_jax``: numpy is LRU-cached,
    ``jnp.asarray`` is fresh per call to keep the DeviceArrays
    trace-local and avoid leaks across JIT boundaries."""
    nodes, weights = legendre_grid(n)
    return jnp.asarray(nodes), jnp.asarray(weights)


# ---------------------------------------------------------------------------
# Kinematic primitives
# ---------------------------------------------------------------------------


def _fermi_dirac_jax(y: jnp.ndarray) -> jnp.ndarray:
    """``1/(e^y + 1)`` clamped to avoid overflow at large ``y``."""
    return 1.0 / (jnp.exp(jnp.minimum(y, 500.0)) + 1.0)


def collision_statistical_factor_jax(f1, f2, f3, f4):
    """Quartic-cancelled six-monomial Pauli factor for diagonal 2-to-2 kernels."""

    return (
        f3 * f4
        - f1 * f2
        + f1 * f2 * f3
        + f1 * f2 * f4
        - f1 * f3 * f4
        - f2 * f3 * f4
    )


def _matrix_element_scat_jax(
    y1: jnp.ndarray,
    y2: jnp.ndarray,
    y3: jnp.ndarray,
    y4: jnp.ndarray,
    GL2_GR2: float,
    GL2_mGR2: float,
) -> jnp.ndarray:
    """Angular-averaged ``|M|^2`` for ``nu + e -> nu + e`` (Hannestad-Madsen)."""
    return (
        GL2_GR2 * ((y1 * y2) ** 2 + (y3 * y4) ** 2)
        + GL2_mGR2 * ((y1 * y4) ** 2 + (y2 * y3) ** 2)
    )


def _matrix_element_ann_jax(
    y1: jnp.ndarray,
    y2: jnp.ndarray,
    y3: jnp.ndarray,
    y4: jnp.ndarray,
    GL2_GR2: float,
    GL2_mGR2: float,
) -> jnp.ndarray:
    """Angular-averaged ``|M|^2`` for ``nu + nubar -> e+ + e-``."""
    return (
        GL2_GR2 * ((y1 * y3) ** 2 + (y2 * y4) ** 2)
        + GL2_mGR2 * ((y1 * y4) ** 2 + (y2 * y3) ** 2)
    )


# ---------------------------------------------------------------------------
# nu-e elastic scattering
# ---------------------------------------------------------------------------


def nu_e_collision_jax(
    f_nu: jnp.ndarray,
    q_nodes: jnp.ndarray,
    T_MeV: jnp.ndarray,
    *,
    y3_nodes: jnp.ndarray,
    y3_weights: jnp.ndarray,
    y2_nodes: jnp.ndarray,
    y2_weights: jnp.ndarray,
    GL2_GR2: float,
    GL2_mGR2: float,
) -> jnp.ndarray:
    """JIT-compatible ``nu + e -> nu + e`` collision integral.

    Mirrors ``NuEScatteringOperator._evaluate_vectorized``:

    * The y3 (outgoing neutrino) integral uses Gauss-Laguerre on the
      precomputed ``y3_nodes`` / ``y3_weights``.
    * The y2 (incoming electron) integral uses Gauss-Laguerre on the
      precomputed ``y2_nodes`` / ``y2_weights`` with thermal FD weight
      ``f_eq(y_2)``.
    * The input ``f_nu`` is interpolated to ``y3_nodes`` via PCHIP cubic
      Hermite when the grids differ; when ``y3_nodes == q_nodes`` the
      interpolation is identity at the nodes.

    The kernel works in dimensionless ``y = p / T_MeV`` units; both
    the input ``f_nu`` and the electron FD inside the integrand
    share that single ``y``-axis.  When ``T_e ≠ T_ν``, callers
    should pass ``T_MeV = T_e`` (the plasma / electron temperature)
    and remap ``q_nodes`` to the electron-frame via
    ``q_remapped = q_bank * T_ν / T_e`` so ``f_nu(q_remapped)`` is
    the same physical distribution evaluated at ``p / T_e``.  This
    matches the convention used by
    ``rabbit.jax.full_boltzmann_collision_preflight._evaluate_direct_kernel_core``
    when feeding SciPy.

    Returns
    -------
    ``C(q_i)`` with shape ``q_nodes.shape`` in units of ``[MeV]`` (divide
    by Hubble to convert to ``d/dN``).
    """
    f_nu = jnp.asarray(f_nu, dtype=jnp.float64)
    q_nodes = jnp.asarray(q_nodes, dtype=jnp.float64)
    T_MeV = jnp.asarray(T_MeV, dtype=jnp.float64)

    prefactor = G_F_MEV**2 * T_MeV**5 / HM_REDUCED_COLLISION_DENOMINATOR

    # Interpolate f_nu onto the y3 quadrature grid (identity if grids match).
    slopes = _pchip_slopes(q_nodes, f_nu)
    f3_raw = _hermite_interp(q_nodes, f_nu, slopes, y3_nodes)
    f3 = jnp.clip(f3_raw, 0.0, 1.0)

    f2_eq = _fermi_dirac_jax(y2_nodes)
    w3_full = y3_weights * jnp.exp(y3_nodes)  # undo Laguerre exp(-y) weight
    w2_full = y2_weights * jnp.exp(y2_nodes)

    def integrand_y3_y2(y1, f1, y3_j, f3_j, w3_j, y2_k, f2_k, w2_k):
        y4 = y1 + y2_k - y3_j
        valid = y4 > 0.0
        f4 = _fermi_dirac_jax(y4)
        M2 = _matrix_element_scat_jax(y1, y2_k, y3_j, y4, GL2_GR2, GL2_mGR2)
        S = collision_statistical_factor_jax(f1, f2_k, f3_j, f4)
        contrib = w3_j * w2_k * M2 * S
        return jnp.where(valid, contrib, 0.0)

    def per_q(y1, f1):
        # Double sum over y3 (Laguerre) and y2 (Laguerre).
        def fold_y3(y3_j, f3_j, w3_j):
            def fold_y2(y2_k, f2_k, w2_k):
                return integrand_y3_y2(y1, f1, y3_j, f3_j, w3_j, y2_k, f2_k, w2_k)

            return jnp.sum(jax.vmap(fold_y2)(y2_nodes, f2_eq, w2_full))

        total = jnp.sum(jax.vmap(fold_y3)(y3_nodes, f3, w3_full))
        return prefactor * total / jnp.maximum(y1**2, 1e-30)

    result = jax.vmap(per_q)(q_nodes, f_nu)
    # SciPy reference skips y1 < 1e-15 (returns 0 there).
    result = jnp.where(q_nodes < 1e-15, 0.0, result)
    return result


# ---------------------------------------------------------------------------
# Pair annihilation / creation
# ---------------------------------------------------------------------------


def pair_collision_jax(
    f_nu: jnp.ndarray,
    f_nubar: jnp.ndarray,
    q_nodes: jnp.ndarray,
    T_MeV: jnp.ndarray,
    *,
    y2_nodes: jnp.ndarray,
    y2_weights: jnp.ndarray,
    leg_nodes: jnp.ndarray,
    leg_weights: jnp.ndarray,
    GL2_GR2: float,
    GL2_mGR2: float,
) -> jnp.ndarray:
    """JIT-compatible ``nu + nubar <-> e+ + e-`` collision integral.

    Mirrors ``PairProcessOperator.evaluate``:

    * y2 (antineutrino momentum) — Gauss-Laguerre.
    * y3 (outgoing positron momentum) — Gauss-Legendre on ``[0, y1+y2]``
      with mapping ``y3 = 0.5*(y1+y2)*(leg+1)``.
    * Off-node values of ``f_nubar`` are obtained via the JAX-native
      not-a-knot natural cubic spline (``cubic_spline_jax``), which
      matches ``scipy.interpolate.interp1d(kind='cubic')`` element-
      wise to ``< 1e-12``.  Boundary clamping mirrors the SciPy
      reference's ``fill_value=(f_nubar[0], 0.0)`` (extrapolation
      below ``q_nodes[0]`` returns ``f_nubar[0]``; above
      ``q_nodes[-1]`` returns ``0.0``, consistent with the FD tail
      vanishing).

    Returns ``C(q_i)`` for the ``nu`` species; pass ``f_nubar`` and
    ``f_nu`` swapped to recover the antineutrino response (the matrix
    element is symmetric under that swap).
    """
    f_nu = jnp.asarray(f_nu, dtype=jnp.float64)
    f_nubar = jnp.asarray(f_nubar, dtype=jnp.float64)
    q_nodes = jnp.asarray(q_nodes, dtype=jnp.float64)
    T_MeV = jnp.asarray(T_MeV, dtype=jnp.float64)

    prefactor = G_F_MEV**2 * T_MeV**5 / HM_REDUCED_COLLISION_DENOMINATOR

    f2_at_y2 = cubic_spline_eval_with_clamp(
        q_nodes, f_nubar, y2_nodes,
        fill_low=f_nubar[0],
        fill_high=jnp.asarray(0.0, dtype=jnp.float64),
    )
    f2_at_y2 = jnp.clip(f2_at_y2, 0.0, 1.0)

    w2_full = y2_weights * jnp.exp(y2_nodes)

    def per_q(y1, f1):
        def fold_y2(y2_k, f2_k, w2_k):
            y_sum = y1 + y2_k
            valid_sum = y_sum > 1e-15
            half_sum = 0.5 * y_sum
            y3_local = half_sum * (leg_nodes + 1.0)  # (N_quad,)
            y3_local_w = half_sum * leg_weights      # (N_quad,)

            def fold_y3(y3_m, w3_m):
                y4 = y_sum - y3_m
                f3 = _fermi_dirac_jax(y3_m)
                f4 = _fermi_dirac_jax(y4)
                M2 = _matrix_element_ann_jax(y1, y2_k, y3_m, y4, GL2_GR2, GL2_mGR2)
                S = collision_statistical_factor_jax(f1, f2_k, f3, f4)
                return w3_m * M2 * S

            inner = jnp.sum(jax.vmap(fold_y3)(y3_local, y3_local_w))
            return jnp.where(valid_sum, w2_k * inner, 0.0)

        total = jnp.sum(jax.vmap(fold_y2)(y2_nodes, f2_at_y2, w2_full))
        return prefactor * total / jnp.maximum(y1**2, 1e-30)

    # f_nu is evaluated only at q_nodes inside the per-q closure (no
    # off-node interpolation needed for the per-q1 outer loop); only
    # f_nubar requires off-grid evaluation at y2_nodes, handled above.
    result = jax.vmap(per_q)(q_nodes, f_nu)
    result = jnp.where(q_nodes < 1e-15, 0.0, result)
    return result


# ---------------------------------------------------------------------------
# Factory bundles aligned with the SciPy module's species couplings
# ---------------------------------------------------------------------------


def nu_e_couplings(species: str = "nue") -> tuple[float, float, float, float]:
    """Return ``(G_L, G_R, GL2_GR2, GL2_mGR2)`` for ``nue``/``nux``."""
    sp = species.strip().lower()
    if sp == "nue":
        gl, gr = G_L_NUE, G_R_NUE
    elif sp == "nux":
        gl, gr = G_L_NUX, G_R_NUX
    else:
        raise ValueError(f"unknown species: {species!r}")
    return gl, gr, gl**2 + gr**2, gl**2 - gr**2


def make_nu_e_kernel(
    *,
    species: str = "nue",
    N_q: int,
    N_int: int | None = None,
    N_quad_electron: int = 32,
):
    """Build a JIT-compiled ``nu-e`` kernel + concrete quadrature inputs.

    Returns a triple ``(kernel, aux)`` where ``aux`` is a dict with the
    precomputed ``y3_nodes``, ``y3_weights``, ``y2_nodes`` and
    ``y2_weights`` arrays so callers can pass them to
    ``nu_e_collision_jax`` without re-querying the cached grids.
    """
    if N_int is None:
        N_int = min(int(N_q), 24)

    _, _, GL2_GR2, GL2_mGR2 = nu_e_couplings(species)
    y3_nodes, y3_weights = laguerre_grid_jax(int(N_int))
    y2_nodes, y2_weights = laguerre_grid_jax(int(N_quad_electron))

    @jax.jit
    def kernel(
        f_nu: jnp.ndarray,
        q_nodes: jnp.ndarray,
        T_MeV: jnp.ndarray,
    ) -> jnp.ndarray:
        return nu_e_collision_jax(
            f_nu,
            q_nodes,
            T_MeV,
            y3_nodes=y3_nodes,
            y3_weights=y3_weights,
            y2_nodes=y2_nodes,
            y2_weights=y2_weights,
            GL2_GR2=GL2_GR2,
            GL2_mGR2=GL2_mGR2,
        )

    aux = {
        "y3_nodes": y3_nodes,
        "y3_weights": y3_weights,
        "y2_nodes": y2_nodes,
        "y2_weights": y2_weights,
        "GL2_GR2": GL2_GR2,
        "GL2_mGR2": GL2_mGR2,
        "species": species,
        "N_int": int(N_int),
        "N_quad_electron": int(N_quad_electron),
    }
    return kernel, aux


def make_pair_kernel(
    *,
    species: str = "nue",
    N_quad: int = 24,
):
    """Build a JIT-compiled pair-process kernel + concrete quadrature inputs."""
    _, _, GL2_GR2, GL2_mGR2 = nu_e_couplings(species)
    y2_nodes, y2_weights = laguerre_grid_jax(int(N_quad))
    leg_nodes, leg_weights = legendre_grid_jax(int(N_quad))

    @jax.jit
    def kernel(
        f_nu: jnp.ndarray,
        f_nubar: jnp.ndarray,
        q_nodes: jnp.ndarray,
        T_MeV: jnp.ndarray,
    ) -> jnp.ndarray:
        return pair_collision_jax(
            f_nu,
            f_nubar,
            q_nodes,
            T_MeV,
            y2_nodes=y2_nodes,
            y2_weights=y2_weights,
            leg_nodes=leg_nodes,
            leg_weights=leg_weights,
            GL2_GR2=GL2_GR2,
            GL2_mGR2=GL2_mGR2,
        )

    aux = {
        "y2_nodes": y2_nodes,
        "y2_weights": y2_weights,
        "leg_nodes": leg_nodes,
        "leg_weights": leg_weights,
        "GL2_GR2": GL2_GR2,
        "GL2_mGR2": GL2_mGR2,
        "species": species,
        "N_quad": int(N_quad),
    }
    return kernel, aux


__all__ = [
    "laguerre_grid",
    "laguerre_grid_jax",
    "legendre_grid",
    "legendre_grid_jax",
    "collision_statistical_factor_jax",
    "nu_e_collision_jax",
    "pair_collision_jax",
    "nu_e_couplings",
    "make_nu_e_kernel",
    "make_pair_kernel",
]
