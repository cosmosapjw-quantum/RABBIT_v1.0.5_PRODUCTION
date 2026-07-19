"""JAX-native diagonal nu-nu scattering preflight (PR-T3C-PF).

Structural skeleton for the Standard-Model diagonal ν–ν elastic
scattering operator ``ν_α + ν_β -> ν_α + ν_β`` (Fierz-diagonal).
The kernel mirrors the algebraic structure used by the
Hannestad-Madsen ν-e port in :mod:`rabbit.jax.collisions_jax` but
replaces the electron equilibrium FD inside the integrand with the
neutrino distributions of the partner species, and absorbs the
species mixing prefactor into a Fierz-aware coefficient
``epsilon_alpha_beta`` (= ``2`` for identical species, ``1`` for
distinguishable species).

The matrix element is taken in the angular-averaged
``GL2_GR2 * ((y1*y2)**2 + (y3*y4)**2)`` form for transparent
detailed-balance and energy-conservation invariants:

* ``S`` is evaluated as the quartic-cancelled six-monomial Pauli polynomial
  equivalent to ``f_3 f_4 (1 - f_1)(1 - f_2) - f_1 f_2 (1 - f_3)(1 - f_4)``.
  Together with energy conservation ``y_4 = y_1 + y_2 - y_3``, this produces
  ``S = 0`` whenever ``f_1, f_2, f_3, f_4`` are all FD at the same temperature.
  Detailed balance therefore collapses algebraically to floating-point
  reduction order.
* Energy conservation ``int y_1^3 [C_alpha + C_beta](y_1) dy_1 = 0``
  follows from the symmetry of the integrand under
  ``(y_1, y_2) <-> (y_3, y_4)`` summed over species.

The absolute numerical normalisation of ``C_νν`` (and hence of the
``N_eff`` shift it produces against Mangano 2005) is **not** locked
in this preflight: the matrix-element prefactor used here is the
ν-e ``(GL² + GR²)`` form rather than the Dolgov-Hansen-Semikoz 1997
appendix-A coefficient table.  Closing that calibration is part of
the follow-up PR-T3C runtime patch and is tracked in
``docs/audit/PR-T3C_preflight.md``.

This module does **not** wire ν-ν into any driver.  Use
:func:`make_nu_nu_kernel` to obtain a JIT-compiled per-species
collision evaluator and pass the gathered bank monopoles in.
"""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from rabbit.collisions.kernels import G_F_MEV, HM_REDUCED_COLLISION_DENOMINATOR
from rabbit.jax.collisions_jax import (
    collision_statistical_factor_jax,
    laguerre_grid as _laguerre_grid_np,
    laguerre_grid_jax as _laguerre_grid_jax,
)
from rabbit.jax.cubic_spline_jax import cubic_spline_eval_with_clamp

jax.config.update("jax_enable_x64", True)


def nu_nu_diagonal_collision_jax(
    f_alpha: jnp.ndarray,
    f_beta: jnp.ndarray,
    q_nodes: jnp.ndarray,
    T_MeV: jnp.ndarray,
    *,
    y3_nodes: jnp.ndarray,
    y3_weights: jnp.ndarray,
    y2_nodes: jnp.ndarray,
    y2_weights: jnp.ndarray,
    epsilon_alpha_beta: float = 1.0,
    matrix_coeff: float = 1.0,
) -> jnp.ndarray:
    """Diagonal ``ν_α + ν_β -> ν_α + ν_β`` collision integral on q_nodes.

    Returns ``C_α(q_i)`` in units of ``[MeV]`` (divide by Hubble to
    convert to ``d/dN``).  Pass ``f_alpha == f_beta`` and
    ``epsilon_alpha_beta = 2`` for identical-species channels;
    pass ``epsilon_alpha_beta = 1`` for distinguishable pairs.

    Structurally mirrors :func:`rabbit.jax.collisions_jax.nu_e_collision_jax`
    with the electron Fermi-Dirac at ``y_2`` replaced by the partner
    neutrino distribution ``f_β`` evaluated at ``y_2`` via the
    JAX-native not-a-knot natural cubic spline
    (``cubic_spline_jax``), matching ``scipy.interpolate.interp1d
    (kind='cubic')`` element-wise to ``< 1e-12``.  Boundary clamping
    mirrors scipy ``interp1d``'s ``fill_value=(f[0], 0.0)`` (below
    ``q_nodes[0]`` returns ``f[0]``; above ``q_nodes[-1]`` returns
    ``0``, consistent with the FD tail vanishing).
    """
    f_alpha = jnp.asarray(f_alpha, dtype=jnp.float64)
    f_beta = jnp.asarray(f_beta, dtype=jnp.float64)
    q_nodes = jnp.asarray(q_nodes, dtype=jnp.float64)
    T_MeV = jnp.asarray(T_MeV, dtype=jnp.float64)

    prefactor = (
        float(epsilon_alpha_beta)
        * float(matrix_coeff)
        * G_F_MEV**2
        * T_MeV**5
        / HM_REDUCED_COLLISION_DENOMINATOR
    )

    f3_at_y3 = jnp.clip(
        cubic_spline_eval_with_clamp(
            q_nodes, f_alpha, y3_nodes,
            fill_low=f_alpha[0],
            fill_high=jnp.asarray(0.0, dtype=jnp.float64),
        ),
        0.0,
        1.0,
    )
    f2_at_y2 = jnp.clip(
        cubic_spline_eval_with_clamp(
            q_nodes, f_beta, y2_nodes,
            fill_low=f_beta[0],
            fill_high=jnp.asarray(0.0, dtype=jnp.float64),
        ),
        0.0,
        1.0,
    )

    w3_full = y3_weights * jnp.exp(y3_nodes)
    w2_full = y2_weights * jnp.exp(y2_nodes)

    def per_q(y1, f1):
        def fold_y3(y3_j, f3_j, w3_j):
            def fold_y2(y2_k, f2_k, w2_k):
                y4 = y1 + y2_k - y3_j
                valid = y4 > 0.0
                # f_4 is the partner species at y_4 — evaluate via the
                # JAX-native cubic spline (matches scipy interp1d cubic
                # at 1e-12 element-wise).  This evaluation is genuinely
                # off-grid for generic ``(y_1, y_2, y_3)`` because
                # ``y_4 = y_1 + y_2 - y_3`` does not coincide with the
                # input Laguerre grid; the spline is the load-bearing
                # interpolant for the diagonal nu-nu detailed-balance
                # and energy-conservation invariants.
                f4_raw = cubic_spline_eval_with_clamp(
                    q_nodes, f_beta, y4,
                    fill_low=f_beta[0],
                    fill_high=jnp.asarray(0.0, dtype=jnp.float64),
                )
                f4 = jnp.clip(f4_raw, 0.0, 1.0)
                # Hannestad-Madsen-style angular-averaged matrix element:
                # ``(y1*y2)^2 + (y3*y4)^2``.  Symmetric under
                # ``(1,2) <-> (3,4)``, so it preserves detailed balance
                # algebraically.
                M2 = (y1 * y2_k) ** 2 + (y3_j * y4) ** 2
                S = collision_statistical_factor_jax(f1, f2_k, f3_j, f4)
                contrib = w3_j * w2_k * M2 * S
                return jnp.where(valid, contrib, 0.0)

            return jnp.sum(jax.vmap(fold_y2)(y2_nodes, f2_at_y2, w2_full))

        total = jnp.sum(jax.vmap(fold_y3)(y3_nodes, f3_at_y3, w3_full))
        return prefactor * total / jnp.maximum(y1**2, 1e-30)

    result = jax.vmap(per_q)(q_nodes, f_alpha)
    result = jnp.where(q_nodes < 1e-15, 0.0, result)
    return result


def _solve_not_a_knot_M_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """NumPy mirror of ``_solve_not_a_knot_M`` for static tensor packs."""

    n = int(x.shape[0])
    h = x[1:] - x[:-1]
    s = (y[1:] - y[:-1]) / h
    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    A[0, 0] = -h[1]
    A[0, 1] = h[0] + h[1]
    A[0, 2] = -h[0]
    A[n - 1, n - 3] = -h[n - 2]
    A[n - 1, n - 2] = h[n - 3] + h[n - 2]
    A[n - 1, n - 1] = -h[n - 3]
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1] / 6.0
        A[i, i] = (h[i - 1] + h[i]) / 3.0
        A[i, i + 1] = h[i] / 6.0
        b[i] = s[i] - s[i - 1]
    return np.linalg.solve(A, b)


def _cubic_eval_with_clamp_np(
    x: np.ndarray,
    y: np.ndarray,
    x_query: np.ndarray,
) -> np.ndarray:
    M = _solve_not_a_knot_M_np(x, y)
    flat = np.ravel(x_query).astype(np.float64)
    idx = np.clip(np.searchsorted(x, flat, side="right") - 1, 0, x.shape[0] - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    M0 = M[idx]
    M1 = M[idx + 1]
    h_seg = np.where(x1 - x0 == 0.0, 1.0, x1 - x0)
    A_coef = (x1 - flat) / h_seg
    B_coef = 1.0 - A_coef
    values = (
        A_coef * y0
        + B_coef * y1
        + ((A_coef**3 - A_coef) * M0 + (B_coef**3 - B_coef) * M1)
        * (h_seg**2)
        / 6.0
    )
    values = np.where(flat < x[0], y[0], values)
    values = np.where(flat > x[-1], 0.0, values)
    return values.reshape(np.shape(x_query))


def _cubic_clamp_interpolation_matrix_np(
    q_nodes: np.ndarray,
    x_query: np.ndarray,
) -> np.ndarray:
    """Return the linear map implementing the existing clamped cubic spline.

    ``cubic_spline_eval_with_clamp(q_nodes, f, x_query, f[0], 0)`` is linear
    in ``f``.  Building the matrix once removes the runtime spline solve from
    the ν-ν spectral collision core without changing the interpolation
    contract.
    """

    q_nodes = np.asarray(q_nodes, dtype=np.float64)
    x_query = np.asarray(x_query, dtype=np.float64)
    basis = np.eye(int(q_nodes.shape[0]), dtype=np.float64)
    values_by_basis = np.stack(
        [_cubic_eval_with_clamp_np(q_nodes, row, x_query) for row in basis],
        axis=-1,
    )
    return values_by_basis


@lru_cache(maxsize=16)
def _cached_laguerre_tensor_pack(
    N_q: int,
    epsilon_alpha_beta: float = 1.0,
    matrix_coeff: float = 1.0,
) -> dict[str, np.ndarray]:
    """Static tensor pack for the fixed Laguerre ν-ν spectral quadrature."""

    q_np, w_np = _laguerre_grid_np(int(N_q))
    q_nodes_np = np.asarray(q_np, dtype=np.float64)
    q_weights_np = np.asarray(w_np, dtype=np.float64)
    y1 = q_nodes_np[:, None, None]
    y2 = q_nodes_np[None, :, None]
    y3 = q_nodes_np[None, None, :]
    y4 = y1 + y2 - y3
    valid = y4 > 0.0
    w2_full = q_weights_np * np.exp(q_nodes_np)
    w3_full = q_weights_np * np.exp(q_nodes_np)
    M2 = (y1 * y2) ** 2 + (y3 * y4) ** 2
    weighted_m2 = (
        np.where(valid, 1.0, 0.0)
        * w2_full[None, :, None]
        * w3_full[None, None, :]
        * M2
        / np.maximum(y1**2, 1.0e-30)
    )
    weighted_m2 = np.where(q_nodes_np[:, None, None] < 1.0e-15, 0.0, weighted_m2)
    interp_y4 = _cubic_clamp_interpolation_matrix_np(q_nodes_np, y4)
    prefactor_scale = (
        float(epsilon_alpha_beta)
        * float(matrix_coeff)
        * G_F_MEV**2
        / HM_REDUCED_COLLISION_DENOMINATOR
    )
    return {
        "q_nodes": q_nodes_np,
        "weighted_m2": weighted_m2,
        "interp_y4": interp_y4,
        "prefactor_scale": np.asarray(prefactor_scale, dtype=np.float64),
        "N_q": np.asarray(int(N_q), dtype=np.int32),
    }


def nu_nu_diagonal_collision_tensorized_jax(
    f_alpha: jnp.ndarray,
    f_beta: jnp.ndarray,
    q_nodes: jnp.ndarray,
    T_MeV: jnp.ndarray,
    *,
    tensor_pack: dict[str, jnp.ndarray] | None = None,
    epsilon_alpha_beta: float = 1.0,
    matrix_coeff: float = 1.0,
) -> jnp.ndarray:
    """Tensorized equivalent of :func:`nu_nu_diagonal_collision_jax`.

    The expensive off-grid ``f_beta(y4)`` cubic-spline evaluations are replaced
    by a cached interpolation matrix, then the ``(y1, y2, y3)`` quadrature is
    evaluated as a dense tensor contraction.  The function remains pure JAX and
    differentiable; no host callback or PyO3 path is used.

    This accelerated path is intentionally scoped to the package's fixed
    Laguerre momentum grid.  Passing a different grid shape reuses the matching
    Laguerre tensor pack by ``N_q`` and is not a generic interpolation backend.
    """

    f_alpha = jnp.asarray(f_alpha, dtype=jnp.float64)
    f_beta = jnp.asarray(f_beta, dtype=jnp.float64)
    q_nodes = jnp.asarray(q_nodes, dtype=jnp.float64)
    T_MeV = jnp.asarray(T_MeV, dtype=jnp.float64)
    if tensor_pack is None:
        tensor_pack = _cached_laguerre_tensor_pack(
            int(q_nodes.shape[0]),
            float(epsilon_alpha_beta),
            float(matrix_coeff),
        )
    weighted_m2 = jnp.asarray(tensor_pack["weighted_m2"], dtype=jnp.float64)
    interp_y4 = jnp.asarray(tensor_pack["interp_y4"], dtype=jnp.float64)
    prefactor_scale = jnp.asarray(tensor_pack["prefactor_scale"], dtype=jnp.float64)

    f_alpha_clip = jnp.clip(f_alpha, 0.0, 1.0)
    f_beta_clip = jnp.clip(f_beta, 0.0, 1.0)
    f1 = f_alpha[:, None, None]
    f2 = f_beta_clip[None, :, None]
    f3 = f_alpha_clip[None, None, :]
    f4 = jnp.clip(jnp.einsum("ijkm,m->ijk", interp_y4, f_beta), 0.0, 1.0)
    total = jnp.sum(weighted_m2 * collision_statistical_factor_jax(f1, f2, f3, f4), axis=(1, 2))
    result = prefactor_scale * T_MeV**5 * total
    return jnp.where(q_nodes < 1.0e-15, 0.0, result)


def make_nu_nu_kernel(
    *,
    N_q: int,
    epsilon_alpha_beta: float = 1.0,
    matrix_coeff: float = 1.0,
):
    """Build a JIT-compiled diagonal ν-ν kernel + cached quadrature inputs.

    The y3 and y2 grids are both ``laguerre(N_q)`` (matched to the
    expected ``q_nodes = laguerre(N_q)`` so PCHIP collapses to
    identity at the input nodes and detailed balance holds
    algebraically).
    """
    y3_nodes, y3_weights = _laguerre_grid_jax(int(N_q))
    y2_nodes, y2_weights = _laguerre_grid_jax(int(N_q))

    @jax.jit
    def kernel(
        f_alpha: jnp.ndarray,
        f_beta: jnp.ndarray,
        q_nodes: jnp.ndarray,
        T_MeV: jnp.ndarray,
    ) -> jnp.ndarray:
        return nu_nu_diagonal_collision_jax(
            f_alpha,
            f_beta,
            q_nodes,
            T_MeV,
            y3_nodes=y3_nodes,
            y3_weights=y3_weights,
            y2_nodes=y2_nodes,
            y2_weights=y2_weights,
            epsilon_alpha_beta=float(epsilon_alpha_beta),
            matrix_coeff=float(matrix_coeff),
        )

    aux = {
        "y3_nodes": y3_nodes,
        "y3_weights": y3_weights,
        "y2_nodes": y2_nodes,
        "y2_weights": y2_weights,
        "epsilon_alpha_beta": float(epsilon_alpha_beta),
        "matrix_coeff": float(matrix_coeff),
        "N_q": int(N_q),
    }
    return kernel, aux


def make_nu_nu_tensorized_kernel(
    *,
    N_q: int,
    epsilon_alpha_beta: float = 1.0,
    matrix_coeff: float = 1.0,
):
    """Build a JIT-compiled tensorized diagonal ν-ν kernel."""

    tensor_pack = _cached_laguerre_tensor_pack(
        int(N_q),
        float(epsilon_alpha_beta),
        float(matrix_coeff),
    )
    q_nodes = jnp.asarray(tensor_pack["q_nodes"], dtype=jnp.float64)

    @jax.jit
    def kernel(
        f_alpha: jnp.ndarray,
        f_beta: jnp.ndarray,
        _q_nodes: jnp.ndarray,
        T_MeV: jnp.ndarray,
    ) -> jnp.ndarray:
        return nu_nu_diagonal_collision_tensorized_jax(
            f_alpha,
            f_beta,
            q_nodes,
            T_MeV,
            tensor_pack=tensor_pack,
        )

    aux = {
        "N_q": int(N_q),
        "epsilon_alpha_beta": float(epsilon_alpha_beta),
        "matrix_coeff": float(matrix_coeff),
        "implementation": "tensorized_static_kernel_v1",
        "grid_contract": "fixed_laguerre_grid_by_N_q",
    }
    return kernel, aux


__all__ = [
    "nu_nu_diagonal_collision_jax",
    "nu_nu_diagonal_collision_tensorized_jax",
    "make_nu_nu_kernel",
    "make_nu_nu_tensorized_kernel",
]
