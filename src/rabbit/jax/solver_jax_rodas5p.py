"""
rabbit.jax.solver_jax_rodas5p — JAX-native Rodas5P adaptive ODE solver.

The namesake "R" in RABBIT, now JIT-compiled.

Architecture:
    lax.while_loop(cond, body, state)
    ├── state = (N, y, h, err_prev, n_steps, n_reject, status)
    ├── body: one adaptive step (Jacobian + 8 stages + error + controller)
    └── cond: N < N_final AND n_steps < max_steps AND status == 0

Jacobian: jax.jacfwd(rhs, argnums=1) — exact forward-mode AD.
Linear solve: jnp.linalg.solve (dense, N=17 is small enough).
Controller: Gustafsson predictor (safety=0.9, β=0.04).

CRITICAL: btilde = [0,...,0,1] stores error weights directly.
          error = Σ btilde·k = k₈ (last stage vector).
          Wrong interpretation (b-bhat) gives error norm ~10⁷.

Convention: SciML
    W = I/(γh) − J
    rhs_stage = f(t + c_i h, y_stage) + Σ C·k / h + h d_i f_t
    y_new = y + Σ b·k        (no h factor on b)

References:
    Steinebach (2023), BIT Numer Math 63:27 — Rodas5P coefficients
    Gustafsson (1991), ACM TOMS 17:533 — predictive controller
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import weakref
import numpy as np
from rabbit.jax._cache_guard import _CACHE_LOCK, register_cache_clearer

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.scipy.linalg
from functools import partial

jax.config.update("jax_enable_x64", True)


_JACFWD_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_SPARSE_JAC_CACHE: "dict[tuple, Callable]" = {}
_SOLVE_RUNNER_CACHE: "dict[tuple, Callable]" = {}
_BATCH_SOLVE_RUNNER_CACHE: "dict[tuple, Callable]" = {}
_BATCH_LINEAR_SOLVER_BACKENDS = frozenset(
    ("auto", "jax_tensorized", "gpu_tensorized")
)

def _clear_solver_caches():
    _SPARSE_JAC_CACHE.clear()
    _SOLVE_RUNNER_CACHE.clear()
    _BATCH_SOLVE_RUNNER_CACHE.clear()

register_cache_clearer(_clear_solver_caches)


def _cached_jacfwd(rhs_fn: Callable) -> Callable:
    with _CACHE_LOCK:
        cached = _JACFWD_CACHE.get(rhs_fn)
        if cached is not None:
            return cached
        jac_fn = jax.jacfwd(rhs_fn, argnums=1)
        _JACFWD_CACHE[rhs_fn] = jac_fn
    return jac_fn


# ═══════════════════════════════════════════════════════════════════════
# Block-sparse Jacobian for systems with a large "passive" transport sector.
#
# State vector: y = [active_head | passive(transport) | active_tail]
#   active indices: geometry (Σ) + thermo (T) + network (X)
#   passive indices: transport hierarchy Ψ (collisionless, no self-coupling)
#
# Jacobian structure (Type I collisionless):
#   J[passive, passive] = 0  (exact: no Ψ-Ψ coupling)
#   J[passive, active]  = constant source (known analytically by RHS structure)
#   J[active,  active]  = full AD (small block, n_active forward passes)
#   J[active,  passive] = chain-rule through projectors (n_active reverse passes)
#
# Cost: 2 × n_active RHS-equivalent passes instead of D passes.
#   At N_q=20: 14 passes instead of 247 → ~17× Jacobian speedup.
# ═══════════════════════════════════════════════════════════════════════

def build_block_sparse_jac_fn(
    rhs_fn: Callable,
    active_indices: jnp.ndarray,
    state_dim: int,
) -> Callable:
    """Build a block-sparse Jacobian function exploiting transport sector sparsity.

    Parameters
    ----------
    rhs_fn : Callable
        ODE right-hand side f(N, y) → dy/dN.
    active_indices : array of int
        Indices of "active" state variables (geometry + thermo + network).
        All other indices are "passive" (transport hierarchy).
    state_dim : int
        Total state dimension D.

    Returns
    -------
    jac_fn : Callable
        Function (N, y) → J of shape (D, D), computing only non-trivial blocks.
    """
    active_idx = jnp.asarray(active_indices, dtype=jnp.int32)
    n_active = active_idx.shape[0]
    all_idx = jnp.arange(state_dim, dtype=jnp.int32)
    passive_mask = jnp.ones(state_dim, dtype=bool).at[active_idx].set(False)
    passive_idx = all_idx[passive_mask]
    n_passive = passive_idx.shape[0]

    # Pre-build tangent basis for active columns (forward mode)
    active_tangents = jnp.zeros((n_active, state_dim), dtype=jnp.float64)
    active_tangents = active_tangents.at[jnp.arange(n_active), active_idx].set(1.0)

    # Pre-build cotangent basis for active rows (reverse mode)
    active_cotangents = jnp.zeros((n_active, state_dim), dtype=jnp.float64)
    active_cotangents = active_cotangents.at[jnp.arange(n_active), active_idx].set(1.0)

    @jax.jit
    def block_sparse_jac(N, y):
        f_of_y = lambda y_: rhs_fn(N, y_)

        # Reuse one linearization / one pullback instead of rebuilding the AD
        # transform inside every vmapped column/row probe.
        _, lin_f = jax.linearize(f_of_y, y)
        active_cols = jax.vmap(lin_f)(active_tangents)  # (n_active, D)

        _, vjp_f = jax.vjp(f_of_y, y)
        passive_grads = jax.vmap(lambda cotangent: vjp_f(cotangent)[0])(active_cotangents)
        active_rows_passive = passive_grads[:, passive_idx]  # (n_active, n_passive)

        # Assemble dense Jacobian
        J = jnp.zeros((state_dim, state_dim), dtype=y.dtype)
        J = J.at[:, active_idx].set(active_cols.T)
        J = J.at[jnp.ix_(active_idx, passive_idx)].set(active_rows_passive)
        # J[passive, passive] = 0 (Type I collisionless: no Ψ-Ψ coupling)
        return J

    return block_sparse_jac


def _get_block_sparse_jac_fn(rhs_fn: Callable, active_indices, state_dim: int) -> Callable:
    """Cached version of build_block_sparse_jac_fn."""
    key = (id(rhs_fn), tuple(int(x) for x in active_indices), int(state_dim))
    cached = _SPARSE_JAC_CACHE.get(key)
    if cached is not None:
        return cached
    jac_fn = build_block_sparse_jac_fn(rhs_fn, jnp.array(active_indices), state_dim)
    _SPARSE_JAC_CACHE[key] = jac_fn
    return jac_fn


# ═══════════════════════════════════════════════════════════════════════
# Low-rank Jacobian preflight utilities for future tier-3 transport.
#
# Planned full-collision structure:
#   J = J_base + U @ C @ V
#
# with
#   J_base : q-advection band block + scalar couplings
#   U, V   : fixed gather/apply maps between ray space and moment space
#   C      : small moment-space collision core Jacobian
#
# The full tier-3 transport block is dense if materialized on rays, but
# low-rank through the moment-space core. These helpers let us test the
# Woodbury solve surface without yet changing the public solve contract.
# ═══════════════════════════════════════════════════════════════════════

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LowRankJacobianFactors:
    """Factorized Jacobian representation J = J_base + U @ C @ V."""

    base_jacobian: jnp.ndarray
    left_factor: jnp.ndarray
    core_matrix: jnp.ndarray
    right_factor: jnp.ndarray


def materialize_low_rank_jacobian(factors: LowRankJacobianFactors) -> jnp.ndarray:
    """Materialize a factorized Jacobian for diagnostics or testing."""
    return (
        factors.base_jacobian
        + factors.left_factor @ factors.core_matrix @ factors.right_factor
    )


def prepare_low_rank_linear_solver(
    N_val: jnp.ndarray,
    y: jnp.ndarray,
    h: jnp.ndarray,
    factors: LowRankJacobianFactors,
):
    """Prepare a Woodbury linear solve state for ``W = I/(γh) - J``."""
    del N_val, y
    base_matrix = (
        jnp.eye(factors.base_jacobian.shape[0], dtype=factors.base_jacobian.dtype) / (GAMMA * h)
        - factors.base_jacobian
    )
    lu, piv = jax.scipy.linalg.lu_factor(base_matrix)
    base_left = jax.scipy.linalg.lu_solve((lu, piv), factors.left_factor)
    update_core = -factors.core_matrix
    condensed = (
        jnp.eye(update_core.shape[0], dtype=base_matrix.dtype)
        + update_core @ (factors.right_factor @ base_left)
    )
    condensed_lu, condensed_piv = jax.scipy.linalg.lu_factor(condensed)
    return (
        lu,
        piv,
        base_left,
        condensed_lu,
        condensed_piv,
        update_core,
        factors.right_factor,
    )


def solve_prepared_low_rank_linear_system(prepared_state, rhs: jnp.ndarray) -> jnp.ndarray:
    """Apply a prepared low-rank Woodbury solve to one RHS vector."""
    lu, piv, base_left, condensed_lu, condensed_piv, update_core, right_factor = prepared_state
    base_rhs = jax.scipy.linalg.lu_solve((lu, piv), rhs)
    correction_rhs = update_core @ (right_factor @ base_rhs)
    z = jax.scipy.linalg.lu_solve((condensed_lu, condensed_piv), correction_rhs)
    return base_rhs - base_left @ z


def _canonical_batch_linear_solver_backend(name: str | None) -> str:
    """Normalize the public batch linear-solver backend name."""

    key = "auto" if name is None else str(name).strip().lower()
    aliases = {
        "jax": "jax_tensorized",
        "jax_dense": "jax_tensorized",
        "tensorized": "jax_tensorized",
        "xla": "jax_tensorized",
        "gpu": "gpu_tensorized",
        "jax_gpu": "gpu_tensorized",
        "xla_gpu": "gpu_tensorized",
    }
    key = aliases.get(key, key)
    if key not in _BATCH_LINEAR_SOLVER_BACKENDS:
        allowed = "', '".join(sorted(_BATCH_LINEAR_SOLVER_BACKENDS | frozenset(aliases)))
        raise ValueError(
            f"unknown batch_linear_solver_backend={name!r}; choose '{allowed}'"
        )
    return key


def _array_device_platform(value) -> str:
    """Best-effort platform name for a JAX array."""

    try:
        devices = value.devices()
    except Exception:
        devices = None
    if devices:
        try:
            return str(next(iter(devices)).platform)
        except Exception:
            pass
    try:
        return str(value.device.platform)
    except Exception:
        return ""


def _resolve_batch_linear_solver_backend(
    name: str | None,
    *,
    platform: str | None = None,
) -> str:
    """Choose the effective batch dense linear solver.

    ``jax_tensorized`` is the portable XLA path. ``gpu_tensorized`` is the
    same XLA batched dense solve, but requires a GPU/CUDA/ROCm device and is
    reported explicitly as the GPU implementation. ``auto`` selects
    ``gpu_tensorized`` for GPU arrays and ``jax_tensorized`` otherwise.
    """

    requested = _canonical_batch_linear_solver_backend(name)
    active_platform = str(platform or "").lower()
    if requested == "auto":
        if not active_platform:
            try:
                active_platform = str(jax.default_backend()).lower()
            except Exception:
                active_platform = "unknown"
        if active_platform in {"gpu", "cuda", "rocm"}:
            return "gpu_tensorized"
        return "jax_tensorized"
    if requested == "gpu_tensorized":
        if not active_platform:
            try:
                active_platform = str(jax.default_backend()).lower()
            except Exception:
                active_platform = "unknown"
        if active_platform not in {"gpu", "cuda", "rocm"}:
            raise ValueError(
                "batch_linear_solver_backend='gpu_tensorized' requires a "
                "JAX GPU/CUDA/ROCm device. Use 'jax_tensorized' for a portable "
                "XLA path or 'auto' for device-aware dispatch."
            )
        return requested
    return requested


def _solve_batched_dense_jax(matrix_batch: jnp.ndarray, rhs_batch: jnp.ndarray) -> jnp.ndarray:
    """Solve ``matrix_batch @ x = rhs_batch`` using XLA batched dense solve."""

    return jnp.linalg.solve(matrix_batch, rhs_batch[..., None])[..., 0]


def _batch_dense_solve_fn(backend: str) -> Callable:
    """Return the JIT-callable batched dense solve function."""

    return _solve_batched_dense_jax


def _solve_woodbury_low_rank_update(
    base_matrix: jnp.ndarray,
    left_factor: jnp.ndarray,
    core_matrix: jnp.ndarray,
    right_factor: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """Solve (A - U C V) x = rhs via Woodbury.

    Parameters
    ----------
    base_matrix : array, shape (D, D)
        The base matrix ``A``.
    left_factor : array, shape (D, K)
        Low-rank left factor ``U``.
    core_matrix : array, shape (K, K)
        Small dense core ``C``.
    right_factor : array, shape (K, D)
        Low-rank right factor ``V``.
    rhs : array, shape (D,)
        Right-hand side.
    """
    lu, piv = jax.scipy.linalg.lu_factor(base_matrix)
    base_left = jax.scipy.linalg.lu_solve((lu, piv), left_factor)
    update_core = -core_matrix
    condensed = (
        jnp.eye(update_core.shape[0], dtype=base_matrix.dtype)
        + update_core @ (right_factor @ base_left)
    )
    condensed_lu, condensed_piv = jax.scipy.linalg.lu_factor(condensed)
    prepared_state = (
        lu,
        piv,
        base_left,
        condensed_lu,
        condensed_piv,
        update_core,
        right_factor,
    )
    return solve_prepared_low_rank_linear_system(prepared_state, rhs)


# ═══════════════════════════════════════════════════════════════════════
# §1. Rodas5P Tableau (Steinebach 2023, SciML convention)
# ═══════════════════════════════════════════════════════════════════════

GAMMA = 0.21193756319429014
ORDER = 5
_STATUS_SPAN_COMPLETE = 0
_STATUS_H_MIN = -1
_STATUS_MAX_STEPS_EXHAUSTED = -2
_STATUS_EVENT_TRIGGERED = 2
_STATUS_CONTRACT = "0=span_complete,2=event_triggered,-1=h_min,-2=max_steps_exhausted"
N_STAGES = 8

# A matrix (lower triangular, 8×8)
A = jnp.zeros((8, 8))
A = A.at[1, 0].set(3.0)
A = A.at[2, 0].set(2.849394379747939)
A = A.at[2, 1].set(0.45842242204463923)
A = A.at[3, 0].set(-6.954028509809101)
A = A.at[3, 1].set(2.489845061869568)
A = A.at[3, 2].set(-10.358996098473584)
A = A.at[4, 0].set(2.8029986275628964)
A = A.at[4, 1].set(0.5072464736228206)
A = A.at[4, 2].set(-0.3988312541770524)
A = A.at[4, 3].set(-0.04721187230404641)
A = A.at[5, 0].set(-7.502846399306121)
A = A.at[5, 1].set(2.561846144803919)
A = A.at[5, 2].set(-11.627539656261098)
A = A.at[5, 3].set(-0.18268767659942256)
A = A.at[5, 4].set(0.030198172008377946)
# Stage 6 = copy of stage 5 + A[6,5]=1
A = A.at[6, 0].set(-7.502846399306121)
A = A.at[6, 1].set(2.561846144803919)
A = A.at[6, 2].set(-11.627539656261098)
A = A.at[6, 3].set(-0.18268767659942256)
A = A.at[6, 4].set(0.030198172008377946)
A = A.at[6, 5].set(1.0)
# Stage 7 = copy of stage 5 + A[7,5]=1 + A[7,6]=1
A = A.at[7, 0].set(-7.502846399306121)
A = A.at[7, 1].set(2.561846144803919)
A = A.at[7, 2].set(-11.627539656261098)
A = A.at[7, 3].set(-0.18268767659942256)
A = A.at[7, 4].set(0.030198172008377946)
A = A.at[7, 5].set(1.0)
A = A.at[7, 6].set(1.0)

# C matrix (lower triangular, 8×8) — coupling with /h division
C = jnp.zeros((8, 8))
C = C.at[1, 0].set(-14.155112264123755)
C = C.at[2, 0].set(-17.97296035885952)
C = C.at[2, 1].set(-2.859693295451294)
C = C.at[3, 0].set(147.12150275711716)
C = C.at[3, 1].set(-1.41221402718213)
C = C.at[3, 2].set(71.68940251302358)
C = C.at[4, 0].set(165.43517024871676)
C = C.at[4, 1].set(-0.4592823456491126)
C = C.at[4, 2].set(42.90938336958603)
C = C.at[4, 3].set(-5.961986721573306)
C = C.at[5, 0].set(24.854864614690072)
C = C.at[5, 1].set(-3.0009227002832186)
C = C.at[5, 2].set(47.4931110020768)
C = C.at[5, 3].set(5.5814197821558125)
C = C.at[5, 4].set(-0.6610691825249471)
C = C.at[6, 0].set(30.91273214028599)
C = C.at[6, 1].set(-3.1208243349937974)
C = C.at[6, 2].set(77.79954646070892)
C = C.at[6, 3].set(34.28646028294783)
C = C.at[6, 4].set(-19.097331116725623)
C = C.at[6, 5].set(-28.087943162872662)
C = C.at[7, 0].set(37.80277123390563)
C = C.at[7, 1].set(-3.2571969029072276)
C = C.at[7, 2].set(112.26918849496327)
C = C.at[7, 3].set(66.9347231244047)
C = C.at[7, 4].set(-40.06618937091002)
C = C.at[7, 5].set(-54.66780262877968)
C = C.at[7, 6].set(-9.48861652309627)

# Solution weights (SciML: y_new = y + Σ b·k, no h factor)
b_weights = jnp.array([
    -7.502846399306121, 2.561846144803919, -11.627539656261098,
    -0.18268767659942256, 0.030198172008377946, 1.0, 1.0, 1.0
])

# Error weights: btilde = [0,...,0,1] → error = k₈
# CRITICAL: this IS the error weight vector, NOT b-bhat
btilde = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

# Stage-time nodes and partial-time derivative weights. These are the
# Steinebach/SciML Rodas5P values for
#
#   f(t + c_i h, y_i) + sum_j(C_ij k_j / h) + h d_i partial_t f(t_n, y_n).
#
# ``partial_t f`` is evaluated once at the attempted-step start and shared by
# all stages. Keep the public names lower-case to match upstream notation and
# the NumPy twin.
c_nodes = jnp.array([
    0.0,
    0.6358126895828704,
    0.4095798393397535,
    0.9769306725060716,
    0.4288403609558664,
    1.0,
    1.0,
    1.0,
])
d_nodes = jnp.array([
    0.21193756319429014,
    -0.42387512638858027,
    -0.3384627126235924,
    1.8046452872882734,
    2.325825639765069,
    0.0,
    0.0,
    0.0,
])


# ═══════════════════════════════════════════════════════════════════════
# §2. Error norm
# ═══════════════════════════════════════════════════════════════════════

def _err_norm(e, y, rtol, atol):
    """Weighted RMS error norm."""
    sc = atol + rtol * jnp.abs(y)
    return jnp.sqrt(jnp.mean((e / sc) ** 2))


def _err_norm_batch(e, y, rtol, atol):
    """Weighted RMS error norm for a leading batch dimension."""

    sc = atol + rtol * jnp.abs(y)
    return jnp.sqrt(jnp.mean((e / sc) ** 2, axis=1))


# ═══════════════════════════════════════════════════════════════════════
# §3. Single Rodas5P step (8 stages, SciML convention)
# ═══════════════════════════════════════════════════════════════════════

def _finite_difference_dfdt(rhs_fn, N_val, y, h):
    """Return ``partial rhs / partial N`` at fixed ``y`` via central FD."""

    dtype = y.dtype
    eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    delta = jnp.cbrt(eps) * jnp.maximum(
        one,
        jnp.maximum(jnp.abs(jnp.asarray(N_val, dtype=dtype)), jnp.abs(h)),
    )
    return (
        rhs_fn(N_val + delta, y) - rhs_fn(N_val - delta, y)
    ) / (2.0 * delta)


def _step_start_dfdt(rhs_fn, dfdt_fn, N_val, y, h):
    """Evaluate the analytic or fixed-state partial exactly once per attempt."""

    if dfdt_fn is None:
        return _finite_difference_dfdt(rhs_fn, N_val, y, h)
    return jnp.asarray(dfdt_fn(N_val, y), dtype=y.dtype)


def _step_start_dfdt_batch(
    rhs_fn,
    dfdt_fn,
    N_arr,
    y_batch,
    h_signed,
    active=None,
):
    """Vectorized step-start partial-time derivative for a solve batch."""

    if active is not None:
        def evaluate_lane(args):
            N_val, y, h, is_active = args
            return lax.cond(
                is_active,
                lambda _: _step_start_dfdt(rhs_fn, dfdt_fn, N_val, y, h),
                lambda _: jnp.zeros_like(y),
                operand=None,
            )

        return lax.cond(
            jnp.all(active),
            lambda _: _step_start_dfdt_batch(
                rhs_fn, dfdt_fn, N_arr, y_batch, h_signed
            ),
            lambda _: lax.map(
                evaluate_lane,
                (N_arr, y_batch, h_signed, active),
            ),
            operand=None,
        )

    if dfdt_fn is None:
        dtype = y_batch.dtype
        eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        delta = jnp.cbrt(eps) * jnp.maximum(
            one,
            jnp.maximum(jnp.abs(N_arr), jnp.abs(h_signed)),
        )
        return (
            jax.vmap(rhs_fn)(N_arr + delta, y_batch)
            - jax.vmap(rhs_fn)(N_arr - delta, y_batch)
        ) / (2.0 * delta[:, None])
    return jnp.asarray(jax.vmap(dfdt_fn)(N_arr, y_batch), dtype=y_batch.dtype)


def _rodas5p_step(rhs_fn, N_val, y, h, J, rtol, atol, dfdt=None):
    """Execute one Rodas5P step (dense LU)."""
    if dfdt is None:
        dfdt = _finite_difference_dfdt(rhs_fn, N_val, y, h)
    else:
        dfdt = jnp.asarray(dfdt, dtype=y.dtype)
    N_dim = y.shape[0]
    W = jnp.eye(N_dim) / (GAMMA * h) - J
    lu, piv = jax.scipy.linalg.lu_factor(W)
    k_all = jnp.zeros((N_STAGES, N_dim))

    def stage_body(i, k_all):
        y_stage = y + A[i] @ k_all
        c_sum = C[i] @ k_all / h
        rhs_vec = (
            rhs_fn(N_val + c_nodes[i] * h, y_stage)
            + c_sum
            + h * d_nodes[i] * dfdt
        )
        k_i = jax.scipy.linalg.lu_solve((lu, piv), rhs_vec)
        k_all = k_all.at[i].set(k_i)
        return k_all

    k_all = lax.fori_loop(0, N_STAGES, stage_body, k_all)
    y_new = y + b_weights @ k_all
    err_vec = btilde @ k_all
    en = _err_norm(err_vec, y_new, rtol, atol)
    return y_new, en, k_all


def _rodas5p_step_batched_dense(
    rhs_fn: Callable,
    N_arr: jnp.ndarray,
    y_batch: jnp.ndarray,
    h_signed: jnp.ndarray,
    J_batch: jnp.ndarray,
    rtol: float,
    atol: float,
    solve_batch_fn: Callable,
    dfdt=None,
):
    """Execute one tensorized dense Rodas5P step for a solve batch.

    This path is deliberately explicit instead of ``vmap(_rodas5p_step)``:
    the linear solve surface is now a real batched matrix problem.  On GPU,
    ``solve_batch_fn`` is the XLA batched dense solve.  On CPU, the same
    JAX-native tensorized surface is used.
    """

    if dfdt is None:
        dfdt = _step_start_dfdt_batch(
            rhs_fn, None, N_arr, y_batch, h_signed
        )
    else:
        dfdt = jnp.asarray(dfdt, dtype=y_batch.dtype)

    n_dim = y_batch.shape[1]
    eye = jnp.eye(n_dim, dtype=y_batch.dtype)
    W_batch = eye[None, :, :] / (GAMMA * h_signed[:, None, None]) - J_batch

    # Python-level unroll keeps the eight-stage tableau static inside XLA.
    # This avoids dynamic A/C row indexing in the nested loop and lets XLA
    # simplify the zero upper-triangular coefficients in this small-system path.
    k_rows = []
    zero_batch = jnp.zeros_like(y_batch)
    for stage in range(N_STAGES):
        if stage == 0:
            y_stage = y_batch
            c_sum = zero_batch
        else:
            k_prefix = jnp.stack(k_rows, axis=0)
            y_stage = y_batch + jnp.tensordot(A[stage, :stage], k_prefix, axes=(0, 0))
            c_sum = (
                jnp.tensordot(C[stage, :stage], k_prefix, axes=(0, 0))
                / h_signed[:, None]
            )
        rhs_batch = (
            jax.vmap(rhs_fn)(N_arr + c_nodes[stage] * h_signed, y_stage)
            + c_sum
            + h_signed[:, None] * d_nodes[stage] * dfdt
        )
        k_rows.append(solve_batch_fn(W_batch, rhs_batch))

    k_all = jnp.stack(k_rows, axis=0)
    y_new = y_batch + jnp.tensordot(b_weights, k_all, axes=(0, 0))
    err_vec = k_rows[-1]
    en = _err_norm_batch(err_vec, y_new, rtol, atol)
    return y_new, en, k_all


def _rodas5p_step_low_rank(
    rhs_fn: Callable,
    N_val: jnp.ndarray,
    y: jnp.ndarray,
    h: jnp.ndarray,
    factors: LowRankJacobianFactors,
    rtol: float,
    atol: float,
    dfdt=None,
):
    """Execute one Rodas5P step using ``J = J_base + U C V``.

    This is an experimental preflight path for future tier-3 transport.
    It avoids materializing the lifted dense collision block inside the
    stage solve by applying a Woodbury correction to the base matrix

        W_base = I/(γh) - J_base
        W      = W_base - U C V

    The current public solver does not route through this path yet; the
    function exists to validate the algebra and future hook surface.
    """
    return _rodas5p_step_custom_linear_solver(
        rhs_fn,
        N_val,
        y,
        h,
        factors,
        rtol,
        atol,
        prepare_low_rank_linear_solver,
        solve_prepared_low_rank_linear_system,
        dfdt=dfdt,
    )


def _rodas5p_step_custom_linear_solver(
    rhs_fn: Callable,
    N_val: jnp.ndarray,
    y: jnp.ndarray,
    h: jnp.ndarray,
    jacobian_payload,
    rtol: float,
    atol: float,
    linear_solver_prepare_fn: Callable,
    linear_solver_solve_fn: Callable,
    dfdt=None,
):
    """Execute one Rodas5P step using a custom prepared linear solver."""
    if dfdt is None:
        dfdt = _finite_difference_dfdt(rhs_fn, N_val, y, h)
    else:
        dfdt = jnp.asarray(dfdt, dtype=y.dtype)
    solver_state = linear_solver_prepare_fn(N_val, y, h, jacobian_payload)
    k_all = jnp.zeros((N_STAGES, y.shape[0]), dtype=y.dtype)

    def stage_body(i, k_all):
        y_stage = y + A[i] @ k_all
        c_sum = C[i] @ k_all / h
        rhs_vec = (
            rhs_fn(N_val + c_nodes[i] * h, y_stage)
            + c_sum
            + h * d_nodes[i] * dfdt
        )
        k_i = linear_solver_solve_fn(solver_state, rhs_vec)
        k_all = k_all.at[i].set(k_i)
        return k_all

    k_all = lax.fori_loop(0, N_STAGES, stage_body, k_all)
    y_new = y + b_weights @ k_all
    err_vec = btilde @ k_all
    en = _err_norm(err_vec, y_new, rtol, atol)
    return y_new, en, k_all


def _rodas5p_step_schur(
    rhs_fn,
    N_val,
    y,
    h,
    J,
    rtol,
    atol,
    active_idx,
    passive_idx,
    dfdt=None,
):
    """Execute one Rodas5P step with Schur complement linear solve.

    Exploits J[passive, passive] = 0 (Type I collisionless transport).
    W_pp = I/(γh) is diagonal → Schur complement reduces LU from D×D to n_active×n_active.

    Cost: O(n_active² × n_passive + n_active³) instead of O(D³).
    At N_q=20 phase 1: 7×7 LU instead of 247×247 LU → ~1000× cheaper.
    """
    if dfdt is None:
        dfdt = _finite_difference_dfdt(rhs_fn, N_val, y, h)
    else:
        dfdt = jnp.asarray(dfdt, dtype=y.dtype)

    N_dim = y.shape[0]
    gamma_h = GAMMA * h
    inv_gamma_h = 1.0 / gamma_h

    # Extract blocks of W = I/(γh) - J
    J_aa = J[jnp.ix_(active_idx, active_idx)]
    J_ap = J[jnp.ix_(active_idx, passive_idx)]
    J_pa = J[jnp.ix_(passive_idx, active_idx)]
    # J_pp = 0, so W_pp = I/(γh) (diagonal scalar)

    W_aa = jnp.eye(active_idx.shape[0]) * inv_gamma_h - J_aa
    W_ap = -J_ap
    W_pa = -J_pa

    # Schur complement: S = W_aa - W_ap · W_pp⁻¹ · W_pa = W_aa - γh · W_ap · W_pa
    S = W_aa - gamma_h * (W_ap @ W_pa)
    lu_s, piv_s = jax.scipy.linalg.lu_factor(S)

    def schur_solve(rhs_vec):
        r_a = rhs_vec[active_idx]
        r_p = rhs_vec[passive_idx]
        # Schur RHS: r_a - W_ap · (γh · r_p)
        schur_rhs = r_a - gamma_h * (W_ap @ r_p)
        k_a = jax.scipy.linalg.lu_solve((lu_s, piv_s), schur_rhs)
        # Back-substitute: k_p = γh · (r_p - W_pa · k_a)
        k_p = gamma_h * (r_p - W_pa @ k_a)
        k = jnp.empty_like(rhs_vec)
        k = k.at[active_idx].set(k_a)
        k = k.at[passive_idx].set(k_p)
        return k

    k_all = jnp.zeros((N_STAGES, N_dim))

    def stage_body(i, k_all):
        y_stage = y + A[i] @ k_all
        c_sum = C[i] @ k_all / h
        rhs_vec = (
            rhs_fn(N_val + c_nodes[i] * h, y_stage)
            + c_sum
            + h * d_nodes[i] * dfdt
        )
        k_i = schur_solve(rhs_vec)
        k_all = k_all.at[i].set(k_i)
        return k_all

    k_all = lax.fori_loop(0, N_STAGES, stage_body, k_all)
    y_new = y + b_weights @ k_all
    err_vec = btilde @ k_all
    en = _err_norm(err_vec, y_new, rtol, atol)
    return y_new, en, k_all


# ═══════════════════════════════════════════════════════════════════════
# §4. Gustafsson predictive step controller
# ═══════════════════════════════════════════════════════════════════════

_SAFETY = 0.9
_BETA = 0.04
_F_MIN = 0.2
_F_MAX = 6.0


def _gustafsson_h(h, err, err_prev, h_min, h_max):
    """Compute new step size via Gustafsson predictor.

    h_new = h × clip(safety × (1/err)^{1/p} × err_prev^β, f_min, f_max)
    """
    # Handle near-zero error → max growth
    fac = lax.cond(
        err < 1e-30,
        lambda _: _F_MAX,
        lambda _: _SAFETY * (1.0 / err) ** (1.0 / ORDER) * err_prev ** _BETA,
        None,
    )
    fac = jnp.clip(fac, _F_MIN, _F_MAX)
    h_new = h * fac
    h_new = jnp.clip(h_new, h_min, h_max)
    return h_new


# ═══════════════════════════════════════════════════════════════════════
# §5. Adaptive solver via lax.while_loop
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class JaxSolverResult:
    """Result of JAX Rodas5P solve."""
    y_final: jnp.ndarray     # final state vector
    N_final: float            # final independent variable value
    n_steps: int              # accepted steps
    n_reject: int             # rejected steps
    success: bool             # solver converged
    message: str = ""
    diagnostics: Optional[dict] = None
    status: int = 0           # raw status code: 0=normal, 2=event_triggered, -1=h_min, -2=max_steps
    h_final: float = 0.0     # step size at termination
    event_triggered: bool = False  # whether a terminal event was detected


@dataclass
class JaxBatchSolverResult:
    """Result of batched JAX Rodas5P solves."""
    y_final: jnp.ndarray
    N_final: jnp.ndarray
    h_final: jnp.ndarray
    n_steps: jnp.ndarray
    n_reject: jnp.ndarray
    success: jnp.ndarray
    status: jnp.ndarray
    diagnostics: Optional[dict] = None
    event_triggered: Optional[jnp.ndarray] = None


def _cubic_hermite_state(theta: jnp.ndarray, y0: jnp.ndarray, y1: jnp.ndarray, f0: jnp.ndarray, f1: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
    """Cubic Hermite interpolation of the accepted step state.

    This is only used when an accepted step brackets a terminal event, so the
    fast no-event path remains unchanged while the event state becomes much
    more accurate than the raw step endpoint.
    """
    th2 = theta * theta
    th3 = th2 * theta
    h00 = 2.0 * th3 - 3.0 * th2 + 1.0
    h10 = th3 - 2.0 * th2 + theta
    h01 = -2.0 * th3 + 3.0 * th2
    h11 = th3 - th2
    return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1


def _refine_terminal_event(
    event_fn: Callable,
    N0: jnp.ndarray,
    y0: jnp.ndarray,
    ev0: jnp.ndarray,
    N1: jnp.ndarray,
    y1: jnp.ndarray,
    ev1: jnp.ndarray,
    f0: jnp.ndarray,
    f1: jnp.ndarray,
    refine_steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Refine a terminal event inside an accepted step.

    We use a fixed-count secant-seeded bisection on a cubic-Hermite dense
    approximation of the accepted step. The extra work only happens on the
    rare step that brackets a terminal event, preserving GPU-friendliness and
    the existing fast path.
    """
    dt = N1 - N0

    def ev_theta(theta: jnp.ndarray) -> jnp.ndarray:
        N_theta = N0 + theta * dt
        y_theta = _cubic_hermite_state(theta, y0, y1, f0, f1, dt)
        return event_fn(N_theta, y_theta)

    theta_secant = jnp.clip(ev0 / jnp.maximum(ev0 - ev1, 1e-300), 0.0, 1.0)
    ev_secant = ev_theta(theta_secant)
    lo = jnp.where(ev_secant > 0.0, theta_secant, 0.0)
    hi = jnp.where(ev_secant > 0.0, 1.0, theta_secant)

    def bisect_body(_, interval):
        lo, hi = interval
        mid = 0.5 * (lo + hi)
        ev_mid = ev_theta(mid)
        lo_new = jnp.where(ev_mid > 0.0, mid, lo)
        hi_new = jnp.where(ev_mid > 0.0, hi, mid)
        return (lo_new, hi_new)

    lo, hi = lax.fori_loop(0, refine_steps, bisect_body, (lo, hi))
    theta_root = 0.5 * (lo + hi)
    N_root = N0 + theta_root * dt
    y_root = _cubic_hermite_state(theta_root, y0, y1, f0, f1, dt)
    return N_root, y_root




def _solver_runner_cache_key(
    rhs_fn: Callable,
    event_fn: Optional[Callable],
    event_refine_steps: int,
    jac_fn: Optional[Callable] = None,
    linear_solver_prepare_fn: Optional[Callable] = None,
    linear_solver_solve_fn: Optional[Callable] = None,
    active_indices_tuple: Optional[tuple] = None,
    state_dim: Optional[int] = None,
    reuse_enabled: bool = False,
    dependency_enabled: bool = False,
    dfdt_fn: Optional[Callable] = None,
) -> tuple:
    return (
        int(id(rhs_fn)),
        int(id(event_fn)) if event_fn is not None else 0,
        int(event_refine_steps),
        int(id(jac_fn)) if jac_fn is not None else 0,
        int(id(linear_solver_prepare_fn)) if linear_solver_prepare_fn is not None else 0,
        int(id(linear_solver_solve_fn)) if linear_solver_solve_fn is not None else 0,
        active_indices_tuple or (),
        int(state_dim) if state_dim is not None else -1,
        int(bool(reuse_enabled)),
        int(bool(dependency_enabled)),
        int(id(dfdt_fn)) if dfdt_fn is not None else 0,
    )


def _cached_solver_runner(
    rhs_fn: Callable,
    event_fn: Optional[Callable],
    *,
    event_refine_steps: int,
    jac_fn: Optional[Callable] = None,
    linear_solver_prepare_fn: Optional[Callable] = None,
    linear_solver_solve_fn: Optional[Callable] = None,
    active_indices: Optional[jnp.ndarray] = None,
    state_dim: Optional[int] = None,
    reuse_enabled: bool = False,
    dependency_enabled: bool = False,
    dfdt_fn: Optional[Callable] = None,
) -> Callable:
    ai_tuple = tuple(int(x) for x in active_indices) if active_indices is not None else None
    key = _solver_runner_cache_key(
        rhs_fn,
        event_fn,
        event_refine_steps,
        jac_fn,
        linear_solver_prepare_fn,
        linear_solver_solve_fn,
        ai_tuple,
        state_dim,
        reuse_enabled=reuse_enabled,
        dependency_enabled=dependency_enabled,
        dfdt_fn=dfdt_fn,
    )
    cached = _SOLVE_RUNNER_CACHE.get(key)
    if cached is not None:
        return cached

    actual_jac_fn = jac_fn if jac_fn is not None else _cached_jacfwd(rhs_fn)
    has_event = event_fn is not None
    refine_steps = int(event_refine_steps)
    use_schur = active_indices is not None
    use_custom_linear_solver = (
        linear_solver_prepare_fn is not None and linear_solver_solve_fn is not None
    )

    # Pre-compute passive indices for Schur complement eagerly so repeated
    # same-process solves do not leak traced values into the cached closure.
    if use_schur:
        if state_dim is None:
            raise ValueError('state_dim is required when active_indices are provided')
        _active_idx = jnp.asarray(active_indices, dtype=jnp.int32)
        _active_set = set(int(x) for x in active_indices)
        _passive_idx = jnp.array([i for i in range(int(state_dim)) if i not in _active_set], dtype=jnp.int32)

    def _runner(
        y0,
        N_start,
        N_end,
        rtol,
        atol,
        max_steps,
        h_init_val,
        h_min,
        h_max,
        ev_init,
        J0,
        dependency_masks,
        dependency_thresholds,
        dependency_scale_floors,
        jacobian_refresh_stride,
    ):
        sgn = jnp.sign(N_end - N_start)

        init_state = (
            N_start,
            y0,
            h_init_val,
            jnp.float64(1.0),
            jnp.int32(0),
            jnp.int32(0),
            jnp.int32(0),
            ev_init,
            J0,
            y0,
            jnp.int32(0),
            jnp.int32(1),
            jnp.int32(0),
            jnp.int32(0),
            jnp.int32(0),
            jnp.int32(0),
            jnp.zeros((dependency_masks.shape[0],), dtype=jnp.float64),
        )

        def cond_fn(state):
            N, y, h, ep, ns, nr, status, ev, J_cache, y_refresh, reuse_age, jac_refresh_count, jac_reuse_count, dep_refresh_count, stride_refresh_count, err_refresh_count, dep_max = state
            still_running = (status == 0)
            not_done = (sgn * (N - N_end) < -1e-14)
            under_max = (ns + nr < max_steps)
            return still_running & not_done & under_max

        def body_fn(state):
            N, y, h, ep, ns, nr, status, ev_prev, J_cache, y_refresh, reuse_age, jac_refresh_count, jac_reuse_count, dep_refresh_count, stride_refresh_count, err_refresh_count, dep_max = state

            remaining = jnp.abs(N_end - N)
            h_use = jnp.minimum(h, remaining)

            if dependency_enabled:
                base_scale = jnp.maximum(jnp.abs(y_refresh), 1.0e-8)
                block_scale = jnp.maximum(base_scale[None, :], dependency_scale_floors[:, None])
                masked_delta = jnp.where(
                    dependency_masks > 0.0,
                    jnp.abs(y - y_refresh)[None, :] / block_scale,
                    -jnp.inf,
                )
                dep_max_now = jnp.max(masked_delta, axis=1)
                dep_trigger = jnp.any(dep_max_now > dependency_thresholds)
            else:
                dep_max_now = dep_max
                dep_trigger = jnp.asarray(False)

            attempted = (ns + nr) > 0
            if reuse_enabled:
                stride_trigger = reuse_age >= jnp.int32(jacobian_refresh_stride - 1)
                err_trigger = attempted & (ep > 0.25)
                refresh_due = stride_trigger | err_trigger | dep_trigger
            else:
                stride_trigger = jnp.asarray(False)
                err_trigger = jnp.asarray(False)
                refresh_due = attempted
            J = lax.cond(refresh_due, lambda _: actual_jac_fn(N, y), lambda _: J_cache, operand=None)
            h_signed = h_use * sgn
            dfdt = _step_start_dfdt(rhs_fn, dfdt_fn, N, y, h_signed)

            if use_custom_linear_solver:
                y_new, err, k_all = _rodas5p_step_custom_linear_solver(
                    rhs_fn,
                    N,
                    y,
                    h_signed,
                    J,
                    rtol,
                    atol,
                    linear_solver_prepare_fn,
                    linear_solver_solve_fn,
                    dfdt=dfdt,
                )
            elif use_schur:
                y_new, err, k_all = _rodas5p_step_schur(
                    rhs_fn,
                    N,
                    y,
                    h_signed,
                    J,
                    rtol,
                    atol,
                    _active_idx,
                    _passive_idx,
                    dfdt=dfdt,
                )
            else:
                y_new, err, k_all = _rodas5p_step(
                    rhs_fn, N, y, h_signed, J, rtol, atol, dfdt=dfdt
                )
            accepted = (err <= 1.0)
            h_new = _gustafsson_h(
                h_use,
                jnp.where(accepted, err, jnp.maximum(err, 2.0)),
                ep,
                h_min,
                h_max,
            )

            N_new = jnp.where(accepted, N + h_use * sgn, N)
            y_out = jnp.where(accepted, y_new, y)
            ep_new = jnp.where(accepted, err, ep)
            ns_new = jnp.where(accepted, ns + 1, ns)
            nr_new = jnp.where(accepted, nr, nr + 1)

            jac_refresh_count_new = jac_refresh_count + jnp.where(refresh_due, 1, 0)
            jac_reuse_count_new = jac_reuse_count + jnp.where(reuse_enabled & (~refresh_due), 1, 0)
            dep_refresh_count_new = dep_refresh_count + jnp.where(refresh_due & dep_trigger, 1, 0)
            stride_refresh_count_new = stride_refresh_count + jnp.where(refresh_due & stride_trigger, 1, 0)
            err_refresh_count_new = err_refresh_count + jnp.where(refresh_due & err_trigger, 1, 0)
            if reuse_enabled:
                reuse_age_new = jnp.where(
                    accepted,
                    jnp.where(refresh_due, jnp.int32(0), reuse_age + 1),
                    jnp.int32(jacobian_refresh_stride),
                )
            else:
                reuse_age_new = jnp.int32(0)
            J_cache_new = J
            y_refresh_new = jnp.where(refresh_due, y, y_refresh)
            dep_max_new = jnp.where(refresh_due, dep_max_now, dep_max)

            ev_new = ev_prev
            status_new = status

            if has_event:
                ev_val = event_fn(N_new, y_out)
                ev_candidate = jnp.where(accepted, ev_val, ev_prev)
                crossed = accepted & (ev_prev > 0.0) & (ev_candidate <= 0.0)

                def refine_event(_):
                    f_start = rhs_fn(N, y)
                    f_end = rhs_fn(N_new, y_out)
                    N_root, y_root = _refine_terminal_event(
                        event_fn, N, y, ev_prev, N_new, y_out, ev_candidate,
                        f_start, f_end, refine_steps,
                    )
                    return N_root, y_root, jnp.float64(0.0), jnp.int32(2)

                def keep_endpoint(_):
                    return N_new, y_out, ev_candidate, status_new

                N_new, y_out, ev_new, status_new = lax.cond(
                    crossed, refine_event, keep_endpoint, operand=None
                )

            status_new = jnp.where((~accepted) & (h_new <= h_min * 1.01), jnp.int32(-1), status_new)

            return (
                N_new, y_out, h_new, ep_new, ns_new, nr_new, status_new, ev_new,
                J_cache_new, y_refresh_new, reuse_age_new, jac_refresh_count_new, jac_reuse_count_new,
                dep_refresh_count_new, stride_refresh_count_new, err_refresh_count_new, dep_max_new,
            )

        return lax.while_loop(cond_fn, body_fn, init_state)

    compiled = jax.jit(_runner, donate_argnums=(0, 10))
    _SOLVE_RUNNER_CACHE[key] = compiled
    return compiled


def jax_rodas5p_solve(
    rhs_fn: Callable,
    y0: jnp.ndarray,
    N_span: Tuple[float, float],
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 2000,
    h_init: Optional[float] = None,
    h_min: float = 1e-14,
    h_max: float = 0.5,
    event_fn: Optional[Callable] = None,
    event_refine_steps: int = 24,
    jac_fn: Optional[Callable] = None,
    linear_solver_prepare_fn: Optional[Callable] = None,
    linear_solver_solve_fn: Optional[Callable] = None,
    jacobian_reuse_experimental: bool = False,
    jacobian_refresh_stride: int = 1,
    jacobian_dependency_masks: Optional[jnp.ndarray] = None,
    jacobian_dependency_thresholds: Optional[jnp.ndarray] = None,
    jacobian_dependency_scale_floors: Optional[jnp.ndarray] = None,
    active_indices: Optional[jnp.ndarray] = None,
    dfdt_fn: Optional[Callable] = None,
) -> JaxSolverResult:
    """Solve ODE system using JAX-native Rodas5P.

    Parameters
    ----------
    jac_fn : callable, optional
        Explicit Jacobian callback with signature ``jac_fn(N, y) -> J``.
        When provided, bypasses the default dense autodiff path. Mutually
        exclusive with ``active_indices``.
    linear_solver_prepare_fn, linear_solver_solve_fn : callable, optional
        Experimental hook pair for custom per-step linear solves. When
        provided together, the solver prepares one linear-system state per
        accepted/rejected Rodas step via
        ``linear_solver_prepare_fn(N, y, h, jacobian_payload)`` and uses
        ``linear_solver_solve_fn(prepared_state, rhs_vec)`` for the eight
        stage solves. This is intended for future structured Jacobian
        surfaces such as low-rank tier-3 transport. Mutually exclusive with
        ``active_indices``.
    active_indices : array of int, optional
        If provided, enables block-sparse Jacobian optimization.
        Must list indices of "active" state variables (geometry + thermo + network).
        All other indices are treated as passive (transport hierarchy with J_pp=0).
        Provides ~D/n_active × Jacobian speedup for large transport sectors.
    dfdt_fn : callable, optional
        Partial derivative callback ``dfdt_fn(N, y) -> partial rhs / partial N``
        at fixed ``y``. When omitted, a scale-aware symmetric finite difference
        is evaluated once per attempted step.
    """
    y0 = jnp.asarray(y0, dtype=jnp.float64)
    N_start, N_end = N_span

    if jac_fn is not None and active_indices is not None:
        raise ValueError("jac_fn and active_indices cannot be combined.")
    if (linear_solver_prepare_fn is None) != (linear_solver_solve_fn is None):
        raise ValueError("linear_solver_prepare_fn and linear_solver_solve_fn must be provided together.")
    if linear_solver_prepare_fn is not None and active_indices is not None:
        raise ValueError("custom linear solver hooks and active_indices cannot be combined.")

    if h_init is None:
        f0 = rhs_fn(N_start, y0)
        sc = atol + rtol * jnp.abs(y0)
        d0 = jnp.sqrt(jnp.mean((y0 / sc) ** 2))
        d1 = jnp.sqrt(jnp.mean((f0 / sc) ** 2))
        h_auto = jnp.where(d0 > 1e-10, 0.01 * d0 / jnp.maximum(d1, 1e-30), 1e-6)
        h_init_val = jnp.minimum(h_auto, jnp.minimum(h_max, jnp.abs(N_end - N_start) * 0.1))
    else:
        h_init_val = jnp.abs(jnp.float64(h_init))

    # Block-sparse Jacobian if active_indices provided, else dense jacfwd
    if jac_fn is not None:
        custom_jac_fn = jac_fn
    elif active_indices is not None:
        custom_jac_fn = _get_block_sparse_jac_fn(rhs_fn, active_indices, y0.shape[0])
    else:
        custom_jac_fn = _cached_jacfwd(rhs_fn)

    jac_fn_for_runner = custom_jac_fn
    jacobian_refresh_stride = max(int(jacobian_refresh_stride), 1)
    reuse_enabled = bool(jacobian_reuse_experimental) and jacobian_refresh_stride > 1
    if jacobian_dependency_masks is None or jacobian_dependency_thresholds is None:
        dependency_masks = jnp.zeros((1, y0.shape[0]), dtype=jnp.float64)
        dependency_thresholds = jnp.array([jnp.inf], dtype=jnp.float64)
        dependency_scale_floors = jnp.zeros((1,), dtype=jnp.float64)
        dependency_enabled = False
    else:
        dependency_masks = jnp.asarray(jacobian_dependency_masks, dtype=jnp.float64)
        dependency_thresholds = jnp.asarray(jacobian_dependency_thresholds, dtype=jnp.float64)
        if jacobian_dependency_scale_floors is None:
            dependency_scale_floors = jnp.zeros_like(dependency_thresholds)
        else:
            dependency_scale_floors = jnp.asarray(jacobian_dependency_scale_floors, dtype=jnp.float64)
        dependency_enabled = bool(reuse_enabled) and dependency_masks.shape[0] > 0
    J0 = jac_fn_for_runner(N_start, y0)

    ev_init = event_fn(N_start, y0) if event_fn is not None else 1.0
    runner = _cached_solver_runner(
        rhs_fn,
        event_fn,
        event_refine_steps=event_refine_steps,
        jac_fn=custom_jac_fn,
        linear_solver_prepare_fn=linear_solver_prepare_fn,
        linear_solver_solve_fn=linear_solver_solve_fn,
        active_indices=active_indices,
        state_dim=int(y0.shape[0]),
        reuse_enabled=reuse_enabled,
        dependency_enabled=dependency_enabled,
        dfdt_fn=dfdt_fn,
    )
    # Donate a private working copy so callers can safely reuse their input
    # arrays across repeated solves in the same Python scope.
    y0_work = y0 + jnp.zeros_like(y0)
    final_state = runner(
        y0_work, jnp.float64(N_start), jnp.float64(N_end),
        jnp.float64(rtol), jnp.float64(atol), jnp.int32(max_steps),
        jnp.asarray(h_init_val, dtype=jnp.float64), jnp.float64(h_min), jnp.float64(h_max),
        jnp.float64(ev_init), J0, dependency_masks, dependency_thresholds, dependency_scale_floors,
        jnp.int32(jacobian_refresh_stride),
    )
    (
        N_final, y_final, h_final, ep_final, n_steps, n_reject, status, ev_final,
        J_final, y_refresh_final, reuse_age_final, jac_refresh_count, jac_reuse_count,
        dep_refresh_count, stride_refresh_count, err_refresh_count, dep_max_final,
    ) = final_state

    (
        N_final_host,
        h_final_host,
        n_steps_host,
        n_reject_host,
        status_host,
        jac_refresh_count_host,
        jac_reuse_count_host,
        dep_refresh_count_host,
        stride_refresh_count_host,
        err_refresh_count_host,
        dep_max_host,
        dependency_thresholds_host,
        dependency_scale_floors_host,
    ) = jax.device_get(
        (
            N_final,
            h_final,
            n_steps,
            n_reject,
            status,
            jac_refresh_count,
            jac_reuse_count,
            dep_refresh_count,
            stride_refresh_count,
            err_refresh_count,
            dep_max_final,
            dependency_thresholds,
            dependency_scale_floors,
        )
    )

    status_int = int(status_host)
    span_sign = float(np.sign(float(N_end) - float(N_start)))
    endpoint_reached = bool(
        span_sign == 0.0
        or span_sign * (float(N_final_host) - float(N_end)) >= -1.0e-14
    )
    max_steps_exhausted = bool(
        status_int == _STATUS_SPAN_COMPLETE and not endpoint_reached
    )
    if max_steps_exhausted:
        status_int = _STATUS_MAX_STEPS_EXHAUSTED
    success = (status_int >= 0)
    event_triggered = (status_int == _STATUS_EVENT_TRIGGERED)
    if linear_solver_prepare_fn is not None:
        linear_solver_policy = "custom_hook"
    elif active_indices is not None:
        linear_solver_policy = "block_sparse_schur"
    else:
        linear_solver_policy = "dense_lu"
    diagnostics = {
        'event_triggered': event_triggered,
        'linear_solver_policy': linear_solver_policy,
        'jacobian_policy': ('experimental_dependency_cadence' if dependency_enabled else ('experimental_reuse' if reuse_enabled else 'exact_per_step')),
        'jacobian_refresh_stride': int(jacobian_refresh_stride),
        'jacobian_refresh_count': int(jac_refresh_count_host),
        'jacobian_reuse_count': int(jac_reuse_count_host),
        'jacobian_reuse_enabled': bool(reuse_enabled),
        'jacobian_dependency_enabled': bool(dependency_enabled),
        'jacobian_dependency_trigger_count': int(dep_refresh_count_host),
        'jacobian_stride_trigger_count': int(stride_refresh_count_host),
        'jacobian_error_trigger_count': int(err_refresh_count_host),
        'jacobian_dependency_last_maxima': [float(x) for x in np.asarray(dep_max_host)],
        'jacobian_dependency_thresholds': [float(x) for x in np.asarray(dependency_thresholds_host)],
        'jacobian_dependency_scale_floors': [float(x) for x in np.asarray(dependency_scale_floors_host)],
        'jacobian_dependency_floor_enabled': bool(np.any(np.asarray(dependency_scale_floors_host) > 0.0)),
        'endpoint_reached': bool(endpoint_reached),
        'max_steps_exhausted': bool(max_steps_exhausted),
        'status_contract': _STATUS_CONTRACT,
    }
    message = (
        "status=-2:max_steps_exhausted"
        if max_steps_exhausted
        else f"status={status_int}"
    )

    return JaxSolverResult(
        y_final=y_final,
        N_final=float(N_final_host),
        n_steps=int(n_steps_host),
        n_reject=int(n_reject_host),
        success=bool(success),
        message=message,
        diagnostics=diagnostics,
        status=status_int,
        h_final=float(h_final_host),
        event_triggered=event_triggered,
    )


# ═══════════════════════════════════════════════════════════════════════
# §5b. Pure-JAX solve core (vmap-compatible, no Python conversions)
# ═══════════════════════════════════════════════════════════════════════

def _solve_core(
    rhs_fn: Callable,
    y0: jnp.ndarray,
    N_start: float,
    N_end: float,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 2000,
    h_max: float = 0.5,
    h_min: float = 1e-14,
    dfdt_fn: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pure-JAX solve returning only jnp arrays. vmap-compatible.

    Returns
    -------
    (y_final, N_final, n_steps, status)
        All as jnp scalar/array — no Python type conversions.
    """
    y0 = jnp.asarray(y0, dtype=jnp.float64)
    N_start = jnp.float64(N_start)
    N_end = jnp.float64(N_end)
    sgn = jnp.sign(N_end - N_start)

    # Auto h_init
    f0 = rhs_fn(N_start, y0)
    sc = atol + rtol * jnp.abs(y0)
    d0 = jnp.sqrt(jnp.mean((y0 / sc) ** 2))
    d1 = jnp.sqrt(jnp.mean((f0 / sc) ** 2))
    h_auto = jnp.where(d0 > 1e-10, 0.01 * d0 / jnp.maximum(d1, 1e-30), 1e-6)
    h_init_val = jnp.minimum(h_auto, jnp.minimum(h_max, jnp.abs(N_end - N_start) * 0.1))

    jac_fn = jax.jacfwd(rhs_fn, argnums=1)

    init_state = (
        N_start,
        y0,
        h_init_val,
        jnp.float64(1.0),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(0),
        jnp.float64(1.0),
    )

    def cond_fn(state):
        N, y, h, ep, ns, nr, status, ev = state
        return (status == 0) & (sgn * (N - N_end) < -1e-14) & (ns + nr < max_steps)

    def body_fn(state):
        N, y, h, ep, ns, nr, status, ev_prev = state
        remaining = jnp.abs(N_end - N)
        h_use = jnp.minimum(h, remaining)
        J = jac_fn(N, y)
        h_signed = h_use * sgn
        dfdt = _step_start_dfdt(rhs_fn, dfdt_fn, N, y, h_signed)
        y_new, err, k_all = _rodas5p_step(
            rhs_fn, N, y, h_signed, J, rtol, atol, dfdt=dfdt
        )
        accepted = (err <= 1.0)
        h_new = _gustafsson_h(
            h_use,
            jnp.where(accepted, err, jnp.maximum(err, 2.0)),
            ep, h_min, h_max,
        )
        N_new = jnp.where(accepted, N + h_use * sgn, N)
        y_out = jnp.where(accepted, y_new, y)
        ep_new = jnp.where(accepted, err, ep)
        ns_new = jnp.where(accepted, ns + 1, ns)
        nr_new = jnp.where(accepted, nr, nr + 1)
        status_new = jnp.where(
            (~accepted) & (h_new <= h_min * 1.01),
            jnp.int32(-1), status,
        )
        return (N_new, y_out, h_new, ep_new, ns_new, nr_new, status_new, ev_prev)

    final = lax.while_loop(cond_fn, body_fn, init_state)
    N_final, y_final, _, _, n_steps, n_reject, status, _ = final
    endpoint_reached = (sgn == 0.0) | (sgn * (N_final - N_end) >= -1.0e-14)
    status = jnp.where(
        (status == _STATUS_SPAN_COMPLETE) & (~endpoint_reached),
        jnp.int32(_STATUS_MAX_STEPS_EXHAUSTED),
        status,
    )
    return y_final, N_final, n_steps, status


def _batch_solver_runner_cache_key(
    rhs_fn: Callable,
    event_fn: Optional[Callable],
    event_refine_steps: int,
    jac_fn: Optional[Callable] = None,
    active_indices_tuple: Optional[tuple] = None,
    state_dim: Optional[int] = None,
    batch_linear_solver_backend: str = "jax_tensorized",
    reuse_enabled: bool = False,
    dfdt_fn: Optional[Callable] = None,
) -> tuple:
    return (
        "batch",
        int(id(rhs_fn)),
        int(id(event_fn)) if event_fn is not None else 0,
        int(event_refine_steps),
        int(id(jac_fn)) if jac_fn is not None else 0,
        active_indices_tuple or (),
        int(state_dim) if state_dim is not None else -1,
        str(batch_linear_solver_backend),
        int(bool(reuse_enabled)),
        int(id(dfdt_fn)) if dfdt_fn is not None else 0,
    )


def _cached_batched_solver_runner(
    rhs_fn: Callable,
    event_fn: Optional[Callable],
    *,
    event_refine_steps: int,
    jac_fn: Optional[Callable] = None,
    active_indices: Optional[jnp.ndarray] = None,
    state_dim: Optional[int] = None,
    batch_linear_solver_backend: str = "jax_tensorized",
    jacobian_reuse_experimental: bool = False,
    dfdt_fn: Optional[Callable] = None,
) -> Callable:
    batch_linear_solver_backend = _canonical_batch_linear_solver_backend(
        batch_linear_solver_backend
    )
    reuse_enabled = bool(jacobian_reuse_experimental)
    ai_tuple = tuple(int(x) for x in active_indices) if active_indices is not None else None
    key = _batch_solver_runner_cache_key(
        rhs_fn,
        event_fn,
        event_refine_steps,
        jac_fn=jac_fn,
        active_indices_tuple=ai_tuple,
        state_dim=state_dim,
        batch_linear_solver_backend=batch_linear_solver_backend,
        reuse_enabled=reuse_enabled,
        dfdt_fn=dfdt_fn,
    )
    cached = _BATCH_SOLVE_RUNNER_CACHE.get(key)
    if cached is not None:
        return cached

    if jac_fn is not None and active_indices is not None:
        raise ValueError("jac_fn and active_indices cannot be combined.")

    use_schur = active_indices is not None
    if use_schur:
        if state_dim is None:
            raise ValueError("state_dim is required when active_indices are provided")
        _active_idx = jnp.asarray(active_indices, dtype=jnp.int32)
        _active_set = set(int(x) for x in active_indices)
        _passive_idx = jnp.array(
            [i for i in range(int(state_dim)) if i not in _active_set],
            dtype=jnp.int32,
        )
        actual_jac_fn = _get_block_sparse_jac_fn(rhs_fn, active_indices, int(state_dim))
    else:
        actual_jac_fn = jac_fn if jac_fn is not None else _cached_jacfwd(rhs_fn)

    dense_solve_fn = _batch_dense_solve_fn(batch_linear_solver_backend)
    refine_steps = int(event_refine_steps)
    has_event = event_fn is not None

    def _runner(
        y0_batch,
        N_start_arr,
        N_end_arr,
        rtol,
        atol,
        max_steps,
        h_min,
        h_max,
        jacobian_refresh_stride,
    ):
        y0_batch = jnp.asarray(y0_batch, dtype=jnp.float64)
        N_start_arr = jnp.asarray(N_start_arr, dtype=jnp.float64)
        N_end_arr = jnp.asarray(N_end_arr, dtype=jnp.float64)
        sgn = jnp.sign(N_end_arr - N_start_arr)
        jacobian_refresh_stride = jnp.maximum(
            jnp.int32(jacobian_refresh_stride),
            jnp.int32(1),
        )

        f0 = jax.vmap(rhs_fn)(N_start_arr, y0_batch)
        sc = atol + rtol * jnp.abs(y0_batch)
        d0 = jnp.sqrt(jnp.mean((y0_batch / sc) ** 2, axis=1))
        d1 = jnp.sqrt(jnp.mean((f0 / sc) ** 2, axis=1))
        h_auto = jnp.where(d0 > 1e-10, 0.01 * d0 / jnp.maximum(d1, 1e-30), 1e-6)
        h_init_val = jnp.minimum(h_auto, jnp.minimum(h_max, jnp.abs(N_end_arr - N_start_arr) * 0.1))
        if has_event:
            ev_init = jax.vmap(event_fn)(N_start_arr, y0_batch)
        else:
            ev_init = jnp.ones_like(N_start_arr, dtype=jnp.float64)
        if reuse_enabled:
            J_init = jax.vmap(actual_jac_fn)(N_start_arr, y0_batch)
            jac_refresh_init = jnp.ones_like(N_start_arr, dtype=jnp.int32)
        else:
            dim = int(y0_batch.shape[1])
            J_init = jnp.zeros((int(y0_batch.shape[0]), dim, dim), dtype=jnp.float64)
            jac_refresh_init = jnp.zeros_like(N_start_arr, dtype=jnp.int32)

        init_state = (
            N_start_arr,
            y0_batch,
            h_init_val,
            jnp.ones_like(N_start_arr, dtype=jnp.float64),
            jnp.zeros_like(N_start_arr, dtype=jnp.int32),
            jnp.zeros_like(N_start_arr, dtype=jnp.int32),
            jnp.zeros_like(N_start_arr, dtype=jnp.int32),
            ev_init,
            J_init,
            jnp.zeros_like(N_start_arr, dtype=jnp.int32),
            jac_refresh_init,
            jnp.zeros_like(N_start_arr, dtype=jnp.int32),
            jnp.zeros_like(N_start_arr, dtype=jnp.int32),
            jnp.zeros_like(N_start_arr, dtype=jnp.int32),
        )

        def cond_fn(state):
            (
                N_arr,
                y_batch,
                h,
                ep,
                ns,
                nr,
                status,
                ev_prev,
                J_cache,
                reuse_age,
                jac_refresh_count,
                jac_reuse_count,
                stride_refresh_count,
                err_refresh_count,
            ) = state
            done_by_span = sgn * (N_arr - N_end_arr) >= -1e-14
            under_max = (ns + nr) < max_steps
            active = (status == 0) & (~done_by_span) & under_max
            return jnp.any(active)

        def body_fn(state):
            (
                N_arr,
                y_batch,
                h,
                ep,
                ns,
                nr,
                status,
                ev_prev,
                J_cache,
                reuse_age,
                jac_refresh_count,
                jac_reuse_count,
                stride_refresh_count,
                err_refresh_count,
            ) = state
            remaining = jnp.abs(N_end_arr - N_arr)
            h_use = jnp.minimum(h, remaining)
            done_by_span = sgn * (N_arr - N_end_arr) >= -1e-14
            under_max = (ns + nr) < max_steps
            active = (status == 0) & (~done_by_span) & under_max
            safe_h = jnp.where(active, h_use, jnp.full_like(h_use, jnp.maximum(h_min, 1.0e-6)))

            attempted = (ns + nr) > 0
            if reuse_enabled:
                stride_trigger = jnp.any(active & (reuse_age >= (jacobian_refresh_stride - 1)))
                err_trigger = jnp.any(active & attempted & (ep > 0.25))
                refresh_due = stride_trigger | err_trigger
                J_batch = lax.cond(
                    refresh_due,
                    lambda _: jax.vmap(actual_jac_fn)(N_arr, y_batch),
                    lambda _: J_cache,
                    operand=None,
                )
            else:
                stride_trigger = jnp.asarray(False)
                err_trigger = jnp.asarray(False)
                refresh_due = jnp.asarray(True)
                J_batch = jax.vmap(actual_jac_fn)(N_arr, y_batch)
            h_signed = safe_h * sgn
            dfdt_batch = _step_start_dfdt_batch(
                rhs_fn, dfdt_fn, N_arr, y_batch, h_signed, active=active
            )

            if use_schur:
                def step_one(N_val, y_val, h_val, J_val, dfdt_val):
                    return _rodas5p_step_schur(
                        rhs_fn,
                        N_val,
                        y_val,
                        h_val,
                        J_val,
                        rtol,
                        atol,
                        _active_idx,
                        _passive_idx,
                        dfdt=dfdt_val,
                    )

                y_new_batch, err_batch, _ = jax.vmap(step_one)(
                    N_arr,
                    y_batch,
                    h_signed,
                    J_batch,
                    dfdt_batch,
                )
            else:
                y_new_batch, err_batch, _ = _rodas5p_step_batched_dense(
                    rhs_fn,
                    N_arr,
                    y_batch,
                    h_signed,
                    J_batch,
                    rtol,
                    atol,
                    dense_solve_fn,
                    dfdt=dfdt_batch,
                )
            accepted = active & (err_batch <= 1.0)
            err_for_h = jnp.where(accepted, err_batch, jnp.maximum(err_batch, 2.0))
            fac = jnp.where(
                err_for_h < 1e-30,
                jnp.full_like(err_for_h, _F_MAX),
                _SAFETY
                * (1.0 / jnp.maximum(err_for_h, 1e-300)) ** (1.0 / ORDER)
                * jnp.maximum(ep, 1e-30) ** _BETA,
            )
            fac = jnp.clip(fac, _F_MIN, _F_MAX)
            h_candidate = jnp.clip(safe_h * fac, h_min, h_max)
            h_new = jnp.where(active, h_candidate, h)
            N_new = jnp.where(accepted, N_arr + safe_h * sgn, N_arr)
            y_out = jnp.where(accepted[:, None], y_new_batch, y_batch)
            ep_new = jnp.where(accepted, err_batch, ep)
            ns_new = jnp.where(accepted, ns + 1, ns)
            nr_new = jnp.where(active & (~accepted), nr + 1, nr)
            jac_refresh_count_new = jac_refresh_count + jnp.where(
                active & refresh_due,
                jnp.int32(1),
                jnp.int32(0),
            )
            jac_reuse_count_new = jac_reuse_count + jnp.where(
                active & (jnp.asarray(reuse_enabled) & (~refresh_due)),
                jnp.int32(1),
                jnp.int32(0),
            )
            stride_refresh_count_new = stride_refresh_count + jnp.where(
                active & refresh_due & stride_trigger,
                jnp.int32(1),
                jnp.int32(0),
            )
            err_refresh_count_new = err_refresh_count + jnp.where(
                active & refresh_due & err_trigger,
                jnp.int32(1),
                jnp.int32(0),
            )
            reuse_age_new = jnp.where(
                active,
                jnp.where(
                    accepted,
                    jnp.where(refresh_due, jnp.int32(0), reuse_age + jnp.int32(1)),
                    jacobian_refresh_stride,
                ),
                reuse_age,
            )
            status_new = jnp.where(
                active & (~accepted) & (h_candidate <= h_min * 1.01),
                jnp.int32(-1),
                status,
            )

            if has_event:
                ev_val = jax.vmap(event_fn)(N_new, y_out)
                ev_candidate = jnp.where(accepted, ev_val, ev_prev)
                crossed = accepted & (ev_prev > 0.0) & (ev_candidate <= 0.0)

                def refine_events(_):
                    f_start = jax.vmap(rhs_fn)(N_arr, y_batch)
                    f_end = jax.vmap(rhs_fn)(N_new, y_out)
                    N_root_all, y_root_all = jax.vmap(
                        lambda n0, y0, e0, n1, y1, e1, f0_i, f1_i: _refine_terminal_event(
                            event_fn,
                            n0,
                            y0,
                            e0,
                            n1,
                            y1,
                            e1,
                            f0_i,
                            f1_i,
                            refine_steps,
                        )
                    )(
                        N_arr,
                        y_batch,
                        ev_prev,
                        N_new,
                        y_out,
                        ev_candidate,
                        f_start,
                        f_end,
                    )
                    return (
                        jnp.where(crossed, N_root_all, N_new),
                        jnp.where(crossed[:, None], y_root_all, y_out),
                        jnp.where(crossed, jnp.zeros_like(ev_candidate), ev_candidate),
                        jnp.where(crossed, jnp.full_like(status_new, 2), status_new),
                    )

                def keep_events(_):
                    return N_new, y_out, ev_candidate, status_new

                N_new, y_out, ev_new, status_new = lax.cond(
                    jnp.any(crossed),
                    refine_events,
                    keep_events,
                    operand=None,
                )
            else:
                ev_new = ev_prev

            return (
                N_new,
                y_out,
                h_new,
                ep_new,
                ns_new,
                nr_new,
                status_new,
                ev_new,
                J_batch,
                reuse_age_new,
                jac_refresh_count_new,
                jac_reuse_count_new,
                stride_refresh_count_new,
                err_refresh_count_new,
            )

        final = lax.while_loop(cond_fn, body_fn, init_state)
        (
            N_final,
            y_final,
            h_final,
            _,
            n_steps,
            n_reject,
            status,
            ev_final,
            _J_final,
            _reuse_age_final,
            jac_refresh_count,
            jac_reuse_count,
            stride_refresh_count,
            err_refresh_count,
        ) = final
        return (
            y_final,
            N_final,
            h_final,
            n_steps,
            n_reject,
            status,
            ev_final,
            jac_refresh_count,
            jac_reuse_count,
            stride_refresh_count,
            err_refresh_count,
        )

    compiled = jax.jit(_runner, donate_argnums=(0,))
    _BATCH_SOLVE_RUNNER_CACHE[key] = compiled
    return compiled


def _solve_core_batch(
    rhs_fn: Callable,
    y0_batch: jnp.ndarray,
    N_start,
    N_end,
    *,
    event_fn: Optional[Callable] = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 2000,
    h_max: float = 0.5,
    h_min: float = 1e-14,
    event_refine_steps: int = 24,
    jac_fn: Optional[Callable] = None,
    active_indices: Optional[jnp.ndarray] = None,
    batch_linear_solver_backend: str = "jax_tensorized",
    jacobian_reuse_experimental: bool = False,
    jacobian_refresh_stride: int = 1,
    dfdt_fn: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, ...]:
    """Pure-JAX batched adaptive solve for independent trajectories."""
    y0_batch = jnp.asarray(y0_batch, dtype=jnp.float64)
    if y0_batch.ndim != 2:
        raise ValueError(f"y0_batch must have shape (batch, dim); got {y0_batch.shape!r}.")
    if jac_fn is not None and active_indices is not None:
        raise ValueError("jac_fn and active_indices cannot be combined.")
    requested_linear_backend = _canonical_batch_linear_solver_backend(
        batch_linear_solver_backend
    )
    y0_platform = _array_device_platform(y0_batch)
    effective_linear_backend = (
        "jax_tensorized"
        if active_indices is not None and requested_linear_backend == "auto"
        else _resolve_batch_linear_solver_backend(
            requested_linear_backend,
            platform=y0_platform,
        )
    )
    if active_indices is not None and effective_linear_backend != "jax_tensorized":
        raise ValueError(
            "batch_linear_solver_backend applies only to dense batch solves; "
            "active_indices uses the block-sparse Schur solver."
        )
    batch_size = int(y0_batch.shape[0])

    def _broadcast_bounds(value, name: str):
        arr = jnp.asarray(value, dtype=jnp.float64)
        if arr.ndim == 0:
            return jnp.full((batch_size,), arr, dtype=jnp.float64)
        if arr.shape != (batch_size,):
            raise ValueError(f"{name} must be scalar or shape ({batch_size},); got {arr.shape!r}.")
        return arr

    N_start_arr = _broadcast_bounds(N_start, "N_start")
    N_end_arr = _broadcast_bounds(N_end, "N_end")
    runner = _cached_batched_solver_runner(
        rhs_fn,
        event_fn,
        event_refine_steps=event_refine_steps,
        jac_fn=jac_fn,
        active_indices=active_indices,
        state_dim=int(y0_batch.shape[1]),
        batch_linear_solver_backend=effective_linear_backend,
        jacobian_reuse_experimental=bool(jacobian_reuse_experimental)
        and int(jacobian_refresh_stride) > 1,
        dfdt_fn=dfdt_fn,
    )
    y0_work = y0_batch + jnp.zeros_like(y0_batch)
    return runner(
        y0_work,
        N_start_arr,
        N_end_arr,
        jnp.float64(rtol),
        jnp.float64(atol),
        jnp.int32(max_steps),
        jnp.float64(h_min),
        jnp.float64(h_max),
        jnp.int32(max(int(jacobian_refresh_stride), 1)),
    )


def _solve_core_event_masked(
    rhs_fn: Callable,
    y0_batch: jnp.ndarray,
    N_start,
    N_end,
    *,
    event_fn: Callable,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 2000,
    h_max: float = 0.5,
    h_min: float = 1e-14,
    event_refine_steps: int = 24,
    dfdt_fn: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pure-JAX event-aware batched solve for independent trajectories.

    Parameters
    ----------
    y0_batch : array, shape (batch, dim)
        Initial states for each independent solve.
    N_start, N_end : scalar or array shape (batch,)
        Initial and terminal integration bounds. Scalars are broadcast
        across the batch; per-element arrays are accepted for phase-2
        handoff runs where the event time differs per element.

    Returns
    -------
    (y_final, N_final, n_steps, n_reject, status)
        All arrays with leading batch dimension. ``status == 2`` means
        the terminal event was triggered; ``status == -1`` means the
        step size hit ``h_min``; ``status == 0`` means the interval end
        was reached without triggering the terminal event.
    """
    if event_fn is None:
        raise ValueError("_solve_core_event_masked requires event_fn")
    (
        y_final,
        N_final,
        _h_final,
        n_steps,
        n_reject,
        status,
        _ev_final,
        _jac_refresh_count,
        _jac_reuse_count,
        _stride_refresh_count,
        _err_refresh_count,
    ) = _solve_core_batch(
        rhs_fn,
        y0_batch,
        N_start,
        N_end,
        event_fn=event_fn,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        h_max=h_max,
        h_min=h_min,
        event_refine_steps=event_refine_steps,
        dfdt_fn=dfdt_fn,
    )
    return y_final, N_final, n_steps, n_reject, status


def jax_rodas5p_solve_batch(
    rhs_fn: Callable,
    y0_batch: jnp.ndarray,
    N_span: Tuple[float, float],
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 2000,
    h_min: float = 1e-14,
    h_max: float = 0.5,
    event_fn: Optional[Callable] = None,
    event_refine_steps: int = 24,
    jac_fn: Optional[Callable] = None,
    active_indices: Optional[jnp.ndarray] = None,
    batch_linear_solver_backend: str = "jax_tensorized",
    jacobian_reuse_experimental: bool = False,
    jacobian_refresh_stride: int = 1,
    dfdt_fn: Optional[Callable] = None,
) -> JaxBatchSolverResult:
    """Solve a batch of independent ODE systems with shared RHS structure.

    This is the public vectorized companion to ``jax_rodas5p_solve``.
    ``N_span`` may be a pair of scalars or per-element arrays with
    shape ``(batch,)``.

    Parameters
    ----------
    jac_fn : callable, optional
        Explicit Jacobian callback with signature ``jac_fn(N, y) -> J``.
        When provided, bypasses the default dense autodiff path. Mutually
        exclusive with ``active_indices``.
    active_indices : array of int, optional
        Enables the same block-sparse Schur linear solve available in the
        scalar Rodas5P path. The indices mark active variables; all remaining
        variables are treated as passive with ``J_pp = 0``.
    batch_linear_solver_backend : {'jax_tensorized', 'gpu_tensorized', 'auto'}
        Dense batch linear solve policy. ``jax_tensorized`` stays inside XLA
        and is CPU/GPU/JIT/AD safe. ``gpu_tensorized`` is the explicit GPU
        XLA implementation and requires GPU-resident input arrays. ``auto``
        selects the explicit GPU tensorized path for GPU arrays and the
        portable tensorized path otherwise.
    jacobian_reuse_experimental : bool, optional
        When true with ``jacobian_refresh_stride > 1``, reuses the previous
        batch Jacobian inside the XLA loop until the stride or rejection/error
        trigger requests a refresh. This is an experimental throughput knob
        for profiling, not a default numerical contract.
    dfdt_fn : callable, optional
        Partial derivative callback ``dfdt_fn(N, y) -> partial rhs / partial N``
        at fixed ``y``. When omitted, a symmetric finite difference is
        evaluated once per attempted step and batch element.
    """
    if jac_fn is not None and active_indices is not None:
        raise ValueError("jac_fn and active_indices cannot be combined.")
    y0_batch = jnp.asarray(y0_batch, dtype=jnp.float64)
    requested_linear_backend = _canonical_batch_linear_solver_backend(
        batch_linear_solver_backend
    )
    y0_platform = _array_device_platform(y0_batch)
    effective_linear_backend = (
        "jax_tensorized"
        if active_indices is not None and requested_linear_backend == "auto"
        else _resolve_batch_linear_solver_backend(
            requested_linear_backend,
            platform=y0_platform,
        )
    )
    jacobian_refresh_stride = max(int(jacobian_refresh_stride), 1)
    reuse_enabled = bool(jacobian_reuse_experimental) and jacobian_refresh_stride > 1
    (
        y_final,
        N_final,
        h_final,
        n_steps,
        n_reject,
        status,
        ev_final,
        jac_refresh_count,
        jac_reuse_count,
        stride_refresh_count,
        err_refresh_count,
    ) = _solve_core_batch(
        rhs_fn,
        y0_batch,
        N_span[0],
        N_span[1],
        event_fn=event_fn,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        h_max=h_max,
        h_min=h_min,
        event_refine_steps=event_refine_steps,
        jac_fn=jac_fn,
        active_indices=active_indices,
        batch_linear_solver_backend=effective_linear_backend,
        jacobian_reuse_experimental=reuse_enabled,
        jacobian_refresh_stride=jacobian_refresh_stride,
        dfdt_fn=dfdt_fn,
    )
    span_sign = jnp.sign(jnp.asarray(N_span[1], dtype=jnp.float64) - jnp.asarray(N_span[0], dtype=jnp.float64))
    endpoint_reached = (span_sign == 0.0) | (
        span_sign * (N_final - jnp.asarray(N_span[1], dtype=jnp.float64)) >= -1.0e-14
    )
    max_steps_exhausted = (status == _STATUS_SPAN_COMPLETE) & (~endpoint_reached)
    status = jnp.where(
        max_steps_exhausted,
        jnp.asarray(_STATUS_MAX_STEPS_EXHAUSTED, dtype=jnp.int32),
        status,
    )
    event_triggered = (
        (status == _STATUS_EVENT_TRIGGERED)
        if event_fn is not None
        else jnp.zeros_like(status, dtype=jnp.bool_)
    )
    success = status >= 0
    if active_indices is not None:
        linear_solver_policy = "block_sparse_schur"
        jacobian_mode = "block_sparse"
    elif jac_fn is not None:
        if effective_linear_backend == "gpu_tensorized":
            linear_solver_policy = "dense_lu_gpu_tensorized_jax"
        else:
            linear_solver_policy = "dense_lu_tensorized_jax"
        jacobian_mode = "custom"
    else:
        if effective_linear_backend == "gpu_tensorized":
            linear_solver_policy = "dense_lu_gpu_tensorized_jax"
        else:
            linear_solver_policy = "dense_lu_tensorized_jax"
        jacobian_mode = "dense"
    diagnostics = {
        "event_enabled": bool(event_fn is not None),
        "event_refine_steps": int(event_refine_steps),
        "linear_solver_policy": linear_solver_policy,
        "batch_linear_solver_requested": requested_linear_backend,
        "batch_linear_solver_effective": effective_linear_backend,
        "batch_linear_solver_gpu_safe": bool(
            effective_linear_backend in {"jax_tensorized", "gpu_tensorized"}
        ),
        "batch_linear_solver_gpu_resident": bool(
            effective_linear_backend == "gpu_tensorized"
        ),
        "batch_linear_solver_external_call": False,
        "jacobian_mode": jacobian_mode,
        "jacobian_policy": "experimental_reuse" if reuse_enabled else "exact_per_step",
        "jacobian_reuse_enabled": bool(reuse_enabled),
        "jacobian_refresh_stride": int(jacobian_refresh_stride),
        "jacobian_refresh_count": jac_refresh_count,
        "jacobian_reuse_count": jac_reuse_count,
        "jacobian_stride_trigger_count": stride_refresh_count,
        "jacobian_error_trigger_count": err_refresh_count,
        "endpoint_reached_count": int(jax.device_get(jnp.sum(endpoint_reached))),
        "max_steps_exhausted_count": int(jax.device_get(jnp.sum(max_steps_exhausted))),
        "status_contract": _STATUS_CONTRACT,
    }
    return JaxBatchSolverResult(
        y_final=y_final,
        N_final=N_final,
        h_final=h_final,
        n_steps=n_steps,
        n_reject=n_reject,
        success=success,
        status=status,
        diagnostics=diagnostics,
        event_triggered=event_triggered,
    )


# ═══════════════════════════════════════════════════════════════════════
# §6. Non-JIT convenience wrapper (for testing and diagnostics)
# ═══════════════════════════════════════════════════════════════════════

def solve_ode_jax(rhs_fn, y0, N_span, **kwargs):
    """Convenience wrapper matching solve_ode() interface.

    Not JIT-compiled itself (event_fn may use Python control flow),
    but calls JIT-compiled internals.
    """
    return jax_rodas5p_solve(rhs_fn, y0, N_span, **kwargs)
