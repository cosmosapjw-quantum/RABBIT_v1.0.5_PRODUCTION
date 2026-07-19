"""
rabbit.jax.gradient_bridge — Differentiable BBN forward model.

Three backends are exposed:

1. ``bridge_mode='rosenbrock_native'`` (v3.0 default — Phase A):
   Rodas5P discrete adjoint replay via ``lax.scan`` over a predetermined
   h-tape. See :mod:`rabbit.jax.solver_jax_rodas5p_adjoint` (Phase RA from
   v2 plan §13). Reuses the same hand-tuned Rodas5P stage solver as
   production but in an AD-friendly form. Cost ≈ 2-3× a single forward
   solve. Recommended for production inference.

2. ``bridge_mode='diffrax_native'`` (Phase α — cross-validation reference):
   Diffrax ``Kvaerno5`` + ``RecursiveCheckpointAdjoint(checkpoints=16)`` true
   reverse-mode adjoint. Cost ``≈ 2× a single forward solve regardless of
   parameter dimension``; O(log N_steps) memory. Used as the cross-validation
   reference for the production rosenbrock_native path (1e-6 parity gate).

3. ``bridge_mode='fd_legacy'`` (legacy — deprecated):
   ``jax.custom_vjp`` whose backward pass uses central finite differences
   through the Rodas5P forward solver. Cost ``2n`` forward solves per
   gradient. Retained for back-compat tests; new code should use
   rosenbrock_native.

Problem (legacy fd_legacy path)
-------------------------------
Reverse-mode AD through ``lax.while_loop`` with dynamic stop conditions is
not supported in JAX. Forward-mode tangents diverge over ~800 adaptive steps
due to step-accept/reject discontinuities. The ``fd_legacy`` bridge solves
this by abandoning AD through the loop and using FD on the complete forward
solve as a black box.

Solution (diffrax_native path)
------------------------------
Diffrax provides a discretize-then-optimise reverse-mode adjoint via
``RecursiveCheckpointAdjoint``. Native ``jax.grad`` propagates through the
entire integration. See ``rabbit.jax.solver_diffrax_canonical``.

Reference: Diffrax (Kidger 2021), ``smsharma/clax``,
DISCO-DJ (arXiv:2311.03291), ABCMB (arXiv:2602.15104).

Public API
----------
    differentiable_solve         — fd_legacy custom_vjp (legacy)
    make_differentiable_solve    — fd_legacy factory (legacy)
    make_differentiable_solve_diffrax — diffrax_native factory (Phase α)
    make_log_posterior            — log-posterior factory; bridge_mode arg
    gradient_check                — validate gradient against direct FD
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import jax.lax as lax

jax.config.update("jax_enable_x64", True)

from rabbit.jax.solver_jax_rodas5p import _solve_core


# ═══════════════════════════════════════════════════════════════════════
# §1. Core: differentiable solve via custom_vjp
# ═══════════════════════════════════════════════════════════════════════

# Default FD epsilon (optimal for double precision central FD: eps ≈ 10⁻⁵)
_DEFAULT_EPS = 1e-5


def make_differentiable_solve(
    rhs_factory: Callable,
    y0: jnp.ndarray,
    N_start: float,
    N_end: float,
    param_indices: Sequence[int] = (0, 1),
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 2000,
    h_max: float = 0.5,
    fd_eps: float = _DEFAULT_EPS,
):
    """Create a custom_vjp-wrapped differentiable solve function.

    Parameters
    ----------
    rhs_factory : callable(params_array) → rhs_fn(N, y)
        Given a 1D parameter array, returns the ODE RHS function.
    y0 : array (N_dim,)
        Initial conditions.
    N_start, N_end : float
        Integration bounds.
    param_indices : sequence of int
        Which elements of the parameter array to differentiate through.
        Others are treated as static (no gradient computed).
    rtol, atol, max_steps, h_max : solver settings.
    fd_eps : float
        Relative perturbation for central FD gradient.

    Returns
    -------
    solve_fn : callable(params) → y_final
        A function that maps parameter array → final state vector,
        and supports jax.grad / jax.value_and_grad.
    """

    def _raw_solve(params):
        """Forward solve (no grad)."""
        rhs_fn = rhs_factory(params)
        y_final, _, _, _ = _solve_core(
            rhs_fn, y0, N_start, N_end,
            rtol=rtol, atol=atol, max_steps=max_steps, h_max=h_max,
        )
        return y_final

    @jax.custom_vjp
    def solve_fn(params):
        """Forward solve with custom_vjp for reverse-mode gradient."""
        return _raw_solve(params)

    def solve_fwd(params):
        """Forward pass: compute y_final and save params as residual."""
        y_final = _raw_solve(params)
        return y_final, params  # residuals for backward

    def solve_bwd(params, g):
        """Backward pass: central FD gradient.

        g : cotangent (same shape as y_final), typically from upstream loss.
        Returns: cotangent for params (same shape as params).
        """
        n_params = params.shape[0]
        grad_params = jnp.zeros(n_params)

        for i in param_indices:
            # Central FD: perturb param[i] by ±eps
            eps_abs = jnp.maximum(jnp.abs(params[i]) * fd_eps, fd_eps * 1e-10)
            params_plus = params.at[i].set(params[i] + eps_abs)
            params_minus = params.at[i].set(params[i] - eps_abs)

            y_plus = _raw_solve(params_plus)
            y_minus = _raw_solve(params_minus)

            # dy/dθ_i ≈ (y⁺ − y⁻) / (2ε)
            dy_dtheta = (y_plus - y_minus) / (2.0 * eps_abs)

            # Chain rule: ∂L/∂θ_i = g · (∂y/∂θ_i)
            grad_i = jnp.sum(g * dy_dtheta)
            grad_params = grad_params.at[i].set(grad_i)

        return (grad_params,)

    solve_fn.defvjp(solve_fwd, solve_bwd)
    return solve_fn


# ═══════════════════════════════════════════════════════════════════════
# §2. Convenience: BBN-like forward model
# ═══════════════════════════════════════════════════════════════════════

def differentiable_solve(
    rhs_factory: Callable,
    y0: jnp.ndarray,
    N_span: Tuple[float, float],
    params: jnp.ndarray,
    param_indices: Sequence[int] = (0, 1),
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 2000,
    h_max: float = 0.5,
    fd_eps: float = _DEFAULT_EPS,
) -> jnp.ndarray:
    """One-shot differentiable ODE solve.

    Parameters
    ----------
    rhs_factory : callable(params) → rhs_fn(N, y)
    y0 : array (N_dim,)
    N_span : (N_start, N_end)
    params : array (n_params,)
        Parameter vector. Gradients computed for param_indices.
    param_indices : which params to differentiate.
    Other args: solver settings.

    Returns
    -------
    y_final : array (N_dim,)
        Supports jax.grad upstream.
    """
    solve_fn = make_differentiable_solve(
        rhs_factory, y0, N_span[0], N_span[1],
        param_indices=param_indices,
        rtol=rtol, atol=atol, max_steps=max_steps, h_max=h_max,
        fd_eps=fd_eps,
    )
    return solve_fn(params)


# ═══════════════════════════════════════════════════════════════════════
# §3. Log-posterior for HMC
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BBNObservation:
    """Single observational constraint."""
    name: str
    value: float
    sigma: float


# Standard observations (Paper I)
OBS_YP = BBNObservation("Y_p", 0.2449, 0.0040)
OBS_DH = BBNObservation("D/H", 2.547e-5, 0.029e-5)


# ═══════════════════════════════════════════════════════════════════════
# §3a. Diffrax-native factory (Phase α — true reverse-mode adjoint)
# ═══════════════════════════════════════════════════════════════════════

def make_differentiable_solve_diffrax(
    rhs_fn: Callable,
    y0: jnp.ndarray,
    N_span: Tuple[float, float],
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 8192,
    checkpoints: int = 16,
    solver: str = "kvaerno5",
) -> Callable:
    """Build a Diffrax-adjoint solve function with native ``jax.grad`` support.

    Unlike ``make_differentiable_solve``, this does *not* use ``custom_vjp``
    — Diffrax's ``RecursiveCheckpointAdjoint`` provides true reverse-mode
    AD with O(log N_steps) memory. Cost is ≈ 2× a single forward solve
    regardless of parameter count.

    Parameters
    ----------
    rhs_fn : callable(t, y, theta) -> dy
        Modern Diffrax RHS signature. ``theta`` is the flat parameter vector
        that ``jax.grad`` will differentiate through.
    y0 : array (D,)
        Initial state.
    N_span : (N_start, N_end)
        Integration bounds.
    rtol, atol, max_steps, checkpoints, solver : Diffrax knobs.
        See ``rabbit.jax.solver_diffrax_canonical`` for documentation.

    Returns
    -------
    solve_fn : callable(theta) -> y_final
        Function returning the final state. Supports
        ``jax.grad`` / ``jax.value_and_grad`` / ``jax.jacrev`` over ``theta``
        without custom_vjp.

    Notes
    -----
    Hallucination guard: any ``rhs_fn`` containing ``jax.pure_callback``,
    ``jax.debug.print``, or host-side numpy operations will silently
    succeed in forward mode but yield wrong / NaN gradients. Validate
    with ``gradient_check`` against Richardson FD before trusting in
    inference.
    """
    from rabbit.jax.solver_diffrax_canonical import diffrax_adjoint_solve

    t_span = (float(N_span[0]), float(N_span[1]))

    def solve_fn(theta):
        return diffrax_adjoint_solve(
            rhs_fn,
            y0,
            t_span,
            theta,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            checkpoints=checkpoints,
            solver=solver,
        )

    return solve_fn


# ═══════════════════════════════════════════════════════════════════════
# §3b. Rosenbrock-native factory (Phase RA — production default in v3.0)
# ═══════════════════════════════════════════════════════════════════════

def make_differentiable_solve_rodas5p_native(
    rhs_fn: Callable,
    y0: jnp.ndarray,
    N_span: Tuple[float, float],
    *,
    n_steps: int = 800,
    schedule: str = "uniform",
    t_min_log: Optional[float] = None,
) -> Callable:
    """Build a Rodas5P-discrete-adjoint solve function.

    Phase RA discrete adjoint replay: run Rodas5P stages inside a
    ``lax.scan`` over a predetermined h-tape so ``jax.grad`` works
    natively. Reuses the production stage solver from
    :mod:`rabbit.jax.solver_jax_rodas5p_adjoint`.

    Parameters
    ----------
    rhs_fn : callable(t, y, theta) -> dy
        Modern signature; ``theta`` is the flat parameter vector.
    y0 : array (D,)
        Initial state.
    N_span : (N_start, N_end)
        Integration bounds.
    n_steps : int
        Number of fixed steps in the replay schedule. Default 800
        matches the BBN-typical accepted-step count for the canonical
        Type-I forward.
    schedule : {'uniform', 'log'}
        Step spacing. 'uniform' is correct when the time variable is
        already log-scale (e.g. ``N = ln(a)``). 'log' is preferred for
        physical-time stiff systems whose dynamics span many decades.
    t_min_log : float, optional
        For schedule='log', the start of the log-spaced grid. Default
        ``max(t1 * 1e-6, t0 + 1e-12)``.

    Returns
    -------
    solve_fn : callable(theta) -> y_final
        Function returning the final state. Supports ``jax.grad`` /
        ``jax.value_and_grad`` / ``jax.jacrev`` over ``theta`` natively
        (no custom_vjp required).

    Notes
    -----
    Same forward-mode hallucination guards as
    :func:`make_differentiable_solve_diffrax` apply: any ``rhs_fn``
    containing ``jax.pure_callback`` or host-side numpy operations
    will silently succeed in forward mode but yield wrong / NaN
    gradients.
    """
    from rabbit.jax.solver_jax_rodas5p_adjoint import (
        solve_uniform_steps,
        solve_log_steps,
    )

    t0_f, t1_f = float(N_span[0]), float(N_span[1])

    def solve_fn(theta):
        if schedule == "uniform":
            return solve_uniform_steps(
                rhs_fn, y0, theta, t0_f, t1_f, int(n_steps),
            )
        if schedule == "log":
            return solve_log_steps(
                rhs_fn, y0, theta, t0_f, t1_f, int(n_steps),
                t_min_log=t_min_log,
            )
        raise ValueError(
            f"unknown schedule={schedule!r}; choose 'uniform' or 'log'"
        )

    return solve_fn


# Default bridge mode for new code. v3.0 Phase A flips this from
# 'fd_legacy' to 'rosenbrock_native'.
DEFAULT_BRIDGE_MODE = "rosenbrock_native"

# Valid bridge modes; checked by the make_log_posterior dispatcher and
# tests that lock the production default.
VALID_BRIDGE_MODES = ("rosenbrock_native", "diffrax_native", "fd_legacy")


def make_log_posterior(
    rhs_factory: Callable,
    y0: jnp.ndarray,
    N_span: Tuple[float, float],
    observable_indices: dict,
    observations: Sequence[BBNObservation],
    param_indices: Sequence[int] = (0, 1),
    prior_fn: Optional[Callable] = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 2000,
    h_max: float = 0.5,
    fd_eps: float = _DEFAULT_EPS,
    bridge_mode: Optional[str] = None,
    diffrax_checkpoints: int = 16,
    diffrax_solver: str = "kvaerno5",
    rodas5p_n_steps: int = 800,
    rodas5p_schedule: str = "uniform",
) -> Callable:
    """Create a differentiable log-posterior for HMC.

    Parameters
    ----------
    rhs_factory : callable
        - ``bridge_mode='fd_legacy'``: ``rhs_factory(params) -> rhs_fn(N, y)``
          (legacy two-step convention).
        - ``bridge_mode='diffrax_native'``: ``rhs_factory(t, y, theta) -> dy``
          (modern Diffrax convention; the "factory" is itself the RHS).
        - ``bridge_mode='rosenbrock_native'``: same modern signature as
          diffrax_native (``rhs_factory`` is the RHS itself).
    y0 : initial conditions
    N_span : integration bounds
    observable_indices : dict mapping obs name → index in y_final
    observations : list of BBNObservation
    param_indices : which params have gradients (fd_legacy only)
    prior_fn : callable(params) → log_prior (optional)
    bridge_mode : {'rosenbrock_native', 'diffrax_native', 'fd_legacy'}, optional
        Selects backward-pass implementation. ``None`` (default) routes to
        :data:`DEFAULT_BRIDGE_MODE` which is ``'rosenbrock_native'`` in v3.0.
        ``'diffrax_native'`` is the cross-validation reference;
        ``'fd_legacy'`` is retained for back-compat.
    diffrax_checkpoints, diffrax_solver : passed through to Diffrax.
    rodas5p_n_steps : int
        Step count for the Rodas5P-native h-tape replay (default 800).
    rodas5p_schedule : {'uniform', 'log'}
        Step spacing for rosenbrock_native.
    Other args: solver settings (fd_legacy path only for h_max, fd_eps).

    Returns
    -------
    log_post_fn : callable(params) → float
        Supports ``jax.grad`` for HMC. The returned closure semantics depend
        on ``bridge_mode``.
    """
    if bridge_mode is None:
        bridge_mode = DEFAULT_BRIDGE_MODE
    if bridge_mode not in VALID_BRIDGE_MODES:
        raise ValueError(
            f"unknown bridge_mode={bridge_mode!r}; "
            f"choose from {set(VALID_BRIDGE_MODES)!r}"
        )
    if bridge_mode == "rosenbrock_native":
        solve_fn = make_differentiable_solve_rodas5p_native(
            rhs_factory,  # modern signature: rhs_factory is itself rhs_fn(t, y, theta)
            y0,
            N_span,
            n_steps=rodas5p_n_steps,
            schedule=rodas5p_schedule,
        )
    elif bridge_mode == "diffrax_native":
        solve_fn = make_differentiable_solve_diffrax(
            rhs_factory,  # in this mode, rhs_factory IS the rhs_fn(t, y, theta)
            y0,
            N_span,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            checkpoints=diffrax_checkpoints,
            solver=diffrax_solver,
        )
    elif bridge_mode == "fd_legacy":
        solve_fn = make_differentiable_solve(
            rhs_factory, y0, N_span[0], N_span[1],
            param_indices=param_indices,
            rtol=rtol, atol=atol, max_steps=max_steps, h_max=h_max,
            fd_eps=fd_eps,
        )
    else:
        # Unreachable: caught by VALID_BRIDGE_MODES check above.
        raise AssertionError(bridge_mode)

    def log_posterior(params):
        y_final = solve_fn(params)

        # Gaussian log-likelihood
        log_lik = 0.0
        for obs in observations:
            if obs.name in observable_indices:
                idx = observable_indices[obs.name]
                predicted = y_final[idx]
                log_lik = log_lik - 0.5 * ((predicted - obs.value) / obs.sigma) ** 2

        # Prior
        log_prior = 0.0
        if prior_fn is not None:
            log_prior = prior_fn(params)

        return log_lik + log_prior

    return log_posterior


# ═══════════════════════════════════════════════════════════════════════
# §4. Gradient validation utility
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GradientCheckResult:
    """Result of gradient validation."""
    param_names: Tuple[str, ...]
    grad_custom_vjp: jnp.ndarray    # gradient from custom_vjp
    grad_direct_fd: jnp.ndarray     # gradient from independent FD
    rel_errors: jnp.ndarray          # relative error per parameter
    max_rel_error: float
    passed: bool                      # all rel errors < threshold


def gradient_check(
    solve_fn_or_factory,
    params: jnp.ndarray,
    loss_fn: Callable,
    param_names: Optional[Tuple[str, ...]] = None,
    fd_eps: float = 1e-6,
    threshold: float = 0.01,
    **factory_kwargs,
) -> GradientCheckResult:
    """Validate custom_vjp gradient against independent central FD.

    Parameters
    ----------
    solve_fn_or_factory : callable
        Either a custom_vjp-wrapped solve function, or make_differentiable_solve args.
    params : array (n_params,)
    loss_fn : callable(y_final) → scalar
        Loss function applied to solver output.
    param_names : optional tuple of parameter name strings.
    fd_eps : perturbation for independent FD check.
    threshold : max relative error for pass.

    Returns
    -------
    GradientCheckResult
    """
    if param_names is None:
        param_names = tuple(f"θ_{i}" for i in range(params.shape[0]))

    # Gradient via custom_vjp (jax.grad through loss ∘ solve)
    if callable(solve_fn_or_factory):
        full_fn = lambda p: loss_fn(solve_fn_or_factory(p))
    else:
        raise ValueError("solve_fn_or_factory must be callable")

    grad_vjp = jax.grad(full_fn)(params)

    # Independent FD gradient
    n = params.shape[0]
    grad_fd = jnp.zeros(n)
    for i in range(n):
        eps_abs = jnp.maximum(jnp.abs(params[i]) * fd_eps, fd_eps * 1e-10)
        p_plus = params.at[i].set(params[i] + eps_abs)
        p_minus = params.at[i].set(params[i] - eps_abs)
        L_plus = full_fn(p_plus)
        L_minus = full_fn(p_minus)
        grad_fd = grad_fd.at[i].set((L_plus - L_minus) / (2.0 * eps_abs))

    # Relative errors with absolute fallback for near-zero gradients
    abs_diff = jnp.abs(grad_vjp - grad_fd)
    scale = jnp.maximum(jnp.abs(grad_vjp), jnp.abs(grad_fd))
    # If both gradients are very small, use absolute error criterion
    abs_threshold = 1e-8  # gradients below this are "effectively zero"
    both_small = scale < abs_threshold
    rel_errors = jnp.where(both_small, 0.0, abs_diff / jnp.maximum(scale, 1e-100))
    max_rel = float(jnp.max(rel_errors))

    return GradientCheckResult(
        param_names=param_names,
        grad_custom_vjp=grad_vjp,
        grad_direct_fd=grad_fd,
        rel_errors=rel_errors,
        max_rel_error=max_rel,
        passed=(max_rel < threshold),
    )
