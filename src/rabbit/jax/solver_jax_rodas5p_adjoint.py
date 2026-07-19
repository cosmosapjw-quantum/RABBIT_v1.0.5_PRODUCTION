"""rabbit.jax.solver_jax_rodas5p_adjoint — Native reverse-mode AD via discrete adjoint replay.

Plan §13 (v2.1 strategic pivot). The in-tree adaptive Rodas5P solver
``rabbit.jax.solver_jax_rodas5p._solve_core`` uses ``lax.while_loop`` with
adaptive step accept/reject — a JAX limitation that blocks reverse-mode
AD. This module mirrors the primary dense stage equation using the shared
tableau coefficients inside a ``lax.scan`` over a *predetermined* step
schedule. Because
``lax.scan`` natively supports reverse-mode AD, ``jax.grad`` /
``jax.jacrev`` work end-to-end without a custom_vjp.

Two execution modes are exposed:

1. ``replay_solve_with_h_tape(rhs_fn, y0, theta, t0, h_tape)`` — caller
   supplies the time-step schedule. Use this when you've already solved
   adaptively (e.g. via ``_solve_core``) and want to re-execute the
   accepted-step trajectory in an AD-friendly form.

2. ``solve_uniform_steps(rhs_fn, y0, theta, t0, t1, n_steps)`` — uniform
   stepping with ``h = (t1 - t0) / n_steps``. Cheaper to set up; the
   trade-off is that error control is implicit in ``n_steps``.

Mathematical correctness
------------------------
Each step mirrors the Rodas5P dense stage solve used by the primary adaptive
path. The sequence of steps is what changes: adaptive ``while_loop`` becomes
a predetermined ``scan``. The frozen parity probe establishes equivalent
action for the same RHS, Jacobian, partial-time derivative, and ``h_n``; the
implementations do not share one step function.

The reverse-mode adjoint produced by ``jax.grad`` on this routine is
the **discrete adjoint** of the Rosenbrock-Wanner method (Hairer-Wanner
Ch. IV.8; Sandu+ 2003), modulo the choice of step schedule. JAX's
automatic VJP through ``lax.scan`` recomputes the per-step Jacobian
during the backward pass (no LU re-use); this is correct but ~3× the
forward solve cost. A future optimisation can cache the LU factor in
the forward residuals — Phase RA-2.

Why we're not introducing a custom_vjp here
-------------------------------------------
The plan considered a ``jax.custom_vjp`` that explicitly stores ``LU``
factors in the forward residuals and replays a transposed-LU adjoint
in the backward. That is a worthwhile follow-on (see
``docs/audit/v2_derivations/rodas5p_discrete_adjoint.md``) but the
``lax.scan`` approach already gives **correct gradients** at full
precision and clean composability with ``jax.jit``, ``jax.vmap``,
``BlackJAX``. We commit the simpler-and-correct version first; the
LU-caching optimisation lands in Phase RA-2 once we profile real BBN
gradient cost.

Reference
---------
- Steinebach 2023, BIT 63:27 — Rodas5P coefficients (already in
  ``solver_jax_rodas5p``).
- Hairer-Wanner, *Solving Ordinary Differential Equations II*, Ch. IV.8
  — Discrete adjoints of Runge-Kutta-Wanner methods.
- Sandu, Daescu, Carmichael 2003, ATCM 27:1577 — Discrete adjoint
  method for Rosenbrock-Wanner methods.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax

jax.config.update("jax_enable_x64", True)

from rabbit.jax.solver_jax_rodas5p import (
    GAMMA,
    A,
    C,
    b_weights,
    btilde,
    c_nodes,
    d_nodes,
    N_STAGES,
    _finite_difference_dfdt,
)


# ═══════════════════════════════════════════════════════════════════════
# §1. Single Rodas5P step that takes theta through a closure
# ═══════════════════════════════════════════════════════════════════════

def _single_step_with_theta(
    rhs_fn: Callable,
    t: jnp.ndarray,
    y: jnp.ndarray,
    h: jnp.ndarray,
    theta: jnp.ndarray,
    dfdt_fn: Optional[Callable] = None,
) -> jnp.ndarray:
    """One Rodas5P step with parameters threaded explicitly.

    Returns ``y_new`` only — the diagnostic ``en`` from the adaptive
    primitive is not needed for the deterministic-schedule replay.

    The Jacobian and explicit time partial are evaluated at the step start.
    ``dfdt_fn(t, y, theta)`` supplies the partial directly when provided;
    otherwise a symmetric, fixed-``y`` finite difference is used.  The
    parameter vector remains a JAX leaf so reverse-mode AD can flow through
    both the right-hand side and the time-partial calculation.
    """
    def f_local(y_local):
        return rhs_fn(t, y_local, theta)

    jac = jax.jacfwd(f_local)(y)
    if dfdt_fn is None:
        def rhs_with_theta(t_local, y_local):
            return rhs_fn(t_local, y_local, theta)

        dfdt = _finite_difference_dfdt(rhs_with_theta, t, y, h)
    else:
        dfdt = jnp.asarray(dfdt_fn(t, y, theta), dtype=y.dtype)

    D = y.shape[0]
    W = jnp.eye(D) / (GAMMA * h) - jac
    lu, piv = jax.scipy.linalg.lu_factor(W)

    def stage_body(i, k_all):
        y_stage = y + A[i] @ k_all
        c_sum = C[i] @ k_all / h
        t_stage = t + c_nodes[i] * h
        rhs_vec = (
            rhs_fn(t_stage, y_stage, theta)
            + c_sum
            + h * d_nodes[i] * dfdt
        )
        k_i = jax.scipy.linalg.lu_solve((lu, piv), rhs_vec)
        return k_all.at[i].set(k_i)

    k_all = lax.fori_loop(0, N_STAGES, stage_body, jnp.zeros((N_STAGES, D)))
    return y + b_weights @ k_all


# ═══════════════════════════════════════════════════════════════════════
# §2. Predetermined-schedule (h-tape) replay solve
# ═══════════════════════════════════════════════════════════════════════

def replay_solve_with_h_tape(
    rhs_fn: Callable,
    y0: jnp.ndarray,
    theta: jnp.ndarray,
    t0: float,
    h_tape: jnp.ndarray,
    dfdt_fn: Optional[Callable] = None,
) -> jnp.ndarray:
    """Run Rodas5P with a precomputed step schedule.

    Parameters
    ----------
    rhs_fn : callable(t, y, theta) -> dy
        Right-hand side. Must be JAX-traceable end-to-end (no
        ``pure_callback``, no host numpy operations).
    y0 : array (D,)
        Initial state.
    theta : array (P,)
        Parameter vector. ``jax.grad`` differentiates through this.
    t0 : float
        Initial time.
    h_tape : array (N,)
        Signed step sizes. Padded entries (e.g. masked beyond the
        accepted-step count) should use ``h_tape == 0`` and will be skipped
        via the ``jnp.where`` mask inside the body.
    dfdt_fn : callable(t, y, theta) -> dy/dt, optional
        Explicit fixed-state partial derivative of ``rhs_fn`` with respect
        to time.  When omitted, each active step uses the symmetric fallback
        described in :func:`_single_step_with_theta`.

    Returns
    -------
    y_final : array (D,)
        State at ``t0 + sum(h_tape)``. Supports ``jax.grad``.
    """
    def step_body(carry, h):
        t, y = carry
        is_active = h != 0.0
        # Use a safe step size for inactive entries to keep the linear
        # solve well-defined. ``cond`` also prevents a padded slot from
        # evaluating the right-hand side or its explicit time partial.
        h_safe = jnp.where(is_active, h, 1.0)
        y_new = lax.cond(
            is_active,
            lambda _: _single_step_with_theta(
                rhs_fn,
                t,
                y,
                h_safe,
                theta,
                dfdt_fn=dfdt_fn,
            ),
            lambda _: y,
            operand=None,
        )
        t_new = jnp.where(is_active, t + h, t)
        return (t_new, y_new), None

    (_, y_final), _ = lax.scan(step_body, (jnp.asarray(t0), y0), h_tape)
    return y_final


def solve_uniform_steps(
    rhs_fn: Callable,
    y0: jnp.ndarray,
    theta: jnp.ndarray,
    t0: float,
    t1: float,
    n_steps: int,
    *,
    dfdt_fn: Optional[Callable] = None,
) -> jnp.ndarray:
    """Run Rodas5P with a uniform step schedule ``h = (t1-t0)/n_steps``.

    Useful when the time variable is already log-scale (e.g. ``N = ln(a)``
    in BBN, where uniform-N spacing corresponds to log-spaced ``a``).
    For physical-time stiff systems where dynamics span many decades,
    use :func:`solve_log_steps` instead.

    Choose ``n_steps`` large enough to capture the stiffest dynamics
    (typically ~600-1024 for BBN-scale integrations of N = ln(a) over
    ~10 e-folds).
    """
    h = (float(t1) - float(t0)) / int(n_steps)
    h_tape = jnp.full((int(n_steps),), h)
    return replay_solve_with_h_tape(
        rhs_fn,
        y0,
        theta,
        t0,
        h_tape,
        dfdt_fn=dfdt_fn,
    )


def solve_log_steps(
    rhs_fn: Callable,
    y0: jnp.ndarray,
    theta: jnp.ndarray,
    t0: float,
    t1: float,
    n_steps: int,
    *,
    t_min_log: Optional[float] = None,
    dfdt_fn: Optional[Callable] = None,
) -> jnp.ndarray:
    """Run Rodas5P with a log-spaced time grid.

    Recommended for stiff systems whose dynamics span multiple decades
    in physical time (e.g. Robertson, plasma evolution, neutrino
    decoupling). The grid starts at ``t0``, jumps to
    ``t_min_log`` (default ``max(t1 * 1e-6, t0 + 1e-12)``), then
    geometrically progresses to ``t1``.

    Parameters
    ----------
    n_steps : int
        Total step count (including the initial linear "ignition" step
        from ``t0`` to ``t_min_log``).
    t_min_log : float, optional
        Start of log-spacing; defaults to ``max(t1 * 1e-6, t0 + 1e-12)``.
    """
    import numpy as np
    t0_f, t1_f = float(t0), float(t1)
    if t_min_log is None:
        t_min_log = max(t1_f * 1.0e-6, t0_f + 1.0e-12)
    if t_min_log <= t0_f:
        raise ValueError(
            f"t_min_log ({t_min_log}) must be greater than t0 ({t0_f})"
        )
    t_log = np.geomspace(t_min_log, t1_f, int(n_steps))
    t_grid = np.concatenate(([t0_f], t_log))
    h_tape = jnp.asarray(np.diff(t_grid))
    return replay_solve_with_h_tape(
        rhs_fn,
        y0,
        theta,
        t0,
        h_tape,
        dfdt_fn=dfdt_fn,
    )


# ═══════════════════════════════════════════════════════════════════════
# §3. h-tape extraction from the primary adaptive solver
# ═══════════════════════════════════════════════════════════════════════

def extract_h_tape_from_adaptive_solve(
    rhs_fn: Callable,
    y0: jnp.ndarray,
    theta: jnp.ndarray,
    t0: float,
    t1: float,
    *,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    max_steps: int = 4096,
    dfdt_fn: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, int]:
    """Build an approximate offline step schedule with adaptive step doubling.

    The primary adaptive ``_solve_core`` does not expose its internal
    ``h`` trajectory. This non-JIT shim uses a separate step-doubling error
    estimate and controller to produce a practical replay schedule; it does
    not reproduce ``_solve_core`` decisions bitwise.

    Returns
    -------
    h_tape : array (max_steps,) padded with zeros beyond ``n_accepted``
    n_accepted : int

    Notes
    -----
    Use this only when an approximate offline schedule is acceptable. For
    controlled parity work, record the primary solver's actual accepted
    steps externally or use a pre-registered uniform schedule.
    """
    # Lightweight Python-level adaptive loop using the same step body
    import numpy as np

    h = (float(t1) - float(t0)) / 64.0  # initial step heuristic
    t = float(t0)
    y = jnp.asarray(y0)
    accepted_h = []
    n_step = 0

    while t < float(t1) and n_step < max_steps:
        h = min(h, float(t1) - t)
        y_new = _single_step_with_theta(
            rhs_fn,
            jnp.asarray(t),
            y,
            jnp.asarray(h),
            theta,
            dfdt_fn=dfdt_fn,
        )
        # Crude error estimate: compare to half-step (cost ~3× per step,
        # acceptable since this is offline tape-extraction).
        y_half_a = _single_step_with_theta(
            rhs_fn,
            jnp.asarray(t),
            y,
            jnp.asarray(h * 0.5),
            theta,
            dfdt_fn=dfdt_fn,
        )
        y_half_b = _single_step_with_theta(
            rhs_fn,
            jnp.asarray(t + h * 0.5),
            y_half_a,
            jnp.asarray(h * 0.5),
            theta,
            dfdt_fn=dfdt_fn,
        )
        err = float(jnp.max(jnp.abs(y_new - y_half_b)
                             / (atol + rtol * jnp.abs(y_half_b))))
        if err < 1.0 or h < 1.0e-10:
            accepted_h.append(h)
            t = t + h
            y = y_half_b  # use the more accurate half-step result
            # Adaptively grow h: PI controller heuristic
            factor = min(2.0, max(0.5, 0.9 / max(err, 1.0e-3) ** 0.2))
            h = h * factor
            n_step += 1
        else:
            h = h * max(0.2, 0.9 / err ** 0.2)

    h_tape_np = np.zeros(max_steps, dtype=np.float64)
    n = len(accepted_h)
    h_tape_np[:n] = accepted_h
    return jnp.asarray(h_tape_np), n


# ═══════════════════════════════════════════════════════════════════════
# §4. High-level convenience: forward + AD-able solve in one call
# ═══════════════════════════════════════════════════════════════════════

def solve_with_adjoint(
    rhs_fn: Callable,
    y0: jnp.ndarray,
    theta: jnp.ndarray,
    t0: float,
    t1: float,
    *,
    n_steps: int = 1024,
    h_tape: Optional[jnp.ndarray] = None,
    dfdt_fn: Optional[Callable] = None,
) -> jnp.ndarray:
    """Top-level entry: returns ``y_final`` with native ``jax.grad`` support.

    By default uses uniform stepping with ``n_steps``. Pass ``h_tape`` to
    use a precomputed adaptive schedule.
    """
    if h_tape is not None:
        return replay_solve_with_h_tape(
            rhs_fn,
            y0,
            theta,
            t0,
            h_tape,
            dfdt_fn=dfdt_fn,
        )
    return solve_uniform_steps(
        rhs_fn,
        y0,
        theta,
        t0,
        t1,
        n_steps,
        dfdt_fn=dfdt_fn,
    )
