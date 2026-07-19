"""rabbit.jax.solver_diffrax_canonical — Diffrax-based reverse-mode AD path.

Physics meaning
---------------
End-to-end differentiable stiff-ODE solve. Wraps the existing canonical RHS
(any callable ``rhs_fn(t, y, args) -> dy`` whose `args` is a flat parameter
array) in Diffrax's ``Kvaerno5`` SDIRK solver with a PID step controller and
``RecursiveCheckpointAdjoint`` so that ``jax.grad`` propagates through the
entire integration via discretize-then-optimise reverse-mode AD with
O(log N_steps) memory.

Why this module exists
----------------------
The legacy ``rabbit.jax.gradient_bridge`` uses a custom_vjp whose backward is
*central finite differences* over the full forward solve. That costs ``2n``
forward solves per gradient and is not native AD. Diffrax with
``RecursiveCheckpointAdjoint`` gives a true reverse-mode adjoint that scales
with one extra solve regardless of parameter dimension.

Reference
---------
- Diffrax: Kidger 2021; https://docs.kidger.site/diffrax/
- ``RecursiveCheckpointAdjoint`` is the production default in ``smsharma/clax``
  and ``DISCO-DJ`` (arXiv:2311.03291). For BBN, see ``ABCMB``
  (arXiv:2602.15104).

Public API
----------
- ``diffrax_adjoint_solve(rhs_fn, y0, t_span, theta, ...)`` — a single
  reverse-mode-AD-friendly solve. Supports ``jax.grad(solve)(theta)`` directly.
- ``DEFAULT_CHECKPOINTS`` — 16, balances memory vs. recompute (per
  ``docs.kidger.site/diffrax/api/adjoints/`` recommendation for typical
  BBN-shaped integrations of ~800 accepted steps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import diffrax as dfx

jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# §1. Defaults
# ═══════════════════════════════════════════════════════════════════════

# Checkpoint count tuned for BBN-shape integrations (~800 accepted steps).
# Memory ∝ checkpoints; recompute work ∝ N_steps / checkpoints.
DEFAULT_CHECKPOINTS = 16

# Production-grade tolerances; tighten to (1e-10, 1e-12) for promotion gates.
DEFAULT_RTOL = 1.0e-8
DEFAULT_ATOL = 1.0e-10

# Maximum solver steps. BBN typically uses 200–800; budget 8x as headroom.
DEFAULT_MAX_STEPS = 8192


# ═══════════════════════════════════════════════════════════════════════
# §2. Result container
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DiffraxAdjointResult:
    """Diagnostics from a Diffrax adjoint-mode solve.

    Notes
    -----
    ``y_final`` is the only field that participates in autodiff; the rest
    are post-solve diagnostics extracted on the host.
    """
    y_final: jnp.ndarray
    t_final: float
    n_accepted_steps: int
    n_rejected_steps: int
    success: bool


# ═══════════════════════════════════════════════════════════════════════
# §3. Core differentiable solve
# ═══════════════════════════════════════════════════════════════════════

def diffrax_adjoint_solve(
    rhs_fn: Callable,
    y0: jnp.ndarray,
    t_span: Tuple[float, float],
    theta: jnp.ndarray,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_steps: int = DEFAULT_MAX_STEPS,
    dt0: Optional[float] = None,
    checkpoints: int = DEFAULT_CHECKPOINTS,
    solver: str = "kvaerno5",
) -> jnp.ndarray:
    """Solve the stiff IVP and return ``y_final`` as a reverse-mode AD primitive.

    Parameters
    ----------
    rhs_fn : callable(t, y, theta) -> dy
        Right-hand side. Must be JAX-traceable (no host callbacks, no
        Python control flow on traced quantities). ``theta`` is the flat
        parameter vector that ``jax.grad`` will differentiate through.
    y0 : array (D,)
        Initial state. Must be a concrete JAX array; shape participates in
        compile-time specialization.
    t_span : (t_start, t_end)
        Integration bounds (inclusive).
    theta : array (P,)
        Flat parameter vector.
    rtol, atol : floats
        PID step-controller tolerances.
    max_steps : int
        Hard upper bound on accepted steps; raises if exceeded.
    dt0 : optional float
        Initial step. Defaults to ``min(0.01, 1e-3 * (t_end - t_start))``.
    checkpoints : int
        ``RecursiveCheckpointAdjoint`` checkpoint count. Memory ∝
        ``checkpoints``; recompute ∝ ``N_steps / checkpoints``.
    solver : {'kvaerno3', 'kvaerno4', 'kvaerno5', 'tsit5'}
        ``kvaerno5`` is L-stable SDIRK; preferred for BBN stiff systems.
        ``tsit5`` is a non-stiff explicit method retained only for adjoint
        cross-checks.

    Returns
    -------
    y_final : array (D,)
        Solution at ``t_end``. Supports ``jax.grad`` over ``theta``.

    Notes
    -----
    The function is intentionally *not* jitted at the module level so that
    callers can choose static-vs-dynamic argument boundaries appropriate to
    their use case (e.g., Fisher matrices benefit from outer-jit; one-shot
    HMC gradients do not). For the common case, wrap with
    ``jax.jit(..., static_argnames=('rhs_fn',))``.
    """
    t0 = float(t_span[0])
    t1 = float(t_span[1])
    if dt0 is None:
        dt0 = float(min(0.01, 1.0e-3 * (t1 - t0)))

    solvers = {
        "kvaerno3": dfx.Kvaerno3,
        "kvaerno4": dfx.Kvaerno4,
        "kvaerno5": dfx.Kvaerno5,
        "tsit5": dfx.Tsit5,
    }
    if solver not in solvers:
        raise ValueError(
            f"unknown solver={solver!r}; choose from {sorted(solvers)}"
        )
    solver_obj = solvers[solver]()

    term = dfx.ODETerm(rhs_fn)
    controller = dfx.PIDController(rtol=rtol, atol=atol)
    saveat = dfx.SaveAt(t1=True)
    adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints=checkpoints)

    sol = dfx.diffeqsolve(
        term,
        solver_obj,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=theta,
        stepsize_controller=controller,
        saveat=saveat,
        adjoint=adjoint,
        max_steps=max_steps,
    )
    # sol.ys has shape (1, D) because SaveAt(t1=True)
    return sol.ys[-1]


def diffrax_adjoint_solve_with_diagnostics(
    rhs_fn: Callable,
    y0: jnp.ndarray,
    t_span: Tuple[float, float],
    theta: jnp.ndarray,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_steps: int = DEFAULT_MAX_STEPS,
    dt0: Optional[float] = None,
    checkpoints: int = DEFAULT_CHECKPOINTS,
    solver: str = "kvaerno5",
) -> DiffraxAdjointResult:
    """Same as ``diffrax_adjoint_solve`` but returns ``DiffraxAdjointResult``.

    Use the diagnostic variant for benchmarking / one-shot calls; use the
    plain ``diffrax_adjoint_solve`` for autodiff (host-side step counts
    cannot be differentiated through).
    """
    t0 = float(t_span[0])
    t1 = float(t_span[1])
    if dt0 is None:
        dt0 = float(min(0.01, 1.0e-3 * (t1 - t0)))

    solvers = {
        "kvaerno3": dfx.Kvaerno3,
        "kvaerno4": dfx.Kvaerno4,
        "kvaerno5": dfx.Kvaerno5,
        "tsit5": dfx.Tsit5,
    }
    solver_obj = solvers[solver]()

    term = dfx.ODETerm(rhs_fn)
    controller = dfx.PIDController(rtol=rtol, atol=atol)
    saveat = dfx.SaveAt(t1=True)
    adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints=checkpoints)

    sol = dfx.diffeqsolve(
        term,
        solver_obj,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=theta,
        stepsize_controller=controller,
        saveat=saveat,
        adjoint=adjoint,
        max_steps=max_steps,
        throw=False,
    )
    y_final = sol.ys[-1] if sol.ys is not None else y0
    t_final = float(sol.ts[-1]) if sol.ts is not None else t1
    accepted = int(sol.stats.get("num_accepted_steps", 0))
    rejected = int(sol.stats.get("num_rejected_steps", 0))
    success = bool(sol.result == dfx.RESULTS.successful)
    return DiffraxAdjointResult(
        y_final=y_final,
        t_final=t_final,
        n_accepted_steps=accepted,
        n_rejected_steps=rejected,
        success=success,
    )


# ═══════════════════════════════════════════════════════════════════════
# §4. Convenience factories
# ═══════════════════════════════════════════════════════════════════════

def make_diffrax_solver(
    rhs_fn: Callable,
    y0: jnp.ndarray,
    t_span: Tuple[float, float],
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_steps: int = DEFAULT_MAX_STEPS,
    checkpoints: int = DEFAULT_CHECKPOINTS,
    solver: str = "kvaerno5",
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Curry the solver to a single-argument function ``solve(theta) -> y_final``.

    Useful when wrapping for HMC / NUTS, where the sampler expects a callable
    that takes only the parameter vector.
    """
    def solve(theta: jnp.ndarray) -> jnp.ndarray:
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
    return solve
