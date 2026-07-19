"""
rabbit.solver.rodas5p_adapter — scipy.solve_ivp-shaped wrapper around the
in-tree Rosenbrock-W solver (``rabbit.solver.rodas5p``).

The Type-I driver (``full_coupled_typeI``) consumes a ``scipy.integrate.solve_ivp``
result object: it reads ``sol.t_events`` (target-reached gate), ``sol.y[:, -1]``,
``sol.t[-1]``, ``sol.status``, ``sol.success`` and ``sol.message`` and feeds the
last four to ``classify_solve_ivp_result``. ``rabbit.solver.rodas5p.solve``
returns a ``Rodas5PResult`` that carries the raw state, exact attempt counters,
and a failure reason but has no ``.status`` or ``.t_events``. This adapter
(BD615) synthesizes those scipy fields while preserving the solver diagnostics.

The adapter is opt-in: it is only reached when a caller explicitly selects
``SolverMethod.RODAS5P``. The default production lane (BDF) is byte-untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from rabbit.solver.rodas5p import Rodas5PConfig, RODAS5P, solve as _rodas5p_solve


@dataclass
class Rodas5PIvpResult:
    """A ``scipy.integrate.OdeResult``-shaped view of a ``Rodas5PResult``.

    Only the fields the Type-I driver + ``classify_solve_ivp_result`` actually
    read are guaranteed meaningful: ``t``, ``y`` (shape ``(n_states, n_steps)``,
    already transposed by ``rodas5p.solve``), ``success``, ``status``,
    ``message``, and ``t_events`` (list with one entry — the single event).
    ``y_events``/``nfev``/``njev`` are populated for parity of shape but are not
    load-bearing in the current driver.
    """
    t: np.ndarray
    y: np.ndarray
    success: bool
    status: int
    message: str
    t_events: list
    y_events: list
    n_steps: int
    n_rejected: int
    n_attempts: int
    failure_reason: Optional[str]
    nfev: int = 0
    njev: int = 0
    # scipy sets sol.sol (dense interpolant) only when dense_output=True; the
    # driver never requests it, so it is absent here (dense_output raises below).
    sol: Optional[object] = None


def solve_ivp_rodas5p(
    fun: Callable,
    t_span: Sequence[float],
    y0: np.ndarray,
    events: Optional[Callable] = None,
    *,
    dfdt_fn: Optional[Callable] = None,
    rtol: float,
    atol: float,
    max_step: float,
    first_step: Optional[float] = None,
    dense_output: bool = False,
    jac_reuse_max_steps: int = 1,
    base_config: Optional[Rodas5PConfig] = None,
) -> Rodas5PIvpResult:
    """Run ``rabbit.solver.rodas5p.solve`` under a scipy-``solve_ivp`` call shape.

    Parameters mirror the subset of ``solve_ivp`` kwargs the driver passes.
    ``events`` is a single terminal scalar callable ``events(t, y) -> float``
    carrying an optional ``.direction`` attribute (scipy convention); it is
    routed to rodas5p's single-event machinery with the matching
    ``event_direction``. ``dense_output=True`` is unsupported (fail-loud).

    SolverConfig → Rodas5PConfig mapping:
      rtol/atol      → rtol/atol (direct)
      max_step       → max_step_N AND h_max (both capped; strict is safe — the
                       driver's max_step is a hard ceiling on the step in N)
      first_step     → h_init
      jac_reuse_max_steps → threaded directly (default 1 = exact per-step
                       Jacobian); >1 amortizes the dense FD Jacobian (BD617).
    The remaining Rodas5PConfig knobs (tableau, step-control tolerances) come
    from ``base_config`` (default: a fresh Rodas5PConfig with the RODAS5P tableau).
    """
    if dense_output:
        raise NotImplementedError(
            "rodas5p_adapter does not support dense_output=True; the Type-I "
            "driver never requests a dense interpolant on this lane (BD615)."
        )

    base = base_config if base_config is not None else Rodas5PConfig(tableau=RODAS5P)
    cfg = Rodas5PConfig(
        rtol=float(rtol),
        atol=float(atol),
        max_steps=base.max_steps,
        h_init=float(first_step) if first_step is not None else None,
        h_min=base.h_min,
        h_max=float(max_step),
        f_safety=base.f_safety,
        f_min=base.f_min,
        f_max=base.f_max,
        beta=base.beta,
        max_step_N=float(max_step),
        tableau=base.tableau,
        jac_reuse_max_steps=int(jac_reuse_max_steps),
    )

    direction = int(getattr(events, "direction", 0) or 0) if events is not None else 0

    r = _rodas5p_solve(
        fun,
        (float(t_span[0]), float(t_span[1])),
        np.asarray(y0, dtype=float),
        config=cfg,
        events=events,
        jac_fn=None,
        event_direction=direction,
        dfdt_fn=dfdt_fn,
    )

    event_fired = (r.message == "Event")
    if event_fired:
        status = 1                       # scipy: 1 = a termination event occurred
        t_events = [np.array([r.t[-1]])]
        y_events = [np.array([r.y[:, -1]])]
    elif r.success:
        status = 0                       # scipy: 0 = reached end of t_span
        t_events = [np.array([])]
        y_events = [np.array([]).reshape(0, r.y.shape[0])]
    else:
        status = -1                      # scipy: -1 = integration step failed
        t_events = [np.array([])]
        y_events = [np.array([]).reshape(0, r.y.shape[0])]

    return Rodas5PIvpResult(
        t=r.t,
        y=r.y,
        success=bool(r.success or event_fired),
        status=status,
        message=r.message,
        t_events=t_events,
        y_events=y_events,
        n_steps=int(r.n_steps),
        n_rejected=int(r.n_rejected),
        n_attempts=int(r.n_attempts),
        failure_reason=r.failure_reason,
        nfev=0,
        njev=int(r.n_jac),
    )
