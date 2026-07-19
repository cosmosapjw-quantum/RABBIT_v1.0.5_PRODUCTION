"""BD615 — solver-level upgrades to rabbit.solver.rodas5p.

Covers (all standalone ODEs, no BBN driver): event direction filtering,
event-crossing refinement to machine precision, LU-once-per-step accuracy,
and Jacobian reuse (n_jac drops, endpoint stays within envelope).
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from rabbit.solver.rodas5p import (
    solve, Rodas5PConfig, RODAS5P, ROS34PW2, _event_fires, _refine_event, _step,
)


def _cfg(**kw):
    return Rodas5PConfig(tableau=RODAS5P, **kw)


_FIXED_STEP_COUNTS = (8, 16, 32, 64)
_NONAUTONOMOUS_EXACT = 0.5 * (np.sin(1.0) - np.cos(1.0) + np.exp(-1.0))
_AUTONOMOUS_ENDPOINT_BEFORE = 0.006737947000718982


def _nonautonomous_rhs(t, y):
    return -y + np.sin(t)


def _nonautonomous_jac(_t, _y):
    return np.array([[-1.0]])


def _nonautonomous_dfdt(t, _y):
    return np.array([np.cos(t)])


def _fixed_step_endpoint(
    rhs,
    jac,
    *,
    tableau,
    n_steps,
    y0,
    dfdt_fn=None,
):
    """Run exactly ``n_steps`` accepted steps through the public NumPy API."""
    h = 1.0 / float(n_steps)
    cfg = Rodas5PConfig(
        tableau=tableau,
        rtol=1.0e6,
        atol=1.0e6,
        max_steps=n_steps + 1,
        h_init=h,
        h_min=h,
        h_max=h,
        max_step_N=h,
    )
    result = solve(
        rhs,
        (0.0, 1.0),
        np.asarray(y0, dtype=float),
        config=cfg,
        jac_fn=jac,
        dfdt_fn=dfdt_fn,
    )
    assert result.success, result.message
    assert result.n_steps == n_steps
    assert result.n_rejected == 0
    return float(result.y[0, -1])


def _global_order(errors):
    h = 1.0 / np.asarray(_FIXED_STEP_COUNTS, dtype=float)
    return float(np.polyfit(np.log(h), np.log(np.asarray(errors, dtype=float)), 1)[0])


def _nan_rejection_worker(send_conn, config_kwargs, t_span):
    """Child-only worker: a broken controller may spin, so never run it inline."""
    try:
        result = solve(
            lambda _t, y: np.full_like(y, np.nan),
            t_span,
            np.array([1.0]),
            config=_cfg(**config_kwargs),
            jac_fn=lambda _t, _y: np.zeros((1, 1)),
        )
        send_conn.send({
            "success": bool(result.success),
            "accepted": int(result.n_steps),
            "rejected": int(result.n_rejected),
            "attempts": int(result.n_attempts),
            "failure_reason": result.failure_reason,
            "raw_final_state": np.asarray(result.y[:, -1], dtype=float).tolist(),
        })
    except BaseException as exc:  # pragma: no cover - surfaced in the parent assertion
        send_conn.send({"exception": f"{type(exc).__name__}: {exc}"})
    finally:
        send_conn.close()


def _run_nan_rejection_case(config_kwargs, t_span=(0.0, 1.0)):
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_nan_rejection_worker,
        args=(send_conn, config_kwargs, t_span),
    )
    process.start()
    send_conn.close()
    process.join(timeout=2.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
        if process.is_alive():
            process.kill()
            process.join(timeout=0.5)
        recv_conn.close()
        pytest.fail("Rodas5P rejection controller exceeded the frozen 2-second wall")
    assert process.exitcode == 0
    assert recv_conn.poll(), "rejection child exited without returning an outcome"
    payload = recv_conn.recv()
    recv_conn.close()
    assert "exception" not in payload, payload.get("exception")
    return payload


# ── Harmonic oscillator with a real Jacobian (Rosenbrock-accurate) ──
_W = 2.0 * np.pi
_A = np.array([[0.0, 1.0], [-_W * _W, 0.0]])


def _osc(t, y):
    return _A @ y


def _osc_jac(t, y):
    return _A


def test_nonautonomous_order_numpy_analytic_and_fd():
    analytic_endpoints = []
    fallback_endpoints = []
    for n_steps in _FIXED_STEP_COUNTS:
        analytic_endpoints.append(_fixed_step_endpoint(
            _nonautonomous_rhs,
            _nonautonomous_jac,
            tableau=RODAS5P,
            n_steps=n_steps,
            y0=[0.0],
            dfdt_fn=_nonautonomous_dfdt,
        ))
        fallback_endpoints.append(_fixed_step_endpoint(
            _nonautonomous_rhs,
            _nonautonomous_jac,
            tableau=RODAS5P,
            n_steps=n_steps,
            y0=[0.0],
        ))

    analytic_errors = np.abs(np.asarray(analytic_endpoints) - _NONAUTONOMOUS_EXACT)
    fallback_errors = np.abs(np.asarray(fallback_endpoints) - _NONAUTONOMOUS_EXACT)
    analytic_order = _global_order(analytic_errors)
    fallback_order = _global_order(fallback_errors)

    assert analytic_order >= 4.5, (
        f"analytic dfdt global order={analytic_order:.12g}; errors={analytic_errors.tolist()}"
    )
    assert fallback_order >= 4.5, (
        f"finite-difference dfdt global order={fallback_order:.12g}; "
        f"errors={fallback_errors.tolist()}"
    )
    assert abs(analytic_endpoints[2] - fallback_endpoints[2]) <= 1.0e-10


def test_ros34pw2_coefficient_sum_and_third_order():
    assert abs(float(np.sum(ROS34PW2.b)) - 1.0) <= 1.0e-15
    assert float(ROS34PW2.b[2]) == 1.5452602553351023

    endpoints = [
        _fixed_step_endpoint(
            lambda _t, y: -y,
            lambda _t, _y: np.array([[-1.0]]),
            tableau=ROS34PW2,
            n_steps=n_steps,
            y0=[1.0],
        )
        for n_steps in _FIXED_STEP_COUNTS
    ]
    errors = np.abs(np.asarray(endpoints) - np.exp(-1.0))
    order = _global_order(errors)
    assert order >= 2.5, f"ROS34PW2 global order={order:.12g}; errors={errors.tolist()}"


def test_autonomous_endpoint_lock_numpy():
    result = solve(
        lambda _t, y: -y,
        (0.0, 5.0),
        np.array([1.0]),
        config=_cfg(rtol=1.0e-8, atol=1.0e-10, max_step_N=0.5),
        jac_fn=lambda _t, _y: np.array([[-1.0]]),
    )
    assert result.success, result.message
    assert abs(float(result.y[0, -1]) - _AUTONOMOUS_ENDPOINT_BEFORE) <= 1.0e-12


def test_persistent_rejection_respects_attempt_budget_in_subprocess():
    payload = _run_nan_rejection_case({
        "max_steps": 3,
        "h_init": 0.1,
        "h_min": 1.0e-6,
    })
    assert payload == {
        "success": False,
        "accepted": 0,
        "rejected": 3,
        "attempts": 3,
        "failure_reason": "max_steps",
        "raw_final_state": [1.0],
    }


def test_h_min_rejection_returns_exact_outcome_in_subprocess():
    payload = _run_nan_rejection_case({
        "max_steps": 10,
        "h_init": 1.0e-6,
        "h_min": 1.0e-6,
    })
    assert payload == {
        "success": False,
        "accepted": 0,
        "rejected": 1,
        "attempts": 1,
        "failure_reason": "h_min",
        "raw_final_state": [1.0],
    }


def test_rejected_truncated_final_step_fails_at_h_min_immediately():
    payload = _run_nan_rejection_case(
        {
            "max_steps": 10,
            "h_init": 1.0e-3,
            "h_min": 1.0e-6,
        },
        t_span=(0.0, 5.0e-7),
    )
    assert payload == {
        "success": False,
        "accepted": 0,
        "rejected": 1,
        "attempts": 1,
        "failure_reason": "h_min",
        "raw_final_state": [1.0],
    }


def test_rejected_truncated_step_rescales_the_attempted_step():
    result = solve(
        lambda _t, y: -100.0 * y,
        (0.0, 0.01),
        np.array([1.0]),
        config=_cfg(
            rtol=1.0e-8,
            atol=1.0e-10,
            max_steps=1,
            h_init=1.0,
            h_min=1.0e-14,
            h_max=1.0,
            max_step_N=1.0,
        ),
        jac_fn=lambda _t, _y: np.array([[-100.0]]),
    )
    assert result.success is False
    assert result.n_steps == 0
    assert result.n_rejected == result.n_attempts == 1
    assert result.failure_reason == "max_steps"
    assert result.h_final == pytest.approx(0.002, rel=0.0, abs=1.0e-15)


def test_dfdt_is_evaluated_once_at_each_attempt_start():
    calls = []

    def dfdt_fn(t, y):
        calls.append((float(t), np.asarray(y, dtype=float).copy()))
        return np.array([np.cos(t)])

    n_steps = 4
    h = 1.0 / n_steps
    result = solve(
        _nonautonomous_rhs,
        (0.0, 1.0),
        np.array([0.0]),
        config=_cfg(
            rtol=1.0e6,
            atol=1.0e6,
            max_steps=n_steps + 1,
            h_init=h,
            h_min=h,
            h_max=h,
            max_step_N=h,
        ),
        jac_fn=_nonautonomous_jac,
        dfdt_fn=dfdt_fn,
    )
    assert result.success, result.message
    assert len(calls) == result.n_steps + result.n_rejected == n_steps
    np.testing.assert_allclose(
        np.asarray([t for t, _ in calls]),
        result.t[:-1],
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray([y for _, y in calls]),
        result.y[:, :-1].T,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_fallback_dfdt_central_pair_is_evaluated_once_at_attempt_start():
    call_times = []

    def rhs(t, y):
        call_times.append(float(t))
        return -y + np.sin(t)

    h = 0.125
    result = solve(
        rhs,
        (0.0, h),
        np.array([0.0]),
        config=_cfg(
            rtol=1.0e6,
            atol=1.0e6,
            max_steps=2,
            h_init=h,
            h_min=h,
            h_max=h,
            max_step_N=h,
        ),
        jac_fn=_nonautonomous_jac,
    )
    assert result.success, result.message
    delta_t = np.cbrt(np.finfo(float).eps) * max(1.0, h)
    backward_calls = [t for t in call_times if abs(t + delta_t) <= 1.0e-15]
    forward_calls = [t for t in call_times if abs(t - delta_t) <= 1.0e-15]
    assert len(backward_calls) == 1
    assert len(forward_calls) == 1


def test_nonautonomous_event_refinement_matches_exact_crossing():
    def event(_t, y):
        return y[0] - 0.3

    result = solve(
        lambda t, _y: np.array([1.0 + t]),
        (0.0, 1.0),
        np.array([0.0]),
        config=_cfg(rtol=1.0e-10, atol=1.0e-12, max_step_N=0.1),
        jac_fn=lambda _t, _y: np.zeros((1, 1)),
        dfdt_fn=lambda _t, _y: np.array([1.0]),
        events=event,
        event_direction=1,
    )
    exact_crossing = np.sqrt(1.6) - 1.0
    assert result.success and result.message == "Event"
    assert abs(float(result.t[-1]) - exact_crossing) <= 1.0e-8
    assert abs(float(result.y[0, -1]) - 0.3) <= 1.0e-8


def test_event_fires_direction_unit():
    # scipy convention: dir<0 fires on +→- crossing, dir>0 on -→+, dir0 either
    assert _event_fires(0.3, -0.2, -1) is True
    assert _event_fires(-0.3, 0.2, -1) is False
    assert _event_fires(-0.3, 0.2, +1) is True
    assert _event_fires(0.3, -0.2, +1) is False
    assert _event_fires(0.3, -0.2, 0) is True
    assert _event_fires(-0.3, 0.2, 0) is True
    assert _event_fires(0.3, 0.2, 0) is False


def test_event_direction_selects_crossing():
    # y0 = cos(2πt): ev = y0 - 0.5 crosses DOWN at t=1/6, UP at t=5/6
    ev = lambda t, y: y[0] - 0.5
    cfg = _cfg(rtol=1e-10, atol=1e-12, max_step_N=0.02)
    y0 = np.array([1.0, 0.0])
    r_any = solve(_osc, (0.0, 1.0), y0, config=cfg, jac_fn=_osc_jac, events=ev, event_direction=0)
    r_dn = solve(_osc, (0.0, 1.0), y0, config=cfg, jac_fn=_osc_jac, events=ev, event_direction=-1)
    r_up = solve(_osc, (0.0, 1.0), y0, config=cfg, jac_fn=_osc_jac, events=ev, event_direction=+1)
    assert r_any.message == "Event" and abs(r_any.t[-1] - 1.0 / 6.0) < 1e-3
    assert r_dn.message == "Event" and abs(r_dn.t[-1] - 1.0 / 6.0) < 1e-3
    assert r_up.message == "Event" and abs(r_up.t[-1] - 5.0 / 6.0) < 1e-3


def test_event_crossing_refined_to_machine_precision():
    # exponential decay: analytic crossing of y=0.1 is at t = ln(10)/k
    k = 3.0
    f = lambda t, y: -k * y
    jac = lambda t, y: np.array([[-k]])
    ev = lambda t, y: y[0] - 0.1
    r = solve(f, (0.0, 5.0), np.array([1.0]), config=_cfg(rtol=1e-9, atol=1e-12),
              jac_fn=jac, events=ev, event_direction=-1)
    t_exact = np.log(10.0) / k
    assert r.message == "Event"
    # linear interpolation would leave ~1e-4; refinement must reach <1e-8
    assert abs(r.t[-1] - t_exact) < 1e-8
    assert abs(r.y[0, -1] - 0.1) < 1e-8


def test_lu_once_accuracy_matches_analytic():
    # LU-once-per-step must not degrade accuracy vs the analytic solution
    k = 5.0
    f = lambda t, y: -k * y
    jac = lambda t, y: np.array([[-k]])
    r = solve(f, (0.0, 1.0), np.array([1.0]), config=_cfg(rtol=1e-10, atol=1e-13), jac_fn=jac)
    exact = np.exp(-k)
    assert abs(r.y[0, -1] - exact) / exact < 1e-7


def test_step_returns_finite_or_flags_failure():
    # a singular W (J with a huge eigenvalue making W ill-conditioned) must not
    # produce silent NaNs — _step returns (y, inf, False) on non-finite result
    y = np.array([1.0, 1.0])
    Jbad = np.array([[np.inf, 0.0], [0.0, -1.0]])
    f = lambda t, yy: -yy
    yn, err, ok = _step(f, 0.0, y, 0.1, Jbad, RODAS5P, _cfg())
    assert (not ok) and not np.isfinite(err) or np.all(np.isfinite(yn))


def test_jacobian_reuse_drops_njac_within_envelope():
    y0 = np.array([1.0, 0.0])
    base = _cfg(rtol=1e-9, atol=1e-11, max_step_N=0.02)
    reuse = _cfg(rtol=1e-9, atol=1e-11, max_step_N=0.02, jac_reuse_max_steps=5)
    r1 = solve(_osc, (0.0, 0.15), y0, config=base, jac_fn=_osc_jac)
    r5 = solve(_osc, (0.0, 0.15), y0, config=reuse, jac_fn=_osc_jac)
    assert r5.n_jac < r1.n_jac                          # reuse cuts Jacobian builds
    # linear system → constant J → reuse is exact; endpoint must agree tightly
    assert np.max(np.abs(r5.y[:, -1] - r1.y[:, -1])) < 1e-8


def test_default_reuse_is_every_step():
    # jac_reuse_max_steps default (1) recomputes the Jacobian every accepted step
    k = 5.0
    f = lambda t, y: -k * y
    jac = lambda t, y: np.array([[-k]])
    r = solve(f, (0.0, 1.0), np.array([1.0]), config=_cfg(rtol=1e-8, atol=1e-10), jac_fn=jac)
    # with reuse=1 the Jacobian count equals accepted-step evaluations (>= n_steps)
    assert r.n_jac >= r.n_steps
