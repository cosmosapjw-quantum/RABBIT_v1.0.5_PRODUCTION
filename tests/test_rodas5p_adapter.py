"""BD615 — rabbit.solver.rodas5p_adapter scipy-solve_ivp-shape contract.

The adapter lets the Type-I driver run the in-tree Rosenbrock-W solver through
the identical result contract it uses for scipy.solve_ivp. These tests lock the
status/t_events synthesis, the event-direction plumbing, the fail-loud
dense_output guard, the to_scipy_kwargs rejection of RODAS5P, and (slow) the
end-to-end driver parity vs the BDF baseline.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from rabbit.config.solver_config import SolverConfig, SolverMethod
from rabbit.solver.rodas5p import Rodas5PConfig
from rabbit.solver.rodas5p_adapter import solve_ivp_rodas5p, Rodas5PIvpResult
from rabbit.solver.ivp_outcome import classify_solve_ivp_result


def _decay(k=3.0):
    return lambda t, y: -k * y


def _adapter_h_min_worker(send_conn):
    try:
        result = solve_ivp_rodas5p(
            lambda _t, y: np.full_like(y, np.nan),
            [0.0, 1.0],
            np.array([1.0]),
            rtol=1.0e-8,
            atol=1.0e-10,
            max_step=1.0e-6,
            first_step=1.0e-6,
            base_config=Rodas5PConfig(max_steps=10, h_min=1.0e-6),
        )
        send_conn.send({
            "success": bool(result.success),
            "status": int(result.status),
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


def _adapter_h_min_outcome():
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_adapter_h_min_worker, args=(send_conn,))
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
        pytest.fail("Rodas5P adapter h_min failure exceeded the frozen 2-second wall")
    assert process.exitcode == 0
    assert recv_conn.poll(), "adapter child exited without returning an outcome"
    payload = recv_conn.recv()
    recv_conn.close()
    assert "exception" not in payload, payload.get("exception")
    return payload


def test_to_scipy_kwargs_rejects_rodas5p():
    cfg = SolverConfig(method=SolverMethod.RODAS5P, rtol=1e-8, atol=1e-10, max_step=0.1)
    assert cfg.is_scipy is False
    with pytest.raises(ValueError, match="non-scipy"):
        cfg.to_scipy_kwargs()
    # scipy methods still work
    assert SolverConfig(method=SolverMethod.BDF).is_scipy is True
    assert SolverConfig(method=SolverMethod.BDF).to_scipy_kwargs()["method"] == "BDF"


def test_adapter_event_fired_status_and_classification():
    def stop(t, y):
        return y[0] - 0.1
    stop.terminal = True
    stop.direction = -1
    sol = solve_ivp_rodas5p(_decay(3.0), [0.0, 5.0], np.array([1.0]), events=stop,
                            rtol=1e-9, atol=1e-12, max_step=0.5)
    assert isinstance(sol, Rodas5PIvpResult)
    assert sol.status == 1 and sol.success is True
    assert len(sol.t_events[0]) == 1
    assert sol.y.shape[0] == 1                       # (n_states, n_steps) scipy layout
    assert abs(sol.y[0, -1] - 0.1) < 1e-8            # crossing localized on target
    tgt = bool(getattr(sol, "t_events", None)) and len(sol.t_events[0]) > 0
    diag = classify_solve_ivp_result(sol, target_reached=tgt, phase="p")
    assert diag["solver_outcome"] == "target_reached"


def test_adapter_direction_filter_skips_wrong_crossing():
    # y0=cos(2πt): ev crosses DOWN at 1/6, UP at 5/6; require the UP crossing
    W = 2.0 * np.pi
    A = np.array([[0.0, 1.0], [-W * W, 0.0]])
    def stop(t, y):
        return y[0] - 0.5
    stop.terminal = True
    stop.direction = +1
    sol = solve_ivp_rodas5p(lambda t, y: A @ y, [0.0, 1.0], np.array([1.0, 0.0]),
                            events=stop, rtol=1e-10, atol=1e-12, max_step=0.02)
    assert sol.status == 1
    assert abs(sol.t[-1] - 5.0 / 6.0) < 1e-3          # skipped the down-crossing at 1/6


def test_adapter_interval_exhausted_without_event():
    def nofire(t, y):
        return y[0] + 100.0                            # never reaches zero
    nofire.terminal = True
    nofire.direction = -1
    sol = solve_ivp_rodas5p(_decay(3.0), [0.0, 1.0], np.array([1.0]), events=nofire,
                            rtol=1e-9, atol=1e-12, max_step=0.5)
    assert sol.status == 0 and sol.success is True
    assert len(sol.t_events[0]) == 0
    diag = classify_solve_ivp_result(sol, target_reached=False, phase="p")
    assert diag["solver_outcome"] == "interval_exhausted_without_event"


def test_adapter_dense_output_raises():
    with pytest.raises(NotImplementedError, match="dense_output"):
        solve_ivp_rodas5p(_decay(3.0), [0.0, 1.0], np.array([1.0]),
                          rtol=1e-9, atol=1e-12, max_step=0.5, dense_output=True)


def test_adapter_threads_jac_reuse_into_config():
    # BD617: jac_reuse_max_steps must reach the solver's Jacobian budget.
    # reuse>1 recomputes the Jacobian less often -> strictly fewer Jacobian builds
    # on a problem that takes several steps.
    k = 8.0
    f = _decay(k)
    # a stiff-ish decay that takes many steps at tight tolerance
    r1 = solve_ivp_rodas5p(f, [0.0, 3.0], np.array([1.0]), rtol=1e-10, atol=1e-13,
                           max_step=0.05, jac_reuse_max_steps=1)
    r5 = solve_ivp_rodas5p(f, [0.0, 3.0], np.array([1.0]), rtol=1e-10, atol=1e-13,
                           max_step=0.05, jac_reuse_max_steps=5)
    assert r5.njev < r1.njev                          # reuse cut the Jacobian builds
    assert abs(r1.y[0, -1] - r5.y[0, -1]) < 1e-6      # endpoint stays close on a mild problem


def test_adapter_event_time_matches_scipy():
    from scipy.integrate import solve_ivp
    def stop(t, y):
        return y[0] - 0.1
    stop.terminal = True
    stop.direction = -1
    a = solve_ivp_rodas5p(_decay(3.0), [0.0, 5.0], np.array([1.0]), events=stop,
                          rtol=1e-9, atol=1e-12, max_step=0.5)
    s = solve_ivp(_decay(3.0), [0.0, 5.0], [1.0], method="BDF", events=stop,
                  rtol=1e-9, atol=1e-12, max_step=0.5)
    assert abs(a.t[-1] - s.t_events[0][0]) < 1e-6


def test_adapter_failure_surface_preserves_exact_h_min_outcome_in_subprocess():
    assert _adapter_h_min_outcome() == {
        "success": False,
        "status": -1,
        "accepted": 0,
        "rejected": 1,
        "attempts": 1,
        "failure_reason": "h_min",
        "raw_final_state": [1.0],
    }


@pytest.mark.slow
@pytest.mark.production
def test_driver_rodas5p_endpoint_parity_vs_bdf():
    """End-to-end: the RODAS5P driver lane matches the BDF baseline endpoint."""
    from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI

    def _solve(solver):
        return run_full_coupled_typeI(FullCoupledConfig(
            Sigma_H_plus=0.1, N_q=8, N_mu=8, n_reactions=12,
            correction_level=0, tier=1, enable_teff=False, solver=solver))

    bdf = _solve(None)
    rodas = _solve(SolverConfig(method=SolverMethod.RODAS5P, rtol=1e-8, atol=1e-10, max_step=0.1))
    assert rodas.metadata["solver_method_requested"] == "RODAS5P"
    assert rodas.metadata["solver_method_effective"] == "RODAS5P"
    assert rodas.metadata["rodas5p_jac_sparsity_unused"] is True
    delta_yp = abs(rodas.observables.Yp - bdf.observables.Yp)
    relative_delta_dh = abs(rodas.observables.DH - bdf.observables.DH) / bdf.observables.DH
    delta_neff = abs(rodas.observables.N_eff - bdf.observables.N_eff)
    print(
        "Type-I BDF/Rodas endpoint gaps: "
        f"abs(delta_Yp)={delta_yp:.17g}, "
        f"relative_delta_DH={relative_delta_dh:.17g}, "
        f"abs(delta_N_eff)={delta_neff:.17g}"
    )
    assert delta_yp < 1e-5
    assert relative_delta_dh < 1e-3
    assert delta_neff <= 1e-4
