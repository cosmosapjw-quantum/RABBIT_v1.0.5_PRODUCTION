from types import SimpleNamespace

from rabbit.solver.ivp_outcome import classify_solve_ivp_result


def _dummy(status, success, n_steps, message="msg"):
    return SimpleNamespace(
        status=status,
        success=success,
        t=list(range(n_steps)),
        message=message,
    )


def test_target_reached():
    d = classify_solve_ivp_result(_dummy(1, True, 5), target_reached=True, phase="phase1")
    assert d["solver_outcome"] == "target_reached"


def test_interval_exhausted_without_event():
    d = classify_solve_ivp_result(_dummy(0, True, 5), target_reached=False, phase="phase1")
    assert d["solver_outcome"] == "interval_exhausted_without_event"


def test_startup_failure():
    d = classify_solve_ivp_result(_dummy(-1, False, 1), target_reached=False, phase="phase1")
    assert d["solver_outcome"] == "startup_failure"


def test_integration_failure():
    d = classify_solve_ivp_result(_dummy(-1, False, 6), target_reached=False, phase="phase2")
    assert d["solver_outcome"] == "integration_failure"
