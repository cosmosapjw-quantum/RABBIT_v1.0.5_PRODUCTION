from __future__ import annotations


def classify_solve_ivp_result(sol, *, target_reached: bool, phase: str) -> dict:
    status_raw = int(getattr(sol, "status", 99))
    success = bool(getattr(sol, "success", False))
    t_values = getattr(sol, "t", None)
    n_steps = 0 if t_values is None else len(t_values)
    message = str(getattr(sol, "message", ""))

    if target_reached:
        outcome = "target_reached"
        reason = "event_triggered"
    elif not success and n_steps < 2:
        outcome = "startup_failure"
        reason = "solver_failed_before_progress"
    elif not success:
        outcome = "integration_failure"
        reason = "solver_reported_failure"
    elif success and status_raw == 0:
        outcome = "interval_exhausted_without_event"
        reason = "finished_tspan_but_target_not_hit"
    else:
        outcome = "terminated_other"
        reason = "unclassified"

    return {
        "phase": str(phase),
        "solver_outcome": outcome,
        "solver_terminal_reason": reason,
        "solver_status_raw": status_raw,
        "solver_success": success,
        "solver_n_steps": int(n_steps),
        "solver_message": message,
        "target_reached": bool(target_reached),
    }
