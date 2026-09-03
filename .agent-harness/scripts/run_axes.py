#!/usr/bin/env python3
"""Run the independent axes of one frozen contract concurrently.

Why this exists: the audit drivers in ``scripts/audit`` are single-threaded
integrations that take hours each (the D-069 r4 trajectory run was projected at
13.1 h sequential), and
until now a multi-axis contract was executed one axis after another. The axes of
a contract are independent by construction, and the host has more physical cores
than a single pinned integration can use, so the wall clock of a contract should
be the wall clock of its slowest axis, not the sum of all of them.

This launcher knows nothing about physics. It runs ``argv``.

Contract this file is written to satisfy:

* Every axis in the spec appears in the status JSON with its own outcome. No
  summarising, no truncation, no dropping a failed axis -- "all axes gated, all
  reported" is the governance rule, and a launcher that loses an axis defeats it.
* The status JSON is rewritten atomically (temp file + ``os.replace``) after
  every axis state change, so an interrupted run still leaves a readable,
  non-torn record. The write pattern is the one already used by ``Reporter`` in
  ``scripts/audit/_trajectory_core.py``.
* Children inherit the single-thread pins. They are placed in the environment
  handed to ``execve``, not set by the child after ``import numpy``: OpenBLAS
  reads its thread count at library init, so a late assignment is a silently
  multi-threaded -- and therefore non-bitwise-reproducible -- run.
* Concurrency is real. ``subprocess.run`` or ``Popen.communicate()`` inside the
  scheduling loop blocks on one child until it exits and serialises the whole
  batch while still looking concurrent; that exact mistake was caught in review
  here before. The loop below is ``Popen`` plus ``poll()`` and never blocks on a
  single child.
* A wall-budget breach is mechanical. It is recorded as ``wall_budget_exceeded``
  and nothing more -- it is not a verdict about the axis, and siblings keep
  running.
* Declared outputs are verified after the axis terminates, not only before it
  starts. The pre-flight check in ``validate_paths`` proves the parent directory
  is usable; it cannot prove the child ever wrote anything. An axis that exits 0
  having produced no output file is exactly the hole "all axes gated, all
  reported" exists to close, so every path in an axis's ``outputs`` must exist
  and be non-empty once the axis ends.

Outcome vocabulary. Terminal: ``completed`` (the child exited on its own; its
``exit_code`` may still be non-zero), ``wall_budget_exceeded``, ``launch_failed``,
``terminated`` (killed by launcher shutdown), ``not_started`` (never launched
because the launcher was already shutting down). Transitional, and only ever
observed in a status file read while the run is live: ``pending``, ``running``.

Success is narrower than ``completed``. An axis counts as clean only when its
outcome is ``completed``, its ``exit_code`` is 0, **and** ``outputs_present`` is
true. ``outputs_present`` is a separate per-axis flag rather than a new outcome
so that the vocabulary above keeps its exact meaning: ``completed`` still says
only "the child exited on its own", and the missing-output failure is reported
next to it in ``outputs_missing`` instead of being folded into it.

Usage::

    venv/bin/python .agent-harness/scripts/run_axes.py \
        --spec .agent-harness/axes/<contract>.json \
        --status .agent-harness/axes/<contract>_status.json [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Callable

from _harness import root

SPEC_VERSION = 1

# Polling granularity of the scheduler loop. Small enough that a wall budget is
# enforced promptly, large enough that the launcher itself costs no measurable
# CPU next to multi-hour integrations.
POLL_INTERVAL_SECONDS = 0.2

# Time a child gets between SIGTERM and SIGKILL, so a driver can flush its own
# report before it is destroyed.
DEFAULT_TERM_GRACE_SECONDS = 10.0

# Bitwise reproducibility depends on every one of these. Assigned, never
# defaulted: an inherited environment that already set one to something else
# must not survive into the child.
THREAD_PINS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

SPEC_KEYS = {
    "spec_version",
    "label",
    "max_concurrent",
    "nice",
    "axes",
    "cwd",
    "term_grace_seconds",
}
AXIS_KEYS = {"id", "argv", "wall_budget_seconds", "log", "outputs"}

# Best-effort recognition of an output path inside an otherwise opaque argv, so
# that a missing results directory is caught at --dry-run instead of hours in.
OUTPUT_FLAGS = ("--out", "--output", "-o")
OUTPUT_ASSIGNMENT_PREFIXES = ("--out=", "--output=")

# Signals received by the launcher itself. Appended by the handler, read by the
# scheduler loop: the handler does nothing that is unsafe to do asynchronously.
_SHUTDOWN_SIGNALS: list[int] = []


class SpecError(Exception):
    """The spec is not usable. Always fatal -- a launcher that guesses is worse
    than a launcher that stops."""


def _now_iso() -> str:
    # Milliseconds rather than the seconds used elsewhere in the project: the
    # per-axis intervals in this file are the evidence that axes overlapped.
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class AxisPlan:
    axis_id: str
    argv: tuple[str, ...]
    wall_budget_seconds: float
    log_declared: str
    log_path: Path
    # Every path whose parent directory is checked before launch: the spec's
    # `outputs` plus the ones sniffed out of argv.
    output_paths: tuple[Path, ...]
    # Only the spec's `outputs`, and the only paths whose existence is enforced
    # after the axis ends. The argv sniff is a best-effort heuristic (`-o` need
    # not mean "output file"), so it stays advisory: it may not turn a correct
    # run into a reported failure.
    declared_outputs: tuple[Path, ...]


@dataclass(frozen=True)
class Spec:
    path: Path
    sha256: str
    label: str
    max_concurrent: int
    nice: int
    cwd: Path
    term_grace_seconds: float
    axes: tuple[AxisPlan, ...]


@dataclass
class AxisState:
    plan: AxisPlan
    outcome: str = "pending"
    pid: int | None = None
    exit_code: int | None = None
    started_at: str | None = None
    completed_at: str | None = None
    wall_seconds: float | None = None
    error: str | None = None
    started_monotonic: float | None = None
    term_deadline: float | None = None
    killed: bool = False
    # None until the axis terminates and the check runs; an axis that never ran
    # is never credited with outputs it cannot have written.
    outputs_present: bool | None = None
    outputs_missing: tuple[str, ...] = ()
    proc: subprocess.Popen[bytes] | None = field(default=None, repr=False)
    log_handle: IO[str] | None = field(default=None, repr=False)


@dataclass
class StatusWriter:
    """Atomic single-destination status writer.

    Same pattern as ``Reporter.write`` in ``scripts/audit/_trajectory_core.py``:
    serialise fully, write to a pid-suffixed temp file in the destination
    directory, then ``os.replace``. A reader therefore sees either the previous
    document or the new one, never a partial one.
    """

    path: str

    def write(self, report: dict[str, Any]) -> None:
        text = json.dumps(report, sort_keys=True, indent=1) + "\n"
        tmp = f"{self.path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, self.path)


# --- spec loading and validation ---------------------------------------------


def _require_positive_number(value: Any, label: str, errors: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label} must be a number, got {value!r}.")
        return 0.0
    number = float(value)
    if not number > 0.0:
        errors.append(f"{label} must be > 0, got {number!r}.")
        return 0.0
    return number


def _require_positive_int(value: Any, label: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} must be an integer, got {value!r}.")
        return 0
    if value < 1:
        errors.append(f"{label} must be >= 1, got {value!r}.")
        return 0
    return value


def _declared_output_paths(argv: tuple[str, ...]) -> tuple[str, ...]:
    found: list[str] = []
    for index, token in enumerate(argv):
        if token in OUTPUT_FLAGS:
            if index + 1 < len(argv):
                found.append(argv[index + 1])
            continue
        for prefix in OUTPUT_ASSIGNMENT_PREFIXES:
            if token.startswith(prefix):
                found.append(token[len(prefix) :])
                break
    return tuple(found)


def _resolve(base: Path, candidate: str) -> Path:
    path = Path(candidate)
    return path if path.is_absolute() else base / path


def load_spec(spec_path: Path, repo_root: Path) -> Spec:
    """Parse and structurally validate the spec. Every problem is collected and
    reported together: a spec with three mistakes should need one round trip."""

    try:
        raw_bytes = spec_path.read_bytes()
    except OSError as exc:
        raise SpecError(f"Spec is unreadable: {spec_path} ({exc})") from exc
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SpecError(f"Spec is not valid JSON: {spec_path} ({exc})") from exc
    if not isinstance(raw, dict):
        raise SpecError(f"Spec must be a JSON object: {spec_path}")

    errors: list[str] = []
    unknown = sorted(set(raw) - SPEC_KEYS)
    if unknown:
        # Fail closed on typos: a misspelled "max_concurrent" that is silently
        # ignored would run the whole contract at the wrong concurrency.
        errors.append(f"Unknown spec key(s): {', '.join(unknown)}.")

    if raw.get("spec_version") != SPEC_VERSION:
        errors.append(
            f"spec_version must be {SPEC_VERSION}, got {raw.get('spec_version')!r}."
        )

    label = raw.get("label")
    if not isinstance(label, str) or not label.strip():
        errors.append(f"label must be a non-empty string, got {label!r}.")
        label = ""

    max_concurrent = _require_positive_int(
        raw.get("max_concurrent"), "max_concurrent", errors
    )

    nice_value = raw.get("nice", 0)
    if isinstance(nice_value, bool) or not isinstance(nice_value, int):
        errors.append(f"nice must be an integer, got {nice_value!r}.")
        nice_value = 0
    elif not -20 <= nice_value <= 19:
        errors.append(f"nice must be within [-20, 19], got {nice_value!r}.")
        nice_value = 0

    cwd_value = raw.get("cwd")
    if cwd_value is None:
        cwd = repo_root
    elif isinstance(cwd_value, str) and cwd_value.strip():
        cwd = _resolve(repo_root, cwd_value)
    else:
        errors.append(f"cwd must be a non-empty string when present, got {cwd_value!r}.")
        cwd = repo_root

    grace_value = raw.get("term_grace_seconds", DEFAULT_TERM_GRACE_SECONDS)
    grace = _require_positive_number(grace_value, "term_grace_seconds", errors)
    if grace <= 0.0:
        grace = DEFAULT_TERM_GRACE_SECONDS

    raw_axes = raw.get("axes")
    axes: list[AxisPlan] = []
    if not isinstance(raw_axes, list) or not raw_axes:
        errors.append("axes must be a non-empty list.")
        raw_axes = []

    seen_ids: set[str] = set()
    seen_logs: dict[Path, str] = {}
    for position, raw_axis in enumerate(raw_axes):
        where = f"axes[{position}]"
        if not isinstance(raw_axis, dict):
            errors.append(f"{where} must be an object, got {raw_axis!r}.")
            continue
        unknown_axis = sorted(set(raw_axis) - AXIS_KEYS)
        if unknown_axis:
            errors.append(f"{where} has unknown key(s): {', '.join(unknown_axis)}.")

        axis_id = raw_axis.get("id")
        if not isinstance(axis_id, str) or not axis_id.strip():
            errors.append(f"{where}.id must be a non-empty string, got {axis_id!r}.")
            axis_id = ""
        elif axis_id in seen_ids:
            errors.append(f"{where}.id is duplicated: {axis_id!r}.")
        else:
            seen_ids.add(axis_id)
        where = f"axes[{position}] ({axis_id or 'unnamed'})"

        raw_argv = raw_axis.get("argv")
        if not isinstance(raw_argv, list) or not raw_argv:
            errors.append(f"{where}.argv must be a non-empty list.")
            argv: tuple[str, ...] = ()
        elif not all(isinstance(token, str) and token != "" for token in raw_argv):
            errors.append(f"{where}.argv must contain only non-empty strings.")
            argv = ()
        else:
            argv = tuple(str(token) for token in raw_argv)

        budget = _require_positive_number(
            raw_axis.get("wall_budget_seconds"), f"{where}.wall_budget_seconds", errors
        )

        log_declared = raw_axis.get("log")
        if not isinstance(log_declared, str) or not log_declared.strip():
            errors.append(f"{where}.log must be a non-empty string, got {log_declared!r}.")
            log_declared = ""
            log_path = cwd
        else:
            log_path = _resolve(cwd, log_declared)
            previous = seen_logs.get(log_path)
            if previous is not None:
                # Two axes writing one log interleaves their output and destroys
                # both records.
                errors.append(
                    f"{where}.log collides with axis {previous!r}: {log_path}."
                )
            else:
                seen_logs[log_path] = axis_id

        declared_outputs = raw_axis.get("outputs", [])
        if not isinstance(declared_outputs, list) or not all(
            isinstance(item, str) and item != "" for item in declared_outputs
        ):
            errors.append(f"{where}.outputs must be a list of non-empty strings.")
            declared_outputs = []
        declared_output_paths = tuple(
            _resolve(cwd, candidate) for candidate in declared_outputs
        )
        output_paths = declared_output_paths + tuple(
            _resolve(cwd, candidate) for candidate in _declared_output_paths(argv)
        )

        if not axis_id or not argv or budget <= 0.0 or not log_declared:
            continue
        axes.append(
            AxisPlan(
                axis_id=axis_id,
                argv=argv,
                wall_budget_seconds=budget,
                log_declared=log_declared,
                log_path=log_path,
                output_paths=output_paths,
                declared_outputs=declared_output_paths,
            )
        )

    if errors:
        raise SpecError("\n".join(f"  - {error}" for error in errors))

    return Spec(
        path=spec_path.resolve(),
        sha256="sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
        label=str(label),
        max_concurrent=max_concurrent,
        nice=nice_value,
        cwd=cwd,
        term_grace_seconds=grace,
        axes=tuple(axes),
    )


def _directory_error(directory: Path, label: str, create: bool) -> str | None:
    if create:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"{label} directory cannot be created: {directory} ({exc})"
    if not directory.is_dir():
        return f"{label} directory does not exist: {directory}"
    if not os.access(directory, os.W_OK | os.X_OK):
        return f"{label} directory is not writable: {directory}"
    return None


def validate_paths(spec: Spec, status_path: Path, create: bool) -> None:
    """Filesystem preconditions. ``create`` is false for --dry-run, which must
    predict the real run without leaving anything behind."""

    errors: list[str] = []
    if not spec.cwd.is_dir():
        errors.append(f"cwd does not exist: {spec.cwd}")
    # Log and status files belong to the launcher, so a real run creates their
    # directories; output files belong to the child, so their directories must
    # already exist and are never created here.
    checks: list[tuple[Path, str, bool]] = [(status_path.parent, "status", create)]
    for plan in spec.axes:
        checks.append((plan.log_path.parent, f"axis {plan.axis_id} log", create))
        checks.extend(
            (output.parent, f"axis {plan.axis_id} output", False)
            for output in plan.output_paths
        )
    for directory, label, create_this in checks:
        error = _directory_error(directory, label, create_this)
        if error is not None:
            errors.append(error)
    if errors:
        raise SpecError("\n".join(f"  - {error}" for error in errors))


# --- execution ---------------------------------------------------------------


def child_environment() -> dict[str, str]:
    """The environment handed to execve, so the pins are already in place when
    the child's BLAS library initialises."""

    env = os.environ.copy()
    for pin in THREAD_PINS:
        env[pin] = "1"
    return env


def _launch(state: AxisState, spec: Spec, env: dict[str, str]) -> None:
    plan = state.plan
    try:
        handle = open(plan.log_path, "w", encoding="utf-8")
    except OSError as exc:
        state.outcome = "launch_failed"
        state.error = f"log file could not be opened: {exc}"
        state.started_at = _now_iso()
        state.completed_at = state.started_at
        state.wall_seconds = 0.0
        return

    def _apply_nice() -> None:  # pragma: no cover -- runs in the forked child
        os.nice(spec.nice)

    try:
        proc = subprocess.Popen(
            list(plan.argv),
            cwd=str(spec.cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            # Its own session, so a budget breach or a launcher shutdown can take
            # down the child's whole process group and never leave orphans.
            start_new_session=True,
            preexec_fn=_apply_nice if spec.nice != 0 else None,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        handle.close()
        state.outcome = "launch_failed"
        state.error = f"{exc.__class__.__name__}: {exc}"
        state.started_at = _now_iso()
        state.completed_at = state.started_at
        state.wall_seconds = 0.0
        return

    state.proc = proc
    state.log_handle = handle
    state.pid = proc.pid
    state.outcome = "running"
    state.started_at = _now_iso()
    state.started_monotonic = time.monotonic()


def _signal_axis(state: AxisState, sig: int) -> None:
    proc = state.proc
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        # Group signalling refused; the direct child is still ours.
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            return


def _check_outputs(state: AxisState) -> None:
    """Verify every declared output now exists and is non-empty.

    Run once the axis has terminated. An axis with no declared outputs passes
    vacuously -- the spec asked for nothing, so nothing is owed. A path that
    cannot be stat'ed, is not a regular file, or is zero bytes is a missing
    output: a driver that created its results file and then died before writing
    to it has produced no result.
    """

    missing: list[str] = []
    for path in state.plan.declared_outputs:
        try:
            stat_result = path.stat()
        except OSError as exc:
            missing.append(f"{path} ({exc.__class__.__name__}: {exc.strerror or exc})")
            continue
        if not os.path.isfile(path):
            missing.append(f"{path} (not a regular file)")
        elif stat_result.st_size == 0:
            missing.append(f"{path} (empty)")
    state.outputs_missing = tuple(missing)
    state.outputs_present = not missing


def _reap(state: AxisState, exit_code: int) -> None:
    state.exit_code = exit_code
    state.completed_at = _now_iso()
    if state.started_monotonic is not None:
        state.wall_seconds = round(time.monotonic() - state.started_monotonic, 3)
    if state.outcome == "running":
        state.outcome = "completed"
    if state.log_handle is not None:
        state.log_handle.close()
        state.log_handle = None
    # After the log is flushed and the child is gone: whatever it wrote is on
    # disk now, and nothing more will appear.
    _check_outputs(state)


def axis_record(state: AxisState) -> dict[str, Any]:
    plan = state.plan
    return {
        "id": plan.axis_id,
        "argv": list(plan.argv),
        "log": str(plan.log_path),
        "wall_budget_seconds": plan.wall_budget_seconds,
        "pid": state.pid,
        "started_at": state.started_at,
        "completed_at": state.completed_at,
        "wall_seconds": state.wall_seconds,
        "exit_code": state.exit_code,
        "outcome": state.outcome,
        # true/false once the axis has terminated and the declared outputs were
        # checked; null while it is still pending or running, or if it never ran.
        "outputs_present": state.outputs_present,
        "outputs_declared": [str(path) for path in plan.declared_outputs],
        "outputs_missing": list(state.outputs_missing),
        "error": state.error,
    }


def status_document(
    spec: Spec,
    states: list[AxisState],
    started_at: str,
    started_monotonic: float,
    finished: bool,
) -> dict[str, Any]:
    records = [axis_record(state) for state in states]
    counts: dict[str, int] = {}
    for record in records:
        counts[record["outcome"]] = counts.get(record["outcome"], 0) + 1
    # `completed` + exit 0 is not sufficient: an axis that produced none of its
    # declared outputs did not do the work the spec gated on. `is True` rather
    # than truthiness, so an unchecked (null) axis can never pass.
    ok = all(
        record["outcome"] == "completed"
        and record["exit_code"] == 0
        and record["outputs_present"] is True
        for record in records
    )
    missing_outputs = [
        record["id"] for record in records if record["outputs_present"] is False
    ]
    return {
        "schema": "run_axes/status/1",
        "label": spec.label,
        "spec_path": str(spec.path),
        "spec_sha256": spec.sha256,
        "launcher_pid": os.getpid(),
        "cwd": str(spec.cwd),
        "max_concurrent": spec.max_concurrent,
        "nice": spec.nice,
        "thread_pins": {pin: "1" for pin in THREAD_PINS},
        "started_at": started_at,
        "updated_at": _now_iso(),
        "wall_seconds": round(time.monotonic() - started_monotonic, 3),
        "finished": finished,
        "interrupted_by_signal": _SHUTDOWN_SIGNALS[0] if _SHUTDOWN_SIGNALS else None,
        "axis_count": len(records),
        "outcome_counts": counts,
        "missing_outputs_axis_ids": missing_outputs,
        "ok": finished and ok,
        # Never summarised, never truncated: one record per spec axis, in spec
        # order, for the whole life of the run.
        "axes": records,
    }


def _record_signal(signum: int, _frame: Any) -> None:
    _SHUTDOWN_SIGNALS.append(signum)


def run(spec: Spec, status: StatusWriter) -> int:
    states = [AxisState(plan=plan) for plan in spec.axes]
    queue = list(states)
    running: list[AxisState] = []
    env = child_environment()
    started_at = _now_iso()
    started_monotonic = time.monotonic()

    def emit(finished: bool = False) -> None:
        status.write(
            status_document(spec, states, started_at, started_monotonic, finished)
        )

    def log(message: str) -> None:
        print(f"[{time.monotonic() - started_monotonic:9.1f}s] {message}", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _record_signal)

    # Written before anything is launched: an instantly killed launcher still
    # leaves a status file naming every axis.
    emit()
    log(
        f"{spec.label}: {len(states)} axes, max_concurrent={spec.max_concurrent}, "
        f"nice={spec.nice}, cwd={spec.cwd}"
    )

    while not _SHUTDOWN_SIGNALS:
        while queue and len(running) < spec.max_concurrent and not _SHUTDOWN_SIGNALS:
            state = queue.pop(0)
            _launch(state, spec, env)
            if state.proc is None:
                log(f"{state.plan.axis_id}: launch failed -- {state.error}")
            else:
                running.append(state)
                log(
                    f"{state.plan.axis_id}: started pid={state.pid} "
                    f"budget={state.plan.wall_budget_seconds:g}s "
                    f"log={state.plan.log_path}"
                )
            emit()

        changed = False
        now = time.monotonic()
        for state in list(running):
            proc = state.proc
            if proc is None:
                running.remove(state)
                continue
            # poll(), never communicate()/run(): blocking on one child here would
            # serialise the batch and silently defeat the whole launcher.
            exit_code = proc.poll()
            if exit_code is None:
                started = state.started_monotonic
                if started is None:
                    continue
                elapsed = now - started
                if (
                    state.outcome == "running"
                    and elapsed >= state.plan.wall_budget_seconds
                ):
                    # Mechanical, not a verdict: the axis ran out of its budget.
                    state.outcome = "wall_budget_exceeded"
                    state.term_deadline = now + spec.term_grace_seconds
                    _signal_axis(state, signal.SIGTERM)
                    log(
                        f"{state.plan.axis_id}: wall budget "
                        f"{state.plan.wall_budget_seconds:g}s exceeded at "
                        f"{elapsed:.1f}s -- SIGTERM sent"
                    )
                    changed = True
                elif (
                    state.term_deadline is not None
                    and not state.killed
                    and now >= state.term_deadline
                ):
                    state.killed = True
                    _signal_axis(state, signal.SIGKILL)
                    log(f"{state.plan.axis_id}: grace period elapsed -- SIGKILL sent")
                    changed = True
                continue
            _reap(state, exit_code)
            running.remove(state)
            log(
                f"{state.plan.axis_id}: {state.outcome} exit={state.exit_code} "
                f"wall={state.wall_seconds}s"
            )
            if state.outputs_present is False:
                log(
                    f"{state.plan.axis_id}: declared output(s) missing after exit "
                    f"{state.exit_code}: {', '.join(state.outputs_missing)}"
                )
            changed = True

        if changed:
            emit()
        if not running and not queue:
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    if _SHUTDOWN_SIGNALS:
        _shutdown(spec, queue, running, log)
        emit()

    emit(finished=True)
    print_summary(spec, states)

    if _SHUTDOWN_SIGNALS:
        return 128 + _SHUTDOWN_SIGNALS[0]
    clean = all(
        state.outcome == "completed"
        and state.exit_code == 0
        and state.outputs_present is True
        for state in states
    )
    return 0 if clean else 1


def _shutdown(
    spec: Spec,
    queue: list[AxisState],
    running: list[AxisState],
    log: Callable[[str], None],
) -> None:
    """Terminate every survivor and reap it. Never leave orphans."""

    signum = _SHUTDOWN_SIGNALS[0]
    log(f"signal {signum} received -- terminating {len(running)} running axes")
    for state in queue:
        state.outcome = "not_started"
    for state in running:
        state.outcome = "terminated"
        _signal_axis(state, signal.SIGTERM)

    deadline = time.monotonic() + spec.term_grace_seconds
    while running and time.monotonic() < deadline:
        for state in list(running):
            proc = state.proc
            if proc is None:
                running.remove(state)
                continue
            exit_code = proc.poll()
            if exit_code is not None:
                _reap(state, exit_code)
                running.remove(state)
        if running:
            time.sleep(POLL_INTERVAL_SECONDS)

    for state in list(running):
        proc = state.proc
        if proc is None:
            running.remove(state)
            continue
        _signal_axis(state, signal.SIGKILL)
        try:
            exit_code = proc.wait(timeout=spec.term_grace_seconds)
        except subprocess.TimeoutExpired:
            # Unkillable (uninterruptible sleep). Say so rather than pretend.
            state.error = "child did not exit after SIGKILL"
            log(f"{state.plan.axis_id}: still alive after SIGKILL")
            running.remove(state)
            continue
        _reap(state, exit_code)
        running.remove(state)


def print_summary(spec: Spec, states: list[AxisState]) -> None:
    width = max((len(state.plan.axis_id) for state in states), default=1)
    print(f"--- {spec.label}: {len(states)} axes ---", flush=True)
    for state in states:
        exit_code = "-" if state.exit_code is None else str(state.exit_code)
        wall = "-" if state.wall_seconds is None else f"{state.wall_seconds:.3f}"
        line = (
            f"{state.plan.axis_id:<{width}}  {state.outcome:<21} "
            f"exit={exit_code:<5} wall={wall:>10}s  log={state.plan.log_path}"
        )
        if state.outputs_present is False:
            # Printed even though the outcome reads `completed`: the summary must
            # not let a no-output axis look like a clean success.
            line += f"  MISSING_OUTPUTS={'; '.join(state.outputs_missing)}"
        if state.error:
            line += f"  error={state.error}"
        print(line, flush=True)


def print_plan(spec: Spec, status_path: Path) -> None:
    print(f"plan: {spec.label}")
    print(f"  spec:           {spec.path}")
    print(f"  spec_sha256:    {spec.sha256}")
    print(f"  status:         {status_path}")
    print(f"  cwd:            {spec.cwd}")
    print(f"  max_concurrent: {spec.max_concurrent} (host cpus: {os.cpu_count()})")
    print(f"  nice:           {spec.nice}")
    print(f"  term_grace:     {spec.term_grace_seconds:g}s")
    print(f"  thread pins:    {', '.join(f'{pin}=1' for pin in THREAD_PINS)}")
    print(f"  axes:           {len(spec.axes)}")
    for plan in spec.axes:
        print(
            f"    {plan.axis_id}  budget={plan.wall_budget_seconds:g}s  "
            f"log={plan.log_path}"
        )
        print(f"      argv: {shlex.join(plan.argv)}")
        if plan.declared_outputs:
            print(
                "      outputs (verified non-empty after the axis ends): "
                + ", ".join(str(path) for path in plan.declared_outputs)
            )
    print("dry-run: spec is valid; nothing was launched.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the independent axes of one frozen contract concurrently."
    )
    parser.add_argument("--spec", required=True, help="Path to the JSON axis spec.")
    parser.add_argument(
        "--status", required=True, help="Path of the status JSON to (re)write."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the spec and print the plan without launching anything.",
    )
    args = parser.parse_args()

    repo_root = root()
    status_path = Path(args.status)
    if not status_path.is_absolute():
        status_path = (Path.cwd() / status_path).resolve()

    try:
        spec = load_spec(Path(args.spec), repo_root)
        validate_paths(spec, status_path, create=not args.dry_run)
    except SpecError as exc:
        print(f"spec error: {args.spec}\n{exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc

    if args.dry_run:
        print_plan(spec, status_path)
        raise SystemExit(0)

    raise SystemExit(run(spec, StatusWriter(str(status_path))))


if __name__ == "__main__":
    main()
