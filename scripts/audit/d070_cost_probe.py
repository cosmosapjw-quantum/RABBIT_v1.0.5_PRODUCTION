#!/usr/bin/env python3
"""BD622 D-070 per-evaluation COST PROBE. Timing only -- NOT SCIENTIFIC EVIDENCE.

Purpose: measure the wall cost of a single right-hand-side evaluation for a set
of candidate angular/radial/spectral resolutions, so that the next robustness-envelope contract can
state defensible wall budgets *before* it is frozen and before any output byte
exists. Nothing here adjudicates anything.

This file is deliberately built so that it cannot be mistaken for, or promoted
into, evidence:

  1. NO INTEGRATION. It evaluates ``_trajectory_core.make_rhs`` at one fixed
     state and never advances it. It does not import ``solve_ivp``, does not
     construct a terminal event, and never calls the trajectory module's
     endpoint / residual / moment / density helpers. ``source_guard`` parses
     this file's own AST at startup and refuses to run if any of that changed.
  2. NO ANCHORS. It never imports ``_f10c2_anchors``, so the Rust comparison
     spectrum is not even resident in the process. The one hash it checks is
     the frozen module's own sha256, hardcoded below.
  3. NO PHYSICS IN THE OUTPUT. The right-hand-side return value is checked for
     finiteness -- a bool -- and dropped. ``report_key_violations`` rejects the
     report before it is written if any key looks like a physics readout.
  4. LABELLED. Every record carries ``artifact_class`` and ``disclosure``.
  5. ``--out`` is required, exactly as in the r4 driver, so a mis-invocation
     cannot silently discard the run while exiting 0.

The measured cost is a *budgeting* input only. Per-evaluation cost varies along
a real trajectory (the whole-reaction domain-rejection count is state
dependent), so a projection built from a single fixed state is an
order-of-magnitude figure, not a trajectory average. Say so wherever it is used.

Usage:
  PYTHONPATH=src:scripts/audit venv/bin/python scripts/audit/d070_cost_probe.py \
      --out PATH [--config-id ID] [--evals N] [--warmup N] [--dry-run]

Exit codes: 0 ok, 20 mechanical (wall budget, evaluation failure, unwritable
``--out``), 30 environment or anti-evidence guard refusal.
"""

from __future__ import annotations

import argparse
import ast
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

# BLAS threading is pinned BEFORE `import numpy`, and before importing anything
# that imports numpy. OpenBLAS reads its thread configuration at library
# initialisation, so a pin set after the import is a silent no-op and the run
# would be multi-threaded and non-reproducible. This ordering is not optional.
# (`_trajectory_core` re-asserts three of these at its own import for the same
# reason; the values are identical, so there is no conflict.)
THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _pin in THREAD_ENV_VARS:
    os.environ[_pin] = "1"

import numpy as np  # noqa: E402
import scipy  # noqa: E402

import _trajectory_core as core  # noqa: E402

# --- identity ----------------------------------------------------------------

ARTIFACT_CLASS = "COST_PROBE_NOT_EVIDENCE"

EXPECTED_MODULE_SHA256 = (
    "760a7c044081e507fae9d5695b301bd44f6466d96322c46f53b77161e32b558a"
)

PURPOSE = (
    "Measure wall seconds per right-hand-side evaluation for candidate "
    "resolutions, to set wall budgets in a contract that is frozen before any "
    "output. This artifact adjudicates nothing and supports no claim."
)

CAVEAT = (
    "Per-evaluation cost is state dependent (the whole-reaction domain-"
    "rejection count varies along a trajectory), so a projection built from a "
    "single fixed state is an order-of-magnitude budgeting figure, not a "
    "trajectory average. Do not quote it as a measured trajectory cost."
)

# The one fixed, disclosed, non-anchor state at which every configuration is
# timed: the N = 0 equilibrium initial condition of the r4 driver.
FIXED_E_FOLDS = 0.0

DISCLOSURE = (
    "Timing only. The right-hand side is evaluated at one fixed, disclosed, "
    "non-anchor state: the N = 0 equilibrium initial condition built by "
    "_trajectory_core.initial_state -- the complementary-log-log encoding of "
    "the equilibrium pair logits -y on each configuration's own grid, with the "
    "photon and comoving temperatures both at Setup.t_start and cosmic time "
    "zero. That state is a fixed input chosen before any band was set, it is "
    "not any code's endpoint, and it is never compared to any anchor. No "
    "integration is performed, no right-hand-side value is retained, and no "
    "physics quantity is computed, stored or printed."
)

# --- anti-evidence guards ----------------------------------------------------


class ProbeGuardError(RuntimeError):
    """An anti-evidence guard refused to let the probe continue."""


# Importing any of these would put integration machinery or the Rust comparison
# anchors into this process. Both lists are checked against this file's own AST.
FORBIDDEN_IMPORT_NAMES = ("solve_ivp", "odeint", "quad", "_f10c2_anchors")

# Calling any of these would turn a cost probe into a physics run. They appear
# below only as string constants, which the AST reports as Constant nodes and
# never as Call nodes, so the guard cannot trip on its own definition.
FORBIDDEN_CALL_NAMES = (
    "solve_ivp",
    "odeint",
    "quad",
    "run_integration",
    "make_terminal_event",
    "endpoint_summary",
    "comoving_energies",
    "coupled_residuals",
    "checkpoint_states",
    "spectral_moments",
    "anchor_moments",
    "pair_densities",
    "equilibrium_tail_fraction",
    "evaluate_independent_collision_action",
    "independent_thermodynamics",
    "cloglog_to_occupation",
    "electromagnetic_eos_adaptive",
)

# If a report key contains any of these, a reader could extract a physics
# number from this artifact, which is exactly what it must not carry.
FORBIDDEN_REPORT_KEY_SUBSTRINGS = (
    "n_eff",
    "neff",
    "n_end",
    "t_end",
    "t_gamma",
    "t_cm",
    "spectr",
    "moment",
    "occupation",
    "cloglog",
    "anchor",
    "rust",
    "residual",
    "enclos",
    "hubble",
    "rho",
    "verdict",
    "heating",
    "enhance",
    "split_ratio",
    "rejection",
    "roundoff",
    "mutant",
    "checkpoint",
)


def source_guard() -> dict[str, Any]:
    """Structural proof, from this file's own AST, that it cannot integrate.

    A comment promising "no integration here" is not a guard. Parsing the file
    and refusing to start when a forbidden import or call appears is.
    """

    try:
        with open(__file__, "rb") as handle:
            source = handle.read()
    except OSError as exc:
        raise ProbeGuardError(f"cannot read own source for the guard: {exc!r}")

    tree = ast.parse(source)
    bad_imports: list[str] = []
    bad_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORT_NAMES:
                    bad_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in FORBIDDEN_IMPORT_NAMES:
                bad_imports.append(node.module or "<relative>")
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORT_NAMES:
                    bad_imports.append(alias.name)
        elif isinstance(node, ast.Call):
            called = node.func
            if isinstance(called, ast.Name):
                name: str | None = called.id
            elif isinstance(called, ast.Attribute):
                name = called.attr
            else:
                name = None
            if name is not None and name in FORBIDDEN_CALL_NAMES:
                bad_calls.append(name)

    return {
        "method": "ast parse of this file; forbidden imports and calls",
        "forbidden_imports": list(FORBIDDEN_IMPORT_NAMES),
        "forbidden_calls": list(FORBIDDEN_CALL_NAMES),
        "violations_import": sorted(set(bad_imports)),
        "violations_call": sorted(set(bad_calls)),
        "ok": not bad_imports and not bad_calls,
    }


def report_key_violations(node: Any, path: str = "") -> list[str]:
    """Every key path in ``node`` that looks like a physics readout."""

    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            lowered = str(key).lower()
            if any(bad in lowered for bad in FORBIDDEN_REPORT_KEY_SUBSTRINGS):
                found.append(here)
            found.extend(report_key_violations(value, here))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(report_key_violations(value, f"{path}[{index}]"))
    return found


def label_violations(report: dict[str, Any]) -> list[str]:
    """Records missing the artifact class or the disclosure string."""

    missing: list[str] = []
    records = [("report", report)]
    records += [
        (f"results[{i}]", row) for i, row in enumerate(report.get("results", []))
    ]
    for name, record in records:
        if record.get("artifact_class") != ARTIFACT_CLASS:
            missing.append(f"{name}.artifact_class")
        if not isinstance(record.get("disclosure"), str) or not record["disclosure"]:
            missing.append(f"{name}.disclosure")
    return missing


def write_guarded(reporter: core.Reporter, report: dict[str, Any]) -> int | None:
    """Write only after both content guards pass, on the exact bytes to be written.

    Returns None on success, or the exit code to fail closed with: 30 when a
    guard refuses, 20 when the report cannot be written at all.
    """

    reporter.stamp(report)  # stamp first: the timestamps are guarded too
    keys = report_key_violations(report)
    if keys:
        print(f"REPORT_GUARD_FAIL physics-shaped report keys: {keys}")
        return 30
    labels = label_violations(report)
    if labels:
        print(f"REPORT_GUARD_FAIL unlabelled records: {labels}")
        return 30
    try:
        reporter.write(report)
    except OSError as exc:
        print(f"ERROR cannot write --out path: {exc}")
        return 20
    return None


# --- configurations ----------------------------------------------------------


@dataclass(frozen=True)
class ProbeConfig:
    """One candidate resolution. Inputs only -- nothing measured lives here."""

    id: str
    order: int
    y_max: float
    incoming_polar_order: int
    final_polar_order: int
    electron_radial_order: int


BASE_ID = "BASE"

# BASE first: every other configuration is reported as a ratio to it.
CONFIGS: tuple[ProbeConfig, ...] = (
    ProbeConfig(BASE_ID, 48, 24.0, 4, 4, 24),
    ProbeConfig("R1", 48, 24.0, 8, 4, 24),
    ProbeConfig("R2", 48, 24.0, 4, 8, 24),
    ProbeConfig("R3", 48, 24.0, 4, 4, 48),
    ProbeConfig("R4", 48, 24.0, 6, 6, 32),
    ProbeConfig("R5", 60, 24.0, 4, 4, 24),
    ProbeConfig("R6", 72, 24.0, 4, 4, 24),
    ProbeConfig("R7", 96, 24.0, 4, 4, 24),
    ProbeConfig("D60", 60, 30.0, 4, 4, 24),
)

# The frozen module rejects order < 8 and non-positive y_max, and fixes the
# final azimuth order at four midpoint nodes. Restated here so a bad
# configuration is refused before anything is built.
MIN_ORDER = 8
FIXED_FINAL_AZIMUTH_ORDER = 4

# The deepest node of the Rust comparison rule is y = 22.18412259016373. Any
# configuration whose domain stops short of that could not, later, be used for
# a comparison at all -- so a probe that timed one would be timing a
# configuration nobody can use. This bound is a usability filter on the inputs;
# no anchor value is imported, read, or compared here.
MIN_Y_MAX = 22.185

# Measured evaluation counts of the r3/r4 trajectories, used only to project
# wall hours from the measured per-evaluation cost.
EVALS_BASE_LENGTH = 3694
EVALS_HOLDOUT_LENGTH = 4331

# The D-069 contract asserts this per-evaluation cost at order 60 / y_max 30,
# but no committed artifact backs the number. The D60 configuration exists to
# check it. The comparison is a cost comparison and carries no scientific
# weight either way.
ASSERTED_CONFIG_ID = "D60"
ASSERTED_SECONDS_PER_EVAL = 4.754

DEFAULT_EVALS = 5
DEFAULT_WARMUP = 1
# Generous but finite: an unbounded probe is not fail-closed. Breach is a
# mechanical exit 20 that keeps everything already measured.
DEFAULT_WALL_BUDGET_SECONDS = 3.0 * 3600.0


def config_problem(config: ProbeConfig) -> str | None:
    """Why ``config`` is unusable, or None."""

    if config.order < MIN_ORDER:
        return f"{config.id}: order {config.order} < {MIN_ORDER}"
    if not (config.y_max >= MIN_Y_MAX):
        return f"{config.id}: y_max {config.y_max} < {MIN_Y_MAX}"
    for name in ("incoming_polar_order", "final_polar_order", "electron_radial_order"):
        if int(getattr(config, name)) < 2:
            return f"{config.id}: {name} must be at least two"
    return None


def resolved_inputs(config: ProbeConfig) -> dict[str, Any]:
    """The fully resolved configuration, derivable without building anything."""

    return {
        "id": config.id,
        "order": int(config.order),
        "y_max": float(config.y_max),
        "incoming_polar_order": int(config.incoming_polar_order),
        "final_polar_order": int(config.final_polar_order),
        "final_azimuth_order": FIXED_FINAL_AZIMUTH_ORDER,
        "electron_radial_order": int(config.electron_radial_order),
        "node_density_per_unit_y": float(config.order) / float(config.y_max),
        "state_size": 3 * int(config.order) + 2,
    }


# --- timing ------------------------------------------------------------------


def time_config(
    config: ProbeConfig,
    evals: int,
    warmup: int,
    deadline: core.Deadline,
) -> dict[str, Any]:
    """Wall seconds per right-hand-side call at the fixed disclosed state."""

    setup = core.build_setup(
        config.order,
        config.y_max,
        incoming_polar_order=config.incoming_polar_order,
        final_polar_order=config.final_polar_order,
        electron_radial_order=config.electron_radial_order,
        label=f"cost-probe-{config.id}",
    )
    if int(setup.config.final_azimuth_order) != FIXED_FINAL_AZIMUTH_ORDER:
        raise ProbeGuardError(f"{config.id}: final azimuth order is not fixed at four")
    expected_state_size = 3 * int(config.order) + 2
    if int(setup.state_size) != expected_state_size:
        raise ProbeGuardError(
            f"{config.id}: state size {setup.state_size} != {expected_state_size}"
        )

    _c0, fixed_state = core.initial_state(setup)
    stats = core.Stats()
    # log=None deliberately: the core right-hand-side logger prints the comoving
    # and photon temperatures and their ratio, which this probe is forbidden to
    # emit. The Stats object still accumulates the module's own per-evaluation
    # diagnostics; none of them are read, and none reach the report.
    rhs = core.make_rhs(setup, stats, deadline, log=None)

    def one_call() -> tuple[float, bool]:
        state = fixed_state.copy()
        start = time.perf_counter()
        derivative = rhs(FIXED_E_FOLDS, state)
        elapsed = time.perf_counter() - start
        # The derivative is a physics quantity. It is reduced to a finiteness
        # bool here and then goes out of scope; nothing derived from it, and no
        # summary of it, reaches the report.
        return elapsed, bool(np.all(np.isfinite(derivative)))

    finite = True
    for _ in range(warmup):
        _discarded, ok = one_call()
        finite = finite and ok

    samples: list[float] = []
    for _ in range(evals):
        elapsed, ok = one_call()
        samples.append(float(elapsed))
        finite = finite and ok

    median = float(statistics.median(samples))
    inputs = resolved_inputs(config)
    inputs["first_node"] = float(setup.grid.nodes[0])
    inputs["last_node"] = float(setup.grid.nodes[-1])

    record: dict[str, Any] = {
        "artifact_class": ARTIFACT_CLASS,
        "disclosure": DISCLOSURE,
        "id": config.id,
        "resolved_config": inputs,
        "state_size": int(setup.state_size),
        "warmup_discarded": int(warmup),
        "evals": int(evals),
        "seconds_per_eval_median": median,
        "seconds_per_eval_min": float(min(samples)),
        "seconds_per_eval_max": float(max(samples)),
        "seconds_per_eval_samples": samples,
        "ratio_to_base": None,
        "projections": {
            "base_length_evals": EVALS_BASE_LENGTH,
            "projected_hours_base_length": median * EVALS_BASE_LENGTH / 3600.0,
            "holdout_length_evals": EVALS_HOLDOUT_LENGTH,
            "projected_hours_holdout_length": median * EVALS_HOLDOUT_LENGTH / 3600.0,
        },
        "derivative_finite": finite,
        "derivative_retained": False,
        "status": "ok",
    }

    if config.id == ASSERTED_CONFIG_ID:
        record["d069_contract_assertion"] = {
            "asserted_seconds_per_eval": ASSERTED_SECONDS_PER_EVAL,
            "asserted_for": "order 60 / y_max 30",
            "measured_seconds_per_eval_median": median,
            "measured_minus_asserted_seconds": median - ASSERTED_SECONDS_PER_EVAL,
            "measured_over_asserted": median / ASSERTED_SECONDS_PER_EVAL,
            "note": (
                "The D-069 contract asserts this per-evaluation cost but no "
                "committed artifact backs it. This is the first measurement. "
                "It is a cost comparison at one fixed state and carries no "
                "scientific weight."
            ),
        }
    return record


def apply_ratios(records: list[dict[str, Any]]) -> None:
    """Express every measured cost as a ratio to BASE, in place."""

    base = next((r for r in records if r["id"] == BASE_ID), None)
    if base is None:
        for record in records:
            record["ratio_to_base"] = None
            record["ratio_note"] = (
                f"{BASE_ID} was not run in this invocation, so no ratio is defined"
            )
        return
    denominator = float(base["seconds_per_eval_median"])
    for record in records:
        record["ratio_to_base"] = float(record["seconds_per_eval_median"]) / denominator


# --- plan printing -----------------------------------------------------------


def print_plan(report: dict[str, Any], out_path: str, dry_run: bool) -> None:
    parameters = report["probe_parameters"]
    print(f"artifact_class = {report['artifact_class']}")
    print(f"module_sha256  = {report['module_sha256']}")
    print(
        "versions       = "
        f"python {report['versions']['python']} "
        f"numpy {report['versions']['numpy']} "
        f"scipy {report['versions']['scipy']}"
    )
    print(f"processor      = {report['platform']['processor']!r}")
    print(f"thread pins    = {report['thread_env_as_seen_after_import']}")
    print(
        "probe          = "
        f"{parameters['evals']} timed evals, "
        f"{parameters['warmup_discarded']} warmup discarded, "
        f"wall budget {parameters['wall_budget_seconds']:.0f} s"
    )
    print(
        "fixed state    = "
        f"N={parameters['fixed_state_e_folds']} "
        f"T_photon=T_comoving={parameters['fixed_state_temperature_mev']} MeV "
        f"from {parameters['fixed_state_source']}"
    )
    print(
        "projections    = "
        f"{EVALS_BASE_LENGTH} evals (base length), "
        f"{EVALS_HOLDOUT_LENGTH} evals (holdout length)"
    )
    print(f"out            = {out_path}" + (" (NOT written: dry run)" if dry_run else ""))
    print()
    header = (
        f"{'id':<5} {'order':>5} {'y_max':>7} {'in_pol':>6} {'fin_pol':>7} "
        f"{'fin_azi':>7} {'e_radial':>8} {'density':>8} {'state':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in report["plan"]:
        print(
            f"{row['id']:<5} {row['order']:>5} {row['y_max']:>7.2f} "
            f"{row['incoming_polar_order']:>6} {row['final_polar_order']:>7} "
            f"{row['final_azimuth_order']:>7} {row['electron_radial_order']:>8} "
            f"{row['node_density_per_unit_y']:>8.4f} {row['state_size']:>6}"
        )
    print()
    print(
        f"{ASSERTED_CONFIG_ID} will be reported against the unbacked D-069 "
        f"contract assertion of {ASSERTED_SECONDS_PER_EVAL} s/eval."
    )


# --- entry point -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D-070 per-evaluation cost probe (timing only, NOT EVIDENCE)"
    )
    parser.add_argument("--out", required=True, help="JSON report path (required)")
    parser.add_argument(
        "--config-id",
        choices=[config.id for config in CONFIGS],
        default=None,
        help="time only this configuration (default: all, BASE first)",
    )
    parser.add_argument("--evals", type=int, default=DEFAULT_EVALS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--wall-budget-seconds", type=float, default=DEFAULT_WALL_BUDGET_SECONDS
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "resolve and print the plan, then exit 0. Builds no grid, calls no "
            "right-hand side, writes no file."
        ),
    )
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv[1:])
    if args.evals < 1:
        parser.error("--evals must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if not args.wall_budget_seconds > 0.0:
        parser.error("--wall-budget-seconds must be positive")

    try:
        guard = source_guard()
    except ProbeGuardError as exc:
        print(f"SOURCE_GUARD_ERROR {exc}")
        return 30
    if not guard["ok"]:
        print(
            "SOURCE_GUARD_FAIL "
            f"imports={guard['violations_import']} calls={guard['violations_call']}"
        )
        return 30

    digest = core.module_sha256()
    if digest != EXPECTED_MODULE_SHA256:
        print(f"MODULE_SHA_MISMATCH {digest}")
        return 30

    selected = [c for c in CONFIGS if args.config_id is None or c.id == args.config_id]
    for config in selected:
        problem = config_problem(config)
        if problem is not None:
            print(f"CONFIG_REJECTED {problem}")
            return 30

    # Both temperatures of the fixed state equal Setup.t_start at N = 0:
    # T_comoving = t_start * exp(-0) and T_photon is the t_start component of
    # the initial state vector.
    fixed_temperature = float(core.Setup.t_start)

    report: dict[str, Any] = {
        "artifact_class": ARTIFACT_CLASS,
        "disclosure": DISCLOSURE,
        "purpose": PURPOSE,
        "caveat": CAVEAT,
        "module_sha256": digest,
        "module_sha256_expected": EXPECTED_MODULE_SHA256,
        "source_guard": guard,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "platform": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
        },
        "thread_env_as_seen_after_import": {
            name: os.environ.get(name) for name in THREAD_ENV_VARS
        },
        "probe_parameters": {
            "evals": int(args.evals),
            "warmup_discarded": int(args.warmup),
            "config_id_filter": args.config_id,
            "base_config_id": BASE_ID,
            "wall_budget_seconds": float(args.wall_budget_seconds),
            "fixed_state_e_folds": FIXED_E_FOLDS,
            "fixed_state_temperature_mev": fixed_temperature,
            "fixed_state_source": "_trajectory_core.initial_state",
            "timer": "time.perf_counter",
            "projection_eval_counts": {
                "base_length": EVALS_BASE_LENGTH,
                "holdout_length": EVALS_HOLDOUT_LENGTH,
            },
        },
        "plan": [resolved_inputs(config) for config in selected],
        "results": [],
        "status": "IN_PROGRESS",
    }

    if args.dry_run:
        print_plan(report, args.out, dry_run=True)
        return 0

    reporter = core.Reporter(args.out)
    failure = write_guarded(reporter, report)
    if failure is not None:
        return failure

    print_plan(report, args.out, dry_run=False)
    deadline = core.Deadline(float(args.wall_budget_seconds))
    records: list[dict[str, Any]] = report["results"]

    for config in selected:
        reporter.log(f"{config.id}: timing {args.evals} evaluations")
        try:
            record = time_config(config, args.evals, args.warmup, deadline)
        except ProbeGuardError as exc:
            print(f"REPORT_GUARD_FAIL {exc}")
            return 30
        except TimeoutError as exc:
            report["status"] = "ABORTED_WALL_BUDGET"
            report["error"] = f"{config.id}: {exc}"
            apply_ratios(records)
            write_guarded(reporter, report)
            reporter.log(f"aborted at {config.id}: wall budget")
            return 20
        except Exception as exc:  # noqa: BLE001 - fail closed, keep what is measured
            report["status"] = "ABORTED_ERROR"
            report["error"] = f"{config.id}: {type(exc).__name__}: {exc}"
            apply_ratios(records)
            write_guarded(reporter, report)
            reporter.log(f"aborted at {config.id}: {type(exc).__name__}")
            return 20
        records.append(record)
        apply_ratios(records)
        failure = write_guarded(reporter, report)
        if failure is not None:
            return failure
        reporter.log(
            f"{config.id}: {record['seconds_per_eval_median']:.4f} s/eval "
            f"(min {record['seconds_per_eval_min']:.4f}, "
            f"max {record['seconds_per_eval_max']:.4f})"
        )

    report["status"] = "COMPLETE"
    failure = write_guarded(reporter, report)
    if failure is not None:
        return failure

    for record in records:
        ratio = record["ratio_to_base"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.3f}x"
        reporter.log(
            f"  {record['id']:<5} {record['seconds_per_eval_median']:9.4f} s/eval "
            f"{ratio_text:>8}  "
            f"base-length {record['projections']['projected_hours_base_length']:7.2f} h  "
            f"holdout-length "
            f"{record['projections']['projected_hours_holdout_length']:7.2f} h"
        )
    asserted = next(
        (r for r in records if r["id"] == ASSERTED_CONFIG_ID), None
    )
    if asserted is not None:
        block = asserted["d069_contract_assertion"]
        reporter.log(
            f"{ASSERTED_CONFIG_ID} vs unbacked D-069 assertion "
            f"{ASSERTED_SECONDS_PER_EVAL} s/eval: measured "
            f"{block['measured_seconds_per_eval_median']:.4f} s/eval "
            f"({block['measured_over_asserted']:.3f}x, "
            f"{block['measured_minus_asserted_seconds']:+.4f} s)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
