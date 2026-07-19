#!/usr/bin/env python
"""Sweep phase-1 non-LRS Rodas5P step caps with runtime and parity gates."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from profile_nonlrs_hotpath_multitool import _build_nonlrs_problem, _device_for_platform, _sync
from rabbit.jax.solver_jax_rodas5p import jax_rodas5p_solve_batch


jax.config.update("jax_enable_x64", True)


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _phase1(problem: dict[str, Any], args: argparse.Namespace, *, h_max: float) -> Any:
    return jax_rodas5p_solve_batch(
        problem["rhs_p1"],
        problem["y0"],
        (0.0, 50.0),
        event_fn=problem["event_p1"],
        rtol=float(args.rtol),
        atol=float(args.atol),
        max_steps=int(args.max_steps),
        h_max=float(h_max),
        h_min=float(args.h_min),
        event_refine_steps=int(args.event_refine_steps),
        batch_linear_solver_backend=str(args.batch_linear_solver_backend),
        jacobian_reuse_experimental=False,
        jacobian_refresh_stride=1,
    )


def _time_phase1(
    problem: dict[str, Any],
    args: argparse.Namespace,
    *,
    h_max: float,
) -> tuple[Any, dict[str, Any]]:
    def once() -> Any:
        result = _phase1(problem, args, h_max=h_max)
        _sync(result)
        return result

    rss0 = _rss_mb()
    t0 = time.perf_counter()
    result = once()
    cold_s = time.perf_counter() - t0
    samples = []
    for _ in range(int(args.warm_reps)):
        t1 = time.perf_counter()
        result = once()
        samples.append(time.perf_counter() - t1)
    return result, {
        "cold_s": float(cold_s),
        "warm_samples_s": [float(x) for x in samples],
        "warm_min_s": float(min(samples)) if samples else None,
        "warm_median_s": float(sorted(samples)[len(samples) // 2]) if samples else None,
        "warm_max_s": float(max(samples)) if samples else None,
        "rss_before_mb": float(rss0),
        "rss_after_mb": float(_rss_mb()),
    }


def _arr(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=np.float64)


def _finite_max(value: np.ndarray) -> float:
    finite = value[np.isfinite(value)]
    return float(np.max(finite)) if finite.size else float("nan")


def _summary(result: Any) -> dict[str, Any]:
    success = np.asarray(jax.device_get(result.success), dtype=bool)
    event = np.asarray(jax.device_get(result.event_triggered), dtype=bool)
    n_steps = np.asarray(jax.device_get(result.n_steps), dtype=np.int64)
    n_reject = np.asarray(jax.device_get(result.n_reject), dtype=np.int64)
    h_final = _arr(result.h_final)
    n_final = _arr(result.N_final)
    diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    return {
        "success_fraction": float(np.mean(success)),
        "event_fraction": float(np.mean(event)),
        "n_steps_max": int(np.max(n_steps)),
        "n_steps_mean": float(np.mean(n_steps)),
        "n_reject_max": int(np.max(n_reject)),
        "n_reject_mean": float(np.mean(n_reject)),
        "h_final_min": float(np.min(h_final)),
        "h_final_max": float(np.max(h_final)),
        "N_final_min": float(np.min(n_final)),
        "N_final_max": float(np.max(n_final)),
        "jacobian_refresh_count_max": int(np.max(jax.device_get(diagnostics.get("jacobian_refresh_count")))),
        "jacobian_reuse_count_max": int(np.max(jax.device_get(diagnostics.get("jacobian_reuse_count")))),
    }


def _diff(reference: Any, candidate: Any) -> dict[str, float]:
    y_ref = _arr(reference.y_final)
    y_got = _arr(candidate.y_final)
    n_ref = _arr(reference.N_final)
    n_got = _arr(candidate.N_final)
    y_abs = np.abs(y_got - y_ref)
    y_rel = y_abs / np.maximum(np.abs(y_ref), 1.0e-300)
    n_abs = np.abs(n_got - n_ref)
    return {
        "max_y_abs": _finite_max(y_abs),
        "max_y_rel": _finite_max(y_rel),
        "max_N_abs": _finite_max(n_abs),
        "finite_y_pair_fraction": float(np.mean(np.isfinite(y_ref) & np.isfinite(y_got))),
    }


def _gate(
    row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    summary = row["summary"]
    if float(summary["success_fraction"]) < float(args.min_success_fraction):
        reasons.append("success_fraction")
    if float(summary["event_fraction"]) < float(args.min_event_fraction):
        reasons.append("event_fraction")
    diff = row.get("diff_vs_baseline")
    if diff:
        if float(diff["finite_y_pair_fraction"]) < 1.0:
            reasons.append("finite_y_pair")
        y_abs_ok = float(diff["max_y_abs"]) <= float(args.y_atol)
        y_rel_ok = float(diff["max_y_rel"]) <= float(args.y_rtol)
        if not (y_abs_ok or y_rel_ok):
            reasons.append("y_final_parity")
        if float(diff["max_N_abs"]) > float(args.N_atol):
            reasons.append("N_final_parity")
    return not reasons, reasons


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Non-LRS Phase-1 Step Policy Sweep",
        "",
        f"- platform: `{payload['platform']}`",
        f"- batch size: `{payload['args']['batch_size']}`",
        f"- grid: `N_theta={payload['args']['N_theta']}, N_phi={payload['args']['N_phi']}, N_q={payload['args']['N_q']}`",
        f"- baseline h_max: `{payload['baseline_h_max']}`",
        "",
        "| h_max | gate | warm median | steps | rejects | jac refresh | max y abs diff | max N diff |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        diff = row.get("diff_vs_baseline") or {}
        lines.append(
            f"| {row['h_max']} | `{row['gate']}` | {row['timing'].get('warm_median_s')} | "
            f"{row['summary']['n_steps_max']} | {row['summary']['n_reject_max']} | "
            f"{row['summary']['jacobian_refresh_count_max']} | "
            f"{float(diff.get('max_y_abs', 0.0)):.3e} | {float(diff.get('max_N_abs', 0.0)):.3e} |"
        )
    passing = [row for row in payload["rows"] if row["gate"] == "pass"]
    lines += ["", "## Recommendation", ""]
    if passing:
        fastest = sorted(
            passing,
            key=lambda row: float(row["timing"].get("warm_median_s") or row["timing"]["cold_s"]),
        )[0]
        fewest_rejects = sorted(passing, key=lambda row: int(row["summary"]["n_reject_max"]))[0]
        lines.append(
            f"Fastest passing cap: `h_max={fastest['h_max']}` "
            f"({fastest['timing'].get('warm_median_s')} s warm median)."
        )
        lines.append(
            f"Fewest-reject passing cap: `h_max={fewest_rejects['h_max']}` "
            f"({fewest_rejects['summary']['n_reject_max']} rejects)."
        )
    else:
        lines.append("No h_max candidate passed the event and final-state parity gate.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="artifacts/nonlrs_phase1_step_policy_sweep")
    parser.add_argument("--platform", choices=["cpu", "gpu", "rocm", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--N-theta", type=int, default=2)
    parser.add_argument("--N-phi", type=int, default=2)
    parser.add_argument("--N-q", type=int, default=4)
    parser.add_argument("--n-reactions", type=int, default=12)
    parser.add_argument("--correction-level", type=int, default=0)
    parser.add_argument("--N-eff", type=float, default=3.044)
    parser.add_argument("--f-nu", type=float, default=0.40523)
    parser.add_argument("--eta", type=float, default=6.1e-10)
    parser.add_argument("--tau-n", type=float, default=879.4)
    parser.add_argument("--sigma-scale", type=float, default=1.0e-3)
    parser.add_argument("--T-start", type=float, default=10.0)
    parser.add_argument("--T-handoff", type=float, default=0.08)
    parser.add_argument("--T-end", type=float, default=0.005)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--h-min", type=float, default=1.0e-14)
    parser.add_argument("--baseline-h-max", type=float, default=0.5)
    parser.add_argument("--h-max-values", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    parser.add_argument("--event-refine-steps", type=int, default=8)
    parser.add_argument(
        "--batch-linear-solver-backend",
        choices=["auto", "jax_tensorized", "gpu_tensorized"],
        default="auto",
    )
    parser.add_argument("--warm-reps", type=int, default=1)
    parser.add_argument("--y-atol", type=float, default=1.0e-6)
    parser.add_argument("--y-rtol", type=float, default=1.0e-4)
    parser.add_argument("--N-atol", type=float, default=1.0e-5)
    parser.add_argument("--min-success-fraction", type=float, default=1.0)
    parser.add_argument("--min-event-fraction", type=float, default=1.0)
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = _device_for_platform(str(args.platform))
    problem = _build_nonlrs_problem(args, device)
    baseline_result, baseline_timing = _time_phase1(problem, args, h_max=float(args.baseline_h_max))

    values = []
    seen = set()
    for value in [float(args.baseline_h_max), *[float(x) for x in args.h_max_values]]:
        key = float(value)
        if key not in seen:
            seen.add(key)
            values.append(key)

    payload: dict[str, Any] = {
        "profile_contract": "nonlrs_phase1_step_policy_sweep_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": jax.default_backend(),
        "device": str(device),
        "args": vars(args),
        "baseline_h_max": float(args.baseline_h_max),
        "baseline_timing": baseline_timing,
        "rows": [],
    }

    for h_max in values:
        if h_max == float(args.baseline_h_max):
            result = baseline_result
            timing = baseline_timing
        else:
            result, timing = _time_phase1(problem, args, h_max=float(h_max))
        row = {
            "h_max": float(h_max),
            "timing": timing,
            "summary": _summary(result),
            "diff_vs_baseline": _diff(baseline_result, result),
        }
        passed, reasons = _gate(row, args)
        row["gate"] = "pass" if passed else "fail"
        row["gate_reasons"] = reasons
        payload["rows"].append(row)

    json_path = outdir / "step_policy_sweep_summary.json"
    md_path = outdir / "step_policy_sweep_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_safe), encoding="utf-8")
    _write_markdown(md_path, payload)
    passing = [
        {"h_max": row["h_max"], "warm_median_s": row["timing"].get("warm_median_s"), "rejects": row["summary"]["n_reject_max"]}
        for row in payload["rows"]
        if row["gate"] == "pass"
    ]
    print(json.dumps({"summary": str(json_path), "report": str(md_path), "passing": passing}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
