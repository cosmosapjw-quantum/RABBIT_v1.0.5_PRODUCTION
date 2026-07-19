#!/usr/bin/env python
"""Compare adaptive batched Rodas5P CPU/GPU JAX/XLA linear-solve paths.

The benchmark is intentionally end-to-end at the public batched solver level:
each row runs ``jax_rodas5p_solve_batch`` on the same linear stiff-ish problem.
It compares:

* CPU ``jax_tensorized``: portable XLA dense batch solve.
* GPU ``gpu_tensorized``: explicit XLA GPU dense batch solve.

This script measures the current JAX/XLA GPU implementation and its accuracy
relative to CPU XLA.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp

from rabbit.jax.solver_jax_rodas5p import jax_rodas5p_solve_batch


jax.config.update("jax_enable_x64", True)


def _block(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    if hasattr(value, "y_final"):
        value.y_final.block_until_ready()
    return value


def _time(fn, *, reps: int, warmups: int) -> tuple[dict[str, float], Any]:
    for _ in range(warmups):
        _block(fn())
    samples = []
    last = None
    for _ in range(reps):
        t0 = time.perf_counter()
        last = _block(fn())
        samples.append(time.perf_counter() - t0)
    return (
        {
            "min_s": float(min(samples)),
            "median_s": float(statistics.median(samples)),
            "max_s": float(max(samples)),
            "reps": int(reps),
            "warmups": int(warmups),
        },
        last,
    )


def _device_for(label: str):
    if label == "cpu":
        return jax.devices("cpu")[0]
    for alias in ("gpu", "cuda", "rocm"):
        try:
            devices = jax.devices(alias)
        except Exception:
            devices = []
        if devices:
            return devices[0]
    return None


def _problem(dim: int, batch: int, device):
    idx = jnp.arange(dim, dtype=jnp.float64)
    diag = -0.35 - 0.03 * idx
    matrix = jnp.diag(diag)
    matrix = matrix + 0.015 / (1.0 + jnp.abs(idx[:, None] - idx[None, :]))
    phase = jnp.arange(batch * dim, dtype=jnp.float64).reshape(batch, dim)
    y0 = 0.5 + 0.2 * jnp.sin(phase + 1.0)
    return jax.device_put(matrix, device), jax.device_put(y0, device)


def _make_solve(matrix, y0, *, backend: str, n_end: float, max_steps: int):
    def rhs_fn(_N, y):
        return matrix @ y

    def jac_fn(_N, _y):
        return matrix

    def solve_once():
        return jax_rodas5p_solve_batch(
            rhs_fn,
            y0,
            (0.0, float(n_end)),
            rtol=1.0e-7,
            atol=1.0e-9,
            h_max=float(n_end),
            max_steps=int(max_steps),
            jac_fn=jac_fn,
            batch_linear_solver_backend=backend,
        )

    return solve_once


def _run_one(
    *,
    device_label: str,
    backend: str,
    dim: int,
    batch: int,
    reps: int,
    warmups: int,
    n_end: float,
    max_steps: int,
) -> dict[str, Any]:
    device = _device_for(device_label)
    row: dict[str, Any] = {
        "device_label": device_label,
        "backend": backend,
        "dim": int(dim),
        "batch": int(batch),
        "benchmark_contract": "rodas5p_gpu_vs_cpu_batch_solver_v1",
    }
    if device is None:
        row["skipped"] = True
        row["skip_reason"] = f"no_{device_label}_device"
        return row
    row["device"] = str(device)
    row["device_platform"] = str(getattr(device, "platform", device_label))
    if backend == "gpu_tensorized" and device_label != "gpu":
        row["skipped"] = True
        row["skip_reason"] = "gpu_tensorized_requires_gpu_row"
        return row
    matrix, y0 = _problem(dim, batch, device)
    solve_once = _make_solve(
        matrix,
        y0,
        backend=backend,
        n_end=n_end,
        max_steps=max_steps,
    )
    try:
        timing, out = _time(solve_once, reps=reps, warmups=warmups)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    row["timing"] = timing
    row["success_all"] = bool(jnp.all(out.success))
    row["diagnostics"] = dict(out.diagnostics)
    row["y_final"] = jax.device_get(out.y_final)
    return row


def _attach_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("device_label") == "cpu" and row.get("backend") == "jax_tensorized" and "y_final" in row:
            baselines[(int(row["dim"]), int(row["batch"]))] = row
    for row in rows:
        if "y_final" not in row:
            continue
        base = baselines.get((int(row["dim"]), int(row["batch"])))
        if base is None:
            continue
        ref = jnp.asarray(base["y_final"], dtype=jnp.float64)
        got = jnp.asarray(row["y_final"], dtype=jnp.float64)
        row["max_abs_diff_vs_cpu_jax"] = float(jnp.max(jnp.abs(got - ref)))
        row["max_rel_diff_vs_cpu_jax"] = float(
            jnp.max(jnp.abs(got - ref) / jnp.maximum(jnp.abs(ref), 1.0e-300))
        )
        row["speedup_vs_cpu_jax"] = float(base["timing"]["median_s"]) / float(
            row["timing"]["median_s"]
        )
    for row in rows:
        row.pop("y_final", None)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", nargs="+", type=int, default=[6, 13])
    parser.add_argument("--batches", nargs="+", type=int, default=[16, 128])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--n-end", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    rows = []
    specs = (
        ("cpu", "jax_tensorized"),
        ("gpu", "gpu_tensorized"),
        ("gpu", "auto"),
    )
    for dim in args.dims:
        for batch in args.batches:
            for device_label, backend in specs:
                rows.append(
                    _run_one(
                        device_label=device_label,
                        backend=backend,
                        dim=int(dim),
                        batch=int(batch),
                        reps=max(1, int(args.reps)),
                        warmups=max(0, int(args.warmups)),
                        n_end=float(args.n_end),
                        max_steps=int(args.max_steps),
                    )
                )
    rows = _attach_comparisons(rows)
    payload = {
        "benchmark_contract": "rodas5p_gpu_vs_cpu_batch_solver_v1",
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "env": {"JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "")},
        "rows": rows,
        "summary": {
            "gpu_implementation": "gpu_tensorized_xla_batched_dense_rodas5p_stage_solve",
            "external_custom_call": False,
            "differentiability": "jax_native_for_tensorized_xla_paths",
        },
    }
    if args.json_output:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        for row in rows:
            if "timing" not in row:
                print(row)
            else:
                print(
                    row["device_label"],
                    row["backend"],
                    "dim",
                    row["dim"],
                    "batch",
                    row["batch"],
                    "median_s",
                    f'{row["timing"]["median_s"]:.6e}',
                    "speedup_vs_cpu_jax",
                    f'{row.get("speedup_vs_cpu_jax", 1.0):.3f}',
                    "max_abs",
                    f'{row.get("max_abs_diff_vs_cpu_jax", 0.0):.3e}',
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
