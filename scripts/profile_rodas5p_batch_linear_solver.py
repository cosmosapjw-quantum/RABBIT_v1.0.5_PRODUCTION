#!/usr/bin/env python
"""Profile JAX/XLA dense batch Rodas5P linear solves.

This benchmark exercises the public adaptive batched solver, not only the
standalone LU helper.  It uses a constant linear RHS with an explicit Jacobian
so the timing is dominated by the Rodas5P stage solves and controller rather
than Jacobian construction.
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


def _sync(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    if hasattr(value, "y_final"):
        value.y_final.block_until_ready()
    return value


def _time_call(fn, *, reps: int, warmups: int) -> dict[str, Any]:
    for _ in range(warmups):
        _sync(fn())
    samples = []
    last = None
    for _ in range(reps):
        t0 = time.perf_counter()
        last = _sync(fn())
        samples.append(time.perf_counter() - t0)
    return {
        "reps": int(reps),
        "warmups": int(warmups),
        "min_s": float(min(samples)),
        "median_s": float(statistics.median(samples)),
        "max_s": float(max(samples)),
        "last_shape": tuple(getattr(getattr(last, "y_final", None), "shape", ())),
    }


def _platform_devices(platforms: list[str]) -> list[tuple[str, Any]]:
    aliases = {
        "gpu": ("gpu", "cuda", "rocm"),
        "cuda": ("cuda", "gpu"),
        "rocm": ("rocm", "gpu"),
        "cpu": ("cpu",),
    }
    rows: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for requested in platforms:
        for alias in aliases.get(requested, (requested,)):
            try:
                devices = list(jax.devices(alias))
            except Exception:
                devices = []
            if devices:
                break
        else:
            devices = []
        for device in devices:
            ident = id(device)
            if ident not in seen:
                seen.add(ident)
                rows.append((requested, device))
    return rows


def _device_kind(device: Any) -> str:
    platform = getattr(device, "platform", "")
    return str(platform or device)


def _problem(dim: int, batch: int, device: Any):
    idx = jnp.arange(dim, dtype=jnp.float64)
    matrix = -0.15 * jnp.eye(dim, dtype=jnp.float64)
    matrix = matrix + 0.02 / (1.0 + jnp.abs(idx[:, None] - idx[None, :]))
    y0 = jnp.sin(jnp.arange(batch * dim, dtype=jnp.float64).reshape(batch, dim) + 1.0)
    return jax.device_put(matrix, device), jax.device_put(y0, device)


def _make_solve_fn(
    matrix,
    y0,
    *,
    backend: str,
    n_end: float,
    rtol: float,
    atol: float,
    max_steps: int,
):
    def rhs_fn(_N, y):
        return matrix @ y

    def jac_fn(_N, _y):
        return matrix

    def solve_once():
        return jax_rodas5p_solve_batch(
            rhs_fn,
            y0,
            (0.0, float(n_end)),
            rtol=float(rtol),
            atol=float(atol),
            max_steps=int(max_steps),
            h_max=float(n_end),
            jac_fn=jac_fn,
            batch_linear_solver_backend=backend,
        )

    return solve_once


def _bench(
    *,
    device: Any,
    requested_platform: str,
    backend: str,
    dim: int,
    batch: int,
    reps: int,
    warmups: int,
    n_end: float,
    rtol: float,
    atol: float,
    max_steps: int,
) -> dict[str, Any]:
    device_platform = _device_kind(device)
    row: dict[str, Any] = {
        "benchmark_contract": "rodas5p_batch_linear_solver_profile_v1",
        "requested_platform": requested_platform,
        "device": str(device),
        "device_platform": device_platform,
        "backend": backend,
        "dim": int(dim),
        "batch": int(batch),
    }
    if backend == "gpu_tensorized" and device_platform not in {"gpu", "cuda", "rocm"}:
        row["skipped"] = True
        row["skip_reason"] = "gpu_tensorized_requires_gpu_device"
        return row
    matrix, y0 = _problem(int(dim), int(batch), device)
    solve_once = _make_solve_fn(
        matrix,
        y0,
        backend=backend,
        n_end=n_end,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    try:
        row["forward"] = _time_call(
            solve_once,
            reps=reps,
            warmups=warmups,
        )
    except Exception as exc:
        row["forward_error"] = f"{type(exc).__name__}: {exc}"
        return row
    if backend != "jax_tensorized":
        ref_solve = _make_solve_fn(
            matrix,
            y0,
            backend="jax_tensorized",
            n_end=n_end,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
        )
        got_solve = _make_solve_fn(
            matrix,
            y0,
            backend=backend,
            n_end=n_end,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
        )
        ref = ref_solve()
        got = got_solve()
        row["max_abs_diff_vs_jax_tensorized"] = float(
            jnp.max(jnp.abs(got.y_final - ref.y_final))
        )
        row["max_rel_diff_vs_jax_tensorized"] = float(
            jnp.max(
                jnp.abs(got.y_final - ref.y_final)
                / jnp.maximum(jnp.abs(ref.y_final), 1.0e-300)
            )
        )
        row["diagnostics"] = got.diagnostics
    return row


def _speedups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline: dict[tuple[str, int, int], float] = {}
    for row in rows:
        if row.get("backend") == "jax_tensorized" and "forward" in row:
            baseline[(row["device_platform"], int(row["dim"]), int(row["batch"]))] = float(
                row["forward"]["median_s"]
            )
    out = []
    for row in rows:
        if "forward" not in row:
            continue
        key = (row["device_platform"], int(row["dim"]), int(row["batch"]))
        base = baseline.get(key)
        if base:
            out.append(
                {
                    "device_platform": row["device_platform"],
                    "backend": row["backend"],
                    "dim": int(row["dim"]),
                    "batch": int(row["batch"]),
                    "forward_median_s": float(row["forward"]["median_s"]),
                    "same_device_speedup_vs_jax_tensorized": base
                    / float(row["forward"]["median_s"]),
                }
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platforms", nargs="+", default=["cpu", "gpu"])
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["jax_tensorized", "gpu_tensorized"],
    )
    parser.add_argument("--dims", nargs="+", type=int, default=[3, 8, 17])
    parser.add_argument("--batches", nargs="+", type=int, default=[8, 64, 256])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--n-end", type=float, default=0.25)
    parser.add_argument("--rtol", type=float, default=1.0e-7)
    parser.add_argument("--atol", type=float, default=1.0e-9)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    rows = []
    for requested_platform, device in _platform_devices([str(x).lower() for x in args.platforms]):
        for backend in args.backends:
            for dim in args.dims:
                for batch in args.batches:
                    rows.append(
                        _bench(
                            device=device,
                            requested_platform=requested_platform,
                            backend=str(backend),
                            dim=int(dim),
                            batch=int(batch),
                            reps=max(1, int(args.reps)),
                            warmups=max(0, int(args.warmups)),
                            n_end=float(args.n_end),
                            rtol=float(args.rtol),
                            atol=float(args.atol),
                            max_steps=int(args.max_steps),
                        )
                    )
    payload = {
        "benchmark_contract": "rodas5p_batch_linear_solver_profile_v1",
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "env": {"JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "")},
        "rows": rows,
        "speedups": _speedups(rows),
        "promotion_note": (
            "gpu_tensorized is the explicit JAX/XLA GPU path; jax_tensorized "
            "is the portable XLA path. No Rust, FFI, or custom-call backend "
            "is part of this benchmark."
        ),
    }
    if args.json_output:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        for row in rows:
            if "forward" not in row:
                print(row)
            else:
                print(
                    row["device_platform"],
                    row["backend"],
                    "dim",
                    row["dim"],
                    "batch",
                    row["batch"],
                    "median_s",
                    f'{row["forward"]["median_s"]:.6e}',
                )
        print("speedups", _speedups(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
