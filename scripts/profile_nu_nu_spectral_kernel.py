#!/usr/bin/env python
"""Profile legacy vs tensorized diagonal nu-nu spectral kernels."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from rabbit.jax.collisions_jax import laguerre_grid
from rabbit.jax.nu_nu_scattering_jax import (
    nu_nu_diagonal_collision_jax,
    nu_nu_diagonal_collision_tensorized_jax,
)


jax.config.update("jax_enable_x64", True)


def _sync(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    return value


def _time_call(fn: Callable[[], Any], *, reps: int, warmups: int) -> dict[str, Any]:
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
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "last_shape": list(getattr(last, "shape", ())),
    }


def _row(N_q: int, *, reps: int, warmups: int) -> dict[str, Any]:
    q_nodes, q_weights = laguerre_grid(int(N_q))
    q = jnp.asarray(q_nodes, dtype=jnp.float64)
    w = jnp.asarray(q_weights, dtype=jnp.float64)
    f_eq = 1.0 / (jnp.exp(jnp.minimum(q, 500.0)) + 1.0)
    f_alpha = jnp.clip(f_eq + 0.012 * f_eq * (1.0 - f_eq), 0.0, 1.0)
    f_beta = jnp.clip(f_eq - 0.009 * f_eq * (1.0 - f_eq) * q / jnp.mean(q), 0.0, 1.0)
    T = jnp.asarray(2.9, dtype=jnp.float64)

    legacy = jax.jit(
        lambda a, b: nu_nu_diagonal_collision_jax(
            a,
            b,
            q,
            T,
            y3_nodes=q,
            y3_weights=w,
            y2_nodes=q,
            y2_weights=w,
        )
    )
    tensorized = jax.jit(
        lambda a, b: nu_nu_diagonal_collision_tensorized_jax(a, b, q, T)
    )
    legacy_out = _sync(legacy(f_alpha, f_beta))
    tensor_out = _sync(tensorized(f_alpha, f_beta))
    max_abs = float(jnp.max(jnp.abs(legacy_out - tensor_out)))
    denom = jnp.maximum(jnp.max(jnp.abs(legacy_out)), 1.0e-300)
    max_rel = float(max_abs / denom)
    legacy_time = _time_call(lambda: legacy(f_alpha, f_beta), reps=reps, warmups=warmups)
    tensor_time = _time_call(
        lambda: tensorized(f_alpha, f_beta),
        reps=reps,
        warmups=warmups,
    )
    return {
        "N_q": int(N_q),
        "legacy": legacy_time,
        "tensorized": tensor_time,
        "speedup_median": legacy_time["median_s"] / max(tensor_time["median_s"], 1.0e-30),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N-q", nargs="+", type=int, default=[6, 12, 20, 40])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    rows = [
        _row(int(N_q), reps=max(1, int(args.reps)), warmups=max(0, int(args.warmups)))
        for N_q in args.N_q
    ]
    payload = {
        "benchmark_contract": "legacy_vs_tensorized_nu_nu_spectral_kernel_v1",
        "platform": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "rows": rows,
    }
    if args.json_output:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print("Nu-nu spectral kernel profile")
        print("=============================")
        print(f"platform={payload['platform']} devices={payload['devices']}")
        for row in rows:
            print(
                f"- N_q={row['N_q']} speedup={row['speedup_median']:.2f}x "
                f"legacy={row['legacy']['median_s']:.6e}s "
                f"tensorized={row['tensorized']['median_s']:.6e}s "
                f"max_rel={row['max_rel_diff']:.3e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
