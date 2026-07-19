#!/usr/bin/env python
"""Profile JAX/XLA collision-rate helper kernels."""

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

from rabbit.jax.collision_rates_jax import (
    collision_rate_kernel_backend_report,
    total_rate_jax,
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
        "reps": reps,
        "warmups": warmups,
        "min_s": min(samples),
        "median_s": statistics.median(samples),
        "max_s": max(samples),
        "last_shape": tuple(getattr(last, "shape", ())),
    }


def _backend_row(
    *,
    backend: str,
    species: str,
    size: int,
    reps: int,
    warmups: int,
    include_grad: bool,
    include_jit_probe: bool,
) -> dict[str, Any]:
    t = jnp.linspace(0.05, 10.0, int(size), dtype=jnp.float64)
    row: dict[str, Any] = {
        "backend": backend,
        "species": species,
        "size": int(size),
        "metadata": collision_rate_kernel_backend_report(backend),
    }

    def forward():
        return total_rate_jax(t, species, kernel_backend=backend)

    try:
        row["forward"] = _time_call(forward, reps=reps, warmups=warmups)
    except Exception as exc:
        row["forward_error"] = f"{type(exc).__name__}: {exc}"
        return row

    if include_grad:
        def grad_call():
            return jax.grad(
                lambda x: jnp.sum(total_rate_jax(x, species, kernel_backend=backend))
            )(t)

        try:
            row["grad"] = _time_call(grad_call, reps=reps, warmups=warmups)
        except Exception as exc:
            row["grad_error"] = f"{type(exc).__name__}: {exc}"

    if include_jit_probe:
        try:
            compiled = jax.jit(lambda x: total_rate_jax(x, species, kernel_backend=backend))
            _sync(compiled(t))
        except Exception as exc:
            row["jit_probe"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            row["jit_probe"] = {"ok": True}

    return row


def _format_text(rows: list[dict[str, Any]]) -> str:
    lines = [
        "Kernel backend profile",
        "======================",
        "",
        "Times are helper-level JAX/XLA timings.",
        "",
    ]
    for row in rows:
        meta = row["metadata"]
        lines.append(
            f"- backend={row['backend']} species={row['species']} size={row['size']} "
            f"effective={meta['kernel_backend_effective']} "
            f"available={meta['kernel_backend_available']} "
            f"jit_safe={meta['kernel_backend_jit_safe']} "
            f"grad_safe={meta['kernel_backend_grad_safe']} "
            f"host_mediated={meta['kernel_backend_host_mediated']}"
        )
        if "forward" in row:
            fwd = row["forward"]
            lines.append(
                f"  forward median={fwd['median_s']:.6e}s min={fwd['min_s']:.6e}s"
            )
        if "forward_error" in row:
            lines.append(f"  forward_error={row['forward_error']}")
        if "grad" in row:
            grad = row["grad"]
            lines.append(
                f"  grad median={grad['median_s']:.6e}s min={grad['min_s']:.6e}s"
            )
        if "grad_error" in row:
            lines.append(f"  grad_error={row['grad_error']}")
        if "jit_probe" in row:
            lines.append(f"  jit_probe={row['jit_probe']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["jax"],
        help="Kernel backends to profile. Only 'jax' and 'auto' are supported.",
    )
    parser.add_argument("--species", choices=["nue", "nux"], default="nue")
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 256, 4096])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--no-grad", action="store_true")
    parser.add_argument("--no-jit-probe", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    rows = []
    for backend in args.backends:
        for size in args.sizes:
            rows.append(
                _backend_row(
                    backend=backend,
                    species=args.species,
                    size=size,
                    reps=max(1, int(args.reps)),
                    warmups=max(0, int(args.warmups)),
                    include_grad=not args.no_grad,
                    include_jit_probe=not args.no_jit_probe,
                )
            )

    if args.json_output:
        print(json.dumps(rows, indent=2, sort_keys=True, default=str))
    else:
        print(_format_text(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
