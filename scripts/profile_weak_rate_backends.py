#!/usr/bin/env python3
"""Profile live weak-rate backends on CPU/GPU batch workloads.

The benchmark is intentionally scoped to the production live-monopole weak
rate contract.  Today it compares the legacy level-specialized JAX path with
the fused JAX path.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np


def _device_for(label: str):
    import jax

    label = str(label).lower()
    if label == "cpu":
        devices = jax.devices("cpu")
        return devices[0] if devices else None
    if label in ("gpu", "rocm"):
        devices = [d for d in jax.devices() if str(getattr(d, "platform", "")).lower() == "gpu"]
        return devices[0] if devices else None
    raise ValueError(f"unknown platform label {label!r}")


def _make_inputs(batch: int, n_q: int):
    import jax.numpy as jnp

    q_np, _ = np.polynomial.laguerre.laggauss(int(n_q))
    q = jnp.asarray(q_np, dtype=jnp.float64)
    idx = jnp.arange(int(batch), dtype=jnp.float64)
    f0 = 1.0 / (jnp.exp(q) + 1.0)
    amp_nue = 0.02 * (idx + 1.0) / max(float(batch), 1.0)
    amp_nuebar = -0.015 * (idx + 1.0) / max(float(batch), 1.0)
    shape = jnp.exp(-q / 2.5)
    f_nue = jnp.clip(f0[None, :] * (1.0 + amp_nue[:, None] * shape[None, :]), 0.0, 1.0)
    f_nuebar = jnp.clip(f0[None, :] * (1.0 + amp_nuebar[:, None] * shape[None, :]), 0.0, 1.0)
    return q, f_nue, f_nuebar


def _build_solver(backend: str, correction_level: int):
    import jax
    import jax.numpy as jnp
    from rabbit.jax.weak_live_jax import compute_live_rates_from_monopoles_level_specialized_jax
    from rabbit.jax.weak_live_fused import compute_live_rates_from_monopoles_fused_jax

    backend = str(backend)
    cl = int(correction_level)

    def one_unfused(f_nue, f_nuebar, q):
        return compute_live_rates_from_monopoles_level_specialized_jax(
            jnp.asarray(1.0),
            jnp.asarray(0.97),
            jnp.asarray(878.4),
            q,
            f_nue,
            f_nuebar,
            correction_level=cl,
        )

    def one_fused(f_nue, f_nuebar, q):
        return compute_live_rates_from_monopoles_fused_jax(
            jnp.asarray(1.0),
            jnp.asarray(0.97),
            jnp.asarray(878.4),
            q,
            f_nue,
            f_nuebar,
            correction_level=cl,
        )

    one = one_fused if backend == "jax_fused" else one_unfused

    @jax.jit
    def run(q, f_nue, f_nuebar):
        lnp, lpn, i0 = jax.vmap(lambda a, b: one(a, b, q))(f_nue, f_nuebar)
        return jnp.stack([lnp, lpn, i0 * jnp.ones_like(lnp)], axis=-1)

    return run


def _time_call(fn, args, *, warmups: int, reps: int) -> tuple[float, Any]:
    out = None
    for _ in range(int(warmups)):
        out = fn(*args)
        out.block_until_ready()
    times = []
    for _ in range(int(reps)):
        t0 = time.perf_counter()
        out = fn(*args)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def _run_row(platform: str, backend: str, batch: int, n_q: int, correction_level: int, args):
    import jax
    import jax.numpy as jnp

    device = _device_for(platform)
    row: dict[str, Any] = {
        "platform": platform,
        "backend": backend,
        "batch": int(batch),
        "N_q": int(n_q),
        "correction_level": int(correction_level),
    }
    if device is None:
        row["skipped"] = True
        row["skip_reason"] = f"{platform}_device_unavailable"
        return row

    with jax.default_device(device):
        q, f_nue, f_nuebar = _make_inputs(int(batch), int(n_q))
        q = jax.device_put(q, device)
        f_nue = jax.device_put(f_nue, device)
        f_nuebar = jax.device_put(f_nuebar, device)
        fn = _build_solver(backend, int(correction_level))
        elapsed, out = _time_call(
            fn,
            (q, f_nue, f_nuebar),
            warmups=int(args.warmups),
            reps=int(args.reps),
        )
    row["skipped"] = False
    row["seconds_median"] = elapsed
    row["rates_per_second"] = float(int(batch) / elapsed) if elapsed > 0.0 else float("inf")
    row["device_platform"] = str(getattr(device, "platform", "unknown"))
    row["device_kind"] = str(getattr(device, "device_kind", "unknown"))
    row["finite"] = bool(jnp.all(jnp.isfinite(out)))
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platforms", nargs="+", default=["cpu"])
    parser.add_argument("--backends", nargs="+", default=["jax_unfused", "jax_fused"])
    parser.add_argument("--batches", nargs="+", type=int, default=[64, 512, 2048])
    parser.add_argument("--N-q", type=int, default=20)
    parser.add_argument("--correction-levels", nargs="+", type=int, default=[0, 3])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)

    rows = []
    for platform in args.platforms:
        for correction_level in args.correction_levels:
            for batch in args.batches:
                reference = None
                for backend in args.backends:
                    row = _run_row(platform, backend, batch, args.N_q, correction_level, args)
                    rows.append(row)
                    if row.get("skipped"):
                        continue
                    if backend == "jax_unfused":
                        reference = dict(row)
                    elif reference is not None and row["seconds_median"] > 0:
                        row["speedup_vs_jax_unfused"] = (
                            reference["seconds_median"] / row["seconds_median"]
                        )

    payload = {
        "rows": rows,
        "external_call_backend": False,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for row in rows:
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
