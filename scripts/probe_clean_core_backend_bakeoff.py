#!/usr/bin/env python
"""BD616 — clean-core backend bake-off (E5 of BD611).

Measures the same physics case across backends so a Rust CPU collision-kernel
lane (PR5a) is gated on evidence, not feel. Three levels, compile separated
from warm steady-state:

  1. kernel   — one deterministic collision-field evaluation per lane
                (numpy nu-e / pair triple-loop vs the JAX ports) at each n_q.
                THIS is what the >=3x same-case migration rule is evaluated on.
  2. rhs      — best-effort single collision-source assembly (the numpy nu-e +
                pair operators together), a proxy for the per-RHS collision cost.
  3. endpoint — full run_full_coupled_typeI at each n_q for the BDF baseline and
                (with --rodas5p) the in-tree Rodas5P adapter lane. Wall + Yp/DH
                cross-check only; NOT part of the >=3x kernel decision.

Decision rule: proceed to a Rust kernel lane iff any target kernel shows
numpy_median / jax_median >= 3.0 at the largest n_q. The Amdahl bound (kernel
wins are capped by the collision fraction of the endpoint wall, ~33% per BD591)
is recorded so a kernel-level win is never oversold as an endpoint win.

Conventions mirror scripts/profile_kernel_backends.py: JAX_PLATFORMS=cpu,
jax_enable_x64, warmups discarded, min/median/max_s, compile probed separately,
json.dumps(indent=2, sort_keys=True). No endpoint claim is made from segment
timings.

Usage:
  env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu \
    venv/bin/python scripts/probe_clean_core_backend_bakeoff.py \
    --n-q 16 24 --repeat 5 --out audit_outputs/bd616_backend_bakeoff.json
"""
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

import numpy as np

# ── collision fraction of the endpoint wall (BD591 attribution) ──
# payload build ~= 33% of the q4 endpoint wall; used only for the Amdahl bound
# on how much a collision-kernel-only speedup can move the endpoint.
_COLLISION_WALL_FRACTION = 0.33


def _sync(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    return value


def _time_call(fn: Callable[[], Any], *, reps: int, warmups: int) -> dict[str, Any]:
    for _ in range(warmups):
        _sync(fn())
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _sync(fn())
        samples.append(time.perf_counter() - t0)
    return {
        "reps": reps,
        "warmups": warmups,
        "min_s": min(samples),
        "median_s": statistics.median(samples),
        "max_s": max(samples),
    }


def _fermi_dirac(q: np.ndarray) -> np.ndarray:
    return 1.0 / (np.exp(np.minimum(q, 500.0)) + 1.0)


def _kernel_rows(n_q: int, *, reps: int, warmups: int, T_MeV: float = 2.0) -> list[dict]:
    """numpy vs JAX collision-field eval for nu-e scattering and pair processes."""
    import jax
    import jax.numpy as jnp
    from rabbit.jax.collisions_jax import laguerre_grid, make_nu_e_kernel, make_pair_kernel
    from rabbit.collisions.nu_e_scattering import NuEScatteringOperator
    from rabbit.collisions.pair_processes import PairProcessOperator

    jax.config.update("jax_enable_x64", True)

    q_nodes, _ = laguerre_grid(n_q)
    rng = np.random.default_rng(12345)
    f_np = np.clip(_fermi_dirac(q_nodes) * (1.0 + 0.05 * rng.standard_normal(n_q)), 0.0, 1.0)
    fbar_np = np.clip(_fermi_dirac(q_nodes) * (1.0 + 0.05 * rng.standard_normal(n_q)), 0.0, 1.0)
    q_jax, f_jax, fbar_jax = jnp.asarray(q_nodes), jnp.asarray(f_np), jnp.asarray(fbar_np)
    T_jax = jnp.asarray(T_MeV)

    rows: list[dict] = []

    # ── nu-e scattering ──
    nue_op = NuEScatteringOperator()
    nue_kernel, _ = make_nu_e_kernel(species="nue", N_q=n_q, N_int=min(n_q, 24), N_quad_electron=32)
    _sync(nue_kernel(f_jax, q_jax, T_jax))  # trigger compile before timing
    t_compile0 = time.perf_counter()
    nue_kernel2, _ = make_nu_e_kernel(species="nue", N_q=n_q, N_int=min(n_q, 24), N_quad_electron=32)
    _sync(nue_kernel2(f_jax, q_jax, T_jax))
    nue_compile_s = time.perf_counter() - t_compile0
    rows.append({
        "kernel": "nu_e_scatter", "backend": "numpy", "n_q": n_q,
        **_time_call(lambda: nue_op._evaluate_vectorized(f_np, q_nodes, T_MeV), reps=reps, warmups=warmups),
    })
    rows.append({
        "kernel": "nu_e_scatter", "backend": "jax", "n_q": n_q, "compile_s": nue_compile_s,
        **_time_call(lambda: nue_kernel(f_jax, q_jax, T_jax), reps=reps, warmups=warmups),
    })

    # ── pair processes ──
    pair_op = PairProcessOperator()
    pair_kernel, _ = make_pair_kernel(species="nue", N_quad=min(n_q, 24))
    _sync(pair_kernel(f_jax, fbar_jax, q_jax, T_jax))
    t_compile1 = time.perf_counter()
    pair_kernel2, _ = make_pair_kernel(species="nue", N_quad=min(n_q, 24))
    _sync(pair_kernel2(f_jax, fbar_jax, q_jax, T_jax))
    pair_compile_s = time.perf_counter() - t_compile1
    rows.append({
        "kernel": "pair_process", "backend": "numpy", "n_q": n_q,
        **_time_call(lambda: pair_op.evaluate(f_np, fbar_np, q_nodes, T_MeV), reps=reps, warmups=warmups),
    })
    rows.append({
        "kernel": "pair_process", "backend": "jax", "n_q": n_q, "compile_s": pair_compile_s,
        **_time_call(lambda: pair_kernel(f_jax, fbar_jax, q_jax, T_jax), reps=reps, warmups=warmups),
    })
    return rows


def _rhs_rows(n_q: int, *, reps: int, warmups: int, T_MeV: float = 2.0) -> list[dict]:
    """Proxy per-RHS collision cost: numpy nu-e + pair source assembled together."""
    from rabbit.jax.collisions_jax import laguerre_grid
    from rabbit.collisions.nu_e_scattering import NuEScatteringOperator
    from rabbit.collisions.pair_processes import PairProcessOperator

    q_nodes, _ = laguerre_grid(n_q)
    rng = np.random.default_rng(777)
    f = np.clip(_fermi_dirac(q_nodes) * (1.0 + 0.05 * rng.standard_normal(n_q)), 0.0, 1.0)
    fbar = np.clip(_fermi_dirac(q_nodes) * (1.0 + 0.05 * rng.standard_normal(n_q)), 0.0, 1.0)
    nue, pair = NuEScatteringOperator(), PairProcessOperator()

    def assemble():
        c = nue._evaluate_vectorized(f, q_nodes, T_MeV)
        c = c + pair.evaluate(f, fbar, q_nodes, T_MeV)
        return c

    return [{
        "lane": "numpy_collision_source", "n_q": n_q,
        **_time_call(assemble, reps=reps, warmups=warmups),
    }]


def _endpoint_rows(n_q: int, *, reps: int, use_rodas5p: bool) -> list[dict]:
    """Full endpoint solve wall for the BDF baseline (and Rodas5P if requested)."""
    from rabbit.config.solver_config import SolverConfig, SolverMethod
    from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI

    def _run(solver):
        cfg = FullCoupledConfig(
            Sigma_H_plus=0.1, N_q=n_q, N_mu=8, n_reactions=12,
            correction_level=0, tier=1, enable_teff=False, solver=solver,
        )
        return run_full_coupled_typeI(cfg)

    lanes = [("BDF", None)]
    if use_rodas5p:
        lanes.append(("RODAS5P", SolverConfig(method=SolverMethod.RODAS5P, rtol=1e-8, atol=1e-10, max_step=0.1)))

    rows: list[dict] = []
    for label, solver in lanes:
        walls, yp, dh = [], None, None
        for _ in range(max(1, reps)):
            t0 = time.perf_counter()
            r = _run(solver)
            walls.append(time.perf_counter() - t0)
            yp, dh = float(r.observables.Yp), float(r.observables.DH)
        rows.append({
            "method": label, "n_q": n_q,
            "wall_time_s_runs": walls, "median_s": statistics.median(walls),
            "min_s": min(walls), "Yp": yp, "DH": dh,
        })
    return rows


def _decide(kernel_rows: list[dict], n_q_max: int) -> dict:
    speedups: dict[str, float] = {}
    jax_runtime: dict[str, float] = {}
    jax_compile: dict[str, float] = {}
    for kname in ("nu_e_scatter", "pair_process"):
        num = next((r for r in kernel_rows if r["kernel"] == kname and r["backend"] == "numpy" and r["n_q"] == n_q_max), None)
        jax_ = next((r for r in kernel_rows if r["kernel"] == kname and r["backend"] == "jax" and r["n_q"] == n_q_max), None)
        if num and jax_ and jax_["median_s"] > 0:
            speedups[kname] = num["median_s"] / jax_["median_s"]
            jax_runtime[kname] = jax_["median_s"]
            jax_compile[kname] = jax_.get("compile_s", 0.0)
    best = max(speedups.values(), default=0.0)
    verdict = "proceed" if best >= 3.0 else "stop"
    amdahl = 1.0 / (1.0 - _COLLISION_WALL_FRACTION * (1.0 - 1.0 / best)) if best > 0 else 1.0
    return {
        "rule": f"same_case_compiled_over_numpy_speedup >= 3.0 at n_q={n_q_max}",
        "measured_speedups_compiled_over_numpy": speedups,
        "best_speedup": best,
        "jax_kernel_runtime_s": jax_runtime,
        "jax_kernel_compile_s": jax_compile,
        "collision_wall_fraction_assumed": _COLLISION_WALL_FRACTION,
        "amdahl_bound_end_to_end": amdahl,
        "verdict": verdict,
        "interpretation": (
            "The measured ratio is JAX(compiled)/numpy(Python triple-loop); it "
            "shows the collision kernel MUST be compiled, not that Rust beats JAX. "
            "JAX already captures this runtime win in production. The Rust lane is "
            "justified only under the meta-goal of DROPPING JAX: Rust (AOT) would "
            "preserve the compiled runtime (jax_kernel_runtime_s) while eliminating "
            "JAX's per-shape compile latency (jax_kernel_compile_s) and the "
            "numpy/JAX twin-maintenance surface. Rust-vs-JAX runtime is expected "
            "~1x, so Rust's value is JAX-removal + twin reduction, NOT raw speed."
        ),
        "amdahl_note": (
            "Even a large kernel speedup is Amdahl-capped by the collision "
            "fraction of the endpoint wall (~33%, BD591): the endpoint win cannot "
            "exceed amdahl_bound_end_to_end regardless of kernel gain."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-q", nargs="+", type=int, default=[16, 24])
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--out", type=str, default="audit_outputs/bd616_backend_bakeoff.json")
    p.add_argument("--skip-solver", action="store_true", help="skip the endpoint-solve level (kernel+rhs only)")
    p.add_argument("--rodas5p", action="store_true", help="add the in-tree Rodas5P adapter endpoint lane")
    args = p.parse_args(argv)

    n_qs = sorted(set(int(n) for n in args.n_q))
    reps, warmups = max(1, int(args.repeat)), max(0, int(args.warmups))

    kernel_rows: list[dict] = []
    rhs_rows: list[dict] = []
    for n_q in n_qs:
        kernel_rows.extend(_kernel_rows(n_q, reps=reps, warmups=warmups))
        rhs_rows.extend(_rhs_rows(n_q, reps=reps, warmups=warmups))

    endpoint_rows: list[dict] = []
    if not args.skip_solver:
        for n_q in n_qs:
            endpoint_rows.extend(_endpoint_rows(n_q, reps=min(reps, 3), use_rodas5p=args.rodas5p))

    import platform
    result = {
        "meta": {
            "argv": sys.argv,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "n_q": n_qs,
            "reps": reps,
            "warmups": warmups,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        },
        "kernel": kernel_rows,
        "rhs": rhs_rows,
        "endpoint": endpoint_rows,
        "decision": _decide(kernel_rows, max(n_qs)),
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
