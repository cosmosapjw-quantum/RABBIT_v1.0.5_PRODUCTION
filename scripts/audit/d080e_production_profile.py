#!/usr/bin/env python3
"""Generate the D-080E same-physics production-order reuse profile.

The two arms are:

1. frozen serial D-079, which recomputes the primal collision path for every
   spectral tangent direction;
2. prepared fixed-state reuse, which keeps the equations, quadrature,
   tolerances, output ordering, and derivative algebra unchanged while caching
   direction-independent kinematics, weak matrices, and modal basis values.

Timings are empirical diagnostics from one CI host.  They are not committed as
portable performance constants and they do not constitute solver admission.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_rhs_jvp import evaluate_c_only_rhs_jvp
from scripts.audit._d080c_tgamma_rhs import evaluate_tgamma_rhs_column
from scripts.audit._d080d_static_jacobian import rhs_block_relative
from scripts.audit._d080e_prepared_jvp import (
    EXPECTED_COMPARATOR_BLOB_SHA,
    FixedStateReusePolicy,
    evaluate_prepared_c_only_rhs_jvp,
    evaluate_prepared_c_only_rhs_jvps,
    prepare_static_rhs_reuse,
)


ORDER60_FIXTURE_SHA256 = (
    "c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380"
)


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    framed = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    return hashlib.sha1(framed).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_memory_bytes() -> dict[str, int | None]:
    result: dict[str, int | None] = {"rss": None, "hwm": None}
    status = Path("/proc/self/status")
    if not status.is_file():
        return result
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            result["rss"] = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            result["hwm"] = int(line.split()[1]) * 1024
    return result


def time_call(function: Callable[[], Any], *, repeats: int = 1) -> tuple[Any, dict[str, Any]]:
    samples: list[float] = []
    result: Any = None
    for _ in range(int(repeats)):
        gc.collect()
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
    return result, {
        "samples_s": samples,
        "minimum_s": float(min(samples)),
        "median_s": float(np.median(samples)),
        "maximum_s": float(max(samples)),
    }


def deterministic_directions(order: int, count: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, int(order), dtype=np.float64)
    candidates = [
        np.stack((np.cos(1.3 * x), np.sin(0.7 * x), x)),
        np.stack((x * x - 0.2, np.cos(0.4 + x), -np.sin(1.1 * x))),
        np.stack((np.exp(-x * x), x - x.mean(), np.cos(2.0 * x))),
        np.stack((np.sin(2.3 * x), np.exp(-0.7 * (x + 0.2) ** 2), x**3)),
        np.stack((np.cos(3.1 * x), x * np.exp(-x * x), np.sin(0.9 - x))),
        np.stack((1.0 + 0.1 * x, -0.4 + x * x, np.cos(np.pi * x))),
        np.stack((np.sin(np.pi * x / 2.0), np.cos(1.7 * x), x**2 - x.mean())),
        np.stack((np.exp(-2.0 * (x - 0.3) ** 2), np.sin(1.4 * x), -x)),
    ]
    if count > len(candidates):
        raise ValueError("requested more deterministic directions than defined")
    values = np.asarray(candidates[:count], dtype=np.float64)
    norms = np.linalg.norm(values.reshape(count, -1), axis=1)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise RuntimeError("invalid deterministic direction")
    return values / norms[:, None, None]


def thermal_case() -> tuple[
    ind.IndependentNoQkeGrid,
    ind.IndependentCollisionConfig,
    np.ndarray,
    float,
    float,
]:
    grid = ind.build_independent_grid(8, 8.0)
    logits = np.stack(
        [
            -grid.nodes + 0.04 * np.exp(-grid.nodes / 3.0),
            -grid.nodes - 0.02 * np.exp(-grid.nodes / 4.0),
            -grid.nodes - 0.02 * np.exp(-grid.nodes / 4.0),
        ]
    )
    config = ind.IndependentCollisionConfig(
        incoming_polar_order=2,
        final_polar_order=2,
        final_azimuth_order=4,
        electron_radial_order=8,
    )
    return grid, config, ind.pair_logits_to_cloglog(logits), 2.0, 2.05


def serial_values(
    *,
    grid: ind.IndependentNoQkeGrid,
    config: ind.IndependentCollisionConfig,
    pair_cloglog: np.ndarray,
    tcm: float,
    tg: float,
    directions: np.ndarray,
) -> np.ndarray:
    values = [
        evaluate_c_only_rhs_jvp(
            grid=grid,
            pair_cloglog=pair_cloglog,
            direction_cloglog=direction,
            temperature_cm_mev=tcm,
            temperature_gamma_mev=tg,
            config=config,
        ).jvp
        for direction in directions
    ]
    return np.asarray(np.stack(values), dtype=np.float64)


def max_block_residual(
    left: np.ndarray,
    right: np.ndarray,
    order: int,
) -> float:
    if left.shape != right.shape:
        raise ValueError("batched residual operands have different shapes")
    return float(
        max(
            rhs_block_relative(a, b, order)
            for a, b in zip(left, right)
        )
    )


def benchmark_order8() -> dict[str, Any]:
    grid, config, c, tcm, tg = thermal_case()
    all_directions = deterministic_directions(grid.order, 8)
    counts = (1, 2, 4, 8)
    rows: list[dict[str, Any]] = []

    for count in counts:
        directions = all_directions[:count]
        serial, serial_time = time_call(
            lambda: serial_values(
                grid=grid,
                config=config,
                pair_cloglog=c,
                tcm=tcm,
                tg=tg,
                directions=directions,
            ),
            repeats=2,
        )
        prepared, preparation_time = time_call(
            lambda: prepare_static_rhs_reuse(
                grid=grid,
                pair_cloglog=c,
                temperature_cm_mev=tcm,
                temperature_gamma_mev=tg,
                config=config,
            ),
            repeats=1,
        )
        candidate, apply_time = time_call(
            lambda: evaluate_prepared_c_only_rhs_jvps(prepared, directions),
            repeats=2,
        )
        residual = max_block_residual(candidate, serial, grid.order)
        prepared_total = preparation_time["median_s"] + apply_time["median_s"]
        rows.append(
            {
                "direction_count": count,
                "serial": serial_time,
                "prepared_preparation": preparation_time,
                "prepared_apply": apply_time,
                "prepared_total_s": prepared_total,
                "same_physics_residual": residual,
                "total_speedup": serial_time["median_s"] / prepared_total,
                "marginal_speedup": (
                    serial_time["median_s"] / count
                ) / (apply_time["median_s"] / count),
                "cache": prepared.cache.snapshot(),
            }
        )

    policies = {
        "full reuse": FixedStateReusePolicy(),
        "no modal cache": FixedStateReusePolicy(cache_modal_basis=False),
        "no matrix cache": FixedStateReusePolicy(cache_matrices=False),
        "no fixed-state caches": FixedStateReusePolicy(
            cache_kinematics=False,
            cache_matrices=False,
            cache_modal_basis=False,
        ),
    }
    ablations: dict[str, Any] = {}
    directions = all_directions[:2]
    reference = serial_values(
        grid=grid,
        config=config,
        pair_cloglog=c,
        tcm=tcm,
        tg=tg,
        directions=directions,
    )
    for name, policy in policies.items():
        prepared, preparation_time = time_call(
            lambda policy=policy: prepare_static_rhs_reuse(
                grid=grid,
                pair_cloglog=c,
                temperature_cm_mev=tcm,
                temperature_gamma_mev=tg,
                config=config,
                policy=policy,
            )
        )
        value, apply_time = time_call(
            lambda prepared=prepared: evaluate_prepared_c_only_rhs_jvps(
                prepared, directions
            )
        )
        ablations[name] = {
            "preparation": preparation_time,
            "apply": apply_time,
            "total_s": preparation_time["median_s"] + apply_time["median_s"],
            "same_physics_residual": max_block_residual(
                value, reference, grid.order
            ),
            "cache": prepared.cache.snapshot(),
        }

    return {
        "order": grid.order,
        "state_size": 3 * grid.order + 2,
        "spectral_size": 3 * grid.order,
        "y_max": grid.y_max,
        "temperature_cm_mev": tcm,
        "temperature_gamma_mev": tg,
        "direction_scaling": rows,
        "cache_policy_ablations": ablations,
    }


def load_retained_case(path: Path) -> tuple[
    ind.IndependentNoQkeGrid,
    ind.IndependentCollisionConfig,
    np.ndarray,
    float,
    float,
    float,
]:
    if sha256(path) != ORDER60_FIXTURE_SHA256:
        raise RuntimeError("retained-state SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        state = np.asarray(archive["y"], dtype=np.float64)
    order, y_max = 60, 30.0
    if state.shape != (3 * order + 2,) or not np.all(np.isfinite(state)):
        raise RuntimeError("retained state has an invalid layout")
    grid = ind.build_independent_grid(order, y_max)
    config = ind.IndependentCollisionConfig(
        incoming_polar_order=4,
        final_polar_order=4,
        final_azimuth_order=4,
        electron_radial_order=24,
    )
    c = state[: 3 * order].reshape(3, order)
    tg = float(state[3 * order])
    elapsed = float(state[-1])
    expansion = 0.16286930247517223
    tcm = 10.0 * np.exp(-expansion)
    return grid, config, c, tcm, tg, elapsed


def benchmark_order60(path: Path) -> dict[str, Any]:
    grid, config, c, tcm, tg, elapsed = load_retained_case(path)
    directions = deterministic_directions(grid.order, 2)
    memory_before = process_memory_bytes()

    serial, serial_time = time_call(
        lambda: serial_values(
            grid=grid,
            config=config,
            pair_cloglog=c,
            tcm=tcm,
            tg=tg,
            directions=directions,
        )
    )
    prepared, preparation_time = time_call(
        lambda: prepare_static_rhs_reuse(
            grid=grid,
            pair_cloglog=c,
            temperature_cm_mev=tcm,
            temperature_gamma_mev=tg,
            config=config,
            policy=FixedStateReusePolicy(max_modal_basis_bytes=768 * 1024**2),
        )
    )
    memory_after_preparation = process_memory_bytes()
    candidate, apply_time = time_call(
        lambda: evaluate_prepared_c_only_rhs_jvps(prepared, directions)
    )
    first = evaluate_prepared_c_only_rhs_jvp(prepared, directions[0])
    memory_after_apply = process_memory_bytes()

    before_tgamma = prepared.cache.snapshot()
    with prepared.cache.patch():
        tgamma, tgamma_time = time_call(
            lambda: evaluate_tgamma_rhs_column(
                grid=prepared.grid,
                pair_cloglog=prepared.pair_cloglog,
                temperature_cm_mev=tcm,
                temperature_gamma_mev=tg,
                config=config,
            )
        )
    after_tgamma = prepared.cache.snapshot()

    residual = max_block_residual(candidate, serial, grid.order)
    spectral_size = 3 * grid.order
    serial_per_direction = serial_time["median_s"] / directions.shape[0]
    prepared_per_direction = apply_time["median_s"] / directions.shape[0]
    projected_serial = spectral_size * serial_per_direction + tgamma_time["median_s"]
    projected_prepared = (
        preparation_time["median_s"]
        + spectral_size * prepared_per_direction
        + tgamma_time["median_s"]
    )
    projected_speedup = projected_serial / projected_prepared
    cache_bytes = int(prepared.cache.snapshot()["estimated_cache_bytes"])
    explicit_matrix_bytes = (3 * grid.order + 2) ** 2 * 8

    if (
        residual < 5.0e-11
        and projected_prepared <= 900.0
        and cache_bytes <= 2 * 1024**3
        and projected_speedup >= 1.5
    ):
        route = "EXPLICIT_FULL_BUILD_MEASUREMENT_ADMISSIBLE"
    elif residual < 5.0e-11 and cache_bytes <= 4 * 1024**3:
        route = "REUSE_VALID_TRUE_BATCHING_OR_MATRIX_FREE_DECISION_REQUIRED"
    else:
        route = "REUSE_ROUTE_NOT_ADMITTED"

    return {
        "order": grid.order,
        "state_size": 3 * grid.order + 2,
        "spectral_size": spectral_size,
        "y_max": grid.y_max,
        "temperature_cm_mev": tcm,
        "temperature_gamma_mev": tg,
        "elapsed_time_mev_inverse": elapsed,
        "fixture_sha256": ORDER60_FIXTURE_SHA256,
        "direction_count": int(directions.shape[0]),
        "serial": serial_time,
        "prepared_preparation": preparation_time,
        "prepared_apply": apply_time,
        "tgamma_column": tgamma_time,
        "same_physics_residual": residual,
        "first_law_tangent_residual": first.first_law_tangent_residual,
        "serial_per_direction_s": serial_per_direction,
        "prepared_marginal_per_direction_s": prepared_per_direction,
        "marginal_speedup": serial_per_direction / prepared_per_direction,
        "projected_serial_full_matrix_s": projected_serial,
        "projected_prepared_full_matrix_s": projected_prepared,
        "projected_full_matrix_speedup": projected_speedup,
        "projection_scope": (
            "linear extrapolation from two retained-state spectral directions; "
            "not an executed 180-column matrix build"
        ),
        "route_decision": route,
        "cache": prepared.cache.snapshot(),
        "cache_delta_during_tgamma": {
            key: int(after_tgamma[key] - before_tgamma[key])
            for key in after_tgamma
            if isinstance(after_tgamma[key], int)
            and isinstance(before_tgamma[key], int)
        },
        "explicit_matrix_bytes": explicit_matrix_bytes,
        "memory_before": memory_before,
        "memory_after_preparation": memory_after_preparation,
        "memory_after_apply": memory_after_apply,
        "tgamma_base_reconstruction_residual": tgamma.base_reconstruction_residual,
    }


def line_plot(path: Path, x: list[int], series: dict[str, list[float]], title: str, ylabel: str) -> None:
    figure = plt.figure(figsize=(6.6, 4.4), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    markers = ("o", "s", "^", "D")
    for marker, (label, values) in zip(markers, series.items()):
        axis.plot(x, values, marker=marker, label=label)
    axis.set_xlabel("number of spectral directions")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def bar_plot(path: Path, values: dict[str, float], title: str, ylabel: str, *, log: bool = False) -> None:
    figure = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    labels = list(values)
    data = list(values.values())
    axis.bar(labels, data)
    if log:
        axis.set_yscale("log")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(True, axis="y", which="both", alpha=0.3)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def generate_plots(output: Path, receipt: dict[str, Any]) -> None:
    rows = receipt["order8"]["direction_scaling"]
    counts = [int(row["direction_count"]) for row in rows]
    line_plot(
        output / "order8_total_wall_time.png",
        counts,
        {
            "frozen serial": [row["serial"]["median_s"] for row in rows],
            "prepared total": [row["prepared_total_s"] for row in rows],
            "prepared marginal only": [row["prepared_apply"]["median_s"] for row in rows],
        },
        "Order-8 same-physics construction cost",
        "wall time [s]",
    )
    line_plot(
        output / "order8_speedup.png",
        counts,
        {
            "total speedup": [row["total_speedup"] for row in rows],
            "marginal speedup": [row["marginal_speedup"] for row in rows],
        },
        "Order-8 fixed-state reuse speedup",
        "serial/prepared ratio",
    )
    ablations = receipt["order8"]["cache_policy_ablations"]
    bar_plot(
        output / "cache_policy_ablation.png",
        {name: item["total_s"] for name, item in ablations.items()},
        "Two-direction cache-policy ablation",
        "prepare + apply wall time [s]",
    )
    production = receipt["order60_retained"]
    bar_plot(
        output / "order60_projected_full_matrix.png",
        {
            "serial projection": production["projected_serial_full_matrix_s"],
            "prepared projection": production["projected_prepared_full_matrix_s"],
        },
        "Retained order-60 projected 180-column construction",
        "projected wall time [s]",
        log=True,
    )
    cache = production["cache"]
    bar_plot(
        output / "order60_cache_activity.png",
        {
            "kinematic hits": cache["kinematic_hits"],
            "kinematic misses": cache["kinematic_misses"],
            "matrix hits": cache["matrix_hits"],
            "matrix misses": cache["matrix_misses"],
            "modal hits": cache["modal_basis_hits"],
            "modal misses": cache["modal_basis_misses"],
        },
        "Retained order-60 cache activity",
        "request count",
        log=True,
    )


def write_summary(output: Path, receipt: dict[str, Any]) -> None:
    order8 = receipt["order8"]["direction_scaling"][-1]
    production = receipt["order60_retained"]
    text = f"""# D-080E production-order Jacobian profile

## Scope

This is a same-physics construction profile.  It does not call an ODE solver,
complete a trajectory, change a tolerance, or move the F10 gate.

## Order-8 executable matrix precursor

- eight-direction serial wall time: `{order8['serial']['median_s']:.9g} s`
- eight-direction prepared total: `{order8['prepared_total_s']:.9g} s`
- total speedup: `{order8['total_speedup']:.6g}`
- marginal speedup: `{order8['marginal_speedup']:.6g}`
- maximum same-physics residual: `{order8['same_physics_residual']:.6e}`

## Retained order-60 profile

- serial time per spectral direction: `{production['serial_per_direction_s']:.9g} s`
- prepared marginal time per direction: `{production['prepared_marginal_per_direction_s']:.9g} s`
- measured marginal speedup: `{production['marginal_speedup']:.6g}`
- projected serial 180-column time: `{production['projected_serial_full_matrix_s']:.9g} s`
- projected prepared 180-column time: `{production['projected_prepared_full_matrix_s']:.9g} s`
- projected speedup: `{production['projected_full_matrix_speedup']:.6g}`
- estimated retained cache bytes: `{production['cache']['estimated_cache_bytes']}`
- explicit 182x182 matrix bytes: `{production['explicit_matrix_bytes']}`
- same-physics residual: `{production['same_physics_residual']:.6e}`
- first-law tangent residual: `{production['first_law_tangent_residual']:.6e}`
- route decision: `{production['route_decision']}`

The 180-column values are linear projections from two measured directions, not
an executed production matrix build.  The next DAG node must follow the route
decision and obtain direct evidence before any solver callback is admitted.
"""
    (output / "probe_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--retained-state", required=True)
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    retained = Path(args.retained_state)

    comparator_sha = git_blob_sha(Path(ind.__file__))
    if comparator_sha != EXPECTED_COMPARATOR_BLOB_SHA:
        raise RuntimeError("frozen comparator Git-blob identity changed")

    receipt: dict[str, Any] = {
        "schema": "rabbit.d080e.production_jacobian_profile.v1",
        "classification": "FIXED_STATE_REUSE_TWO_ARM_PROFILE",
        "claim_ceiling": (
            "same-physics fixed-state reuse correctness and single-host construction-cost "
            "profile; no solver, trajectory, endpoint, N_eff, speedup portability, gate, "
            "release, or publication claim"
        ),
        "tested_head": os.environ.get("GITHUB_SHA"),
        "comparator_blob_sha": comparator_sha,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "fairness_contract": {
            "same_equations": True,
            "same_collision_catalogue": True,
            "same_quadrature": True,
            "same_matrix_roundoff_policy": True,
            "same_state_and_directions": True,
            "same_output_ordering": True,
            "tolerances_changed": False,
            "approximation_added": False,
        },
        "order8": benchmark_order8(),
        "order60_retained": benchmark_order60(retained),
    }

    maximum_residual = max(
        [
            row["same_physics_residual"]
            for row in receipt["order8"]["direction_scaling"]
        ]
        + [
            item["same_physics_residual"]
            for item in receipt["order8"]["cache_policy_ablations"].values()
        ]
        + [receipt["order60_retained"]["same_physics_residual"]]
    )
    receipt["maximum_same_physics_residual"] = float(maximum_residual)

    generate_plots(output, receipt)
    (output / "research_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(output, receipt)

    files = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    manifest = "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    (output / "SHA256SUMS").write_text(manifest, encoding="utf-8")


if __name__ == "__main__":
    main()
