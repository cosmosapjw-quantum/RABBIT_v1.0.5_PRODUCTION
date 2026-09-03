#!/usr/bin/env python3
"""Locate cross-host binary64 divergence in retained packed-RHS fixtures."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import scipy

MASK64 = (1 << 64) - 1
SIGN64 = 1 << 63

ARRAY_LAYERS: list[tuple[str, tuple[str, ...]]] = [
    (
        "fixed_input",
        ("packed_state", "pair_cloglog"),
    ),
    (
        "quadrature_grid",
        ("grid_nodes", "grid_weights"),
    ),
    (
        "chart",
        ("occupation", "cloglog_chain"),
    ),
    (
        "collision",
        (
            "self_native",
            "electron_native",
            "total_native",
            "self_modal",
            "electron_modal",
            "total_modal",
            "pair_rate",
        ),
    ),
    (
        "packed_rhs",
        (
            "spectral_rhs",
            "packed_rhs_trajectory_core",
            "packed_rhs_reconstructed",
        ),
    ),
]

SCALAR_LAYERS: list[tuple[str, tuple[str, ...]]] = [
    (
        "background_scalars",
        (
            "configuration.expansion_n_bits",
            "configuration.temperature_start_mev_bits",
            "configuration.temperature_cm_mev_bits",
            "configuration.temperature_gamma_mev_bits",
            "configuration.elapsed_mev_inverse_bits",
        ),
    ),
    ("thermodynamics", ("thermodynamics",)),
    ("electromagnetic_eos", ("electromagnetic_eos",)),
    ("action_diagnostics", ("action_diagnostics",)),
    ("moments", ("moments",)),
    ("rhs_scalars", ("scalars",)),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_array(value: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    shape = tuple(int(item) for item in value["shape"])
    bits = np.asarray([int(str(item), 16) for item in value["bits"]], dtype=np.uint64)
    floats = np.ascontiguousarray(bits.view(np.float64).reshape(shape))
    return floats, bits.reshape(shape)


def ordered_integer(bits: int) -> int:
    return ((~bits) & MASK64) if bits & SIGN64 else bits | SIGN64


def array_difference(left_value: dict[str, Any], right_value: dict[str, Any]) -> dict[str, Any]:
    left, left_bits = decode_array(left_value)
    right, right_bits = decode_array(right_value)
    if left.shape != right.shape:
        return {
            "shape_equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "changed": True,
        }

    changed_mask = left_bits != right_bits
    changed_flat = np.flatnonzero(changed_mask.ravel())
    if changed_flat.size == 0:
        return {
            "shape_equal": True,
            "changed": False,
            "changed_count": 0,
            "total_count": int(left.size),
            "max_abs": 0.0,
            "max_relative": 0.0,
            "max_ulp": 0,
        }

    left_flat = left.ravel()
    right_flat = right.ravel()
    absolute = np.abs(left_flat - right_flat)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scale = np.maximum(
            np.maximum(np.abs(left_flat), np.abs(right_flat)),
            np.finfo(np.float64).tiny,
        )
        relative = absolute / scale

    max_abs_flat = int(np.nanargmax(absolute))
    max_relative_flat = int(np.nanargmax(relative))
    max_ulp = -1
    max_ulp_flat = int(changed_flat[0])
    for flat in changed_flat.tolist():
        distance = abs(
            ordered_integer(int(left_bits.ravel()[flat]))
            - ordered_integer(int(right_bits.ravel()[flat]))
        )
        if distance > max_ulp:
            max_ulp = distance
            max_ulp_flat = int(flat)

    return {
        "shape_equal": True,
        "changed": True,
        "changed_count": int(changed_flat.size),
        "total_count": int(left.size),
        "max_abs": float(absolute[max_abs_flat]),
        "max_abs_flat_index": max_abs_flat,
        "max_abs_left": float(left_flat[max_abs_flat]),
        "max_abs_right": float(right_flat[max_abs_flat]),
        "max_relative": float(relative[max_relative_flat]),
        "max_relative_flat_index": max_relative_flat,
        "max_ulp": int(max_ulp),
        "max_ulp_flat_index": max_ulp_flat,
        "max_ulp_left": float(left_flat[max_ulp_flat]),
        "max_ulp_right": float(right_flat[max_ulp_flat]),
    }


def is_hex_float(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 16:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def decode_hex_float(value: str) -> tuple[float, int]:
    bits = int(value, 16)
    floating = np.asarray([bits], dtype=np.uint64).view(np.float64)[0]
    return float(floating), bits


def flatten_hex_scalars(value: Any, prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if is_hex_float(item):
                result[path] = item
            elif isinstance(item, dict) and "bits" not in item:
                result.update(flatten_hex_scalars(item, path))
    return result


def select_prefixes(flat: dict[str, str], prefixes: tuple[str, ...]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for key, value in flat.items():
        if any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes):
            selected[key] = value
    return selected


def scalar_difference(left: dict[str, str], right: dict[str, str]) -> dict[str, Any]:
    keys = sorted(set(left) | set(right))
    changed: dict[str, Any] = {}
    for key in keys:
        if key not in left or key not in right:
            changed[key] = {"missing": True}
            continue
        left_float, left_bits = decode_hex_float(left[key])
        right_float, right_bits = decode_hex_float(right[key])
        if left_bits == right_bits:
            continue
        absolute = abs(left_float - right_float)
        scale = max(abs(left_float), abs(right_float), np.finfo(np.float64).tiny)
        changed[key] = {
            "left": left_float,
            "right": right_float,
            "absolute": absolute,
            "relative": absolute / scale,
            "ulp": abs(ordered_integer(left_bits) - ordered_integer(right_bits)),
        }
    return {
        "changed": bool(changed),
        "changed_count": len(changed),
        "values": changed,
    }


def compare_payloads(left_path: Path, right_path: Path, left_label: str, right_label: str) -> dict[str, Any]:
    left = load(left_path)
    right = load(right_path)
    arrays: dict[str, Any] = {}
    layer_changes: dict[str, bool] = {}
    for layer, names in ARRAY_LAYERS:
        layer_changed = False
        for name in names:
            difference = array_difference(left["arrays"][name], right["arrays"][name])
            arrays[name] = difference
            layer_changed |= bool(difference["changed"])
        layer_changes[layer] = layer_changed

    left_scalars = flatten_hex_scalars(left)
    right_scalars = flatten_hex_scalars(right)
    scalar_groups: dict[str, Any] = {}
    for layer, prefixes in SCALAR_LAYERS:
        difference = scalar_difference(
            select_prefixes(left_scalars, prefixes),
            select_prefixes(right_scalars, prefixes),
        )
        scalar_groups[layer] = difference
        layer_changes[layer] = bool(difference["changed"])

    ordered_layers = [layer for layer, _ in ARRAY_LAYERS] + [layer for layer, _ in SCALAR_LAYERS]
    first_divergent = next((layer for layer in ordered_layers if layer_changes.get(layer)), None)

    authority_equal = left.get("authorities") == right.get("authorities")
    environment_equal = left.get("environment") == right.get("environment")
    plain_configuration_equal = {
        key: left["configuration"].get(key) == right["configuration"].get(key)
        for key in left["configuration"]
        if not is_hex_float(left["configuration"].get(key))
    }

    changed_arrays = [name for name, diff in arrays.items() if diff["changed"]]
    return {
        "left_label": left_label,
        "right_label": right_label,
        "left_sha256": sha256(left_path),
        "right_sha256": sha256(right_path),
        "byte_identical": left_path.read_bytes() == right_path.read_bytes(),
        "authority_equal": authority_equal,
        "environment_equal": environment_equal,
        "plain_configuration_equal": plain_configuration_equal,
        "first_divergent_layer": first_divergent,
        "layer_changes": layer_changes,
        "changed_arrays": changed_arrays,
        "arrays": arrays,
        "scalar_groups": scalar_groups,
    }


def host_metadata() -> dict[str, Any]:
    numpy_config = io.StringIO()
    with contextlib.redirect_stdout(numpy_config):
        np.show_config()
    try:
        lscpu = subprocess.run(
            ["lscpu", "--json"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        lscpu_payload: Any = json.loads(lscpu)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        lscpu_payload = {"error": repr(error)}
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "environment": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "numpy_show_config": numpy_config.getvalue(),
        "lscpu": lscpu_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    first = Path(args.first)
    second = Path(args.second)
    current = Path(args.current)
    report = {
        "schema": "rabbit.d081r1e0.cross_host_fixture_diagnostic.v1",
        "status": "DIAGNOSTIC_ONLY_NO_CANONICAL_MUTATION",
        "host": host_metadata(),
        "comparisons": {
            "first_vs_second": compare_payloads(
                first, second, "run33751080647-westus3", "run33751351930-eastus"
            ),
            "first_vs_current": compare_payloads(
                first, current, "run33751080647-westus3", "current-diagnostic-run"
            ),
            "second_vs_current": compare_payloads(
                second, current, "run33751351930-eastus", "current-diagnostic-run"
            ),
        },
    }

    fixed_input_diverged = any(
        comparison["layer_changes"]["fixed_input"]
        for comparison in report["comparisons"].values()
    )
    report["classification"] = (
        "INVALID_FIXED_INPUT_DIVERGENCE"
        if fixed_input_diverged
        else "CROSS_HOST_FLOATING_OPERATOR_DIVERGENCE"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if fixed_input_diverged:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
