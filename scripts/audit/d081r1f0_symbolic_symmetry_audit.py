#!/usr/bin/env python3
"""Auxiliary symbolic/high-precision audit for D-081R1F0 symmetry metrology.

This script is not a physics authority.  It checks the chain-rule covariance
on an explicit polynomial equivariant map, verifies the exact ratio identity
used in the propagated bound, evaluates the preserved calibration numbers at
high precision, and records availability of optional CAS/formal executables.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp


def probe_executable(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    if path is None:
        return {"available": False, "path": None, "probe": None}
    commands = {
        "octave": [path, "--version"],
        "octave-cli": [path, "--version"],
        "sage": [path, "--version"],
        "Singular": [path, "--version"],
        "lean": [path, "--version"],
        "lake": [path, "--version"],
        "elan": [path, "--version"],
    }
    command = commands.get(name, [path, "--version"])
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = (completed.stdout + completed.stderr).strip().splitlines()
        return {
            "available": True,
            "path": path,
            "returncode": completed.returncode,
            "probe": output[:3],
        }
    except Exception as exc:  # pragma: no cover - environment receipt only
        return {
            "available": True,
            "path": path,
            "probe_error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    e, mu, tau = sp.symbols("e mu tau", real=True)
    ve, vmu, vtau = sp.symbols("ve vmu vtau", real=True)
    x = sp.Matrix([e, mu, tau])
    v = sp.Matrix([ve, vmu, vtau])
    swap = sp.Matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    equivariant_map = sp.Matrix(
        [
            e**2 + mu * tau,
            e * mu + mu**2 + tau,
            e * tau + tau**2 + mu,
        ]
    )
    substitutions = {e: e, mu: tau, tau: mu}
    swapped_map = equivariant_map.subs(substitutions, simultaneous=True)
    map_covariance = sp.simplify(swapped_map - swap * equivariant_map)
    assert map_covariance == sp.zeros(3, 1)

    jacobian = equivariant_map.jacobian(x)
    swapped_jacobian = jacobian.subs(substitutions, simultaneous=True)
    jvp_covariance = sp.simplify(swapped_jacobian * swap * v - swap * jacobian * v)
    assert jvp_covariance == sp.zeros(3, 1)

    symmetric_direction = {ve: 1, vmu: 1, vtau: 1}
    nonsymmetric_state = {e: 1, mu: 2, tau: 3}
    fixed_state = {e: 1, mu: 2, tau: 2}
    original_jvp = jacobian * v
    symmetric_direction_only_difference = sp.Matrix(
        original_jvp.subs({**symmetric_direction, **nonsymmetric_state})
        - swap * original_jvp.subs({**symmetric_direction, **nonsymmetric_state})
    )
    fixed_state_difference = sp.Matrix(
        original_jvp.subs({**symmetric_direction, **fixed_state})
        - swap * original_jvp.subs({**symmetric_direction, **fixed_state})
    )
    assert symmetric_direction_only_difference != sp.zeros(3, 1)
    assert fixed_state_difference == sp.zeros(3, 1)

    n_r, n_p, s_r, s_p = sp.symbols("n_r n_p s_r s_p", positive=True)
    ratio_identity = sp.simplify(
        n_r / s_r
        - n_p / s_p
        - ((n_r - n_p) / s_r + n_p * (s_p - s_r) / (s_r * s_p))
    )
    assert ratio_identity == 0

    mp.mp.dps = 80
    rust_numerator = mp.mpf("3.22152047171665042e-29")
    rust_scale = mp.mpf("1.27304202805501452e-20")
    python_numerator = mp.mpf("3.21284039573526329e-29")
    python_scale = mp.mpf("1.27304202805500941e-20")
    mu_array_relative = mp.mpf("1.33925624268166116e-11")
    tau_array_relative = mp.mpf("6.57418896388758385e-12")
    scale_upper = max(rust_scale, python_scale)
    delta_mu_upper = mu_array_relative * scale_upper
    delta_tau_upper = tau_array_relative * scale_upper
    rust_ratio = rust_numerator / rust_scale
    python_ratio = python_numerator / python_scale
    observed_difference = abs(rust_ratio - python_ratio)
    binary64_allowance = (
        mp.mpf(64)
        * mp.power(2, -52)
        * max(mp.mpf(1), abs(rust_ratio), abs(python_ratio))
    )
    propagated_bound = (
        (delta_mu_upper + delta_tau_upper) / rust_scale
        + python_numerator
        * max(delta_mu_upper, delta_tau_upper)
        / (rust_scale * python_scale)
        + binary64_allowance
    )
    assert observed_difference <= propagated_bound

    tools = {
        name: probe_executable(name)
        for name in ("octave", "octave-cli", "sage", "Singular", "lean", "lake", "elan")
    }
    payload = {
        "schema": "rabbit.d081r1f0.symbolic_symmetry_audit.v1",
        "classification": "PASS_AUXILIARY_SYMPY_MPMATH_AUDIT",
        "authority_status": "auxiliary-only; repository equations and executed Rust/Python receipts remain authoritative",
        "runtime": {
            "sympy": sp.__version__,
            "mpmath": mp.__version__,
            "mpmath_dps": mp.mp.dps,
        },
        "equivariance": {
            "tested_identity": "DF(Sx) S v = S DF(x) v",
            "explicit_polynomial_map_residual": [str(item) for item in jvp_covariance],
            "symmetric_direction_without_fixed_state_counterexample": [
                str(item) for item in symmetric_direction_only_difference
            ],
            "symmetric_direction_at_fixed_state_residual": [
                str(item) for item in fixed_state_difference
            ],
        },
        "ratio_bound": {
            "exact_algebraic_identity_residual": str(ratio_identity),
            "rust_ratio": mp.nstr(rust_ratio, 40),
            "python_ratio": mp.nstr(python_ratio, 40),
            "observed_difference": mp.nstr(observed_difference, 40),
            "conservative_propagated_bound": mp.nstr(propagated_bound, 40),
            "binary64_allowance": mp.nstr(binary64_allowance, 40),
            "passes": bool(observed_difference <= propagated_bound),
        },
        "optional_tool_probes": tools,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
