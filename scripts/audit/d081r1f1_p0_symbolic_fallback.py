#!/usr/bin/env python3
"""Independent symbolic/high-precision checks for D-081R1F1 P0.

This is a validation-only oracle.  It does not import or execute the Rust
production implementation and does not inspect any retained or holdout state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp

EXPECTED_SYMPY = "1.14.0"
EXPECTED_MPMATH = "1.3.0"
MP_DPS = 80
NUMERIC_CAP = mp.mpf("1e-55")


def symbolic_zero(expression: sp.Expr, label: str) -> str:
    reduced = sp.simplify(sp.factor(sp.together(expression)))
    if reduced != 0:
        raise AssertionError(f"{label}: symbolic residual is {reduced!r}")
    return "PASS"


def relative(left: mp.mpf, right: mp.mpf) -> mp.mpf:
    return abs(left - right) / max(mp.mpf(1), abs(left), abs(right))


def mp_text(value: mp.mpf) -> str:
    return mp.nstr(value, 70)


def exact_symbolic_checks() -> dict[str, str]:
    T, q, k = sp.symbols("T q k", positive=True, finite=True)
    u, W, ymax = sp.symbols("u W ymax", positive=True, finite=True)
    rho = sp.Function("rho")(T)
    p = sp.Function("p")(T)
    Q = sp.Function("Q")(T)
    P = sp.Function("P")(T)
    chi = sp.Function("chi")(T)
    H = sp.sqrt(k * rho)

    checks: dict[str, str] = {}
    checks["hubble_log_derivative"] = symbolic_zero(
        sp.diff(H, T) / H - sp.diff(rho, T) / (2 * rho),
        "hubble_log_derivative",
    )

    spectral = P / (H * q)
    spectral_t = sp.diff(P, T) / (H * q) - spectral * sp.diff(H, T) / H
    checks["spectral_row_quotient"] = symbolic_zero(
        sp.diff(spectral, T) - spectral_t,
        "spectral_row_quotient",
    )

    numerator = -3 * (rho + p) + Q / H
    numerator_t = (
        -3 * (sp.diff(rho, T) + sp.diff(p, T))
        + sp.diff(Q, T) / H
        - (Q / H) * sp.diff(H, T) / H
    )
    checks["photon_numerator"] = symbolic_zero(
        sp.diff(numerator, T) - numerator_t,
        "photon_numerator",
    )

    photon_row = numerator / chi
    photon_row_t = numerator_t / chi - photon_row * sp.diff(chi, T) / chi
    checks["photon_row_quotient"] = symbolic_zero(
        sp.diff(photon_row, T) - photon_row_t,
        "photon_row_quotient",
    )

    elapsed_row = 1 / H
    checks["elapsed_row_reciprocal"] = symbolic_zero(
        sp.diff(elapsed_row, T) + elapsed_row * sp.diff(H, T) / H,
        "elapsed_row_reciprocal",
    )

    momentum = T * u / (1 - u)
    weight = T * W / (1 - u) ** 2
    checks["moving_half_line_node"] = symbolic_zero(
        sp.diff(momentum, T) - momentum / T,
        "moving_half_line_node",
    )
    checks["moving_half_line_weight"] = symbolic_zero(
        sp.diff(weight, T) - weight / T,
        "moving_half_line_weight",
    )

    energy = sp.Function("E")(T)
    occupation = 1 / (1 + sp.exp(energy / T))
    occupation_t = occupation * (1 - occupation) * (
        energy / T**2 - sp.diff(energy, T) / T
    )
    checks["fermi_dirac_chain_rule"] = symbolic_zero(
        sp.diff(occupation, T) - occupation_t,
        "fermi_dirac_chain_rule",
    )

    x = sp.Function("x")(T)
    c0, c1 = sp.symbols("c0 c1", finite=True)
    linear_interpolant = (1 - x) * c0 + x * c1
    checks["moving_linear_interpolant"] = symbolic_zero(
        sp.diff(linear_interpolant, T) - (c1 - c0) * sp.diff(x, T),
        "moving_linear_interpolant",
    )

    y = sp.Function("y")(T)
    xi = 2 * y / ymax - 1
    legendre_two = (3 * xi**2 - 1) / 2
    checks["moving_mapped_basis"] = symbolic_zero(
        sp.diff(legendre_two, T) - 3 * xi * sp.diff(xi, T),
        "moving_mapped_basis",
    )

    checks["first_law_transfer_tangent"] = symbolic_zero(
        sp.diff(Q, T) + sp.diff(-Q, T),
        "first_law_transfer_tangent",
    )
    return checks


def dimension_checks() -> tuple[dict[str, int], dict[str, bool]]:
    # Integer powers of MeV.  The fixed chart variables and quadrature
    # coordinates are dimensionless.
    dims = {
        "Tgamma": 1,
        "H": 1,
        "rho": 4,
        "p": 4,
        "chi_gamma": 3,
        "chi_gamma_T": 2,
        "Q": 5,
        "Q_T": 4,
        "spectral_numerator": 1,
        "spectral_numerator_T": 0,
        "F_c": 0,
        "F_c_T": -1,
        "N_gamma": 4,
        "N_gamma_T": 3,
        "F_gamma": 1,
        "F_gamma_T": 0,
        "F_elapsed": -1,
        "F_elapsed_T": -2,
        "H_T_over_H": -1,
    }
    checks = {
        "hubble_feedback": dims["chi_gamma"] - dims["rho"] == dims["H_T_over_H"],
        "spectral_primal": dims["spectral_numerator"] - dims["H"] == dims["F_c"],
        "spectral_tangent_direct": dims["spectral_numerator_T"] - dims["H"] == dims["F_c_T"],
        "spectral_tangent_feedback": dims["F_c"] + dims["H_T_over_H"] == dims["F_c_T"],
        "photon_numerator_transfer": dims["Q"] - dims["H"] == dims["N_gamma"],
        "photon_numerator_tangent_transfer": dims["Q_T"] - dims["H"] == dims["N_gamma_T"],
        "photon_row_primal": dims["N_gamma"] - dims["chi_gamma"] == dims["F_gamma"],
        "photon_row_tangent_direct": dims["N_gamma_T"] - dims["chi_gamma"] == dims["F_gamma_T"],
        "photon_row_tangent_denominator": dims["F_gamma"] + dims["chi_gamma_T"] - dims["chi_gamma"] == dims["F_gamma_T"],
        "elapsed_primal": -dims["H"] == dims["F_elapsed"],
        "elapsed_tangent": dims["F_elapsed"] + dims["H_T_over_H"] == dims["F_elapsed_T"],
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise AssertionError(f"dimension ledger failed: {failed}")
    return dims, checks


def high_precision_checks() -> dict[str, str]:
    mp.mp.dps = MP_DPS
    T = mp.mpf("1.7")
    q = mp.mpf("0.83")
    k = mp.mpf("2.31")

    rho = lambda t: t**4 + mp.mpf("0.31") * t**2 + mp.mpf("2.7")
    pressure = lambda t: mp.mpf("0.21") * t**4 + mp.mpf("0.17") * t**2
    transfer = lambda t: mp.mpf("0.013") * t**5 / (1 + t)
    spectral_numerator = lambda t: mp.mpf("0.37") * t**2 + mp.mpf("0.19") * t + mp.mpf("0.07")
    hubble = lambda t: mp.sqrt(k * rho(t))

    rho_t = mp.diff(rho, T)
    pressure_t = mp.diff(pressure, T)
    transfer_t = mp.diff(transfer, T)
    H = hubble(T)
    h_t = rho_t / (2 * rho(T))

    residuals: dict[str, mp.mpf] = {}
    residuals["hubble_log_derivative"] = relative(mp.diff(hubble, T) / H, h_t)

    spectral = lambda t: spectral_numerator(t) / (hubble(t) * q)
    spectral_formula = mp.diff(spectral_numerator, T) / (H * q) - spectral(T) * h_t
    residuals["spectral_row_quotient"] = relative(mp.diff(spectral, T), spectral_formula)

    numerator = lambda t: -3 * (rho(t) + pressure(t)) + transfer(t) / hubble(t)
    numerator_formula = (
        -3 * (rho_t + pressure_t)
        + transfer_t / H
        - (transfer(T) / H) * h_t
    )
    residuals["photon_numerator"] = relative(mp.diff(numerator, T), numerator_formula)

    chi = lambda t: mp.diff(rho, t)
    photon_row = lambda t: numerator(t) / chi(t)
    chi_t = mp.diff(chi, T)
    photon_formula = numerator_formula / chi(T) - photon_row(T) * chi_t / chi(T)
    residuals["photon_row_quotient"] = relative(mp.diff(photon_row, T), photon_formula)

    elapsed = lambda t: 1 / hubble(t)
    residuals["elapsed_row_reciprocal"] = relative(
        mp.diff(elapsed, T),
        -elapsed(T) * h_t,
    )

    u = mp.mpf("0.37")
    base_weight = mp.mpf("0.83")
    momentum = lambda t: t * u / (1 - u)
    weight = lambda t: t * base_weight / (1 - u) ** 2
    residuals["moving_half_line_node"] = relative(mp.diff(momentum, T), momentum(T) / T)
    residuals["moving_half_line_weight"] = relative(mp.diff(weight, T), weight(T) / T)

    mass = mp.mpf("0.511")
    scale = mp.mpf("0.71")
    energy = lambda t: mp.sqrt((scale * t) ** 2 + mass**2)
    occupation = lambda t: 1 / (1 + mp.exp(energy(t) / t))
    energy_t = scale**2 * T / energy(T)
    occupation_formula = occupation(T) * (1 - occupation(T)) * (
        energy(T) / T**2 - energy_t / T
    )
    residuals["fermi_dirac_chain_rule"] = relative(
        mp.diff(occupation, T),
        occupation_formula,
    )

    x = lambda t: t / (1 + t)
    c0 = mp.mpf("-0.23")
    c1 = mp.mpf("0.61")
    interpolant = lambda t: (1 - x(t)) * c0 + x(t) * c1
    residuals["moving_linear_interpolant"] = relative(
        mp.diff(interpolant, T),
        (c1 - c0) * mp.diff(x, T),
    )

    ymax = mp.mpf("30")
    y = lambda t: mp.sqrt(t**2 + mp.mpf("0.4"))
    xi = lambda t: 2 * y(t) / ymax - 1
    basis = lambda t: (3 * xi(t) ** 2 - 1) / 2
    basis_formula = 3 * xi(T) * (2 / ymax) * (T / y(T))
    residuals["moving_mapped_basis"] = relative(mp.diff(basis, T), basis_formula)

    qnu = lambda t: t**5 / (1 + t)
    qem = lambda t: -qnu(t)
    residuals["first_law_transfer_tangent"] = abs(mp.diff(qnu, T) + mp.diff(qem, T))

    maximum = max(residuals.values())
    if maximum > NUMERIC_CAP:
        rendered = {name: mp_text(value) for name, value in residuals.items()}
        raise AssertionError(f"high-precision residual exceeded cap: {rendered}")
    return {name: mp_text(value) for name, value in sorted(residuals.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sp.__version__ != EXPECTED_SYMPY:
        raise SystemExit(f"SymPy mismatch: {sp.__version__} != {EXPECTED_SYMPY}")
    if mp.__version__ != EXPECTED_MPMATH:
        raise SystemExit(f"mpmath mismatch: {mp.__version__} != {EXPECTED_MPMATH}")

    symbolic = exact_symbolic_checks()
    dimensions, dimension_results = dimension_checks()
    numeric = high_precision_checks()
    maximum_numeric = max(mp.mpf(value) for value in numeric.values())

    payload: dict[str, Any] = {
        "schema": "rabbit.d081r1f1.p0_symbolic_fallback.v1",
        "classification": "PASS_WITH_SYMPY_MPMATH_P0_IDENTITIES_ONLY",
        "runtime": {
            "python": platform.python_version(),
            "sympy": sp.__version__,
            "mpmath": mp.__version__,
            "mpmath_decimal_digits": MP_DPS,
        },
        "symbolic_checks": symbolic,
        "dimension_ledger_mev_powers": dimensions,
        "dimension_checks": dimension_results,
        "numeric_relative_residuals": numeric,
        "maximum_numeric_relative_residual": mp_text(maximum_numeric),
        "numeric_cap": mp_text(NUMERIC_CAP),
        "wolfram_status": "BLOCKED_EXTERNAL_HTTP_502",
        "retained_or_holdout_state_accessed": False,
        "production_rust_imported_or_executed": False,
        "claim_ceiling": (
            "independent P0 symbolic identities, dimensional consistency, and "
            "80-digit toy-function differentiation only; no Rust thermal "
            "primitive, collision JVP, packed-RHS JVP, retained calibration, "
            "unseen holdout, solver, trajectory, endpoint, N_eff, performance, "
            "publication, or F10 gate movement"
        ),
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print("D081R1F1_P0_SYMBOLIC_FALLBACK_PASS")
    print(f"output={output}")
    print(f"sha256={digest}")
    print(f"maximum_numeric_relative_residual={payload['maximum_numeric_relative_residual']}")


if __name__ == "__main__":
    main()
