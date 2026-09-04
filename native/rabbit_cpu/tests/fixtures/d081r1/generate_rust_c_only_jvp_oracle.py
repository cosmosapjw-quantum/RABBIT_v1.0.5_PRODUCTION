#!/usr/bin/env python3
"""Generate deterministic D-081R1F0 spectral-c analytic-JVP fixtures.

Production Rust code is never imported here.  The oracle is the frozen D-079
Python derivative path applied to the unchanged independent comparator.  The
three cases are deliberately separate so the unseen retained holdout can be
generated only after the order-eight and retained-calibration lanes are GREEN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[5]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rabbit.decoupling import _independent_noqke as ind  # noqa: E402
from scripts.audit._d079_rhs_jvp import (  # noqa: E402
    c_only_state_validator,
    evaluate_c_only_rhs_jvp,
    evaluate_static_rhs_from_packed_state,
)

EXPECTED_PARENT = "8cef907e704149340774214f4da1bd28b79608e9"
EXPECTED_PARENT_TREE = "189e100de980fdbbe654e579d83c939cbdb1cef1"
EXPECTED_CONTRACT_BLOB = "ac7149fe5d5ec327cdc168d1eba7fe4a68ce3221"
EXPECTED_COMPARATOR_BLOB = "de44feee0aa484abe26976c7dc34c579643005b5"
EXPECTED_TANGENT_BLOB = "668f3fab76ffc3ad7f29335a79fcd5daf47d429e"
EXPECTED_COLLISION_JVP_BLOB = "591a64702c58a2de265fb88636f186e2d1b7e019"
EXPECTED_RHS_JVP_BLOB = "6bcff2bc5627c0af0ad4df61c908d09e62ffaba5"
EXPECTED_CARGO_LOCK_BLOB = "a1b5035da5c20712d1a2a4ab077da255ff94a014"
EXPECTED_CONTROL_FIXTURE_BLOB = "c94d2e72a1f8300b7c20c9c793417a5c4a5fa302"
EXPECTED_RETAINED_SHA256 = (
    "c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380"
)
EXPECTED_NUMPY = "2.4.4"
EXPECTED_SCIPY = "1.17.1"
T_START_MEV = 10.0

CONTRACT_PATH = ROOT / "docs/audit/D081R1F0_RUST_C_ONLY_PACKED_RHS_JVP_CONTRACT_2026-09-04.md"
COMPARATOR_PATH = ROOT / "src/rabbit/decoupling/_independent_noqke.py"
TANGENT_PATH = ROOT / "scripts/audit/_d079_tangent_primitives.py"
COLLISION_JVP_PATH = ROOT / "scripts/audit/_d079_collision_jvp.py"
RHS_JVP_PATH = ROOT / "scripts/audit/_d079_rhs_jvp.py"
CARGO_LOCK_PATH = ROOT / "native/rabbit_cpu/Cargo.lock"
CONTROL_FIXTURE_PATH = Path(__file__).with_name("full_collision_action_case.json")


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def git_blob(path: Path) -> str:
    return run("git", "hash-object", str(path.resolve().relative_to(ROOT)))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float_bits(value: float) -> str:
    return f"{np.float64(value).view(np.uint64).item():016x}"


def encode_array(values: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return {
        "shape": list(array.shape),
        "bits": [f"{item:016x}" for item in array.view(np.uint64).ravel().tolist()],
    }


def decode_float(value: Any) -> float:
    text = str(value)
    if text.startswith("0x"):
        text = text[2:]
    raw = np.asarray([int(text, 16)], dtype=np.uint64)
    return float(raw.view(np.float64)[0])


def decode_array(value: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(item) for item in value["shape"])
    bits = np.asarray([int(str(item).removeprefix("0x"), 16) for item in value["bits"]], dtype=np.uint64)
    return bits.view(np.float64).reshape(shape).copy()


def safe_relative(actual: Any, expected: Any) -> float:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    scale = max(
        float(np.max(np.abs(left), initial=0.0)),
        float(np.max(np.abs(right), initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    return float(np.max(np.abs(left - right), initial=0.0) / scale)


def normalized(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector.ravel()))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("direction norm must be finite and positive")
    return vector / norm


def authority_identities() -> dict[str, str]:
    subprocess.check_call(["git", "merge-base", "--is-ancestor", EXPECTED_PARENT, "HEAD"], cwd=ROOT)
    if run("git", "rev-parse", f"{EXPECTED_PARENT}^{{tree}}") != EXPECTED_PARENT_TREE:
        raise SystemExit("parent tree identity mismatch")
    observed = {
        "contract_git_blob": git_blob(CONTRACT_PATH),
        "python_comparator_git_blob": git_blob(COMPARATOR_PATH),
        "python_tangent_git_blob": git_blob(TANGENT_PATH),
        "python_collision_jvp_git_blob": git_blob(COLLISION_JVP_PATH),
        "python_rhs_jvp_git_blob": git_blob(RHS_JVP_PATH),
        "cargo_lock_git_blob": git_blob(CARGO_LOCK_PATH),
        "control_fixture_git_blob": git_blob(CONTROL_FIXTURE_PATH),
    }
    expected = {
        "contract_git_blob": EXPECTED_CONTRACT_BLOB,
        "python_comparator_git_blob": EXPECTED_COMPARATOR_BLOB,
        "python_tangent_git_blob": EXPECTED_TANGENT_BLOB,
        "python_collision_jvp_git_blob": EXPECTED_COLLISION_JVP_BLOB,
        "python_rhs_jvp_git_blob": EXPECTED_RHS_JVP_BLOB,
        "cargo_lock_git_blob": EXPECTED_CARGO_LOCK_BLOB,
        "control_fixture_git_blob": EXPECTED_CONTROL_FIXTURE_BLOB,
    }
    if observed != expected:
        raise SystemExit(f"authority mismatch: {observed} != {expected}")
    return observed


def order8_case() -> tuple[Any, np.ndarray, float, float, float, np.ndarray, dict[str, Any]]:
    document = json.loads(CONTROL_FIXTURE_PATH.read_text(encoding="utf-8"))
    case = next(item for item in document["cases"] if item["name"] == "thermal_split")
    grid = ind.build_independent_grid(order=8, y_max=8.0)
    pair_cloglog = decode_array(case["pair_cloglog"])
    temperature_cm = decode_float(case["temperature_cm_bits"])
    temperature_gamma = decode_float(case["temperature_gamma_bits"])
    ln_a = float(np.log(T_START_MEV / temperature_cm))
    state = np.concatenate((pair_cloglog.ravel(), [temperature_gamma, 0.0]))
    coordinate = np.linspace(-1.0, 1.0, grid.order, dtype=np.float64)
    direction = normalized(
        np.stack(
            (
                0.3 + coordinate,
                -0.2 + coordinate**2,
                -0.2 + coordinate**2,
            )
        )
    )
    metadata = {
        "source": "full_collision_action_case.json:thermal_split",
        "retained_sha256": None,
        "retained_h": None,
        "direction_definition": "ve=0.3+x; vmu=vtau=-0.2+x^2; x=linspace(-1,1,n); global L2 normalization",
    }
    return grid, state, ln_a, temperature_cm, temperature_gamma, direction, metadata


def retained_case(
    retained: Path,
    *,
    holdout: bool,
) -> tuple[Any, np.ndarray, float, float, float, np.ndarray, dict[str, Any]]:
    retained = retained.resolve()
    if not retained.is_file():
        raise SystemExit(f"retained state missing: {retained}")
    digest = sha256(retained)
    if digest != EXPECTED_RETAINED_SHA256:
        raise SystemExit(f"retained SHA-256 mismatch: {digest}")
    with np.load(retained, allow_pickle=False) as archive:
        if set(archive.files) != {"t", "y", "raw", "h", "order"}:
            raise SystemExit(f"unexpected retained keys: {sorted(archive.files)}")
        ln_a = float(np.asarray(archive["t"], dtype=np.float64).reshape(-1)[0])
        state = np.asarray(archive["y"], dtype=np.float64).reshape(-1).copy()
        retained_h = np.asarray(archive["h"], dtype=np.float64).reshape(-1).copy()
    if state.shape != (182,) or not np.all(np.isfinite(state)):
        raise SystemExit("invalid retained packed state")
    grid = ind.build_independent_grid(order=60, y_max=30.0)
    temperature_cm = T_START_MEV * float(np.exp(-ln_a))
    temperature_gamma = float(state[180])
    phase = np.pi * np.arange(grid.order, dtype=np.float64) / (grid.order - 1)
    if holdout:
        direction = np.stack(
            (
                0.25 + np.cos(2.0 * phase),
                -0.15 + np.sin(3.0 * phase),
                0.35 * np.cos(phase) - 0.20 * np.sin(2.0 * phase),
            )
        )
        definition = "ve=0.25+cos(2phi); vmu=-0.15+sin(3phi); vtau=0.35cos(phi)-0.20sin(2phi); global L2 normalization"
    else:
        direction = np.stack((np.cos(phase), np.sin(phase), np.sin(phase)))
        definition = "ve=cos(phi); vmu=vtau=sin(phi); phi=pi*i/(n-1); global L2 normalization"
    metadata = {
        "source": "historical state_1200.npz",
        "retained_sha256": digest,
        "retained_h": encode_array(retained_h),
        "direction_definition": definition,
    }
    return (
        grid,
        state,
        ln_a,
        temperature_cm,
        temperature_gamma,
        normalized(direction),
        metadata,
    )


def branch_signature(action: Any) -> dict[str, Any]:
    return {
        "whole_reaction_domain_rejections": int(action.whole_reaction_domain_rejections),
        "matrix_roundoff_corrections": int(action.matrix_roundoff_corrections),
        "largest_matrix_roundoff_correction_bits": float_bits(
            float(action.largest_matrix_roundoff_correction)
        ),
    }


def centered_witnesses(
    *,
    grid: Any,
    state: np.ndarray,
    direction: np.ndarray,
    temperature_cm: float,
    temperature_gamma: float,
    analytic: Any,
    epsilons: tuple[float, ...],
) -> list[dict[str, Any]]:
    config = ind.IndependentCollisionConfig()
    output: list[dict[str, Any]] = []
    full_direction = np.concatenate((direction.ravel(), [0.0, 0.0]))
    base_signature = branch_signature(analytic.collision.base)
    for epsilon in epsilons:
        plus_state = state + epsilon * full_direction
        minus_state = state - epsilon * full_direction
        valid = c_only_state_validator(grid, plus_state) and c_only_state_validator(grid, minus_state)
        if not valid:
            output.append({"epsilon_bits": float_bits(epsilon), "state_valid": False})
            continue
        plus_rhs = evaluate_static_rhs_from_packed_state(
            grid=grid,
            packed_state=plus_state,
            temperature_cm_mev=temperature_cm,
            config=config,
        )
        minus_rhs = evaluate_static_rhs_from_packed_state(
            grid=grid,
            packed_state=minus_state,
            temperature_cm_mev=temperature_cm,
            config=config,
        )
        packed_fd = (plus_rhs - minus_rhs) / (2.0 * epsilon)
        plus_action = ind.evaluate_independent_collision_action(
            grid=grid,
            pair_cloglog=plus_state[: 3 * grid.order].reshape(3, grid.order),
            temperature_cm_mev=temperature_cm,
            temperature_gamma_mev=temperature_gamma,
            config=config,
        )
        minus_action = ind.evaluate_independent_collision_action(
            grid=grid,
            pair_cloglog=minus_state[: 3 * grid.order].reshape(3, grid.order),
            temperature_cm_mev=temperature_cm,
            temperature_gamma_mev=temperature_gamma,
            config=config,
        )
        modal_fd = (
            np.asarray(plus_action.modal_total, dtype=np.float64)
            - np.asarray(minus_action.modal_total, dtype=np.float64)
        ) / (2.0 * epsilon)
        plus_signature = branch_signature(plus_action)
        minus_signature = branch_signature(minus_action)
        same_branch = plus_signature == base_signature == minus_signature
        output.append(
            {
                "epsilon_bits": float_bits(epsilon),
                "state_valid": True,
                "same_support_and_correction_branch": same_branch,
                "plus_branch": plus_signature,
                "minus_branch": minus_signature,
                "packed_fd": encode_array(packed_fd),
                "collision_modal_fd": encode_array(modal_fd),
                "packed_residual": safe_relative(packed_fd, analytic.jvp),
                "collision_modal_residual": safe_relative(
                    modal_fd, analytic.collision.modal_total
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=("order8", "retained-calibration", "retained-holdout"),
    )
    parser.add_argument("--retained", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if np.__version__ != EXPECTED_NUMPY:
        raise SystemExit(f"NumPy mismatch: {np.__version__} != {EXPECTED_NUMPY}")
    if scipy.__version__ != EXPECTED_SCIPY:
        raise SystemExit(f"SciPy mismatch: {scipy.__version__} != {EXPECTED_SCIPY}")
    identities = authority_identities()

    if args.case == "order8":
        grid, state, ln_a, tcm, tgamma, direction, metadata = order8_case()
        epsilons = (3.0e-4, 1.0e-4, 3.0e-5)
    else:
        if args.retained is None:
            raise SystemExit("--retained is required for retained cases")
        grid, state, ln_a, tcm, tgamma, direction, metadata = retained_case(
            args.retained,
            holdout=args.case == "retained-holdout",
        )
        epsilons = (3.0e-6,)

    pair_cloglog = state[: 3 * grid.order].reshape(3, grid.order)
    analytic = evaluate_c_only_rhs_jvp(
        grid=grid,
        pair_cloglog=pair_cloglog,
        direction_cloglog=direction,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tgamma,
        config=ind.IndependentCollisionConfig(),
    )
    witnesses = centered_witnesses(
        grid=grid,
        state=state,
        direction=direction,
        temperature_cm=tcm,
        temperature_gamma=tgamma,
        analytic=analytic,
        epsilons=epsilons,
    )
    if not any(item.get("same_support_and_correction_branch") for item in witnesses):
        raise SystemExit("no centered witness remained on the frozen branch")

    collision = analytic.collision
    payload = {
        "schema": "rabbit.d081r1f0.c_only_jvp_oracle.v1",
        "case": args.case,
        "repository": "cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION",
        "parent_commit": EXPECTED_PARENT,
        "parent_tree": EXPECTED_PARENT_TREE,
        **identities,
        "generator_git_blob": git_blob(Path(__file__).resolve()),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "order": int(grid.order),
        "y_max_bits": float_bits(float(grid.y_max)),
        "ln_a_bits": float_bits(ln_a),
        "temperature_cm_bits": float_bits(tcm),
        "temperature_gamma_bits": float_bits(tgamma),
        "metadata": metadata,
        "grid_nodes": encode_array(grid.nodes),
        "grid_weights": encode_array(grid.weights),
        "packed_state": encode_array(state),
        "pair_cloglog": encode_array(pair_cloglog),
        "direction_cloglog": encode_array(direction),
        "direction_full": encode_array(analytic.full_direction),
        "direction_logit": encode_array(collision.direction_logit),
        "base_rhs": encode_array(analytic.base_rhs),
        "packed_rhs_jvp": encode_array(analytic.jvp),
        "delta_rho_neutrino_bits": float_bits(analytic.delta_rho_neutrino),
        "delta_hubble_over_hubble_bits": float_bits(
            analytic.delta_hubble_over_hubble
        ),
        "collision": {
            "self_native": encode_array(collision.self_interaction),
            "electron_native": encode_array(collision.electron),
            "total_native": encode_array(collision.total),
            "self_modal": encode_array(collision.modal_self_interaction),
            "electron_modal": encode_array(collision.modal_electron),
            "total_modal": encode_array(collision.modal_total),
            "neutrino_energy_transfer_bits": float_bits(
                collision.neutrino_energy_transfer
            ),
            "electromagnetic_energy_transfer_bits": float_bits(
                collision.electron_bath_energy_transfer
            ),
            "first_law_tangent_residual_bits": float_bits(
                collision.first_law_tangent_residual
            ),
            "self_event_energy_residual_bits": float_bits(
                collision.self_event_energy_residual
            ),
            "self_number_moment_bits": float_bits(collision.self_number_moment),
            "self_energy_moment_bits": float_bits(collision.self_energy_moment),
            "charge_conjugation_residual_bits": float_bits(
                collision.charge_conjugation_residual
            ),
            "mu_tau_residual_bits": float_bits(collision.mu_tau_residual),
            "base_branch": branch_signature(collision.base),
        },
        "centered_witnesses": witnesses,
        "frozen_thresholds": {
            "linearity": 5.0e-12,
            "order8_self_modal": 1.0e-7,
            "order8_electron_modal": 1.0e-7,
            "order8_total_modal": 1.0e-7,
            "order8_packed_rhs_jvp": 5.0e-7,
            "order8_first_law": 2.0e-11,
            "order8_centered_collision": 2.0e-6,
            "order8_centered_packed_rhs": 3.0e-6,
            "retained_component_modal": 1.0e-7,
            "retained_packed_rhs_jvp": 2.0e-4,
            "retained_first_law": 2.0e-9,
            "retained_centered_packed_rhs": 2.0e-4,
        },
        "claim_ceiling": (
            "frozen Python static spectral-c JVP oracle only; no thermal or "
            "expansion input columns, dense Jacobian, solver, trajectory, "
            "endpoint, N_eff, performance, publication, or F10 gate movement"
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote={output}")
    print(f"sha256={sha256(output)}")
    print(f"case={args.case}")
    print(f"order={grid.order}")
    print(f"best_packed_fd={min(item.get('packed_residual', float('inf')) for item in witnesses):.17e}")
    print(f"best_collision_fd={min(item.get('collision_modal_residual', float('inf')) for item in witnesses):.17e}")


if __name__ == "__main__":
    main()
