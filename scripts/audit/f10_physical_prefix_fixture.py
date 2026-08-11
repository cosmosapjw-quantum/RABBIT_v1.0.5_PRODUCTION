"""Build and verify the F-10 physical-prefix provenance fixture.

This branch-local audit utility does not move a public capability or gate.  It
keeps deterministic input bytes separate from physical receipt execution so a
Git commit can prospectively seal the latter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from collections.abc import Callable, Mapping
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit import _trajectory_core as core


BASE_BRANCH = "f10-independent-validation-b3v2"
BASE_COMMIT = "719987d0bc5a018d57fded1df2c8ad3f0c3fc24f"
SOLVER_ZIP_NAME = "RABBIT_F10_SolverAlgorithm_Blocker_Research_Loop_2026-08-06.zip"
MATHPHYS_ZIP_NAME = "RABBIT_F10_MathPhysics_Blocker_Research_Loop_2026-08-06.zip"
SOLVER_ZIP_SHA256 = "8ffb9c34019e4bc9e431985df9fe69a347ced5da11f68308a1943187e3829fd8"
MATHPHYS_ZIP_SHA256 = "bb3ca057d1ecee6b11e33bba5dbcd8325a23d95dfe925bb5a235866d05ed4fb0"
SOLVER_ARCHIVE_ROOT = "RABBIT_F10_SolverAlgorithm_Blocker_Research_Loop_2026-08-06"
MATHPHYS_ARCHIVE_ROOT = "RABBIT_F10_MathPhysics_Blocker_Research_Loop_2026-08-06"
SOLVER_HISTORY_COMMIT = "b8f11b03d9d59746c4ceddbb0712dfbd3f5386ab"
DIAGNOSIS_DIR_NAME = "00_F10_PHYSICAL_PREFIX_DIAGNOSIS"

CHECKPOINT_PATHS = (
    ".agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_1200.npz",
    ".agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_2000.npz",
    ".agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_3000.npz",
)

PREFIX_SOURCE_PATHS = (
    "src/rabbit/decoupling/_independent_noqke.py",
    "scripts/audit/_trajectory_core.py",
    "scripts/audit/d069_independent_trajectory_r4.py",
    "native/rabbit_cpu/src/isotropic_boltzmann.rs",
    "native/rabbit_cpu/src/electron_catalog.rs",
    "native/rabbit_cpu/src/quadrature.rs",
    "tests/test_independent_noqke_comparator.py",
    "docs/audit/BD622_D071_trajectory_closure_2026-08-04.md",
    "docs/audit/BD622_V2_option3_closed_and_protocol_2026-08-04.md",
    "docs/audit/BD622_V2_result_2026-08-05.md",
    "docs/audit/BD622_V3_protocol_2026-08-05.md",
    "docs/audit/BD622_V3_report_2026-08-06.md",
)

V3_PROVENANCE_PATHS = (
    ".agent-harness/runs/run-20260804-f10-v1-diagnostic/instrument/instrumented_rhs.py",
    ".agent-harness/runs/run-20260804-f10-v1-diagnostic/instrument/instrumented_bdf.py",
    ".agent-harness/runs/run-20260805-f10-v3-campaign/run_v3.py",
    ".agent-harness/runs/run-20260805-f10-v3-campaign/analyse_v3.py",
    ".agent-harness/runs/run-20260805-f10-v3-campaign/render_v3.py",
    ".agent-harness/runs/run-20260805-f10-v3-campaign/ANALYSIS_V3.json",
    ".agent-harness/runs/run-20260805-f10-v3-campaign/report_verification_output.json",
    ".agent-harness/runs/run-20260805-f10-v3-campaign/r4_reference.json",
    ".agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/pins_verified.json",
    ".agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/selftest_result.json",
    ".agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/driver.log",
    ".agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/nohup.log",
)

V3_DOMAIN_FILENAMES = (
    "accepted.jsonl",
    "jac_factor.jsonl",
    "jacobian_events.jsonl",
    "newton_calls.jsonl",
    "obs_jac_1200.npz",
    "obs_jac_2000.npz",
    "obs_jac_3000.npz",
    "ratcheted_cols_1200.npz",
    "ratcheted_cols_2000.npz",
    "ratcheted_cols_3000.npz",
    "rhs_calls.jsonl",
    "state_1200.npz",
    "state_2000.npz",
    "state_3000.npz",
    "step_events.jsonl",
    "summary.json",
    "trials.jsonl",
)
V3_DOMAIN_PATHS = tuple(
    ".agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/"
    + filename
    for filename in V3_DOMAIN_FILENAMES
)
RETAINED_EVIDENCE_PATHS = (
    SOLVER_ZIP_NAME,
    MATHPHYS_ZIP_NAME,
    *V3_PROVENANCE_PATHS,
    *V3_DOMAIN_PATHS,
)
RECEIPT_RELATIVE_PATHS = (
    "receipts/PHYSICAL_RHS_JVP_RECEIPTS.json",
    "receipts/PHYSICAL_RHS_JVP_VECTORS.npz",
    "receipts/RECEIPT_RUN_LOG.json",
)
REQUIRED_STATE_LABELS = (
    "initial",
    "creep_1200",
    "creep_2000",
    "creep_3000",
)
REQUIREMENT_IDS = (
    "REQ-SOURCE",
    "REQ-CHECKPOINTS",
    "REQ-INPUTS",
    "REQ-RHS-JVP",
    "REQ-RECEIPTS",
    "REQ-CONTRACT",
    "REQ-ENTRYPOINT",
)


def canonical_json_bytes(value: object) -> bytes:
    """Return UTF-8 JSON bytes with one stable, finite-number encoding."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def float64_le_bytes(values: np.ndarray) -> bytes:
    """Return contiguous little-endian float64 bytes independent of host order."""

    return np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes(order="C")


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of ``data``."""

    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    """Hash one file without interpreting or normalizing its bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_deterministic_npz(
    path: Path, arrays: Mapping[str, np.ndarray]
) -> None:
    """Write numeric NPY members in a name-sorted, timestamp-fixed ZIP."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"invalid NPZ member name: {name!r}")
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError("object dtype is forbidden in a sealed NPZ fixture")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            entry = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(
                entry,
                buffer.getvalue(),
                compress_type=ZIP_DEFLATED,
                compresslevel=9,
            )


def load_numeric_npz(path: Path) -> dict[str, np.ndarray]:
    """Load a sealed NPZ without pickle and reject object-bearing members."""

    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("object dtype is forbidden in a sealed NPZ fixture")
    return arrays


@dataclass(frozen=True)
class PhysicalEvaluation:
    """One direct evaluation of the frozen physical collision/RHS path."""

    N: float
    state: np.ndarray
    pair_cloglog: np.ndarray
    temperature_cm_mev: float
    temperature_gamma_mev: float
    elapsed_time_mev_inverse: float
    occupations: np.ndarray
    occupation_min: float
    occupation_max: float
    occupations_strict_open: bool
    rhs: np.ndarray
    collision_electron: np.ndarray
    collision_self_interaction: np.ndarray
    collision_total: np.ndarray
    collision_modal_total: np.ndarray
    electron_bath_energy_transfer: float
    first_law_residual: float
    whole_reaction_domain_rejections: int
    matrix_roundoff_corrections: int
    largest_matrix_roundoff_correction: float
    equilibrium_tail_number_fraction: float
    equilibrium_tail_energy_fraction: float
    tail_edge_relative_distortion_max: float
    tail_edge_occupation_max: float
    reaction_tail_authority_validated: bool
    collision_diagnostics: dict[str, float]


def evaluate_physical_state(
    setup: core.Setup, N: float, state: np.ndarray
) -> PhysicalEvaluation:
    """Evaluate the frozen collision action and its full trajectory RHS once."""

    expansion = float(N)
    packed = np.asarray(state, dtype=np.float64)
    if not np.isfinite(expansion):
        raise ValueError("N must be finite")
    if packed.shape != (setup.state_size,) or not np.all(np.isfinite(packed)):
        raise ValueError("state must be a finite vector with the frozen layout")
    pair_cloglog, temperature_gamma, elapsed_time = core.unpack(setup, packed)
    temperature_cm = setup.t_start * float(np.exp(-expansion))
    occupations = ind.cloglog_to_occupation(pair_cloglog)
    strict_open = bool(
        np.all(np.isfinite(occupations))
        and np.all(occupations > 0.0)
        and np.all(occupations < 1.0)
    )
    if not strict_open:
        raise ind.IndependentNoQkeError("occupation left the strict-open domain")

    action = ind.evaluate_independent_collision_action(
        grid=setup.grid,
        pair_cloglog=pair_cloglog,
        temperature_cm_mev=temperature_cm,
        temperature_gamma_mev=temperature_gamma,
        config=setup.config,
    )
    thermo = ind.independent_thermodynamics(
        grid=setup.grid,
        pair_cloglog=pair_cloglog,
        temperature_cm_mev=temperature_cm,
        temperature_gamma_mev=temperature_gamma,
    )
    total = np.asarray(action.total)
    pair_rate = 0.5 * np.stack(
        (total[0] + total[1], total[2] + total[3], total[4] + total[5])
    )
    chain = ind.cloglog_chain_factor(pair_cloglog)
    dc_dN = pair_rate / (thermo.hubble_mev * chain)
    eos = ind.electromagnetic_eos_adaptive(temperature_gamma)
    dtemperature_gamma_dN = (
        -3.0 * (eos.rho + eos.pressure)
        + action.electron_bath_energy_transfer / thermo.hubble_mev
    ) / eos.drho_dtemperature
    rhs = np.concatenate(
        (dc_dN.ravel(), [dtemperature_gamma_dN, 1.0 / thermo.hubble_mev])
    )
    if rhs.shape != packed.shape or not np.all(np.isfinite(rhs)):
        raise ind.IndependentNoQkeError("physical RHS is nonfinite or wrong-shaped")

    equilibrium = 1.0 / (1.0 + np.exp(setup.grid.nodes))
    tail_slice = slice(max(0, setup.order - 4), setup.order)
    tail_scale = np.maximum(equilibrium[tail_slice], np.finfo(np.float64).tiny)
    tail_distortion = np.abs(
        occupations[:, tail_slice] - equilibrium[None, tail_slice]
    ) / tail_scale[None, :]
    diagnostics = {
        str(name): float(value) for name, value in action.diagnostics.items()
    }
    return PhysicalEvaluation(
        N=expansion,
        state=packed.copy(),
        pair_cloglog=np.asarray(pair_cloglog, dtype=np.float64).copy(),
        temperature_cm_mev=float(temperature_cm),
        temperature_gamma_mev=float(temperature_gamma),
        elapsed_time_mev_inverse=float(elapsed_time),
        occupations=np.asarray(occupations, dtype=np.float64).copy(),
        occupation_min=float(np.min(occupations)),
        occupation_max=float(np.max(occupations)),
        occupations_strict_open=strict_open,
        rhs=np.asarray(rhs, dtype=np.float64),
        collision_electron=np.asarray(action.electron, dtype=np.float64).copy(),
        collision_self_interaction=np.asarray(
            action.self_interaction, dtype=np.float64
        ).copy(),
        collision_total=np.asarray(action.total, dtype=np.float64).copy(),
        collision_modal_total=np.asarray(action.modal_total, dtype=np.float64).copy(),
        electron_bath_energy_transfer=float(action.electron_bath_energy_transfer),
        first_law_residual=float(diagnostics["first_law_residual"]),
        whole_reaction_domain_rejections=int(
            action.whole_reaction_domain_rejections
        ),
        matrix_roundoff_corrections=int(action.matrix_roundoff_corrections),
        largest_matrix_roundoff_correction=float(
            action.largest_matrix_roundoff_correction
        ),
        equilibrium_tail_number_fraction=core.equilibrium_tail_fraction(
            setup.y_max, power=2
        ),
        equilibrium_tail_energy_fraction=core.equilibrium_tail_fraction(
            setup.y_max, power=3
        ),
        tail_edge_relative_distortion_max=float(np.max(tail_distortion)),
        tail_edge_occupation_max=float(np.max(occupations[:, tail_slice])),
        reaction_tail_authority_validated=False,
        collision_diagnostics=diagnostics,
    )


def load_exact_arnoldi(solver_zip: Path) -> Callable[..., object]:
    """Load ``arnoldi`` directly from the SHA-locked solver research ZIP."""

    archive_path = Path(solver_zip).resolve()
    if sha256_path(archive_path) != SOLVER_ZIP_SHA256:
        raise ValueError("solver ZIP digest does not match the exact source lock")
    member = f"{SOLVER_ARCHIVE_ROOT}/src/f10_solver_research/jvp.py"
    with ZipFile(archive_path) as archive:
        source = archive.read(member)
    module_digest = sha256_bytes(source)
    with tempfile.TemporaryDirectory(prefix="rabbit-f10-jvp-") as raw:
        extracted = Path(raw) / "jvp.py"
        extracted.write_bytes(source)
        module_name = f"_rabbit_f10_exact_jvp_{module_digest}"
        specification = importlib.util.spec_from_file_location(module_name, extracted)
        if specification is None or specification.loader is None:
            raise ImportError("could not construct an exact JVP module specification")
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        try:
            specification.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    arnoldi = getattr(module, "arnoldi", None)
    if not callable(arnoldi):
        raise ImportError("exact solver JVP source does not define callable arnoldi")
    setattr(arnoldi, "_rabbit_exact_source_sha256", module_digest)
    return arnoldi


def _evaluation_summary(evaluation: PhysicalEvaluation) -> dict[str, object]:
    return {
        "N": evaluation.N,
        "temperature_cm_mev": evaluation.temperature_cm_mev,
        "temperature_gamma_mev": evaluation.temperature_gamma_mev,
        "elapsed_time_mev_inverse": evaluation.elapsed_time_mev_inverse,
        "state_sha256": sha256_bytes(float64_le_bytes(evaluation.state)),
        "rhs_sha256": sha256_bytes(float64_le_bytes(evaluation.rhs)),
        "collision_total_sha256": sha256_bytes(
            float64_le_bytes(evaluation.collision_total)
        ),
        "electron_bath_energy_transfer": evaluation.electron_bath_energy_transfer,
        "first_law_residual": evaluation.first_law_residual,
        "occupation": {
            "strict_open": evaluation.occupations_strict_open,
            "minimum": evaluation.occupation_min,
            "maximum": evaluation.occupation_max,
        },
        "domain": {
            "whole_reaction_rejections": (
                evaluation.whole_reaction_domain_rejections
            ),
            "matrix_roundoff_corrections": evaluation.matrix_roundoff_corrections,
            "largest_matrix_roundoff_correction": (
                evaluation.largest_matrix_roundoff_correction
            ),
        },
        "tail": {
            "equilibrium_number_fraction_beyond_ymax": (
                evaluation.equilibrium_tail_number_fraction
            ),
            "equilibrium_energy_fraction_beyond_ymax": (
                evaluation.equilibrium_tail_energy_fraction
            ),
            "last_four_node_relative_distortion_max": (
                evaluation.tail_edge_relative_distortion_max
            ),
            "last_four_node_occupation_max": evaluation.tail_edge_occupation_max,
            "reaction_tail_authority_validated": (
                evaluation.reaction_tail_authority_validated
            ),
        },
        "collision_diagnostics": evaluation.collision_diagnostics,
    }


def run_state_arnoldi_receipt(
    setup: core.Setup,
    label: str,
    source_path: str,
    N: float,
    state: np.ndarray,
    arnoldi: Callable[..., object],
    *,
    relative_step: float = 1.0e-3,
    krylov_dim: int = 10,
    tolerance: float = 1.0e-12,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run direct time-augmented JVPs and retain the exact Arnoldi evidence."""

    if not label:
        raise ValueError("state label must be nonempty")
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    if int(krylov_dim) < 1 or int(krylov_dim) > setup.state_size + 1:
        raise ValueError("krylov_dim lies outside the augmented state")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Arnoldi tolerance must be finite and positive")

    packed = np.asarray(state, dtype=np.float64)
    base = evaluate_physical_state(setup, float(N), packed)
    augmented_state = np.concatenate((packed, [float(N)]))
    start_vector = np.concatenate((base.rhs, [1.0]))
    jvp_calls: list[dict[str, object]] = []
    vectors: dict[str, np.ndarray] = {
        "base_collision_total": base.collision_total,
        "base_occupations": base.occupations,
        "base_rhs": base.rhs,
        "base_state": packed.copy(),
    }

    def augmented_operator(direction: np.ndarray) -> np.ndarray:
        vector = np.asarray(direction, dtype=np.float64)
        vector_norm = float(np.linalg.norm(vector))
        if vector.shape != augmented_state.shape or not np.all(np.isfinite(vector)):
            raise ValueError("Arnoldi direction is nonfinite or wrong-shaped")
        if vector_norm == 0.0:
            return np.zeros_like(vector)
        epsilon = (
            float(relative_step)
            * max(1.0, float(np.linalg.norm(augmented_state)))
            / vector_norm
        )
        shifted_N = float(N) + epsilon * float(vector[-1])
        shifted_state = packed + epsilon * vector[:-1]
        shifted = evaluate_physical_state(setup, shifted_N, shifted_state)
        delta = shifted.rhs - base.rhs
        result = np.concatenate((delta / epsilon, [0.0]))
        call_index = len(jvp_calls)
        prefix = f"jvp_{call_index:02d}"
        vectors[f"{prefix}_direction"] = vector.copy()
        vectors[f"{prefix}_jvp"] = result
        vectors[f"{prefix}_shifted_collision_total"] = shifted.collision_total
        vectors[f"{prefix}_shifted_rhs"] = shifted.rhs
        denominator = max(
            float(np.linalg.norm(shifted.rhs)) + float(np.linalg.norm(base.rhs)),
            np.finfo(np.float64).tiny,
        )
        difference_norm = float(np.linalg.norm(delta))
        relative_difference = difference_norm / denominator
        subtractive_condition = (
            None if difference_norm == 0.0 else denominator / difference_norm
        )
        jvp_calls.append(
            {
                "call_index": call_index,
                "scheme": "forward_time_augmented",
                "epsilon": float(epsilon),
                "direction_norm": vector_norm,
                "shifted_N": shifted_N,
                "shifted_state_sha256": sha256_bytes(
                    float64_le_bytes(shifted_state)
                ),
                "shifted_rhs_sha256": sha256_bytes(
                    float64_le_bytes(shifted.rhs)
                ),
                "jvp_sha256": sha256_bytes(float64_le_bytes(result)),
                "rhs_difference_norm": difference_norm,
                "relative_difference_signal": relative_difference,
                "subtractive_condition_ratio": subtractive_condition,
                "physical_diagnostics": _evaluation_summary(shifted),
            }
        )
        return result

    result = arnoldi(
        augmented_operator,
        start_vector,
        max_dim=int(krylov_dim),
        tolerance=float(tolerance),
    )
    basis = np.asarray(getattr(result, "basis"), dtype=np.float64)
    hessenberg = np.asarray(getattr(result, "hessenberg"), dtype=np.float64)
    dimension = int(getattr(result, "dimension"))
    breakdown = bool(getattr(result, "breakdown"))
    if dimension:
        gram_error = basis.T @ basis - np.eye(dimension, dtype=np.float64)
        orthogonality_residual = float(np.linalg.norm(gram_error, ord=np.inf))
    else:
        orthogonality_residual = 0.0
    vectors["arnoldi_basis"] = basis
    vectors["arnoldi_hessenberg"] = hessenberg
    receipt = {
        "schema": "rabbit.f10.physical_rhs_jvp_state_receipt.v1",
        "status": (
            "EXECUTED_WITH_RECORDED_BREAKDOWN" if breakdown else "EXECUTED"
        ),
        "label": label,
        "source_path": source_path,
        "base": _evaluation_summary(base),
        "jvp_rule": {
            "augmentation": "z=(y,N); G(z)=(F(N,y),1)",
            "relative_step": float(relative_step),
            "scheme": "forward_time_augmented",
            "persistent_finite_difference_factor": False,
        },
        "arnoldi": {
            "requested_dimension": int(krylov_dim),
            "dimension": dimension,
            "breakdown": breakdown,
            "breakdown_tolerance": float(tolerance),
            "orthogonalization": "double_modified_gram_schmidt",
            "orthogonality_residual_inf": orthogonality_residual,
            "source_sha256": getattr(
                arnoldi, "_rabbit_exact_source_sha256", None
            ),
        },
        "rhs_call_accounting": {
            "base_calls": 1,
            "shifted_calls": len(jvp_calls),
            "full_rhs_equivalent_calls": 1 + len(jvp_calls),
        },
        "jvp_calls": jvp_calls,
    }
    return receipt, vectors


def run_receipt_set(
    setup: core.Setup,
    states: list[tuple[str, str, float, np.ndarray]],
    arnoldi: Callable[..., object],
    *,
    relative_step: float,
    krylov_dim: int,
    tolerance: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run each requested state independently and retain every failure."""

    state_receipts: list[dict[str, object]] = []
    combined_vectors: dict[str, np.ndarray] = {}
    total_calls = 0
    failure_count = 0
    for label, source_path, expansion, state in states:
        if not label or not label.replace("_", "").isalnum():
            raise ValueError(f"unsafe receipt state label: {label!r}")
        try:
            receipt, vectors = run_state_arnoldi_receipt(
                setup,
                label,
                source_path,
                expansion,
                state,
                arnoldi,
                relative_step=relative_step,
                krylov_dim=krylov_dim,
                tolerance=tolerance,
            )
            total_calls += int(
                receipt["rhs_call_accounting"]["full_rhs_equivalent_calls"]  # type: ignore[index]
            )
            state_receipts.append(receipt)
            for name, array in vectors.items():
                combined_vectors[f"{label}__{name}"] = np.asarray(array)
        except Exception as error:
            failure_count += 1
            state_receipts.append(
                {
                    "schema": "rabbit.f10.physical_rhs_jvp_state_receipt.v1",
                    "status": "ERROR_RETAINED",
                    "label": label,
                    "source_path": source_path,
                    "N": float(expansion),
                    "state_sha256": (
                        sha256_bytes(float64_le_bytes(np.asarray(state)))
                        if np.asarray(state).dtype.kind in "biufc"
                        else None
                    ),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    return (
        {
            "overall_status": (
                "EXECUTED"
                if failure_count == 0
                else "EXECUTED_WITH_RETAINED_FAILURES"
            ),
            "state_count": len(states),
            "failure_count": failure_count,
            "full_rhs_equivalent_calls": total_calls,
            "states": state_receipts,
        },
        combined_vectors,
    )


def verify_protected_paths(repo: Path, contract: Mapping[str, object]) -> int:
    """Verify every contract-protected working-tree path by exact SHA-256."""

    root = Path(repo).resolve()
    entries = contract.get("protected_paths")
    if not isinstance(entries, list) or not entries:
        raise ValueError("contract protected_paths must be a nonempty list")
    checked = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("protected-path entry must be an object")
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("protected-path entry lacks path or sha256")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"protected path escapes repository: {relative}") from error
        if not candidate.is_file():
            raise ValueError(f"protected path is missing: {relative}")
        observed = sha256_path(candidate)
        if observed != expected:
            raise ValueError(
                "protected-path digest mismatch: "
                f"{relative}: {observed} != {expected}"
            )
        checked += 1
    return checked


def _verify_contract_digest(repo: Path, output_dir: Path) -> str:
    root = Path(repo).resolve()
    destination = Path(output_dir).resolve()
    contract_path = destination / "PREFIX_CONTRACT.json"
    checksum_path = destination / "PREFIX_CONTRACT.sha256"
    fields = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2:
        raise ValueError("PREFIX_CONTRACT.sha256 must contain one digest/path pair")
    expected_digest, recorded_path = fields
    expected_path = contract_path.relative_to(root).as_posix()
    if recorded_path != expected_path:
        raise ValueError(
            f"contract checksum path {recorded_path!r} != {expected_path!r}"
        )
    observed = sha256_path(contract_path)
    if observed != expected_digest:
        raise ValueError(f"contract digest mismatch: {observed} != {expected_digest}")
    return observed


def verify_seal(
    repo: Path,
    output_dir: Path,
    seal_commit: str,
    *,
    require_clean: bool,
) -> dict[str, object]:
    """Verify Git chronology plus working-tree and committed protected bytes."""

    root = Path(repo).resolve()
    destination = Path(output_dir).resolve()
    if len(seal_commit) != 40 or any(char not in "0123456789abcdef" for char in seal_commit):
        raise ValueError("seal commit must be a lowercase 40-character Git OID")
    resolved = str(_git(root, "rev-parse", f"{seal_commit}^{{commit}}"))
    if resolved != seal_commit:
        raise ValueError(f"seal commit resolves to {resolved}, expected {seal_commit}")
    head = str(_git(root, "rev-parse", "HEAD"))
    if require_clean and head != seal_commit:
        raise ValueError(f"HEAD {head} is not the prospective seal {seal_commit}")
    status = str(_git(root, "status", "--porcelain"))
    if require_clean and status:
        raise ValueError("working tree is not clean at prospective receipt execution")

    contract_digest = _verify_contract_digest(root, destination)
    contract_path = destination / "PREFIX_CONTRACT.json"
    contract_relative = contract_path.relative_to(root).as_posix()
    committed_contract = _git(
        root, "show", f"{seal_commit}:{contract_relative}", binary=True
    )
    if not isinstance(committed_contract, bytes):
        raise TypeError("committed contract bytes were decoded unexpectedly")
    if sha256_bytes(committed_contract) != contract_digest:
        raise ValueError("seal commit does not contain the working contract bytes")
    contract = read_json(contract_path)
    checked = verify_protected_paths(root, contract)
    for entry in contract["protected_paths"]:  # type: ignore[index]
        relative = entry["path"]  # type: ignore[index]
        expected = entry["sha256"]  # type: ignore[index]
        committed = _git(root, "show", f"{seal_commit}:{relative}", binary=True)
        if not isinstance(committed, bytes):
            raise TypeError("committed protected bytes were decoded unexpectedly")
        observed = sha256_bytes(committed)
        if observed != expected:
            raise ValueError(
                f"seal commit protected digest mismatch: {relative}: "
                f"{observed} != {expected}"
            )
    return {
        "contract_sha256": contract_digest,
        "head": head,
        "protected_path_count": checked,
        "seal_commit": seal_commit,
        "working_tree_clean": not bool(status),
    }


def _vector_manifest(vectors: Mapping[str, np.ndarray]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for name in sorted(vectors):
        array = np.asarray(vectors[name])
        if array.dtype.hasobject:
            raise ValueError("receipt vectors may not contain object dtype")
        entries.append(
            {
                "dtype": str(array.dtype),
                "name": name,
                "sha256": sha256_bytes(np.ascontiguousarray(array).tobytes()),
                "shape": list(array.shape),
            }
        )
    return entries


def write_receipt_artifacts(
    output_dir: Path,
    payload: Mapping[str, object],
    vectors: Mapping[str, np.ndarray],
    *,
    seal_commit: str,
    contract_sha256: str,
    started_utc: str,
    finished_utc: str,
    wall_seconds: float,
) -> None:
    """Write source-bound receipt JSON, numeric vectors, and one run log."""

    if len(seal_commit) != 40:
        raise ValueError("receipt seal commit must be a 40-character OID")
    if len(contract_sha256) != 64:
        raise ValueError("receipt contract digest must be a SHA-256")
    if not np.isfinite(wall_seconds) or wall_seconds < 0.0:
        raise ValueError("receipt wall_seconds must be finite and nonnegative")
    destination = Path(output_dir).resolve() / "receipts"
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / "PHYSICAL_RHS_JVP_RECEIPTS.json"
    vector_path = destination / "PHYSICAL_RHS_JVP_VECTORS.npz"
    run_log_path = destination / "RECEIPT_RUN_LOG.json"
    if any(path.exists() for path in (receipt_path, vector_path, run_log_path)):
        raise FileExistsError("receipt output exists; overwrite/refit is forbidden")

    vector_arrays = {name: np.asarray(array) for name, array in vectors.items()}
    write_deterministic_npz(vector_path, vector_arrays)
    try:
        import scipy

        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = "UNAVAILABLE"
    receipt = {
        "schema": "rabbit.f10.physical_rhs_jvp_receipts.v1",
        "seal_commit": seal_commit,
        "contract_sha256": contract_sha256,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "wall_seconds": float(wall_seconds),
        "environment": {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": sys.version,
            "scipy": scipy_version,
        },
        "vector_manifest": _vector_manifest(vector_arrays),
        "results": dict(payload),
    }
    _write_json(receipt_path, receipt)
    _write_json(
        run_log_path,
        {
            "schema": "rabbit.f10.physical_rhs_jvp_run_log.v1",
            "seal_commit": seal_commit,
            "contract_sha256": contract_sha256,
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "wall_seconds": float(wall_seconds),
            "receipt_sha256": sha256_path(receipt_path),
            "vectors_sha256": sha256_path(vector_path),
            "overall_status": payload.get("overall_status"),
        },
    )


def verify_receipt_artifacts(
    output_dir: Path, seal_commit: str, contract_sha256: str
) -> dict[str, object]:
    """Verify receipt metadata, vector members, counts, and run-log hashes."""

    destination = Path(output_dir).resolve() / "receipts"
    receipt_path = destination / "PHYSICAL_RHS_JVP_RECEIPTS.json"
    vector_path = destination / "PHYSICAL_RHS_JVP_VECTORS.npz"
    run_log_path = destination / "RECEIPT_RUN_LOG.json"
    receipt = read_json(receipt_path)
    if receipt.get("seal_commit") != seal_commit:
        raise ValueError("receipt seal commit does not match the requested seal")
    if receipt.get("contract_sha256") != contract_sha256:
        raise ValueError("receipt contract digest does not match")
    vectors = load_numeric_npz(vector_path)
    manifest = receipt.get("vector_manifest")
    if not isinstance(manifest, list) or len(manifest) != len(vectors):
        raise ValueError("receipt vector manifest count does not match NPZ members")
    for entry in manifest:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError("invalid receipt vector manifest entry")
        name = entry["name"]
        if name not in vectors:
            raise ValueError(f"receipt vector member is missing: {name}")
        array = vectors[name]
        observed = sha256_bytes(np.ascontiguousarray(array).tobytes())
        if observed != entry.get("sha256"):
            raise ValueError(f"receipt vector digest mismatch: {name}")
        if list(array.shape) != entry.get("shape") or str(array.dtype) != entry.get(
            "dtype"
        ):
            raise ValueError(f"receipt vector shape/dtype mismatch: {name}")
    results = receipt.get("results")
    if not isinstance(results, dict) or not isinstance(results.get("states"), list):
        raise ValueError("receipt results lack a states list")
    if results.get("state_count") != len(results["states"]):
        raise ValueError("receipt state count does not match retained states")
    run_log = read_json(run_log_path)
    if run_log.get("seal_commit") != seal_commit:
        raise ValueError("run-log seal commit does not match")
    if run_log.get("receipt_sha256") != sha256_path(receipt_path):
        raise ValueError("run-log receipt digest does not match")
    if run_log.get("vectors_sha256") != sha256_path(vector_path):
        raise ValueError("run-log vectors digest does not match")
    return {
        "failure_count": results.get("failure_count"),
        "overall_status": results.get("overall_status"),
        "state_count": results["state_count"],
        "vector_count": len(vectors),
    }


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(
        value, (int, float, np.integer, np.floating)
    ) and bool(np.isfinite(value))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def validate_static_receipt_payload(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Validate four-state RHS/JVP diagnostic presence without regrading values."""

    results = receipt.get("results")
    if not isinstance(results, dict):
        raise ValueError("receipt lacks results")
    states = results.get("states")
    if not isinstance(states, list):
        raise ValueError("receipt results lack states")
    labels = [state.get("label") if isinstance(state, dict) else None for state in states]
    if labels != list(REQUIRED_STATE_LABELS) or results.get("state_count") != len(
        REQUIRED_STATE_LABELS
    ):
        raise ValueError("receipt state labels do not match the four-state contract")

    allowed_statuses = {"EXECUTED", "EXECUTED_WITH_RECORDED_BREAKDOWN"}
    all_executed = True
    diagnostics_present = True
    direct_jvp_present = True
    per_state: list[dict[str, object]] = []
    for state in states:
        if not isinstance(state, dict):
            raise ValueError("receipt state must be an object")
        status = state.get("status")
        executed = status in allowed_statuses
        all_executed = all_executed and executed
        base = state.get("base")
        first_law = False
        occupation = False
        domain = False
        tail = False
        rhs = False
        jvp = False
        if isinstance(base, dict):
            first_law = _is_finite_number(base.get("first_law_residual"))
            rhs = _is_sha256(base.get("rhs_sha256")) and _is_sha256(
                base.get("collision_total_sha256")
            )
            occupation_payload = base.get("occupation")
            occupation = bool(
                isinstance(occupation_payload, dict)
                and isinstance(occupation_payload.get("strict_open"), bool)
                and _is_finite_number(occupation_payload.get("minimum"))
                and _is_finite_number(occupation_payload.get("maximum"))
            )
            domain_payload = base.get("domain")
            domain = bool(
                isinstance(domain_payload, dict)
                and isinstance(
                    domain_payload.get("whole_reaction_rejections"), int
                )
                and isinstance(
                    domain_payload.get("matrix_roundoff_corrections"), int
                )
                and _is_finite_number(
                    domain_payload.get("largest_matrix_roundoff_correction")
                )
            )
            tail_payload = base.get("tail")
            tail = bool(
                isinstance(tail_payload, dict)
                and all(
                    _is_finite_number(tail_payload.get(field))
                    for field in (
                        "equilibrium_number_fraction_beyond_ymax",
                        "equilibrium_energy_fraction_beyond_ymax",
                        "last_four_node_relative_distortion_max",
                        "last_four_node_occupation_max",
                    )
                )
                and tail_payload.get("reaction_tail_authority_validated") is False
            )
        arnoldi = state.get("arnoldi")
        jvp_calls = state.get("jvp_calls")
        jvp = bool(
            isinstance(arnoldi, dict)
            and _is_sha256(arnoldi.get("source_sha256"))
            and isinstance(jvp_calls, list)
            and len(jvp_calls) > 0
            and all(
                isinstance(call, dict)
                and call.get("scheme") == "forward_time_augmented"
                and _is_finite_number(call.get("epsilon"))
                and float(call["epsilon"]) > 0.0
                for call in jvp_calls
            )
        )
        state_diagnostics = first_law and occupation and domain and tail and rhs
        diagnostics_present = diagnostics_present and state_diagnostics
        direct_jvp_present = direct_jvp_present and jvp
        per_state.append(
            {
                "label": state["label"],
                "status": status,
                "rhs_present": rhs,
                "first_law_present": first_law,
                "occupation_present": occupation,
                "domain_present": domain,
                "tail_present": tail,
                "direct_jvp_present": jvp,
            }
        )

    if results.get("historical_observation_jacobians_used") is not False:
        raise ValueError("direct receipt must reject historical observation Jacobians")
    for field in (
        "physical_prefix_executed",
        "reaction_tail_authority_validated",
        "d071_reopen_earned",
    ):
        if results.get(field) is not False:
            raise ValueError(f"receipt claim ceiling changed: {field}")
    return {
        "all_states_executed": all_executed,
        "all_required_diagnostics_present": diagnostics_present,
        "direct_jvp_provenance_present": direct_jvp_present,
        "historical_observation_jacobians_used": False,
        "physical_prefix_executed": False,
        "reaction_tail_authority_validated": False,
        "d071_reopen_earned": False,
        "states": per_state,
    }


def load_receipt_states(
    repo: Path, output_dir: Path
) -> list[tuple[str, str, float, np.ndarray]]:
    """Load the derived initial state and three exact retained V3a states."""

    root = Path(repo).resolve()
    destination = Path(output_dir).resolve()
    initial_path = destination / "initial_state_order60_ymax30.npz"
    initial = load_numeric_npz(initial_path)
    if set(initial) != {"N", "order", "state_dim", "y", "y_max"}:
        raise ValueError("initial-state NPZ fields do not match the sealed contract")
    states: list[tuple[str, str, float, np.ndarray]] = [
        (
            "initial",
            initial_path.relative_to(root).as_posix(),
            float(initial["N"]),
            np.asarray(initial["y"], dtype=np.float64),
        )
    ]
    for relative in CHECKPOINT_PATHS:
        arrays = load_numeric_npz(root / relative)
        if set(arrays) != {"t", "y", "raw", "h", "order"}:
            raise ValueError(f"retained checkpoint fields changed: {relative}")
        checkpoint_id = Path(relative).stem.removeprefix("state_")
        states.append(
            (
                f"creep_{checkpoint_id}",
                relative,
                float(arrays["t"]),
                np.asarray(arrays["y"], dtype=np.float64),
            )
        )
    expected_labels = ["initial", "creep_1200", "creep_2000", "creep_3000"]
    if [state[0] for state in states] != expected_labels:
        raise ValueError("retained receipt state labels do not match the contract")
    if any(
        state.shape != (182,) or not np.all(np.isfinite(state))
        for _, _, _, state in states
    ):
        raise ValueError("receipt state does not satisfy the finite 182-vector contract")
    return states


def verify_preseal(repo: Path, output_dir: Path) -> dict[str, object]:
    """Verify complete pre-receipt bytes while proving output absence."""

    root = Path(repo).resolve()
    destination = Path(output_dir).resolve()
    contract_digest = _verify_contract_digest(root, destination)
    contract = read_json(destination / "PREFIX_CONTRACT.json")
    protected_count = verify_protected_paths(root, contract)
    states = load_receipt_states(root, destination)
    receipt_paths = (
        destination / "receipts/PHYSICAL_RHS_JVP_RECEIPTS.json",
        destination / "receipts/PHYSICAL_RHS_JVP_VECTORS.npz",
        destination / "receipts/RECEIPT_RUN_LOG.json",
    )
    present = [path.as_posix() for path in receipt_paths if path.exists()]
    if present:
        raise ValueError(f"physical receipt output exists before sealing: {present}")
    return {
        "contract_sha256": contract_digest,
        "protected_path_count": protected_count,
        "receipt_outputs_absent": True,
        "state_count": len(states),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def execute_receipts(
    repo: Path, output_dir: Path, seal_commit: str
) -> dict[str, object]:
    """Execute the four-state direct physical receipt set exactly once."""

    root = Path(repo).resolve()
    destination = Path(output_dir).resolve()
    seal = verify_seal(root, destination, seal_commit, require_clean=True)
    states = load_receipt_states(root, destination)
    setup = core.build_setup(order=60, y_max=30.0, label="f10-prefix")
    manifest = read_json(destination / "QUADRATURE_CATALOG_MANIFEST.json")
    if manifest.get("collision_config") != asdict(setup.config):
        raise ValueError("runtime collision configuration differs from the sealed manifest")
    arnoldi = load_exact_arnoldi(root / SOLVER_ZIP_NAME)
    started_utc = _utc_now()
    started = time.perf_counter()
    payload, vectors = run_receipt_set(
        setup,
        states,
        arnoldi,
        relative_step=1.0e-3,
        krylov_dim=10,
        tolerance=1.0e-12,
    )
    finished_utc = _utc_now()
    wall_seconds = time.perf_counter() - started
    payload.update(
        {
            "schema": "rabbit.f10.physical_rhs_jvp_receipt_set.v1",
            "resolution": {"order": 60, "state_dim": 182, "y_max": 30.0},
            "source_bundle_sha256": sha256_path(root / SOLVER_ZIP_NAME),
            "historical_observation_jacobians_used": False,
            "physical_prefix_executed": False,
            "reaction_tail_authority_validated": False,
            "d071_reopen_earned": False,
        }
    )
    if not vectors:
        vectors = {"receipt_state_count": np.array(len(states), dtype=np.int64)}
    write_receipt_artifacts(
        destination,
        payload,
        vectors,
        seal_commit=seal_commit,
        contract_sha256=str(seal["contract_sha256"]),
        started_utc=started_utc,
        finished_utc=finished_utc,
        wall_seconds=wall_seconds,
    )
    return verify_receipt_artifacts(
        destination, seal_commit, str(seal["contract_sha256"])
    )


def verify_receipts(
    repo: Path, output_dir: Path, seal_commit: str
) -> dict[str, object]:
    """Verify a retained receipt set without requiring a clean worktree."""

    seal = verify_seal(repo, output_dir, seal_commit, require_clean=False)
    receipt = verify_receipt_artifacts(
        output_dir, seal_commit, str(seal["contract_sha256"])
    )
    return {"seal": seal, "receipt": receipt}


def _write_json(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, object]:
    """Load one JSON object and reject non-object top-level values."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document {path} must contain an object")
    return payload


def write_sha256sums(
    repo: Path, output_dir: Path, external_paths: tuple[str, ...]
) -> int:
    """Write a sorted checksum list without recursively hashing the list itself."""

    root = Path(repo).resolve()
    destination = Path(output_dir).resolve()
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise ValueError("checksum output directory must be inside repository") from error
    checksum_path = destination / "SHA256SUMS"
    candidates = [
        path
        for path in destination.rglob("*")
        if path.is_file() and path != checksum_path
    ]
    for relative in external_paths:
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"checksum path escapes repository: {relative}") from error
        if not candidate.is_file():
            raise ValueError(f"checksum input is missing: {relative}")
        candidates.append(candidate)
    unique = {path.relative_to(root).as_posix(): path for path in candidates}
    lines = [
        f"{sha256_path(unique[relative])}  {relative}"
        for relative in sorted(unique)
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def verify_sha256sums(repo: Path, output_dir: Path) -> int:
    """Verify every exact digest/path pair in the diagnosis checksum list."""

    root = Path(repo).resolve()
    checksum_path = Path(output_dir).resolve() / "SHA256SUMS"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("SHA256SUMS is empty")
    observed_paths: set[str] = set()
    for line in lines:
        if "  " not in line:
            raise ValueError("invalid SHA256SUMS line")
        expected, relative = line.split("  ", 1)
        if (
            len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)
            or not relative
            or relative in observed_paths
        ):
            raise ValueError("invalid or duplicate SHA256SUMS entry")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"checksum path escapes repository: {relative}") from error
        if not candidate.is_file():
            raise ValueError(f"checksum path is missing: {relative}")
        actual = sha256_path(candidate)
        if actual != expected:
            raise ValueError(
                f"checksum mismatch: {relative}: {actual} != {expected}"
            )
        observed_paths.add(relative)
    checksum_relative = checksum_path.relative_to(root).as_posix()
    if checksum_relative in observed_paths:
        raise ValueError("SHA256SUMS must not contain its own digest")
    return len(observed_paths)


def _git_blob_oid(repo: Path, path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "hash-object", "--no-filters", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    oid = result.stdout.strip()
    if len(oid) != 40:
        raise ValueError(f"invalid Git blob OID for {path}")
    return oid


def _seal_lacks_receipts(repo: Path, output_dir: Path, seal_commit: str) -> bool:
    root = Path(repo).resolve()
    destination = Path(output_dir).resolve()
    for suffix in RECEIPT_RELATIVE_PATHS:
        relative = (destination / suffix).relative_to(root).as_posix()
        result = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{seal_commit}:{relative}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return False
    return True


def _artifact_metadata(path: str, output_relative: str) -> tuple[
    list[str], str, str, str
]:
    """Return requirement IDs, role, claim status, and bounded note."""

    diagnosis_prefix = output_relative.rstrip("/") + "/"
    if path in PREFIX_SOURCE_PATHS:
        return ["REQ-SOURCE", "REQ-RHS-JVP"], "source", "IMPLEMENTED", (
            "Runtime/source byte locked to the exact F-10 base commit."
        )
    if path == SOLVER_ZIP_NAME:
        return ["REQ-SOURCE", "REQ-RHS-JVP"], "source_bundle", "VALIDATED", (
            "Retained solver archive; exact ZIP and internal JVP/history bytes checked."
        )
    if path == MATHPHYS_ZIP_NAME:
        return ["REQ-SOURCE", "REQ-RECEIPTS"], "source_bundle", "VALIDATED", (
            "Retained math/physics archive including bounded tail evidence and warning."
        )
    if path in CHECKPOINT_PATHS:
        return ["REQ-CHECKPOINTS", "REQ-RECEIPTS"], "checkpoint", "VALIDATED", (
            "Original order-60 retained state at its canonical campaign path."
        )
    if path in V3_DOMAIN_PATHS or path in V3_PROVENANCE_PATHS:
        return ["REQ-CHECKPOINTS", "REQ-RHS-JVP"], "audit", "IMPLEMENTED", (
            "Canonical retained campaign or instrumentation provenance; not a direct JVP receipt."
        )
    if path == "README.md" or path in {
        diagnosis_prefix + "README.md",
        diagnosis_prefix + "FILE_LOCATIONS.md",
    }:
        return ["REQ-ENTRYPOINT"], "audit", "IMPLEMENTED", (
            "Branch-local external-reader navigation."
        )
    suffix = path.removeprefix(diagnosis_prefix)
    if suffix == "SOURCE_BUNDLE.json":
        return ["REQ-SOURCE"], "source_bundle", "VALIDATED", (
            "Machine-readable source/tree/archive identity."
        )
    if suffix in {
        "PREFIX_INPUTS.json",
        "QUADRATURE_CATALOG_MANIFEST.json",
        "initial_state_order60_ymax30.npz",
    }:
        return ["REQ-INPUTS"], "input", "VALIDATED", (
            "Deterministic order-60, y_max=30 input or value-level manifest."
        )
    if suffix in {"PREFIX_CONTRACT.json", "PREFIX_CONTRACT.sha256"}:
        return ["REQ-CONTRACT"], "contract", "SPECIFIED", (
            "Prospective contract committed before direct receipt output."
        )
    if suffix == RECEIPT_RELATIVE_PATHS[0]:
        return ["REQ-RHS-JVP", "REQ-RECEIPTS"], "receipt", "VALIDATED", (
            "Executed four-state physical RHS/JVP diagnostic receipt."
        )
    if suffix == RECEIPT_RELATIVE_PATHS[1]:
        return ["REQ-RHS-JVP", "REQ-RECEIPTS"], "jvp_provenance", "VALIDATED", (
            "Raw direct RHS, collision, Arnoldi, direction, and JVP vectors."
        )
    if suffix == RECEIPT_RELATIVE_PATHS[2]:
        return ["REQ-RHS-JVP", "REQ-RECEIPTS"], "rhs_provenance", "VALIDATED", (
            "Execution chronology and receipt/vector byte bindings."
        )
    raise ValueError(f"no provenance metadata route for {path}")


def build_provenance_index(
    repo: Path, output_dir: Path, seal_commit: str
) -> dict[str, object]:
    """Build the nonrecursive requested-artifact map from actual repository bytes."""

    root = Path(repo).resolve()
    destination = Path(output_dir).resolve()
    output_relative = destination.relative_to(root).as_posix()
    generated = (
        "SOURCE_BUNDLE.json",
        "PREFIX_INPUTS.json",
        "QUADRATURE_CATALOG_MANIFEST.json",
        "initial_state_order60_ymax30.npz",
        "PREFIX_CONTRACT.json",
        "PREFIX_CONTRACT.sha256",
        *RECEIPT_RELATIVE_PATHS,
    )
    candidate_paths = list(
        dict.fromkeys(
            [
                *PREFIX_SOURCE_PATHS,
                *RETAINED_EVIDENCE_PATHS,
                *(f"{output_relative}/{suffix}" for suffix in generated),
                "README.md",
                f"{output_relative}/README.md",
                f"{output_relative}/FILE_LOCATIONS.md",
            ]
        )
    )
    artifacts: list[dict[str, object]] = []
    for relative in candidate_paths:
        candidate = root / relative
        if not candidate.is_file():
            continue
        requirement_ids, role, claim_status, notes = _artifact_metadata(
            relative, output_relative
        )
        source_commit = BASE_COMMIT if relative in PREFIX_SOURCE_PATHS else seal_commit
        derived = relative.startswith(output_relative + "/") or relative == "README.md"
        artifacts.append(
            {
                "artifact_id": "artifact-"
                + sha256_bytes(relative.encode("utf-8"))[:16],
                "requirement_ids": requirement_ids,
                "repo_path": relative,
                "role": role,
                "status": (
                    "PRESENT_DERIVED_VALIDATED" if derived else "PRESENT_TRACKED"
                ),
                "sha256": sha256_path(candidate),
                "git_blob_oid": _git_blob_oid(root, candidate),
                "source_commit": source_commit,
                "claim_status": claim_status,
                "notes": notes,
            }
        )
    return {
        "schema": "rabbit.f10.provenance_index.v1",
        "base_commit": BASE_COMMIT,
        "seal_commit": seal_commit,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def _branch_scope_payload(seal_commit: str) -> dict[str, object]:
    return {
        "schema": "rabbit.f10.diagnosis_branch_scope.v1",
        "branch": "diagnosis_report",
        "base": {"branch": BASE_BRANCH, "commit": BASE_COMMIT},
        "seal_commit": seal_commit,
        "intended_remote_ref": "refs/remotes/origin/diagnosis_report",
        "merge_to_main_authorized": False,
        "purpose": (
            "Externally inspectable source/input/retained-state provenance and "
            "direct static physical RHS/JVP receipts."
        ),
        "claim_ceiling": {
            "physical_prefix_executed": False,
            "reaction_tail_authority_validated": False,
            "d071_reopen_earned": False,
            "public_production_support": "FORBIDDEN",
        },
        "non_goals": [
            "trajectory or physical-prefix execution",
            "D-071 or public capability promotion",
            "QKE implementation",
            "merge into origin/main",
        ],
        "controlling_documents": [
            "AGENTS.md",
            "docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md",
            "bbn_codex_anti_drift_cost_effective_policy.md",
            "docs/superpowers/specs/2026-08-11-f10-diagnosis-report-branch-design.md",
        ],
    }


def _receipt_index_payload(
    output_relative: str, receipt_validation: Mapping[str, object]
) -> dict[str, object]:
    receipt = f"{output_relative}/{RECEIPT_RELATIVE_PATHS[0]}"
    vectors = f"{output_relative}/{RECEIPT_RELATIVE_PATHS[1]}"
    historical = [
        path for path in V3_DOMAIN_PATHS if "/obs_jac_" in path
    ]
    return {
        "schema": "rabbit.f10.receipt_index.v1",
        "state_labels": list(REQUIRED_STATE_LABELS),
        "direct_receipts": {
            "physical_rhs": {
                "path": receipt,
                "selector": "/results/states/*/base/rhs_sha256",
                "raw_vectors": vectors,
            },
            "direct_time_augmented_jvp": {
                "path": receipt,
                "selector": "/results/states/*/jvp_calls",
                "raw_vectors": vectors,
            },
            "first_law": {
                "path": receipt,
                "selector": "/results/states/*/base/first_law_residual",
            },
            "strict_open_occupation": {
                "path": receipt,
                "selector": "/results/states/*/base/occupation",
            },
            "domain_rejection_and_roundoff": {
                "path": receipt,
                "selector": "/results/states/*/base/domain",
            },
            "finite_domain_tail": {
                "path": receipt,
                "selector": "/results/states/*/base/tail",
                "reaction_tail_authority_validated": False,
            },
        },
        "historical_observation_jacobians": {
            "paths": historical,
            "classification": "RETAINED_FINITE_DIFFERENCE_OBSERVATIONS_ONLY",
            "used_for_direct_jvp_receipts": False,
        },
        "validation": dict(receipt_validation),
    }


def _required_artifact_set_complete(
    repo: Path, output_dir: Path, provenance: Mapping[str, object]
) -> bool:
    root = Path(repo).resolve()
    destination = Path(output_dir).resolve()
    required_files = [
        root / "README.md",
        destination / "README.md",
        destination / "FILE_LOCATIONS.md",
        destination / "SOURCE_BUNDLE.json",
        destination / "PREFIX_INPUTS.json",
        destination / "QUADRATURE_CATALOG_MANIFEST.json",
        destination / "initial_state_order60_ymax30.npz",
        destination / "PREFIX_CONTRACT.json",
        destination / "PREFIX_CONTRACT.sha256",
        *(destination / suffix for suffix in RECEIPT_RELATIVE_PATHS),
        *(root / path for path in RETAINED_EVIDENCE_PATHS),
    ]
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    covered = {
        requirement
        for entry in artifacts
        if isinstance(entry, dict)
        for requirement in entry.get("requirement_ids", [])
        if isinstance(requirement, str)
    }
    return all(path.is_file() for path in required_files) and set(
        REQUIREMENT_IDS
    ) <= covered


def finalize_fixture(repo: Path, output_dir: Path, seal_commit: str) -> dict[str, object]:
    """Create deterministic final indexes from the sealed inputs and raw receipts."""

    root = Path(repo).resolve()
    destination = Path(output_dir).resolve()
    output_relative = destination.relative_to(root).as_posix()
    seal = verify_seal(root, destination, seal_commit, require_clean=False)
    receipt_summary = verify_receipt_artifacts(
        destination, seal_commit, str(seal["contract_sha256"])
    )
    receipt = read_json(destination / RECEIPT_RELATIVE_PATHS[0])
    receipt_validation = validate_static_receipt_payload(receipt)
    chronology = _seal_lacks_receipts(root, destination, seal_commit)
    if not chronology:
        raise ValueError("prospective seal already contains receipt output")

    _write_json(destination / "BRANCH_SCOPE.json", _branch_scope_payload(seal_commit))
    _write_json(
        destination / "RECEIPT_INDEX.json",
        _receipt_index_payload(output_relative, receipt_validation),
    )
    provenance = build_provenance_index(root, destination, seal_commit)
    _write_json(destination / "PROVENANCE_INDEX.json", provenance)
    requested_complete = _required_artifact_set_complete(
        root, destination, provenance
    )
    static_executed = bool(
        receipt_validation["all_states_executed"]
        and receipt_validation["all_required_diagnostics_present"]
        and receipt_validation["direct_jvp_provenance_present"]
    )
    readiness = {
        "schema": "rabbit.f10.physical_prefix_readiness.v1",
        "requested_artifact_set_complete": requested_complete,
        "fixture_hashes_validated": True,
        "static_physical_receipts_executed": static_executed,
        "prospective_contract_sealed_before_receipts": chronology,
        "physical_prefix_executed": False,
        "reaction_tail_authority_validated": False,
        "d071_reopen_earned": False,
        "evidence": {
            "seal": seal,
            "receipt": receipt_summary,
            "static_receipt_validation": receipt_validation,
            "limitations": [
                "Static four-state calls are not a trajectory or physical prefix.",
                "Equilibrium and edge-tail metrics are not a reaction-tail authority.",
                "No gate or public capability status is changed by this index.",
            ],
        },
    }
    _write_json(destination / "READINESS.json", readiness)
    validation_path = destination / "VALIDATION_LEDGER.json"
    if not validation_path.exists():
        _write_json(
            validation_path,
            {
                "schema": "rabbit.f10.validation_ledger.v1",
                "seal_commit": seal_commit,
                "contract_sha256": seal["contract_sha256"],
                "automatic_checks": [
                    {
                        "check": "protected seal bytes and contract digest",
                        "status": "PASS",
                        "evidence": seal,
                    },
                    {
                        "check": "receipt/vector/run-log byte bindings",
                        "status": "PASS",
                        "evidence": receipt_summary,
                    },
                    {
                        "check": "four-state required static diagnostics",
                        "status": "PASS" if static_executed else "FAIL",
                        "evidence": receipt_validation,
                    },
                ],
                "executed_commands": [],
                "anti_drift_cost": {
                    "added_lines": None,
                    "deleted_lines": None,
                    "net_lines": None,
                    "token_use_exact": "UNAVAILABLE",
                    "token_use_reason": "Harness exposes no exact per-task token counter.",
                    "blocker_movement_ratio": None,
                    "cost_effectiveness_verdict": "PENDING_FINAL_MEASUREMENT",
                },
            },
        )
    checksum_count = write_sha256sums(
        root,
        destination,
        tuple(dict.fromkeys(("README.md", *RETAINED_EVIDENCE_PATHS))),
    )
    return {
        "checksum_count": checksum_count,
        "readiness": readiness,
        "receipt": receipt_summary,
        "seal": seal,
    }


def _tracked_or_staged(repo: Path, relative: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", relative],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _ignored_even_if_tracked(repo: Path, relative: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "--no-index", "--quiet", "--", relative],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise ValueError(f"git check-ignore failed for {relative}: {result.stderr.strip()}")
    return result.returncode == 0


def _verify_markdown_links(repo: Path, documents: tuple[Path, ...]) -> int:
    root = Path(repo).resolve()
    checked = 0
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for raw_target in pattern.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            candidate = (document.parent / target).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"Markdown link escapes repository: {document}: {raw_target}"
                ) from error
            if not candidate.exists():
                raise ValueError(
                    f"Markdown link target is missing: {document}: {raw_target}"
                )
            checked += 1
    return checked


def _verify_checksum_coverage(
    repo: Path, output_dir: Path, external_paths: tuple[str, ...]
) -> int:
    root = Path(repo).resolve()
    destination = Path(output_dir).resolve()
    checksum_path = destination / "SHA256SUMS"
    listed = {
        line.split("  ", 1)[1]
        for line in checksum_path.read_text(encoding="utf-8").splitlines()
        if "  " in line
    }
    expected = {
        path.relative_to(root).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path != checksum_path
    }
    expected.update(external_paths)
    if listed != expected:
        missing = sorted(expected - listed)
        extra = sorted(listed - expected)
        raise ValueError(f"checksum coverage mismatch: missing={missing}, extra={extra}")
    return len(listed)


def verify_final(repo: Path, output_dir: Path, seal_commit: str) -> dict[str, object]:
    """Fail closed on final hashes, tracking, chronology, receipts, and navigation."""

    root = Path(repo).resolve()
    destination = Path(output_dir).resolve()
    output_relative = destination.relative_to(root).as_posix()
    seal = verify_seal(root, destination, seal_commit, require_clean=False)
    receipt_summary = verify_receipt_artifacts(
        destination, seal_commit, str(seal["contract_sha256"])
    )
    receipt_validation = validate_static_receipt_payload(
        read_json(destination / RECEIPT_RELATIVE_PATHS[0])
    )
    if not _seal_lacks_receipts(root, destination, seal_commit):
        raise ValueError("receipt output predates or appears in prospective seal")

    machine_json_names = (
        "BRANCH_SCOPE.json",
        "SOURCE_BUNDLE.json",
        "PROVENANCE_INDEX.json",
        "PREFIX_INPUTS.json",
        "QUADRATURE_CATALOG_MANIFEST.json",
        "RECEIPT_INDEX.json",
        "READINESS.json",
        "PREFIX_CONTRACT.json",
        "VALIDATION_LEDGER.json",
    )
    machine_documents = {
        name: read_json(destination / name) for name in machine_json_names
    }
    readiness = machine_documents["READINESS.json"]
    for field in (
        "requested_artifact_set_complete",
        "fixture_hashes_validated",
        "static_physical_receipts_executed",
        "prospective_contract_sealed_before_receipts",
    ):
        if readiness.get(field) is not True:
            raise ValueError(f"required artifact readiness is not true: {field}")
    for field in (
        "physical_prefix_executed",
        "reaction_tail_authority_validated",
        "d071_reopen_earned",
    ):
        if readiness.get(field) is not False:
            raise ValueError(f"scientific claim ceiling is not false: {field}")
    if not (
        receipt_validation["all_states_executed"]
        and receipt_validation["all_required_diagnostics_present"]
        and receipt_validation["direct_jvp_provenance_present"]
    ):
        raise ValueError("static physical receipts are structurally incomplete")

    provenance = machine_documents["PROVENANCE_INDEX.json"]
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, list) or provenance.get("artifact_count") != len(
        artifacts
    ):
        raise ValueError("provenance index artifact count is invalid")
    coverage: set[str] = set()
    publish_paths: set[str] = set()
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise ValueError("provenance entry must be an object")
        relative = entry.get("repo_path")
        if not isinstance(relative, str):
            raise ValueError("provenance entry lacks repo_path")
        candidate = root / relative
        if not candidate.is_file() or sha256_path(candidate) != entry.get("sha256"):
            raise ValueError(f"provenance digest/path mismatch: {relative}")
        if _git_blob_oid(root, candidate) != entry.get("git_blob_oid"):
            raise ValueError(f"provenance Git blob mismatch: {relative}")
        requirements = entry.get("requirement_ids")
        if not isinstance(requirements, list):
            raise ValueError(f"provenance requirement list missing: {relative}")
        coverage.update(item for item in requirements if isinstance(item, str))
        publish_paths.add(relative)
    if set(REQUIREMENT_IDS) - coverage:
        raise ValueError(
            f"provenance requirements missing: {sorted(set(REQUIREMENT_IDS) - coverage)}"
        )

    publish_paths.update(
        f"{output_relative}/{name}"
        for name in (
            *machine_json_names,
            "PREFIX_CONTRACT.sha256",
            "SHA256SUMS",
            "initial_state_order60_ymax30.npz",
            *RECEIPT_RELATIVE_PATHS,
            "README.md",
            "FILE_LOCATIONS.md",
        )
    )
    publish_paths.update(RETAINED_EVIDENCE_PATHS)
    publish_paths.add("README.md")
    for relative in sorted(publish_paths):
        if not _tracked_or_staged(root, relative):
            raise ValueError(f"published artifact is not tracked or staged: {relative}")
        if _ignored_even_if_tracked(root, relative):
            raise ValueError(f"published artifact remains ignored: {relative}")

    checksum_count = verify_sha256sums(root, destination)
    external_checksums = tuple(
        dict.fromkeys(("README.md", *RETAINED_EVIDENCE_PATHS))
    )
    _verify_checksum_coverage(root, destination, external_checksums)

    setup = core.build_setup(order=60, y_max=30.0, label="f10-prefix")
    with tempfile.TemporaryDirectory(prefix="rabbit-f10-final-verify-") as raw:
        regenerated = Path(raw) / "initial.npz"
        write_deterministic_npz(regenerated, build_initial_arrays(setup))
        if regenerated.read_bytes() != (
            destination / "initial_state_order60_ymax30.npz"
        ).read_bytes():
            raise ValueError("initial state does not regenerate byte-for-byte")
    if build_quadrature_catalog_manifest(setup) != machine_documents[
        "QUADRATURE_CATALOG_MANIFEST.json"
    ]:
        raise ValueError("quadrature/catalog manifest does not independently regenerate")
    if build_source_bundle_manifest(root) != machine_documents["SOURCE_BUNDLE.json"]:
        raise ValueError("source bundle manifest does not independently regenerate")

    root_readme = root / "README.md"
    notice = root_readme.read_text(encoding="utf-8").lstrip()
    required_link = f"[{DIAGNOSIS_DIR_NAME}]({DIAGNOSIS_DIR_NAME}/README.md)"
    if required_link not in notice[:1200] or "diagnosis_report" not in notice[:1200]:
        raise ValueError("root README does not begin with the diagnosis branch entrypoint")
    link_count = _verify_markdown_links(
        root,
        (
            root_readme,
            destination / "README.md",
            destination / "FILE_LOCATIONS.md",
        ),
    )
    return {
        "checksum_count": checksum_count,
        "link_count": link_count,
        "provenance_artifact_count": len(artifacts),
        "receipt": receipt_summary,
        "seal": seal,
    }


def build_initial_arrays(setup: core.Setup) -> dict[str, np.ndarray]:
    """Build the analytic order/ymax initial state without a collision call."""

    _, state = core.initial_state(setup)
    state_array = np.asarray(state, dtype=np.float64)
    if state_array.shape != (setup.state_size,) or not np.all(np.isfinite(state_array)):
        raise ValueError("initial state does not satisfy the frozen state contract")
    return {
        "N": np.array(0.0, dtype=np.float64),
        "order": np.array(setup.order, dtype=np.int64),
        "state_dim": np.array(setup.state_size, dtype=np.int64),
        "y": state_array,
        "y_max": np.array(setup.y_max, dtype=np.float64),
    }


def _array_manifest(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "canonical_byte_order": "little-endian",
        "count": int(array.size),
        "dtype": "float64",
        "first": float(array.flat[0]),
        "last": float(array.flat[-1]),
        "shape": list(array.shape),
        "values_sha256": sha256_bytes(float64_le_bytes(array)),
    }


def _catalog_manifest(items: tuple[object, ...]) -> dict[str, object]:
    records = [asdict(item) for item in items]
    return {
        "canonical_json_sha256": sha256_bytes(canonical_json_bytes(records)),
        "count": len(records),
        "records": records,
    }


def build_quadrature_catalog_manifest(setup: core.Setup) -> dict[str, object]:
    """Describe the frozen value-level grid and reaction/event catalogues."""

    return {
        "schema": "rabbit.f10.quadrature_catalog_manifest.v1",
        "generator": {
            "grid": "rabbit.decoupling._independent_noqke.build_independent_grid",
            "catalogs": [
                "independent_self_reactions",
                "independent_electron_reactions",
                "independent_self_events",
                "independent_electron_events",
            ],
        },
        "grid": {
            "order": int(setup.order),
            "y_max": float(setup.y_max),
            "nodes": _array_manifest(setup.grid.nodes),
            "weights": _array_manifest(setup.grid.weights),
        },
        "collision_config": asdict(setup.config),
        "catalogs": {
            "self_reactions": _catalog_manifest(ind.independent_self_reactions()),
            "electron_reactions": _catalog_manifest(ind.independent_electron_reactions()),
            "self_events": _catalog_manifest(ind.independent_self_events()),
            "electron_events": _catalog_manifest(ind.independent_electron_events()),
        },
    }


def _git(repo: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout if binary else result.stdout.strip()


def _git_source_entry(repo: Path, relative_path: str) -> dict[str, object]:
    data = _git(repo, "show", f"{BASE_COMMIT}:{relative_path}", binary=True)
    if not isinstance(data, bytes):
        raise TypeError("binary git output was decoded unexpectedly")
    blob_oid = _git(repo, "rev-parse", f"{BASE_COMMIT}:{relative_path}")
    return {
        "git_blob_oid": str(blob_oid),
        "path": relative_path,
        "sha256": sha256_bytes(data),
        "size": len(data),
    }


def _solver_archive_manifest(path: Path) -> dict[str, object]:
    root = SOLVER_ARCHIVE_ROOT
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        reproducibility_name = f"{root}/REPRODUCIBILITY_MANIFEST.json"
        history_name = f"{root}/RABBIT_F10_solver_algorithm_research_history.bundle"
        exprb_name = f"{root}/src/f10_solver_research/solvers/exprb.py"
        jvp_name = f"{root}/src/f10_solver_research/jvp.py"
        required = (reproducibility_name, history_name, exprb_name, jvp_name)
        missing = [name for name in required if name not in names]
        if missing:
            raise ValueError(f"solver archive lacks required members: {missing}")
        reproducibility = json.loads(archive.read(reproducibility_name))
        internal_commit = reproducibility["research_history"]["commit"]
        if internal_commit != SOLVER_HISTORY_COMMIT:
            raise ValueError(
                f"solver history commit {internal_commit} != {SOLVER_HISTORY_COMMIT}"
            )
        history_bytes = archive.read(history_name)
        with tempfile.TemporaryDirectory(prefix="rabbit-f10-bundle-verify-") as raw:
            temporary = Path(raw)
            bundle_path = temporary / "history.bundle"
            repository = temporary / "verify.git"
            bundle_path.write_bytes(history_bytes)
            subprocess.run(
                ["git", "init", "--bare", "--quiet", str(repository)],
                check=True,
                capture_output=True,
                text=True,
            )
            heads_result = subprocess.run(
                ["git", "bundle", "list-heads", str(bundle_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "bundle", "verify", str(bundle_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        bundle_heads = [
            {"commit": line.split(maxsplit=1)[0], "ref": line.split(maxsplit=1)[1]}
            for line in heads_result.stdout.splitlines()
            if line.strip()
        ]
        expected_ref = "refs/heads/" + reproducibility["research_history"]["branch"]
        if {"commit": internal_commit, "ref": expected_ref} not in bundle_heads:
            raise ValueError("solver history bundle lacks the manifest commit/ref")
        return {
            "entry_count": len(names),
            "internal_exprb_sha256": sha256_bytes(archive.read(exprb_name)),
            "internal_history_bundle_sha256": sha256_bytes(history_bytes),
            "internal_history_commit": internal_commit,
            "internal_history_ref": reproducibility["research_history"]["branch"],
            "internal_bundle_heads": bundle_heads,
            "internal_bundle_verified": True,
            "internal_jvp_sha256": sha256_bytes(archive.read(jvp_name)),
            "internal_manifest": reproducibility_name,
            "internal_manifest_sha256": sha256_bytes(
                archive.read(reproducibility_name)
            ),
            "path": path.name,
            "sha256": sha256_path(path),
            "size": path.stat().st_size,
        }


def _mathphysics_archive_manifest(path: Path) -> dict[str, object]:
    root = MATHPHYS_ARCHIVE_ROOT
    tail_name = f"{root}/research_artifacts/results/tail_bounds.json"
    report_name = f"{root}/RABBIT_F10_MATHPHYS_BLOCKER_RESEARCH_REPORT.md"
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = [name for name in (tail_name, report_name) if name not in names]
        if missing:
            raise ValueError(f"mathphysics archive lacks required members: {missing}")
        return {
            "entry_count": len(names),
            "internal_report_sha256": sha256_bytes(archive.read(report_name)),
            "internal_tail_bounds_sha256": sha256_bytes(archive.read(tail_name)),
            "path": path.name,
            "sha256": sha256_path(path),
            "size": path.stat().st_size,
        }


def build_source_bundle_manifest(repo: Path) -> dict[str, object]:
    """Bind the exact RABBIT tree and retained external research archives."""

    root = Path(repo).resolve()
    solver_path = root / SOLVER_ZIP_NAME
    mathphysics_path = root / MATHPHYS_ZIP_NAME
    solver = _solver_archive_manifest(solver_path)
    mathphysics = _mathphysics_archive_manifest(mathphysics_path)
    if solver["sha256"] != SOLVER_ZIP_SHA256:
        raise ValueError("solver research archive digest does not match the retained byte lock")
    if mathphysics["sha256"] != MATHPHYS_ZIP_SHA256:
        raise ValueError("mathphysics archive digest does not match the retained byte lock")
    commit = str(_git(root, "rev-parse", BASE_COMMIT))
    if commit != BASE_COMMIT:
        raise ValueError(f"base commit resolved to {commit}, expected {BASE_COMMIT}")
    return {
        "schema": "rabbit.f10.source_bundle.v1",
        "rabbit_source": {
            "branch": BASE_BRANCH,
            "commit": commit,
            "tree_oid": str(_git(root, "rev-parse", f"{BASE_COMMIT}^{{tree}}")),
            "subtree_oids": {
                path: str(_git(root, "rev-parse", f"{BASE_COMMIT}:{path}"))
                for path in ("docs/audit", "native", "scripts", "src", "tests")
            },
            "prefix_files": [
                _git_source_entry(root, relative_path)
                for relative_path in PREFIX_SOURCE_PATHS
            ],
            "reconstruction": {
                "command": f"git archive --format=tar {BASE_COMMIT}",
                "authority": "Git commit and tree objects on diagnosis_report ancestry",
            },
        },
        "solver_research_archive": solver,
        "mathphysics_research_archive": mathphysics,
    }


def _checkpoint_entry(repo: Path, relative_path: str) -> dict[str, object]:
    path = repo / relative_path
    arrays = load_numeric_npz(path)
    required = {"t", "y", "raw", "h", "order"}
    if set(arrays) != required:
        raise ValueError(
            f"checkpoint {relative_path} fields {sorted(arrays)} != {sorted(required)}"
        )
    state = np.asarray(arrays["y"], dtype=np.float64)
    if state.shape != (182,) or not np.all(np.isfinite(state)):
        raise ValueError(f"checkpoint {relative_path} has an invalid 182-state vector")
    return {
        "fields": sorted(arrays),
        "h": float(arrays["h"]),
        "integrator_order": int(arrays["order"]),
        "N": float(arrays["t"]),
        "path": relative_path,
        "raw_call": int(arrays["raw"]),
        "sha256": sha256_path(path),
        "state_sha256": sha256_bytes(float64_le_bytes(state)),
    }


def _prefix_inputs_payload(
    repo: Path,
    output_dir: Path,
    initial_path: Path,
    grid_manifest_path: Path,
) -> dict[str, object]:
    return {
        "schema": "rabbit.f10.prefix_inputs.v1",
        "resolution": {"order": 60, "state_dim": 182, "y_max": 30.0},
        "initial_state": {
            "claim_status": "DERIVED",
            "derivation": [
                "scripts/audit/_trajectory_core.py::build_setup",
                "scripts/audit/_trajectory_core.py::initial_state",
            ],
            "path": initial_path.relative_to(repo).as_posix(),
            "sha256": sha256_path(initial_path),
        },
        "quadrature_catalog_manifest": {
            "path": grid_manifest_path.relative_to(repo).as_posix(),
            "sha256": sha256_path(grid_manifest_path),
        },
        "retained_creep_checkpoints": [
            _checkpoint_entry(repo, relative_path)
            for relative_path in CHECKPOINT_PATHS
        ],
        "npz_loading": {"allow_pickle": False},
        "output_directory": output_dir.relative_to(repo).as_posix(),
    }


def _contract_payload(
    repo: Path,
    output_dir: Path,
    source_manifest_path: Path,
    inputs_path: Path,
    grid_manifest_path: Path,
    initial_path: Path,
) -> dict[str, object]:
    protected_paths = list(
        dict.fromkeys(
            [
                ".gitignore",
                "scripts/audit/f10_physical_prefix_fixture.py",
                "tests/test_f10_physical_prefix_fixture.py",
                *PREFIX_SOURCE_PATHS,
                SOLVER_ZIP_NAME,
                MATHPHYS_ZIP_NAME,
                source_manifest_path.relative_to(repo).as_posix(),
                inputs_path.relative_to(repo).as_posix(),
                grid_manifest_path.relative_to(repo).as_posix(),
                initial_path.relative_to(repo).as_posix(),
                *CHECKPOINT_PATHS,
                *V3_PROVENANCE_PATHS,
                *V3_DOMAIN_PATHS,
            ]
        )
    )
    protected = [
        {"path": path, "sha256": sha256_path(repo / path)}
        for path in protected_paths
    ]
    return {
        "schema": "rabbit.f10.physical_prefix_contract.v1",
        "contract_status": "PROSPECTIVE_UNEXECUTED",
        "claim_ceiling": {
            "d071_reopen_earned": False,
            "physical_prefix_executed": False,
            "public_production_support": "FORBIDDEN",
        },
        "source_identity": {
            "branch": BASE_BRANCH,
            "commit": BASE_COMMIT,
            "source_manifest": source_manifest_path.relative_to(repo).as_posix(),
            "source_manifest_sha256": sha256_path(source_manifest_path),
        },
        "inputs": {
            "order": 60,
            "y_max": 30.0,
            "state_dim": 182,
            "prefix_inputs": inputs_path.relative_to(repo).as_posix(),
            "prefix_inputs_sha256": sha256_path(inputs_path),
            "quadrature_catalog_manifest_sha256": sha256_path(grid_manifest_path),
            "initial_state_sha256": sha256_path(initial_path),
        },
        "static_receipt_discriminator": {
            "states": ["initial", "creep_1200", "creep_2000", "creep_3000"],
            "physical_evaluator": (
                "rabbit.decoupling._independent_noqke."
                "evaluate_independent_collision_action"
            ),
            "jvp": {
                "augmentation": "z=(y,N); G(z)=(F(N,y),1)",
                "formula": (
                    "epsilon=r*max(1,||z||_2)/||v||_2; "
                    "J_Gv=((F(N+epsilon*v_N,y+epsilon*v_y)-F(N,y))/epsilon,0)"
                ),
                "relative_step": 1.0e-3,
                "scheme": "forward_time_augmented",
                "persistent_finite_difference_factor": False,
            },
            "arnoldi": {
                "krylov_dimension": 10,
                "orthogonalization": "double_modified_gram_schmidt",
                "breakdown_tolerance": 1.0e-12,
                "source": f"{SOLVER_ZIP_NAME}::src/f10_solver_research/jvp.py",
            },
        },
        "full_prefix_obligation": {
            "coverage": {
                "start": "physical_initial_state",
                "must_include": [0.14, 0.22],
                "N_end_minimum": 0.25,
            },
            "method": {
                "candidate": "EC-EXPRB-K",
                "krylov_dimension": 10,
                "total_comoving_energy_coordinate_required": True,
                "algebraic_T_gamma_recovery_required": True,
                "executor_status": "SPECIFIED_NOT_IMPLEMENTED_ON_THIS_BRANCH",
            },
            "hard_caps": {
                "full_rhs_equivalent_call_projection_max": 5500,
                "wall_seconds_max": 64800,
            },
        },
        "required_receipts": [
            "physical_rhs",
            "direct_time_augmented_jvp",
            "first_law",
            "strict_open_occupation",
            "whole_reaction_domain_rejection",
            "matrix_roundoff",
            "finite_domain_tail",
        ],
        "tail_boundary": {
            "equilibrium_tail_receipt_required": True,
            "reaction_tail_authority_validated": False,
            "missing_authority": (
                "extended-domain high-precision lost-action oracle for direct, "
                "collision-source, and propagated-feedback channels"
            ),
        },
        "output_paths": {
            "receipts": (
                f"{output_dir.relative_to(repo).as_posix()}/receipts/"
                "PHYSICAL_RHS_JVP_RECEIPTS.json"
            ),
            "vectors": (
                f"{output_dir.relative_to(repo).as_posix()}/receipts/"
                "PHYSICAL_RHS_JVP_VECTORS.npz"
            ),
            "run_log": (
                f"{output_dir.relative_to(repo).as_posix()}/receipts/"
                "RECEIPT_RUN_LOG.json"
            ),
        },
        "kill_semantics": [
            "missing or nonfinite diagnostic",
            "strict-open occupation failure",
            "Arnoldi/JVP failure or hidden fallback",
            "source, input, or contract digest mismatch",
            "call or wall cap breach",
            "post-output epsilon, krylov dimension, step, threshold, or source change",
            "incomplete required interval",
        ],
        "no_post_output_refit": True,
        "failure_retention_required": True,
        "protected_paths": protected,
    }


def prepare_fixture(repo: Path, output_dir: Path) -> None:
    """Generate only pre-receipt inputs/manifests and the prospective contract."""

    root = Path(repo).resolve()
    destination = Path(output_dir).resolve()
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise ValueError("fixture output directory must be inside the repository") from error
    destination.mkdir(parents=True, exist_ok=True)
    receipt_paths = (
        destination / "receipts/PHYSICAL_RHS_JVP_RECEIPTS.json",
        destination / "receipts/PHYSICAL_RHS_JVP_VECTORS.npz",
        destination / "receipts/RECEIPT_RUN_LOG.json",
    )
    if any(path.exists() for path in receipt_paths):
        raise FileExistsError("prepare refuses to overwrite or coexist with receipt output")

    setup = core.build_setup(order=60, y_max=30.0, label="f10-prefix")
    initial_path = destination / "initial_state_order60_ymax30.npz"
    source_manifest_path = destination / "SOURCE_BUNDLE.json"
    grid_manifest_path = destination / "QUADRATURE_CATALOG_MANIFEST.json"
    inputs_path = destination / "PREFIX_INPUTS.json"
    contract_path = destination / "PREFIX_CONTRACT.json"

    write_deterministic_npz(initial_path, build_initial_arrays(setup))
    _write_json(source_manifest_path, build_source_bundle_manifest(root))
    _write_json(grid_manifest_path, build_quadrature_catalog_manifest(setup))
    _write_json(
        inputs_path,
        _prefix_inputs_payload(root, destination, initial_path, grid_manifest_path),
    )
    _write_json(
        contract_path,
        _contract_payload(
            root,
            destination,
            source_manifest_path,
            inputs_path,
            grid_manifest_path,
            initial_path,
        ),
    )
    contract_relative = contract_path.relative_to(root).as_posix()
    (destination / "PREFIX_CONTRACT.sha256").write_text(
        f"{sha256_path(contract_path)}  {contract_relative}\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="generate deterministic inputs and prospective contract"
    )
    verify_preseal_parser = subparsers.add_parser(
        "verify-preseal", help="verify sealed inputs before any receipt execution"
    )
    run_receipts_parser = subparsers.add_parser(
        "run-receipts", help="execute the sealed four-state physical receipt set"
    )
    verify_receipts_parser = subparsers.add_parser(
        "verify-receipts", help="verify raw receipt bytes against the seal"
    )
    finalize_parser = subparsers.add_parser(
        "finalize", help="write final provenance, readiness, and checksum indexes"
    )
    verify_final_parser = subparsers.add_parser(
        "verify-final", help="verify the complete externally navigable artifact set"
    )
    for command_parser in (
        prepare,
        verify_preseal_parser,
        run_receipts_parser,
        verify_receipts_parser,
        finalize_parser,
        verify_final_parser,
    ):
        command_parser.add_argument("--repo", type=Path, required=True)
        command_parser.add_argument("--output-dir", type=Path, required=True)
    for command_parser in (
        run_receipts_parser,
        verify_receipts_parser,
        finalize_parser,
        verify_final_parser,
    ):
        command_parser.add_argument("--seal-commit", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        prepare_fixture(args.repo, args.output_dir)
        print(json.dumps({"status": "PREPARED"}, sort_keys=True))
        return 0
    if args.command == "verify-preseal":
        result = verify_preseal(args.repo, args.output_dir)
    elif args.command == "run-receipts":
        result = execute_receipts(args.repo, args.output_dir, args.seal_commit)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result.get("failure_count") == 0 else 2
    elif args.command == "verify-receipts":
        result = verify_receipts(args.repo, args.output_dir, args.seal_commit)
    elif args.command == "finalize":
        result = finalize_fixture(args.repo, args.output_dir, args.seal_commit)
    elif args.command == "verify-final":
        result = verify_final(args.repo, args.output_dir, args.seal_commit)
    else:
        raise AssertionError(f"unhandled command {args.command!r}")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
