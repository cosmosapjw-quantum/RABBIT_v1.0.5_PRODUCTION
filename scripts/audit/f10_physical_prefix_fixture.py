"""Build and verify the F-10 physical-prefix provenance fixture.

This branch-local audit utility does not move a public capability or gate.  It
keeps deterministic input bytes separate from physical receipt execution so a
Git commit can prospectively seal the latter.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from dataclasses import asdict, dataclass
from collections.abc import Mapping
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
        return {
            "entry_count": len(names),
            "internal_exprb_sha256": sha256_bytes(archive.read(exprb_name)),
            "internal_history_bundle_sha256": sha256_bytes(archive.read(history_name)),
            "internal_history_commit": internal_commit,
            "internal_history_ref": reproducibility["research_history"]["branch"],
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
    protected_paths = [
        "scripts/audit/f10_physical_prefix_fixture.py",
        "tests/test_f10_physical_prefix_fixture.py",
        SOLVER_ZIP_NAME,
        MATHPHYS_ZIP_NAME,
        source_manifest_path.relative_to(repo).as_posix(),
        inputs_path.relative_to(repo).as_posix(),
        grid_manifest_path.relative_to(repo).as_posix(),
        initial_path.relative_to(repo).as_posix(),
        *CHECKPOINT_PATHS,
        *V3_PROVENANCE_PATHS,
    ]
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
    prepare.add_argument("--repo", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        prepare_fixture(args.repo, args.output_dir)
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
