"""Sealed fixed-state full-Jacobian construction for D-080F.

This module does not change the frozen comparator equations.  It strengthens
D-080E by (i) saturating the direction-independent caches, (ii) making every
reachable numerical array read-only, (iii) binding the prepared state to a
content fingerprint and explicit byte inventory, and (iv) assembling an
actually measured square matrix without allowing cache growth during the
build.

Natural units ``hbar=c=k_B=1`` and packed ordering
``(c_e,c_mu,c_tau,T_gamma,elapsed_time)`` are inherited unchanged.  The final
elapsed-time input column is exact structural zero.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import inspect
import json
import time
from types import ModuleType
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_rhs_jvp import evaluate_c_only_rhs_jvp
from scripts.audit._d080c_tgamma_rhs import evaluate_tgamma_rhs_column
from scripts.audit._d080d_static_jacobian import rhs_block_relative
from scripts.audit._d080e_prepared_jvp import (
    FixedStateReusePolicy,
    PreparedStaticRhs,
    evaluate_prepared_c_only_rhs_jvp,
    evaluate_prepared_c_only_rhs_jvps,
    prepare_static_rhs_reuse,
)

FloatArray = NDArray[np.float64]


class D080FSealError(RuntimeError):
    """Fail-closed error for a changed or incompletely frozen prepared state."""


@dataclass(frozen=True)
class PreparedStateSeal:
    schema: str
    fingerprint_sha256: str
    array_count: int
    unique_array_bytes: int
    cache_unique_bytes: int
    all_arrays_readonly: bool
    cache_snapshot: dict[str, int]
    structural_cache_snapshot: dict[str, int]
    contract: dict[str, Any]
    array_manifest: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SealedPreparedStaticRhs:
    prepared: PreparedStaticRhs
    seal: PreparedStateSeal


@dataclass(frozen=True)
class FrozenStaticJacobianBuild:
    base_rhs: FloatArray
    jacobian: FloatArray
    spectral_columns: FloatArray
    tgamma_column: FloatArray
    elapsed_column: FloatArray
    build_seconds: float
    matrix_sha256: str
    seal_before_sha256: str
    seal_after_sha256: str
    cache_snapshot_before: dict[str, int]
    cache_snapshot_after: dict[str, int]
    cache_miss_delta: int
    cache_entry_delta: int
    maximum_prepared_action_residual: float
    maximum_serial_oracle_residual: float


_STRUCTURAL_CACHE_KEYS = (
    "kinematic_misses",
    "self_matrix_misses",
    "electron_matrix_misses",
    "modal_basis_misses",
    "kinematic_entries",
    "self_matrix_entries",
    "electron_matrix_entries",
    "modal_basis_entries",
    "estimated_cache_bytes",
)
_MISS_KEYS = (
    "kinematic_misses",
    "self_matrix_misses",
    "electron_matrix_misses",
    "modal_basis_misses",
)
_ENTRY_KEYS = (
    "kinematic_entries",
    "self_matrix_entries",
    "electron_matrix_entries",
    "modal_basis_entries",
)


def _is_traversable_object(value: Any) -> bool:
    if isinstance(value, ModuleType) or inspect.isroutine(value) or inspect.isclass(value):
        return False
    module = type(value).__module__
    return module.startswith("rabbit.") or module.startswith("scripts.audit.")


def _walk_arrays(
    value: Any,
    path: str,
    *,
    seen_objects: set[int],
    seen_arrays: set[int],
    output: list[tuple[str, np.ndarray]],
) -> None:
    if isinstance(value, np.ndarray):
        identity = id(value)
        if identity not in seen_arrays:
            seen_arrays.add(identity)
            output.append((path, value))
        return
    if value is None or isinstance(value, (str, bytes, int, float, bool, complex)):
        return

    identity = id(value)
    if identity in seen_objects:
        return
    seen_objects.add(identity)

    if is_dataclass(value):
        for field in fields(value):
            _walk_arrays(
                getattr(value, field.name),
                f"{path}.{field.name}",
                seen_objects=seen_objects,
                seen_arrays=seen_arrays,
                output=output,
            )
        return
    if isinstance(value, dict):
        # Dictionary keys can contain process-local object identities.  The
        # content seal intentionally fingerprints values rather than those
        # unstable keys.
        for index, item in enumerate(value.values()):
            _walk_arrays(
                item,
                f"{path}.value[{index}]",
                seen_objects=seen_objects,
                seen_arrays=seen_arrays,
                output=output,
            )
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _walk_arrays(
                item,
                f"{path}[{index}]",
                seen_objects=seen_objects,
                seen_arrays=seen_arrays,
                output=output,
            )
        return
    if _is_traversable_object(value) and hasattr(value, "__dict__"):
        for name, item in sorted(vars(value).items()):
            if callable(item):
                continue
            _walk_arrays(
                item,
                f"{path}.{name}",
                seen_objects=seen_objects,
                seen_arrays=seen_arrays,
                output=output,
            )


def _prepared_arrays(prepared: PreparedStaticRhs) -> list[tuple[str, np.ndarray]]:
    output: list[tuple[str, np.ndarray]] = []
    _walk_arrays(
        prepared,
        "prepared",
        seen_objects=set(),
        seen_arrays=set(),
        output=output,
    )
    return output


def _cache_arrays(prepared: PreparedStaticRhs) -> list[tuple[str, np.ndarray]]:
    output: list[tuple[str, np.ndarray]] = []
    roots = (
        ("cache.kinematics", prepared.cache.kinematic_cache),
        ("cache.self_matrix", prepared.cache.self_matrix_cache),
        ("cache.electron_matrix", prepared.cache.electron_matrix_cache),
        ("cache.modal_basis", prepared.cache.modal_basis_cache),
    )
    seen_objects: set[int] = set()
    seen_arrays: set[int] = set()
    for path, value in roots:
        _walk_arrays(
            value,
            path,
            seen_objects=seen_objects,
            seen_arrays=seen_arrays,
            output=output,
        )
    return output


def _root_array(array: np.ndarray) -> np.ndarray:
    root = array
    seen: set[int] = set()
    while isinstance(root.base, np.ndarray) and id(root.base) not in seen:
        seen.add(id(root))
        root = root.base
    return root


def _unique_allocation_bytes(arrays: Iterable[tuple[str, np.ndarray]]) -> int:
    roots: dict[int, np.ndarray] = {}
    for _path, array in arrays:
        root = _root_array(array)
        roots.setdefault(id(root), root)
    return int(sum(array.nbytes for array in roots.values()))


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _structural_cache_snapshot(snapshot: dict[str, int]) -> dict[str, int]:
    return {key: int(snapshot[key]) for key in _STRUCTURAL_CACHE_KEYS}


def _contract(prepared: PreparedStaticRhs) -> dict[str, Any]:
    config = prepared.config
    policy = prepared.cache.policy
    return {
        "order": int(prepared.raw_grid.order),
        "y_max": float(prepared.raw_grid.y_max),
        "temperature_cm_mev": float(prepared.temperature_cm_mev),
        "temperature_gamma_mev": float(prepared.temperature_gamma_mev),
        "electron_mass_mev": float(prepared.electron_mass_mev),
        "collision_config": {
            "incoming_polar_order": int(config.incoming_polar_order),
            "final_polar_order": int(config.final_polar_order),
            "final_azimuth_order": int(config.final_azimuth_order),
            "electron_radial_order": int(config.electron_radial_order),
            "matrix_roundoff_ulps": float(config.matrix_roundoff_ulps),
        },
        "reuse_policy": {
            "cache_kinematics": bool(policy.cache_kinematics),
            "cache_matrices": bool(policy.cache_matrices),
            "cache_modal_basis": bool(policy.cache_modal_basis),
            "max_modal_basis_bytes": int(policy.max_modal_basis_bytes),
        },
        "state_ordering": "(c_e,c_mu,c_tau,T_gamma,elapsed_time)",
        "natural_units": {"hbar": 1, "c": 1, "k_B": 1},
    }


def _build_seal(prepared: PreparedStaticRhs, *, freeze: bool) -> PreparedStateSeal:
    arrays = _prepared_arrays(prepared)
    if not arrays:
        raise D080FSealError("prepared state contains no numerical arrays")

    if freeze:
        # Freeze both views and ultimate backing allocations.  This prevents a
        # writable alias from changing bytes behind a nominally read-only view.
        for _path, array in arrays:
            root = _root_array(array)
            root.setflags(write=False)
            array.setflags(write=False)

    records: list[dict[str, Any]] = []
    for path, array in arrays:
        records.append(
            {
                "path": path,
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "nbytes": int(array.nbytes),
                "sha256": _array_digest(array),
                "writeable": bool(array.flags.writeable),
            }
        )

    # Process-local cache keys may include id(batch), so the fingerprint is a
    # sorted multiset of array content descriptors plus the physical contract.
    stable_records = sorted(
        (
            record["dtype"],
            tuple(record["shape"]),
            record["nbytes"],
            record["sha256"],
        )
        for record in records
    )
    snapshot = prepared.cache.snapshot()
    structural = _structural_cache_snapshot(snapshot)
    payload = {
        "schema": "rabbit.d080f.prepared_state_seal.v1",
        "contract": _contract(prepared),
        "structural_cache": structural,
        "arrays": stable_records,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_arrays = _cache_arrays(prepared)
    all_readonly = all(not array.flags.writeable for _path, array in arrays)
    return PreparedStateSeal(
        schema="rabbit.d080f.prepared_state_seal.v1",
        fingerprint_sha256=fingerprint,
        array_count=len(records),
        unique_array_bytes=_unique_allocation_bytes(arrays),
        cache_unique_bytes=_unique_allocation_bytes(cache_arrays),
        all_arrays_readonly=bool(all_readonly),
        cache_snapshot={key: int(value) for key, value in snapshot.items()},
        structural_cache_snapshot=structural,
        contract=_contract(prepared),
        array_manifest=tuple(records),
    )


def _warm_direction(order: int, phase: float) -> FloatArray:
    x = np.linspace(-1.0, 1.0, int(order), dtype=np.float64)
    direction = np.stack(
        (
            np.cos((0.7 + phase) * x),
            np.sin((1.1 + phase) * x + 0.2),
            x * x - 0.3 + phase * x,
        )
    )
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise D080FSealError("invalid cache-warming direction")
    return np.asarray(direction / norm, dtype=np.float64)


def prepare_and_seal_static_rhs(
    *,
    grid: ind.IndependentNoQkeGrid,
    pair_cloglog: ArrayLike,
    temperature_cm_mev: float,
    temperature_gamma_mev: float,
    config: ind.IndependentCollisionConfig = ind.IndependentCollisionConfig(),
    electron_mass_mev: float = ind.M_ELECTRON_MEV,
    max_modal_basis_bytes: int = 768 * 1024**2,
) -> SealedPreparedStaticRhs:
    """Prepare, saturate, deep-freeze, and content-seal one fixed state."""

    prepared = prepare_static_rhs_reuse(
        grid=grid,
        pair_cloglog=pair_cloglog,
        temperature_cm_mev=temperature_cm_mev,
        temperature_gamma_mev=temperature_gamma_mev,
        config=config,
        electron_mass_mev=electron_mass_mev,
        policy=FixedStateReusePolicy(
            max_modal_basis_bytes=int(max_modal_basis_bytes)
        ),
    )

    # A dense tangent and the thermal column visit every direction-independent
    # kinematic, matrix, and modal-basis query used by later basis columns.
    evaluate_prepared_c_only_rhs_jvp(prepared, _warm_direction(grid.order, 0.0))
    with prepared.cache.patch():
        evaluate_tgamma_rhs_column(
            grid=prepared.grid,
            pair_cloglog=prepared.pair_cloglog,
            temperature_cm_mev=prepared.temperature_cm_mev,
            temperature_gamma_mev=prepared.temperature_gamma_mev,
            config=prepared.config,
            electron_mass_mev=prepared.electron_mass_mev,
        )
    first = _structural_cache_snapshot(prepared.cache.snapshot())
    evaluate_prepared_c_only_rhs_jvp(prepared, _warm_direction(grid.order, 0.37))
    with prepared.cache.patch():
        evaluate_tgamma_rhs_column(
            grid=prepared.grid,
            pair_cloglog=prepared.pair_cloglog,
            temperature_cm_mev=prepared.temperature_cm_mev,
            temperature_gamma_mev=prepared.temperature_gamma_mev,
            config=prepared.config,
            electron_mass_mev=prepared.electron_mass_mev,
        )
    second = _structural_cache_snapshot(prepared.cache.snapshot())
    if first != second:
        raise D080FSealError(
            "direction-independent cache did not saturate before sealing"
        )

    seal = _build_seal(prepared, freeze=True)
    if not seal.all_arrays_readonly:
        raise D080FSealError("not every prepared array is read-only")
    return SealedPreparedStaticRhs(prepared=prepared, seal=seal)


def verify_prepared_state_seal(sealed: SealedPreparedStaticRhs) -> bool:
    """Fail closed if content, contract, cache structure, or mutability changed."""

    current = _build_seal(sealed.prepared, freeze=False)
    if not current.all_arrays_readonly:
        raise D080FSealError("prepared array regained write permission")
    if current.fingerprint_sha256 != sealed.seal.fingerprint_sha256:
        raise D080FSealError("prepared-state content fingerprint changed")
    if current.structural_cache_snapshot != sealed.seal.structural_cache_snapshot:
        raise D080FSealError("prepared cache structure changed after sealing")
    return True


def _matrix_digest(matrix_value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(matrix_value, dtype=np.float64))
    header = json.dumps(
        {"shape": list(array.shape), "dtype": array.dtype.str},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _deterministic_probe_directions(order: int) -> tuple[FloatArray, FloatArray]:
    return _warm_direction(order, 0.19), _warm_direction(order, 0.83)


def assemble_sealed_static_jacobian(
    sealed: SealedPreparedStaticRhs,
    *,
    direction_block_size: int = 8,
    serial_oracle_columns: tuple[int, ...] = (),
) -> FrozenStaticJacobianBuild:
    """Execute a complete explicit matrix build without changing sealed data."""

    verify_prepared_state_seal(sealed)
    prepared = sealed.prepared
    grid = prepared.raw_grid
    spectral_size = 3 * grid.order
    state_size = spectral_size + 2
    block_size = int(direction_block_size)
    if block_size <= 0:
        raise ValueError("direction_block_size must be positive")
    for column in serial_oracle_columns:
        if int(column) < 0 or int(column) >= spectral_size:
            raise ValueError("serial oracle column is outside the spectral block")

    before = prepared.cache.snapshot()
    seal_before = _build_seal(prepared, freeze=False)
    start = time.perf_counter()

    with prepared.cache.patch():
        tgamma = evaluate_tgamma_rhs_column(
            grid=prepared.grid,
            pair_cloglog=prepared.pair_cloglog,
            temperature_cm_mev=prepared.temperature_cm_mev,
            temperature_gamma_mev=prepared.temperature_gamma_mev,
            config=prepared.config,
            electron_mass_mev=prepared.electron_mass_mev,
        )
    if rhs_block_relative(prepared.base_rhs, tgamma.base_rhs, grid.order) > 5.0e-13:
        raise D080FSealError("prepared and thermal-column base RHS disagree")

    spectral_columns = np.empty((state_size, spectral_size), dtype=np.float64)
    for block_start in range(0, spectral_size, block_size):
        block_stop = min(block_start + block_size, spectral_size)
        directions = np.zeros(
            (block_stop - block_start, 3, grid.order), dtype=np.float64
        )
        for local_index, column in enumerate(range(block_start, block_stop)):
            directions[local_index].ravel()[column] = 1.0
        values = evaluate_prepared_c_only_rhs_jvps(prepared, directions)
        spectral_columns[:, block_start:block_stop] = values.T

    jacobian = np.zeros((state_size, state_size), dtype=np.float64)
    jacobian[:, :spectral_size] = spectral_columns
    jacobian[:, spectral_size] = tgamma.tgamma_column
    elapsed_column = jacobian[:, -1]
    if np.any(elapsed_column != 0.0) or not np.all(np.isfinite(jacobian)):
        raise D080FSealError("invalid explicit static Jacobian")
    build_seconds = float(time.perf_counter() - start)

    maximum_prepared_residual = 0.0
    for direction in _deterministic_probe_directions(grid.order):
        direct = evaluate_prepared_c_only_rhs_jvp(prepared, direction)
        full_direction = np.concatenate((direction.ravel(), [0.0, 0.0]))
        maximum_prepared_residual = max(
            maximum_prepared_residual,
            rhs_block_relative(
                jacobian @ full_direction,
                direct.jvp,
                grid.order,
            ),
        )

    maximum_serial_residual = 0.0
    for column in serial_oracle_columns:
        direction = np.zeros((3, grid.order), dtype=np.float64)
        direction.ravel()[int(column)] = 1.0
        reference = evaluate_c_only_rhs_jvp(
            grid=grid,
            pair_cloglog=prepared.pair_cloglog,
            direction_cloglog=direction,
            temperature_cm_mev=prepared.temperature_cm_mev,
            temperature_gamma_mev=prepared.temperature_gamma_mev,
            config=prepared.config,
            electron_mass_mev=prepared.electron_mass_mev,
        )
        maximum_serial_residual = max(
            maximum_serial_residual,
            rhs_block_relative(
                spectral_columns[:, int(column)],
                reference.jvp,
                grid.order,
            ),
        )

    after = prepared.cache.snapshot()
    seal_after = _build_seal(prepared, freeze=False)
    cache_miss_delta = int(sum(after[key] - before[key] for key in _MISS_KEYS))
    cache_entry_delta = int(sum(after[key] - before[key] for key in _ENTRY_KEYS))
    if cache_miss_delta != 0 or cache_entry_delta != 0:
        raise D080FSealError("sealed full build grew a direction-independent cache")
    if seal_after.fingerprint_sha256 != seal_before.fingerprint_sha256:
        raise D080FSealError("prepared-state bytes changed during full build")
    verify_prepared_state_seal(sealed)

    jacobian.setflags(write=False)
    spectral_columns.setflags(write=False)
    elapsed_column.setflags(write=False)
    tgamma_column = np.asarray(tgamma.tgamma_column, dtype=np.float64).copy()
    tgamma_column.setflags(write=False)
    base_rhs = np.asarray(prepared.base_rhs, dtype=np.float64).copy()
    base_rhs.setflags(write=False)
    return FrozenStaticJacobianBuild(
        base_rhs=base_rhs,
        jacobian=jacobian,
        spectral_columns=spectral_columns,
        tgamma_column=tgamma_column,
        elapsed_column=elapsed_column,
        build_seconds=build_seconds,
        matrix_sha256=_matrix_digest(jacobian),
        seal_before_sha256=seal_before.fingerprint_sha256,
        seal_after_sha256=seal_after.fingerprint_sha256,
        cache_snapshot_before={key: int(value) for key, value in before.items()},
        cache_snapshot_after={key: int(value) for key, value in after.items()},
        cache_miss_delta=cache_miss_delta,
        cache_entry_delta=cache_entry_delta,
        maximum_prepared_action_residual=float(maximum_prepared_residual),
        maximum_serial_oracle_residual=float(maximum_serial_residual),
    )


def classify_full_build_route(
    *,
    build_seconds: float,
    cache_bytes: int,
    maximum_correctness_residual: float,
    seal_unchanged: bool,
    cache_miss_delta: int,
    cache_entry_delta: int,
    full_matrix_measured: bool,
    wall_budget_seconds: float = 900.0,
    cache_budget_bytes: int = 2 * 1024**3,
) -> str:
    """Prospective route decision; never infer admission from a projection."""

    values = (
        float(build_seconds),
        float(maximum_correctness_residual),
        float(wall_budget_seconds),
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("timing, residual, and budget inputs must be finite/nonnegative")
    if int(cache_bytes) < 0 or int(cache_budget_bytes) < 0:
        raise ValueError("cache byte counts must be nonnegative")
    if not full_matrix_measured:
        return "EXPLICIT_CONSTRUCTION_NOT_YET_ADMISSIBLE"
    if not seal_unchanged or int(cache_miss_delta) != 0 or int(cache_entry_delta) != 0:
        return "PREPARED_STATE_INTEGRITY_FAILED"
    if maximum_correctness_residual > 5.0e-10:
        return "SAME_PHYSICS_EQUIVALENCE_FAILED"
    if build_seconds <= wall_budget_seconds and cache_bytes <= cache_budget_bytes:
        return "EXPLICIT_CALLBACK_CANDIDATE"
    return "MATRIX_FREE_OR_SPLIT_CANDIDATE"
