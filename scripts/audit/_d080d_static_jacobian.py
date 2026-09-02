"""Square static Jacobian of the frozen private comparator RHS.

D-080D assembles the already admitted derivative pieces in the packed state
ordering

    Y = (c_e[0:n], c_mu[0:n], c_tau[0:n], T_gamma, elapsed_time).

The first ``3*n`` columns are exact D-079 spectral-direction JVPs evaluated on
basis vectors.  The next column is the D-080C fixed-support ``T_gamma`` RHS
column.  The final column is exactly zero because the frozen static RHS does
not depend on stored elapsed time.

This is a research-only static operator.  It does not call an integrator,
construct a BDF step, claim a speedup, cross a support branch, or move a
physics gate.  The comparator uses natural units ``hbar=c=k_B=1``.  Packed
state and RHS blocks have heterogeneous dimensions, so validation residuals
must normalize spectral, photon-temperature, and elapsed-output blocks
separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_rhs_jvp import (
    RhsJvpResult,
    evaluate_c_only_rhs_jvp,
)
from scripts.audit._d079_tangent_primitives import matrix
from scripts.audit._d080c_tgamma_rhs import (
    TgammaRhsColumnResult,
    evaluate_tgamma_rhs_column,
)

FloatArray = NDArray[np.float64]
EXPECTED_COMPARATOR_BLOB_SHA = "de44feee0aa484abe26976c7dc34c579643005b5"
EXPECTED_D079_RHS_BLOB_SHA = "6bcff2bc5627c0af0ad4df61c908d09e62ffaba5"
EXPECTED_D080C_RHS_BLOB_SHA = "c18feacbd57c9519af14504027b7d465758eb1ef"


class D080DStaticJacobianError(RuntimeError):
    """Fail-closed error for an invalid square static-Jacobian assembly."""


@dataclass(frozen=True)
class StaticJacobianLayout:
    """Machine-readable ordering and dimensional contract."""

    state_labels: tuple[str, ...]
    rhs_labels: tuple[str, ...]
    state_dimensions: tuple[str, ...]
    rhs_dimensions: tuple[str, ...]
    spectral_slice: slice
    temperature_index: int
    elapsed_index: int


@dataclass(frozen=True)
class StaticDirectionalJvpResult:
    """Exact combined static directional derivative before matrix assembly."""

    base_rhs: FloatArray
    jvp: FloatArray
    full_direction: FloatArray
    spectral_component: FloatArray
    tgamma_component: FloatArray
    elapsed_component: FloatArray
    tgamma_direction: float
    elapsed_direction: float
    branch_signature: str
    base_reconstruction_residual: float
    spectral: RhsJvpResult
    tgamma: TgammaRhsColumnResult


@dataclass(frozen=True)
class StaticJacobianResult:
    """Explicit square Jacobian and the independent source columns."""

    base_rhs: FloatArray
    jacobian: FloatArray
    active_jacobian: FloatArray
    spectral_columns: FloatArray
    tgamma_column: FloatArray
    elapsed_time_column: FloatArray
    state_size: int
    spectral_size: int
    layout: StaticJacobianLayout
    branch_signature: str
    base_reconstruction_residual: float
    column_assembly_residual: float
    tgamma: TgammaRhsColumnResult


def _finite_vector(name: str, value: ArrayLike, size: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise D080DStaticJacobianError(
            f"{name} must be a finite vector of length {size}"
        )
    return result.copy()


def _finite_matrix(
    name: str,
    value: ArrayLike,
    shape: tuple[int, int],
) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise D080DStaticJacobianError(
            f"{name} must be a finite matrix of shape {shape}"
        )
    return result.copy()


def _relative(left: ArrayLike, right: ArrayLike) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("relative-residual operands must share a shape")
    scale = max(
        float(np.linalg.norm(a)),
        float(np.linalg.norm(b)),
        np.finfo(np.float64).tiny,
    )
    return float(np.linalg.norm(a - b) / scale)


def _scalar_relative(left: float, right: float) -> float:
    a = float(left)
    b = float(right)
    scale = max(abs(a), abs(b), np.finfo(np.float64).tiny)
    return float(abs(a - b) / scale)


def rhs_block_relative(
    left: ArrayLike,
    right: ArrayLike,
    order: int,
) -> float:
    """Dimension-aware residual for packed static RHS vectors.

    The three blocks have dimensions ``1``, ``MeV``, and ``MeV^-1`` before an
    input derivative is applied.  Combining them in one Euclidean norm can
    hide a wrong spectral or temperature block behind the large elapsed-time
    row, so each block is normalized independently.
    """

    size = 3 * int(order) + 2
    a = _finite_vector("left packed RHS", left, size)
    b = _finite_vector("right packed RHS", right, size)
    spectral_size = 3 * int(order)
    return max(
        _relative(a[:spectral_size], b[:spectral_size]),
        _scalar_relative(a[-2], b[-2]),
        _scalar_relative(a[-1], b[-1]),
    )


def static_jacobian_layout(order: int) -> StaticJacobianLayout:
    """Return the exact packed ordering and block dimensions."""

    n = int(order)
    if n <= 0:
        raise ValueError("order must be positive")
    labels = tuple(
        f"c_{flavour}[{index}]"
        for flavour in ("e", "mu", "tau")
        for index in range(n)
    )
    state_labels = labels + ("T_gamma", "elapsed_time")
    rhs_labels = tuple(f"d{label}/dN" for label in labels) + (
        "dT_gamma/dN",
        "dt_elapsed/dN",
    )
    state_dimensions = ("1",) * (3 * n) + ("MeV", "MeV^-1")
    rhs_dimensions = ("1",) * (3 * n) + ("MeV", "MeV^-1")
    return StaticJacobianLayout(
        state_labels=state_labels,
        rhs_labels=rhs_labels,
        state_dimensions=state_dimensions,
        rhs_dimensions=rhs_dimensions,
        spectral_slice=slice(0, 3 * n),
        temperature_index=3 * n,
        elapsed_index=3 * n + 1,
    )


def evaluate_static_rhs_direction_jvp(
    *,
    grid: ind.IndependentNoQkeGrid,
    pair_cloglog: ArrayLike,
    full_direction: ArrayLike,
    temperature_cm_mev: float,
    temperature_gamma_mev: float,
    config: ind.IndependentCollisionConfig = ind.IndependentCollisionConfig(),
    electron_mass_mev: float = ind.M_ELECTRON_MEV,
) -> StaticDirectionalJvpResult:
    """Combine admitted spectral and ``T_gamma`` derivatives for one direction.

    This path is deliberately independent of explicit basis-column assembly.
    It is therefore the direct oracle used to verify ``J @ v``.  The stored
    elapsed-time direction is retained in the receipt but contributes exactly
    zero to the derivative.
    """

    c = matrix("pair_cloglog", pair_cloglog, (3, grid.order))
    spectral_size = 3 * grid.order
    size = spectral_size + 2
    direction = _finite_vector("full_direction", full_direction, size)
    spectral_direction = direction[:spectral_size].reshape(3, grid.order)

    spectral = evaluate_c_only_rhs_jvp(
        grid=grid,
        pair_cloglog=c,
        direction_cloglog=spectral_direction,
        temperature_cm_mev=float(temperature_cm_mev),
        temperature_gamma_mev=float(temperature_gamma_mev),
        config=config,
        electron_mass_mev=float(electron_mass_mev),
    )
    tgamma = evaluate_tgamma_rhs_column(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=float(temperature_cm_mev),
        temperature_gamma_mev=float(temperature_gamma_mev),
        config=config,
        electron_mass_mev=float(electron_mass_mev),
    )

    base_residual = rhs_block_relative(
        spectral.base_rhs,
        tgamma.base_rhs,
        grid.order,
    )
    if base_residual > 5.0e-13:
        raise D080DStaticJacobianError(
            "D-079 and D-080C base RHS values do not share one operator"
        )

    spectral_component = _finite_vector(
        "spectral directional component", spectral.jvp, size
    )
    tgamma_component = _finite_vector(
        "T_gamma directional component",
        direction[-2] * tgamma.tgamma_column,
        size,
    )
    elapsed_component = np.zeros(size, dtype=np.float64)
    jvp = _finite_vector(
        "combined static directional JVP",
        spectral_component + tgamma_component + elapsed_component,
        size,
    )
    return StaticDirectionalJvpResult(
        base_rhs=_finite_vector("base RHS", tgamma.base_rhs, size),
        jvp=jvp,
        full_direction=direction,
        spectral_component=spectral_component,
        tgamma_component=tgamma_component,
        elapsed_component=elapsed_component,
        tgamma_direction=float(direction[-2]),
        elapsed_direction=float(direction[-1]),
        branch_signature=tgamma.branch_signature,
        base_reconstruction_residual=float(
            max(base_residual, tgamma.base_reconstruction_residual)
        ),
        spectral=spectral,
        tgamma=tgamma,
    )


def assemble_static_jacobian(
    *,
    grid: ind.IndependentNoQkeGrid,
    pair_cloglog: ArrayLike,
    temperature_cm_mev: float,
    temperature_gamma_mev: float,
    config: ind.IndependentCollisionConfig = ind.IndependentCollisionConfig(),
    electron_mass_mev: float = ind.M_ELECTRON_MEV,
) -> StaticJacobianResult:
    """Assemble the explicit ``(3*n+2) x (3*n+2)`` static Jacobian."""

    c = matrix("pair_cloglog", pair_cloglog, (3, grid.order))
    layout = static_jacobian_layout(grid.order)
    spectral_size = 3 * grid.order
    size = spectral_size + 2
    tgamma = evaluate_tgamma_rhs_column(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=float(temperature_cm_mev),
        temperature_gamma_mev=float(temperature_gamma_mev),
        config=config,
        electron_mass_mev=float(electron_mass_mev),
    )
    base_rhs = _finite_vector("base RHS", tgamma.base_rhs, size)

    spectral_columns = np.empty((size, spectral_size), dtype=np.float64)
    maximum_base_residual = float(tgamma.base_reconstruction_residual)
    basis = np.zeros((3, grid.order), dtype=np.float64)
    flat_basis = basis.ravel()
    for column in range(spectral_size):
        flat_basis.fill(0.0)
        flat_basis[column] = 1.0
        result = evaluate_c_only_rhs_jvp(
            grid=grid,
            pair_cloglog=c,
            direction_cloglog=basis,
            temperature_cm_mev=float(temperature_cm_mev),
            temperature_gamma_mev=float(temperature_gamma_mev),
            config=config,
            electron_mass_mev=float(electron_mass_mev),
        )
        base_residual = rhs_block_relative(result.base_rhs, base_rhs, grid.order)
        if base_residual > 5.0e-13:
            raise D080DStaticJacobianError(
                f"spectral basis column {column} uses a different base RHS"
            )
        spectral_columns[:, column] = result.jvp
        maximum_base_residual = max(maximum_base_residual, base_residual)

    spectral_columns = _finite_matrix(
        "spectral Jacobian columns",
        spectral_columns,
        (size, spectral_size),
    )
    jacobian = np.zeros((size, size), dtype=np.float64)
    jacobian[:, :spectral_size] = spectral_columns
    jacobian[:, layout.temperature_index] = tgamma.tgamma_column
    jacobian[:, layout.elapsed_index] = tgamma.elapsed_time_input_column
    jacobian = _finite_matrix("square static Jacobian", jacobian, (size, size))

    elapsed_column = _finite_vector(
        "elapsed-time input column",
        jacobian[:, layout.elapsed_index],
        size,
    )
    if np.any(elapsed_column != 0.0):
        raise D080DStaticJacobianError(
            "elapsed-time input column is not exact structural zero"
        )

    assembly_residual = max(
        _relative(jacobian[:, :spectral_size], spectral_columns),
        _relative(jacobian[:, layout.temperature_index], tgamma.tgamma_column),
        float(np.linalg.norm(elapsed_column)),
    )
    if not np.isfinite(assembly_residual):
        raise D080DStaticJacobianError("nonfinite column-assembly residual")

    # The final stored-time variable is passive.  Removing its input column and
    # accumulator output row leaves the dynamically active (c,T_gamma) block.
    active_jacobian = _finite_matrix(
        "active static Jacobian",
        jacobian[:-1, :-1],
        (size - 1, size - 1),
    )
    return StaticJacobianResult(
        base_rhs=base_rhs,
        jacobian=jacobian,
        active_jacobian=active_jacobian,
        spectral_columns=spectral_columns,
        tgamma_column=_finite_vector(
            "T_gamma input column", tgamma.tgamma_column, size
        ),
        elapsed_time_column=elapsed_column,
        state_size=size,
        spectral_size=spectral_size,
        layout=layout,
        branch_signature=tgamma.branch_signature,
        base_reconstruction_residual=float(maximum_base_residual),
        column_assembly_residual=float(assembly_residual),
        tgamma=tgamma,
    )


def static_newton_matrix(jacobian: ArrayLike, gamma: float) -> FloatArray:
    """Return ``I-gamma*J`` without invoking any nonlinear or ODE solver.

    The helper exists only to expose the passive-accumulator block identity.
    For a valid D-080D matrix the final column of this Newton matrix is the
    final Euclidean basis vector, so its determinant equals that of the active
    block.  No conditioning or solver-performance claim follows.
    """

    matrix_value = np.asarray(jacobian, dtype=np.float64)
    if (
        matrix_value.ndim != 2
        or matrix_value.shape[0] != matrix_value.shape[1]
        or not np.all(np.isfinite(matrix_value))
    ):
        raise ValueError("jacobian must be a finite square matrix")
    coefficient = float(gamma)
    if not np.isfinite(coefficient):
        raise ValueError("gamma must be finite")
    result = np.eye(matrix_value.shape[0], dtype=np.float64) - coefficient * matrix_value
    return _finite_matrix("static Newton matrix", result, matrix_value.shape)
