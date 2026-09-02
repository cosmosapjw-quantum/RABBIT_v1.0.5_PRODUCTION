"""Cancellation-aware equivalence metric for an explicit static Jacobian action.

For a matrix ``J`` and direction ``v``, ordinary forward relative error
``||Jv-r||/max(||Jv||,||r||)`` is ill-conditioned when physically large column
contributions cancel in ``Jv``.  This module additionally uses the standard
componentwise contribution scale ``|J| |v|``.  The scale does not hide a wrong
basis column: for a Euclidean basis direction it reduces to the norm of that
column.  Spectral, photon-temperature, and elapsed-output blocks are treated
separately because they carry different dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MatrixActionBlockResidual:
    """Dimension-aware backward-stable report for ``J @ v``."""

    maximum: float
    spectral: float
    temperature: float
    elapsed: float
    spectral_difference_norm: float
    spectral_forward_scale: float
    spectral_contribution_scale: float
    temperature_absolute_difference: float
    temperature_forward_scale: float
    temperature_contribution_scale: float
    elapsed_absolute_difference: float
    elapsed_forward_scale: float
    elapsed_contribution_scale: float


def _finite_vector(name: str, value: ArrayLike, size: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return array


def matrix_action_block_residual(
    *,
    jacobian: ArrayLike,
    direction: ArrayLike,
    reference_action: ArrayLike,
    order: int,
) -> MatrixActionBlockResidual:
    """Compare an explicit matrix action with an independent directional JVP.

    For each native-dimensional output block, the denominator is

    ``max(||Jv||, ||reference||, || |J| |v| ||, tiny)``.

    The third term is a contribution/backward-error scale.  It is required for
    dense directions whose individually large column contributions cancel.  A
    basis direction has ``|J| |e_j| = |J[:,j]|``, so selected-column admission
    is not weakened.
    """

    n = int(order)
    if n <= 0:
        raise ValueError("order must be positive")
    size = 3 * n + 2
    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"jacobian must be a finite {size} x {size} matrix")
    vector = _finite_vector("direction", direction, size)
    reference = _finite_vector("reference_action", reference_action, size)

    candidate = np.asarray(matrix @ vector, dtype=np.float64)
    contribution = np.asarray(np.abs(matrix) @ np.abs(vector), dtype=np.float64)
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(contribution)):
        raise ValueError("matrix action or contribution scale is nonfinite")

    spectral_size = 3 * n
    tiny = np.finfo(np.float64).tiny

    spectral_difference = float(
        np.linalg.norm(candidate[:spectral_size] - reference[:spectral_size])
    )
    spectral_forward_scale = max(
        float(np.linalg.norm(candidate[:spectral_size])),
        float(np.linalg.norm(reference[:spectral_size])),
    )
    spectral_contribution_scale = float(
        np.linalg.norm(contribution[:spectral_size])
    )
    spectral = spectral_difference / max(
        spectral_forward_scale,
        spectral_contribution_scale,
        tiny,
    )

    temperature_difference = float(abs(candidate[-2] - reference[-2]))
    temperature_forward_scale = max(abs(float(candidate[-2])), abs(float(reference[-2])))
    temperature_contribution_scale = abs(float(contribution[-2]))
    temperature = temperature_difference / max(
        temperature_forward_scale,
        temperature_contribution_scale,
        tiny,
    )

    elapsed_difference = float(abs(candidate[-1] - reference[-1]))
    elapsed_forward_scale = max(abs(float(candidate[-1])), abs(float(reference[-1])))
    elapsed_contribution_scale = abs(float(contribution[-1]))
    elapsed = elapsed_difference / max(
        elapsed_forward_scale,
        elapsed_contribution_scale,
        tiny,
    )

    values = (
        spectral,
        temperature,
        elapsed,
        spectral_difference,
        spectral_forward_scale,
        spectral_contribution_scale,
        temperature_difference,
        temperature_forward_scale,
        temperature_contribution_scale,
        elapsed_difference,
        elapsed_forward_scale,
        elapsed_contribution_scale,
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("action residual report is nonfinite or negative")

    return MatrixActionBlockResidual(
        maximum=float(max(spectral, temperature, elapsed)),
        spectral=float(spectral),
        temperature=float(temperature),
        elapsed=float(elapsed),
        spectral_difference_norm=spectral_difference,
        spectral_forward_scale=float(spectral_forward_scale),
        spectral_contribution_scale=spectral_contribution_scale,
        temperature_absolute_difference=temperature_difference,
        temperature_forward_scale=float(temperature_forward_scale),
        temperature_contribution_scale=float(temperature_contribution_scale),
        elapsed_absolute_difference=elapsed_difference,
        elapsed_forward_scale=float(elapsed_forward_scale),
        elapsed_contribution_scale=float(elapsed_contribution_scale),
    )
