"""Primitive exact tangents for D-079.

The frozen comparator uses
    c = log(-log(1-f)),  q = df/dc = exp(c-exp(c)),
    u = log(f/(1-f)),   du/dc = exp(c)/f.
All chart variables are dimensionless. No clipping or projection is allowed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, log_expit

from rabbit.decoupling import _independent_noqke as ind

FloatArray = NDArray[np.float64]
EXPECTED_COMPARATOR_BLOB_SHA = "de44feee0aa484abe26976c7dc34c579643005b5"


class D079LinearizationError(RuntimeError):
    """Fail-closed error for an inadmissible static tangent."""


def matrix(name: str, value: ArrayLike, shape: tuple[int, ...]) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN/Inf")
    return array.copy()


def safe_relative(left: ArrayLike, right: ArrayLike) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    scale = max(
        float(np.max(np.abs(a), initial=0.0)),
        float(np.max(np.abs(b), initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    return float(np.max(np.abs(a - b), initial=0.0) / scale)


def cloglog_chart_tangent(
    pair_cloglog: ArrayLike,
    direction_cloglog: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(df, du, dlog(df/dc))`` for a c-direction."""

    c = np.asarray(pair_cloglog, dtype=np.float64)
    v = np.asarray(direction_cloglog, dtype=np.float64)
    if c.shape != v.shape or c.ndim == 0:
        raise ValueError("state and direction must share a nonempty shape")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(v)):
        raise ValueError("state/direction contains NaN/Inf")
    f = ind.cloglog_to_occupation(c)
    q = ind.cloglog_chain_factor(c)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        df = q * v
        du = np.exp(c) * v / f
        dlogq = (1.0 - np.exp(c)) * v
    if not all(np.all(np.isfinite(x)) for x in (df, du, dlogq)):
        raise D079LinearizationError("unrepresentable cloglog chart tangent")
    return tuple(np.asarray(x, dtype=np.float64) for x in (df, du, dlogq))


def pauli_gain_minus_loss_jvp(
    logit_1: ArrayLike,
    logit_2: ArrayLike,
    logit_3: ArrayLike,
    logit_4: ArrayLike,
    direction_1: ArrayLike,
    direction_2: ArrayLike,
    direction_3: ArrayLike,
    direction_4: ArrayLike,
) -> FloatArray:
    """Exact JVP of the stable Pauli factor.

    With P=gain-loss, affinity a=u3+u4-u1-u2 and
    L=f1*f2*(1-f3)*(1-f4),
        dP = P*d(log L) + gain*da.
    At detailed balance P=0, hence dP=gain*da.
    """

    raw = [
        np.asarray(x, dtype=np.float64)
        for x in (
            logit_1, logit_2, logit_3, logit_4,
            direction_1, direction_2, direction_3, direction_4,
        )
    ]
    try:
        shape = np.broadcast_shapes(*(x.shape for x in raw))
        u1, u2, u3, u4, du1, du2, du3, du4 = [
            np.broadcast_to(x, shape) for x in raw
        ]
    except ValueError as error:
        raise ValueError("Pauli inputs are not broadcast-compatible") from error
    if not all(np.all(np.isfinite(x)) for x in (u1,u2,u3,u4,du1,du2,du3,du4)):
        raise ValueError("Pauli inputs contain NaN/Inf")

    base = ind._stable_pauli_gain_minus_loss(u1, u2, u3, u4)
    f1, f2, f3, f4 = expit(u1), expit(u2), expit(u3), expit(u4)
    log_gain = (
        log_expit(-u1) + log_expit(-u2)
        + log_expit(u3) + log_expit(u4)
    )
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            gain = np.exp(log_gain)
            dlog_loss = (
                (1.0-f1)*du1 + (1.0-f2)*du2
                - f3*du3 - f4*du4
            )
            daffinity = du3 + du4 - du1 - du2
            result = base*dlog_loss + gain*daffinity
        except FloatingPointError as error:
            raise D079LinearizationError("unrepresentable Pauli tangent") from error
    if not np.all(np.isfinite(result)):
        raise D079LinearizationError("nonfinite Pauli tangent")
    return np.asarray(result, dtype=np.float64)


class TangentSpectralLogits:
    """Logit tangent interpolated on the comparator's exact modal basis."""

    def __init__(
        self,
        grid: ind.IndependentNoQkeGrid,
        pair_cloglog: ArrayLike,
        direction_cloglog: ArrayLike,
    ) -> None:
        c = matrix("pair_cloglog", pair_cloglog, (3, grid.order))
        v = matrix("direction_cloglog", direction_cloglog, c.shape)
        _df, du, _dlogq = cloglog_chart_tangent(c, v)
        self.grid = grid
        self.native_values = du
        self.coefficients = np.stack(
            [grid.modal_coefficients(row) for row in du]
        )

    def native(self, species: str) -> FloatArray:
        index = ind.PAIR_INDEX[ind._species_flavour(species)]
        return self.native_values[index]

    def at(self, species: str, y: ArrayLike) -> FloatArray:
        query = np.asarray(y, dtype=np.float64)
        if (
            not np.all(np.isfinite(query))
            or np.any(query < 0.0)
            or np.any(query > self.grid.y_max)
        ):
            raise D079LinearizationError("tangent query outside [0,y_max]")
        index = ind.PAIR_INDEX[ind._species_flavour(species)]
        result = self.grid.modal_basis(query) @ self.coefficients[index]
        if not np.all(np.isfinite(result)):
            raise D079LinearizationError("nonfinite interpolated tangent")
        return np.asarray(result, dtype=np.float64)


def modal_product(
    rates: FloatArray,
    y: FloatArray,
    grid: ind.IndependentNoQkeGrid,
) -> FloatArray:
    """Match the comparator's fixed-block modal contraction."""

    result = np.zeros((rates.shape[0], grid.order), dtype=np.float64)
    for start in range(0, rates.shape[1], ind._EVENT_BLOCK):
        stop = min(start + ind._EVENT_BLOCK, rates.shape[1])
        result += rates[:, start:stop] @ grid.modal_basis(y[start:stop])
    return result
