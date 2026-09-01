"""Exact piecewise logit push-forward for the D-078 research lane.

This module is deliberately research-only.  It does not import or call the
private comparator, does not integrate a trajectory, and does not alter the
D-069/D-071 evidence.  Its sole authority is the algebra frozen in
BD622_D078_ANALYTIC_JACOBIAN_RESEARCH_CONTRACT_2026-09-01.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class LinearizationError(ValueError):
    """Invalid or uncertifiable logit linearization input."""


class LinearizationKinkError(LinearizationError):
    """The effective chain is nondifferentiable at the requested state."""


def _readonly_float(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=float).copy()
    array.setflags(write=False)
    return array


def _readonly_bool(value: ArrayLike) -> BoolArray:
    array = np.asarray(value, dtype=bool).copy()
    array.setflags(write=False)
    return array


def _vector(name: str, value: ArrayLike, *, length: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise LinearizationError(f"{name} must be a one-dimensional array")
    if length is not None and array.size != length:
        raise LinearizationError(
            f"{name} has length {array.size}, expected {length}"
        )
    if not np.all(np.isfinite(array)):
        raise LinearizationError(f"{name} contains NaN or infinity")
    return array.copy()


def _matrix(name: str, value: ArrayLike, *, size: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size, size):
        raise LinearizationError(
            f"{name} has shape {array.shape}, expected {(size, size)}"
        )
    if not np.all(np.isfinite(array)):
        raise LinearizationError(f"{name} contains NaN or infinity")
    return array.copy()


@dataclass(frozen=True)
class LogitChainTerms:
    """Piecewise chain data at one occupation state.

    ``raw_chain`` is always ``f(1-f)`` and is used on the input side of
    ``J_f D``.  ``effective_chain`` is ``max(raw_chain, floor)`` and is used
    only as the transformed-RHS denominator.  At an exact floor kink the
    derivative is undefined and represented by NaN; all logit-direction
    push-forwards refuse such a state.
    """

    occupation: FloatArray
    raw_chain: FloatArray
    effective_chain: FloatArray
    derivative_wrt_logit: FloatArray
    floor_active: BoolArray
    kink: BoolArray
    floor: float


@dataclass(frozen=True)
class LogitJacobianResult:
    jacobian: FloatArray
    correction_diagonal: FloatArray
    terms: LogitChainTerms


@dataclass(frozen=True)
class LogitJvpResult:
    jvp: FloatArray
    correction: FloatArray
    terms: LogitChainTerms


@dataclass(frozen=True)
class LogitAuxiliaryColumnResult:
    column: FloatArray
    terms: LogitChainTerms


def logit_chain_terms(
    occupation: ArrayLike,
    *,
    floor: float = 1.0e-12,
    kink_rtol: float = 128.0 * np.finfo(float).eps,
) -> LogitChainTerms:
    """Return the exact branch data for ``D_eff=max(f(1-f), floor)``.

    Parameters are checked fail-closed.  The physical occupation domain is
    strict: every component must lie in ``(0,1)``.  ``floor=1/4`` is allowed
    so the nondifferentiable point can be tested exactly; larger floors are not
    meaningful for a logistic chain.
    """

    f = _vector("occupation", occupation)
    if f.size == 0:
        raise LinearizationError("occupation must not be empty")
    if not np.all((f > 0.0) & (f < 1.0)):
        raise LinearizationError("occupation must lie strictly inside (0,1)")
    if not np.isfinite(floor) or not (0.0 < floor <= 0.25):
        raise LinearizationError("floor must be finite and lie in (0, 1/4]")
    if not np.isfinite(kink_rtol) or kink_rtol < 0.0:
        raise LinearizationError("kink_rtol must be finite and nonnegative")

    raw = f * (1.0 - f)
    effective = np.maximum(raw, floor)
    active = raw < floor
    kink = np.isclose(raw, floor, rtol=kink_rtol, atol=0.0)
    derivative = np.where(raw > floor, raw * (1.0 - 2.0 * f), 0.0)
    derivative = derivative.astype(float, copy=False)
    derivative[kink] = np.nan

    return LogitChainTerms(
        occupation=_readonly_float(f),
        raw_chain=_readonly_float(raw),
        effective_chain=_readonly_float(effective),
        derivative_wrt_logit=_readonly_float(derivative),
        floor_active=_readonly_bool(active),
        kink=_readonly_bool(kink),
        floor=float(floor),
    )


def _require_differentiable(terms: LogitChainTerms) -> None:
    indices = np.flatnonzero(terms.kink)
    if indices.size:
        rendered = ", ".join(str(int(index)) for index in indices)
        raise LinearizationKinkError(
            "effective logit chain is nondifferentiable at indices " + rendered
        )


def push_forward_occupation_jacobian(
    occupation: ArrayLike,
    occupation_rhs: ArrayLike,
    occupation_jacobian: ArrayLike,
    *,
    floor: float = 1.0e-12,
) -> LogitJacobianResult:
    """Push an occupation-space Jacobian into the piecewise logit chart.

    Away from the floor kink this evaluates

    ``E^-1 J_f D - diag(F * dE/dz / E^2)``

    exactly in binary64 operation order.  It never substitutes ``E`` for the
    input-side raw logistic chain ``D``.
    """

    terms = logit_chain_terms(occupation, floor=floor)
    _require_differentiable(terms)
    size = terms.occupation.size
    rhs = _vector("occupation_rhs", occupation_rhs, length=size)
    jacobian_f = _matrix("occupation_jacobian", occupation_jacobian, size=size)

    transformed = (
        jacobian_f * terms.raw_chain[np.newaxis, :]
    ) / terms.effective_chain[:, np.newaxis]
    correction_diagonal = (
        rhs
        * terms.derivative_wrt_logit
        / np.square(terms.effective_chain)
    )
    transformed[np.diag_indices(size)] -= correction_diagonal
    if not np.all(np.isfinite(transformed)):
        raise LinearizationError("logit Jacobian contains NaN or infinity")

    return LogitJacobianResult(
        jacobian=_readonly_float(transformed),
        correction_diagonal=_readonly_float(correction_diagonal),
        terms=terms,
    )


def push_forward_occupation_jvp(
    occupation: ArrayLike,
    occupation_rhs: ArrayLike,
    occupation_jvp: ArrayLike,
    logit_direction: ArrayLike,
    *,
    floor: float = 1.0e-12,
) -> LogitJvpResult:
    """Push ``J_f (D v_z)`` into the piecewise logit chart.

    ``occupation_jvp`` must already be the occupation-space directional
    derivative in direction ``D v_z``.  Keeping that contract explicit avoids
    silently applying the logistic chain twice.
    """

    terms = logit_chain_terms(occupation, floor=floor)
    _require_differentiable(terms)
    size = terms.occupation.size
    rhs = _vector("occupation_rhs", occupation_rhs, length=size)
    jvp_f = _vector("occupation_jvp", occupation_jvp, length=size)
    direction = _vector("logit_direction", logit_direction, length=size)

    correction = (
        rhs
        * terms.derivative_wrt_logit
        * direction
        / np.square(terms.effective_chain)
    )
    transformed = jvp_f / terms.effective_chain - correction
    if not np.all(np.isfinite(transformed)):
        raise LinearizationError("logit JVP contains NaN or infinity")

    return LogitJvpResult(
        jvp=_readonly_float(transformed),
        correction=_readonly_float(correction),
        terms=terms,
    )


def push_forward_auxiliary_column(
    occupation: ArrayLike,
    partial_occupation_rhs: ArrayLike,
    *,
    floor: float = 1.0e-12,
) -> LogitAuxiliaryColumnResult:
    """Push a non-chart auxiliary derivative into the logit RHS.

    The auxiliary variable is assumed not to redefine ``f=sigmoid(z)`` or the
    floor.  Its column is therefore divided by ``D_eff`` exactly once.  Unlike
    a logit-direction derivative this column remains defined at a floor kink.
    """

    terms = logit_chain_terms(occupation, floor=floor)
    size = terms.occupation.size
    partial = _vector(
        "partial_occupation_rhs", partial_occupation_rhs, length=size
    )
    column = partial / terms.effective_chain
    if not np.all(np.isfinite(column)):
        raise LinearizationError("logit auxiliary column contains NaN or infinity")
    return LogitAuxiliaryColumnResult(column=_readonly_float(column), terms=terms)
