"""True directional-derivative certificates for the D-078 research lane.

The design adapts one narrow VigilODE discipline: an inexpensive or projected
signal may trigger a candidate, but success is granted only after checking the
original residual.  Here the original residual is the centered finite-
difference mismatch against a prospectively supplied JVP.  No Krylov solver or
VigilODE implementation is copied.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Hashable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
RhsFunction = Callable[[FloatArray], ArrayLike]
StateValidator = Callable[[FloatArray], bool]
BranchSignature = Callable[[FloatArray], Hashable]


class TangentStatus(str, Enum):
    CERTIFIED = "CERTIFIED"
    UNRESOLVED = "UNRESOLVED"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    BRANCH_CROSSING = "BRANCH_CROSSING"
    NONFINITE = "NONFINITE"


class TangentSampleStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    BRANCH_CROSSING = "BRANCH_CROSSING"
    NONFINITE = "NONFINITE"


@dataclass(frozen=True)
class TangentSample:
    epsilon: float
    status: TangentSampleStatus
    error_norm: float | None
    comparison_scale: float | None
    threshold: float | None
    normalized_residual: float | None


@dataclass(frozen=True)
class TangentCertificate:
    status: TangentStatus
    samples: tuple[TangentSample, ...]
    valid_samples: int
    passing_samples: int
    max_consecutive_passes: int
    best_normalized_residual: float
    rtol: float
    atol: float
    min_valid_samples: int
    required_consecutive_passes: int


def _vector(name: str, value: ArrayLike, *, length: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and array.size != length:
        raise ValueError(f"{name} has length {array.size}, expected {length}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array.copy()


def _safe_norm(value: FloatArray) -> float:
    scale = float(np.max(np.abs(value), initial=0.0))
    if scale == 0.0:
        return 0.0
    if not np.isfinite(scale):
        return float("nan")
    return scale * float(np.linalg.norm(value / scale))


def _evaluate(rhs: RhsFunction, state: FloatArray, size: int) -> FloatArray | None:
    try:
        value = np.asarray(rhs(state.copy()), dtype=float)
    except Exception:
        return None
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        return None
    return value.copy()


def _valid_state(validator: StateValidator | None, state: FloatArray) -> bool:
    if validator is None:
        return True
    try:
        return bool(validator(state.copy()))
    except Exception:
        return False


def _signature(signature: BranchSignature, state: FloatArray) -> Hashable | None:
    try:
        value = signature(state.copy())
        hash(value)
    except Exception:
        return None
    return value


def certify_directional_derivative(
    rhs: RhsFunction,
    state: ArrayLike,
    direction: ArrayLike,
    analytic_jvp: ArrayLike,
    *,
    epsilons: Iterable[float],
    rtol: float = 1.0e-6,
    atol: float = 1.0e-12,
    min_valid_samples: int = 3,
    required_consecutive_passes: int = 2,
    state_validator: StateValidator | None = None,
    branch_signature: BranchSignature | None = None,
) -> TangentCertificate:
    """Certify a JVP with a centered, same-branch epsilon ladder.

    Success requires both a minimum number of valid samples and a prospectively
    specified run of consecutive passing samples.  A single favorable epsilon
    is never sufficient.  Invalid shifted states and branch crossings are
    recorded rather than projected, clipped, or silently skipped into a pass.

    The caller's ``state`` and ``direction`` arrays are never passed directly to
    user callbacks, so even a hostile callback cannot mutate them.
    """

    x = _vector("state", state)
    v = _vector("direction", direction, length=x.size)
    jvp = _vector("analytic_jvp", analytic_jvp, length=x.size)
    if _safe_norm(v) == 0.0:
        raise ValueError("direction must be nonzero")
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be finite and nonnegative")
    if min_valid_samples <= 0 or required_consecutive_passes <= 0:
        raise ValueError("sample and consecutive-pass requirements must be positive")
    if required_consecutive_passes > min_valid_samples:
        raise ValueError("required_consecutive_passes cannot exceed min_valid_samples")

    epsilon_values = tuple(float(value) for value in epsilons)
    if not epsilon_values:
        raise ValueError("epsilons must not be empty")
    if any(not np.isfinite(value) or value <= 0.0 for value in epsilon_values):
        raise ValueError("every epsilon must be finite and positive")

    if not _valid_state(state_validator, x):
        raise ValueError("base state is outside the declared strict domain")
    base_rhs = _evaluate(rhs, x, x.size)
    if base_rhs is None:
        return TangentCertificate(
            status=TangentStatus.NONFINITE,
            samples=(),
            valid_samples=0,
            passing_samples=0,
            max_consecutive_passes=0,
            best_normalized_residual=float("inf"),
            rtol=float(rtol),
            atol=float(atol),
            min_valid_samples=min_valid_samples,
            required_consecutive_passes=required_consecutive_passes,
        )

    base_signature: Hashable | None = None
    if branch_signature is not None:
        base_signature = _signature(branch_signature, x)
        if base_signature is None:
            raise ValueError("base branch signature is invalid or unhashable")

    samples: list[TangentSample] = []
    valid_count = 0
    passing_count = 0
    consecutive = 0
    max_consecutive = 0
    best = float("inf")
    domain_failures = 0
    branch_failures = 0
    nonfinite_failures = 0

    for epsilon in epsilon_values:
        plus = x + epsilon * v
        minus = x - epsilon * v

        if not _valid_state(state_validator, plus) or not _valid_state(
            state_validator, minus
        ):
            domain_failures += 1
            consecutive = 0
            samples.append(
                TangentSample(
                    epsilon=epsilon,
                    status=TangentSampleStatus.OUT_OF_DOMAIN,
                    error_norm=None,
                    comparison_scale=None,
                    threshold=None,
                    normalized_residual=None,
                )
            )
            continue

        if branch_signature is not None:
            plus_signature = _signature(branch_signature, plus)
            minus_signature = _signature(branch_signature, minus)
            if (
                plus_signature is None
                or minus_signature is None
                or plus_signature != base_signature
                or minus_signature != base_signature
            ):
                branch_failures += 1
                consecutive = 0
                samples.append(
                    TangentSample(
                        epsilon=epsilon,
                        status=TangentSampleStatus.BRANCH_CROSSING,
                        error_norm=None,
                        comparison_scale=None,
                        threshold=None,
                        normalized_residual=None,
                    )
                )
                continue

        plus_rhs = _evaluate(rhs, plus, x.size)
        minus_rhs = _evaluate(rhs, minus, x.size)
        if plus_rhs is None or minus_rhs is None:
            nonfinite_failures += 1
            consecutive = 0
            samples.append(
                TangentSample(
                    epsilon=epsilon,
                    status=TangentSampleStatus.NONFINITE,
                    error_norm=None,
                    comparison_scale=None,
                    threshold=None,
                    normalized_residual=None,
                )
            )
            continue

        centered = (plus_rhs - minus_rhs) / (2.0 * epsilon)
        if not np.all(np.isfinite(centered)):
            nonfinite_failures += 1
            consecutive = 0
            samples.append(
                TangentSample(
                    epsilon=epsilon,
                    status=TangentSampleStatus.NONFINITE,
                    error_norm=None,
                    comparison_scale=None,
                    threshold=None,
                    normalized_residual=None,
                )
            )
            continue

        error_norm = _safe_norm(centered - jvp)
        comparison_scale = max(_safe_norm(centered), _safe_norm(jvp))
        threshold = float(atol + rtol * comparison_scale)
        normalized = error_norm / max(
            comparison_scale,
            atol,
            np.finfo(float).tiny,
        )
        if not all(
            np.isfinite(value)
            for value in (error_norm, comparison_scale, threshold, normalized)
        ):
            nonfinite_failures += 1
            consecutive = 0
            samples.append(
                TangentSample(
                    epsilon=epsilon,
                    status=TangentSampleStatus.NONFINITE,
                    error_norm=None,
                    comparison_scale=None,
                    threshold=None,
                    normalized_residual=None,
                )
            )
            continue

        valid_count += 1
        best = min(best, normalized)
        passed = error_norm <= threshold
        if passed:
            passing_count += 1
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
            sample_status = TangentSampleStatus.PASS
        else:
            consecutive = 0
            sample_status = TangentSampleStatus.FAIL
        samples.append(
            TangentSample(
                epsilon=epsilon,
                status=sample_status,
                error_norm=error_norm,
                comparison_scale=comparison_scale,
                threshold=threshold,
                normalized_residual=normalized,
            )
        )

    if (
        valid_count >= min_valid_samples
        and max_consecutive >= required_consecutive_passes
    ):
        status = TangentStatus.CERTIFIED
    elif nonfinite_failures:
        status = TangentStatus.NONFINITE
    elif valid_count < min_valid_samples and branch_failures:
        status = TangentStatus.BRANCH_CROSSING
    elif valid_count < min_valid_samples and domain_failures:
        status = TangentStatus.OUT_OF_DOMAIN
    else:
        status = TangentStatus.UNRESOLVED

    return TangentCertificate(
        status=status,
        samples=tuple(samples),
        valid_samples=valid_count,
        passing_samples=passing_count,
        max_consecutive_passes=max_consecutive,
        best_normalized_residual=best,
        rtol=float(rtol),
        atol=float(atol),
        min_valid_samples=min_valid_samples,
        required_consecutive_passes=required_consecutive_passes,
    )
