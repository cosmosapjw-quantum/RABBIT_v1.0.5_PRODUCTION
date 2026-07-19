"""Shared observable-name resolution for BBN inference code."""

from __future__ import annotations

from collections.abc import Mapping
from operator import index
from typing import Any

import numpy as np


OBSERVABLE_ALIASES = {
    "Yp": "Yp",
    "Y_p": "Yp",
    "yp": "Yp",
    "y_p": "Yp",
    "DH": "DH",
    "D/H": "DH",
    "D_H": "DH",
    "dh": "DH",
    "d/h": "DH",
}


BBN_JAX_SAMPLER_UNAVAILABLE = (
    "BBN_JAX_SAMPLER_UNAVAILABLE: host forward solver is not JAX-traceable; "
    "use host scalar/profile likelihood until B-05"
)


def resolve_observable_key(name: str) -> str:
    """Return the canonical prediction key for a public observation name."""
    key = OBSERVABLE_ALIASES.get(str(name))
    if key is None:
        raise ValueError(
            f"Unsupported BBN observation name {name!r}; "
            "supported names are Yp/Y_p and DH/D/H."
        )
    return key


def prediction_value_for_observation(pred: Any, obs: Any) -> float:
    """Return a prediction object's value for an observation."""
    return float(getattr(pred, resolve_observable_key(obs.name)))


def mapping_value_for_observation(pred: Mapping[str, Any], obs: Any) -> float:
    """Return a prediction mapping's value for an observation."""
    return float(pred[resolve_observable_key(obs.name)])


_MISSING = object()


def _prediction_field(pred: Any, key: str, default: Any = _MISSING) -> Any:
    if isinstance(pred, Mapping):
        return pred.get(key, default)
    return getattr(pred, key, default)


def _prediction_metadata(pred: Any) -> Mapping[str, Any]:
    metadata = _prediction_field(pred, "metadata", _MISSING)
    if metadata is _MISSING:
        return {}
    if not isinstance(metadata, Mapping):
        raise ValueError("Inference prediction metadata must be a mapping.")
    return metadata


def _reject_surrogate_prediction(pred: Any) -> None:
    surrogate = _prediction_metadata(pred).get("surrogate", False)
    if not isinstance(surrogate, (bool, np.bool_)):
        raise ValueError(
            "Inference prediction metadata.surrogate must be a scalar boolean."
        )
    if bool(surrogate):
        raise ValueError(
            "Inference received a SURROGATE prediction; use a canonical "
            "forward model for likelihood or posterior evaluation."
        )


def validate_prediction_for_inference(
    pred: Any,
    observation_names: Any = ("Yp", "DH"),
    *,
    extra_prediction_fields: Any = (),
) -> bool:
    """Validate one object/mapping before any inference observable is scored.

    Surrogates are a caller error and raise.  Failed, missing, non-scalar, or
    non-finite predictions return ``False`` so scalar likelihoods can fail
    closed with negative infinity.
    """
    _reject_surrogate_prediction(pred)
    success = _prediction_field(pred, "success", _MISSING)
    if success is _MISSING:
        return False
    try:
        success_arr = np.ma.asarray(success)
        success_masked = bool(np.ma.getmaskarray(success_arr))
    except (TypeError, ValueError):
        return False
    if success_masked or success_arr.ndim != 0:
        return False
    if not np.issubdtype(success_arr.dtype, np.bool_) or not bool(success_arr):
        return False
    required_fields = [resolve_observable_key(name) for name in observation_names]
    required_fields.extend(str(name) for name in extra_prediction_fields)
    for field in required_fields:
        value = _prediction_field(pred, field, _MISSING)
        if value is _MISSING:
            return False
        try:
            value_arr = np.ma.asarray(value)
            value_masked = bool(np.ma.getmaskarray(value_arr))
            value_is_real = (
                np.issubdtype(value_arr.dtype, np.number)
                and not np.issubdtype(value_arr.dtype, np.complexfloating)
            )
            value_finite = value_is_real and bool(np.isfinite(value_arr))
        except (TypeError, ValueError):
            return False
        if value_masked or value_arr.ndim != 0 or not value_finite:
            return False
    return True


def prediction_valid_mask_for_inference(
    pred: Any,
    observation_names: Any = ("Yp", "DH"),
    *,
    extra_prediction_fields: Any = (),
    expected_shape: Any = None,
    xp: Any = np,
) -> Any:
    """Return a validity mask for NumPy/JAX batch predictions.

    Without ``expected_shape`` this retains the legacy broadcast-based
    behaviour.  With a requested shape, scalar boolean status is the only
    value that may broadcast; status arrays, observables, and masked-array
    masks must otherwise match that shape exactly.
    """
    _reject_surrogate_prediction(pred)

    if expected_shape is not None:
        try:
            if isinstance(expected_shape, (int, np.integer)):
                batch_shape = (index(expected_shape),)
            else:
                batch_shape = tuple(index(size) for size in expected_shape)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("expected_shape must contain integer dimensions") from exc
        if any(size < 0 for size in batch_shape):
            raise ValueError("expected_shape dimensions must be non-negative")

        invalid = lambda: xp.zeros(batch_shape, dtype=bool)
        success = _prediction_field(pred, "success", _MISSING)
        if success is _MISSING:
            return invalid()

        success_mask = None
        if np.ma.isMaskedArray(success):
            success_mask = xp.asarray(np.ma.getmaskarray(success), dtype=bool)
            success = np.ma.getdata(success)
        try:
            success_arr = xp.asarray(success)
        except (TypeError, ValueError):
            return invalid()
        if not np.issubdtype(np.dtype(success_arr.dtype), np.bool_):
            return invalid()
        success_shape = tuple(success_arr.shape)
        if success_shape == ():
            if success_mask is not None:
                return invalid()
            valid = xp.ones(batch_shape, dtype=bool) & success_arr
        elif success_shape == batch_shape:
            valid = success_arr
            if success_mask is not None:
                if tuple(success_mask.shape) != batch_shape:
                    return invalid()
                valid = valid & ~success_mask
        else:
            return invalid()

        required_fields = [
            resolve_observable_key(name) for name in observation_names
        ]
        required_fields.extend(str(name) for name in extra_prediction_fields)
        for field in required_fields:
            value = _prediction_field(pred, field, _MISSING)
            if value is _MISSING:
                return invalid()
            value_mask = None
            if np.ma.isMaskedArray(value):
                value_mask = xp.asarray(np.ma.getmaskarray(value), dtype=bool)
                value = np.ma.getdata(value)
            try:
                value_arr = xp.asarray(value)
            except (TypeError, ValueError):
                return invalid()
            if tuple(value_arr.shape) != batch_shape:
                return invalid()
            value_dtype = np.dtype(value_arr.dtype)
            if (
                not np.issubdtype(value_dtype, np.number)
                or np.issubdtype(value_dtype, np.complexfloating)
            ):
                return invalid()
            if value_mask is not None and tuple(value_mask.shape) != batch_shape:
                return invalid()
            valid = valid & xp.isfinite(value_arr)
            if value_mask is not None:
                valid = valid & ~value_mask
        return valid

    success = _prediction_field(pred, "success", _MISSING)
    if success is _MISSING:
        return xp.asarray(False, dtype=bool)
    success_mask = None
    if np.ma.isMaskedArray(success):
        success_mask = xp.asarray(np.ma.getmaskarray(success), dtype=bool)
        success = np.ma.getdata(success)
    try:
        success_arr = xp.asarray(success)
    except (TypeError, ValueError):
        return xp.asarray(False, dtype=bool)
    if not np.issubdtype(np.dtype(success_arr.dtype), np.bool_):
        return xp.zeros_like(success_arr, dtype=bool)
    valid = success_arr
    if success_mask is not None:
        valid = valid & ~success_mask
    required_fields = [resolve_observable_key(name) for name in observation_names]
    required_fields.extend(str(name) for name in extra_prediction_fields)
    for field in required_fields:
        value = _prediction_field(pred, field, _MISSING)
        if value is _MISSING:
            return xp.zeros_like(valid, dtype=bool)
        value_mask = None
        if np.ma.isMaskedArray(value):
            value_mask = xp.asarray(np.ma.getmaskarray(value), dtype=bool)
            value = np.ma.getdata(value)
        try:
            value_arr = xp.asarray(value)
        except (TypeError, ValueError):
            return xp.zeros_like(valid, dtype=bool)
        value_dtype = np.dtype(value_arr.dtype)
        if (
            not np.issubdtype(value_dtype, np.number)
            or np.issubdtype(value_dtype, np.complexfloating)
        ):
            return xp.zeros_like(valid, dtype=bool)
        finite = xp.isfinite(value_arr)
        valid = valid & finite
        if value_mask is not None:
            valid = valid & ~value_mask
    return valid


def _safe_real_scalar_for_inference(value: Any, fallback: float, *, xp: Any) -> Any:
    """Return a real scalar for log-density arithmetic after validity checks."""
    if np.ma.isMaskedArray(value):
        value = np.ma.getdata(value)
    try:
        value_arr = xp.asarray(value)
    except (TypeError, ValueError):
        return xp.asarray(fallback, dtype=float)
    value_dtype = np.dtype(value_arr.dtype)
    if (
        value_arr.ndim != 0
        or not np.issubdtype(value_dtype, np.number)
        or np.issubdtype(value_dtype, np.complexfloating)
    ):
        return xp.asarray(fallback, dtype=float)
    return xp.asarray(value_arr, dtype=float)


__all__ = [
    "BBN_JAX_SAMPLER_UNAVAILABLE",
    "OBSERVABLE_ALIASES",
    "resolve_observable_key",
    "prediction_value_for_observation",
    "mapping_value_for_observation",
    "validate_prediction_for_inference",
    "prediction_valid_mask_for_inference",
]
