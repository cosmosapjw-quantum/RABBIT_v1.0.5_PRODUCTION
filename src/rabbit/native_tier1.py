"""Fail-closed private bridge to the Rust Tier-1 LRS core."""

from __future__ import annotations

import importlib

import numpy as np


class NativeTier1UnavailableError(RuntimeError):
    """Raised when explicitly requested native execution is unavailable."""


def _load_extension():
    return importlib.import_module("_rabbit_cpu")


def _vector(name: str, value, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or (size is not None and array.size != size):
        raise ValueError(f"native {name} must be a 1-D float64 array of length {size}")
    return np.ascontiguousarray(array)


def validate_native_outputs(derivative, monopole, aux, ray_work) -> None:
    """Validate raw native outputs without repairing or truncating them."""
    arrays = (derivative, monopole, aux, ray_work)
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise RuntimeError("native Tier-1 output contains a non-finite value")
    if aux[0] <= 0.0 or aux[1] <= 0.0:
        raise RuntimeError("native Tier-1 Hubble output must be positive")
    if np.any(monopole < 0.0) or np.any(monopole > 1.0):
        raise RuntimeError("native Tier-1 monopole lies outside [0, 1]")


class NativeTier1Workspace:
    """Python-owned immutable-grid workspace around ``NativeTier1Core``."""

    def __init__(self, *, ray_grid, momentum_grid, N_eff, f_nu,
                 ablate_hubble_anisotropy=False):
        mu0, w0, X0, signs = (_vector("ray grid", item) for item in ray_grid)
        q_nodes, q_weights = (_vector("momentum grid", item) for item in momentum_grid)
        if not (mu0.size == w0.size == X0.size == signs.size) or mu0.size == 0:
            raise ValueError("native ray-grid arrays must have one common nonzero length")
        if q_nodes.size != q_weights.size or q_nodes.size == 0:
            raise ValueError("native momentum-grid arrays must have one common nonzero length")
        try:
            extension = _load_extension()
        except Exception as exc:
            raise NativeTier1UnavailableError(
                "native Tier-1 extension is required but unavailable"
            ) from exc
        self.n_mu = int(mu0.size)
        self.n_q = int(q_nodes.size)
        self.core = extension.NativeTier1Core(
            mu0, w0, X0, signs, q_nodes, q_weights,
            float(N_eff), float(f_nu), bool(ablate_hubble_anisotropy),
        )

    def evaluate_into(self, state, derivative, monopole, aux, ray_work) -> None:
        state = _vector("state", state, 4)
        outputs = (("derivative", derivative, 4), ("monopole", monopole, self.n_q),
                   ("aux", aux, 9), ("ray work", ray_work, 5 * self.n_mu))
        for name, value, size in outputs:
            if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                    or value.ndim != 1 or value.size != size
                    or not value.flags.c_contiguous or not value.flags.writeable):
                raise ValueError(f"native {name} must be a writable C-contiguous float64[{size}]")
        arrays = (state,) + tuple(value for _name, value, _size in outputs)
        if any(np.shares_memory(left, right) for i, left in enumerate(arrays)
               for right in arrays[i + 1:]):
            raise ValueError("native state and output buffers must not alias")
        self.core.evaluate_into(state, derivative, monopole, aux, ray_work)
        validate_native_outputs(derivative, monopole, aux, ray_work)
