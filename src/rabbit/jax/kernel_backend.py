"""Kernel-backend metadata for the JAX/XLA-only runtime.

The current runtime exposes only JAX-native helper kernels. External Rust,
JAX-FFI, and wgpu helper backends were removed from the active code surface so
production metadata cannot accidentally report a host-mediated or custom-call
path as promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_ALIASES = {
    "auto": "jax",
    "jax": "jax",
    "xla": "jax",
    "jax_xla": "jax",
}
_REMOVED_BACKENDS = {
    "rust",
    "rust_cpu",
    "ffi",
    "rust_ffi",
    "jax_ffi",
    "gpu_rust",
    "rust_gpu",
    "rocm_rust",
    "gpu_rust_ffi",
    "wgpu",
}


@dataclass(frozen=True)
class KernelBackendReport:
    """Honest capability report for one helper-level kernel backend."""

    requested: str
    effective: str
    scope: str
    available: bool
    jit_safe: bool
    grad_safe: bool
    gpu_resident: bool
    host_mediated: bool
    fallback_reason: str = ""
    precision: str = "f64"
    adapter_info: Any = None

    def as_dict(self) -> dict[str, Any]:
        """Return stable metadata keys for inference/profiling payloads."""

        return {
            "kernel_backend_requested": self.requested,
            "kernel_backend_effective": self.effective,
            "kernel_backend_scope": self.scope,
            "kernel_backend_available": bool(self.available),
            "kernel_backend_jit_safe": bool(self.jit_safe),
            "kernel_backend_grad_safe": bool(self.grad_safe),
            "kernel_backend_gpu_resident": bool(self.gpu_resident),
            "kernel_backend_host_mediated": bool(self.host_mediated),
            "kernel_backend_fallback_reason": self.fallback_reason,
            "kernel_backend_precision": self.precision,
            "kernel_backend_adapter_info": self.adapter_info,
            "kernel_backend_external_call": False,
        }


def canonical_kernel_backend_name(name: str | None) -> str:
    """Normalize public backend aliases to the JAX-only surface."""

    key = "jax" if name is None else str(name).strip().lower()
    if key in _REMOVED_BACKENDS:
        raise ValueError(
            f"backend/kernel_backend={name!r} was removed; use 'jax' or 'auto'."
        )
    try:
        return _ALIASES[key]
    except KeyError as exc:
        allowed = "', '".join(sorted(_ALIASES))
        raise ValueError(
            f"unknown backend/kernel_backend={name!r}; choose '{allowed}'"
        ) from exc


def resolve_kernel_backend(backend: str = "jax", kernel_backend: str | None = None) -> str:
    """Resolve legacy ``backend`` and staged ``kernel_backend`` names."""

    resolved_backend = canonical_kernel_backend_name(backend)
    if kernel_backend is None:
        return resolved_backend
    resolved_kernel = canonical_kernel_backend_name(kernel_backend)
    if resolved_backend != "jax" and resolved_backend != resolved_kernel:
        raise ValueError(
            f"Conflicting backend={backend!r} and kernel_backend={kernel_backend!r}."
        )
    return resolved_kernel


def inspect_kernel_backend(
    kernel_backend: str | None = "jax",
    *,
    scope: str = "helper",
    require_available: bool = False,
) -> KernelBackendReport:
    """Return an honesty report for the JAX-only helper backend."""

    del require_available
    requested = "jax" if kernel_backend is None else str(kernel_backend)
    resolved = canonical_kernel_backend_name(kernel_backend)
    return KernelBackendReport(
        requested=requested,
        effective=resolved,
        scope=scope,
        available=True,
        jit_safe=True,
        grad_safe=True,
        gpu_resident=False,
        host_mediated=False,
    )


__all__ = [
    "KernelBackendReport",
    "canonical_kernel_backend_name",
    "inspect_kernel_backend",
    "resolve_kernel_backend",
]
