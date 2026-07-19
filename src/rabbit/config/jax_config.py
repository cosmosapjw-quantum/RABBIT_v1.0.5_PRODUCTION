"""
rabbit.config.jax_config — JAX solver backend configuration.

The JAX Rodas5P solver is a frozen numerical-method and parity oracle. Rust AOT
is the active implementation target and SciPy/BDF is the temporary
number-of-record. This module retains the method enum, explicit compatibility
dispatch, and configuration presets needed by existing JAX regression checks.

The JAX backends require jax/jaxlib.  If not installed, the SciPy
validation path is used and JAX methods raise ImportError at call time
(not at import time).

Usage
-----
    from rabbit.config.jax_config import JAXSolverMethod, is_jax_backend

    method = JAXSolverMethod.JAX_RODAS5P
    assert is_jax_backend(method)

    # The unified solve_ode() dispatcher in solver_config.py routes here:
    from rabbit.config.solver_config import solve_ode
    result = solve_ode(rhs, y0, t_span, method=JAXSolverMethod.JAX_RODAS5P, ...)
"""
from __future__ import annotations

import os
import glob
from enum import Enum
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# §-1. Stale accelerator plugin guard
# ═══════════════════════════════════════════════════════════════════════

def local_accelerator_device_hint_available() -> bool:
    """Return whether this process can see a plausible local GPU device node.

    JAX accelerator plugin packages may be installed in a CPU-only container.
    In that state, the first backend query can fail before application-level
    retry logic has a chance to select the CPU backend.  This probe deliberately
    uses OS-level device hints and does not import JAX.
    """

    if os.path.exists("/dev/kfd"):
        return True
    if os.path.exists("/dev/nvidiactl") or os.path.exists("/dev/dxg"):
        return True
    if glob.glob("/dev/nvidia[0-9]*"):
        return True
    return False


def maybe_pin_jax_platform_to_cpu_without_accelerator(
    *,
    reason: str = "",
) -> dict:
    """Set ``JAX_PLATFORMS=cpu`` before JAX import when no GPU is visible.

    The guard is intentionally conservative:
      - explicit ``JAX_PLATFORMS`` / ``JAX_PLATFORM_NAME`` is never overridden;
      - ``RABBIT_JAX_ASSUME_ACCELERATOR=1`` disables the guard;
      - no JAX module is imported here.
    """

    explicit = os.environ.get("JAX_PLATFORMS") or os.environ.get("JAX_PLATFORM_NAME")
    if explicit:
        return {
            "applied": False,
            "reason": "explicit_jax_platform",
            "platforms": explicit,
        }
    assume_accelerator = os.environ.get("RABBIT_JAX_ASSUME_ACCELERATOR", "").strip().lower()
    if assume_accelerator in ("1", "true", "yes", "on"):
        return {
            "applied": False,
            "reason": "assume_accelerator_env",
            "platforms": "",
        }
    if local_accelerator_device_hint_available():
        return {
            "applied": False,
            "reason": "accelerator_device_hint_present",
            "platforms": "",
        }
    os.environ["JAX_PLATFORMS"] = "cpu"
    return {
        "applied": True,
        "reason": str(reason or "no_local_accelerator_device_hint"),
        "platforms": "cpu",
    }


# ═══════════════════════════════════════════════════════════════════════
# §0. AOT compilation cache (OPT-10)
# ═══════════════════════════════════════════════════════════════════════

def resolve_writable_cache_dir(cache_dir: str = "/tmp/rabbit_jax_cache") -> str:
    """Return a JAX compilation-cache directory writable by this process."""
    fallback = "/tmp/rabbit_jax_cache"
    candidates: list[str] = []
    for candidate in (cache_dir, os.environ.get("RABBIT_JAX_CACHE_DIR"), fallback):
        if not candidate or candidate in candidates:
            continue
        candidates.append(candidate)

    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = os.path.join(candidate, ".rabbit_write_test")
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("ok")
            try:
                os.remove(probe)
            except OSError:
                pass
            return candidate
        except OSError:
            continue
    return ""


def enable_compilation_cache(
    cache_dir: str = "/tmp/rabbit_jax_cache",
    *,
    min_compile_time_secs: Optional[float] = None,
) -> bool:
    """Enable persistent JIT compilation caching.

    After first JIT, subsequent runs with the same code can reuse cached
    XLA artifacts across processes. For the frozen CPU-first oracle path we
    default to caching all compilations because many medium-sized compile
    buckets add up materially even when each individual bucket is <2s.

    Parameters
    ----------
    cache_dir : str
        Directory for cached compilation artifacts.
    min_compile_time_secs : float, optional
        Persist only compilations whose wall time exceeds this threshold.
        Defaults to ``RABBIT_JAX_CACHE_MIN_COMPILE_SECS`` or ``0.0``.

    Returns
    -------
    bool
        True if caching was enabled, False if JAX not available.
    """
    try:
        import jax
        cache_dir = resolve_writable_cache_dir(cache_dir)
        if not cache_dir:
            return False
        if min_compile_time_secs is None:
            min_compile_time_secs = float(
                os.environ.get("RABBIT_JAX_CACHE_MIN_COMPILE_SECS", "0.0")
            )
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update(
            "jax_persistent_cache_min_compile_time_secs",
            float(min_compile_time_secs),
        )
        return True
    except (ImportError, Exception):
        return False


# ═══════════════════════════════════════════════════════════════════════
# §1. Extended solver method enum
# ═══════════════════════════════════════════════════════════════════════

class JAXSolverMethod(Enum):
    """JAX-native ODE solver methods.

    These extend the SciPy SolverMethod enum. The solve_ode() dispatcher
    checks both enums when routing.
    """
    JAX_RODAS5P = "jax_rodas5p"     # Full-JIT Rodas5P (Steinebach 2023)
    JAX_KVAERNO3 = "jax_kvaerno3"   # Frozen unused compatibility identifier


def is_jax_backend(method) -> bool:
    """Check whether a solver method requires the JAX backend.

    Parameters
    ----------
    method : SolverMethod or JAXSolverMethod or str
        The solver method to check.

    Returns
    -------
    bool
        True if the method is a JAX-native backend.
    """
    if isinstance(method, JAXSolverMethod):
        return True
    if isinstance(method, str):
        return method.startswith("jax_")
    # SolverMethod enum values from solver_config.py
    if hasattr(method, 'value'):
        return str(method.value).startswith("jax_")
    return False


def ensure_jax_available():
    """Raise ImportError with actionable message if JAX is not installed."""
    try:
        import jax  # noqa: F401
        import jaxlib  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "JAX backend requested but jax/jaxlib not installed. "
            "Install with: pip install 'rabbit-bbn[jax]' "
            "or: pip install jax jaxlib"
        ) from e


# ═══════════════════════════════════════════════════════════════════════
# §2. JAX solver configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class JAXSolverConfig:
    """Configuration for JAX-native ODE solvers.

    This is the JAX counterpart to SolverConfig. It carries the same
    tolerance parameters plus JAX-specific options (JIT control,
    precision mode, trajectory storage).

    Parameters
    ----------
    method : JAXSolverMethod
        Which JAX solver to use. Default: JAX_RODAS5P.
    rtol : float
        Relative tolerance. Default: 1e-8.
    atol : float
        Absolute tolerance. Default: 1e-10.
    max_steps : int
        Maximum adaptive steps before declaring failure. Default: 2000.
    max_step_size : float
        Maximum step in N = ln a. Default: 0.1.
    min_step_size : float
        Minimum step (failure below this). Default: 1e-14.
    enable_x64 : bool
        Whether to enable float64. Default: True (required for BBN precision).
    store_trajectory : bool
        Whether to store the full solution trajectory. Default: False
        (saves memory; enable for diagnostics).
    """
    method: JAXSolverMethod = JAXSolverMethod.JAX_RODAS5P
    rtol: float = 1e-8
    atol: float = 1e-10
    max_steps: int = 2000
    max_step_size: float = 0.1
    min_step_size: float = 1e-14
    enable_x64: bool = True
    store_trajectory: bool = False

    def to_solver_params(self):
        """Convert to SolverParams pytree for JAX-internal use."""
        from rabbit.config.jax_params import SolverParams
        return SolverParams(
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
            max_step_size=self.max_step_size,
            min_step_size=self.min_step_size,
        )


# ═══════════════════════════════════════════════════════════════════════
# §3. Presets
# ═══════════════════════════════════════════════════════════════════════

#: Historical standard-tolerance JAX preset; the public name is compatibility-only.
JAX_PRODUCTION_CONFIG = JAXSolverConfig(
    method=JAXSolverMethod.JAX_RODAS5P,
    rtol=1e-8,
    atol=1e-10,
    max_steps=2000,
)

#: Fast JAX preset for frozen smoke/parity checks only.
JAX_FAST_CONFIG = JAXSolverConfig(
    method=JAXSolverMethod.JAX_RODAS5P,
    rtol=1e-6,
    atol=1e-8,
    max_steps=1000,
)

#: Tight JAX solver for reference validation.
JAX_REFERENCE_CONFIG = JAXSolverConfig(
    method=JAXSolverMethod.JAX_RODAS5P,
    rtol=1e-10,
    atol=1e-12,
    max_steps=5000,
)


# ═══════════════════════════════════════════════════════════════════════
# §4. Dispatch helper (used by solve_ode in solver_config.py)
# ═══════════════════════════════════════════════════════════════════════

def dispatch_jax_solver(method: JAXSolverMethod):
    """Return the JAX solver function for the given method.

    Returns
    -------
    callable
        The solver function with signature:
        solve(rhs_fn, y0, N_span, params, solver_params) → result

    Raises
    ------
    ImportError
        If JAX is not installed.
    NotImplementedError
        If the requested method is not yet implemented.
    """
    ensure_jax_available()

    if method == JAXSolverMethod.JAX_RODAS5P:
        # Deferred import to avoid circular deps and JAX requirement at top level
        from rabbit.jax.solver_jax_rodas5p import jax_rodas5p_solve
        return jax_rodas5p_solve
    elif method == JAXSolverMethod.JAX_KVAERNO3:
        raise NotImplementedError(
            "JAX Kvaerno3 solver is planned (J-EXT-5) but not yet implemented. "
            "Use JAX_RODAS5P instead."
        )
    else:
        raise ValueError(f"Unknown JAX solver method: {method}")
