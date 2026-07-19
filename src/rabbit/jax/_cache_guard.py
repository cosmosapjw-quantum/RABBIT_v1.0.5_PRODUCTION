"""Thread-safe cache guard for JAX JIT compilation caches.

All module-level mutable caches in the JAX backend use this guard
to prevent data races in multi-threaded contexts (e.g., parallel
grid scans, concurrent forward model evaluations).

Contract:
  - All OrderedDict/dict caches in jax/ are guarded by _CACHE_LOCK.
  - lru_cache decorators are inherently thread-safe in CPython (GIL),
    but explicit locking is added for correctness on non-CPython runtimes.
  - clear_all_caches() provides a deterministic reset for reproducibility.
"""
import threading
from typing import List, Callable

_CACHE_LOCK = threading.Lock()

# Registry of all clearable caches
_CACHE_CLEARERS: List[Callable] = []


def register_cache_clearer(fn: Callable) -> None:
    """Register a cache-clearing function."""
    _CACHE_CLEARERS.append(fn)


def clear_all_caches() -> int:
    """Clear all registered JAX caches. Returns number of caches cleared."""
    with _CACHE_LOCK:
        count = 0
        for fn in _CACHE_CLEARERS:
            fn()
            count += 1
        return count
