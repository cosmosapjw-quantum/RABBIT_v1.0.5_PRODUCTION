"""tests/test_registry_drift_gate.py — v3.2 hostile-re-audit P3 gate.

Audit-honest meta-test that fails LOUDLY when the backend-capability
registry drifts from the test-side enumerations. Built after the v3.2
hostile re-audit (2026-05-05) caught that ``jax_tilted_full_coupled``
had been registered in ``CAPABILITY_BY_BACKEND`` since v2.0 Phase γ-3
but never added to the lock-test's HIERARCHY dict + EXPECTED_CATALOG_KEYS
list — silent drift through three release cycles (v3.0 / v3.1 / v3.2).

This test exists to make that class of silent drift impossible:

  - It depends only on the same data structures that
    ``tests/test_inference_hierarchy_lock.py`` uses, so any future
    capability addition must be reflected in both files.
  - It runs as a fast unit test (no JAX driver invocation) so it is
    *never* skipped by the @slow / @expensive / @cross_code markers.
  - It is marked ``@pytest.mark.release_smoke`` so it gates release
    candidates explicitly.
  - It emits an actionable error message (which key is missing where).

Reuse: this file is the single source of truth for the registry-vs-
test consistency contract; ``test_inference_hierarchy_lock.py`` does
the per-key tier/scope checks. Together they prevent regressions from
silent additions and silent removals.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.release_smoke


def _import_registry():
    from rabbit.config.backend_capabilities import (
        CAPABILITY_BY_BACKEND, CAPABILITY_BY_KEY,
    )
    return CAPABILITY_BY_BACKEND, CAPABILITY_BY_KEY


def _import_lock():
    """Load HIERARCHY + EXPECTED_CATALOG_KEYS from the sibling lock test.

    The ``tests/`` directory isn't a Python package, so we use
    ``importlib.util`` against the explicit file path.
    """
    import importlib.util
    from pathlib import Path
    lock_path = Path(__file__).parent / "test_inference_hierarchy_lock.py"
    spec = importlib.util.spec_from_file_location("_lock_module", lock_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HIERARCHY, mod.EXPECTED_CATALOG_KEYS


# ═══════════════════════════════════════════════════════════════════════
# §1. Backend hierarchy ↔ HIERARCHY dict consistency
# ═══════════════════════════════════════════════════════════════════════

def test_no_backend_in_registry_missing_from_hierarchy_lock():
    """Every backend in CAPABILITY_BY_BACKEND must appear in HIERARCHY.

    If this fails, a new backend was registered in
    ``rabbit.config.backend_capabilities`` without updating the lock
    test's HIERARCHY dict. Add the missing entry to
    ``tests/test_inference_hierarchy_lock.py::HIERARCHY``.
    """
    cap_by_backend, _ = _import_registry()
    hierarchy, _ = _import_lock()
    missing = sorted(set(cap_by_backend) - set(hierarchy))
    assert not missing, (
        f"Registry drift: {len(missing)} backend(s) in CAPABILITY_BY_BACKEND "
        f"but not in tests/test_inference_hierarchy_lock.py::HIERARCHY: "
        f"{missing}\n\n"
        "Fix: add each missing backend to HIERARCHY with its key, tier, "
        "and physics_scope (matching backend_capabilities.py)."
    )


def test_no_hierarchy_lock_entry_missing_from_registry():
    """Every entry in HIERARCHY must be a real registered backend.

    If this fails, the lock test references a backend that no longer
    exists. Either remove the stale entry from HIERARCHY or restore
    the registry entry.
    """
    cap_by_backend, _ = _import_registry()
    hierarchy, _ = _import_lock()
    missing = sorted(set(hierarchy) - set(cap_by_backend))
    assert not missing, (
        f"Registry drift: {len(missing)} backend(s) in HIERARCHY but "
        f"not in CAPABILITY_BY_BACKEND: {missing}\n\n"
        "Fix: either remove the stale entry from HIERARCHY or add the "
        "registry entry to backend_capabilities.py."
    )


# ═══════════════════════════════════════════════════════════════════════
# §2. Capability catalog ↔ EXPECTED_CATALOG_KEYS consistency
# ═══════════════════════════════════════════════════════════════════════

def test_no_capability_in_registry_missing_from_expected_catalog():
    """Every key in CAPABILITY_BY_KEY must appear in EXPECTED_CATALOG_KEYS.

    If this fails, a new BackendCapability was added to
    ``rabbit.config.backend_capabilities`` without updating the
    lock-test's EXPECTED_CATALOG_KEYS list.
    """
    _, cap_by_key = _import_registry()
    _, expected = _import_lock()
    missing = sorted(set(cap_by_key) - set(expected))
    assert not missing, (
        f"Registry drift: {len(missing)} capability key(s) in "
        f"CAPABILITY_BY_KEY but not in EXPECTED_CATALOG_KEYS: {missing}\n\n"
        "Fix: add each missing key to "
        "tests/test_inference_hierarchy_lock.py::EXPECTED_CATALOG_KEYS."
    )


def test_no_expected_catalog_entry_missing_from_registry():
    """Every entry in EXPECTED_CATALOG_KEYS must be a real capability key.

    If this fails, the lock test references a capability that no
    longer exists. Either remove the stale entry or restore the
    registry capability.
    """
    _, cap_by_key = _import_registry()
    _, expected = _import_lock()
    missing = sorted(set(expected) - set(cap_by_key))
    assert not missing, (
        f"Registry drift: {len(missing)} key(s) in EXPECTED_CATALOG_KEYS "
        f"but not in CAPABILITY_BY_KEY: {missing}"
    )


# ═══════════════════════════════════════════════════════════════════════
# §3. Cross-consistency: every backend's key must be in catalog
# ═══════════════════════════════════════════════════════════════════════

def test_every_backend_dispatch_key_is_a_real_capability():
    """For every backend → cap mapping, cap.key must be in CAPABILITY_BY_KEY.

    This catches the case where a backend is dispatched to a
    capability whose key is not registered in the by-key catalog
    (e.g. a typo in the dispatch table).
    """
    cap_by_backend, cap_by_key = _import_registry()
    orphaned = []
    for backend, cap in cap_by_backend.items():
        if cap.key not in cap_by_key:
            orphaned.append((backend, cap.key))
    assert not orphaned, (
        f"Dispatch table has {len(orphaned)} entry/entries pointing to "
        f"unregistered capability keys: {orphaned}"
    )


# ═══════════════════════════════════════════════════════════════════════
# §4. Counts (single source of truth)
# ═══════════════════════════════════════════════════════════════════════

def test_capability_counts_match_lock_lengths():
    """Cardinality cross-check: registry and lock test agree on sizes."""
    cap_by_backend, cap_by_key = _import_registry()
    hierarchy, expected = _import_lock()
    assert len(cap_by_backend) == len(hierarchy), (
        f"len(CAPABILITY_BY_BACKEND) = {len(cap_by_backend)}, "
        f"len(HIERARCHY) = {len(hierarchy)}; one was added without the other"
    )
    assert len(cap_by_key) == len(expected), (
        f"len(CAPABILITY_BY_KEY) = {len(cap_by_key)}, "
        f"len(EXPECTED_CATALOG_KEYS) = {len(expected)}"
    )
