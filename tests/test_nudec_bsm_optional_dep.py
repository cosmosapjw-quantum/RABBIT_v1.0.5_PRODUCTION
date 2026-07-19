"""tests/test_nudec_bsm_optional_dep.py — optional-dependency probe safety.

Regression guard for BD598 / PR-A (audit risk E-R1): the NUDEC_BSM
availability probe must answer "is it installed?" without raising when
the optional backend is absent. ``importlib.util.find_spec`` raises
``ModuleNotFoundError`` (not return ``None``) when the *parent* package
of a dotted name is missing, so probing ``BasicModules_source.nudec_v2``
on a clean checkout crashed pytest collection at the ``skipif`` line of
``tests/test_cross_code_live.py`` (exit code 2).

These are always-on unit tests: they need no external backend and must
pass on a fresh checkout with ``RABBIT_NUDEC_BSM_PATH`` unset.
"""

from __future__ import annotations

import sys

from rabbit.external import nudec_bsm


def _force_backend_absent(monkeypatch):
    """Make the probe see no backend, hermetically regardless of test order."""
    monkeypatch.delenv("RABBIT_NUDEC_BSM_PATH", raising=False)
    # BasicModules_source is the (absent) parent package whose missing
    # presence previously made find_spec raise instead of return None.
    # Evict any stray import so the test is order-independent; monkeypatch
    # restores sys.modules afterwards.
    for name in list(sys.modules):
        if name == "BasicModules_source" or name.startswith("BasicModules_source."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_ensure_nudec_path_returns_none_when_parent_pkg_absent(monkeypatch):
    """Probe returns None (does not raise) when the backend is unreachable."""
    _force_backend_absent(monkeypatch)
    result = nudec_bsm._ensure_nudec_path()
    assert result is None


def test_has_nudec_bsm_is_false_not_raising_when_absent(monkeypatch):
    """has_nudec_bsm() returns a bool (False) instead of propagating an error."""
    _force_backend_absent(monkeypatch)
    available = nudec_bsm.has_nudec_bsm()
    assert available is False
