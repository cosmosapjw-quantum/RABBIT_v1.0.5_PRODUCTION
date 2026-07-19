"""tests/test_optional_dependency_probes.py — BD598 PR-F (protects E-R1).

Collection-crash regression guard. Every optional-backend availability probe
must answer "is it installed?" with a plain bool and MUST NOT raise when the
backend is absent — otherwise it crashes pytest collection at the module-level
``skipif`` that calls it (the E-R1 failure: find_spec raising ModuleNotFoundError
on a clean checkout). These probes are imported at collection time across the
suite, so this always-on test is the cheap invariant that keeps a fresh-checkout
collection green.
"""

from __future__ import annotations

import pytest

from rabbit.external.nudec_bsm import has_nudec_bsm
from rabbit.external.alterbbn import has_docker, has_alterbbn_image

PROBES = [has_nudec_bsm, has_docker, has_alterbbn_image]


@pytest.mark.parametrize("probe", PROBES, ids=lambda p: p.__name__)
def test_optional_dep_probe_returns_bool_without_raising(probe, monkeypatch):
    """Each availability probe returns a bool and never propagates an error."""
    # Ensure the optional-backend env hints are absent so we hit the
    # "not installed" branch that previously crashed.
    monkeypatch.delenv("RABBIT_NUDEC_BSM_PATH", raising=False)
    result = probe()
    assert isinstance(result, bool)
