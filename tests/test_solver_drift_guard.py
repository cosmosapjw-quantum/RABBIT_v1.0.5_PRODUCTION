"""tests/test_solver_drift_guard.py — BD613 solver drift lock.

Historically, ``full_coupled_typeI.py`` silently remapped the declared
production solver method (Radau, per ``PRODUCTION_CONFIG``) to BDF whenever
the default config was used, via an identity check against the module
singleton backed by a hidden per-call BDF constant. Zero rationale for the
remap was ever recorded (BD613).

BD613-PR1b retired that remap: ``PRODUCTION_CONFIG`` now declares BDF
directly (matching what has always actually run — see BD598 Radau/BDF parity:
|ΔY_p|=8.25e-8, D/H rel 9.87e-6), Radau-at-production-tolerances is available
as ``PRODUCTION_RADAU_CONFIG`` (what ``classA_driver`` defaults to, preserving
its historical behavior), and ``full_coupled_typeI._resolve_effective_solver``
is an explicit identity seam guarded by a fail-loud ``RuntimeError`` at the
call site if effective ever diverges from requested. This module locks that
contract down: requested must equal effective on every lane, always.
"""

from __future__ import annotations

import pytest

from rabbit.config.solver_config import (
    PRODUCTION_CONFIG,
    PRODUCTION_RADAU_CONFIG,
    REFERENCE_CONFIG,
    SolverMethod,
)
from rabbit.drivers import full_coupled_typeI as mod
from rabbit.drivers.classA_driver import ClassAConfig
from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI
from rabbit.inference.forward_likelihood import canonical_forward_solver


def _solve(*, solver=None):
    cfg = FullCoupledConfig(
        Sigma_H_plus=0.1,
        N_q=8,
        N_mu=8,
        n_reactions=12,
        correction_level=0,
        tier=1,
        enable_teff=False,
        solver=solver,
    )
    return run_full_coupled_typeI(cfg)


def test_requested_equals_effective_default():
    """Default (PRODUCTION_CONFIG) solve: requested and effective solver
    methods agree, and the production lane is honestly declared as BDF."""
    result = _solve()
    requested = result.metadata["solver_method_requested"]
    effective = result.metadata["solver_method_effective"]
    assert requested == effective
    assert effective == "BDF"


@pytest.mark.parametrize(
    "solver, expected_method",
    [
        (PRODUCTION_CONFIG, "BDF"),
        (PRODUCTION_RADAU_CONFIG, "Radau"),
        (REFERENCE_CONFIG, "Radau"),
    ],
)
def test_requested_equals_effective_matrix(solver, expected_method):
    """Requested and effective agree for every production/reference solver
    config, not just the default."""
    result = _solve(solver=solver)
    requested = result.metadata["solver_method_requested"]
    effective = result.metadata["solver_method_effective"]
    assert requested == effective
    assert effective == expected_method


def test_production_config_declares_bdf():
    """Declaration lock (no solve): PRODUCTION_CONFIG is BDF, and Radau at the
    same tolerances is available separately as PRODUCTION_RADAU_CONFIG."""
    assert PRODUCTION_CONFIG.method is SolverMethod.BDF
    assert PRODUCTION_RADAU_CONFIG.method is SolverMethod.RADAU


def test_remap_singleton_deleted():
    """The hidden per-call BDF singleton that powered the silent remap is
    gone; _resolve_effective_solver is the only seam left, and it is
    identity by design."""
    assert not hasattr(mod, "_PRODUCTION_BDF_CONFIG")
    assert mod._resolve_effective_solver(PRODUCTION_CONFIG) is PRODUCTION_CONFIG


def test_drift_guard_raises(monkeypatch):
    """If _resolve_effective_solver ever returns a config with a different
    method than requested, the driver must fail loud rather than run a
    silently different solver. The guard fires before the solve loop, so a
    tiny/failing config is fine here."""
    monkeypatch.setattr(mod, "_resolve_effective_solver", lambda requested: REFERENCE_CONFIG)
    with pytest.raises(RuntimeError, match="drift guard"):
        _solve(solver=PRODUCTION_CONFIG)


def test_forward_likelihood_override_inherits_production_method():
    """An rtol/atol override on the public inference surface must inherit the
    production method rather than hard-coding one (BD613 site B)."""
    pred = canonical_forward_solver(
        Sigma_H=0.05,
        backend="scipy",
        N_q=6,
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    assert pred.success
    requested = pred.metadata["solver_method_requested"]
    effective = pred.metadata["solver_method_effective"]
    assert requested == effective == PRODUCTION_CONFIG.method.value


def test_classA_default_solver_is_radau():
    """classA's __post_init__ default (no solve): preserved as Radau at
    production tolerances via PRODUCTION_RADAU_CONFIG (BD613 site C, no
    remap ever existed here — this locks it stays that way)."""
    config = ClassAConfig(solver=None)
    assert config.solver.method is SolverMethod.RADAU
