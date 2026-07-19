"""tests/test_neff_provenance_guard.py — BD599 W2-PR11 (W2-R3).

The PR-B N_eff_is_derived flag was write-only. This enforces it: BBNPrediction
metadata carries N_eff_is_derived, and get_reported_N_eff warns when a Tier-1
parametric N_eff would be cited as a measurement.
"""

from __future__ import annotations

import warnings

import pytest

from rabbit.inference.forward_likelihood import (
    BBNPrediction,
    canonical_forward_solver,
    get_reported_N_eff,
)


def test_get_reported_neff_warns_when_not_derived():
    pred = BBNPrediction(Yp=0.24, DH=2.5e-5,
                         metadata={"N_eff": 3.011, "N_eff_is_derived": False})
    with pytest.warns(RuntimeWarning, match="parametric Tier-1"):
        assert get_reported_N_eff(pred) == 3.011


def test_get_reported_neff_silent_when_derived():
    pred = BBNPrediction(Yp=0.24, DH=2.5e-5,
                         metadata={"N_eff": 3.044, "N_eff_is_derived": True})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        assert get_reported_N_eff(pred) == 3.044


@pytest.mark.slow
def test_canonical_forward_prediction_carries_provenance_flag():
    """The SciPy canonical (Tier-1) forward stamps N_eff_is_derived=False."""
    pred = canonical_forward_solver(Sigma_H=0.0, N_q=20, backend="scipy")
    assert pred.success
    assert pred.metadata.get("N_eff_is_derived") is False  # Tier-1 parametric


def test_neff_gap_warning_uses_correct_yp_sensitivity():
    """BD622-R3 (audit G6 H-D): the N_eff-gap warning quoted a Y_p offset of
    ~13*delta instead of dY_p/dN_eff ~ 0.013*delta — a factor-1000 overstatement
    (delta=0.095 printed '~1.2e+00', a physically impossible Y_p offset).

    The warning fires only deep inside run_full_coupled_typeI after a full
    solve, so this pins the fix at source level: the mis-scaled literal is
    gone and the corrected coefficient gives a physically sensible offset.
    """
    import inspect

    import rabbit.drivers.full_coupled_typeI as mod

    src = inspect.getsource(mod)
    # Anchor on the f-string brace: "0.013*neff_gap" contains "13*neff_gap"
    # as a bare substring, so match the full interpolation start.
    assert "{13*neff_gap" not in src, "mis-scaled Y_p sensitivity literal is back"
    assert "{0.013*neff_gap" in src
    # Corrected coefficient: a just-above-threshold gap maps to a ~1e-3 Y_p
    # offset (plausible), not ~1e+0 (impossible: Y_p itself is ~0.247).
    assert 0.013 * 0.095 == pytest.approx(1.2e-3, rel=0.05)
    assert 0.013 * 0.095 < 0.01 < 13 * 0.095
