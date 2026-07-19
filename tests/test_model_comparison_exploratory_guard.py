"""tests/test_model_comparison_exploratory_guard.py — BD599 W2-PR8 (INF-4).

The independently-importable evidence primitives (savage_dickey_bf,
profile_likelihood, information_criteria) emit Bayes-factor / p-value / sigma
results, so each must carry the EXPLORATORY warning that previously lived only in
run_full_analysis — otherwise a single direct import is one step from a headline.
``_acknowledge_exploratory=True`` suppresses it (used by run_full_analysis, which
warns once at the top).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rabbit.inference.forward_likelihood import GridResult
from rabbit.inference.model_comparison import (
    information_criteria,
    profile_likelihood,
    savage_dickey_bf,
)


def _grid_result():
    grid = np.array([0.0, 0.25, 0.5, 0.75])
    chi2 = np.array([0.0, 0.5, 2.0, 5.0])
    p1d = np.exp(-0.5 * chi2)
    return GridResult(
        param_names=["Sigma_H"],
        param_grids={"Sigma_H": grid},
        log_likelihood=-0.5 * chi2,
        chi2=chi2,
        best_fit={"Sigma_H": 0.0},
        best_chi2=0.0,
        marginalized_1d={"Sigma_H": (grid, p1d)},
    )


def test_information_criteria_warns_exploratory():
    with pytest.warns(RuntimeWarning, match="EXPLORATORY"):
        information_criteria(chi2_flrw=1.0, chi2_bianchi=0.5)


def test_profile_likelihood_warns_exploratory():
    with pytest.warns(RuntimeWarning, match="EXPLORATORY"):
        profile_likelihood(_grid_result())


def test_savage_dickey_warns_exploratory():
    with pytest.warns(RuntimeWarning, match="EXPLORATORY"):
        savage_dickey_bf(_grid_result())


def test_acknowledge_suppresses_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        information_criteria(1.0, 0.5, _acknowledge_exploratory=True)
        profile_likelihood(_grid_result(), _acknowledge_exploratory=True)
        savage_dickey_bf(_grid_result(), _acknowledge_exploratory=True)
