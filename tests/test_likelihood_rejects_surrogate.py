"""tests/test_likelihood_rejects_surrogate.py — BD599 W2-PR12 (BIA-3).

The calibrated surrogate (_simplified_bbn_solver, analytic shear factor) must
never reach a posterior. It now stamps metadata['surrogate']=True and
BBNLikelihood.log_likelihood refuses any such prediction.
"""

from __future__ import annotations

import pytest

from rabbit.inference.forward_likelihood import (
    BBNLikelihood,
    BBNPrediction,
    Observation,
    _simplified_bbn_solver,
)


class _SurrogateModel:
    def predict(self, **params):
        keep = {k: params[k] for k in ("Sigma_H", "eta", "tau_n") if k in params}
        return _simplified_bbn_solver(**keep)


class _PlainModel:
    def predict(self, **params):
        return BBNPrediction(Yp=0.245, DH=2.5e-5, metadata={"N_eff_is_derived": False})


def _obs():
    return [Observation("Y_p", 0.245, 0.004), Observation("D/H", 2.5e-5, 3e-7)]


def test_likelihood_rejects_surrogate_prediction():
    lik = BBNLikelihood(_SurrogateModel(), observations=_obs())
    with pytest.warns(DeprecationWarning):  # surrogate also warns
        with pytest.raises(ValueError, match="SURROGATE"):
            lik.log_likelihood(Sigma_H=0.0, eta=6.104e-10, tau_n=878.4)


def test_likelihood_accepts_non_surrogate_prediction():
    lik = BBNLikelihood(_PlainModel(), observations=_obs())
    ll = lik.log_likelihood(Sigma_H=0.0, eta=6.104e-10, tau_n=878.4)
    assert ll <= 0.0  # finite log-likelihood, no raise
