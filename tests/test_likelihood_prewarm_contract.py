from rabbit.inference.forward_likelihood import (
    ForwardModel,
    BBNLikelihood,
    BBNPrediction,
    grid_scan,
    derive_sigma_constraint,
)
import numpy as np


def test_bbn_likelihood_auto_prewarm_on_first_loglike():
    calls = []
    def fake_prewarm():
        calls.append('prewarm')
        return {'cache_hit': False}
    def fake_solver(Sigma_H=0.0, eta=6.104e-10, tau_n=878.4):
        calls.append(('solve', float(Sigma_H)))
        return BBNPrediction(Yp=0.245 + 0.01 * float(Sigma_H), DH=2.5e-5)
    model = ForwardModel(solver_fn=fake_solver, prewarm_fn=fake_prewarm, auto_prewarm_on_first_predict=False)
    like = BBNLikelihood(model, auto_prewarm_on_first_loglike=True)
    like.log_likelihood(Sigma_H=0.0)
    like.log_likelihood(Sigma_H=1.0e-3)
    assert calls == ['prewarm', ('solve', 0.0), ('solve', 0.001)]
    assert like._prewarmed is True
    assert like._last_prewarm_summary == {'cache_hit': False}


def test_grid_scan_explicit_prewarm_runs_once_before_loop():
    calls = []
    def fake_prewarm():
        calls.append('prewarm')
        return {'cache_hit': False}
    def fake_solver(Sigma_H=0.0, eta=6.104e-10, tau_n=878.4):
        calls.append(('solve', float(Sigma_H)))
        return BBNPrediction(Yp=0.245 + 0.01 * float(Sigma_H), DH=2.5e-5)
    model = ForwardModel(solver_fn=fake_solver, prewarm_fn=fake_prewarm, auto_prewarm_on_first_predict=False)
    like = BBNLikelihood(model, auto_prewarm_on_first_loglike=False)
    result = grid_scan(like, {'Sigma_H': np.array([0.0, 1.0e-3, 2.0e-3])}, verbose=False, prewarm_likelihood=True)
    assert calls[0] == 'prewarm'
    assert len([c for c in calls if c == 'prewarm']) == 1
    assert [c for c in calls if isinstance(c, tuple)] == [('solve', 0.0), ('solve', 0.001), ('solve', 0.002)]
    assert result.best_fit['Sigma_H'] in (0.0, 0.001, 0.002)


def test_derive_sigma_constraint_explicit_prewarm_runs_once():
    calls = []
    def fake_prewarm():
        calls.append('prewarm')
        return {'cache_hit': False}
    def fake_solver(Sigma_H=0.0, eta=6.104e-10, tau_n=878.4):
        calls.append(('solve', float(Sigma_H)))
        return BBNPrediction(Yp=0.245 + 0.01 * float(Sigma_H), DH=2.5e-5)
    model = ForwardModel(solver_fn=fake_solver, prewarm_fn=fake_prewarm, auto_prewarm_on_first_predict=False)
    like = BBNLikelihood(model, auto_prewarm_on_first_loglike=False)
    out = derive_sigma_constraint(like, Sigma_grid=np.array([0.0, 1.0e-3, 2.0e-3]), prewarm_likelihood=True)
    assert calls[0] == 'prewarm'
    assert len([c for c in calls if c == 'prewarm']) == 1
    assert len([c for c in calls if isinstance(c, tuple)]) == 3
    assert 'upper_limit' in out
