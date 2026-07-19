import numpy as np
from rabbit.inference import model_comparison as mc


def test_run_canonical_full_analysis_builds_scipy_likelihood_without_jax_prewarm(monkeypatch):
    calls = []

    def fake_make_canonical_likelihood(**kwargs):
        calls.append(('build', dict(kwargs)))
        return object()

    def fake_run_full_analysis(likelihood, Sigma_grid=None, prior=None, verbose=True, prewarm_likelihood=False):
        calls.append(('run', likelihood, Sigma_grid, prewarm_likelihood))
        return 'ok'

    monkeypatch.setattr('rabbit.inference.forward_likelihood.make_canonical_likelihood', fake_make_canonical_likelihood)
    monkeypatch.setattr(mc, 'run_full_analysis', fake_run_full_analysis)

    out = mc.run_canonical_full_analysis(
        Sigma_grid=np.array([0.0, 1.0e-3]),
        backend='auto',
        prewarm_likelihood=False,
        N_q=6,
        verbose=False,
    )

    assert out == 'ok'
    assert calls[0][0] == 'build'
    assert calls[0][1]['backend'] == 'auto'
    assert 'prewarm_jax' not in calls[0][1]
    assert calls[0][1]['auto_prewarm_on_first_loglike'] is False
    assert calls[1][0] == 'run'
    assert calls[1][3] is False
