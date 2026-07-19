import sys
import types
import numpy as np

from rabbit.inference import sampler


class _FakeResults:
    def __init__(self):
        self.samples = np.array([[0.1]])
        self.logwt = np.array([0.0])
        self.logz = np.array([0.0])
        self.logzerr = np.array([0.0])
        self.ncall = 1
        self.niter = 1


class _FakeNestedSampler:
    def __init__(self, loglike, prior, ndim, **kwargs):
        self.loglike = loglike
        self.prior = prior
        self.results = _FakeResults()
    def run_nested(self, **kwargs):
        self.loglike(np.array([0.0]))


class _FakeDynesty(types.SimpleNamespace):
    def __init__(self):
        super().__init__(
            DynamicNestedSampler=_FakeNestedSampler,
            NestedSampler=_FakeNestedSampler,
        )


class _FakeUtils(types.SimpleNamespace):
    @staticmethod
    def resample_equal(samples, weights):
        return samples


class FakeLikelihood:
    def __init__(self):
        self.prewarm_calls = 0
        self.loglike_calls = 0
    def prewarm(self):
        self.prewarm_calls += 1
        return {'ok': True}
    def log_likelihood(self, params):
        self.loglike_calls += 1
        return -0.5


def test_make_vector_loglike_adapter_preserves_prewarm_for_run_dynesty(monkeypatch):
    monkeypatch.setitem(sys.modules, 'dynesty', _FakeDynesty())
    monkeypatch.setitem(sys.modules, 'dynesty.utils', _FakeUtils())

    like = FakeLikelihood()
    loglike = sampler.make_vector_loglike_adapter(like, ('Sigma_H',))

    cfg = sampler.DynestyConfig(n_dim=1, dynamic=True, nlive=5, slices=1, walks=1, maxcall=5)
    out = sampler.run_dynesty(loglike, lambda u: u, cfg, verbose=False, prewarm_likelihood=True)

    assert like.prewarm_calls == 1
    assert like.loglike_calls == 1
    assert out['prewarm_applied'] is True


def test_run_canonical_dynesty_uses_adapter_without_retired_jax_prewarm(monkeypatch):
    calls = []

    class FakeLike:
        def prewarm(self):
            calls.append(('prewarm',))
            return {'ok': True}
        def log_likelihood(self, params):
            calls.append(('loglike', dict(params)))
            return -1.0

    def fake_make_canonical_direct_likelihood(**kwargs):
        calls.append(('build', dict(kwargs)))
        return FakeLike()

    def fake_run_dynesty(log_likelihood_fn, prior_transform_fn, config, verbose=True, prewarm_likelihood=False, prewarm_fn=None):
        calls.append(('run', prewarm_likelihood, hasattr(log_likelihood_fn, 'prewarm')))
        if prewarm_likelihood and hasattr(log_likelihood_fn, 'prewarm'):
            log_likelihood_fn.prewarm()
        log_likelihood_fn(np.array([0.0]))
        return {'ok': True}

    monkeypatch.setattr(sampler, 'make_canonical_direct_likelihood', fake_make_canonical_direct_likelihood)
    monkeypatch.setattr(sampler, 'run_dynesty', fake_run_dynesty)

    cfg = sampler.DynestyConfig(n_dim=1, dynamic=True)
    out = sampler.run_canonical_dynesty(
        lambda u: u,
        cfg,
        ('Sigma_H',),
        backend='auto',
        prewarm_likelihood=False,
    )

    assert out == {'ok': True}
    assert calls[0][0] == 'build'
    assert calls[0][1]['backend'] == 'auto'
    assert 'prewarm_jax' not in calls[0][1]
    assert calls[0][1]['auto_prewarm_on_first_loglike'] is False
    assert calls[1] == ('run', False, True)
    assert calls[2][0] == 'loglike'
