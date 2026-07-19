import sys
import types
import numpy as np

from rabbit.inference import sampler


class _FakeResults:
    def __init__(self):
        self.samples = np.array([[0.1], [0.2]])
        self.logwt = np.log(np.array([0.4, 0.6]))
        self.logz = np.array([0.0])
        self.logzerr = np.array([0.1])
        self.ncall = 7
        self.niter = 3


class _FakeDynamicNestedSampler:
    def __init__(self, *args, **kwargs):
        self.results = _FakeResults()

    def run_nested(self, **kwargs):
        return None


class _FakeNestedSampler(_FakeDynamicNestedSampler):
    pass


def test_run_dynesty_bound_loglike_auto_prewarm(monkeypatch):
    calls = []

    class FakeLikelihood:
        def prewarm(self):
            calls.append('prewarm')
            return {'cache_hit': False}

        def loglike(self, theta):
            return -0.5 * theta[0] ** 2

    fake_dynesty = types.ModuleType('dynesty')
    fake_dynesty.DynamicNestedSampler = _FakeDynamicNestedSampler
    fake_dynesty.NestedSampler = _FakeNestedSampler
    fake_utils = types.ModuleType('dynesty.utils')
    fake_utils.resample_equal = lambda samples, weights: samples
    monkeypatch.setitem(sys.modules, 'dynesty', fake_dynesty)
    monkeypatch.setitem(sys.modules, 'dynesty.utils', fake_utils)

    like = FakeLikelihood()
    config = sampler.DynestyConfig(n_dim=1, dynamic=True)
    result = sampler.run_dynesty(
        like.loglike,
        lambda u: u,
        config,
        verbose=False,
        prewarm_likelihood=True,
    )

    assert calls == ['prewarm']
    assert result['prewarm_applied'] is True
    assert result['prewarm_summary'] == {'cache_hit': False}
    assert result['prewarm_wall_time'] >= 0.0


def test_run_dynesty_explicit_prewarm_overrides_auto(monkeypatch):
    calls = []

    def explicit_prewarm():
        calls.append('explicit')
        return {'source': 'explicit'}

    fake_dynesty = types.ModuleType('dynesty')
    fake_dynesty.DynamicNestedSampler = _FakeDynamicNestedSampler
    fake_dynesty.NestedSampler = _FakeNestedSampler
    fake_utils = types.ModuleType('dynesty.utils')
    fake_utils.resample_equal = lambda samples, weights: samples
    monkeypatch.setitem(sys.modules, 'dynesty', fake_dynesty)
    monkeypatch.setitem(sys.modules, 'dynesty.utils', fake_utils)

    config = sampler.DynestyConfig(n_dim=1, dynamic=False)
    result = sampler.run_dynesty(
        lambda theta: -0.5 * theta[0] ** 2,
        lambda u: u,
        config,
        verbose=False,
        prewarm_likelihood=False,
        prewarm_fn=explicit_prewarm,
    )

    assert calls == ['explicit']
    assert result['prewarm_applied'] is True
    assert result['prewarm_summary'] == {'source': 'explicit'}
