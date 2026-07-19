from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np

from rabbit.inference import sampler
from rabbit.inference import model_comparison as mc


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


class FakeLike:
    def __init__(self, prewarm_s=0.08, body_s=0.06):
        self.prewarm_s = prewarm_s
        self.body_s = body_s
        self.prewarm_calls = 0
        self.loglike_calls = 0

    def prewarm(self):
        self.prewarm_calls += 1
        time.sleep(self.prewarm_s)
        return {'slept': self.prewarm_s, 'calls': self.prewarm_calls}

    def log_likelihood(self, params):
        self.loglike_calls += 1
        time.sleep(self.body_s)
        return -0.5


def benchmark_dynesty_adapter():
    sys.modules['dynesty'] = _FakeDynesty()
    sys.modules['dynesty.utils'] = _FakeUtils()
    cfg = sampler.DynestyConfig(n_dim=1, dynamic=True, nlive=5, slices=1, walks=1, maxcall=5)

    like_lambda = FakeLike()
    direct_loglike = lambda theta: like_lambda.log_likelihood({'Sigma_H': float(theta[0])})
    t0 = time.perf_counter()
    out_direct = sampler.run_dynesty(direct_loglike, lambda u: u, cfg, verbose=False, prewarm_likelihood=True)
    direct_elapsed = time.perf_counter() - t0

    like_adapter = FakeLike()
    adapted_loglike = sampler.make_vector_loglike_adapter(like_adapter, ('Sigma_H',))
    t1 = time.perf_counter()
    out_adapter = sampler.run_dynesty(adapted_loglike, lambda u: u, cfg, verbose=False, prewarm_likelihood=True)
    adapter_elapsed = time.perf_counter() - t1

    return {
        'lambda_hidden_prewarm': {
            'elapsed_seconds': direct_elapsed,
            'prewarm_applied': out_direct['prewarm_applied'],
            'prewarm_calls': like_lambda.prewarm_calls,
            'loglike_calls': like_lambda.loglike_calls,
        },
        'adapter_preserved_prewarm': {
            'elapsed_seconds': adapter_elapsed,
            'prewarm_applied': out_adapter['prewarm_applied'],
            'prewarm_calls': like_adapter.prewarm_calls,
            'loglike_calls': like_adapter.loglike_calls,
            'body_wall_time_seconds': out_adapter['wall_time'],
            'prewarm_wall_time_seconds': out_adapter['prewarm_wall_time'],
        },
    }


def benchmark_canonical_analysis_wrapper():
    orig_make = sys.modules['rabbit.inference.forward_likelihood'].make_canonical_likelihood
    orig_run = mc.run_full_analysis

    class FakeAnalysisLike:
        def __init__(self):
            self.prewarm_calls = 0
        def prewarm(self):
            self.prewarm_calls += 1
            time.sleep(0.08)
            return {'calls': self.prewarm_calls}

    fake_like = FakeAnalysisLike()

    def fake_make_canonical_likelihood(**kwargs):
        return fake_like

    def fake_run_full_analysis(likelihood, Sigma_grid=None, prior=None, verbose=True, prewarm_likelihood=False):
        if prewarm_likelihood:
            likelihood.prewarm()
        time.sleep(0.06)
        return {
            'prewarm_likelihood': prewarm_likelihood,
            'prewarm_calls': likelihood.prewarm_calls,
        }

    sys.modules['rabbit.inference.forward_likelihood'].make_canonical_likelihood = fake_make_canonical_likelihood
    mc.run_full_analysis = fake_run_full_analysis
    try:
        t0 = time.perf_counter()
        out = mc.run_canonical_full_analysis(backend='jax', prewarm_jax=True, verbose=False)
        elapsed = time.perf_counter() - t0
    finally:
        sys.modules['rabbit.inference.forward_likelihood'].make_canonical_likelihood = orig_make
        mc.run_full_analysis = orig_run

    return {
        'elapsed_seconds': elapsed,
        'prewarm_likelihood': bool(out['prewarm_likelihood']),
        'prewarm_calls': int(out['prewarm_calls']),
    }


def main():
    out = {
        'dynesty_adapter': benchmark_dynesty_adapter(),
        'canonical_analysis_wrapper': benchmark_canonical_analysis_wrapper(),
    }
    out_path = Path('artifacts/prr17_canonical_wrapper_prewarm_contract_benchmark.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(out_path)


if __name__ == '__main__':
    main()
