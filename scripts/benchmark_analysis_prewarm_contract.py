from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

from rabbit.inference.forward_likelihood import ForwardModel, BBNLikelihood
from rabbit.inference.model_comparison import run_full_analysis


class _FakePred:
    def __init__(self, yp, dh):
        self.success = True
        self.Yp = float(yp)
        self.DH = float(dh)
        self.metadata = {}


def _make_model(counter, prewarm_sleep=0.08, solve_sleep=0.02):
    def fake_prewarm():
        counter['prewarm_calls'] += 1
        t0 = time.perf_counter()
        time.sleep(prewarm_sleep)
        dt = time.perf_counter() - t0
        counter['prewarm_elapsed_seconds'] += dt
        return {'cache_hit': False, 'elapsed_seconds': dt}

    def fake_solver(**params):
        sigma = float(params.get('Sigma_H', 0.0))
        counter['solve_calls'] += 1
        time.sleep(solve_sleep)
        return _FakePred(0.2449 + 0.01 * sigma, 2.547e-5 + 1.0e-6 * sigma)

    return ForwardModel(solver_fn=fake_solver, prewarm_fn=fake_prewarm, auto_prewarm_on_first_predict=False)


def _run_case(staged: bool):
    counter = {'prewarm_calls': 0, 'solve_calls': 0, 'prewarm_elapsed_seconds': 0.0}
    model = _make_model(counter)
    like = BBNLikelihood(model, auto_prewarm_on_first_loglike=not staged)
    sigma_grid = np.array([0.0, 1.0e-3, 2.0e-3])
    t0 = time.perf_counter()
    run_full_analysis(like, Sigma_grid=sigma_grid, verbose=False, prewarm_likelihood=staged)
    elapsed = time.perf_counter() - t0
    out = {
        'elapsed_seconds': elapsed,
        'prewarm_calls': counter['prewarm_calls'],
        'solve_calls': counter['solve_calls'],
    }
    if staged:
        out['prewarm_elapsed_seconds'] = counter['prewarm_elapsed_seconds']
        out['analysis_body_elapsed_seconds'] = elapsed - counter['prewarm_elapsed_seconds']
    return out


def main():
    result = {
        'direct': _run_case(staged=False),
        'staged': _run_case(staged=True),
    }
    out = Path('artifacts/prr15_analysis_prewarm_contract_benchmark.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(out)


if __name__ == '__main__':
    main()
