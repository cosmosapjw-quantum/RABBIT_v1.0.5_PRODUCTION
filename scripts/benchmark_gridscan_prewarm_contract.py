#!/usr/bin/env python3
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from rabbit.inference.forward_likelihood import ForwardModel, BBNLikelihood, BBNPrediction, grid_scan

def make_fake_model(prewarm_delay: float = 0.25, solve_delay: float = 0.02):
    calls = {'prewarm': 0, 'solve': 0}
    def fake_prewarm():
        calls['prewarm'] += 1
        time.sleep(prewarm_delay)
        return {'cache_hit': False, 'simulated_delay_seconds': prewarm_delay}
    def fake_solver(Sigma_H=0.0, eta=6.104e-10, tau_n=878.4):
        calls['solve'] += 1
        time.sleep(solve_delay)
        return BBNPrediction(Yp=0.245 + 0.01 * float(Sigma_H), DH=2.5e-5, params={'Sigma_H': float(Sigma_H), 'eta': float(eta), 'tau_n': float(tau_n)})
    model = ForwardModel(solver_fn=fake_solver, prewarm_fn=fake_prewarm, auto_prewarm_on_first_predict=False)
    like = BBNLikelihood(model, auto_prewarm_on_first_loglike=False)
    return like, calls

def run_direct():
    like, calls = make_fake_model()
    grid = {'Sigma_H': np.array([0.0, 1.0e-3, 2.0e-3])}
    like.auto_prewarm_on_first_loglike = True
    t0 = time.perf_counter()
    _ = grid_scan(like, grid, verbose=False, prewarm_likelihood=False)
    return {'elapsed_seconds': time.perf_counter() - t0, 'calls': calls}

def run_staged():
    like, calls = make_fake_model()
    grid = {'Sigma_H': np.array([0.0, 1.0e-3, 2.0e-3])}
    t0 = time.perf_counter()
    prewarm_summary = like.prewarm()
    prewarm_elapsed = time.perf_counter() - t0
    t1 = time.perf_counter()
    _ = grid_scan(like, grid, verbose=False, prewarm_likelihood=False)
    return {'prewarm_elapsed_seconds': prewarm_elapsed, 'scan_elapsed_seconds': time.perf_counter() - t1, 'calls': calls, 'prewarm_summary': prewarm_summary}

def main() -> None:
    out = {'direct': run_direct(), 'staged': run_staged()}
    path = Path('artifacts/prr14_gridscan_prewarm_contract_benchmark.json')
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f'Wrote {path}')

if __name__ == '__main__':
    main()
