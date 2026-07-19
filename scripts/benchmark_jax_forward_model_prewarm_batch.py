"""Benchmark one-shot ForwardModel prewarm for repeated JAX predictions."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CASES = [0.0, 1.0e-3, 2.0e-3]


def _run_snippet(code: str) -> dict:
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(code)
        path = f.name
    cmd = [sys.executable, path]
    env = dict(PYTHONPATH=str(ROOT / 'src'))
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    return json.loads(proc.stdout)


def main() -> None:
    direct_code = textwrap.dedent(
        f'''
        import json, time
        from rabbit.inference.forward_likelihood import make_canonical_forward_model
        model = make_canonical_forward_model(
            backend='jax',
            prewarm_jax=False,
            N_q=6,
            jax_thermo_tier=2,
            jax_use_live_weak_monopoles=True,
        )
        sigmas = {CASES!r}
        t0 = time.perf_counter()
        preds = [model.predict(Sigma_H=float(s)).success for s in sigmas]
        elapsed = time.perf_counter() - t0
        print(json.dumps({{'solve_elapsed_seconds': elapsed, 'success_vector': preds}}))
        '''
    )
    prewarm_code = textwrap.dedent(
        f'''
        import json, time
        from rabbit.inference.forward_likelihood import make_canonical_forward_model
        model = make_canonical_forward_model(
            backend='jax',
            prewarm_jax=True,
            N_q=6,
            jax_thermo_tier=2,
            jax_use_live_weak_monopoles=True,
        )
        sigmas = {CASES!r}
        t0 = time.perf_counter()
        warm = model.prewarm()
        prewarm_elapsed = time.perf_counter() - t0
        t1 = time.perf_counter()
        preds = [model.predict(Sigma_H=float(s)).success for s in sigmas]
        solve_elapsed = time.perf_counter() - t1
        print(json.dumps({{'prewarm_elapsed_seconds': prewarm_elapsed, 'solve_elapsed_seconds': solve_elapsed, 'prewarm_summary': warm, 'success_vector': preds}}))
        '''
    )
    out = {
        'direct': _run_snippet(direct_code),
        'prewarm_then_run': _run_snippet(prewarm_code),
        'sigma_cases': CASES,
    }
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    path = artifacts / 'prr13_forward_model_prewarm_batch_benchmark.json'
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
