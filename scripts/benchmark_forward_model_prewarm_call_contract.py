"""Structural benchmark for ForwardModel one-shot prewarm contract."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    import rabbit.inference.forward_likelihood as fl

    counts = {
        'direct_solver_style': {'prewarm_calls': 0, 'solve_calls': 0},
        'forward_model_one_shot': {'prewarm_calls': 0, 'solve_calls': 0},
    }

    def fake_prewarm_only(**kwargs):
        counts['forward_model_one_shot']['prewarm_calls'] += 1
        return {'cache_hit': False, 'elapsed_seconds': 0.0}

    def fake_direct_solver(**kwargs):
        if kwargs.get('prewarm_jax', False):
            counts['direct_solver_style']['prewarm_calls'] += 1
        counts['direct_solver_style']['solve_calls'] += 1
        return fl.BBNPrediction(Yp=0.25, DH=2.5e-5, metadata={})

    def fake_model_solver(**kwargs):
        counts['forward_model_one_shot']['solve_calls'] += 1
        return fl.BBNPrediction(Yp=0.25, DH=2.5e-5, metadata={})

    original_prewarm = fl._canonical_jax_prewarm_only
    original_solver = fl.canonical_forward_solver
    try:
        fl._canonical_jax_prewarm_only = fake_prewarm_only
        fl.canonical_forward_solver = fake_model_solver
        model = fl.make_canonical_forward_model(
            backend='jax', prewarm_jax=True,
            N_q=6, jax_thermo_tier=2, jax_use_live_weak_monopoles=True,
        )
        for sigma in (0.0, 1.0e-3, 2.0e-3):
            model.predict(Sigma_H=float(sigma))

        fl.canonical_forward_solver = fake_direct_solver
        for sigma in (0.0, 1.0e-3, 2.0e-3):
            fl.canonical_forward_solver(
                Sigma_H=float(sigma), backend='jax', prewarm_jax=True,
                N_q=6, jax_thermo_tier=2, jax_use_live_weak_monopoles=True,
            )
    finally:
        fl._canonical_jax_prewarm_only = original_prewarm
        fl.canonical_forward_solver = original_solver

    out = {
        'sigma_cases': [0.0, 1.0e-3, 2.0e-3],
        'direct_solver_style': counts['direct_solver_style'],
        'forward_model_one_shot': counts['forward_model_one_shot'],
        'prewarm_call_reduction_factor': (
            counts['direct_solver_style']['prewarm_calls'] / max(counts['forward_model_one_shot']['prewarm_calls'], 1)
        ),
    }
    artifacts = Path(__file__).resolve().parents[1] / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    path = artifacts / 'prr13_forward_model_prewarm_call_contract.json'
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
