import numpy as np

from rabbit.inference.forward_likelihood import ForwardModel, BBNLikelihood
from rabbit.inference.model_comparison import run_full_analysis


def test_run_full_analysis_explicit_prewarm_runs_once_before_grid_scan():
    calls = []

    def fake_prewarm():
        calls.append("prewarm")
        return {"cache_hit": False}

    def fake_solver(**params):
        sigma = float(params.get("Sigma_H", 0.0))
        calls.append(("solve", sigma))
        return type("Pred", (), {
            "success": True,
            "Yp": 0.2449 + 0.01 * sigma,
            "DH": 2.547e-5 + 1.0e-6 * sigma,
            "metadata": {},
        })()

    model = ForwardModel(solver_fn=fake_solver, prewarm_fn=fake_prewarm, auto_prewarm_on_first_predict=False)
    like = BBNLikelihood(model, auto_prewarm_on_first_loglike=False)

    out = run_full_analysis(
        like,
        Sigma_grid=np.array([0.0, 1.0e-3, 2.0e-3]),
        verbose=False,
        prewarm_likelihood=True,
    )

    assert calls[0] == "prewarm"
    assert len([c for c in calls if c == "prewarm"]) == 1
    assert len([c for c in calls if isinstance(c, tuple) and c[0] == "solve"]) == 3
    assert out.grid_result.best_fit["Sigma_H"] in {0.0, 0.001, 0.002}
