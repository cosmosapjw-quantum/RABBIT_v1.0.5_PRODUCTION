from rabbit.inference import sampler


def test_sampler_bbn_likelihood_auto_prewarm_forward_model_like_runs_once():
    calls = []

    class FakeModel:
        def prewarm(self):
            calls.append("prewarm")
            return {"cache_hit": False}

        def predict(self, **params):
            calls.append(("solve", float(params["Sigma_H"])))
            return {"Yp": 0.2449, "DH": 2.547e-5}

    like = sampler.BBNLikelihood(solver_fn=FakeModel(), auto_prewarm_on_first_loglike=True)
    like.log_likelihood({"Sigma_H": 0.0})
    like.log_likelihood({"Sigma_H": 1.0e-3})

    assert calls == ["prewarm", ("solve", 0.0), ("solve", 0.001)]
    assert like._prewarmed is True
    assert like._last_prewarm_summary == {"cache_hit": False}


def test_sampler_bbn_likelihood_bound_predict_method_carries_prewarm():
    calls = []

    class FakeModel:
        def prewarm(self):
            calls.append("prewarm")
            return {"cache_hit": False}

        def predict(self, **params):
            calls.append(("solve", float(params["Sigma_H"])))
            return {"Yp": 0.2449, "DH": 2.547e-5}

    model = FakeModel()
    like = sampler.BBNLikelihood(solver_fn=model.predict, auto_prewarm_on_first_loglike=True)
    like.log_likelihood({"Sigma_H": 0.0})
    like.log_likelihood({"Sigma_H": 2.0e-3})

    assert calls == ["prewarm", ("solve", 0.0), ("solve", 0.002)]
