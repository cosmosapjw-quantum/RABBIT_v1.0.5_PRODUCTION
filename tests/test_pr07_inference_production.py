"""PR-07: Inference Parameter-Estimation Production Lock — TDD tests.

Production: ForwardModel.predict + log_likelihood on canonical Type I {Yp, D/H}.
Exploratory: sampler execution, posterior recovery, Bayes factor.
"""
import pytest
import numpy as np

pytestmark = [pytest.mark.production, pytest.mark.release_smoke]


class TestInferenceRegistryScope:
    """Registry must specify PE-only scope."""

    def test_inference_no_blockers_for_pe_slice(self):
        """After PR-07, PE slice should have no blockers."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['inference_exploratory']
        assert len(f.blockers) == 0, \
            f"Inference PE slice should have no blockers: {f.blockers}"

    def test_inference_validation_mode(self):
        """Inference validation_mode should reflect PE promotion."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['inference_exploratory']
        assert f.validation_mode in ("exploratory", "diagnostic", "candidate"), \
            f"Inference validation_mode: {f.validation_mode}"

    def test_inference_short_summary_mentions_pe(self):
        """Short summary must mention PE or parameter estimation."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['inference_exploratory']
        combined = f"{f.short_summary} {f.notes}".lower()
        has_pe = any(q in combined for q in ['pe', 'parameter', 'forward', 'predict'])
        assert has_pe, "Inference short_summary must mention PE scope"


class TestForwardModelAPI:
    """ForwardModel.predict must return physical observables."""

    @pytest.mark.slow
    def test_forward_model_predict_returns_yp(self):
        """ForwardModel.predict must return BBNPrediction with Yp."""
        from rabbit.inference.forward_likelihood import ForwardModel, canonical_forward_solver
        fm = ForwardModel(solver_fn=canonical_forward_solver)
        pred = fm.predict(Sigma_H=0.0)
        assert hasattr(pred, 'Yp')
        assert 0.2 < pred.Yp < 0.3, f"Yp={pred.Yp} out of physical range"

    @pytest.mark.slow
    def test_forward_model_log_likelihood_finite(self):
        """ForwardModel.log_likelihood must return finite value."""
        from rabbit.inference.forward_likelihood import ForwardModel, canonical_forward_solver, Observation
        fm = ForwardModel(solver_fn=canonical_forward_solver)
        obs = Observation(name='Yp', value=0.245, sigma=0.003)
        ll = fm.log_likelihood(obs, Sigma_H=0.0)
        assert np.isfinite(ll), f"log_likelihood={ll} not finite"


class TestInferenceDocSync:
    """No doc may claim Bayes factor or full evidence maturity."""

    def test_no_bayes_factor_claim(self):
        """No doc may positively claim 'Bayes factor headline' outside Forbidden Claims."""
        for f in ["README.md", "SUPPORTED_CAPABILITIES.md"]:
            text = open(f).read()
            forbidden_start = text.find("## Forbidden Claims")
            for i, line in enumerate(text.split('\n')):
                lower = line.lower()
                # Only catch POSITIVE claims, not blockers or forbidden statements
                if 'bayes factor' in lower and 'headline' in lower:
                    pos = sum(len(l)+1 for l in text.split('\n')[:i])
                    if forbidden_start != -1 and pos >= forbidden_start:
                        continue
                    # Skip lines that negate the claim
                    if any(neg in lower for neg in ['no ', 'not ', 'forbidden', 'exploratory', '—']):
                        continue
                    pytest.fail(f"{f}: Bayes factor claim: {line.strip()[:80]}")

    def test_no_evidence_ready_claim(self):
        """No doc may claim 'evidence-grade' or 'model comparison ready'."""
        for f in ["README.md", "SUPPORTED_CAPABILITIES.md"]:
            text = open(f).read()
            forbidden_start = text.find("## Forbidden Claims")
            for i, line in enumerate(text.split('\n')):
                lower = line.lower()
                if 'evidence-grade' in lower or 'model comparison ready' in lower:
                    pos = sum(len(l)+1 for l in text.split('\n')[:i])
                    if forbidden_start != -1 and pos >= forbidden_start:
                        continue
                    pytest.fail(f"{f}: evidence claim: {line.strip()[:80]}")
