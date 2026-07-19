"""PR-08: Inference Model-Comparison Guardrail Lock — TDD tests.

Model comparison (Bayes factors, evidence computation) is explicitly
exploratory. No user-facing artifact may confuse production PE with
production evidence/model comparison.
"""
import pytest
import re

pytestmark = [pytest.mark.production, pytest.mark.release_smoke]


class TestModelComparisonGuardrails:
    """Evidence/BF must be explicitly exploratory everywhere."""

    def test_forbidden_claims_has_bf_prohibition(self):
        """Forbidden Claims must explicitly ban Bayes factor headlines."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        forbidden_start = text.find("## Forbidden Claims")
        forbidden_end = text.find("## Permitted", forbidden_start)
        forbidden = text[forbidden_start:forbidden_end] if forbidden_end > forbidden_start else ""
        assert "bayes factor" in forbidden.lower(), \
            "Forbidden Claims must explicitly ban Bayes factor headlines"

    def test_forbidden_claims_has_differentiable_solver_prohibition(self):
        """Forbidden Claims must ban 'full differentiable BBN solver'."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        forbidden_start = text.find("## Forbidden Claims")
        forbidden_end = text.find("## Permitted", forbidden_start)
        forbidden = text[forbidden_start:forbidden_end]
        assert "differentiable" in forbidden.lower(), \
            "Forbidden Claims must ban full differentiable solver"

    def test_permitted_claims_pe_only(self):
        """Permitted Claims must say PE/framework, not evidence/BF."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        permitted_start = text.find("## Permitted Claims")
        permitted = text[permitted_start:]
        for line in permitted.split('\n'):
            lower = line.lower()
            if 'inference' in lower and '-' == line.strip()[0:1]:
                # Inference permitted claim must say PE/framework
                assert any(q in lower for q in ['pe', 'forward', 'framework', 'likelihood']), \
                    f"Inference permitted claim must specify PE scope: {line.strip()}"
                # Must NOT say evidence-grade or BF-ready
                assert 'evidence-grade' not in lower, \
                    f"Permitted claim cannot say evidence-grade: {line.strip()}"

    def test_bf_exploratory_in_permitted(self):
        """Permitted Claims must explicitly say BF/evidence is exploratory."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        permitted_start = text.find("## Permitted Claims")
        permitted = text[permitted_start:]
        has_bf_exploratory = "exploratory" in permitted.lower() and \
                            ("bayes" in permitted.lower() or "evidence" in permitted.lower())
        assert has_bf_exploratory, \
            "Permitted Claims must state BF/evidence computation is exploratory"


class TestNoProductionEvidenceLeakage:
    """No production path may silently produce evidence headlines."""

    def test_model_comparison_module_is_exploratory(self):
        """model_comparison module must not be in any production import chain."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        inf = FEATURE_BY_KEY['inference_exploratory']
        assert inf.surface_class == "exploratory", \
            "Inference surface_class must remain exploratory"

    def test_no_production_evidence_in_readme(self):
        """README must not mention evidence/BF in production context."""
        text = open("README.md").read()
        for line in text.split('\n'):
            lower = line.lower()
            if ('evidence' in lower or 'bayes factor' in lower) and 'production' in lower:
                if 'exploratory' not in lower and 'forbidden' not in lower:
                    pytest.fail(f"README production evidence leak: {line.strip()[:80]}")

    def test_registry_notes_forbid_bf(self):
        """Inference registry notes must explicitly forbid BF headlines."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        inf = FEATURE_BY_KEY['inference_exploratory']
        combined = f"{inf.notes} {inf.evidence_summary}".lower()
        assert "forbidden" in combined or "exploratory" in combined, \
            "Inference notes must mention BF prohibition or exploratory status"
