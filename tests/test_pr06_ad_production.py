"""PR-06: AD Diagnostic Production Lock — TDD tests.

AD is production-grade ONLY as a diagnostic tool on canonical Type-I,
for the parameter subset {eta, tau_n}. Not a full differentiable solver.
"""
import pytest

pytestmark = [pytest.mark.production, pytest.mark.release_smoke]


class TestADRegistryScope:
    """Registry must specify diagnostic-only scope."""

    def test_ad_is_diagnostic_surface_class(self):
        """AD must have surface_class=diagnostic."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['ad_diagnostic']
        assert f.surface_class == "diagnostic", \
            f"AD surface_class should be 'diagnostic', got '{f.surface_class}'"

    def test_ad_short_summary_says_diagnostic(self):
        """AD short_summary must mention diagnostic scope."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['ad_diagnostic']
        assert "diagnostic" in f.short_summary.lower() or "not full" in f.short_summary.lower(), \
            "AD short_summary must indicate diagnostic scope"

    def test_ad_no_blockers_for_diagnostic_slice(self):
        """After PR-06, diagnostic slice should have no blockers."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['ad_diagnostic']
        assert len(f.blockers) == 0, \
            f"AD diagnostic slice should have no blockers: {f.blockers}"

    def test_ad_validation_mode_is_diagnostic(self):
        """AD validation_mode must remain 'diagnostic', not 'full'."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['ad_diagnostic']
        assert f.validation_mode == "diagnostic", \
            f"AD validation_mode must be 'diagnostic', got '{f.validation_mode}'"


class TestADDocSync:
    """No doc may claim 'full differentiable solver'."""

    def test_no_full_differentiable_claim(self):
        """No doc may positively claim 'differentiable BBN solver' without diagnostic qualifier."""
        for f in ["README.md", "STATUS.md", "SUPPORTED_CAPABILITIES.md",
                   "PROMOTION_GATES.md"]:
            text = open(f).read()
            forbidden_start = text.find("## Forbidden Claims")
            for i, line in enumerate(text.split('\n')):
                lower = line.lower()
                if 'differentiable' in lower and 'solver' in lower:
                    pos = sum(len(l)+1 for l in text.split('\n')[:i])
                    if forbidden_start != -1 and pos >= forbidden_start:
                        continue
                    # Skip lines that negate the claim
                    if any(neg in lower for neg in ['not full', 'not a', 'diagnostic', 'toy']):
                        continue
                    pytest.fail(f"{f}: 'differentiable solver' claim: {line.strip()[:80]}")

    def test_no_inference_ready_claim(self):
        """No doc may say 'gradient-based inference ready' outside Forbidden."""
        for f in ["README.md", "SUPPORTED_CAPABILITIES.md"]:
            text = open(f).read()
            forbidden_start = text.find("## Forbidden Claims")
            for i, line in enumerate(text.split('\n')):
                lower = line.lower()
                if 'inference ready' in lower or 'inference-ready' in lower:
                    pos = sum(len(l)+1 for l in text.split('\n')[:i])
                    if forbidden_start != -1 and pos >= forbidden_start:
                        continue
                    pytest.fail(f"{f}: inference-ready claim: {line.strip()[:80]}")


class TestADComponentLevel:
    """AD gradient bridge must be importable and have correct API."""

    def test_gradient_bridge_importable(self):
        """gradient_bridge module must be importable."""
        from rabbit.jax.gradient_bridge import make_differentiable_solve, gradient_check
        assert callable(make_differentiable_solve)
        assert callable(gradient_check)

    def test_ad_parity_value_in_registry(self):
        """Registry must record the AD/FD parity value."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['ad_diagnostic']
        combined = f"{f.evidence_summary} {f.short_summary}"
        assert "8.2e-8" in combined or "8.2e-08" in combined, \
            "AD registry must record custom_vjp/FD parity value"
