"""PR-02 tilted component locks after public endpoint retirement."""
import pytest

pytestmark = [pytest.mark.production, pytest.mark.release_smoke]

class TestTiltedRegimeGuards:
    """The component survives, while public endpoint dispatch fails closed."""

    def test_tilted_public_dispatch_is_retired(self):
        from rabbit.inference.forward_likelihood import canonical_forward_solver

        with pytest.raises(ValueError, match="retired"):
            canonical_forward_solver(backend='jax_tilted')

    def test_tilted_component_config_remains_constructible(self):
        from rabbit.jax.run_tilted_bbn import TiltedBBNConfig

        cfg = TiltedBBNConfig(v0=1e-4, correction_level=0)
        assert cfg.v0 == pytest.approx(1e-4)


class TestTiltedDocSync:
    """All docs must use the same narrow slice wording."""

    def test_no_broad_tilted_production_claim(self):
        """No doc may claim 'tilted production-ready' without Type I/scalar/v0 qualifier."""
        for f in ["README.md", "STATUS.md", "SUPPORTED_CAPABILITIES.md",
                   "PROMOTION_GATES.md"]:
            text = open(f).read()
            forbidden_start = text.find("## Forbidden Claims")
            for i, line in enumerate(text.split('\n')):
                lower = line.lower()
                if 'tilt' in lower and 'production' in lower:
                    pos = sum(len(l)+1 for l in text.split('\n')[:i])
                    if forbidden_start != -1 and pos >= forbidden_start:
                        continue
                    has_qualifier = any(q in lower for q in
                        ['type i', 'scalar', 'v0', 'documented', 'slice', 'verified'])
                    if not has_qualifier:
                        pytest.fail(f"{f}: broad tilted production claim: {line.strip()[:80]}")

    def test_tilted_registry_has_v0_scope(self):
        """Feature registry must specify v0 window and Type I scope."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        t = FEATURE_BY_KEY['tilted_scalar']
        combined = f"{t.evidence_summary} {t.short_summary} {t.notes}"
        assert "v0" in combined.lower() or "v₀" in combined, \
            "Tilted registry missing v0 scope"
        assert "Type I" in combined or "type I" in combined or "Type_I" in combined, \
            "Tilted registry missing Type I scope"
