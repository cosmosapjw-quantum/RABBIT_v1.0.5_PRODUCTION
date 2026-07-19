"""PR-05: Class A Reduced-Slice Production Lock — TDD tests.

Production slice: 6-type Class A geometry + reduced κ-cascade transport.
Type I reduction closes exactly. Transport is reduced (not exact curved PSTF).
"""
import pytest

pytestmark = [pytest.mark.production, pytest.mark.release_smoke]

CLASS_A_TYPES = ["TYPE_I", "TYPE_II", "TYPE_VI0", "TYPE_VII0", "TYPE_VIII", "TYPE_IX"]


class TestClassAReducedSlice:
    """Registry must specify reduced-slice scope."""

    def test_registry_specifies_reduced_transport(self):
        """Feature registry must mention 'reduced' transport."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['classA_transport']
        combined = f"{f.evidence_summary} {f.short_summary} {f.notes}"
        assert "reduced" in combined.lower(), \
            "Class A registry must specify 'reduced' transport scope"

    def test_registry_specifies_6_types(self):
        """Feature registry must mention 6-type support."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['classA_transport']
        assert "6" in f.evidence_summary, "Class A must mention 6-type scope"

    def test_type_I_reduction_exact(self):
        """N1=N2=N3=0 must give zero curvature (exact Type I closure)."""
        from rabbit.geometry.general_classA import curvature_source_S, gauss_curvature_K
        Sp, Sm = curvature_source_S(0, 0, 0)
        K = gauss_curvature_K(0, 0, 0)
        assert abs(Sp) < 1e-15 and abs(Sm) < 1e-15 and abs(K) < 1e-15, \
            "Type I reduction must be exact (zero curvature)"


class TestClassADocSync:
    """No doc may claim broad family-wide production."""

    def test_no_broad_classA_production_claim(self):
        """No doc may say 'Class A production-ready' without reduced/slice qualifier."""
        for f in ["README.md", "STATUS.md", "SUPPORTED_CAPABILITIES.md",
                   "PROMOTION_GATES.md"]:
            text = open(f).read()
            forbidden_start = text.find("## Forbidden Claims")
            permitted_start = text.find("## Permitted Claims")
            for i, line in enumerate(text.split('\n')):
                lower = line.lower()
                if 'class a' in lower and 'production' in lower:
                    pos = sum(len(l)+1 for l in text.split('\n')[:i])
                    if forbidden_start != -1 and pos >= forbidden_start:
                        continue
                    if permitted_start != -1 and pos >= permitted_start:
                        # Permitted section — must have qualifier
                        has_q = any(q in lower for q in ['reduced', 'slice', 'documented'])
                        if not has_q:
                            pytest.fail(f"{f}: broad Class A permitted claim: {line.strip()[:80]}")
                        continue
                    has_q = any(q in lower for q in ['reduced', 'slice', 'documented', 'locked'])
                    if not has_q:
                        pytest.fail(f"{f}: broad Class A production claim: {line.strip()[:80]}")

    def test_registry_no_blockers_for_reduced_slice(self):
        """Reduced slice should have no blockers after PR-05."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['classA_transport']
        assert len(f.blockers) == 0, f"Class A reduced slice should have no blockers: {f.blockers}"
