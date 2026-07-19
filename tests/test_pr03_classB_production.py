"""PR-03: Class B 6-label Reduced-Mask Production Lock — TDD tests.

Documented candidate slices: V, IV, canonical III, h-locked VI_h/VII_h
representatives, and canonical VI_{-1/9} at CL0. All six reduced-mask labels
now have conservative representative gold gates; full Class B remains candidate.
"""
import pytest
import json
import numpy as np

pytestmark = [pytest.mark.production, pytest.mark.release_smoke]

_PRODUCTION_TYPE = "TYPE_V"


class TestClassBRegimeGuards:
    """Broader Class B must not inherit Type-V-only gold wording."""

    def test_classB_documented_slices_are_all_six_reduced_mask_labels(self):
        """Registry must specify the currently documented 6-label candidate slices."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['classB_bbn']
        combined = f"{f.evidence_summary} {f.short_summary} {f.notes}"
        assert all(tag in combined for tag in ["Type V", "Type VI_h", "Type VII_h", "VI_{-1/9}"]), \
            "Class B registry missing documented 6-label reduced-mask slice specification"

    def test_classB_short_summary_mentions_candidate_not_gold(self):
        """Short summary may mention 6-label gold but must keep candidate posture."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['classB_bbn']
        summary = f.short_summary.lower()
        assert "candidate" in summary and "6-label" in summary and "gold" in summary, \
            "Class B short_summary must scope gold wording to reduced-mask representatives and keep candidate posture"


class TestClassBSmokeAnchors:
    """Type V fixture is now a conservative frame-variable gold anchor."""

    def test_gold_fixture_exists(self):
        """Fixture must still keep the Type V smoke anchor entry."""
        gold = json.load(open("tests/fixtures/jax_bbn_gold.json"))
        assert "classB_typeV" in gold, "Missing classB_typeV gold fixture"

    def test_gold_yp_value(self):
        """Type V anchor Yp should remain at the conservative A=1e-4 value."""
        gold = json.load(open("tests/fixtures/jax_bbn_gold.json"))
        entry = gold.get("classB_typeV", {})
        yp = entry.get("Yp", 0)
        assert abs(yp - 0.24237194337494503) < 1e-12, \
            f"Type V Yp={yp} doesn't match conservative gold anchor"

    def test_gold_delta_yp_positive(self):
        """Type V DYp(A) should remain positive in the retained smoke anchor."""
        gold = json.load(open("tests/fixtures/jax_bbn_gold.json"))
        entry = gold.get("classB_typeV", {})
        dyp = entry.get("DYp_A", 0)
        assert dyp > 0, f"Type V DYp(A) should be positive, got {dyp}"


class TestClassBLayeredScope:
    """6/6/6/6 layered scope must be consistent across all docs."""

    def test_layered_scope_in_supported_capabilities(self):
        """SUPPORTED_CAPABILITIES must have Class B validation layers."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        assert "Class B validation layers" in text
        assert "6 types" in text or "6-type" in text

    def test_layered_scope_in_registry(self):
        """Feature registry must mention the updated 6/6/6/6 layered scope."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['classB_bbn']
        ls = f.layered_scope
        assert (ls["geometry"], ls["family_envelope"], ls["bbn_smoke"], ls["gold_locked"]) == (6, 6, 6, 6), \
            "Class B registry missing updated 6/6/6/6 layered scope"
        assert ls["gold_type"] == "TYPE_V/TYPE_IV/TYPE_III/TYPE_VIH/TYPE_VIIH/TYPE_VI_M19"


class TestClassBDocSync:
    """No doc may imply broader Class B is production."""

    def test_no_broad_classB_production_claim(self):
        """No doc may say 'Class B production-ready' outside Forbidden Claims."""
        for f in ["README.md", "STATUS.md", "SUPPORTED_CAPABILITIES.md",
                   "PROMOTION_GATES.md"]:
            text = open(f).read()
            forbidden_start = text.find("## Forbidden Claims")
            for i, line in enumerate(text.split('\n')):
                lower = line.lower()
                if 'class b' in lower and 'production' in lower:
                    pos = sum(len(l)+1 for l in text.split('\n')[:i])
                    if forbidden_start != -1 and pos >= forbidden_start:
                        continue
                    has_qualifier = any(q in lower for q in
                        ['type v', 'type_v', 'type iii', 'single', 'slice', 'only', 'locked', 'documented'])
                    if not has_qualifier:
                        pytest.fail(f"{f}: broad Class B production claim: {line.strip()[:80]}")
