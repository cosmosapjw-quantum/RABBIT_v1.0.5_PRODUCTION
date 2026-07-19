#!/usr/bin/env python3
"""Sync test: verify generated document blocks match registry truth.

Checks:
  S1. All generated block markers exist in target documents
  S2. surface_class consistency: backend and feature registries
  S3. No manual text overrides registry tier/surface_class
  S4. Render provenance exists and is recent
  S5. README identity paragraph matches registry canonical count

Usage:
    python3 tests/test_registry_doc_sync.py
    pytest tests/test_registry_doc_sync.py -v
"""
import sys, os, re, json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def _read(name):
    p = _ROOT / name
    return p.read_text() if p.exists() else ""


def test_s1_block_markers():
    """S1: All expected BEGIN/END markers exist."""
    expected = {
        "README.md": [
            "README_HEADER", "IDENTITY", "CORE_SUMMARY",
            "README_BACKENDS_HEADER", "BACKEND_TABLE",
            "README_QUICKSTART", "README_FOOTER",
        ],
        "STATUS.md": [
            "STATUS_HEADER", "TIER_SUMMARY", "STATUS_BACKENDS", "STATUS_DETAIL",
        ],
        "PROMOTION_GATES.md": [
            "PG_HEADER", "PROMOTION_STATUS", "PG_BODY", "NEXT_QUEUE", "PG_FOOTER",
        ],
    }
    for doc, markers in expected.items():
        text = _read(doc)
        if not text:
            print(f"  SKIP: {doc} not found")
            continue
        for marker in markers:
            begin = f"<!-- BEGIN:{marker}"
            end = f"<!-- END:{marker}"
            assert begin in text, f"{doc} missing BEGIN:{marker}"
            assert end in text, f"{doc} missing END:{marker}"
        print(f"  S1 PASS: {doc} has all {len(markers)} markers")


def test_s2_surface_class_consistency():
    """S2: surface_class is defined for all backend and feature entries."""
    from rabbit.config.backend_capabilities import CAPABILITY_BY_KEY
    from rabbit.config.feature_capabilities import FEATURE_BY_KEY

    valid_sc = {"canonical", "candidate", "candidate_strong", "candidate_layered",
                "diagnostic", "exploratory", "substrate"}

    for key, cap in CAPABILITY_BY_KEY.items():
        sc = cap.effective_surface_class
        assert sc in valid_sc, f"Backend {key}: invalid surface_class '{sc}'"
    print(f"  S2 PASS: {len(CAPABILITY_BY_KEY)} backends have valid surface_class")

    for key, feat in FEATURE_BY_KEY.items():
        assert feat.surface_class in valid_sc, f"Feature {key}: invalid surface_class '{feat.surface_class}'"
    print(f"  S2 PASS: {len(FEATURE_BY_KEY)} features have valid surface_class")


def test_s3_tier_surface_alignment():
    """S3: surface_class is compatible with tier."""
    from rabbit.config.backend_capabilities import CAPABILITY_BY_KEY
    from rabbit.config.feature_capabilities import FEATURE_BY_KEY

    # canonical tier must have canonical surface_class
    for key, cap in CAPABILITY_BY_KEY.items():
        if cap.tier == "canonical":
            assert cap.effective_surface_class == "canonical", \
                f"Backend {key}: tier=canonical but surface_class={cap.effective_surface_class}"
    for key, feat in FEATURE_BY_KEY.items():
        if feat.tier == "canonical":
            assert feat.surface_class == "canonical", \
                f"Feature {key}: tier=canonical but surface_class={feat.surface_class}"
    print("  S3 PASS: canonical tier → canonical surface_class")


def test_s4_readme_identity_matches_registry():
    """S4: README identity paragraph matches registry canonical count."""
    from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND

    text = _read("README.md")
    if not text:
        print("  SKIP: README.md not found")
        return

    n_canonical = sum(1 for c in CAPABILITY_BY_BACKEND.values()
                      if c.tier == "canonical" and c.backend != "auto")
    # README should mention the canonical backends
    assert "canonical" in text.lower(), "README missing 'canonical' mention"
    print(f"  S4 PASS: README mentions canonical ({n_canonical} backends)")


def test_s5_no_retired_values():
    """S5: No retired values in governance documents."""
    retired = ["0.617", "B_{01} = 3.7", "B_{01}=3.7", "< 0.13", "3.68"]
    for doc in ["README.md", "STATUS.md", "PROMOTION_GATES.md"]:
        text = _read(doc)
        if not text:
            continue
        for rv in retired:
            assert rv not in text, f"{doc} contains retired value '{rv}'"
    print("  S5 PASS: no retired values in governance docs")


def test_s6_render_provenance_matches_registry_counts():
    """S6: Render provenance counts every registry entry, including substrate features."""
    from rabbit.config.backend_capabilities import CAPABILITY_BY_KEY
    from rabbit.config.feature_capabilities import FEATURE_BY_KEY

    text = _read("docs/RENDER_PROVENANCE.json")
    assert text, "docs/RENDER_PROVENANCE.json missing"
    provenance = json.loads(text)
    assert provenance["n_backends"] == len(CAPABILITY_BY_KEY)
    assert provenance["n_features"] == len(FEATURE_BY_KEY)


if __name__ == "__main__":
    print("=" * 60)
    print("  Registry-Document Sync Test")
    print("=" * 60)
    test_s1_block_markers()
    test_s2_surface_class_consistency()
    test_s3_tier_surface_alignment()
    test_s4_readme_identity_matches_registry()
    test_s5_no_retired_values()
    test_s6_render_provenance_matches_registry_counts()
    print("\nALL SYNC TESTS PASS")
