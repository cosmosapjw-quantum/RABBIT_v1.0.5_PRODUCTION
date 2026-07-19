"""PR-00: Promotion Infrastructure Completion — TDD tests.

These tests verify that registry-driven documentation prevents drift
by construction. They enforce:
1. Generated block idempotency
2. No manual summaries that duplicate generated registry content
3. Registry surface_class governs all feature grouping
4. Stale generated blocks are caught
5. Section semantics are structurally sound
"""
import pytest
import subprocess
import sys
import re
import os

pytestmark = [pytest.mark.production, pytest.mark.release_smoke]


class TestGeneratedBlockIdempotency:
    """Verify render --apply is idempotent (running twice changes nothing)."""

    def test_render_apply_is_idempotent(self):
        """Running render --apply twice must produce identical docs."""
        # First apply
        r1 = subprocess.run(
            [sys.executable, "scripts/render_capability_tables.py", "--apply"],
            capture_output=True, text=True, timeout=30
        )
        assert r1.returncode == 0

        # Snapshot all docs
        docs = {}
        for f in ["README.md", "STATUS.md", "SUPPORTED_CAPABILITIES.md",
                   "PROMOTION_GATES.md", "docs/BACKEND_CAPABILITY_MATRIX.md",
                   "docs/RENDER_PROVENANCE.json"]:
            docs[f] = open(f).read()

        # Second apply
        r2 = subprocess.run(
            [sys.executable, "scripts/render_capability_tables.py", "--apply"],
            capture_output=True, text=True, timeout=30
        )
        assert r2.returncode == 0

        # Compare
        for f, content in docs.items():
            current = open(f).read()
            assert current == content, f"{f} changed after second render --apply (not idempotent)"


class TestNoDuplicateSummaries:
    """Verify no manual summary block duplicates generated registry content."""

    def test_no_manual_diagnostic_exploratory_after_feature_table(self):
        """No manual Diagnostic/Exploratory table after generated FEATURE_TABLE."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        end_idx = text.find("<!-- END:FEATURE_TABLE -->")
        if end_idx == -1:
            pytest.skip("No FEATURE_TABLE block")
        after = text[end_idx:]
        assert "### Diagnostic / Exploratory" not in after, \
            "Manual Diagnostic/Exploratory duplicates generated feature registry"
        # Also no manual candidate-strong/layered tables
        assert "### Candidate-strong" not in after, \
            "Manual Candidate-strong duplicates generated feature registry"

    def test_no_manual_backend_table_outside_markers(self):
        """Backend table rows must only appear inside generated blocks."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        end_idx = text.find("<!-- END:BACKEND_TABLE -->")
        feature_start = text.find("<!-- BEGIN:FEATURE_TABLE")
        if end_idx == -1:
            pytest.skip("No BACKEND_TABLE block")
        between = text[end_idx:feature_start] if feature_start > end_idx else ""
        # No duplicate backend tier rows between the blocks
        assert "| `jax`" not in between, \
            "Backend row appears outside generated BACKEND_TABLE block"


class TestSurfaceClassAuthority:
    """Verify surface_class from registry governs all feature grouping."""

    def test_readme_candidate_grouping_matches_registry(self):
        """README candidate perimeter must group by registry surface_class."""
        text = open("README.md").read()
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        for key, feat in FEATURE_BY_KEY.items():
            if feat.tier == "candidate":
                sc_label = feat.surface_class.replace("_", "-").title()
                # The surface class label should appear in README
                assert sc_label.lower() in text.lower() or feat.name in text, \
                    f"README missing surface_class grouping for {feat.name} ({sc_label})"

    def test_promotion_queue_surface_classes_match_registry(self):
        """PROMOTION_GATES next queue must use registry surface_class values."""
        text = open("PROMOTION_GATES.md").read()
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        for key, feat in FEATURE_BY_KEY.items():
            if feat.tier == "candidate":
                sc = feat.surface_class.replace("_", "-")
                assert sc in text, \
                    f"PROMOTION_GATES missing surface_class '{sc}' for {feat.name}"

    def test_feature_table_groups_by_surface_class(self):
        """SUPPORTED_CAPABILITIES feature table must have surface_class headings."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        for sc in ["Canonical", "Candidate-Strong", "Candidate-Layered",
                    "Diagnostic", "Exploratory"]:
            assert f"### {sc}" in text, \
                f"SUPPORTED_CAPABILITIES missing surface_class heading '{sc}'"


class TestSectionSemantics:
    """Verify document section structure is logically sound."""

    def test_no_canonical_under_candidate_heading(self):
        """No canonical feature/backend may appear under a 'Candidate' heading."""
        text = open("README.md").read()
        cand_start = text.find("## Candidate Features")
        if cand_start == -1:
            return
        next_section = text.find("\n## ", cand_start + 1)
        cand_text = text[cand_start:next_section] if next_section != -1 else text[cand_start:]
        from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND
        for name, cap in CAPABILITY_BY_BACKEND.items():
            if cap.tier == "canonical" and name not in ("auto",):
                assert f"`{name}` | **canonical**" not in cand_text, \
                    f"Canonical backend '{name}' in Candidate section"

    def test_generated_blocks_have_matching_markers(self):
        """Every BEGIN marker must have a matching END marker."""
        for f in ["README.md", "STATUS.md", "SUPPORTED_CAPABILITIES.md",
                   "PROMOTION_GATES.md"]:
            text = open(f).read()
            begins = re.findall(r'<!-- BEGIN:(\w+)', text)
            ends = re.findall(r'<!-- END:(\w+)', text)
            for b in begins:
                assert b in ends, f"{f}: BEGIN:{b} has no matching END"

    def test_status_tier_counts_match_registry(self):
        """STATUS tier summary counts must match registry."""
        from rabbit.config.backend_capabilities import CAPABILITY_BY_KEY
        counts = {}
        for k, c in CAPABILITY_BY_KEY.items():
            counts[c.tier] = counts.get(c.tier, 0) + 1
        text = open("STATUS.md").read()
        for tier, count in counts.items():
            pattern = rf'\| \*\*{tier}\*\* \|.*?\| {count} \|'
            assert re.search(pattern, text), \
                f"STATUS tier '{tier}' count should be {count}"


class TestManifestProvenance:
    """Verify RELEASE_MANIFEST has required provenance fields."""

    def test_manifest_has_required_fields(self):
        import json
        if not os.path.exists("RELEASE_MANIFEST.json"):
            pytest.skip("No manifest")
        m = json.load(open("RELEASE_MANIFEST.json"))
        required = ["package", "version", "build_timestamp", "python_version",
                     "platform", "optional_deps", "test_counts",
                     "exact_count_sync_build_env", "counts_portable_across_envs",
                     "fast_gates", "pipeline", "render_tables_applied", "render_scope"]
        for field in required:
            assert field in m, f"RELEASE_MANIFEST missing field: {field}"

    def test_manifest_render_applied(self):
        import json
        if not os.path.exists("RELEASE_MANIFEST.json"):
            pytest.skip("No manifest")
        m = json.load(open("RELEASE_MANIFEST.json"))
        assert m.get("render_tables_applied") is True, \
            "RELEASE_MANIFEST: render_tables_applied must be True"


class TestClaimBoundary:
    """Verify the exact claim boundary of PR-00.

    Claimed: "Promotion-sensitive metadata (tier counts, backend tables,
    feature groupings by surface_class, promotion status, next-promotion queue)
    is registry-generated and self-heals on render --apply."

    NOT claimed: "All doc content is generated" (STATUS capability descriptions,
    policy text, and supplementary detail remain hand-maintained).
    """

    def test_all_promotion_sensitive_blocks_are_generated(self):
        """Every promotion-sensitive block must have generated markers."""
        checks = [
            ("README.md", "BACKEND_TABLE"),
            ("README.md", "CORE_SUMMARY"),
            ("SUPPORTED_CAPABILITIES.md", "BACKEND_TABLE"),
            ("SUPPORTED_CAPABILITIES.md", "FEATURE_TABLE"),
            ("STATUS.md", "TIER_SUMMARY"),
            ("PROMOTION_GATES.md", "PROMOTION_STATUS"),
            ("PROMOTION_GATES.md", "NEXT_QUEUE"),
        ]
        for filepath, block_name in checks:
            text = open(filepath).read()
            assert f"<!-- BEGIN:{block_name}" in text, \
                f"{filepath} missing generated block {block_name}"
            assert f"<!-- END:{block_name} -->" in text, \
                f"{filepath} missing end marker for {block_name}"

    def test_generated_blocks_self_heal(self):
        """Corrupting inside a generated block and re-rendering must restore it."""
        import subprocess
        filepath = "SUPPORTED_CAPABILITIES.md"
        original = open(filepath).read()
        # Corrupt ONLY inside the BACKEND_TABLE generated block
        begin = original.find("<!-- BEGIN:BACKEND_TABLE")
        end = original.find("<!-- END:BACKEND_TABLE -->")
        if begin == -1 or end == -1:
            pytest.skip("No BACKEND_TABLE block")
        block_content = original[begin:end]
        corrupted_block = block_content.replace("**canonical**", "CORRUPTED_TIER")
        corrupted = original[:begin] + corrupted_block + original[end:]
        open(filepath, "w").write(corrupted)
        # Render --apply
        subprocess.run([sys.executable, "scripts/render_capability_tables.py", "--apply"],
                       capture_output=True, timeout=30)
        # Verify healed
        healed = open(filepath).read()
        healed_block = healed[healed.find("<!-- BEGIN:BACKEND_TABLE"):healed.find("<!-- END:BACKEND_TABLE -->")]
        assert "CORRUPTED_TIER" not in healed_block, \
            "Generated BACKEND_TABLE block did not self-heal after corruption"
        assert "canonical" in healed_block, \
            "Generated BACKEND_TABLE block lost canonical tier after healing"

    def test_manual_scope_is_declared(self):
        """Documents must explicitly state their generation scope."""
        for f in ["SUPPORTED_CAPABILITIES.md", "PROMOTION_GATES.md"]:
            text = open(f).read()
            has_scope = ("Generated vs manual scope" in text or
                        "Generated scope" in text)
            assert has_scope, f"{f} missing governance scope declaration"

    def test_render_scope_in_manifest(self):
        """RELEASE_MANIFEST must declare what render --apply covers."""
        import json
        if not os.path.exists("RELEASE_MANIFEST.json"):
            pytest.skip("No manifest")
        m = json.load(open("RELEASE_MANIFEST.json"))
        scope = m.get("render_scope", [])
        assert len(scope) >= 4, \
            f"RELEASE_MANIFEST render_scope too narrow: {scope}"
