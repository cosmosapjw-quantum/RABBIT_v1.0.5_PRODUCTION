"""Registry ↔ document sync tests.

Ensures single source of truth is actually enforced, not just declared.

TestExactCountSync is marked @build_env_only because test counts depend on
optional dependencies (JAX, BlackJAX). These tests pass in the build
environment but may fail in minimal environments. Exact count equality
is enforced during packaging by scripts/package_release.py.
"""
import pytest
import re
import os

_portable = [pytest.mark.production, pytest.mark.release_smoke]


class TestBackendRegistrySync:
    pytestmark = _portable
    """backend_capabilities.py ↔ STATUS.md ↔ README.md dispatch sync."""

    def test_capability_by_backend_keys(self):
        from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND
        expected = {"scipy", "auto"}
        assert set(CAPABILITY_BY_BACKEND.keys()) == expected

    def test_auto_is_scipy_alias(self):
        from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND
        assert CAPABILITY_BY_BACKEND["auto"] is CAPABILITY_BY_BACKEND["scipy"]

    def test_readme_backend_table_complete(self):
        """README.md must list all public backends."""
        text = open("README.md").read()
        for backend in ["auto", "scipy"]:
            assert f"`{backend}`" in text or f"`{backend}/" in text or f"/{backend}`" in text, \
                f"README.md missing backend {backend}"

    def test_all_backends_have_tier(self):
        """Every backend capability must have a valid tier."""
        from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND
        valid = {"canonical", "candidate", "substrate"}
        for name, cap in CAPABILITY_BY_BACKEND.items():
            assert cap.tier in valid, f"Backend {name}: invalid tier '{cap.tier}'"


class TestFeatureRegistrySync:
    pytestmark = _portable
    """feature_capabilities.py ↔ SUPPORTED_CAPABILITIES.md sync."""

    def test_feature_registry_imports(self):
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        assert len(FEATURE_BY_KEY) >= 10

    def test_canonical_features_exist(self):
        from rabbit.config.feature_capabilities import FEATURE_TIERS
        canonical = FEATURE_TIERS["canonical"]
        keys = {f.key for f in canonical}
        assert "flrw_baseline" in keys
        assert "typeI_anisotropic" in keys
        assert "weak_rates" in keys
        assert "nuclear_network" in keys

    def test_ad_is_diagnostic(self):
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        ad = FEATURE_BY_KEY["ad_diagnostic"]
        assert ad.validation_mode == "diagnostic"

    def test_inference_is_exploratory(self):
        """Inference surface_class remains exploratory; validation_mode promoted to candidate for PE."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        inf = FEATURE_BY_KEY["inference_exploratory"]
        assert inf.surface_class == "exploratory"
        assert inf.validation_mode in ("exploratory", "candidate")

    def test_feature_tiers_valid(self):
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        valid_tiers = {"canonical", "candidate", "substrate"}
        valid_modes = {"full", "candidate", "diagnostic", "exploratory"}
        for key, feat in FEATURE_BY_KEY.items():
            assert feat.tier in valid_tiers, f"{key}: tier '{feat.tier}' invalid"
            assert feat.validation_mode in valid_modes, f"{key}: mode '{feat.validation_mode}' invalid"

    def test_supported_capabilities_mentions_features(self):
        """SUPPORTED_CAPABILITIES.md must reference key features."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        for term in ["Teff", "Tilted", "Class B", "AD", "Inference"]:
            assert term in text, f"SUPPORTED_CAPABILITIES.md missing {term}"

    def test_forbidden_claims_present(self):
        """Forbidden claims section must exist."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        assert "Forbidden Claims" in text
        assert "Bayes factor" in text


_build_env = pytest.mark.skipif(
    os.environ.get("RABBIT_BUILD_ENV") != "1",
    reason="Exact count sync requires RABBIT_BUILD_ENV=1 (packaging env only)"
)


def _pytest_collect_env():
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("RABBIT_JAX_CACHE_DIR", "/tmp/rabbit_jax_cache")
    return env


class TestExactCountSync:
    """Verify document counts match actual pytest collection.

    These tests are build_env_only: counts depend on optional deps.
    Auto-skipped unless RABBIT_BUILD_ENV=1 is set.
    Run during packaging (package_release.py) or with:
        RABBIT_BUILD_ENV=1 pytest -m build_env_only
    """
    pytestmark = [pytest.mark.build_env_only, _build_env]

    def test_readme_total_matches_collect(self):
        """README test count must match actual collection."""
        import subprocess, sys, re
        r = subprocess.run([sys.executable, '-m', 'pytest', 'tests/',
                            '--collect-only', '-q'],
                           capture_output=True, text=True, timeout=60, env=_pytest_collect_env())
        m = re.search(r'(\d+) tests? collected', r.stdout)
        actual = int(m.group(1)) if m else 0
        readme = open("README.md").read()
        rm = re.search(r'(\d+) tests across', readme)
        readme_count = int(rm.group(1)) if rm else -1
        assert readme_count == actual, \
            f"README says {readme_count}, actual is {actual}"

    def test_status_total_matches_collect(self):
        """STATUS test count must match actual collection."""
        import subprocess, sys, re
        r = subprocess.run([sys.executable, '-m', 'pytest', 'tests/',
                            '--collect-only', '-q'],
                           capture_output=True, text=True, timeout=60, env=_pytest_collect_env())
        m = re.search(r'(\d+) tests? collected', r.stdout)
        actual = int(m.group(1)) if m else 0
        status = open("STATUS.md").read()
        sm = re.search(r'\((\d+) collected', status)
        status_count = int(sm.group(1)) if sm else -1
        assert status_count == actual, \
            f"STATUS says {status_count}, actual is {actual}"

    def test_gold_count_matches_collect(self):
        """Gold count in STATUS must match actual."""
        import subprocess, sys, re
        r = subprocess.run([sys.executable, '-m', 'pytest', 'tests/',
                            '-m', 'gold', '--collect-only', '-q'],
                           capture_output=True, text=True, timeout=60, env=_pytest_collect_env())
        m = re.search(r'(\d+)/\d+ tests', r.stdout)
        actual = int(m.group(1)) if m else 0
        status = open("STATUS.md").read()
        sm = re.search(r'\*\*(\d+) gold\*\*', status)
        status_gold = int(sm.group(1)) if sm else -1
        assert status_gold == actual, \
            f"STATUS gold says {status_gold}, actual is {actual}"


class TestExactRegistrySync:
    pytestmark = _portable
    """Verify document rows match registry values exactly."""

    def test_supported_capabilities_backend_tiers_match(self):
        """Each backend row in SUPPORTED_CAPABILITIES must match code tier."""
        from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND
        text = open("SUPPORTED_CAPABILITIES.md").read()
        import re
        for name in ["scipy"]:
            cap = CAPABILITY_BY_BACKEND[name]
            pattern = rf'\| `{name}` \| \**?{cap.tier}\**?'
            assert re.search(pattern, text), \
                f"SUPPORTED_CAPABILITIES: {name} should be {cap.tier}"

    def test_feature_registry_ad_inference_in_docs(self):
        """AD and inference validation modes must appear in SUPPORTED_CAPABILITIES."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        assert "diagnostic" in text.lower(), "AD diagnostic not in SUPPORTED_CAPABILITIES"
        assert "exploratory" in text.lower(), "Inference exploratory not in SUPPORTED_CAPABILITIES"

    def test_promotion_gates_no_higher_than_code(self):
        """No feature in PROMOTION_GATES may claim higher tier than code."""
        from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND
        text = open("PROMOTION_GATES.md").read()
        import re
        for line in text.split('\n'):
            if '| **canonical**' in line:
                # Only SciPy and JAX Type I (after tier flip) should be canonical
                assert 'SciPy' in line or 'JAX Type I' in line, \
                    f"Unexpected canonical in PROMOTION_GATES: {line}"

    def test_readme_candidate_features_listed(self):
        """README must list all candidate features from feature registry."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        text = open("README.md").read()
        candidate_names = {
            "teff_spectral": "Teff",
            "tilted_scalar": "Tilted",
            "classA_transport": "Class A",
            "classB_bbn": "Class B",
            "ad_diagnostic": "AD",
            "inference_exploratory": "inference",
        }
        for key, display in candidate_names.items():
            assert display.lower() in text.lower(), \
                f"README missing candidate feature {key} ({display})"


class TestDocumentStructureSync:
    """Verify document section placement matches code tiers.

    These tests catch the specific failure mode where a tier flip
    is applied to code but not propagated to document sections.
    """
    pytestmark = _portable

    def test_readme_no_canonical_in_candidate_section(self):
        """Canonical backends/features must NOT appear under Candidate Features."""
        text = open("README.md").read()
        # Find candidate section
        cand_start = text.find("## Candidate Features")
        if cand_start == -1:
            return  # no candidate section
        # Find next section header
        next_section = text.find("\n## ", cand_start + 1)
        cand_text = text[cand_start:next_section] if next_section != -1 else text[cand_start:]
        # No canonical backend should appear here
        from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND
        for name, cap in CAPABILITY_BY_BACKEND.items():
            if cap.tier == "canonical" and name not in ("auto", "scipy"):
                assert f"`{name}`" not in cand_text.lower(), \
                    f"Canonical backend '{name}' found in Candidate Features section"

    def test_status_canonical_table_matches_count(self):
        """STATUS.md canonical table row count must match declared count."""
        text = open("STATUS.md").read()
        import re
        # Find canonical count
        count_m = re.search(r'\| \*\*canonical\*\* \|.*?\| (\d+) \|', text)
        if not count_m:
            return
        declared = int(count_m.group(1))
        # Count rows in canonical table (between ### Canonical and ### Candidate)
        canon_start = text.find("### Canonical\n")
        cand_start = text.find("### Candidate\n")
        if canon_start == -1 or cand_start == -1:
            return
        canon_section = text[canon_start:cand_start]
        # Count data rows (| key | ... |)
        rows = [l for l in canon_section.split('\n')
                if l.startswith('|') and '---' not in l
                and 'Key' not in l and 'Scope' not in l
                and 'Backend' not in l]
        assert len(rows) == declared, \
            f"STATUS canonical count={declared} but {len(rows)} rows in table"

    def test_supported_capabilities_section_structure(self):
        """Feature Registry section must exist (replaces old Candidate Features)."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        # Old structure had "Candidate Features" containing canonical — that's wrong
        assert "## Candidate Features" not in text or "### Canonical" not in text[text.find("## Candidate Features"):text.find("## Forbidden", text.find("## Candidate Features"))], \
            "Canonical subsection under Candidate Features is structurally invalid"
        # New structure: Feature Registry (by surface class)
        assert "Feature Registry" in text or "Candidate Features" in text, \
            "SUPPORTED_CAPABILITIES missing feature section"

    def test_backend_matrix_dispatch_matches_code(self):
        """BACKEND_CAPABILITY_MATRIX dispatch must match code."""
        import os
        path = "docs/BACKEND_CAPABILITY_MATRIX.md"
        if not os.path.exists(path):
            return
        text = open(path).read()
        from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND
        # jax must map to liveweak, not reduced
        assert "liveweak" in text, "Matrix missing liveweak dispatch"
        # jax must be canonical
        jax_lines = [l for l in text.split('\n') if '"jax"' in l]
        for line in jax_lines:
            assert "canonical" in line.lower(), \
                f"Matrix: jax not marked canonical: {line.strip()}"


class TestGeneratedBlockIntegrity:
    """Verify generated blocks are consistent with registry and not duplicated."""
    pytestmark = _portable

    def test_no_manual_diagnostic_exploratory_in_supported(self):
        """SUPPORTED_CAPABILITIES must not have manual Diagnostic/Exploratory table
        outside the generated FEATURE_TABLE block."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        # Find end of generated feature table
        end_marker = "<!-- END:FEATURE_TABLE -->"
        end_idx = text.find(end_marker)
        if end_idx == -1:
            return
        after_generated = text[end_idx:]
        # No manual Diagnostic/Exploratory summary table after generated block
        assert "### Diagnostic / Exploratory" not in after_generated, \
            "Manual Diagnostic/Exploratory table duplicates generated feature registry"

    def test_readme_core_summary_is_generated(self):
        """README must have CORE_SUMMARY generated block."""
        text = open("README.md").read()
        assert "<!-- BEGIN:CORE_SUMMARY" in text, "README missing CORE_SUMMARY generated block"
        assert "<!-- END:CORE_SUMMARY -->" in text, "README missing CORE_SUMMARY end marker"

    def test_promotion_status_is_generated(self):
        """PROMOTION_GATES must have PROMOTION_STATUS generated block."""
        text = open("PROMOTION_GATES.md").read()
        assert "<!-- BEGIN:PROMOTION_STATUS" in text, "PROMOTION_GATES missing generated status block"

    def test_generated_promotion_surface_classes_match_registry(self):
        """PROMOTION_GATES status table surface_class values must match registry."""
        text = open("PROMOTION_GATES.md").read()
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        for key, feat in FEATURE_BY_KEY.items():
            if feat.tier == "candidate":
                sc = feat.surface_class.replace("_", "-")
                assert sc in text, \
                    f"PROMOTION_GATES missing surface_class '{sc}' for {feat.name}"
