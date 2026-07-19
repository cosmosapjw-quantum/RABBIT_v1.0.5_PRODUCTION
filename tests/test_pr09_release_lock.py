"""PR-09: Final Publication and Release Lock — TDD tests.

The package must be presented consistently as:
  "canonical Type I BBN core + candidate anisotropic extension perimeter"
across ALL public-facing surfaces.
"""
import pytest
import json
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

pytestmark = [pytest.mark.production, pytest.mark.release_smoke]

_IDENTITY_PHRASE = "canonical Type I BBN core"
_PERIMETER_PHRASE = "candidate"


class TestIdentityConsistency:
    """Top-line identity must be consistent across all surfaces."""

    def test_readme_has_canonical_identity(self):
        text = open("README.md").read()
        assert _IDENTITY_PHRASE.lower() in text.lower(), \
            f"README missing identity phrase: '{_IDENTITY_PHRASE}'"

    def test_readme_has_candidate_perimeter(self):
        text = open("README.md").read()
        assert _PERIMETER_PHRASE in text.lower(), \
            f"README missing perimeter qualifier: '{_PERIMETER_PHRASE}'"

    def test_status_mentions_canonical(self):
        text = open("STATUS.md").read()
        assert "canonical" in text.lower(), "STATUS missing 'canonical'"

    def test_supported_capabilities_mentions_canonical(self):
        text = open("SUPPORTED_CAPABILITIES.md").read()
        assert "canonical" in text.lower(), "SUPPORTED_CAPABILITIES missing 'canonical'"

    def test_no_full_production_package_claim(self):
        """No doc may say 'full production anisotropic BBN package'."""
        for f in ["README.md", "STATUS.md", "SUPPORTED_CAPABILITIES.md"]:
            text = open(f).read()
            forbidden_start = text.find("## Forbidden Claims")
            for i, line in enumerate(text.split('\n')):
                lower = line.lower()
                if 'full' in lower and 'production' in lower and 'bbn' in lower and 'package' in lower:
                    pos = sum(len(l)+1 for l in text.split('\n')[:i])
                    if forbidden_start != -1 and pos >= forbidden_start:
                        continue
                    pytest.fail(f"{f}: full production package claim: {line.strip()[:80]}")


class TestGeneratedDocCleanliness:
    """render --apply must be a no-op on a clean tree."""

    def test_render_apply_is_noop(self):
        """Running render --apply on current docs must not change them."""
        docs = {}
        for f in ["README.md", "STATUS.md", "SUPPORTED_CAPABILITIES.md",
                   "PROMOTION_GATES.md", "docs/BACKEND_CAPABILITY_MATRIX.md",
                   "docs/RENDER_PROVENANCE.json"]:
            docs[f] = open(f).read()
        r = subprocess.run(
            [sys.executable, "scripts/render_capability_tables.py", "--apply"],
            capture_output=True, text=True, timeout=30
        )
        assert r.returncode == 0
        for f, content in docs.items():
            current = open(f).read()
            assert current == content, f"{f} changed after render --apply (stale generated block)"

    def test_render_provenance_is_content_addressed(self):
        provenance = json.loads(Path("docs/RENDER_PROVENANCE.json").read_text())
        assert provenance["provenance_schema"] == "content-addressed-v2"
        assert len(provenance["render_fingerprint"]) == 64
        assert "render_timestamp" not in provenance


class TestPytestLaneHygiene:
    """Fast/release marker lanes must not collect opt-in expensive tests."""

    def test_expensive_tests_are_also_slow(self):
        """`-m not slow` must exclude every opt-in expensive test."""
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-m",
                "expensive and not slow",
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert r.returncode in (0, 5), r.stderr
        leaked = [line for line in r.stdout.splitlines() if "::" in line]
        assert not leaked, (
            "Opt-in expensive tests leaked into the '-m not slow' lane: "
            + ", ".join(leaked)
        )


class TestManifestProvenance:
    """RELEASE_MANIFEST must have complete provenance."""

    def test_manifest_has_render_scope_details(self):
        if not __import__('os').path.exists("RELEASE_MANIFEST.json"):
            pytest.skip("No manifest")
        m = json.load(open("RELEASE_MANIFEST.json"))
        scope = m.get("render_scope", [])
        assert len(scope) >= 4, f"render_scope too narrow: {len(scope)}"

    def test_manifest_has_pipeline_version(self):
        if not __import__('os').path.exists("RELEASE_MANIFEST.json"):
            pytest.skip("No manifest")
        m = json.load(open("RELEASE_MANIFEST.json"))
        assert "pipeline" in m, "Missing pipeline field"

    def test_manifest_test_counts_complete(self):
        if not __import__('os').path.exists("RELEASE_MANIFEST.json"):
            pytest.skip("No manifest")
        m = json.load(open("RELEASE_MANIFEST.json"))
        tc = m.get("test_counts", {})
        for field in ["total", "gold", "production_total",
                      "production_not_slow", "smoke"]:
            assert field in tc, f"test_counts missing '{field}'"
            assert tc[field] > 0, f"test_counts['{field}'] is 0"

    def test_package_release_fails_closed_on_tracked_dirtiness(self, monkeypatch):
        """Packaging must stop before render/tests when tracked state is dirty."""
        script = Path("scripts/package_release.py").resolve()
        spec = importlib.util.spec_from_file_location("rabbit_package_release", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout=" M README.md\n",
                stderr="",
            ),
        )

        with pytest.raises(SystemExit) as exc_info:
            module.require_clean_tracked_tree("test")
        assert exc_info.value.code == 1

    def test_runtime_release_excludes_historical_report_tree(self):
        script = Path("scripts/package_release.py").resolve()
        spec = importlib.util.spec_from_file_location("rabbit_package_release", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert not module.release_path_allowed("docs/RABBIT_report/main.pdf")
        assert not module.release_path_allowed(
            "docs/RABBIT_report/sections/18_results.tex"
        )
        assert module.release_path_allowed("docs/ROADMAP_INDEX.md")

    def test_release_version_matches_project_metadata(self):
        script = Path("scripts/package_release.py").resolve()
        spec = importlib.util.spec_from_file_location("rabbit_package_release", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.read_project_version() == "1.0.0"

    def test_release_archive_includes_tracked_native_crate(self):
        script = Path("scripts/package_release.py").resolve()
        spec = importlib.util.spec_from_file_location("rabbit_package_release", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        release_files = module.tracked_release_files()
        assert "LICENSE" in release_files
        assert "native/rabbit_cpu/Cargo.toml" in release_files
        assert "native/rabbit_cpu/src/lib.rs" in release_files
        assert not any(
            path.startswith("docs/RABBIT_report/") for path in release_files
        )

    def test_fast_production_timeout_is_explicit_sufficient_and_wired(
        self, monkeypatch
    ):
        """The measured fast lane must use one inspectable timeout >= 300 s."""
        script = Path("scripts/package_release.py").resolve()
        spec = importlib.util.spec_from_file_location("rabbit_package_release", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        timeout = module.FAST_PRODUCTION_TIMEOUT_SECONDS
        assert isinstance(timeout, (int, float))
        assert timeout >= 300

        run_calls = []

        def fake_run(cmd, timeout=120, check=False):
            run_calls.append((cmd, timeout, check))
            is_fast_lane = "production and not slow" in cmd
            return SimpleNamespace(
                returncode=1 if is_fast_lane else 0,
                stdout="forced stop before packaging\n" if is_fast_lane else "",
                stderr="",
            )

        monkeypatch.setattr(module, "run", fake_run)
        monkeypatch.setattr(module, "require_clean_tracked_tree", lambda _context: None)
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout="",
                stderr="",
            ),
        )
        monkeypatch.setattr(module.sys, "argv", ["package_release.py", "unused.zip"])

        with pytest.raises(SystemExit) as exc_info:
            module.main()
        assert exc_info.value.code == 1

        fast_lane_calls = [
            call for call in run_calls if "production and not slow" in call[0]
        ]
        assert len(fast_lane_calls) == 1
        assert fast_lane_calls[0][1] == timeout

    def test_subprocess_timeout_blocks_release_explicitly(
        self, monkeypatch, capsys
    ):
        """A timeout must become an explicit nonzero release block."""
        script = Path("scripts/package_release.py").resolve()
        spec = importlib.util.spec_from_file_location("rabbit_package_release", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        command = [sys.executable, "-m", "pytest", "tests/"]
        timeout = 300

        def raise_timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(command, timeout)

        monkeypatch.setattr(module.subprocess, "run", raise_timeout)

        with pytest.raises(SystemExit) as exc_info:
            module.run(command, timeout=timeout)
        assert exc_info.value.code not in (None, 0)

        output = capsys.readouterr().out
        assert "RELEASE BLOCKED" in output
        assert "timed out" in output.lower()
        assert str(timeout) in output


class TestForbiddenPermittedEnforcement:
    """Forbidden claims must actually block overclaiming."""

    def test_forbidden_covers_all_pr_features(self):
        """Forbidden Claims must have entries for each production-locked feature boundary."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        forbidden_start = text.find("## Forbidden Claims")
        forbidden_end = text.find("## Permitted", forbidden_start)
        forbidden = text[forbidden_start:forbidden_end].lower()
        required_topics = ["teff", "tilted", "class b", "class a",
                           "differentiable", "bayes factor", "bianchi-bbn"]
        for topic in required_topics:
            assert topic in forbidden, \
                f"Forbidden Claims missing boundary for: {topic}"

    def test_permitted_matches_production_features(self):
        """Permitted Claims must list exactly the production-locked features."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        permitted_start = text.find("## Permitted Claims")
        permitted = text[permitted_start:].lower()
        required = ["type i", "primat", "tilted", "class b", "class a",
                     "ad diagnostic", "pe framework"]
        for topic in required:
            assert topic in permitted, \
                f"Permitted Claims missing: {topic}"


class TestNoManualDrift:
    """Manual blocks must not duplicate generated registry content."""

    def test_supported_capabilities_no_manual_after_claims(self):
        """No manual content after the generated CLAIMS block."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        end = text.find("<!-- END:CLAIMS -->")
        if end == -1:
            pytest.skip("No CLAIMS block")
        after = text[end + len("<!-- END:CLAIMS -->"):].strip()
        assert len(after) < 50, \
            f"Manual content after CLAIMS block ({len(after)} chars): {after[:80]}"

    def test_readme_identity_is_generated(self):
        """README identity paragraph must be inside a generated block."""
        text = open("README.md").read()
        assert "<!-- BEGIN:IDENTITY" in text, "README missing IDENTITY generated block"
        assert "<!-- END:IDENTITY -->" in text, "README missing IDENTITY end marker"

    def test_all_claims_are_generated(self):
        """Forbidden/Permitted Claims must be inside a generated block."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        assert "<!-- BEGIN:CLAIMS" in text, "SUPPORTED_CAPABILITIES missing CLAIMS block"
        forbidden_pos = text.find("## Forbidden Claims")
        claims_begin = text.find("<!-- BEGIN:CLAIMS")
        if forbidden_pos != -1 and claims_begin != -1:
            assert forbidden_pos > claims_begin, \
                "Forbidden Claims must be inside generated CLAIMS block"
