"""Teff deprecated-legacy lock.

Teff is no longer a promotion target.  The low-level closure kernels remain
for reproducibility diagnostics, but public forward-solver paths must reject
enable_teff=True.

These tests verify:
  1. public runtime rejects enable_teff=True
  2. enable_teff=False remains baseline-compatible
  3. docs and registries describe Teff as deprecated diagnostic substrate
"""
import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("RABBIT_JAX_CACHE_DIR", "/tmp/rabbit_jax_cache")

pytestmark = [pytest.mark.production, pytest.mark.release_smoke]


class TestTeffRegimeGuards:
    """Public forward-solver paths reject the deprecated Teff switch."""

    def test_auto_backend_ignores_teff_without_failure(self):
        """backend='auto' rejects enable_teff."""
        from rabbit.inference.forward_likelihood import canonical_forward_solver
        import pytest
        with pytest.raises(ValueError, match="enable_teff=True is deprecated legacy"):
            canonical_forward_solver(
                Sigma_H=0.1, N_q=6, backend='auto',
                enable_teff=True
            )

    def test_retired_jax_advanced_precedes_teff_policy(self):
        from rabbit.inference.forward_likelihood import canonical_forward_solver
        with pytest.raises(ValueError, match="retired from the public forward surface"):
            canonical_forward_solver(
                Sigma_H=0.1, N_q=20, backend='jax_advanced',
                enable_teff=True
            )


class TestTeffBaselineRecovery:
    """Teff off must produce identical results to baseline."""

    @pytest.mark.slow
    def test_teff_off_equals_no_teff(self):
        """enable_teff=False must match enable_teff omitted."""
        from rabbit.inference.forward_likelihood import canonical_forward_solver
        r1 = canonical_forward_solver(Sigma_H=0.1, N_q=6, backend='auto')
        r2 = canonical_forward_solver(Sigma_H=0.1, N_q=6, backend='auto',
                                       enable_teff=False)
        assert abs(r1.Yp - r2.Yp) < 1e-12, \
            f"Teff off should match baseline: {r1.Yp} vs {r2.Yp}"


class TestTeffIsotropicRecovery:
    """Sigma->0 Teff null is superseded by public runtime rejection."""

    def test_retired_jax_advanced_is_not_a_teff_null_surface(self):
        from rabbit.inference.forward_likelihood import canonical_forward_solver
        with pytest.raises(ValueError, match="retired from the public forward surface"):
            canonical_forward_solver(Sigma_H=0.0, N_q=20, backend='jax_advanced',
                                      enable_teff=True)


class TestTeffDocSync:
    """All docs must use the same narrow regime wording."""

    def test_no_broad_teff_production_claim(self):
        """No doc may say Teff is production-ready or a promotion target outside Forbidden Claims."""
        for f in ["README.md", "STATUS.md", "SUPPORTED_CAPABILITIES.md",
                   "PROMOTION_GATES.md"]:
            text = open(f).read()
            # Find Forbidden Claims section and exclude it
            forbidden_start = text.find("## Forbidden Claims")
            for i, line in enumerate(text.split('\n')):
                low = line.lower()
                if 'teff' in low and (
                    'production-ready' in low or
                    'candidate-strong' in low or
                    'promotion target' in low
                ):
                    # Calculate char position
                    char_pos = sum(len(l)+1 for l in text.split('\n')[:i])
                    # Skip if inside Forbidden Claims section
                    if forbidden_start != -1 and char_pos >= forbidden_start:
                        continue
                    pytest.fail(f"{f}: broad 'Teff production-ready' claim: {line.strip()}")

    def test_teff_registry_has_regime_scope(self):
        """Feature registry must record deprecation, not a runtime regime."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        t = FEATURE_BY_KEY['teff_spectral']
        combined = f"{t.evidence_summary} {t.short_summary} {t.notes}"
        assert "Deprecated" in combined or "deprecated" in combined
        assert "no public" in combined

    def test_teff_surface_class_reflects_promotion(self):
        """Teff is diagnostic substrate, not candidate promotion surface."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        t = FEATURE_BY_KEY['teff_spectral']
        assert t.tier == "substrate"
        assert t.validation_mode == "diagnostic"
        assert t.surface_class == "diagnostic"
