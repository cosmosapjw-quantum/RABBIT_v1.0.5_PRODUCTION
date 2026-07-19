"""Deprecated Teff capability audit.

The former Teff candidate-promotion gate is now a deprecation lock: low-level
kernels may remain for reproducibility diagnostics, but no public runtime path
may advertise active Teff support.
"""
from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="JAX required")


class TestTeffMetadataContract:
    def test_capability_is_diagnostic_substrate(self):
        from rabbit.config.backend_capabilities import (
            JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE,
        )

        cap = JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE
        assert cap.tier == "substrate"
        assert cap.effective_surface_class == "diagnostic"
        assert cap.supports_teff is False
        assert cap.teff_kernel_validated is True

    def test_capability_caveat_says_deprecated(self):
        from rabbit.config.backend_capabilities import (
            JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE,
        )

        reason = JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE.teff_blocking_reason
        assert "deprecated" in reason.lower()

    def test_dispatch_contract_has_no_public_teff_surface(self):
        from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND

        teff_dispatch = {k for k, v in CAPABILITY_BY_BACKEND.items() if v.supports_teff}
        assert teff_dispatch == set()


class TestTeffRuntimeFence:
    def test_canonical_forward_solver_rejects_scipy_teff(self):
        from rabbit.inference.forward_likelihood import canonical_forward_solver

        with pytest.raises(ValueError, match="deprecated legacy"):
            canonical_forward_solver(
                Sigma_H=0.1,
                backend="scipy",
                correction_level=2,
                N_q=6,
                enable_teff=True,
            )

    def test_direct_scipy_driver_rejects_teff(self):
        from rabbit.drivers.full_coupled_typeI import FullCoupledConfig

        with pytest.raises(ValueError, match="supersedes the legacy Teff closure"):
            FullCoupledConfig(
                Sigma_H_plus=0.1,
                eta=6.104e-10,
                tau_n=878.4,
                correction_level=2,
                n_reactions=12,
                N_q=6,
                tier=2,
                enable_teff=True,
            )


class TestTeffLanguageAudit:
    def test_feature_registry_marks_deprecated(self):
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY

        feat = FEATURE_BY_KEY["teff_spectral"]
        text = f"{feat.evidence_summary} {feat.short_summary} {feat.notes}".lower()
        assert feat.tier == "substrate"
        assert feat.validation_mode == "diagnostic"
        assert "deprecated" in text
        assert "no public" in text

    def test_teff_capable_is_kernel_only(self):
        from rabbit.config.backend_capabilities import CAPABILITY_BY_KEY

        capable = sorted(k for k, c in CAPABILITY_BY_KEY.items() if c.supports_teff)
        assert capable == ["jax_weak_cl3_kernel"]
