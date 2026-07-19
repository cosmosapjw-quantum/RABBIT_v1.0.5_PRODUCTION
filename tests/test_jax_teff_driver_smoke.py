"""
Test: Teff correction legacy-kernel diagnostic.

Separates kernel parity (PASS) from the removed public runtime feature.

Status: Teff is deprecated legacy.
  - Kernel is parity-locked to 1e-12 vs NumPy.
  - Public forward-solver and driver runtime paths reject enable_teff=True.
"""
import pytest
import numpy as np

pytest.importorskip("jax", reason="JAX required")


# ═══════════════════════════════════════════════════════════════
# §1. Kernel parity (fast, no ODE)
# ═══════════════════════════════════════════════════════════════

class TestTeffKernelParity:
    """JAX Teff kernel matches NumPy reference to machine precision."""

    def test_mapping(self):
        from rabbit.jax.teff_correction_jax import pi_tilde_to_teff_quadrupole_jax
        from rabbit.weak.teff_correction import pi_tilde_to_teff_quadrupole
        for pi in [0.01, 0.05, 0.1, 0.2]:
            ref = pi_tilde_to_teff_quadrupole(pi)
            got = float(pi_tilde_to_teff_quadrupole_jax(pi))
            assert abs(got - ref) < 1e-14, f"pi={pi}: {got} vs {ref}"

    def test_spectral_hardening(self):
        import jax.numpy as jnp
        from rabbit.jax.teff_correction_jax import spectral_hardening_fd_jax
        from rabbit.weak.teff_correction import spectral_hardening_fd
        q = np.linspace(0.2, 8.0, 20)
        for sig2 in [1e-6, 1e-4, 1e-2]:
            ref = spectral_hardening_fd(q, sig2)
            got = np.asarray(spectral_hardening_fd_jax(jnp.asarray(q), sig2))
            assert np.allclose(got, ref, rtol=1e-12)

    def test_exact_closure(self):
        import jax.numpy as jnp
        from numpy.polynomial.laguerre import laggauss
        from rabbit.jax.teff_correction_jax import teff_corrected_monopole_exact_jax
        from rabbit.weak.teff_correction import teff_corrected_monopole_exact
        q, w = laggauss(12)
        f0 = 1.0 / (np.exp(q) + 1.0)
        for pi in [0.01, 0.08, 0.15]:
            ref = teff_corrected_monopole_exact(f0, q, pi)
            got = np.asarray(teff_corrected_monopole_exact_jax(f0, q, pi))
            assert np.allclose(got, ref, rtol=1e-12)


# ═══════════════════════════════════════════════════════════════
# §2. Driver smoke: Teff runs without crash (ODE)
# ═══════════════════════════════════════════════════════════════

"""The endpoint-driver smoke checks were retired with the JAX forward path."""


# ═══════════════════════════════════════════════════════════════
# §3. Channel 2 diagnostic: magnitude + sign instability
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# §4. Capability documentation: Teff deprecated status
# ═══════════════════════════════════════════════════════════════

class TestTeffCapabilityHonesty:

    def test_jax_advanced_teff_candidate_is_deprecated(self):
        from rabbit.config.backend_capabilities import JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE
        assert JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE.supports_teff is False
        assert JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE.tier == "substrate"

    def test_jax_advanced_kernel_validated(self):
        from rabbit.config.backend_capabilities import JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE
        assert JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE.teff_kernel_validated is True

    def test_jax_advanced_has_caveat(self):
        from rabbit.config.backend_capabilities import JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE
        reason = JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE.teff_blocking_reason
        assert "deprecated" in reason.lower()

    def test_scipy_teff_is_deprecated(self):
        from rabbit.config.backend_capabilities import SCIPY_TYPEI_REFERENCE
        assert SCIPY_TYPEI_REFERENCE.supports_teff is False
        assert SCIPY_TYPEI_REFERENCE.teff_kernel_validated is True

    def test_kernel_substrate_supports_teff(self):
        from rabbit.config.backend_capabilities import JAX_WEAK_CL3_KERNEL
        assert JAX_WEAK_CL3_KERNEL.supports_teff is True
        assert JAX_WEAK_CL3_KERNEL.teff_kernel_validated is True

    def test_candidate_summary(self):
        """Summary: kernel=validated, runtime=deprecated."""
        from rabbit.config.backend_capabilities import JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE as cap
        assert cap.teff_kernel_validated is True
        assert cap.supports_teff is False
        assert "deprecated" in cap.teff_blocking_reason.lower()

    def test_caveat_matches_runtime_policy(self):
        from rabbit.config.backend_capabilities import JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE as cap
        assert "deprecated" in cap.teff_blocking_reason.lower()


# ═══════════════════════════════════════════════════════════════
# §5. Catalog-wide Teff consistency audit (no ODE)
# ═══════════════════════════════════════════════════════════════

class TestTeffCatalogConsistency:
    """Verify Teff fields are self-consistent across all 10 capabilities."""

    def test_supports_teff_implies_kernel_validated(self):
        """If supports_teff=True, kernel must also be validated."""
        from rabbit.config.backend_capabilities import CAPABILITY_BY_KEY
        for key, cap in CAPABILITY_BY_KEY.items():
            if cap.supports_teff:
                assert cap.teff_kernel_validated, (
                    f"{key}: supports_teff=True but teff_kernel_validated=False"
                )

    def test_caveat_reason_requires_kernel(self):
        """Teff caveat/blocking reason requires kernel=True."""
        from rabbit.config.backend_capabilities import CAPABILITY_BY_KEY
        for key, cap in CAPABILITY_BY_KEY.items():
            if cap.teff_blocking_reason:
                assert cap.teff_kernel_validated is True, (
                    f"{key}: has reason but kernel not validated"
                )

    def test_exactly_one_with_caveat(self):
        """Only the dedicated legacy Teff surface has a Teff caveat reason."""
        from rabbit.config.backend_capabilities import CAPABILITY_BY_KEY
        with_reason = [k for k, c in CAPABILITY_BY_KEY.items() if c.teff_blocking_reason]
        assert with_reason == ["jax_typeI_liveweak_cl3_tier2_teff_candidate"]

    def test_teff_capable_count(self):
        """Only the low-level weak kernel claims active Teff support."""
        from rabbit.config.backend_capabilities import CAPABILITY_BY_KEY
        capable = [k for k, c in CAPABILITY_BY_KEY.items() if c.supports_teff]
        assert len(capable) == 1
        assert "jax_weak_cl3_kernel" in capable
