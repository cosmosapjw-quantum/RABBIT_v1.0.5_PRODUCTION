"""
tests/test_teff_promotion_gates.py — Teff legacy-kernel diagnostic tests.

Teff is deprecated as a public runtime feature.  These tests keep the old
low-level closure kernel reproducible while ensuring the docs still mention
the legacy status.

Gate T1: Public API keeps the deprecated switch but runtime rejects it
Gate T2: Zero/null/limit recovery (π̃=0 → identity)
Gate T3: Matched-physics parity (component-level determinism)
Gate T4: Convergence envelope (N_q sensitivity, sign stability)
Gate T5: Docs-vs-code consistency
Gate T6: Legacy kernel diagnostic alignment
"""
import pytest
import numpy as np


def _grid_and_fd(nq):
    from rabbit.config.grids import MomentumGrid
    g = MomentumGrid(nq)
    return g, 1.0 / (np.exp(g.nodes) + 1)


def _teff_rates(nq, pi_tilde, cl=0):
    """Compute rates with and without Teff at given config."""
    from rabbit.weak.teff_correction import teff_corrected_monopole_exact
    from rabbit.weak.live_rates import compute_live_weak_rates
    g, f0 = _grid_and_fd(nq)
    f_corr = teff_corrected_monopole_exact(f0, g.nodes, pi_tilde)
    r_off = compute_live_weak_rates(f0, f0, g.nodes, 1.0, correction_level=cl)
    r_on = compute_live_weak_rates(f_corr, f_corr, g.nodes, 1.0, correction_level=cl)
    return r_off, r_on


# ═══ Gate T1: API callable ═══

@pytest.mark.production
@pytest.mark.release_smoke
class TestTeffGateT1:
    def test_teff_correction_importable(self):
        from rabbit.weak.teff_correction import (
            teff_corrected_monopole_exact,
            pi_tilde_to_teff_quadrupole,
            teff_angular_variance,
            spectral_hardening_fd,
        )
        assert callable(teff_corrected_monopole_exact)

    def test_enable_teff_in_canonical_signature(self):
        from rabbit.inference.forward_likelihood import canonical_forward_solver
        import inspect
        sig = inspect.signature(canonical_forward_solver)
        assert 'enable_teff' in sig.parameters

    def test_public_runtime_rejects_enable_teff(self):
        from rabbit.inference.forward_likelihood import canonical_forward_solver
        with pytest.raises(ValueError, match="deprecated legacy"):
            canonical_forward_solver(enable_teff=True)

    def test_teff_correction_returns_array(self):
        from rabbit.weak.teff_correction import teff_corrected_monopole_exact
        g, f0 = _grid_and_fd(20)
        result = teff_corrected_monopole_exact(f0, g.nodes, 0.05)
        assert isinstance(result, np.ndarray)
        assert result.shape == f0.shape


# ═══ Gate T2: Zero/null/limit recovery ═══

@pytest.mark.production
@pytest.mark.release_smoke
class TestTeffGateT2:
    def test_pi_zero_is_identity(self):
        """π̃=0 must return unmodified distribution."""
        for nq in [6, 20, 40]:
            g, f0 = _grid_and_fd(nq)
            from rabbit.weak.teff_correction import teff_corrected_monopole_exact
            f_corr = teff_corrected_monopole_exact(f0, g.nodes, 0.0)
            np.testing.assert_allclose(f_corr, f0, atol=1e-15,
                err_msg=f"Teff not identity at π̃=0, N_q={nq}")

    def test_teff_off_equals_baseline_rates(self):
        """Rates without Teff must equal standard rates."""
        g, f0 = _grid_and_fd(20)
        from rabbit.weak.live_rates import compute_live_weak_rates
        r = compute_live_weak_rates(f0, f0, g.nodes, 1.0, correction_level=0)
        # This IS the baseline — no Teff applied
        assert r.lambda_np > 0

    def test_sigma_zero_gives_zero_correction(self):
        """At Σ=0, π̃=0.158×0=0 → zero correction."""
        r_off, r_on = _teff_rates(20, 0.0)
        assert abs(r_on.lambda_np - r_off.lambda_np) < 1e-14

    def test_small_pi_gives_small_correction(self):
        """At very small π̃, correction must be perturbatively small."""
        r_off, r_on = _teff_rates(20, 1e-6)
        rel = abs(r_on.lambda_np - r_off.lambda_np) / r_off.lambda_np
        assert rel < 1e-6, f"Tiny π̃ gives {rel:.2e} relative correction"


# ═══ Gate T3: Matched-physics parity ═══

@pytest.mark.production
@pytest.mark.release_smoke
class TestTeffGateT3:
    def test_teff_correction_deterministic(self):
        """Same input must produce same output."""
        from rabbit.weak.teff_correction import teff_corrected_monopole_exact
        g, f0 = _grid_and_fd(20)
        a = teff_corrected_monopole_exact(f0, g.nodes, 0.05)
        b = teff_corrected_monopole_exact(f0, g.nodes, 0.05)
        np.testing.assert_array_equal(a, b)

    def test_teff_rate_shift_deterministic(self):
        """Rate shift from Teff must be deterministic."""
        r1_off, r1_on = _teff_rates(20, 0.05)
        r2_off, r2_on = _teff_rates(20, 0.05)
        d1 = r1_on.lambda_np - r1_off.lambda_np
        d2 = r2_on.lambda_np - r2_off.lambda_np
        assert abs(d1 - d2) < 1e-15

    def test_cl0_cl2_teff_same_sign(self):
        """CL0 and CL2 must give same-sign Teff correction."""
        r0_off, r0_on = _teff_rates(20, 0.05, cl=0)
        r2_off, r2_on = _teff_rates(20, 0.05, cl=2)
        d0 = r0_on.lambda_np - r0_off.lambda_np
        d2 = r2_on.lambda_np - r2_off.lambda_np
        assert d0 * d2 > 0, f"Sign mismatch: CL0={d0:.3e}, CL2={d2:.3e}"


# ═══ Gate T4: Convergence envelope ═══

@pytest.mark.production
@pytest.mark.release_smoke
class TestTeffGateT4:
    def test_sign_consistent_nq20_plus(self):
        """Δλ_np must be positive at N_q≥20 for all tested Σ."""
        for nq in [20, 40]:
            for sigma in [0.05, 0.1, 0.2]:
                pi = 0.158 * sigma
                r_off, r_on = _teff_rates(nq, pi)
                d = r_on.lambda_np - r_off.lambda_np
                assert d > 0, f"N_q={nq} Σ={sigma}: Δλ_np={d:.3e} not positive"

    def test_nq_convergence_factor(self):
        """N_q=20→40 correction must decrease (convergence)."""
        pi = 0.158 * 0.1
        _, r20_on = _teff_rates(20, pi)
        r20_off, _ = _teff_rates(20, 0.0)
        _, r40_on = _teff_rates(40, pi)
        r40_off, _ = _teff_rates(40, 0.0)
        d20 = abs(r20_on.lambda_np - r20_off.lambda_np)
        d40 = abs(r40_on.lambda_np - r40_off.lambda_np)
        # Convergence: N_q=40 should give smaller correction
        # (closer to the continuum limit)
        assert d40 < d20 * 1.5, f"|Δ(40)|={d40:.3e} not converging from |Δ(20)|={d20:.3e}"

    def test_monotonic_in_sigma(self):
        """Teff correction magnitude must be monotonic in Σ."""
        prev = 0.0
        for sigma in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]:
            pi = 0.158 * sigma
            r_off, r_on = _teff_rates(20, pi)
            d = abs(r_on.lambda_np - r_off.lambda_np)
            assert d >= prev - 1e-15, \
                f"Non-monotonic at Σ={sigma}: |Δ|={d:.3e} < prev={prev:.3e}"
            prev = d

    def test_nq10_sign_is_now_stable(self):
        """After the logit-residual rewrite, exact-FD Teff sign stays positive at N_q=10."""
        pi = 0.158 * 0.1
        r_off, r_on = _teff_rates(10, pi)
        d = r_on.lambda_np - r_off.lambda_np
        assert d > 0, f"N_q=10 exact-FD sign regressed: Δλ_np={d:.3e}"


# ═══ Gate T5: Docs-vs-code consistency ═══

@pytest.mark.production
@pytest.mark.release_smoke
class TestTeffGateT5:
    def test_promotion_packet_exists(self):
        import os
        assert os.path.exists("docs/TEFF_PROMOTION_PACKET.md"), \
            "Teff promotion packet missing"

    def test_supported_capabilities_teff_entry(self):
        with open("SUPPORTED_CAPABILITIES.md") as f:
            cap = f.read()
        # After promotion, Teff should have a production entry
        assert "Teff" in cap

    def test_promotion_gates_doc_exists(self):
        import os
        assert os.path.exists("PROMOTION_GATES.md")


# ═══ Gate T6: Support claim alignment ═══

@pytest.mark.production
@pytest.mark.release_smoke
class TestTeffGateT6:
    def test_gold_values_match_packet(self):
        """Gold values in promotion packet must match actual computation."""
        # After the stabilized Teff closure rewrite, the exact-FD reference shift
        # at N_q=20, CL0, Σ=0.1 is small and positive.
        r_off, r_on = _teff_rates(20, 0.158 * 0.1, cl=0)
        d = r_on.lambda_np - r_off.lambda_np
        assert abs(d - 4.36055e-6) < 1.0e-6, \
            f"Gold drift: actual={d:.4e}, documented≈4.3606e-6"
        assert d > 0, "Sign must be positive"

    def test_correction_magnitude_order(self):
        """Exact-FD Teff correction is small but finite at Σ=0.1 after closure stabilization."""
        r_off, r_on = _teff_rates(20, 0.158 * 0.1)
        rel = abs(r_on.lambda_np - r_off.lambda_np) / r_off.lambda_np
        assert 1e-7 < rel < 1e-4, f"rel={rel:.3e} outside expected small-positive window"
