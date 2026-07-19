"""tests/test_dhs_correction.py — v3.2 Phase χ-3 closed-form gates.

Replaces the v2.x stub regression (which locked the DH-S factor at
identity = 1.0) with the v3.2 closed-form running-coupling gates.

Plan §χ-3. The structural form active in v3.2 is

    δ_DHS(T_ν) = 1 + (α_em / π) · c_DHS · ln(1 + (T_ν / m_e)²)

with `c_DHS = 1.0` (placeholder pending direct DH-S 1997 Appendix A
transcription). Tests lock structural properties (asymptotics,
monotonicity, sign) while leaving the calibration coefficient as a
documented research-grade follow-on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# §1. v3.2 contract: closed form is now active
# ═══════════════════════════════════════════════════════════════════════

def test_v32_dhs_closed_form_active():
    """v3.2 Phase χ-3 contract: HAS_DHS_TABLE = True (closed form active)."""
    from rabbit.jax.dhs_correction import HAS_DHS_TABLE
    assert HAS_DHS_TABLE is True, (
        "v3.2 Phase χ-3 should activate HAS_DHS_TABLE; if reverting, "
        "update the module docstring + this test together."
    )


def test_v32_dhs_factor_differs_from_unity_at_BBN_temperatures():
    """At BBN-relevant T_ν, the factor differs measurably from 1.

    v3.2 Phase χ-3 fails the v2.x identity contract by construction.
    Lock the new behaviour: factor > 1 (running enhancement) at
    T_ν > m_e/10 with magnitude small but non-trivial.
    """
    from rabbit.jax.dhs_correction import dhs_correction_factor
    for t in (0.1, 0.5, 1.0, 2.0, 5.0):
        f = float(dhs_correction_factor(jnp.asarray(t)))
        assert f > 1.0, f"DH-S factor at T_ν = {t} should be > 1; got {f}"
        # Sub-percent at BBN scales (α/π × ln(1+(T/m_e)²) is bounded)
        assert f < 1.05, (
            f"DH-S factor at T_ν = {t} = {f}; expected < 1.05 in BBN range"
        )


# ═══════════════════════════════════════════════════════════════════════
# §2. Asymptotics
# ═══════════════════════════════════════════════════════════════════════

class TestAsymptotics:

    def test_deep_IR_limit_recovers_unity(self):
        """T_ν → 0: δ_DHS → 1 (FLRW deep-IR limit; no running)."""
        from rabbit.jax.dhs_correction import dhs_correction_factor
        # T_ν << m_e → ln(1 + (T/m_e)²) → 0 → δ → 1
        f_tiny = float(dhs_correction_factor(jnp.asarray(1e-6)))
        assert abs(f_tiny - 1.0) < 1e-12, (
            f"deep-IR DH-S factor: {f_tiny} != 1 (expected within 1e-12)"
        )

    def test_at_electron_mass_threshold(self):
        """T_ν = m_e ≈ 0.51 MeV: ln(1+1) = ln 2 ≈ 0.693."""
        from rabbit.jax.dhs_correction import (
            dhs_correction_factor, _ALPHA_EM, _M_E_MEV, _C_DHS,
        )
        f = float(dhs_correction_factor(jnp.asarray(_M_E_MEV)))
        expected = 1.0 + (_ALPHA_EM / np.pi) * _C_DHS * np.log(2.0)
        rel = abs(f - expected) / expected
        assert rel < 1e-12, (
            f"factor at T = m_e: got {f}, expected {expected}; rel={rel:.3e}"
        )

    def test_high_T_logarithmic_growth(self):
        """T_ν ≫ m_e: δ ~ 1 + (α/π) · 2 ln(T/m_e) (logarithmic growth)."""
        from rabbit.jax.dhs_correction import (
            dhs_correction_factor, _ALPHA_EM, _M_E_MEV, _C_DHS,
        )
        T_high = 50.0  # MeV; deep UV vs m_e
        f = float(dhs_correction_factor(jnp.asarray(T_high)))
        # ln(1 + x²) ~ 2 ln x for x >> 1
        expected = 1.0 + (_ALPHA_EM / np.pi) * _C_DHS * 2.0 * np.log(T_high / _M_E_MEV)
        rel = abs(f - expected) / expected
        # Asymptotic; relax to 1e-3
        assert rel < 1e-3


# ═══════════════════════════════════════════════════════════════════════
# §3. Monotonicity + sign
# ═══════════════════════════════════════════════════════════════════════

class TestMonotonicityAndSign:

    def test_factor_monotone_increasing_in_T(self):
        """δ_DHS is monotonically non-decreasing in T_ν."""
        from rabbit.jax.dhs_correction import dhs_correction_factor
        T = jnp.geomspace(0.01, 100.0, 50)
        f = dhs_correction_factor(T)
        diffs = jnp.diff(f)
        assert jnp.all(diffs >= -1e-15)

    def test_factor_always_above_unity(self):
        """δ_DHS > 1 across the BBN range (running enhancement)."""
        from rabbit.jax.dhs_correction import dhs_correction_factor
        T = jnp.geomspace(0.05, 50.0, 30)
        f = dhs_correction_factor(T)
        assert jnp.all(f > 1.0)


# ═══════════════════════════════════════════════════════════════════════
# §4. Differentiability
# ═══════════════════════════════════════════════════════════════════════

def test_grad_finite_and_positive():
    """jax.grad of δ_DHS w.r.t. T_ν is finite and positive (monotone)."""
    from rabbit.jax.dhs_correction import dhs_correction_factor
    g = float(jax.grad(lambda t: dhs_correction_factor(t).sum())(jnp.asarray(1.0)))
    assert np.isfinite(g)
    assert g > 0.0


def test_grad_matches_richardson_fd():
    """jax.grad ≈ central FD at T_ν = 1 MeV."""
    from rabbit.jax.dhs_correction import dhs_correction_factor
    T0 = 1.0
    eps = 1e-4
    g_ad = float(jax.grad(lambda t: dhs_correction_factor(t).sum())(jnp.asarray(T0)))
    g_fd = float(
        (dhs_correction_factor(jnp.asarray(T0 + eps))
         - dhs_correction_factor(jnp.asarray(T0 - eps)))
        / (2.0 * eps)
    )
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)
    assert rel < 1e-4


# ═══════════════════════════════════════════════════════════════════════
# §5. Corrected-rate wrapper
# ═══════════════════════════════════════════════════════════════════════

def test_corrected_rate_differs_from_leading_order():
    """The corrected rate differs from the leading-order ν-ν rate."""
    from rabbit.jax.collision_rates_jax import total_rate_nu_nu_diagonal_jax
    from rabbit.jax.dhs_correction import total_rate_nu_nu_diagonal_dhs_corrected
    for t in (0.5, 1.0, 2.0):
        ref = float(total_rate_nu_nu_diagonal_jax(jnp.asarray(t)))
        corrected = float(total_rate_nu_nu_diagonal_dhs_corrected(jnp.asarray(t)))
        # v3.2: correction must shift the rate by a sub-percent amount
        rel_shift = abs(corrected - ref) / ref
        assert 1e-5 < rel_shift < 1e-2, (
            f"T_ν = {t}: rate shift = {rel_shift:.3e}; expected sub-percent"
        )
        # Sign: enhancement (running > 1)
        assert corrected > ref, (
            f"corrected rate must exceed leading-order; got {corrected} ≤ {ref}"
        )
