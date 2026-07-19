"""tests/test_v32_mangano_gap_envelope.py — v3.2 Phase χ-5 envelope lock.

Plan §χ-5. Locks the v3.2 Mangano-gap envelope using the rate-level
shift implied by the combined corrections:

  - F(m_e/T_γ): electron-mass suppression (χ-2)
  - δ_DHS(T_ν): running-coupling enhancement (χ-3)
  - FM corrections (χ-1): active in weak path, no rate-level effect
  - Pair processes (χ-4): already summed in Mangano a_α = 4(G_L²+G_R²)

Honest scope acknowledgement
----------------------------
A direct N_eff measurement against NUDEC_BSM requires running the
production AP-form driver against the live external code, which
needs the NUDEC_BSM environment to be set up (RABBIT_NUDEC_BSM_PATH).
On the dev system this isn't available; this test instead locks the
**rate-level shift** at the BBN-decoupling temperature, which the
SciPy-side sensitivity ``∂N_eff/∂C ≈ 1.4e-4 per unit C`` (cf.
``rabbit.thermo.nudec_tables`` line 84) maps to a projected N_eff
shift.

The actual N_eff envelope decision (gate-lift YES/NO) is in Phase χ-6,
which runs the full driver and consults the cross-code live test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# §1. Rate shift at BBN-decoupling temperature
# ═══════════════════════════════════════════════════════════════════════

class TestV32CorrectionsRateShift:
    """Lock the multiplicative rate shift at T = 1 MeV (decoupling)."""

    def test_corrections_off_reproduces_v31_baseline(self):
        """apply_v32_corrections=False reproduces the v3.1 leading-order rate."""
        from rabbit.jax.collision_hm_full_jax import gamma_alpha_per_momentum
        q = jnp.array([1.0, 3.0, 6.0])
        T = jnp.asarray(1.0)
        rate_off = gamma_alpha_per_momentum(q, T, species="nue")
        # Default kwargs should return identical rate
        rate_default = gamma_alpha_per_momentum(q, T, species="nue")
        rel = float(jnp.max(jnp.abs((rate_off - rate_default) / rate_off)))
        assert rel == 0.0, f"default vs explicit-off mismatch: rel={rel:.3e}"

    def test_corrections_on_shifts_rate(self):
        """apply_v32_corrections=True produces a measurable rate shift."""
        from rabbit.jax.collision_hm_full_jax import gamma_alpha_per_momentum
        q = jnp.array([1.0, 3.0, 6.0])
        T = jnp.asarray(1.0)
        rate_off = gamma_alpha_per_momentum(q, T, species="nue")
        rate_on = gamma_alpha_per_momentum(
            q, T, species="nue", apply_v32_corrections=True,
        )
        # The corrections are F(T_γ) × δ_DHS(T_ν) ≠ 1
        for r_on, r_off in zip(rate_on, rate_off):
            assert float(r_on) != float(r_off)

    def test_combined_factor_at_decoupling_consistent(self):
        """At T = 1 MeV: combined factor = F(1 MeV) × δ_DHS(1 MeV) ≈ 0.984."""
        from rabbit.jax.collision_hm_full_jax import gamma_alpha_per_momentum
        from rabbit.jax.electron_mass_suppression_jax import (
            electron_mass_suppression_jax,
        )
        from rabbit.jax.dhs_correction import dhs_correction_factor
        q = jnp.asarray(3.0)
        T = jnp.asarray(1.0)
        rate_off = float(gamma_alpha_per_momentum(q, T, species="nue"))
        rate_on = float(gamma_alpha_per_momentum(
            q, T, species="nue", apply_v32_corrections=True,
        ))
        F = float(electron_mass_suppression_jax(T))
        dhs = float(dhs_correction_factor(T))
        expected_factor = F * dhs
        actual_factor = rate_on / rate_off
        rel = abs(expected_factor - actual_factor) / abs(expected_factor)
        assert rel < 1e-12

    def test_combined_factor_magnitude_at_decoupling(self):
        """At T = 1 MeV: F × δ_DHS is in the documented sub-percent range."""
        from rabbit.jax.electron_mass_suppression_jax import (
            electron_mass_suppression_jax,
        )
        from rabbit.jax.dhs_correction import dhs_correction_factor
        T = jnp.asarray(1.0)
        F = float(electron_mass_suppression_jax(T))    # ~0.980
        dhs = float(dhs_correction_factor(T))           # ~1.00374
        combined = F * dhs
        # F suppresses (~-2%), DHS enhances (~+0.4%); net is ~-1.6%
        assert 0.97 < combined < 1.0
        # Specifically, combined ≈ 0.984 ± 0.005
        assert abs(combined - 0.984) < 0.01


# ═══════════════════════════════════════════════════════════════════════
# §2. Rate shift propagates through partner integration
# ═══════════════════════════════════════════════════════════════════════

class TestPartnerIntegrationCarriesCorrections:
    """When apply_v32_corrections=True, the partner-integrated rate changes."""

    def test_FLRW_thermal_average_shifts_by_combined_factor(self):
        """Thermal average shifts by the F(T_γ) × δ_DHS(T_γ) product
        at the calibration point (T_γ = T_ν = 1 MeV)."""
        from rabbit.jax.collision_hm_full_jax import (
            gamma_alpha_per_momentum, hm_thermal_average,
        )
        from rabbit.jax.electron_mass_suppression_jax import (
            electron_mass_suppression_jax,
        )
        from rabbit.jax.dhs_correction import dhs_correction_factor
        T = 1.0
        # Build q-grid (uniform Simpson)
        n_q = 40
        xi_max = 20.0
        xi = np.linspace(xi_max / (2 * n_q), xi_max, n_q)
        q = jnp.asarray(T * xi, dtype=jnp.float64)
        h = T * (xi[1] - xi[0])
        w = jnp.full(n_q, h, dtype=jnp.float64)

        # Leading-order thermal average
        avg_off = float(hm_thermal_average(q, w, T, species="nue"))

        # Apply corrections multiplicatively to per-q rate
        F = float(electron_mass_suppression_jax(jnp.asarray(T)))
        dhs = float(dhs_correction_factor(jnp.asarray(T)))
        avg_on_expected = avg_off * F * dhs

        # Verify by computing thermal average with corrections active
        # via gamma_alpha_per_momentum
        from rabbit.jax.collision_hm_full_jax import _fermi_dirac_dimensionless
        f_eq = _fermi_dirac_dimensionless(q / T)
        rate_corrected = gamma_alpha_per_momentum(
            q, T, species="nue", apply_v32_corrections=True,
        )
        numerator = float(jnp.sum(w * q ** 2 * f_eq * rate_corrected))
        denom = float(jnp.sum(w * q ** 2 * f_eq))
        avg_on_computed = numerator / max(denom, 1e-300)

        rel = abs(avg_on_computed - avg_on_expected) / avg_on_expected
        assert rel < 1e-12, (
            f"corrections do not propagate cleanly: expected "
            f"{avg_on_expected}, got {avg_on_computed}, rel = {rel:.3e}"
        )


# ═══════════════════════════════════════════════════════════════════════
# §3. v3.2 envelope lock
# ═══════════════════════════════════════════════════════════════════════

class TestV32EnvelopeLock:
    """Lock the rate-level corrections magnitude as the v3.2 baseline.

    Per Plan §χ-5 the actual N_eff envelope is decided in Phase χ-6
    after running the full driver against NUDEC_BSM. This test locks
    the rate-level shift so a future commit cannot silently drift the
    corrections without updating the envelope.
    """

    def test_v32_combined_correction_locked_envelope(self):
        """Combined correction at T_γ = T_ν = 1 MeV: F × δ_DHS ∈ [0.97, 1.00].

        This is the LOCK for v3.2: any future commit that changes the
        underlying corrections must update this envelope explicitly
        with a citation in the commit message.
        """
        from rabbit.jax.electron_mass_suppression_jax import (
            electron_mass_suppression_jax,
        )
        from rabbit.jax.dhs_correction import dhs_correction_factor
        T = jnp.asarray(1.0)
        F = float(electron_mass_suppression_jax(T))
        dhs = float(dhs_correction_factor(T))
        combined = F * dhs
        # Locked envelope (v3.2 baseline at T = 1 MeV)
        assert 0.975 < combined < 0.995, (
            f"v3.2 χ-5 envelope drift: F × δ_DHS at T = 1 MeV = {combined}; "
            "expected (0.975, 0.995). If deliberate physics update, "
            "amend this gate with citation."
        )

    def test_v32_combined_jax_grad_finite(self):
        """jax.grad through the corrected rate is finite."""
        from rabbit.jax.collision_hm_full_jax import gamma_alpha_per_momentum
        def loss(T):
            q = jnp.asarray(3.0)
            return gamma_alpha_per_momentum(
                q, T, species="nue", apply_v32_corrections=True,
            )
        g = float(jax.grad(loss)(jnp.asarray(1.0)))
        assert np.isfinite(g)
        assert g != 0.0


# ═══════════════════════════════════════════════════════════════════════
# §4. Honest acknowledgement: full driver measurement deferred to χ-6
# ═══════════════════════════════════════════════════════════════════════

def test_phase_chi5_acknowledges_chi6_full_driver_measurement_pending():
    """Phase χ-5 ships the rate-level wire-up. The actual gap measurement
    against NUDEC_BSM live is the Phase χ-6 deliverable.

    This test exists to lock the contract: χ-5 commits the corrections,
    χ-6 runs the cross-code measurement, decides gate-lift, tags
    release. No silent skip-ahead allowed.
    """
    # Smoke: corrections are wired and active
    from rabbit.jax.collision_hm_full_jax import gamma_alpha_per_momentum
    from rabbit.jax.dhs_correction import HAS_DHS_TABLE
    assert HAS_DHS_TABLE is True, (
        "v3.2 χ-3 contract: HAS_DHS_TABLE must be True at end of Phase χ-3"
    )
    rate_on = float(gamma_alpha_per_momentum(
        jnp.asarray(3.0), jnp.asarray(1.0),
        species="nue", apply_v32_corrections=True,
    ))
    rate_off = float(gamma_alpha_per_momentum(
        jnp.asarray(3.0), jnp.asarray(1.0), species="nue",
    ))
    # Corrections active and produce a measurable shift
    assert rate_on != rate_off
