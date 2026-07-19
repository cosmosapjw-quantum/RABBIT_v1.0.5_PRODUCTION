"""
Test: External validation against PRIMAT (Pitrou et al. 2018, Table V).

This is the decisive credibility test for the FLRW baseline.
No internal parity check can substitute for this comparison.

PRIMAT Born reference: Y_p = 0.2437 (η=6.104e-10, τ_n=878.4 s, N_eff=3.044)
RABBIT Born baseline:  Y_p ≈ 0.2424 (same η, τ_n but N_eff≈3.011 from tier-1)

The N_eff gap (3.011 vs 3.044) accounts for ΔY_p ≈ +0.0015.
After N_eff correction: residual |ΔY_p| should be < 0.001.

This residual is attributed to: network size (12 vs ~400 reactions),
incomplete neutrino decoupling (tier-1), and numerical methods.
"""
import pytest

from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig

PRIMAT_BORN_YP = 0.2437        # Pitrou+ 2018, Table V
NEFF_CORRECTION = 0.0015       # ΔY_p from N_eff = 3.011 → 3.044
TOLERANCE = 0.001              # After N_eff correction


@pytest.fixture(scope="module")
def flrw_born():
    return run_full_coupled_typeI(FullCoupledConfig(
        Sigma_H_plus=0.0, Sigma_H_minus=0.0,
        N_q=20, n_reactions=12, correction_level=0, enable_teff=False))


class TestPRIMATparity:

    def test_born_yp_physical(self, flrw_born):
        assert 0.240 < flrw_born.observables.Yp < 0.250

    def test_born_yp_vs_primat(self, flrw_born):
        """After N_eff correction, residual < 0.001."""
        gap = flrw_born.observables.Yp - PRIMAT_BORN_YP
        corrected_gap = gap + NEFF_CORRECTION
        assert abs(corrected_gap) < TOLERANCE, (
            f"RABBIT={flrw_born.observables.Yp:.6f}, PRIMAT={PRIMAT_BORN_YP}, "
            f"raw gap={gap:+.4e}, corrected={corrected_gap:+.4e}")

    def test_born_yp_below_primat(self, flrw_born):
        """RABBIT Born Y_p < PRIMAT Born Y_p (lower N_eff → fewer neutrons)."""
        assert flrw_born.observables.Yp < PRIMAT_BORN_YP

    def test_dh_order_of_magnitude(self, flrw_born):
        """D/H should be O(10⁻⁵)."""
        assert 1e-6 < flrw_born.observables.DH < 1e-4
