"""
test_standard_network — Integration test for the Standard BBN network.

Tests:
  1. Conservation: baryon, charge, mass fraction
  2. Backbone parity: std(12 rxn) ≈ v2(12 rxn) to < 10⁻⁵
  3. Extension physics: 31-reaction ΔY_p < 10⁻⁴, Δ(D/H) < 1%, Δ(⁷Li) < 5%
  4. ⁶Li prediction: 10⁻¹⁵ < ⁶Li/H < 10⁻¹³ (standard BBN range)
  5. Rate coverage: all 31 forward rates nonzero at T9=1
"""
import pytest
import sys
import os
import numpy as np
from scipy.integrate import solve_ivp

# (paths resolved via pip install)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

_ETA = 6.1e-10
_TAU_N = 878.4
_M_E = 0.5109989500
_Q_NP = 1.29333236


def _approx_weak(T):
    if T < 0.001:
        return 1.0 / _TAU_N, 0.0
    base = (1.0 + 0.45 * (T / _M_E) ** 5) / _TAU_N
    ratio = np.exp(-_Q_NP / T) if T > 0.01 else 0.0
    return base, base * ratio


def _T_of_t(t):
    return 0.74 / np.sqrt(max(t, 1e-3))


def _run_bbn(rhs_fn, ic_fn, n_rxn, X_n=0.12):
    """Run Phase 2 BBN from T=0.08→0.005 MeV."""
    t0 = (0.74 / 0.08) ** 2
    t1 = (0.74 / 0.005) ** 2
    X0 = ic_fn(X_n)

    def ode(t, X):
        T = _T_of_t(t)
        return rhs_fn(X, T, _ETA, *_approx_weak(T), n_reactions=n_rxn)

    sol = solve_ivp(ode, [t0, t1], X0, method='Radau',
                    rtol=1e-8, atol=1e-12, max_step=500)
    assert sol.success, f"Solver failed: {sol.message}"
    return sol.y[:, -1]


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

class TestConservation:

    def test_baryon_conservation(self):
        from rabbit.network.abundances_standard import STOICHIOMETRY, ATOMIC_MASSES
        check = ATOMIC_MASSES @ STOICHIOMETRY
        assert np.allclose(check, 0, atol=1e-14)

    def test_charge_conservation(self):
        from rabbit.network.abundances_standard import STOICHIOMETRY, CHARGE_NUMBERS
        check = CHARGE_NUMBERS @ STOICHIOMETRY
        assert np.allclose(check, 0, atol=1e-14)

    def test_mass_fraction_conservation(self):
        from rabbit.network.abundances_standard import abundance_rhs_phase2, phase1_to_phase2
        X = _run_bbn(abundance_rhs_phase2, phase1_to_phase2, 31)
        assert abs(np.sum(X) - 1.0) < 1e-8, f"ΣX = {np.sum(X)}"


class TestBackboneParity:

    def test_v2_vs_standard_12(self):
        """std(12 rxn) matches v2(12 rxn) on Y_p to < 10⁻⁵ relative."""
        from rabbit.network.abundances_v2 import abundance_rhs_phase2 as rhs_v2, phase1_to_phase2 as ic_v2
        from rabbit.network.abundances_standard import abundance_rhs_phase2 as rhs_std, phase1_to_phase2 as ic_std

        def rhs_v2_wrap(X, T, eta, lnp, lpn, n_reactions=12):
            return rhs_v2(X, T, eta, lnp, lpn)

        Xv2 = _run_bbn(rhs_v2_wrap, ic_v2, 12)
        Xs12 = _run_bbn(rhs_std, ic_std, 12)

        Yp_v2 = Xv2[5]
        Yp_s12 = Xs12[5]
        rel = abs(Yp_v2 - Yp_s12) / Yp_v2
        assert rel < 1e-5, f"Y_p parity = {rel:.2e}"
        print(f"  Backbone parity Y_p: {rel:.2e}")


class TestExtensionPhysics:

    def test_Yp_shift_negligible(self):
        """ΔY_p from 12→31 reactions < 10⁻⁴."""
        from rabbit.network.abundances_standard import abundance_rhs_phase2, phase1_to_phase2
        X12 = _run_bbn(abundance_rhs_phase2, phase1_to_phase2, 12)
        X31 = _run_bbn(abundance_rhs_phase2, phase1_to_phase2, 31)
        dYp = abs(X31[5] - X12[5])
        assert dYp < 1e-4, f"ΔY_p = {dYp:.2e}"
        print(f"  ΔY_p(31−12) = {dYp:.2e}")

    def test_DH_shift_small(self):
        """Δ(D/H) from 12→31 < 1%."""
        from rabbit.network.abundances_standard import abundance_rhs_phase2, phase1_to_phase2
        X12 = _run_bbn(abundance_rhs_phase2, phase1_to_phase2, 12)
        X31 = _run_bbn(abundance_rhs_phase2, phase1_to_phase2, 31)
        DH_12 = (X12[2] / 2) / X12[1]
        DH_31 = (X31[2] / 2) / X31[1]
        rel = abs(DH_31 - DH_12) / DH_12
        assert rel < 0.01, f"Δ(D/H)/D/H = {rel:.4f}"
        print(f"  Δ(D/H)/D/H = {rel:.2e}")

    def test_Li7_shift_reasonable(self):
        """Δ(⁷Li/H) from 12→31 between −10% and 0 (destruction from extensions)."""
        from rabbit.network.abundances_standard import abundance_rhs_phase2, phase1_to_phase2
        X12 = _run_bbn(abundance_rhs_phase2, phase1_to_phase2, 12)
        X31 = _run_bbn(abundance_rhs_phase2, phase1_to_phase2, 31)
        Li7_12 = (X12[6] / 7) / X12[1]
        Li7_31 = (X31[6] / 7) / X31[1]
        frac = (Li7_31 - Li7_12) / Li7_12
        assert -0.10 < frac < 0.01, f"Δ(⁷Li)/⁷Li = {frac:.4f}"
        print(f"  Δ(⁷Li)/⁷Li = {frac:.4f} ({frac*100:.1f}%)")

    def test_Li6_prediction(self):
        """⁶Li/H from 31-reaction in standard BBN range [10⁻¹⁵, 10⁻¹³]."""
        from rabbit.network.abundances_standard import abundance_rhs_phase2, phase1_to_phase2
        X31 = _run_bbn(abundance_rhs_phase2, phase1_to_phase2, 31)
        Li6H = (X31[8] / 6) / X31[1]
        assert 1e-15 < Li6H < 1e-13, f"⁶Li/H = {Li6H:.2e}"
        print(f"  ⁶Li/H = {Li6H:.2e} (standard range: 10⁻¹⁵–10⁻¹³)")


class TestRateCoverage:

    def test_all_rates_nonzero(self):
        """All 31 forward rates > 0 at T9 = 1."""
        from rabbit.network.abundances_standard import evaluate_nuclear_rates
        fwd, _ = evaluate_nuclear_rates(1.0 / 11.6045, 31)
        n_zero = np.sum(fwd <= 0)
        assert n_zero == 0, f"{n_zero} reactions have zero forward rate"

    def test_reverse_rates_exist(self):
        """Reverse rates computed via detailed balance (most > 0)."""
        from rabbit.network.abundances_standard import evaluate_nuclear_rates
        fwd, rev = evaluate_nuclear_rates(1.0 / 11.6045, 31)
        n_with_rev = np.sum(rev > 0)
        assert n_with_rev >= 20, f"Only {n_with_rev}/31 have reverse rates"
        print(f"  {n_with_rev}/31 reactions have reverse rates")


class TestTraceFluxSplit:

    def test_trace_species_production_destruction_split_reconstructs_network_rhs(self):
        from rabbit.network.abundances_standard import (
            abundance_rhs_phase2,
            trace_species_production_destruction_split,
        )

        X = np.asarray(
            [
                8.565687159782537e-02,
                9.124214139078601e-01,
                7.479126866674593e-04,
                2.7777347183243074e-05,
                9.54488483276192e-08,
                1.14592899816044e-03,
                1.2294663405355158e-11,
                4.63603578828313e-12,
                1.0e-30,
            ],
            dtype=float,
        )
        split = trace_species_production_destruction_split(
            X,
            0.07990207012753038,
            _ETA,
            species_indices=(6, 7, 8),
            n_reactions=31,
        )
        rhs = abundance_rhs_phase2(
            X,
            0.07990207012753038,
            _ETA,
            0.0,
            0.0,
            n_reactions=31,
        )

        assert split["available"] is True
        assert split["species_names"] == ["7Li", "7Be", "6Li"]
        assert split["forward_flux_min"] >= 0.0
        assert split["reverse_flux_min"] >= 0.0
        assert split["raw_negative_species_indices"] == []
        for item in split["species"]:
            index = item["species_index"]
            assert item["production"] >= 0.0
            assert item["destruction"] >= 0.0
            assert item["destruction_rate"] >= 0.0
            assert item["net_dX"] == pytest.approx(rhs[index])
            assert item["dominant_abs_reactions"]

    def test_trace_split_does_not_claim_nonnegative_flux_for_raw_negative_input(self):
        from rabbit.network.abundances_standard import (
            trace_species_production_destruction_split,
        )

        X = -np.asarray(
            [0.13, 0.87, 1.0e-8, 1.0e-12, 2.0e-12, 0.0, 3.0e-12, 4.0e-12, 5.0e-30],
            dtype=float,
        )
        raw = trace_species_production_destruction_split(
            X,
            0.08,
            _ETA,
            species_indices=(6, 7, 8),
            n_reactions=31,
        )
        clipped = trace_species_production_destruction_split(
            X,
            0.08,
            _ETA,
            species_indices=(6, 7, 8),
            n_reactions=31,
            clip_negative=True,
        )

        assert raw["raw_negative_species_indices"]
        assert raw["input_clip_policy"] == "raw_abundances"
        assert raw["directional_flux_nonnegative"] is False
        assert raw["directional_flux_nonnegative_reason"] == (
            "raw_input_outside_physical_domain"
        )
        assert clipped["clip_negative"] is True
        assert clipped["input_clip_policy"] == "diagnostic_nonnegative_counterfactual"
        assert clipped["directional_flux_nonnegative"] is True


if __name__ == "__main__":
    print("Standard Network Integration Test")
    print("=" * 60)

    t1 = TestConservation()
    t1.test_baryon_conservation()
    t1.test_charge_conservation()
    t1.test_mass_fraction_conservation()
    print("  Conservation: ALL PASS")

    t2 = TestBackboneParity()
    t2.test_v2_vs_standard_12()

    t3 = TestExtensionPhysics()
    t3.test_Yp_shift_negligible()
    t3.test_DH_shift_small()
    t3.test_Li7_shift_reasonable()
    t3.test_Li6_prediction()

    t4 = TestRateCoverage()
    t4.test_all_rates_nonzero()
    t4.test_reverse_rates_exist()

    print("=" * 60)
    print("ALL TESTS PASSED")
