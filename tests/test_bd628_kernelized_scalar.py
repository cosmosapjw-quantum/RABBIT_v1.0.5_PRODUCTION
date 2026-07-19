"""tests/test_bd628_kernelized_scalar.py — BD628 W-complete: κ₂ → κ₂_eff(T) in the
production tier-2 CHARACTERISTIC ray-space lane (transport.characteristic_residual).

BD627 W1 wired a full-kernel C₂ into projected_operator.py, but the production tier-2
scipy CHARACTERISTIC lane damps ℓ=2 via ``characteristic_residual.py`` (ray-space
``delta_K = −relax·κ₂·(Γ/H)·(K−1)``, constant κ₂ = 5/3) — so W1's C₂ fired 0× there
(LEDGER F-033).  BD628 replaces that constant with the first-principles, T-dependent
κ₂_eff(T) derived from the validated ν_e ℓ=2 kernel.

Gates:
  (a) FLAG-DEFAULT BIT-IDENTITY: the default mode ("calibrated_rta") is bit-identical
      (np.array_equal on delta_I / delta_J) to the pre-BD628 arithmetic (KAPPA_ELL2).
  (b) kernelized_scalar changes delta_K by exactly the κ₂_eff/κ₂ ratio (~0.24).
  (c) κ₂_eff(T) is finite, positive, O(0.1–1), and H-independent.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.collisions.projected_operator import KAPPA_ELL0, KAPPA_ELL2
from rabbit.collisions.kernelized_ell2 import kappa2_eff
from rabbit.transport.characteristic_residual import apply_species_residual_relaxation
from rabbit.collisions.species import Species, as_species
from rabbit.transport.characteristic_species import SPECIES as CHAR_SPECIES


N_MU = 12
_T_STATE = dict(T_gamma=1.6, T_nu_e=1.4, T_nu_x=1.3, H_inv_sec=2.0e-2)


def _state(seed=628):
    rng = np.random.default_rng(seed)
    I = rng.standard_normal(N_MU) * 1e-2
    J = 1.0 + rng.standard_normal(N_MU) * 1e-2
    w0 = np.abs(rng.standard_normal(N_MU)) + 0.1
    return I, J, w0


def _wmean(vals, w):
    """Bit-for-bit replica of characteristic_residual._weighted_mean."""
    norm = float(np.sum(w))
    return float(np.sum(w * vals) / norm)


def _reference_deltas(I, J, w0, *, relax, g_ell0, g_ell2):
    """Golden replica of the pre-BD628 delta_I / delta_J arithmetic (KAPPA_ELL2)."""
    I_arr = np.asarray(I, dtype=np.float64)
    J_arr = np.asarray(J, dtype=np.float64)
    w = np.asarray(w0, dtype=np.float64)
    I_mean = _wmean(I_arr, w)
    I_res = I_arr - I_mean
    delta_I = -float(relax) * KAPPA_ELL0 * g_ell0 * I_res
    delta_I = delta_I - _wmean(delta_I, w)
    K_res = J_arr * np.exp(-8.0 * I_res)
    delta_K = -float(relax) * KAPPA_ELL2 * g_ell2 * (K_res - 1.0)
    delta_J = np.exp(8.0 * I_res) * delta_K + 8.0 * J_arr * delta_I
    return delta_I, delta_J


# ─────────────────────────────────────────────────────────────────────
# (a) FLAG-DEFAULT BIT-IDENTITY
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("sp", CHAR_SPECIES)
def test_flag_default_bit_identity(sp):
    I, J, w0 = _state()
    r_default = apply_species_residual_relaxation(
        species=sp, I=I, J=J, w0=w0, **_T_STATE)
    r_calib = apply_species_residual_relaxation(
        species=sp, I=I, J=J, w0=w0, ell2_collision_mode="calibrated_rta", **_T_STATE)

    # default == explicit calibrated_rta, bit-for-bit
    assert np.array_equal(r_default.delta_I, r_calib.delta_I)
    assert np.array_equal(r_default.delta_J, r_calib.delta_J)

    # and both match the golden pre-BD628 arithmetic (uses the returned rates so the
    # float-op order is identical to the source)
    ref_I, ref_J = _reference_deltas(
        I, J, w0, relax=1.0,
        g_ell0=r_default.gamma_over_H_ell0, g_ell2=r_default.gamma_over_H_ell2)
    assert np.array_equal(r_default.delta_I, ref_I)
    assert np.array_equal(r_default.delta_J, ref_J)


@pytest.mark.parametrize("sp", CHAR_SPECIES)
def test_kernelized_full_falls_back_to_calibrated(sp):
    """The full q-resolved kernel is not implemented in ray space → calibrated here."""
    I, J, w0 = _state(seed=7)
    r_calib = apply_species_residual_relaxation(
        species=sp, I=I, J=J, w0=w0, ell2_collision_mode="calibrated_rta", **_T_STATE)
    r_kern = apply_species_residual_relaxation(
        species=sp, I=I, J=J, w0=w0, ell2_collision_mode="kernelized", **_T_STATE)
    assert np.array_equal(r_calib.delta_I, r_kern.delta_I)
    assert np.array_equal(r_calib.delta_J, r_kern.delta_J)


# ─────────────────────────────────────────────────────────────────────
# (b) kernelized_scalar changes delta_K by κ₂_eff/κ₂
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("sp", CHAR_SPECIES)
def test_kernelized_scalar_scales_delta_K(sp):
    # Uniform I ⇒ I_res = 0 ⇒ delta_I = 0 and delta_J = delta_K exactly, isolating κ₂.
    I = np.full(N_MU, 3e-3)
    J = np.linspace(0.85, 1.15, N_MU)   # K_res = J ≠ 1 → delta_K ≠ 0
    w0 = np.ones(N_MU)

    r_calib = apply_species_residual_relaxation(
        species=sp, I=I, J=J, w0=w0, ell2_collision_mode="calibrated_rta", **_T_STATE)
    r_kern = apply_species_residual_relaxation(
        species=sp, I=I, J=J, w0=w0, ell2_collision_mode="kernelized_scalar", **_T_STATE)

    # delta_I untouched by the ℓ=2 scalar
    assert np.array_equal(r_calib.delta_I, r_kern.delta_I)
    assert np.allclose(r_calib.delta_I, 0.0)

    # T_bank per species: nue/nuebar → T_nu_e, nux → T_nu_x
    bank = _T_STATE["T_nu_e"] if as_species(sp) in (Species.NUE, Species.NUEBAR) \
        else _T_STATE["T_nu_x"]
    k2 = kappa2_eff(bank, as_species(sp).value)
    expected = k2 / KAPPA_ELL2

    ratio = r_kern.delta_J / r_calib.delta_J   # = delta_K_kern / delta_K_calib
    assert np.allclose(ratio, expected)
    # nue/nuebar ≈ 0.24, nux ≈ 0.17 (weaker NC-only coupling); both < 1 (the kernel
    # damps ℓ=2 strictly WEAKER than the ad-hoc 5/3).
    assert 0.1 < expected < 0.35, f"κ₂_eff/κ₂ = {expected:.3f}"


# ─────────────────────────────────────────────────────────────────────
# (c) κ₂_eff(T): finite, positive, O(0.1–1), H-independent
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("species", ["nue", "nuebar", "nux"])
def test_kappa2_eff_finite_positive_order_unity(species):
    for T in (0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        k = kappa2_eff(T, species)
        assert np.isfinite(k)
        assert 0.05 < k < 1.0, f"κ₂_eff({T},{species}) = {k} not O(0.1–1)"


def test_kappa2_eff_target_at_2MeV():
    """LEDGER F-033: κ₂_eff(2 MeV, nue) ≈ 0.24 × (5/3) ≈ 0.40 (4× under the ad-hoc 5/3)."""
    k = kappa2_eff(2.0, "nue")
    assert 0.3 < k < 0.5, f"κ₂_eff(2 MeV) = {k}, expected 0.3–0.5"
    assert k < KAPPA_ELL2   # kernel under-damps vs the calibrated 5/3


def test_kappa2_eff_H_independent():
    # H cancels by construction (C₂ ∝ 1/H, Γ/H ∝ 1/H); equal to machine precision
    # (the two 1/H divisions round independently, so ~1e-15 not bit-exact).
    for species in ("nue", "nux"):
        a = kappa2_eff(2.0, species, H_inv_sec=1.0)
        b = kappa2_eff(2.0, species, H_inv_sec=10.0)
        c = kappa2_eff(2.0, species, H_inv_sec=1.0e-3)
        assert np.isclose(a, b, rtol=1e-12, atol=0.0)
        assert np.isclose(a, c, rtol=1e-12, atol=0.0)


def test_kappa2_eff_nuebar_shares_nue_bank():
    assert kappa2_eff(2.0, "nuebar") == kappa2_eff(2.0, "nue")


def test_kappa2_eff_low_T_bounded():
    """T ≪ m_e: the bare kernel/Γ ratio diverges (e± suppression asymmetry) — the clamp
    keeps κ₂_eff bounded and finite so it cannot inject a pathological damping rate into
    the BBN-epoch solve (which reaches sub-0.1-MeV temperatures)."""
    for T in (0.02, 0.05, 0.1, 0.2):
        for species in ("nue", "nux"):
            k = kappa2_eff(T, species)
            assert np.isfinite(k)
            assert 0.0 <= k <= KAPPA_ELL2 + 1e-12   # capped at the calibrated 5/3


def test_kappa2_eff_nunu_refused():
    with pytest.raises(NotImplementedError):
        kappa2_eff(2.0, "nu_nu")
