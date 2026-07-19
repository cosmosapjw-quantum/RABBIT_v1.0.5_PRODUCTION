"""tests/test_dynamic_collision_driver.py — PR-C7: FLRW-endpoint decoupling driver.

Acceptance suite for the comoving-frame FLRW neutrino-decoupling driver that wraps the
clean dynamic-collision core (``dynamic_collision_core``) in a stiff ODE integration
reaching the BBN decoupling endpoint. Two load-bearing gates:

  Gate A (frame validation): collisionless ⟹ N_eff = 3.00 exactly and T_ν/T_γ = (4/11)^{1/3},
          purely from the e± entropy boost to T_γ (f frozen in comoving y). If this fails the
          comoving-frame / z-bookkeeping is wrong.
  Gate B (blocker-fixed acceptance): collision-on reaches the endpoint with a small POSITIVE
          N_eff correction (3.00–3.15, NOT the 4.17 anomaly, NOT a heavy-bank collapse to ~0.001).

Plus a non-circular energy-conservation test that exercises the REAL driver path: the plasma
collision-loss rate fed to the γ channel must equal the physical ν energy-gain rate derived
from the SAME df the driver evolves (energy conserved BY CONSTRUCTION). No external anchor
exists for Bianchi-I BBN with a collision term; these are INTERNAL consistency numbers only.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.collisions.deterministic_reference import build_fixed_collision_quadrature
from rabbit.collisions.dynamic_collision_core import (
    _collision_field,
    spectral_energy_moment,
)
from rabbit.thermo.eos_photon_electron import drho_dT_plasma_with_electron_mu
from rabbit.thermo.nudec_coupled import _plasma_temperature_base_rhs, hubble_3T

from rabbit.collisions.dynamic_collision_driver import (
    DecouplingResult,
    _diagnostic_T_nu,
    _make_rhs,
    _map_thermal_field_to_comoving,
    _resample_comoving_to_thermal,
    integrate_flrw_decoupling,
    plasma_prefactor,
)

_I_FD = 7.0 * np.pi**4 / 120.0
_TNU_TGAMMA_STD = (4.0 / 11.0) ** (1.0 / 3.0)  # 0.713765...
_Z_STD = (11.0 / 4.0) ** (1.0 / 3.0)


def _quad(n=24):
    return build_fixed_collision_quadrature(
        N_q=n, N_nue_y2=n, N_nue_y3=n, N_pair_y2=n, N_pair_leg=16
    )


# ---------------------------------------------------------------------------
# Gate A — collisionless frame validation
# ---------------------------------------------------------------------------
def test_collisionless_recovers_analytic_neff_3():
    res = integrate_flrw_decoupling(n_q=24, collisions=False)
    assert isinstance(res, DecouplingResult)
    assert res.reached_endpoint
    assert res.solver_success  # FINDING 2: fail-closed guard must confirm success.
    assert np.isfinite(res.N_eff)
    # N_eff = 3 exactly: f frozen in comoving y (I_pair = I_FD), z_final = (11/4)^{1/3}.
    assert res.N_eff == pytest.approx(3.0, abs=0.02)
    assert res.z_final == pytest.approx(_Z_STD, rel=2e-3)
    # Both banks track (4/11)^{1/3} purely from the plasma entropy boost.
    assert res.T_nu_e_over_gamma == pytest.approx(_TNU_TGAMMA_STD, abs=0.002)
    assert res.T_nu_x_over_gamma == pytest.approx(_TNU_TGAMMA_STD, abs=0.002)
    # FINDING 3: the clamps must not have silently repaired an unphysical state.
    assert res.max_clip_excursion < 1e-6, res.max_clip_excursion
    assert res.min_hubble_MeV > 0.0


# ---------------------------------------------------------------------------
# Gate B — collision-on blocker-fixed acceptance
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_collision_on_reaches_endpoint_physical_neff_no_collapse():
    res = integrate_flrw_decoupling(n_q=24, collisions=True)
    assert res.reached_endpoint
    assert res.solver_success  # FINDING 2: fail-closed guard.
    # Small POSITIVE correction — not the ~4.17 anomaly, not a collapse.
    assert 3.00 <= res.N_eff <= 3.15, f"N_eff landed at {res.N_eff}"
    # Heavy bank did NOT collapse (blocker symptom was T_νx/T_γ ~ 0.001).
    assert res.T_nu_x_over_gamma > 0.5
    assert res.T_nu_e_over_gamma > 0.5
    # FINDING 3: clamps did not silently repair an unphysical state.
    assert res.max_clip_excursion < 1e-6, res.max_clip_excursion
    assert res.min_hubble_MeV > 0.0
    # Everything finite.
    for v in (
        res.N_eff,
        res.T_nu_e_over_gamma,
        res.T_nu_x_over_gamma,
        res.z_final,
        res.T_gamma_final,
        res.N_final,
    ):
        assert np.isfinite(v)
    assert np.all(np.isfinite(res.f_nue_final))
    assert np.all(np.isfinite(res.f_nux_final))


@pytest.mark.slow
def test_collision_on_frame_scale_is_tight():
    # FINDING 4: Gate A never exercises the frame resample (its ν moments are a pinned
    # identity). The collision-on run does; a frame-scale bug the wide N_eff band absorbs
    # shifts T_ν/T_γ off (4/11)^{1/3}. They land ~0.715–0.719, so abs 0.01 is tight.
    res = integrate_flrw_decoupling(n_q=24, collisions=True)
    assert res.T_nu_e_over_gamma == pytest.approx(_TNU_TGAMMA_STD, abs=0.01)
    assert res.T_nu_x_over_gamma == pytest.approx(_TNU_TGAMMA_STD, abs=0.01)


@pytest.mark.slow
def test_nue_warmer_than_nux():
    # ν_e (charged + neutral current) couples longer than ν_x (neutral current only),
    # so it stays warmer relative to the photon bath. Spike: 0.7204 vs 0.7169.
    res = integrate_flrw_decoupling(n_q=24, collisions=True)
    assert res.T_nu_e_over_gamma > res.T_nu_x_over_gamma


# ---------------------------------------------------------------------------
# FINDING 2 — fail-closed on a failed integration (no finite garbage N_eff)
# ---------------------------------------------------------------------------
def test_failed_integration_fails_closed():
    # An impossible span (N_span_max too small for the terminal T_γ event to fire) must
    # NOT silently return a finite N_eff. Sibling nudec_coupled fails closed the same way.
    with pytest.raises(RuntimeError):
        integrate_flrw_decoupling(collisions=False, N_span_max=0.5)


# ---------------------------------------------------------------------------
# FINDING 5 — collisionless N_eff is n_q-convergent
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_q", [16, 24, 32, 48])
def test_collisionless_neff_is_nq_convergent(n_q):
    # The q-Laguerre collapse's signature was discretization-dependence. Collisionless is
    # cheap; N_eff must agree across n_q to abs 1e-3.
    res = integrate_flrw_decoupling(n_q=n_q, collisions=False)
    assert res.reached_endpoint
    assert res.N_eff == pytest.approx(2.9934, abs=1e-3), f"n_q={n_q}: N_eff={res.N_eff}"


# ---------------------------------------------------------------------------
# §Plasma — energy conservation BY CONSTRUCTION (exercises the real driver path)
# ---------------------------------------------------------------------------
def test_plasma_prefactor_is_half_over_pi_squared():
    # PHYS = T_γ⁴/(2π²); the spike's 1/π² is a 2× g double-count.
    T_gamma = 2.0
    assert plasma_prefactor(T_gamma) == pytest.approx(T_gamma**4 / (2.0 * np.pi**2), rel=1e-14)


def test_interp_roundtrip_bounds_error():
    """FINDING 1 support: quantify the comoving↔thermal interp round-trip error.

    The two interpolations (_resample_comoving_to_thermal, _map_thermal_field_to_comoving)
    are DIFFERENT quadratures of the same continuum integral. Their disagreement is exactly
    why the OLD plasma coupling (thermal bank-moment) and the ν df-moment (interp #2) did not
    match — the ~8–12% path split that made energy conservation fail. FIX 1 makes conservation
    IMMUNE to this error (the plasma loses exactly what the ν df gains), but the error still
    limits the collision RATE accuracy, so we pin its documented magnitude here.

    On a coarse q-Laguerre grid (nodes spanning 0.04 → 112) the round-trip of a smooth FD f
    at z=1.2 is only accurate in ABSOLUTE terms:
      - max |f_round − f| ≲ 0.02 (small),
    but the RELATIVE error is LARGE on the physically-populated interior (nodes where the
    collision integrand has weight, f > 1e-6): it reaches tens of percent because linear
    interpolation of a steeply-decaying exponential systematically over-shoots. This large
    relative error is the whole reason FIX 1 conserves BY CONSTRUCTION rather than by trusting
    the two quadratures to agree. We assert the documented bounds so a regression that either
    fixes the interp (tightening the bound) or worsens it (breaking it) is flagged."""
    quad = _quad(32)
    q = np.asarray(quad.q_nodes, dtype=float)
    Y = q.copy()
    z = 1.2
    f = 1.0 / (np.exp(Y) + 1.0)  # smooth FD on Y
    f_th = _resample_comoving_to_thermal(Y, f, q, z)
    f_round = _map_thermal_field_to_comoving(q, f_th, Y, z)

    # Absolute error is small everywhere.
    max_abs_err = float(np.max(np.abs(f_round - f)))
    assert max_abs_err < 0.02, f"max abs interp round-trip error {max_abs_err:.3e}"

    # Relative error on the populated interior (f > 1e-6, inside both interp supports) is
    # LARGE — documented at O(0.1–0.5). This is the motivation for the conserving-by-
    # construction coupling; a naive "the two quadratures agree" assumption is FALSE here.
    lo = q[0] * z
    hi = q[-1] / z
    interior = (Y >= lo) & (Y <= hi) & (f > 1e-6)
    assert interior.sum() >= 5
    max_rel_interior = float(np.max(np.abs(f_round[interior] - f[interior]) / f[interior]))
    assert 0.05 < max_rel_interior < 2.0, (
        f"interior interp round-trip rel error {max_rel_interior:.3f} outside documented "
        "[0.05, 2.0] band — the coupling relies on FIX 1 conserving despite this, so a "
        "large change here is a real regression signal"
    )


def test_energy_conserving_plasma_coupling():
    """Conservation is BY CONSTRUCTION, verified through the REAL ``rhs`` path.

    At a z≠1 cold-ν state, call the built driver ``rhs`` once. The plasma collision-source
    energy-loss rate fed to the γ channel must equal the total physical ν energy-gain rate
    computed from the SAME df the ``rhs`` returns:

      ν physical gain / e-fold = (T_γ⁴/z⁴)/(2π²)·(2·∫Y³ df_nue dY + 4·∫Y³ df_nux dY),
      plasma collision-loss / e-fold = −(dT_γ − dT_base)·drho_dT_plasma,

    with dT_base the collisionless plasma rate at the same T_γ. This runs interp #2 + the
    frame factors (unlike the old tautological version), so it catches any future break in the
    conserving coupling. It holds to rel 1e-10 regardless of interpolation quality."""
    quad = _quad(24)
    q = np.asarray(quad.q_nodes, dtype=float)
    qw = np.asarray(quad.q_weights, dtype=float)
    Y = q.copy()
    n = Y.size

    # Build the actual collision-on driver rhs.
    rhs, Y_r, q_r, qw_r, _ = _make_rhs(quad, collisions=True)
    np.testing.assert_array_equal(Y_r, Y)

    # A genuinely cold-ν state (ν colder than the plasma): FD in comoving y, warm T_γ, z≠1.
    # Pick N so a = exp(N), z = a·T_γ ≈ 1.2 (the in-run regime where the paths disagreed).
    T_gamma = 2.0
    z_target = 1.2
    a = z_target / T_gamma
    N = np.log(a)
    z = a * T_gamma
    f_com = 1.0 / (np.exp(Y) + 1.0)
    state = np.concatenate([f_com.copy(), f_com.copy(), [T_gamma]])

    out = rhs(N, state)
    df_nue = out[:n]
    df_nux = out[n : 2 * n]
    dT_gamma = float(out[2 * n])

    # ── ν physical energy gain per e-fold, from the df the rhs actually returned ──
    G_nue = spectral_energy_moment(df_nue, Y, qw)  # ∫Y³ df_nue dY (single ν_e species)
    G_nux = spectral_energy_moment(df_nux, Y, qw)  # ∫Y³ df_nux dY (single ν_x species)
    frame = (T_gamma**4 / z**4) / (2.0 * np.pi**2)
    nu_gain = frame * (2.0 * G_nue + 4.0 * G_nux)

    # ── plasma collision-loss per e-fold: the extra dT_γ beyond the collisionless base ──
    dT_base = _plasma_temperature_base_rhs(T_gamma, 0.0)
    drho_dT = drho_dT_plasma_with_electron_mu(T_gamma, 0.0)
    plasma_loss = -(dT_gamma - dT_base) * drho_dT

    assert plasma_loss == pytest.approx(nu_gain, rel=1e-10)
    # A genuinely nonzero, correctly-signed cold-ν heating channel (not a trivial 0==0):
    # the plasma LOSES energy (positive loss) and the ν banks GAIN it.
    assert nu_gain > 0.0
    assert plasma_loss > 0.0


def test_collisionless_plasma_coupling_is_pure_base_rate():
    """Sanity partner to the conservation test: with collisions off, dT_γ equals the
    collisionless base rate exactly (no collision source), so plasma_loss = 0."""
    quad = _quad(24)
    q = np.asarray(quad.q_nodes, dtype=float)
    Y = q.copy()
    n = Y.size
    rhs, _, _, _, _ = _make_rhs(quad, collisions=False)
    T_gamma = 2.0
    a = 1.2 / T_gamma
    N = np.log(a)
    f_com = 1.0 / (np.exp(Y) + 1.0)
    state = np.concatenate([f_com.copy(), f_com.copy(), [T_gamma]])
    out = rhs(N, state)
    dT_gamma = float(out[2 * n])
    dT_base = _plasma_temperature_base_rhs(T_gamma, 0.0)
    assert dT_gamma == pytest.approx(dT_base, rel=1e-12)
