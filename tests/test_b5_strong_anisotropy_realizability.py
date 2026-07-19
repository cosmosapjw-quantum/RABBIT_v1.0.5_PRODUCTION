"""tests/test_b5_strong_anisotropy_realizability.py — PR#17 (B5): strong-anisotropy realizability.

The plan's B5 gate was "0 <= f <= 1 on a Lebedev grid at Sigma = 0.3, 0.5". That gate is VACUOUS:
the band-limited logit ``f = 1/(1+exp(-(q+A)))`` is in ``(0, 1)`` for ANY finite A by construction
(`augmented_pstf_distribution.fermi_dirac_from_logit`), so asserting ``0 <= f <= 1`` measures the
sigmoid's codomain, not the physics, and passes on arbitrary finite A. (Lebedev is also roadmap-only;
the real angular grid is tensor-product Gauss-Legendre(mu) x uniform-midpoint(phi).)

This replaces it with the NON-vacuous strong-anisotropy well-posedness measurements, exercising the
existing AP62 operator (`augmented_nonlrs_nonlinear_mode_rhs`) under frozen Sigma at |Sigma| in
{0.3, 0.5}:

  A1 finiteness     -- A_modes and the RHS stay finite (the RHS dataclass hard-raises on NaN/Inf).
  A2 bounded growth -- max|A| stays under an explicit ceiling and grows sub-exponentially (no runaway).
  A3 FD margin      -- min(f.min, 1-f.max) stays strictly > 0 (f approaches but never crosses the
                       boundary) and the logit argument stays below the sigmoid-underflow regime.
  A4 FLRW-quiet     -- the operator's source vanishes exactly at Sigma=0 (FLRW is a fixed point) and
                       is live (non-zero) at strong shear.
  A5 Sigma- antisym -- the W_- source is exactly antisymmetric in Sigma_minus at strong shear.
  A6 moment interior-- the reconstructed quadrupole pi_+ stays inside the realizable interval.
  A7 maxent contrast-- the Levermore/maxent M2 multiplier DIVERGES (and ceases to exist) as pi_+
                       approaches the realizability boundary, whereas the band-limited logit reaches a
                       strong-anisotropy state with a finite, modest coefficient and a positive FD
                       margin -- the well-posedness mechanism the logit buys (B.HR/B5).

Scope: this is OPERATOR-level realizability under frozen Sigma (the decisive slice). Driver-level
integration that co-evolves Sigma to 0.5 is a separate, heavier effort.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.laguerre import laggauss

from rabbit.transport.augmented_nonlrs_transport import (
    augmented_nonlrs_nonlinear_mode_rhs,
    build_nonlrs_nonlinear_transport_grid,
)
from rabbit.transport.augmented_pstf_distribution import reconstruct_distribution
from rabbit.transport.augmented_typeI_observables import stress_moments_from_distribution
from rabbit.transport.realizability_diagnostics import (
    RealizabilityBoundaryError,
    fd_boundary_margin,
    maxent_M2_multiplier,
)

_N_MU, _N_PHI, _N_Q = 20, 20, 10
_N_STEPS, _DN = 250, 1e-3
_MAX_A_CEILING = 12.0  # measured overall max ~7.58 at 250 steps; 1.6x regression-lock margin


def _grid():
    return build_nonlrs_nonlinear_transport_grid(N_mu=_N_MU, N_phi=_N_PHI)


def _q():
    q, _ = laggauss(_N_Q)
    return np.asarray(q, dtype=float)


def _evolve(grid, q, Sigma_plus, Sigma_minus, n_steps=_N_STEPS, dN=_DN):
    """Frozen-Sigma explicit-Euler evolution of the logit A_modes from the FLRW state (A=0).

    Returns (A_final, traj) where traj has per-step max|A|, fd_margin, pi_plus, logit_max.
    The RHS constructor raises on any non-finite array, so a true blow-up surfaces as an exception.
    """
    A = np.zeros((1, 3, q.size))
    basis = grid.s2_grid.basis_matrix
    qw = np.asarray(laggauss(q.size)[1], dtype=float) * np.exp(q) * q ** 3
    aw = grid.s2_grid.angular_weights
    Wp, Wm = grid.s2_grid.W_plus, grid.s2_grid.W_minus
    traj = []
    for _ in range(n_steps):
        rhs = augmented_nonlrs_nonlinear_mode_rhs(
            Sigma_plus=Sigma_plus, Sigma_minus=Sigma_minus, A_modes=A, q_nodes=q, grid=grid)
        A = A + dN * rhs.dA_modes
        f = reconstruct_distribution(A, q, basis)
        sm = stress_moments_from_distribution(f[0], qw, aw, Wp, W_minus=Wm)
        rho = float(np.sum(sm.rho_weighted))
        pi_plus = float(np.sum(sm.rho_weighted * sm.pi_plus_tilde) / rho)
        nodal = np.einsum("smq,ma->sqa", A, basis)  # logit deviation per (q, angle)
        logit_max = float(np.max(np.abs(q[None, :, None] + nodal)))
        traj.append((float(np.max(np.abs(A))), fd_boundary_margin(f), pi_plus, logit_max))
    return A, traj


@pytest.mark.parametrize("S_plus,S_minus", [(0.3, 0.0), (0.5, 0.0), (0.0, 0.5),
                                            (0.5 / np.sqrt(2), 0.5 / np.sqrt(2))])
def test_strong_shear_evolution_is_finite_bounded_and_realizable(S_plus, S_minus):
    """A1/A2/A3/A6: the logit transport stays finite, sub-exponentially bounded, strictly interior to
    the FD boundary, and inside the moment-realizability interval, through |Sigma| up to 0.5."""
    grid, q = _grid(), _q()
    A_final, traj = _evolve(grid, q, S_plus, S_minus)
    max_a = np.array([t[0] for t in traj])
    margins = np.array([t[1] for t in traj])
    pi_plus = np.array([t[2] for t in traj])
    logit_max = np.array([t[3] for t in traj])
    Wp = grid.s2_grid.W_plus

    # A1 finiteness (the RHS would have raised on a blow-up; confirm the final state too)
    assert np.all(np.isfinite(A_final))
    # A2 bounded + sub-exponential growth (no runaway)
    assert max_a[-1] < _MAX_A_CEILING, f"max|A| {max_a[-1]} exceeded ceiling"
    assert max_a[-1] / max_a[len(max_a) // 2] < 3.0, "growth not sub-exponential (runaway)"
    # A3 strictly interior to the FD boundary + below the sigmoid-underflow regime
    assert np.all(margins > 0.0), "f reached the Fermi-Dirac boundary"
    assert np.all(logit_max < 600.0), "logit argument entered the sigmoid-underflow regime"
    # A6 reconstructed W_+ quadrupole stays inside the realizable interval [Wp.min, Wp.max]
    # (for pure Sigma_minus the W_+ moment stays ~0 -- the anisotropy lives in the W_- m=2 mode --
    # so this is a realizability-interval membership check, not a magnitude claim)
    assert np.all(pi_plus > Wp.min() - 1e-9) and np.all(pi_plus < Wp.max() + 1e-9)
    # the state is a GENUINE strong-anisotropy one (not a trivial near-FLRW fixed point: FLRW->0)
    assert max_a[-1] > 1.0
    # discriminator vs a constant-source operator: the AP62 source is proportional to Sigma, so it
    # vanishes EXACTLY at Sigma=0 even at the strongly-perturbed final state (a constant-source op
    # would inject non-zero dA here and fail). This makes test 1 correctness-bearing on its own.
    quiet = augmented_nonlrs_nonlinear_mode_rhs(Sigma_plus=0.0, Sigma_minus=0.0,
                                                A_modes=A_final, q_nodes=q, grid=grid)
    assert np.max(np.abs(quiet.dA_modes)) < 1e-14


def test_flrw_is_an_exact_fixed_point_and_strong_shear_is_live():
    """A4: at A=0 the source vanishes EXACTLY at Sigma=0 (FLRW fixed point) and is non-zero at
    strong shear (the operator is genuinely forcing, not inert)."""
    grid, q = _grid(), _q()
    A0 = np.zeros((1, 3, q.size))
    quiet = augmented_nonlrs_nonlinear_mode_rhs(Sigma_plus=0.0, Sigma_minus=0.0,
                                                A_modes=A0, q_nodes=q, grid=grid)
    live = augmented_nonlrs_nonlinear_mode_rhs(Sigma_plus=0.5, Sigma_minus=0.0,
                                               A_modes=A0, q_nodes=q, grid=grid)
    assert np.max(np.abs(quiet.dA_modes)) < 1e-14
    assert np.max(np.abs(live.dA_modes)) > 1e-3


def test_sigma_minus_source_is_exactly_antisymmetric_at_strong_shear():
    """A5: at strong |Sigma_minus|=0.5 the W_- mode source flips sign exactly with sign(Sigma_minus)
    and the W_+ mode source stays zero (Sigma_minus does not source W_+ at the FLRW state)."""
    grid, q = _grid(), _q()
    A0 = np.zeros((1, 3, q.size))
    rp = augmented_nonlrs_nonlinear_mode_rhs(Sigma_plus=0.0, Sigma_minus=0.5,
                                             A_modes=A0, q_nodes=q, grid=grid)
    rm = augmented_nonlrs_nonlinear_mode_rhs(Sigma_plus=0.0, Sigma_minus=-0.5,
                                             A_modes=A0, q_nodes=q, grid=grid)
    assert np.max(np.abs(rp.dA_modes[0, 2] + rm.dA_modes[0, 2])) < 1e-12  # W_- antisymmetric
    assert np.max(np.abs(rp.dA_modes[0, 1])) < 1e-10                      # W_+ unsourced


def test_maxent_multiplier_diverges_at_boundary_while_logit_stays_finite():
    """A7 (decisive non-vacuity): the maxent / Levermore M2 multiplier diverges as the quadrupole
    approaches the realizability boundary and ceases to exist beyond it, whereas the band-limited
    logit reaches a strong-anisotropy state with a finite, modest coefficient and a positive FD
    margin -- no moment inversion. This is the well-posedness the logit buys over maxent closures."""
    grid, q = _grid(), _q()
    Wp, aw = grid.s2_grid.W_plus, grid.s2_grid.angular_weights

    # maxent multiplier: zero at pi=0, monotone increasing, steeply diverging toward the boundary
    lams = [maxent_M2_multiplier(pi, Wp, aw) for pi in (0.0, 0.5, 0.9, 0.95, 0.97)]
    assert abs(lams[0]) < 1e-9
    assert all(lams[i] < lams[i + 1] for i in range(len(lams) - 1))
    assert lams[-1] / lams[1] > 8.0, f"multiplier not diverging toward boundary: {lams}"
    # does not exist beyond the realizable interval (Wp.max ~ 0.979, Wp.min ~ -0.49)
    with pytest.raises(RealizabilityBoundaryError):
        maxent_M2_multiplier(0.99, Wp, aw)
    with pytest.raises(RealizabilityBoundaryError):
        maxent_M2_multiplier(-0.6, Wp, aw)

    # CONTRAST: the logit reaches strong anisotropy with finite, modest A and stays realizable
    A_final, traj = _evolve(grid, q, 0.5, 0.0)
    assert np.max(np.abs(A_final)) < _MAX_A_CEILING
    assert traj[-1][1] > 0.0  # FD margin strictly positive -- no boundary crossing, no inversion


def test_maxent_multiplier_is_the_correct_closure():
    """Helper CORRECTNESS (not just divergence): on a symmetric {-1,+1} kernel the multiplier is the
    analytic atanh(pi); on the W_+ angular grid a round-trip exp(lambda*W)->moment->invert recovers
    lambda. Without this, a wrong-but-monotone inversion would still pass the A7 contrast."""
    k = np.array([-1.0, 1.0])
    w = np.array([1.0, 1.0])
    for pi in (0.3, 0.6, 0.9):
        assert maxent_M2_multiplier(pi, k, w) == pytest.approx(float(np.arctanh(pi)), rel=1e-10)
    grid = _grid()
    Wp, aw = grid.s2_grid.W_plus, grid.s2_grid.angular_weights
    for lam_true in (1.0, 2.5, 5.0):
        z = lam_true * Wp
        z = z - z.max()
        e = np.exp(z)
        pi = float(np.sum(aw * Wp * e) / np.sum(aw * e))
        assert maxent_M2_multiplier(pi, Wp, aw) == pytest.approx(lam_true, rel=1e-9)


def test_fd_boundary_margin_value_and_guard():
    """fd_boundary_margin = min(f.min, 1-f.max) on a known input, and it rejects non-finite f."""
    assert fd_boundary_margin(np.array([0.1, 0.2, 0.95])) == pytest.approx(0.05)
    assert fd_boundary_margin(np.array([[0.4, 0.6]])) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        fd_boundary_margin(np.array([0.5, np.nan]))
