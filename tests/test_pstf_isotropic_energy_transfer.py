"""tests/test_pstf_isotropic_energy_transfer.py — PR#13 cross-track diagnostic (Bianchi side).

CERTIFIES the Bianchi isotropic nu-nu energy-transfer operator
(:func:`pstf_collision_operator.isotropic_collision_moment_rate`) as an INDEPENDENT exact
reference, in support of the cross-track Mangano-gap diagnostic
(docs/audit/A2_crosstrack_diagnostic_2026-06-30.md).

Why this is a SELF-CONTAINED diagnostic, not a matched two-kernel comparison: the production
exact-kernel N_eff=2.9934 run (jax_kernel_preflight) contains ONLY nu-e elastic + e+e- pair
annihilation and EXCLUDES nu-nu scattering, while the only production nu-nu kernel that exists
is unwired with an uncalibrated matrix_coeff=1.0 (DHS coefficient deferred to PR-T3C). So there
is no production N_eff carrying a production nu-nu rate to compare against — the apples-to-apples
match is a multi-PR effort (wire nu-nu as a 3rd channel + DHS calibration + measure conversion).
What CAN be certified today, in isolation, is that the Bianchi nu-nu operator is energy-conserving
at equilibrium, grid-resolvable, and nonperturbative — the prerequisites for any later match.

The three certifications (the equilibrium null + manual-sum identity live in
test_pstf_collision_operator.py; these are the NEW, non-duplicative measurements):

  1. RESOLUTION: the energy-transfer rate is UNDER-RESOLVED at low momentum-grid order (N<=8 is
     sign-unstable for the localized distortion) and converges only at N>=12 (adjacent N=20 vs 24
     agree to <1%). This is a STRICTER resolution bar than the production N_eff (flat at N_q=4) —
     a fact any future matched comparison must respect.
  2. NONPERTURBATIVE: under a number-density-preserving spectral distortion of amplitude a, the
     rate grows SUPER-LINEARLY (ratio ~3.3-3.8 per doubling, vs 2.0 for a pure linear response;
     effective local exponent ~1.8) — a beyond-linear-response (nonlinear Pauli-blocking)
     signature. NOTE: this is NOT a cubic power law (cubic would give ratio 8); the operator
     contains cubic monomials but the integrated rate's distortion-scaling is sub-quadratic. It is
     distinct from the shear test's Chat_1, which is a parity-FORCED exact odd cubic in the dipole
     amplitude (test_pstf_collision_operator_shear.py) — a different structure.
  3. DETAILED BALANCE at the convergence grid: the rate stays machine-zero at Fermi-Dirac
     equilibrium even at the high N used for (1), confirming the null is not a coarse-grid fluke.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rabbit.collisions.kernels import G_F_MEV
from rabbit.collisions.pstf_collision_operator import isotropic_collision_moment_rate

_GF2 = G_F_MEV ** 2


def _nunu_xpoly_factory(p1, p2, p3):
    """W_r x-polynomial for nu-nu diagonal scattering: 32 G_F^2 (E1 E2 - p1 p2 x)^2 (massless)."""
    a = p1 * p2  # massless E_i = p_i
    coeffs = (32 * _GF2 * a ** 2, -64 * _GF2 * a ** 2, 32 * _GF2 * a ** 2)
    return lambda y: coeffs


def _fermi_dirac(mu=0.0, T=1.0):
    return lambda p: 1.0 / (1.0 + math.exp((p - mu) / T))


def _grid(n, pmin=0.2, pmax=8.0):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (pmax - pmin)
    mid = 0.5 * (pmax + pmin)
    return mid + half * nodes, half * weights


def _energy_rate(f, n, n_y=24):
    p_nodes, p_weights = _grid(n)
    return isotropic_collision_moment_rate(f, p_nodes, p_weights, _nunu_xpoly_factory,
                                           power=3, n_y=n_y)


def _bump():
    """Localized multiplicative distortion (breaks detailed balance)."""
    fd = _fermi_dirac()
    return lambda p: min(0.999, fd(p) * (1.0 + 0.3 * math.exp(-((p - 2.0) ** 2))))


def test_low_Nq_under_resolved_high_Nq_converges():
    """The energy-transfer rate needs momentum-grid order N>=12. The convergence ladder (all
    measured at n_y=24): N=4 is qualitatively wrong (sign-flipped on the localized bump,
    rel-diff ~1.3 vs N=24); N=8 still materially off (~0.51); N=12 converged to order (~0.058);
    N=20 vs N=24 agree to ~3e-4. The production N_eff was flat already at N_q=4 -- this
    independent reference is a STRICTER resolution bar, a fact any matched comparison must respect."""
    f = _bump()
    r4 = _energy_rate(f, 4)
    r8 = _energy_rate(f, 8)
    r12 = _energy_rate(f, 12)
    r20 = _energy_rate(f, 20)
    r24 = _energy_rate(f, 24)
    # high-N adjacent convergence (measured ~3e-4)
    assert abs(r20 - r24) / abs(r24) < 1e-2, f"not converged at high N: r20={r20} r24={r24}"
    # N>=12 converged to order (measured ~0.058)
    assert abs(r12 - r24) / abs(r24) < 0.1, f"N=12 not converged to order: r12={r12} r24={r24}"
    # ... but N<=8 is NOT yet there: N=8 still materially off (~0.51), N=4 sign-flipped (~1.3).
    # The ladder must be monotone-improving (N=8 strictly farther than N=12) to back the prose.
    assert abs(r8 - r24) / abs(r24) > 0.1, f"N=8 unexpectedly converged: r8={r8} r24={r24}"
    assert abs(r4 - r24) / abs(r24) > 0.5, f"N=4 unexpectedly close: r4={r4} r24={r24}"
    assert abs(r8 - r24) > abs(r12 - r24), "ladder not monotone-improving (N=8 should beat N=12)"
    # converged value is a genuine non-zero physical scale, not a coarse-grid zero
    assert abs(r24) > 1e-25


def test_energy_transfer_is_nonperturbative_in_amplitude():
    """Number-preserving distortion of amplitude a -> SUPER-LINEAR rate growth (beyond linear
    response), evidence of the operator's nonlinear Pauli-blocking terms -- NOT a cubic power law.

    delta(p) = a (p - pbar) with pbar the NUMBER-weighted mean momentum (number density n ~ int
    p^2 f dp, so the number weight is w_i p_i^2 f_i -- the 1/2E factors in the collision integral
    are the matrix-element phase-space norm, not the number-density measure) holds the neutrino
    number density fixed, isolating the spectral (energy) response. A pure linear response would
    double the rate when a doubles; the exact operator more-than-triples it.
    """
    n = 20
    p_nodes, p_weights = _grid(n)
    fd = _fermi_dirac()
    num_w = p_weights * p_nodes ** 2 * np.array([fd(p) for p in p_nodes])  # number-density weight
    N_fd = float(np.sum(num_w))
    pbar = float(np.sum(num_w * p_nodes) / N_fd)

    amps = [0.05, 0.1, 0.2]
    rates = []
    for a in amps:
        # self-check the premise: the distortion preserves the number density to machine precision
        # (an exact discrete identity for this pbar -- guards against a future measure mismatch).
        dN = float(np.sum(num_w * a * (p_nodes - pbar)))
        assert abs(dN) / N_fd < 1e-12, f"distortion not number-preserving: dN/N={dN / N_fd}"
        f = lambda p, a=a: min(0.999, max(1e-9, fd(p) * (1.0 + a * (p - pbar))))
        rates.append(isotropic_collision_moment_rate(f, p_nodes, p_weights, _nunu_xpoly_factory,
                                                     power=3, n_y=24))
    # consistent sign (the response does not flip with amplitude for a fixed distortion shape)
    assert len({r > 0 for r in rates}) == 1, f"sign not consistent across amplitude: {rates}"
    # monotone in |a|
    mags = [abs(r) for r in rates]
    assert mags[0] < mags[1] < mags[2], f"not monotone in amplitude: {mags}"
    # SUPER-LINEAR: each doubling more-than-doubles the response (linear would be exactly 2.0)
    assert mags[1] / mags[0] > 2.5, f"a:2a ratio {mags[1] / mags[0]} not super-linear"
    assert mags[2] / mags[1] > 2.5, f"a:2a ratio {mags[2] / mags[1]} not super-linear"


def test_detailed_balance_null_at_convergence_grid():
    """At the high momentum-grid order used for the resolution test, Fermi-Dirac equilibrium
    gives a machine-zero rate -- the equilibrium null is exact, not a coarse-grid artifact, and
    NOT a globally-zero operator: the SAME grid produces an O(1e-24) rate out of equilibrium, so a
    broken operator returning 0 everywhere FAILS this test rather than passing the null vacuously."""
    p_nodes, p_weights = _grid(24)
    # anti-vacuity reference (measured ~1e-24): pins that the operator is alive on this grid.
    ref = isotropic_collision_moment_rate(_bump(), p_nodes, p_weights, _nunu_xpoly_factory,
                                          power=3, n_y=24)
    assert abs(ref) > 1e-25, f"out-of-equilibrium reference unexpectedly ~0: {ref}"
    for T, mu in [(1.0, 0.0), (1.2, 0.3)]:
        f = _fermi_dirac(mu, T)
        e = isotropic_collision_moment_rate(f, p_nodes, p_weights, _nunu_xpoly_factory,
                                            power=3, n_y=24)
        n = isotropic_collision_moment_rate(f, p_nodes, p_weights, _nunu_xpoly_factory,
                                            power=2, n_y=24)
        # absolute floor (measured ~1e-38 to ~1e-40, >=5 orders below) AND >=9 orders below the
        # live out-of-equilibrium scale -- both must hold, so the null cannot pass vacuously.
        assert abs(e) < 1e-33, f"energy rate {e} not ~0 at equilibrium (T={T} mu={mu})"
        assert abs(n) < 1e-33, f"number rate {n} not ~0 at equilibrium (T={T} mu={mu})"
        assert abs(e) < 1e-9 * abs(ref), f"energy null not far below non-eq scale: {e} vs {ref}"
