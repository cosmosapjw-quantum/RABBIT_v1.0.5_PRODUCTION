"""tests/test_pstf_collision_operator.py — Stage 0 isotropic collision-moment evaluator.

Wires the validated reduced kernel F_0 into the isotropic HM collision integral
(neutrino_decoupling_PSTF_HM_generalization_ko.md, eq 616-622):

    Chat_0(p1) = (2/(2pi)^4)(1/2E1) int p2^2 dp2/2E2 int p3^2 dp3/2E3 Lambda_r^(0) F_0,

with Lambda_r^(0) = (1-f1)(1-f2) f3 f4 - f1 f2 (1-f3)(1-f4) the gain-minus-loss factor and
p4 = p1+p2-p3 on-shell. This is the first live collision operator built from the exact
PSTF substrate (isotropic limit).

Rigorous GATE -- detailed balance: at any Fermi-Dirac equilibrium f = 1/(1+e^{(p-mu)/T}),
energy conservation p1+p2 = p3+p4 forces Lambda_r^(0) = f1 f2 f3 f4 (e^{p1+p2}-e^{p3+p4}) = 0
POINTWISE, so Chat_0 = 0 to machine precision regardless of F_0 (evaluating f4 at the exact
p4, not interpolated). Out of equilibrium Chat_0 is finite and non-zero.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rabbit.collisions.kernels import G_F_MEV
from rabbit.collisions.pstf_collision_operator import (
    isotropic_collision_moment_C0,
    isotropic_collision_moment_rate,
)

_GF2 = G_F_MEV ** 2


def _nunu_xpoly_factory(p1, p2, p3):
    """Return the W_r x-polynomial callable y -> (A0,A1,A2) (y-independent for nu-nu)."""
    e1e2 = p1 * p2  # massless E_i = p_i
    p1p2 = p1 * p2
    coeffs = (32 * _GF2 * e1e2 ** 2, -64 * _GF2 * e1e2 * p1p2, 32 * _GF2 * p1p2 ** 2)
    return lambda y: coeffs


def _fermi_dirac(mu=0.0, T=1.0):
    return lambda p: 1.0 / (1.0 + math.exp((p - mu) / T))


def _momentum_grid(n=10, pmin=0.2, pmax=8.0):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (pmax - pmin)
    mid = 0.5 * (pmax + pmin)
    return mid + half * nodes, half * weights


def test_detailed_balance_equilibrium_is_zero():
    """Fermi-Dirac equilibrium -> Lambda_r=0 pointwise -> Chat_0 = 0 (machine precision)."""
    p_nodes, p_weights = _momentum_grid()
    for T, mu in [(1.0, 0.0), (1.3, 0.5), (0.8, -0.2)]:
        f = _fermi_dirac(mu, T)
        for p1 in (0.5, 1.5, 3.0):
            C0 = isotropic_collision_moment_C0(f, p1, p_nodes, p_weights,
                                               _nunu_xpoly_factory, n_y=32)
            assert abs(C0) < 1e-30, f"T={T} mu={mu} p1={p1}: C0={C0}"


def test_out_of_equilibrium_is_finite_nonzero():
    """A non-equilibrium distribution gives a finite, non-zero collision moment."""
    p_nodes, p_weights = _momentum_grid()
    fd = _fermi_dirac()
    # multiplicative distortion (still in (0,1)) -> breaks detailed balance
    f = lambda p: min(0.999, fd(p) * (1.0 + 0.3 * math.exp(-((p - 2.0) ** 2))))
    vals = [isotropic_collision_moment_C0(f, p1, p_nodes, p_weights,
                                          _nunu_xpoly_factory, n_y=32)
            for p1 in (0.5, 1.5, 3.0)]
    assert all(math.isfinite(v) for v in vals)
    assert any(abs(v) > 0.0 for v in vals)


def test_detailed_balance_far_smaller_than_nonequilibrium():
    """Equilibrium moment is many orders below the non-equilibrium scale (not a fluke 0)."""
    p_nodes, p_weights = _momentum_grid()
    fd = _fermi_dirac()
    f_neq = lambda p: min(0.999, fd(p) * (1.0 + 0.3 * math.exp(-((p - 2.0) ** 2))))
    c_eq = isotropic_collision_moment_C0(fd, 1.5, p_nodes, p_weights, _nunu_xpoly_factory, n_y=32)
    c_neq = isotropic_collision_moment_C0(f_neq, 1.5, p_nodes, p_weights, _nunu_xpoly_factory, n_y=32)
    assert abs(c_neq) > 1e-26            # genuinely non-zero physical scale (~8e-26)
    assert abs(c_eq) < 1e-12 * abs(c_neq)  # DB residual ~4e-40 -> ratio ~5e-15


def test_energy_transfer_rate_vanishes_at_equilibrium():
    """Detailed balance: the p^3 (energy) and p^2 (number) moments of Chat_0 vanish at FD."""
    p_nodes, p_weights = _momentum_grid()
    for T, mu in [(1.0, 0.0), (1.2, 0.3)]:
        f = _fermi_dirac(mu, T)
        e_rate = isotropic_collision_moment_rate(f, p_nodes, p_weights, _nunu_xpoly_factory,
                                                 power=3, n_y=24)
        n_rate = isotropic_collision_moment_rate(f, p_nodes, p_weights, _nunu_xpoly_factory,
                                                 power=2, n_y=24)
        # residual ~1e-37 (DB) vs ~1e-24 out-of-equilibrium scale -> 1e-30 bites a real break
        assert abs(e_rate) < 1e-30, f"energy rate {e_rate} not ~0 at equilibrium"
        assert abs(n_rate) < 1e-30, f"number rate {n_rate} not ~0 at equilibrium"


def test_energy_transfer_rate_finite_nonzero_out_of_equilibrium():
    p_nodes, p_weights = _momentum_grid()
    fd = _fermi_dirac()
    f = lambda p: min(0.999, fd(p) * (1.0 + 0.3 * math.exp(-((p - 2.0) ** 2))))
    e_rate = isotropic_collision_moment_rate(f, p_nodes, p_weights, _nunu_xpoly_factory,
                                             power=3, n_y=24)
    assert math.isfinite(e_rate) and abs(e_rate) > 0.0


def test_energy_transfer_rate_equals_manual_moment_sum():
    """The rate is exactly sum_i w_i p_i^power Chat_0(p_i) -- the energy/number moment."""
    p_nodes, p_weights = _momentum_grid(n=6)
    fd = _fermi_dirac()
    f = lambda p: min(0.999, fd(p) * (1.0 + 0.2 * math.exp(-((p - 1.5) ** 2))))
    manual = sum(w * (p ** 3) * isotropic_collision_moment_C0(f, p, p_nodes, p_weights,
                                                              _nunu_xpoly_factory, n_y=20)
                 for p, w in zip(p_nodes, p_weights))
    rate = isotropic_collision_moment_rate(f, p_nodes, p_weights, _nunu_xpoly_factory,
                                           power=3, n_y=20)
    assert rate == pytest.approx(manual, rel=1e-12)
