"""tests/test_deterministic_reference_rate_scaling.py — BD612-F1 regression lock.

The deterministic weak-collision reference returns a physical ``df/dt`` rate in
MeV. Its dimensionless Hannestad-Madsen/DHS reduced kernels therefore require
``C ∝ G_F^2 T^5`` when ``G_F_MEV`` is in MeV^-2. At fixed spectral shape, the
energy-transfer moment must scale as T^5.

The reduced matrix-polynomial convention used by the source operators keeps the
surviving angular denominator at ``4*pi**3``. This file locks the T power and
the source prefactor helper so the clean core consumes raw MeV-rate ``C`` values
without a compensating temperature factor.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.collisions.kernels import (
    G_F_MEV,
    HM_REDUCED_COLLISION_DENOMINATOR,
    hm_reduced_collision_prefactor,
)
from rabbit.collisions.deterministic_reference import build_fixed_collision_quadrature
from rabbit.collisions.dynamic_collision_core import neutrino_collision_energy_transfer


def _quad():
    return build_fixed_collision_quadrature(
        N_q=24, N_nue_y2=24, N_nue_y3=24, N_pair_y2=24, N_pair_leg=16
    )


def _fixed_shape_cold(q, s=0.9):
    """FD at T_nu = s*T_gamma on the dimensionless q-grid — a spectral SHAPE that is
    independent of the absolute plasma temperature T_gamma, so the only T-dependence
    of the energy transfer is the prefactor's."""
    return 1.0 / (np.exp(np.minimum(q / s, 500.0)) + 1.0)


def _energy_transfer_ratio(T_lo, T_hi):
    quad = _quad()
    q = np.asarray(quad.q_nodes, dtype=float)
    f_cold = _fixed_shape_cold(q)
    dQ_lo, _ = neutrino_collision_energy_transfer(
        f_cold, f_cold, quadrature=quad, T_gamma_MeV=T_lo, species="nue"
    )
    dQ_hi, _ = neutrino_collision_energy_transfer(
        f_cold, f_cold, quadrature=quad, T_gamma_MeV=T_hi, species="nue"
    )
    return abs(dQ_hi) / abs(dQ_lo)


def test_energy_transfer_scales_as_T5():
    """At fixed spectral shape the weak energy-transfer moment must scale as T^5
    (physically-normalized df/dt ~ G_F^2 T^5). Doubling T -> factor 2^5 = 32."""
    ratio = _energy_transfer_ratio(2.0, 4.0)
    assert ratio == pytest.approx(2.0**5, rel=1e-3)


def test_hm_reduced_prefactor_uses_t5_and_four_pi_cubed():
    """The source convention is fixed at G_F^2 T^5 / (4*pi^3)."""
    T = 3.0
    expected = G_F_MEV**2 * T**5 / (4.0 * np.pi**3)
    assert HM_REDUCED_COLLISION_DENOMINATOR == pytest.approx(4.0 * np.pi**3)
    assert hm_reduced_collision_prefactor(T) == pytest.approx(expected)
