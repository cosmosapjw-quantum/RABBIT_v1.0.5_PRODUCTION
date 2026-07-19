"""tests/test_pstf_trilinear_kernel.py — Stage 0 foundation (exact PSTF collision).

The exact all-orders PSTF collision (neutrino_large_anisotropy_exact_PSTF_collision_ko.md)
rests on the particle-hole reduction: with G_i = 1-2 f_i the Fermi-blocking factor
Lambda_r collapses to LINEAR + TRILINEAR in G (the quartic/quadratic G-terms cancel),
exact at arbitrary anisotropy. This pins that foundation against the EXISTING tested
deterministic_reference.collision_statistical_factor (the direct gain-loss form), and
the logit form's manifest detailed balance.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.collisions.deterministic_reference import collision_statistical_factor
from rabbit.collisions.pstf_trilinear_kernel import (
    lambda_r_particle_hole,
    lambda_r_logit,
    occupancy_to_logit,
    particle_hole_G,
)


def _random_occ(rng, n=2000):
    # occupancies strictly in (0,1), spanning the full Pauli-blocking range
    return [rng.uniform(1e-6, 1 - 1e-6, size=n) for _ in range(4)]


def test_particle_hole_decomposition_matches_tested_factor():
    """8 Lambda_r = sum eps G + sum eps eps eps G G G equals the tested direct factor."""
    rng = np.random.default_rng(0)
    f1, f2, f3, f4 = _random_occ(rng)
    ph = lambda_r_particle_hole(f1, f2, f3, f4)
    ref = collision_statistical_factor(f1, f2, f3, f4)
    assert np.allclose(ph, ref, rtol=0, atol=1e-13)


def test_logit_form_matches_tested_factor():
    """sinh/cosh logit form equals the tested direct factor."""
    rng = np.random.default_rng(1)
    f1, f2, f3, f4 = _random_occ(rng)
    eta = [occupancy_to_logit(f) for f in (f1, f2, f3, f4)]
    lg = lambda_r_logit(*eta)
    ref = collision_statistical_factor(f1, f2, f3, f4)
    assert np.allclose(lg, ref, rtol=1e-12, atol=1e-13)


def test_detailed_balance_annihilation_with_drift():
    """eta_1+eta_2 = eta_3+eta_4 (any common drift, incl. large dipole) -> Lambda_r = 0."""
    rng = np.random.default_rng(2)
    # Build etas satisfying the balance condition with arbitrary (large) values.
    eta1 = rng.uniform(-8, 8, size=1000)
    eta2 = rng.uniform(-8, 8, size=1000)
    eta3 = rng.uniform(-8, 8, size=1000)
    eta4 = eta1 + eta2 - eta3  # enforce eta1+eta2 = eta3+eta4
    lg = lambda_r_logit(eta1, eta2, eta3, eta4)
    assert np.max(np.abs(lg)) < 1e-13


def test_particle_hole_G_range_and_definition():
    f = np.linspace(1e-6, 1 - 1e-6, 50)
    G = particle_hole_G(f)
    assert np.all(G <= 1.0) and np.all(G >= -1.0)
    assert np.allclose(G, 1.0 - 2.0 * f)
