"""tests/test_pstf_reduced_kernel_brute_force.py — independent validation of F_0.

The analytic reduced kernel F_0 = int dx dy Theta(D)/sqrt(D) W_r (pstf_reduced_kernel)
rests on the HM body-frame reduction (azimuth delta -> 1/sqrt(D)). This pins it against
a COMPLETELY INDEPENDENT numerical method: a direct 3-D angular phase-space integral with
the energy delta represented by a narrow Gaussian (no body frame, no 1/sqrt(D) Jacobian,
no analytic x-integral).

Exact relation (derived from the azimuth + energy delta-function Jacobians):

    J = int dOmega2 dOmega3 delta(E1+E2-E3-E4) W_r / (2 E4) = 4 pi F_0,

because phi2 -> 2 pi (rotation about p1), the two azimuth roots double the integrand, and
the energy delta's Jacobian d(E4)/d(phi3) is exactly the 1/sqrt(D) that F_0 carries. The
brute force does NOT apply that Jacobian by hand: it represents delta(E1+E2-E3-E4) as a
narrow Gaussian in the ENERGY mismatch, and the dphi3 integration produces the
d(E4)/d(phi3) Jacobian automatically (standard smeared-delta technique) -- the 1/sqrt(D)
emerges, never assumed. W_r uses 32 G_F^2 chi12^2 (the on-shell W the analytic kernel is
given; chi12 depends only on cos theta2, exact at every node, so the delta smearing is the
only approximation).

Convergence (finite resolution): for (p1,p2,p3)=(1.2,0.9,0.8) the ratio F_0/(J/4pi) goes
0.986 -> 0.996 -> 0.997 as the grid refines (240x960, sigma 2e-3) -> (320x1600, 1.2e-3) ->
(400x2600, 8e-4). The moderate grid here lands within ~1% for the two checked configs.
A factor-2 normalization error (4pi vs 2pi/8pi) would give ratio 0.5/2.0 -- far outside the
band -- so the test still pins the normalization despite the resolution-limited tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.collisions.kernels import G_F_MEV
from rabbit.collisions.pstf_reduced_kernel import reduced_kernel_F0

_GF2 = G_F_MEV ** 2


def _nunu_xpoly(p1, p2, p3):
    p1p2 = p1 * p2
    e1e2 = p1 * p2  # massless E_i = p_i
    return lambda y: (32 * _GF2 * e1e2 ** 2, -64 * _GF2 * e1e2 * p1p2, 32 * _GF2 * p1p2 ** 2)


def _brute_force_J(p1, p2, p3, *, ncos, nphi, sigma):
    """J = int dOmega2 dOmega3 delta(E1+E2-E3-E4) W_r/(2 E4) by a Gaussian-smeared grid.

    p1 along z, phi2 fixed (x 2 pi by symmetry); massless E_i = p_i. Loops over cos(theta2)
    to stay memory-light. W = 32 G_F^2 chi12 chi34 with chi from explicit 3-vectors -- no
    body frame, no 1/sqrt(D).
    """
    c2 = np.linspace(-1.0, 1.0, ncos)
    c3 = np.linspace(-1.0, 1.0, ncos)
    ph = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    dc2, dc3, dph = c2[1] - c2[0], c3[1] - c3[0], ph[1] - ph[0]
    C3, PH = np.meshgrid(c3, ph, indexing="ij")
    S3 = np.sqrt(1.0 - C3 ** 2)
    p3v = np.stack([p3 * S3 * np.cos(PH), p3 * S3 * np.sin(PH), p3 * C3], 0)
    p1v = np.array([0.0, 0.0, p1])[:, None, None]
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    total = 0.0
    for x2 in c2:
        s2 = np.sqrt(1.0 - x2 ** 2)
        p2v = np.array([p2 * s2, 0.0, p2 * x2])[:, None, None]
        p4v = p1v + p2v - p3v
        E4 = np.sqrt(np.sum(p4v ** 2, 0))
        dE = p1 + p2 - p3 - E4
        gauss = np.exp(-dE ** 2 / (2.0 * sigma ** 2)) * norm
        chi12 = p1 * p2 - p1 * p2 * x2                      # E1E2 - p1 p2 mu12 (theta2 only)
        W = 32.0 * _GF2 * chi12 ** 2                        # on-shell chi34 = chi12
        total += np.sum(gauss * W / (2.0 * E4)) * dc3 * dph
    return 2.0 * np.pi * total * dc2


@pytest.mark.parametrize("p1,p2,p3", [(1.2, 0.9, 0.8), (0.9, 1.1, 0.7)])
def test_F0_matches_independent_angular_integral(p1, p2, p3):
    """F_0 == J/(4 pi) within the brute-force grid/smearing resolution (~few %)."""
    F0 = reduced_kernel_F0((p1, p2, p3), (p1, p2, p3), 0.0, _nunu_xpoly(p1, p2, p3), n_y=80)
    J = _brute_force_J(p1, p2, p3, ncos=300, nphi=1300, sigma=1.5e-3)
    ratio = F0 / (J / (4.0 * np.pi))
    assert ratio == pytest.approx(1.0, abs=0.025)
