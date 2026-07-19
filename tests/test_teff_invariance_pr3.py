from __future__ import annotations

import numpy as np

from rabbit.transport.state import fermi_dirac
from rabbit.weak.teff_correction import (
    teff_corrected_monopole_exact,
    teff_quadrupole_theta_bounds,
    fd_bounds_summary,
)


def test_teff_quadrupole_theta_bounds_piecewise():
    # Positive pi_tilde: minimum at the equator P2=-1/2
    theta_min, theta_max = teff_quadrupole_theta_bounds(0.8)
    assert np.isclose(theta_min, 0.9)
    assert np.isclose(theta_max, 1.2)

    # Negative pi_tilde: minimum on the axis P2=+1
    theta_min, theta_max = teff_quadrupole_theta_bounds(-0.8)
    assert np.isclose(theta_min, 0.8)
    assert np.isclose(theta_max, 1.1)


def test_teff_exact_fd_preserves_fd_bounds_in_realizable_regime():
    q = np.linspace(0.05, 12.0, 64)
    f0 = fermi_dirac(q)
    out = teff_corrected_monopole_exact(f0, q, pi_tilde=0.8, N_mu=16)
    summary = fd_bounds_summary(out)
    assert summary['is_fd_admissible']
    assert summary['min'] >= -1e-12
    assert summary['max'] <= 1.0 + 1e-12


def test_teff_exact_fd_fallback_remains_fd_admissible_when_theta_nonpositive():
    q = np.linspace(0.05, 12.0, 64)
    f0 = fermi_dirac(q)
    # pi_tilde=8 -> T2=2, so Theta_min = 1 - T2/2 = 0 exactly (non-realizable boundary)
    out = teff_corrected_monopole_exact(f0, q, pi_tilde=8.0, N_mu=16)
    summary = fd_bounds_summary(out)
    assert summary['is_fd_admissible']
