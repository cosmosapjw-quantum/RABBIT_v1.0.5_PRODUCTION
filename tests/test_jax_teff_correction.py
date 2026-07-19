from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax", reason="JAX required")

import jax.numpy as jnp
from numpy.polynomial.laguerre import laggauss

from rabbit.jax.teff_correction_jax import (
    pi_tilde_to_teff_quadrupole_jax, teff_angular_variance_jax, spectral_hardening_fd_jax,
    teff_corrected_monopole_exact_jax, teff_tangency_diagnostic_jax,
)
from rabbit.weak.teff_correction import (
    pi_tilde_to_teff_quadrupole, teff_angular_variance, spectral_hardening_fd,
    teff_corrected_monopole_exact, teff_tangency_diagnostic,
)


def test_teff_mapping_matches_numpy_reference():
    pi_tilde = 0.08
    T2_ref = pi_tilde_to_teff_quadrupole(pi_tilde)
    T2 = float(pi_tilde_to_teff_quadrupole_jax(pi_tilde))
    assert abs(T2 - T2_ref) < 1e-15
    sig_ref = teff_angular_variance(T2_ref)
    sig = float(teff_angular_variance_jax(T2))
    assert abs(sig - sig_ref) < 1e-15


def test_perturbative_hardening_matches_numpy_reference():
    q = np.linspace(0.2, 8.0, 25)
    Sigma2 = 3.0e-5
    ref = spectral_hardening_fd(q, Sigma2)
    got = np.asarray(spectral_hardening_fd_jax(jnp.asarray(q), Sigma2))
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-14)


def test_exact_closure_matches_numpy_reference():
    q_nodes, q_weights = laggauss(12)
    f0 = 1.0 / (np.exp(q_nodes) + 1.0)
    pi_tilde = 0.08
    ref = teff_corrected_monopole_exact(f0, q_nodes, pi_tilde)
    got = np.asarray(teff_corrected_monopole_exact_jax(f0, q_nodes, pi_tilde))
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-14)
    D_ref = teff_tangency_diagnostic(f0, q_nodes, pi_tilde, q_weights=q_weights)
    D = teff_tangency_diagnostic_jax(f0, q_nodes, pi_tilde, q_weights=q_weights)
    assert abs(D - D_ref) < 1e-12
