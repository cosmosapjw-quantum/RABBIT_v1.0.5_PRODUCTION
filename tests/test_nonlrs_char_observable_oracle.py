"""tests/test_nonlrs_char_observable_oracle.py — PR#16 (B3-lite): non-LRS characteristic observable
parity at Sigma_minus != 0.

PR#15 established the LRS characteristic-ray path as the exact shear-faithful reference. This closes
the one validation hole blocking the non-LRS (Sigma_minus != 0, biaxial) observable path from
'candidate' scope: the forward MAP is already ODE-cross-checked (test_pr_n1_nonlrs_primitives.py),
but the angular OBSERVABLE extractors that feed the RHS stresses -- extract_monopole_S2 and
extract_stress_minus_S2 -- were pinned only in the LRS reduction (Sigma_minus -> 0). Here they are
anchored against an INDEPENDENT first-principles oracle AT Sigma_minus != 0.

Oracle: the diagonal Type-I stretch A = diag(exp(-(S_+ + sqrt3 S_-)), exp(-(S_+ - sqrt3 S_-)),
exp(2 S_+)) acts on the initial direction n0; the transported FD argument rescales q -> q * ||A n0||
(so exp(2 I) = ||A n0||) and the S^2 push-forward Jacobian is J = det(A)/||A n0||^3 = ||A n0||^{-3}
(det A = 1, trace-free shear). Computing ||A n0|| directly from the geometry (NOT the production
log-domain helpers) independently validates the closed-form I, J and the monopole/stress assembly.

Scope: this validates the QUADRATURE assembly + the I/J closed forms + the measure push-forward at
Sigma_minus != 0. It does NOT re-validate the exact-diagonal-Type-I physics (the ODE test does) nor
the absolute -(sqrt3/2) stress normalization convention (anchored by the AP62/AP63 LRS-reduction
rate tests); the stress oracle independently supplies the (1-mu^2) cos 2phi angular SHAPE.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from rabbit.config.grids import MomentumGrid  # noqa: E402
from rabbit.transport.stageAB_state import fermi_dirac  # noqa: E402
from rabbit.jax.characteristic_rays_jax import (  # noqa: E402
    extract_monopole_jax,
    intensity_shift_jax,
    jacobian_jax,
    mu_current_jax,
)
from rabbit.jax.characteristic_rays_nonlrs_jax import (  # noqa: E402
    extract_monopole_S2,
    extract_stress_minus_S2,
    extract_stress_plus_S2,
    intensity_shift_nonlrs_jax,
    jacobian_nonlrs_jax,
    mu_phi_current_jax,
    setup_ray_grid_S2,
)

_SQRT3 = np.sqrt(3.0)
_N_Q = 20
_STRESS_NORM = -0.5 * _SQRT3  # the project |m|=2 normalization applied to the (1-mu^2)cos2phi shape


def _norm_geom(mu0, phi0, S_plus, S_minus):
    """||A n0|| from first principles (independent of the production log-domain helpers)."""
    sin0 = np.sqrt(np.maximum(1.0 - mu0 ** 2, 0.0))
    ex0, ey0, ez0 = sin0 * np.cos(phi0), sin0 * np.sin(phi0), mu0
    ax = np.exp(-(S_plus + _SQRT3 * S_minus))
    ay = np.exp(-(S_plus - _SQRT3 * S_minus))
    az = np.exp(2.0 * S_plus)
    return np.sqrt((ax * ex0) ** 2 + (ay * ey0) ** 2 + (az * ez0) ** 2)


def _grid_np(n_theta, n_phi):
    mu0, phi0, w_s2, X0, signs = setup_ray_grid_S2(n_theta, n_phi)
    return tuple(np.asarray(a) for a in (mu0, phi0, w_s2, X0, signs))


@pytest.mark.parametrize("S_plus,S_minus", [(0.15, 0.08), (0.20, -0.05)])
def test_intensity_jacobian_monopole_match_first_principles(S_plus, S_minus):
    """exp(2 I) == ||A n0||, J == ||A n0||^{-3}, and extract_monopole_S2 == the hand-assembled
    push-forward monopole -- all to machine precision -- at Sigma_minus != 0."""
    mu0, phi0, w_s2, _, _ = _grid_np(12, 16)
    norm = _norm_geom(mu0, phi0, S_plus, S_minus)
    I = np.asarray(intensity_shift_nonlrs_jax(jnp.asarray(mu0), jnp.asarray(phi0), S_plus, S_minus))
    J = np.asarray(jacobian_nonlrs_jax(jnp.asarray(mu0), jnp.asarray(phi0), S_plus, S_minus))
    assert np.max(np.abs(np.exp(2 * I) - norm) / norm) < 1e-12
    assert np.max(np.abs(J - norm ** -3) / norm ** -3) < 1e-12

    q = MomentumGrid(N_q=_N_Q).nodes
    mono_prod = np.asarray(extract_monopole_S2(jnp.asarray(I), jnp.asarray(J),
                                               jnp.asarray(w_s2), jnp.asarray(q)))
    f_vals = 1.0 / (np.exp(np.minimum(q[:, None] * norm[None, :], 500.0)) + 1.0)
    mono_hand = 0.5 * f_vals @ (w_s2 * norm ** -3)
    assert np.max(np.abs(mono_prod - mono_hand)) < 1e-12
    # Secondary non-triviality guard (the <1e-12 parity above is the real correctness check): the
    # 1/norm^3 measure factor is load-bearing -- dropping it must DISAGREE. Margin is modest at
    # these mild shears (~0.015 vs 1e-2); it only proves J is non-trivial, not that it is correct.
    mono_noJ = 0.5 * f_vals @ w_s2
    assert np.max(np.abs(mono_prod - mono_noJ)) > 1e-2


@pytest.mark.parametrize("S_plus,S_minus", [(0.15, 0.08), (0.20, -0.05)])
def test_stress_minus_matches_independent_shape_assembly(S_plus, S_minus):
    """extract_stress_minus_S2 == NORM * sum w_s2 J (1-mu^2)cos2phi exp(-8I), with the angular
    SHAPE (1-mu^2)cos2phi derived independently from the real |m|=2 harmonic, and it is genuinely
    non-zero at Sigma_minus != 0."""
    mu0, phi0, w_s2, _, _ = _grid_np(12, 16)
    I = intensity_shift_nonlrs_jax(jnp.asarray(mu0), jnp.asarray(phi0), S_plus, S_minus)
    J = jacobian_nonlrs_jax(jnp.asarray(mu0), jnp.asarray(phi0), S_plus, S_minus)
    mu, phi = mu_phi_current_jax(jnp.asarray(mu0), jnp.asarray(phi0), S_plus, S_minus)
    st_prod = float(extract_stress_minus_S2(I, J, mu, phi, jnp.asarray(w_s2), 1.0))

    mu_n, phi_n, I_n, J_n = (np.asarray(a) for a in (mu, phi, I, J))
    shape = (1.0 - mu_n ** 2) * np.cos(2.0 * phi_n)  # independent real |m|=2 angular shape
    st_hand = _STRESS_NORM * float(np.sum(w_s2 * J_n * shape * np.exp(-8.0 * I_n)))
    assert abs(st_prod - st_hand) / abs(st_hand) < 1e-10
    assert abs(st_prod) > 1e-2  # the minus-sector stress is a real, non-trivial signal


@pytest.mark.parametrize("S_plus,S_minus", [(0.15, 0.08), (0.20, -0.05)])
def test_stress_plus_matches_independent_shape_assembly(S_plus, S_minus):
    """The Pi_+ extractor (the m=0 quadrupole feeding dSigma_+/dN) also assembles correctly: it
    matches sum w_s2 J P2(mu) exp(-8I) with the independently-written P2 = (3 mu^2 - 1)/2 shape, and
    is genuinely non-zero (the dominant axisymmetric stress)."""
    mu0, phi0, w_s2, _, _ = _grid_np(12, 16)
    I = intensity_shift_nonlrs_jax(jnp.asarray(mu0), jnp.asarray(phi0), S_plus, S_minus)
    J = jacobian_nonlrs_jax(jnp.asarray(mu0), jnp.asarray(phi0), S_plus, S_minus)
    mu, _phi = mu_phi_current_jax(jnp.asarray(mu0), jnp.asarray(phi0), S_plus, S_minus)
    sp_prod = float(extract_stress_plus_S2(I, J, mu, jnp.asarray(w_s2), 1.0))

    mu_n, I_n, J_n = (np.asarray(a) for a in (mu, I, J))
    shape = 0.5 * (3.0 * mu_n ** 2 - 1.0)  # independent P2 angular shape
    sp_hand = float(np.sum(w_s2 * J_n * shape * np.exp(-8.0 * I_n)))
    assert abs(sp_prod - sp_hand) / abs(sp_hand) < 1e-10
    assert abs(sp_prod) > 1e-2


def test_integrated_observables_are_quadrature_converged():
    """The INTEGRATED observables (not just the pointwise closed forms) are converged in the S^2
    quadrature: stress_minus at (N_theta,N_phi)=(12,16) vs (24,32) agree to <5e-3."""
    def _stress(n_theta, n_phi):
        mu0, phi0, w_s2, _, _ = _grid_np(n_theta, n_phi)
        I = intensity_shift_nonlrs_jax(jnp.asarray(mu0), jnp.asarray(phi0), 0.15, 0.08)
        J = jacobian_nonlrs_jax(jnp.asarray(mu0), jnp.asarray(phi0), 0.15, 0.08)
        mu, phi = mu_phi_current_jax(jnp.asarray(mu0), jnp.asarray(phi0), 0.15, 0.08)
        return float(extract_stress_minus_S2(I, J, mu, phi, jnp.asarray(w_s2), 1.0))
    s_coarse, s_fine = _stress(12, 16), _stress(24, 32)
    assert abs(s_coarse - s_fine) / abs(s_fine) < 5e-3, f"not converged: {s_coarse} vs {s_fine}"


@pytest.mark.parametrize("n_phi", [1, 4, 16])
def test_lrs_reduction_at_zero_minus_matches_trusted_lrs_primitive(n_phi):
    """At Sigma_minus = 0 the non-LRS monopole reproduces the trusted LRS extract_monopole_jax
    (PR#15's exact reference) for N_phi in {1,4,16} -- so the azimuthal quadrature correctly
    integrates out the (absent) m=2 structure, not merely ignores the phi dimension -- and the
    m=2 stress vanishes (axisymmetry)."""
    S_plus = 0.15
    q = MomentumGrid(N_q=_N_Q).nodes
    mu0, phi0, w_s2, _, _ = _grid_np(12, n_phi)
    I = intensity_shift_nonlrs_jax(jnp.asarray(mu0), jnp.asarray(phi0), S_plus, 0.0)
    J = jacobian_nonlrs_jax(jnp.asarray(mu0), jnp.asarray(phi0), S_plus, 0.0)
    mono_nonlrs = np.asarray(extract_monopole_S2(I, J, jnp.asarray(w_s2), jnp.asarray(q)))

    mu0_l = np.polynomial.legendre.leggauss(12)[0]
    X0 = mu0_l ** 2 / (1.0 - mu0_l ** 2)
    signs = np.sign(mu0_l)
    I_l = intensity_shift_jax(jnp.asarray(X0), S_plus)
    mu_l = mu_current_jax(jnp.asarray(X0), jnp.asarray(signs), S_plus)
    J_l = jacobian_jax(jnp.asarray(X0), S_plus, mu_l)
    w_mu = np.polynomial.legendre.leggauss(12)[1]
    mono_lrs = np.asarray(extract_monopole_jax(I_l, J_l, jnp.asarray(w_mu), jnp.asarray(q)))
    assert np.max(np.abs(mono_nonlrs - mono_lrs)) < 1e-12

    mu0b, phi0b, w_s2b, _, _ = _grid_np(12, 16)
    I0 = intensity_shift_nonlrs_jax(jnp.asarray(mu0b), jnp.asarray(phi0b), S_plus, 0.0)
    J0 = jacobian_nonlrs_jax(jnp.asarray(mu0b), jnp.asarray(phi0b), S_plus, 0.0)
    mu0c, phi0c = mu_phi_current_jax(jnp.asarray(mu0b), jnp.asarray(phi0b), S_plus, 0.0)
    st0 = float(extract_stress_minus_S2(I0, J0, mu0c, phi0c, jnp.asarray(w_s2b), 1.0))
    assert abs(st0) < 1e-12, f"m=2 stress should vanish at Sigma_minus=0, got {st0}"


def _delta_rho(S_plus, S_minus, grid):
    q = grid.nodes
    mu0, phi0, w_s2, _, _ = _grid_np(12, 16)
    I = intensity_shift_nonlrs_jax(jnp.asarray(mu0), jnp.asarray(phi0), S_plus, S_minus)
    J = jacobian_nonlrs_jax(jnp.asarray(mu0), jnp.asarray(phi0), S_plus, S_minus)
    f_mono = np.asarray(extract_monopole_S2(I, J, jnp.asarray(w_s2), jnp.asarray(q)))
    fFD = fermi_dirac(q)
    return float(np.trapezoid(q ** 3 * f_mono, q) / np.trapezoid(q ** 3 * fFD, q) - 1.0)


def test_minus_sector_moves_energy_density_and_flips_sign():
    """Sigma_minus measurably changes the energy-density anisotropy (a stub ignoring it FAILS), and
    the m=2 stress flips sign with sign(Sigma_minus) -- the physical signature of the minus sector."""
    grid = MomentumGrid(N_q=_N_Q)
    d0 = _delta_rho(0.15, 0.0, grid)
    d8 = _delta_rho(0.15, 0.08, grid)
    assert abs(d8 / d0 - 1.0) > 0.1, f"Sigma_minus did not move delta-rho: {d0} vs {d8}"

    mu0, phi0, w_s2, _, _ = _grid_np(12, 16)

    def _stress(S_minus):
        I = intensity_shift_nonlrs_jax(jnp.asarray(mu0), jnp.asarray(phi0), 0.15, S_minus)
        J = jacobian_nonlrs_jax(jnp.asarray(mu0), jnp.asarray(phi0), 0.15, S_minus)
        mu, phi = mu_phi_current_jax(jnp.asarray(mu0), jnp.asarray(phi0), 0.15, S_minus)
        return float(extract_stress_minus_S2(I, J, mu, phi, jnp.asarray(w_s2), 1.0))

    assert _stress(0.08) * _stress(-0.08) < 0.0, "m=2 stress should flip sign with Sigma_minus"
