"""rabbit.jax.collision_operator_jax — JAX twin of PhysicalCollisionOperator (B4).

A pure-functional, jittable, differentiable JAX reimplementation of
``rabbit.collisions.projected_operator.PhysicalCollisionOperator.evaluate`` -- the
incomplete-decoupling collision source the SciPy tier-2 driver uses. The numpy operator is
collisionless-blocked on the JAX characteristic transport path (``forward_likelihood.py`` raises a
ValueError on ``enable_collisions`` there); this twin is the first step to wiring it in.

It is **parity-locked** to the numpy operator to machine precision
(``tests/test_jax_collision_operator_parity.py``). To keep parity exact it reuses the SAME physical
constants (``G_F_MEV``, ``A_TOTAL_*``, ``_C_RATE``, ``KAPPA_ELL0/2``, ``_M_E``, ``_MEV2S``) and the
same closed form:

    C_0(q) = kappa0 * (Gamma_s/H) * [ Psi0_target(q) - Psi0(q) ]
    C_2(q) = - kappa2 * (Gamma_s/H) * Psi2(q)
    Psi0_target(q) = (T_g - T_nu)/T_g * q * (1 - f0(q))        (-> 0 at T_g == T_nu, since DT == 0)
    Gamma_s/H = (_C_RATE/pi^5) * G_F^2 * a_s * F(m_e/T_g) * T_g^4 * T_nu / (H_inv_sec/_MEV2S)

Unlike the numpy operator this twin takes EXPLICIT arrays/scalars (no ``None`` defaults, no internal
Hubble estimate) -- the JAX RHS always supplies q-grid, couplings, temperatures and H.

PARITY IS NOT PHYSICAL VALIDATION (external audit BD601, 2026-06-30). The numpy operator this
reproduces is a CALIBRATED RELAXATION-TIME (RTA) model, not a first-principles collision integral:
``_C_RATE = 210`` is calibrated to reproduce ``N_eff ~ 3.044``, the source is linearized in
``dT/T`` (~7% error at dT/T = 1%), ``Gamma`` is momentum-independent, and there is no nu-nu
scattering. Machine-precision agreement with that model is ENGINEERING fidelity (jittable +
differentiable), and is **NOT a physical validation of the collision physics**.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Parity is exact ONLY because this twin shares the numpy operator's constants verbatim -- it
# deliberately imports them (including the private ``_``-prefixed ones) rather than re-defining them,
# so the two operators cannot silently drift. If a refactor renames any of these, the parity test
# (tests/test_jax_collision_operator_parity.py) fails loudly; do NOT fork local copies.
from rabbit.collisions.kernels import A_TOTAL_NUE, A_TOTAL_NUX, G_F_MEV
from rabbit.collisions.projected_operator import (
    KAPPA_ELL0,
    KAPPA_ELL2,
    _M_E,
    _MEV2S,
    _PI,
)
from rabbit.thermo.nudec_tables import _C_RATE

#: Default 3-species coupling vector [nu_e, nu_e-bar, nu_x] (matches PhysicalCollisionOperator).
DEFAULT_A_COUPLING = jnp.array([A_TOTAL_NUE, A_TOTAL_NUE, A_TOTAL_NUX])


class JAXCollisionResult(NamedTuple):
    """Mirror of PhysicalCollisionResult (all arrays jnp)."""
    C_ell0: jnp.ndarray            # (n_species, N_q)
    C_ell2: jnp.ndarray            # (n_species, N_q)
    source_ell0: jnp.ndarray       # (n_species, N_q)
    damping_ell0: jnp.ndarray      # (n_species, N_q)
    Gamma_over_H: jnp.ndarray      # (n_species,)
    energy_transfer: jnp.ndarray   # (n_species,)
    total_energy_transfer: jnp.ndarray  # scalar


def _fermi_dirac_jax(q: jnp.ndarray) -> jnp.ndarray:
    """f0(q) = 1/(e^q + 1); mirrors projected_operator._fermi_dirac (clips q at 500)."""
    return 1.0 / (jnp.exp(jnp.minimum(q, 500.0)) + 1.0)


def electron_mass_suppression_jax(T_MeV):
    """F(m_e/T_g); mirrors projected_operator._electron_mass_suppression (T>50 -> 1, T<0.01 -> 0,
    else the sigmoid). The exponent is clipped so the (jnp.where-evaluated) unused branches stay
    finite for safe gradients."""
    x = _M_E / jnp.maximum(T_MeV, 1e-30)
    sig = 1.0 / (1.0 + jnp.exp(jnp.clip(2.5 * (x - 2.3), -500.0, 500.0)))
    return jnp.where(T_MeV > 50.0, 1.0, jnp.where(T_MeV < 0.01, 0.0, sig))


def collision_rate_per_efold_jax(T_gamma, T_nu, H_inv_sec, a_coupling):
    """Dimensionless Gamma/H; mirrors projected_operator.collision_rate_per_efold (per-species when
    a_coupling is a vector)."""
    F = electron_mass_suppression_jax(T_gamma)
    Gamma_MeV = (_C_RATE / _PI**5) * G_F_MEV**2 * a_coupling * F * T_gamma**4 * T_nu
    H_MeV = H_inv_sec / _MEV2S
    # Guard the division so the (jnp.where-evaluated) ratio stays finite when H_MeV underflows --
    # an inf in the unused branch would nan the gradient. Mirrors the numpy `if H_MeV<1e-50: 0`.
    H_safe = jnp.maximum(H_MeV, 1e-50)
    return jnp.where(H_MeV < 1e-50, 0.0, Gamma_MeV / H_safe)


def physical_collision_rhs_jax(
    Psi0: jnp.ndarray,
    Psi2: jnp.ndarray,
    T_gamma,
    T_nu,
    H_inv_sec,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
    a_coupling: jnp.ndarray = DEFAULT_A_COUPLING,
) -> JAXCollisionResult:
    """JAX twin of ``PhysicalCollisionOperator.evaluate``.

    Parameters
    ----------
    Psi0, Psi2 : (n_species, N_q)
        Monopole / quadrupole perturbations.
    T_gamma, T_nu, H_inv_sec : scalars
        Photon temperature [MeV], neutrino temperature [MeV], Hubble rate [s^-1].
    q_nodes, q_weights : (N_q,)
        Momentum grid nodes + quadrature weights (the same ``MomentumGrid`` the numpy op uses).
    a_coupling : (n_species,)
        Per-species coupling; defaults to the 3-species [nu_e, nu_e-bar, nu_x] vector.
    """
    f0 = _fermi_dirac_jax(q_nodes)

    # Psi0_target(q) = DT/T_g * q * (1 - f0). Mirror PhysicalCollisionOperator.source_term EXACTLY,
    # including its epsilon guard: return zeros when T_g and T_nu agree to within floating-point
    # epsilon (so parity holds across the whole eps band, not only at bit-exact equality).
    DT_over_Tg = (T_gamma - T_nu) / jnp.maximum(T_gamma, 1e-30)
    in_band = jnp.abs(T_gamma - T_nu) < 1e-15 * jnp.maximum(
        jnp.maximum(jnp.abs(T_gamma), jnp.abs(T_nu)), 1e-30)
    target = jnp.where(in_band, 0.0, DT_over_Tg * q_nodes * (1.0 - f0))   # (N_q,)

    gamma_h = collision_rate_per_efold_jax(T_gamma, T_nu, H_inv_sec, a_coupling)  # (n_species,)
    gh_col = gamma_h[:, None]                              # (n_species, 1)

    source = KAPPA_ELL0 * gh_col * target[None, :]         # (n_species, N_q)
    damping = -KAPPA_ELL0 * gh_col * Psi0
    C0 = source + damping
    C2 = -KAPPA_ELL2 * gh_col * Psi2

    w_energy = q_weights * jnp.exp(q_nodes) * q_nodes**3 * f0   # (N_q,)
    energy_xfer = jnp.sum(w_energy[None, :] * C0, axis=1)       # (n_species,)

    return JAXCollisionResult(
        C_ell0=C0, C_ell2=C2, source_ell0=source, damping_ell0=damping,
        Gamma_over_H=gamma_h, energy_transfer=energy_xfer,
        total_energy_transfer=jnp.sum(energy_xfer),
    )
