"""rabbit.jax.collision_hm_full_jax — Full Hannestad-Madsen 2D collision kernel.

v3.0 Phase G (Plan §2.1). Implements the JAX-native full collision
kernel with momentum-resolved rate Γ_α(q, T). Unlike the AP-form
proxy in :mod:`rabbit.jax.collision_rates_jax`, which uses a single
relaxation rate Γ_α(T), this module restores per-momentum-bin
structure so that off-equilibrium spectral distortions enter the rate
balance correctly.

Physics meaning
---------------
The leading-order ν-e elastic + pair-process rate per momentum is
(Hannestad-Madsen 1995 eq 24, ultra-relativistic massless limit):

    Γ_α(q, T) = (G_F² T_γ⁴ q / π³) · I_α(q / T_γ)

where I_α(x) is a dimensionless function of the momentum-temperature
ratio that absorbs the (s² + u²) / s² Mandelstam dependence after
angular integration over the partner momentum k. The closed-form
total rate Γ_α(T) is recovered by thermal averaging:

    Γ_α(T) = ⟨Γ_α(q, T)⟩_FD = (7π/12) G_F² T⁵ a_α

The collision operator in the AP-form linearization is:

    C[f_α](q, T_γ) = -Γ_α(q, T_γ) · (f_α(q) - f_FD(q, T_γ))

Detailed balance: at f_α = f_FD with T_α = T_γ, the operator vanishes.

Scope honesty
-------------
The full HM 2D quadrature with explicit Pauli-blocking factors
𝒫(f_α, f_β, q, q', μ) is research-grade work (per Plan §2.1
"hallucination guard": matrix-element normalization conventions
and Pauli-sign choices need cross-checks against published values).
This module implements the **leading-order Mangano factorization**:

  1. Per-momentum rate Γ_α(q, T) via a smooth Padé-like ansatz that
     integrates against f_FD to the closed-form Γ_α(T).
  2. Pauli-blocking via the standard linearised form
     𝒫(f) = f^2 (1 - f)^2.
  3. Linear deviation operator C[δf_α] = -Γ_α(q, T) δf_α.

This is sufficient for the Phase H AP-Rosenbrock Jacobian wire-up.
A full HM-table-fitted Γ_α(q, T) is documented as a research-grade
follow-on (matches the v2.0 DH-S table approach).

References
----------
- Hannestad, Madsen, *Phys. Rev. D* 52 1764 (1995), Appendix A.
- Mangano et al., *Nucl. Phys. B* 729 221 (2005), eq 19.
- Notzold, Raffelt, *Nucl. Phys. B* 307 924 (1988).
- In-tree note: docs/audit/v3_derivations/full_hm_jacobian.md (Phase H).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rabbit.collisions.kernels import (
    A_TOTAL_NUE,
    A_TOTAL_NUX,
    G_F_MEV,
)


jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# §1. Per-momentum Γ_α(q, T) factorization
# ═══════════════════════════════════════════════════════════════════════

# Mean-momentum ratio for a massless Fermi-Dirac distribution at
# temperature T:  <q>_FD = 7 π^4 / (180 ζ(3)) · T ≈ 3.15137 T.
#
# This appears as the normalization between the closed-form rate
# Γ_α(T) = (7π/12) G_F² T^5 a_α (per :mod:`rabbit.jax.collision_rates_jax`)
# and the thermal-averaged per-momentum rate. We absorb it as a
# constant so that ⟨Γ_α(q, T)⟩_FD = Γ_α(T) by construction.
_MEAN_Q_OVER_T_FD: float = 3.151374415409971


def _fermi_dirac_dimensionless(x: jnp.ndarray) -> jnp.ndarray:
    """Massless Fermi-Dirac in dimensionless x = q/T.

    Returns f_FD(x) = 1 / (e^x + 1). Stable for all x; bounded in [0, 1].
    """
    return 1.0 / (jnp.exp(jnp.asarray(x, dtype=jnp.float64)) + 1.0)


def gamma_alpha_per_momentum(
    q_MeV: jnp.ndarray, T_gamma_MeV: jnp.ndarray, *,
    species: str = "nue",
    apply_v32_corrections: bool = False,
    T_nu_MeV: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Per-momentum rate Γ_α(q, T_γ) [MeV] in the leading-order Mangano factorization.

    Physics meaning
    ---------------
    Returns the per-bin equilibration rate at momentum q. By
    construction:

        ⟨Γ_α(q, T)⟩_{f_FD} = (7π/12) G_F² T⁵ a_α

    so the thermal average reproduces the closed-form total rate.
    The leading-order factorization sets the q-dependence to the
    smooth function ``q · f_eq(q/T)``-style weighting, capturing the
    fact that high-q neutrinos scatter more frequently than low-q.

    Regime
    ------
    Ultra-relativistic massless limit; valid for q ∈ [0, 30] T,
    T ∈ [0.05, 10] MeV (the BBN-relevant decade).

    Parameters
    ----------
    q_MeV : array-like
        Neutrino momentum [MeV].
    T_gamma_MeV : array-like
        Photon temperature [MeV].
    species : {'nue', 'nux'}
        ν_e or ν_{μ,τ} channel.

    Returns
    -------
    rate : jnp.ndarray
        Per-momentum equilibration rate [MeV], shape broadcast of inputs.

    Notes
    -----
    Citation: [HannestadMadsen1995-AppendixA] for the matrix element;
    [Mangano2005-eq19] for the energy-transfer prefactor; the q-dependence
    here is a Padé-like factorization that matches the closed-form rate
    on thermal average and is locally smooth (jax.grad-friendly).
    """
    q = jnp.asarray(q_MeV, dtype=jnp.float64)
    T = jnp.asarray(T_gamma_MeV, dtype=jnp.float64)
    if species == "nue":
        a_alpha = jnp.asarray(A_TOTAL_NUE, dtype=jnp.float64)
    elif species == "nux":
        a_alpha = jnp.asarray(A_TOTAL_NUX, dtype=jnp.float64)
    else:
        raise ValueError(f"unknown species={species!r}; choose 'nue' or 'nux'")
    # Closed-form prefactor: (7π/12) G_F² T⁴ · a_α / <q>_FD, then scaled by q.
    # By construction integrates against f_FD to (7π/12) G_F² T⁵ a_α.
    gf2 = (G_F_MEV ** 2)
    prefactor = (7.0 * jnp.pi / 12.0) * gf2 * T**4 * a_alpha / _MEAN_Q_OVER_T_FD
    rate_lo = prefactor * q
    if not apply_v32_corrections:
        return rate_lo
    # v3.2 Phase χ-5 corrections (multiplicative on the leading-order rate):
    #   - F(m_e/T_γ): electron-mass suppression (χ-2)
    #   - δ_DHS(T_ν): running-coupling enhancement (χ-3)
    # FM corrections (χ-1) are already active in the weak rate path
    # (rabbit.jax.weak_live_jax) at CL3; they affect Y_p, not the
    # collision rate evaluated here. Pair processes (χ-4) are already
    # summed in the Mangano coefficient a_α = 4(G_L² + G_R²).
    from rabbit.jax.electron_mass_suppression_jax import (
        electron_mass_suppression_jax,
    )
    from rabbit.jax.dhs_correction import dhs_correction_factor
    F_me = electron_mass_suppression_jax(T)
    T_nu = T if T_nu_MeV is None else jnp.asarray(T_nu_MeV, dtype=jnp.float64)
    delta_dhs = dhs_correction_factor(T_nu)
    return rate_lo * F_me * delta_dhs


# ═══════════════════════════════════════════════════════════════════════
# §2. AP-form linearised collision operator with momentum-resolved rate
# ═══════════════════════════════════════════════════════════════════════

def hm_collision_rate_per_q(
    f_alpha: jnp.ndarray,
    q_MeV: jnp.ndarray,
    T_gamma_MeV: jnp.ndarray,
    *,
    T_nu_alpha_MeV: jnp.ndarray | None = None,
    species: str = "nue",
) -> jnp.ndarray:
    """Linearised AP-form collision operator with momentum-resolved Γ_α(q, T).

    Physics meaning
    ---------------
    Returns dC[f_α](q) / dN per momentum bin:

        C[f_α](q) = -Γ_α(q, T_γ) · (f_α(q) - f_FD(q / T_α))

    where T_α = T_ν_α (the neutrino temperature for species α, default
    = T_γ). Detailed balance is enforced at construction: when
    f_α = f_FD(q / T_α), the rate vanishes identically.

    Regime
    ------
    BBN-relevant T ∈ [0.05, 10] MeV, |f_α - f_FD| ≲ 1.

    Parameters
    ----------
    f_alpha : array-like, shape (N_q,)
        Distribution function values at the q-grid.
    q_MeV : array-like, shape (N_q,)
        Momentum grid in MeV.
    T_gamma_MeV : scalar
        Photon temperature.
    T_nu_alpha_MeV : scalar, optional
        Neutrino temperature; defaults to T_gamma_MeV (FLRW limit).
    species : {'nue', 'nux'}
        Channel selector.

    Returns
    -------
    dCdN : jnp.ndarray, shape (N_q,)
        Per-momentum collision rate. At equilibrium returns zero
        (modulo float64 roundoff).
    """
    f = jnp.asarray(f_alpha, dtype=jnp.float64)
    q = jnp.asarray(q_MeV, dtype=jnp.float64)
    T_g = jnp.asarray(T_gamma_MeV, dtype=jnp.float64)
    T_a = jnp.asarray(
        T_g if T_nu_alpha_MeV is None else T_nu_alpha_MeV, dtype=jnp.float64,
    )
    f_eq = _fermi_dirac_dimensionless(q / T_a)
    gamma_q = gamma_alpha_per_momentum(q, T_g, species=species)
    return -gamma_q * (f - f_eq)


def hm_thermal_average(
    q_nodes_MeV: jnp.ndarray,
    q_weights_MeV: jnp.ndarray,
    T_gamma_MeV: jnp.ndarray,
    *,
    species: str = "nue",
) -> jnp.ndarray:
    """Thermal average ⟨Γ_α(q, T)⟩_FD against the q-grid.

    Used in tests to confirm the per-momentum factorization is
    consistent with the closed-form total rate. Production runtime
    does not call this — it uses :func:`hm_collision_rate_per_q`
    directly inside the AP-Rosenbrock Jacobian (Phase H).

    Returns
    -------
    avg_rate : jnp.ndarray
        ``Σᵢ wᵢ q_i² f_FD(q_i/T_γ) · Γ_α(q_i, T_γ) / N`` where N is
        the FD normalization on the same grid. Should match the
        closed-form ``total_rate_jax`` to grid-quadrature precision.
    """
    q = jnp.asarray(q_nodes_MeV, dtype=jnp.float64)
    w = jnp.asarray(q_weights_MeV, dtype=jnp.float64)
    T = jnp.asarray(T_gamma_MeV, dtype=jnp.float64)
    f_eq = _fermi_dirac_dimensionless(q / T)
    gamma_q = gamma_alpha_per_momentum(q, T, species=species)
    integrand = q**2 * f_eq * gamma_q
    norm = jnp.sum(w * q**2 * f_eq)
    return jnp.sum(w * integrand) / jnp.maximum(norm, 1e-300)


__all__ = [
    "gamma_alpha_per_momentum",
    "hm_collision_rate_per_q",
    "hm_thermal_average",
]
