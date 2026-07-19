"""rabbit.jax.hm_matrix_elements_jax — Hannestad-Madsen 1995 |M|² closed forms.

v3.1 Phase α-1 (Plan §α-1). Particle-physics layer of the full HM
collision kernel: closed-form, anisotropy-independent matrix elements
for the four BBN-relevant 2→2 weak processes.

Layering (Plan §0.1)
--------------------
This module ships the **particle-physics layer** only:
``|M|²(s, t, u, m_e)`` as Lorentz-invariant closed forms. The
**cosmological-application layer** (partner integration measure,
Bianchi-anisotropic distributions) is in
:mod:`rabbit.jax.collision_hm_full_jax` (Phase α-2).

Why anisotropy-independent
--------------------------
``|M|²`` describes the local particle-physics interaction in the
collision-rest frame; cross sections are Lorentz invariants. Switching
from FLRW to Bianchi I/II/… does not change the matrix element. Only
the *application* (which partner distribution we integrate against)
becomes anisotropic. See ``docs/audit/v3_1_anisotropy_audit.md`` for
the per-source determination.

Process catalog
---------------
- ``M2_nu_e_elastic``   : ν_α + e^- → ν_α + e^- (CC + NC for ν_e; NC for ν_x)
- ``M2_nu_nubar_to_ee`` : ν_α + ν̄_α → e^+ + e^- (annihilation)
- ``M2_nu_nu_diagonal`` : ν_α + ν_α → ν_α + ν_α (Fierz-allowed identical species)
- ``M2_nu_nu_off_diag`` : ν_α + ν_β → ν_α + ν_β  (α ≠ β; Z-exchange only)

Normalization convention
------------------------
Spin-summed, color-summed |M|² in the Standard-Model V−A theory.
Factor conventions match Mangano 2005 eq 8 and Hannestad-Madsen 1995
Appendix A; sign verification cross-references PRIMAT Table 4.2 (the
``test_thermal_average_matches_mangano_a_alpha`` gate locks the
absolute normalization to 1 % within the closed-form Mangano coefficient
``a_α = 4(G_L² + G_R²)``).

References
----------
- Hannestad, Madsen, *Phys. Rev. D* 52 1764 (1995), Appendix A.
- Mangano et al., *Nucl. Phys. B* 729 221 (2005), eq 8.
- Notzold, Raffelt, *Nucl. Phys. B* 307 924 (1988).
- Pitrou et al. PRIMAT, *Phys. Rep.* 754 1 (2018), Table 4.2 (sign check).

Hallucination guards (Plan §0.1 + §0.2)
---------------------------------------
- Non-negativity asserted on a BBN grid (any negative |M|² indicates
  sign-convention error and fails CI).
- Crossing symmetry: ``M²_νe→νe(s, t, u) = M²_νν̄→ee(u, t, s)`` in URM.
- Mandelstam closure: ``s + t + u = 2 m_e²`` (URM: → 0).
- Thermal-average benchmark vs Mangano ``a_α`` coefficient (1 % budget).
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from rabbit.collisions.kernels import (
    G_F_MEV,
    G_L_NUE, G_R_NUE,
    G_L_NUX, G_R_NUX,
)


jax.config.update("jax_enable_x64", True)


# Electron mass [MeV]; PDG 2024 value to 7 significant figures.
M_E_MEV: float = 0.5109989461


# ═══════════════════════════════════════════════════════════════════════
# §1. Mandelstam ↔ (q, q', μ) conversion
# ═══════════════════════════════════════════════════════════════════════

def mandelstam_from_qq_mu_urm(
    q: jnp.ndarray, q_prime: jnp.ndarray, mu: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Mandelstam (s, t, u) from (q, q', μ) in the ultra-relativistic limit.

    Convention: q is the incoming neutrino momentum, q' is the partner
    momentum, μ = cos θ where θ is the angle between p and the partner
    3-momentum in the rest frame of the colliding pair. Both particles
    treated as massless.

    .. math::
        s = 2 q q' (1 - \\mu)
        t = -2 q q' (1 - \\cos\\theta_{\\rm out}) \\to 0  \\text{(forward limit)}
        u = -2 q q' (1 + \\mu) - t

    For BBN-relevant temperatures T ~ 1 MeV, ``q ≫ m_e`` typically holds
    away from the deeply non-relativistic regime; URM is the standard
    leading-order approximation used by Mangano 2005 and HM 1995.

    The μ here is the *initial-state* angle between the two incoming
    particles. In the elastic-scattering 2D quadrature, an additional
    final-state angle (between p and p') determines t individually.
    For the closed-form thermal-average benchmark we use the symmetric
    form below; see Plan §α-2 for the full Bianchi-aware quadrature.

    Parameters
    ----------
    q, q_prime : array-like
        4-momentum magnitudes in MeV, both ≥ 0.
    mu : array-like
        cos(θ) ∈ [-1, 1].

    Returns
    -------
    (s, t, u) : tuple of jnp.ndarray
        Mandelstam invariants in MeV².
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    qp = jnp.asarray(q_prime, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    s = 2.0 * q * qp * (1.0 - mu)
    # In the angle-averaged elastic 2→2 with both particles massless,
    # the symmetric forward kinematic gives t → 0 in the leading
    # collinear limit. The full t-distribution is restored in Phase α-2
    # by integrating over the second angle.
    t = jnp.zeros_like(s)
    u = -s   # consequence of s + t + u = 0 in massless URM
    return s, t, u


def mandelstam_from_qq_mu_with_me(
    q: jnp.ndarray, q_prime: jnp.ndarray, mu: jnp.ndarray, m_e_MeV: float = M_E_MEV,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Mandelstam (s, t, u) from (q, q', μ) including electron mass.

    Convention as :func:`mandelstam_from_qq_mu_urm` but with the partner
    treated as a massive lepton (electron with mass m_e). Closed form:

    .. math::
        E_e = \\sqrt{q'^2 + m_e^2}
        s = m_e^2 + 2 q (E_e - q' \\mu)

    The forward-limit t = 0 approximation is retained; full angular
    quadrature is in Phase α-2.

    Parameters
    ----------
    q : neutrino momentum [MeV]
    q_prime : electron momentum [MeV]
    mu : cos(θ) ∈ [-1, 1]
    m_e_MeV : electron mass [MeV]

    Returns
    -------
    (s, t, u) : Mandelstam invariants in MeV²; satisfy s + t + u = 2 m_e².
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    qp = jnp.asarray(q_prime, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    m_e2 = float(m_e_MeV) ** 2
    E_e = jnp.sqrt(qp * qp + m_e2)
    s = m_e2 + 2.0 * q * (E_e - qp * mu)
    t = jnp.zeros_like(s)
    # s + t + u = 2 m_e² (one massive particle, m_ν = 0)
    u = 2.0 * m_e2 - s - t
    return s, t, u


# ═══════════════════════════════════════════════════════════════════════
# §2. Process matrix elements
# ═══════════════════════════════════════════════════════════════════════

def M2_nu_e_elastic(
    s: jnp.ndarray, t: jnp.ndarray, u: jnp.ndarray, *,
    species: str = "nue",
    m_e_MeV: float = M_E_MEV,
) -> jnp.ndarray:
    """``|M|²`` for ν_α + e⁻ → ν_α + e⁻ in the Standard Model.

    Closed form (Mangano 2005 eq 8; Hannestad-Madsen 1995 App A):

    .. math::
        |M|^2_{\\nu e \\to \\nu e}
        = 8 G_F^2 \\left[ G_L^2 (s - m_e^2)^2 + G_R^2 (u - m_e^2)^2
        \\right]
        + 8 G_F^2 \\, G_L G_R \\, m_e^2 (s + u - 2 m_e^2)

    ``G_L`` and ``G_R`` are the chiral couplings:
    - ν_e: G_L = 1/2 + sin²θ_W (CC + NC), G_R = sin²θ_W (NC only)
    - ν_x: G_L = -1/2 + sin²θ_W (NC only), G_R = sin²θ_W

    URM limit (m_e → 0): ``|M|² = 8 G_F² (G_L² s² + G_R² u²)``.

    Thermal-average benchmark
    -------------------------
    ``⟨|M|²/(2π)³⟩_{f_e f_νe} ∝ 4(G_L² + G_R²) = a_α`` matches Mangano
    2005 ``a_e = 1 + 4 sin²θ_W + 8 sin⁴θ_W ≈ 2.353`` for ν_e and
    ``a_x = 1 - 4 sin²θ_W + 8 sin⁴θ_W ≈ 0.503`` for ν_x.

    Parameters
    ----------
    s, t, u : Mandelstam invariants [MeV²]. Caller supplies via
        :func:`mandelstam_from_qq_mu_with_me`.
    species : {'nue', 'nux'}.
    m_e_MeV : electron mass [MeV].

    Returns
    -------
    |M|² in dimensionless units of ``G_F² · MeV⁴``; multiply by
    ``G_F²·MeV⁴`` to recover physical squared amplitude.
    """
    if species == "nue":
        gL = jnp.asarray(G_L_NUE, dtype=jnp.float64)
        gR = jnp.asarray(G_R_NUE, dtype=jnp.float64)
    elif species == "nux":
        gL = jnp.asarray(G_L_NUX, dtype=jnp.float64)
        gR = jnp.asarray(G_R_NUX, dtype=jnp.float64)
    else:
        raise ValueError(f"unknown species {species!r}; choose 'nue' or 'nux'")
    s = jnp.asarray(s, dtype=jnp.float64)
    t = jnp.asarray(t, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    gf2 = float(G_F_MEV) ** 2
    m_e2 = float(m_e_MeV) ** 2
    # Pure squared-coupling piece (vanishes at m_e=0 only via s,u)
    sq = gL * gL * (s - m_e2) ** 2 + gR * gR * (u - m_e2) ** 2
    # Mass-mixing piece: G_L G_R m_e² (s + u - 2 m_e²); this is the
    # standard interference term in V-A theory (Notzold-Raffelt 1988).
    mix = gL * gR * m_e2 * (s + u - 2.0 * m_e2)
    return 8.0 * gf2 * (sq + mix)


def M2_nu_nubar_to_ee(
    s: jnp.ndarray, t: jnp.ndarray, u: jnp.ndarray, *,
    species: str = "nue",
    m_e_MeV: float = M_E_MEV,
) -> jnp.ndarray:
    """``|M|²`` for ν_α + ν̄_α → e⁺ + e⁻ (annihilation).

    Crossing symmetry: ``|M|²_{νν̄→ee}(s, t, u) = |M|²_{νe→νe}(u, t, s)``
    by interchange of the incoming antifermion and outgoing fermion.
    In URM this is exact; with m_e ≠ 0 the closed form follows by the
    same crossing applied to the squared-coupling and mass-mixing pieces.

    Closed form:
    .. math::
        |M|^2_{\\nu \\bar\\nu \\to ee}
        = 8 G_F^2 \\left[ G_L^2 (u - m_e^2)^2 + G_R^2 (s - m_e^2)^2
        \\right]
        + 8 G_F^2 \\, G_L G_R \\, m_e^2 (s + u - 2 m_e^2)

    Parameters and return: same conventions as :func:`M2_nu_e_elastic`.
    """
    # Crossing s ↔ u in the squared-coupling piece; mixing piece is
    # symmetric in (s + u) so it carries through unchanged.
    return M2_nu_e_elastic(u, t, s, species=species, m_e_MeV=m_e_MeV)


def M2_nu_nu_diagonal(
    s: jnp.ndarray, t: jnp.ndarray, u: jnp.ndarray,
) -> jnp.ndarray:
    """``|M|²`` for ν_α + ν_α → ν_α + ν_α (identical-species diagonal).

    Z-exchange only, with identical-particle symmetrization yielding
    a factor of 2 vs the distinguishable case:

    .. math::
        |M|^2_{\\nu_\\alpha \\nu_\\alpha} = 64 G_F^2 (s^2 + u^2)

    Massless throughout (no m_e); s + t + u = 0 in URM. Citation:
    Hannestad-Madsen 1995 App A; Mangano 2005.
    """
    s = jnp.asarray(s, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    gf2 = float(G_F_MEV) ** 2
    return 64.0 * gf2 * (s * s + u * u)


def M2_nu_nu_off_diagonal(
    s: jnp.ndarray, t: jnp.ndarray, u: jnp.ndarray,
) -> jnp.ndarray:
    """``|M|²`` for ν_α + ν_β → ν_α + ν_β with α ≠ β.

    Distinguishable species: no symmetrization factor of 2.

    .. math::
        |M|^2_{\\nu_\\alpha \\nu_\\beta} = 32 G_F^2 (s^2 + u^2)

    Massless throughout (URM).
    """
    s = jnp.asarray(s, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    gf2 = float(G_F_MEV) ** 2
    return 32.0 * gf2 * (s * s + u * u)


# ═══════════════════════════════════════════════════════════════════════
# §3. Sanity check: closed-form a_α coefficient
# ═══════════════════════════════════════════════════════════════════════

def closed_form_a_alpha(species: str = "nue") -> float:
    """Mangano 2005 ``a_α`` coefficient as ``4(G_L² + G_R²)``.

    Useful for the thermal-average benchmark test: the leading-order
    rate is ``Γ_α = (7π/12) G_F² T⁵ · a_α`` and our matrix-element
    normalization should reproduce ``a_α`` after thermal averaging.
    """
    if species == "nue":
        return 4.0 * (G_L_NUE ** 2 + G_R_NUE ** 2)
    if species == "nux":
        return 4.0 * (G_L_NUX ** 2 + G_R_NUX ** 2)
    raise ValueError(f"unknown species {species!r}")


__all__ = [
    "M_E_MEV",
    "mandelstam_from_qq_mu_urm",
    "mandelstam_from_qq_mu_with_me",
    "M2_nu_e_elastic",
    "M2_nu_nubar_to_ee",
    "M2_nu_nu_diagonal",
    "M2_nu_nu_off_diagonal",
    "closed_form_a_alpha",
]
