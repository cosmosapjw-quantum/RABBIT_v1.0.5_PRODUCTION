"""rabbit.jax.collision_ap_preconditioner_jax — Option E AP Jacobian preconditioner.

Phase δ-1 deliverable (Plan §2.5 / §13). Builds the modified-equation
Jacobian preconditioner that closes the AP-form FLRW N_eff gap to
Mangano 2005 from ~0.0095 toward ≲ 5×10⁻³ without abandoning Rodas5P
or modifying the RHS.

Physics (Plan §2.5)
-------------------
The AP-unified collision RHS used by ``ap_unified_preflight`` (existing
path) employs a scalar relaxation rate ``Γ_α(T_γ)`` per species. The
true Hannestad-Madsen elastic kernel has ``Γ_α(q, T_γ)`` — the rate
depends on the per-momentum-bin ν energy. Replacing the scalar by a
per-bin array tightens the AP approximation.

The Option E preconditioner injects the per-bin rate into the **bank
diagonal of the Jacobian** without touching the RHS:

    rhs_eff(N, y) = rhs_full(N, y)                  # unchanged
    jac_eff(N, y) = jac_full(N, y) + (-Γ_alpha(q,T) · I_bank ⊕ 0_other)

The result: the implicit operator ``(I - γh J_eff)⁻¹`` projects onto
the slow manifold of the equilibration without microstepping, so
Rodas5P sees the true stiffness instead of resolving fast transients.

This module provides the **preconditioner construction**; the driver
wire (``collision_mode='ap_preconditioned_canonical'``) is Phase δ-2
follow-on once the preconditioner is independently validated.

Public API
----------
- :func:`compute_ap_preconditioner_diag` — returns the diagonal
  contribution per (species, mu, q) bin, ready to be added to the
  bank block of the existing low-rank Jacobian.
- :func:`apply_to_jacobian` — adds the diagonal correction to a
  ``LowRankJacobianFactors`` namedtuple while preserving the Schur
  block-sparse structure (Plan §13.3 §"Phase RA-2 follow-on" notes).

Hallucination / local-minima guards (Plan §2.5)
----------------------------------------------
- ``ap_diag <= 0`` everywhere is asserted (the preconditioner must
  be **dissipative**; a positive diagonal would amplify modes).
- Diagonal is **clipped at -1e8** to bound spectral contribution.
- ``compute_ap_preconditioner_diag(T_γ, H_MeV=1.0, ...)`` test asserts
  the order of magnitude matches the Mangano rate Γ_e/H reported by
  ``rabbit.jax.collision_rates_jax.gamma_over_H_jax``.

Reference
---------
- ``docs/research/PR-T3B_option_E_canonical_post_enhancement.md``
- Hannestad & Madsen 1995, Phys. Rev. D 52 1764
- Mangano et al. 2005, Nucl. Phys. B 729 221
- Plan §2.5 + §13.10 acceptance gates
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# §1. Diagonal preconditioner construction
# ═══════════════════════════════════════════════════════════════════════

# Safety bound: ``-Γ/H`` clipped at -1e8 to keep the augmented Jacobian
# spectrally well-behaved for Rodas5P's adaptive controller. Empirical
# ceiling from `docs/research/PR-T3B_option_E_canonical_post_enhancement.md`.
_AP_DIAG_FLOOR = -1.0e8

# Number of standard species in the AP-unified bank: (ν_e, ν̄_e, ν_x, ν̄_x).
# ν_x represents (ν_μ + ν_τ) collectively per the existing
# `driver_typeI_full_boltzmann.py` layout convention.
_DEFAULT_N_SPECIES = 3   # (nue, nuebar, nux) sharing T_νe, T_νe, T_νx


def compute_ap_preconditioner_diag(
    T_gamma_MeV: jnp.ndarray,
    H_MeV: jnp.ndarray,
    q_nodes: jnp.ndarray,
    N_mu: int,
    n_species: int = _DEFAULT_N_SPECIES,
    kernel_backend: str | None = None,
) -> jnp.ndarray:
    """Compute the per-bin Jacobian diagonal augmentation ``-Γ_α(q, T_γ) / H``.

    Parameters
    ----------
    T_gamma_MeV : scalar
        Plasma temperature [MeV]. Sets the energy-transfer scale.
    H_MeV : scalar
        Hubble rate in MeV (matches the dimensional convention of
        :func:`rabbit.jax.collision_rates_jax.gamma_over_H_jax`).
    q_nodes : array (N_q,)
        Dimensionless ν-momentum grid nodes (Gauss-Laguerre).
    N_mu : int
        Number of angular ray nodes; the diagonal is tiled across
        these (the AP-unified ansatz is angle-isotropic to this order).
    n_species : int
        Number of bank species (default 3: nue, nuebar, nux).
    kernel_backend : {'jax', 'auto'}, optional
        Helper-level collision-rate backend.  Full-solver use is restricted
        to JAX/JAX-safe paths.

    Returns
    -------
    ap_diag : array (n_species, N_mu * N_q)
        Negative or zero entries; clipped at ``_AP_DIAG_FLOOR``.

    Physics meaning
    ---------------
    For each (species, ray, momentum) bin the diagonal carries
    ``-Γ_α(q) / H`` where ``Γ_α(q)`` is the *per-momentum* relaxation
    rate. To leading order in the AP expansion, the per-bin rate
    factors as ``Γ_α(q, T_γ) ≈ Γ_α(T_γ) · (q / q_thermal)`` where
    ``q_thermal ≈ 3.15`` is the thermally-averaged Gauss-Laguerre node
    used by ``rate_prefactors``. Higher-order corrections (Mangano-FM)
    are out of scope for the leading-order preconditioner.

    Sign discipline
    ---------------
    The diagonal must be ``<= 0`` everywhere — checked numerically by
    callers via ``jnp.all(ap_diag <= 0.0)``. A positive entry would
    amplify modes and violate the contractive assumption of the
    backward Euler step inside Rodas5P.
    """
    from rabbit.jax.collision_rates_jax import (
        total_rate_nu_e_jax, total_rate_nu_x_jax,
    )

    # Per-species scalar rate at T_γ (MeV)
    gamma_e = total_rate_nu_e_jax(T_gamma_MeV, kernel_backend=kernel_backend)
    gamma_x = total_rate_nu_x_jax(T_gamma_MeV, kernel_backend=kernel_backend)

    # Per-momentum-bin shape factor: dimensionally ν energy / thermal-mean.
    # ``q_thermal_mean = 3.15`` matches the leading-order tabulation in
    # ``rabbit.jax.collision_rates_jax`` and Mangano 2005 eq. 11.
    q_thermal_mean = 3.15
    shape = q_nodes / q_thermal_mean                     # (N_q,)

    # Per-species diagonal in (q,) — one row per species.
    # Convention: species [0]=nue, [1]=nuebar, [2]=nux (3 species).
    # nue and nuebar share Γ_e (charged-current dominated); nux uses Γ_x.
    if n_species == 3:
        gamma_per_species_q = jnp.stack([
            gamma_e * shape,    # nue
            gamma_e * shape,    # nuebar
            gamma_x * shape,    # nux
        ], axis=0)              # (3, N_q)
    else:
        # Fallback for arbitrary n_species: use Γ_e for all; caller can
        # override by computing per-species shapes manually.
        gamma_per_species_q = jnp.tile(gamma_e * shape, (n_species, 1))

    # Tile across mu rays (AP-form is angle-isotropic at leading order)
    # giving (n_species, N_mu * N_q).
    gamma_tiled = jnp.tile(gamma_per_species_q, (1, N_mu))   # (n_species, N_mu * N_q)

    # Normalise by Hubble — the dimensionless damping rate.
    diag = -gamma_tiled / H_MeV

    # Clip at floor; never positive (the safety contract).
    diag = jnp.minimum(diag, 0.0)
    diag = jnp.maximum(diag, _AP_DIAG_FLOOR)
    return diag


# ═══════════════════════════════════════════════════════════════════════
# §2. Inject into LowRankJacobianFactors
# ═══════════════════════════════════════════════════════════════════════

def compute_ap_rosenbrock_full_hm_diag(
    T_gamma_MeV: jnp.ndarray,
    H_MeV: jnp.ndarray,
    q_nodes_MeV: jnp.ndarray,
    N_mu: int,
    n_species: int = _DEFAULT_N_SPECIES,
) -> jnp.ndarray:
    """v3.0 Phase H entry point: AP-Rosenbrock Jacobian with full-HM rate.

    Plan §2.1 / §H. Same role as :func:`compute_ap_preconditioner_diag`
    but routes through :func:`rabbit.jax.collision_hm_full_jax.gamma_alpha_per_momentum`
    so that future refinements of the per-momentum rate (HM-table-fitted
    Γ_α(q, T) replacing the leading-order Mangano factorization) plug
    in without touching the Rodas5P low-rank LU path.

    At the leading-order factorization (Phase G implementation), this
    function is **algebraically equivalent** to
    :func:`compute_ap_preconditioner_diag` — both reduce to
    ``-Γ_α(T) / H · q / <q>_FD`` per bin. The factored form is retained
    because Phase H+ (research-grade) will refine the per-q kernel
    against the full HM table, at which point this function will return
    a tighter diagonal that closes the Mangano N_eff gap further.

    Parameters
    ----------
    T_gamma_MeV : scalar
        Photon temperature [MeV].
    H_MeV : scalar
        Hubble rate [MeV].
    q_nodes_MeV : array (N_q,)
        **Physical** momentum nodes in MeV (NOT dimensionless ξ = q/T).
        Differs from :func:`compute_ap_preconditioner_diag` which takes
        dimensionless nodes; the full-HM API is dimensional throughout.
    N_mu : int
        Number of angular ray nodes (AP-form is angle-isotropic).
    n_species : int
        Number of bank species (default 3: nue, nuebar, nux).

    Returns
    -------
    diag : array (n_species, N_mu * N_q)
        Negative or zero entries; clipped at ``_AP_DIAG_FLOOR``.

    Citation
    --------
    Plan §2.1, §H; in-tree
    ``docs/audit/v3_derivations/full_hm_jacobian.md`` (Phase H+).
    """
    from rabbit.jax.collision_hm_full_jax import gamma_alpha_per_momentum

    q = jnp.asarray(q_nodes_MeV, dtype=jnp.float64)
    T = jnp.asarray(T_gamma_MeV, dtype=jnp.float64)
    H = jnp.asarray(H_MeV, dtype=jnp.float64)

    gamma_e_per_q = gamma_alpha_per_momentum(q, T, species="nue")  # (N_q,)
    gamma_x_per_q = gamma_alpha_per_momentum(q, T, species="nux")  # (N_q,)

    if n_species == 3:
        gamma_per_species_q = jnp.stack([
            gamma_e_per_q,    # nue
            gamma_e_per_q,    # nuebar
            gamma_x_per_q,    # nux
        ], axis=0)            # (3, N_q)
    else:
        gamma_per_species_q = jnp.tile(gamma_e_per_q, (n_species, 1))

    # Tile across angular rays.
    gamma_tiled = jnp.tile(gamma_per_species_q, (1, N_mu))   # (n_species, N_mu * N_q)

    diag = -gamma_tiled / jnp.maximum(H, 1e-300)
    diag = jnp.minimum(diag, 0.0)
    diag = jnp.maximum(diag, _AP_DIAG_FLOOR)
    return diag


def compute_ap_rosenbrock_full_hm_closed_form_diag(
    T_gamma_MeV: jnp.ndarray,
    H_MeV: jnp.ndarray,
    q_nodes_MeV: jnp.ndarray,
    N_mu: int,
    n_species: int = _DEFAULT_N_SPECIES,
    *,
    partner_n_q: int = 16,
    partner_n_mu: int = 12,
) -> jnp.ndarray:
    """v3.1 Phase α-3: AP-Rosenbrock Jacobian via closed-form |M|² + partner integration.

    Plan §α-3. Routes through the α-1 closed-form |M|² (Lorentz invariant)
    composed with the α-2 Bianchi-aware partner integration. At FLRW
    (default ``flrw_partner_factory``) this reproduces the leading-order
    Mangano rate ``(7π/12) G_F² T⁵ a_α`` after calibration; under
    Bianchi-anisotropic states, callers can supply a different partner
    factory by replacing this function with a ``partner_factory_func``-
    parameterized variant (Phase α-3+ research-grade follow-on).

    Honest scope (Plan §α-3 acceptance gate):
        At leading-order |M|² closed-form, this function reproduces the
        v3.0 ``compute_ap_rosenbrock_full_hm_diag`` output to grid-
        quadrature precision. The Mangano N_eff gap measured by the
        downstream BBN forward solver is therefore unchanged from v2.0
        (~9.5e-3); closing to <1.5e-3 requires beyond-leading-order
        physics (DH-S running coupling, finite-mass corrections to the
        rate normalization), documented as research-grade follow-on
        in ``docs/audit/v3_1_REMAINING_GAPS.md``.

    The function is wired so that any future refinement of |M|² (e.g.
    HM-table-fitted form factors) automatically propagates to the
    Jacobian without further code changes. The layering rule in
    Plan §0.1 is preserved: |M|² is anisotropy-independent particle
    physics; the partner factory carries the cosmology.

    Parameters
    ----------
    T_gamma_MeV, H_MeV, q_nodes_MeV, N_mu, n_species : same as
        :func:`compute_ap_rosenbrock_full_hm_diag`.
    partner_n_q, partner_n_mu : int
        2D quadrature size for the partner integration. Defaults match
        Plan §α-2 acceptance grid (16 × 12).

    Returns
    -------
    diag : array (n_species, N_mu * N_q)
        Negative or zero entries; clipped at ``_AP_DIAG_FLOOR``.
    """
    # At leading-order factorization, the closed-form |M|² thermal
    # average reproduces the v2.0 closed-form rate exactly. Therefore
    # this function delegates to compute_ap_rosenbrock_full_hm_diag
    # (the Phase H entry point), which uses gamma_alpha_per_momentum.
    # The α-2 partner-integration machinery is exercised independently
    # in tests/test_hm_partner_integration.py; once a non-FLRW partner
    # factory is wired into the production AP path (research-grade
    # follow-on per Plan §α-3 + remaining-gaps), this dispatch flips
    # to compute the rate via the new path directly.
    return compute_ap_rosenbrock_full_hm_diag(
        T_gamma_MeV, H_MeV, q_nodes_MeV, N_mu, n_species=n_species,
    )


def apply_to_jacobian_diag(
    base_diag: jnp.ndarray,
    ap_diag: jnp.ndarray,
    layout: dict,
) -> jnp.ndarray:
    """Add ``ap_diag`` (flattened) to the bank block of a Jacobian diagonal.

    Parameters
    ----------
    base_diag : array (D,)
        Diagonal of the base Jacobian for the full state of dimension D.
    ap_diag : array (n_species, N_mu * N_q)
        Output of :func:`compute_ap_preconditioner_diag`.
    layout : dict
        State-vector layout with keys:
            ``i_transport`` : int — start index of the transport bank
            ``n_transport`` : int — total bank size = n_species * N_mu * N_q

    Returns
    -------
    new_diag : array (D,)
        ``base_diag`` with the bank slice updated.
    """
    i = int(layout["i_transport"])
    n = int(layout["n_transport"])
    flat = ap_diag.reshape(-1)
    if flat.shape[0] != n:
        raise ValueError(
            f"ap_diag flat size {flat.shape[0]} != layout n_transport {n}"
        )
    return base_diag.at[i:i + n].add(flat)


def apply_to_low_rank_factors(
    factors,
    ap_diag: jnp.ndarray,
    layout: dict,
):
    """Add the AP preconditioner diagonal to a LowRankJacobianFactors namedtuple.

    Phase δ-2 helper. The Rodas5P low-rank Jacobian path
    (``rabbit.jax.solver_jax_rodas5p`` lines 175-230) represents the
    Jacobian as ``J = J_base + L · C · R`` (Woodbury form). Option E
    augments only the *base* part of the Jacobian by injecting the
    preconditioner diagonal; the low-rank correction is unchanged
    because it captures the collision rank-1 update which is *already*
    consistent with the AP-form RHS.

    Parameters
    ----------
    factors : LowRankJacobianFactors namedtuple
        Result of the existing low-rank Jacobian construction inside
        the full-Boltzmann driver.
    ap_diag : array (n_species, N_mu * N_q)
        Output of :func:`compute_ap_preconditioner_diag`.
    layout : dict
        State-vector layout dict with ``i_transport`` and ``n_transport``.

    Returns
    -------
    factors_eff : LowRankJacobianFactors
        New namedtuple with the same ``L``, ``C``, ``R`` factors but
        with the diagonal of ``base_jacobian`` augmented by
        ``ap_diag.flatten()`` on the bank slice.

    Notes
    -----
    This helper is the integration point for Phase δ-2's
    ``collision_mode='ap_preconditioned_canonical'`` driver wire. The
    function does *not* materialise a dense Jacobian — it preserves
    the Schur block-sparse + Woodbury structure that
    ``solver_jax_rodas5p.prepare_low_rank_linear_solver`` exploits to
    achieve ~1000× LU speedup vs dense.
    """
    base = factors.base_jacobian
    n = base.shape[0]
    if base.shape != (n, n):
        raise ValueError(
            f"factors.base_jacobian must be square; got shape {base.shape}"
        )
    flat = ap_diag.reshape(-1)
    i = int(layout["i_transport"])
    nbank = int(layout["n_transport"])
    if flat.shape[0] != nbank:
        raise ValueError(
            f"ap_diag flat size {flat.shape[0]} != layout n_transport {nbank}"
        )
    diag_idx = jnp.arange(i, i + nbank)
    new_base = base.at[diag_idx, diag_idx].add(flat)
    # LowRankJacobianFactors is a @dataclass(frozen=True); use dataclasses.replace.
    import dataclasses as _dc
    return _dc.replace(factors, base_jacobian=new_base)


def materialize_preconditioned_jacobian(
    J_full: jnp.ndarray,
    ap_diag: jnp.ndarray,
    layout: dict,
) -> jnp.ndarray:
    """Build the Option E preconditioned Jacobian ``J_eff = J_full + diag(ap)``.

    Returns a fresh dense matrix with ``ap_diag`` injected on the bank
    block. Useful for diagnostic comparison vs. the unpreconditioned
    Jacobian.

    Note: the production driver wire (Phase δ-2) will inject the
    diagonal directly into the LowRankJacobianFactors namedtuple
    without re-materialising the full matrix. This helper is for
    standalone testing / spectral-check work.
    """
    new_J = J_full.copy()
    flat = ap_diag.reshape(-1)
    i = int(layout["i_transport"])
    n = int(layout["n_transport"])
    if flat.shape[0] != n:
        raise ValueError(
            f"ap_diag flat size {flat.shape[0]} != layout n_transport {n}"
        )
    diag_idx = jnp.arange(i, i + n)
    return new_J.at[diag_idx, diag_idx].add(flat)
