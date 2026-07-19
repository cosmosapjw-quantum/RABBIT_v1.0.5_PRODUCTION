"""tests/test_ap_rosenbrock_full_hm.py — v3.0 Phase H acceptance gates.

Plan §2.1 / §H. Validates the AP-Rosenbrock Jacobian preconditioner
with the full-HM per-momentum rate.

The Phase H deliverable is the API entry point that Phase H+
(research-grade) will refine. At the leading-order factorization
(Phase G implementation), the new function is algebraically
equivalent to the existing AP preconditioner — and these tests lock
that equivalence so Phase H+ refinements are transparent shifts.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# §1. Equivalence with the leading-order AP preconditioner
# ═══════════════════════════════════════════════════════════════════════

def test_full_hm_equivalent_to_leading_order_ap():
    """At leading-order Phase G factorization, the full-HM diag equals AP.

    Both functions reduce to ``-Γ_α(T) · q / <q>_FD / H`` per bin. Any
    future refinement that breaks this equivalence (e.g. HM-table fit)
    will fail this test as a documented physics-shift signal.
    """
    from rabbit.jax.collision_ap_preconditioner_jax import (
        compute_ap_preconditioner_diag,
        compute_ap_rosenbrock_full_hm_diag,
    )
    T_g = 1.0
    H = 1.0e-20      # arbitrary; cancels in the ratio
    N_q = 6
    N_mu = 4
    # Existing AP path uses dimensionless q nodes; full-HM uses dimensional.
    # At leading order both reduce to T·xi / <q>_FD; produce nodes in both
    # conventions from a shared dimensionless grid.
    xi = jnp.linspace(0.5, 8.0, N_q)
    q_dimensional = T_g * xi
    diag_ap = compute_ap_preconditioner_diag(T_g, H, xi, N_mu, n_species=3)
    diag_hm = compute_ap_rosenbrock_full_hm_diag(T_g, H, q_dimensional, N_mu, n_species=3)
    rel = float(jnp.max(jnp.abs((diag_ap - diag_hm) / jnp.maximum(jnp.abs(diag_ap), 1e-300))))
    assert rel < 5e-3, (
        f"Phase H leading-order equivalence shifted: rel={rel:.3e}\n"
        "If this is a deliberate Phase H+ refinement, update the test "
        "with a citation to docs/audit/v3_derivations/full_hm_jacobian.md."
    )


# ═══════════════════════════════════════════════════════════════════════
# §2. Sign discipline + shape contract
# ═══════════════════════════════════════════════════════════════════════

def test_full_hm_diag_is_nonpositive():
    """All entries must be ≤ 0 (dissipative preconditioner contract)."""
    from rabbit.jax.collision_ap_preconditioner_jax import (
        compute_ap_rosenbrock_full_hm_diag,
    )
    diag = compute_ap_rosenbrock_full_hm_diag(
        1.0, 1.0e-20, jnp.linspace(0.1, 10.0, 8), N_mu=4, n_species=3,
    )
    assert jnp.all(diag <= 0.0), f"sign discipline violated: max={float(jnp.max(diag))}"


def test_full_hm_diag_shape():
    from rabbit.jax.collision_ap_preconditioner_jax import (
        compute_ap_rosenbrock_full_hm_diag,
    )
    N_q = 6
    N_mu = 4
    diag = compute_ap_rosenbrock_full_hm_diag(
        1.0, 1.0, jnp.linspace(0.1, 10.0, N_q), N_mu=N_mu, n_species=3,
    )
    assert diag.shape == (3, N_mu * N_q)


def test_full_hm_diag_clipped_at_floor():
    """When Γ/H exceeds 1e8, clipping kicks in."""
    from rabbit.jax.collision_ap_preconditioner_jax import (
        compute_ap_rosenbrock_full_hm_diag,
    )
    # Strongly stiff regime: H very small so Γ/H >> 1e8.
    diag = compute_ap_rosenbrock_full_hm_diag(
        10.0, 1.0e-50, jnp.linspace(0.1, 10.0, 8), N_mu=4, n_species=3,
    )
    assert jnp.all(diag >= -1e8), "diagonal not clipped at floor"


# ═══════════════════════════════════════════════════════════════════════
# §3. Mangano-gap envelope honesty
# ═══════════════════════════════════════════════════════════════════════
# Phase H ships the API. The numerical Mangano gap envelope is determined
# by the underlying per-q rate, which Phase G implements at leading-order
# factorization (algebraically identical to the v2 AP path). Closing the
# gap below 1.5e-3 (Plan §2.1 target) requires refining the per-q rate
# against the HM table — a research-grade follow-on. The test below locks
# this honest scope acknowledgement: if a future commit claims the gap
# has closed, it must update both this test AND the underlying physics.

def test_phase_h_documented_gap_envelope_unchanged_until_hm_refinement():
    """v3.0 Phase H lock: the leading-order AP-Rosenbrock has the same
    Mangano gap envelope as v2.0 ap_preconditioned_canonical (~9.5e-3).
    A claimed closure below 1.5e-3 must be accompanied by an HM-table-
    fitted Γ_α(q, T) refinement in collision_hm_full_jax.py.
    """
    from rabbit.jax.collision_hm_full_jax import (
        gamma_alpha_per_momentum, _MEAN_Q_OVER_T_FD,
    )
    # The leading-order factorization produces:
    #   <Γ_α(q, T)>_FD = (7π/12) G_F² T⁵ a_α
    # which is identical to the closed-form Mangano rate. Hence the
    # Mangano N_eff gap of ~9.5e-3 is unchanged at leading order.
    # The test asserts this leading-order property: closure to <1.5e-3
    # requires refining the q-shape function in gamma_alpha_per_momentum.
    T = 1.0
    q = jnp.linspace(0.5, 10.0, 6) * T
    rate = gamma_alpha_per_momentum(q, T, species="nue")
    # The shape factor is q/<q>_FD; integral against FD reproduces the
    # closed form. This is the documented leading-order behaviour.
    shape_proportional_to_q = rate / (rate[0] / q[0])
    rel = float(jnp.max(jnp.abs((shape_proportional_to_q - q) / q)))
    assert rel < 1e-12, (
        "Phase G factorization: gamma_alpha_per_momentum should be "
        f"linear in q (leading order); deviation {rel:.3e} suggests an "
        "HM-table refinement landed without updating this gate"
    )
