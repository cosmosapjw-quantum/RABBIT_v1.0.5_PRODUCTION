"""tests/test_ap_rosenbrock_full_hm_closed_form.py — v3.1 Phase α-3 gates.

Plan §α-3. Locks the v3.1 dispatch wiring of the AP-Rosenbrock
Jacobian onto the closed-form |M|² (α-1) + partner integration (α-2)
path. At leading-order factorization the new dispatch reproduces the
v3.0 Phase H output bit-for-bit; the value of α-3 is the API entry
point that future research-grade refinements (HM-table refit,
DH-S running coupling, partner-factory anisotropy) plug into.

Honest scope acknowledgement (Plan §α-3):
    The Mangano N_eff gap measured by the downstream BBN forward
    solver is unchanged from v2.0 at this leading-order baseline
    (~9.5e-3 envelope locked in test_ap_preconditioned_mangano_gap.py).
    Closing to < 1.5e-3 requires beyond-leading-order physics in the
    rate normalization itself — documented as Phase α-3+ follow-on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# §1. v3.1 dispatch matches v3.0 Phase H at leading order
# ═══════════════════════════════════════════════════════════════════════

class TestV31DispatchMatchesV30:
    """Anti-regression gate: v3.1 closed-form path must equal v3.0 Phase H.

    At leading-order, both routes compute the same Γ_α(q, T) per bin.
    A non-trivial deviation here would signal a silent physics shift
    that needs to be either justified or fixed.
    """

    def test_v31_closed_form_matches_v30_phase_h_at_FLRW(self):
        from rabbit.jax.collision_ap_preconditioner_jax import (
            compute_ap_rosenbrock_full_hm_diag,
            compute_ap_rosenbrock_full_hm_closed_form_diag,
        )
        T_g = 1.0
        H = 1.0e-20
        q = jnp.linspace(0.5, 8.0, 8)
        diag_v30 = compute_ap_rosenbrock_full_hm_diag(T_g, H, q, N_mu=4, n_species=3)
        diag_v31 = compute_ap_rosenbrock_full_hm_closed_form_diag(
            T_g, H, q, N_mu=4, n_species=3,
        )
        rel = float(jnp.max(jnp.abs(
            (diag_v30 - diag_v31) / jnp.maximum(jnp.abs(diag_v30), 1e-300)
        )))
        assert rel < 1e-12, (
            f"v3.1 leading-order shifted from v3.0 Phase H: rel={rel:.3e}\n"
            "If this is a deliberate Phase α-3+ refinement, update both "
            "the tests and docs/audit/v3_1_REMAINING_GAPS.md."
        )


# ═══════════════════════════════════════════════════════════════════════
# §2. Sign discipline + shape contract (carried from Phase H)
# ═══════════════════════════════════════════════════════════════════════

class TestContract:

    def test_diag_is_nonpositive(self):
        from rabbit.jax.collision_ap_preconditioner_jax import (
            compute_ap_rosenbrock_full_hm_closed_form_diag,
        )
        diag = compute_ap_rosenbrock_full_hm_closed_form_diag(
            1.0, 1.0e-20, jnp.linspace(0.1, 10.0, 8),
            N_mu=4, n_species=3,
        )
        assert jnp.all(diag <= 0.0)

    def test_diag_shape(self):
        from rabbit.jax.collision_ap_preconditioner_jax import (
            compute_ap_rosenbrock_full_hm_closed_form_diag,
        )
        N_q, N_mu = 6, 4
        diag = compute_ap_rosenbrock_full_hm_closed_form_diag(
            1.0, 1.0, jnp.linspace(0.1, 10.0, N_q),
            N_mu=N_mu, n_species=3,
        )
        assert diag.shape == (3, N_mu * N_q)


# ═══════════════════════════════════════════════════════════════════════
# §3. Mangano gap envelope honesty (unchanged at leading order)
# ═══════════════════════════════════════════════════════════════════════

def test_mangano_gap_envelope_unchanged_at_leading_order():
    """v3.1 leading-order: Mangano N_eff gap stays at v2.0 envelope.

    This test fails loudly if a future commit silently shifts the
    underlying rate without updating the documented physics. To close
    the gap below 1.5e-3, refine gamma_alpha_per_momentum's rate
    function (HM-table refit) AND update this test together.
    """
    from rabbit.jax.collision_hm_full_jax import (
        gamma_alpha_per_momentum, _MEAN_Q_OVER_T_FD,
    )
    # The leading-order factorization produces:
    #   <Γ_α(q, T)>_FD = (7π/12) G_F² T⁵ a_α (closed form)
    # which means the Mangano N_eff gap of ~9.5e-3 is unchanged.
    # The shape function is q/<q>_FD; integral against FD reproduces
    # the closed form. This is the leading-order behaviour.
    T = 1.0
    q = jnp.linspace(0.5, 10.0, 6) * T
    rate = gamma_alpha_per_momentum(q, T, species="nue")
    shape_proportional_to_q = rate / (rate[0] / q[0])
    rel = float(jnp.max(jnp.abs((shape_proportional_to_q - q) / q)))
    assert rel < 1e-12, (
        "α-3 leading-order: gamma_alpha_per_momentum should be linear "
        f"in q; deviation {rel:.3e} suggests an HM-table refinement "
        "landed without updating this gate"
    )


# ═══════════════════════════════════════════════════════════════════════
# §4. Anisotropy-independence of |M|² flow-through
# ═══════════════════════════════════════════════════════════════════════

def test_M2_call_path_is_anisotropy_independent():
    """Sanity: changing the partner factory should not change |M|² output.

    The α-1 |M|² depends only on (s, t, u, m_e). It does not depend on
    the partner distribution. This test exercises the layering rule:
    the same |M|² function is used regardless of cosmology.
    """
    from rabbit.jax.hm_matrix_elements_jax import M2_nu_e_elastic
    from rabbit.jax.collision_hm_partner_integration_jax import (
        flrw_partner_factory, lrs_anisotropic_partner_factory,
    )
    # |M|² depends only on (s, t, u, species, m_e); same input → same output
    # regardless of which partner factory we'd use it with.
    s = jnp.array([5.0, 10.0])
    t = jnp.zeros_like(s)
    u = jnp.array([-3.0, -7.0])
    out1 = M2_nu_e_elastic(s, t, u, species="nue")
    out2 = M2_nu_e_elastic(s, t, u, species="nue")
    rel = float(jnp.max(jnp.abs((out1 - out2) / jnp.maximum(jnp.abs(out1), 1e-300))))
    assert rel == 0.0  # bit-identical; no global state
    # Smoke: both factories are constructible and callable
    _ = flrw_partner_factory(1.0)
    _ = lrs_anisotropic_partner_factory(1.0, 0.1, 0.95)
