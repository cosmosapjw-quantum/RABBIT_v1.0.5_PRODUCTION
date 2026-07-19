"""tests/test_ap_preconditioner.py — Phase δ-1 acceptance gates.

Standalone validation of the Option E AP Jacobian preconditioner
:mod:`rabbit.jax.collision_ap_preconditioner_jax`. The driver wire-up
(``collision_mode='ap_preconditioned_canonical'``) is Phase δ-2
follow-on; this gate validates the preconditioner correctness in
isolation so the integration step has a known-good kernel.

Plan §2.5 / §13.10 acceptance:
    1. ap_diag is everywhere ≤ 0 (the dissipativity contract).
    2. Magnitude matches Γ/H from the in-tree
       :func:`rabbit.jax.collision_rates_jax.gamma_over_H_jax` at the
       same (T_γ, H) within an order of magnitude (we use the same
       scalar Γ; per-bin shape factor centers around 1).
    3. Layout-injected dense Jacobian has the correct diagonal updated
       on the bank block and identity elsewhere.
    4. Jacobian Re-eigenvalues are all ≤ 0 after preconditioning a
       sample test problem (numerical contractivity).
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from rabbit.jax.collision_ap_preconditioner_jax import (
    compute_ap_preconditioner_diag,
    apply_to_jacobian_diag,
    materialize_preconditioned_jacobian,
)


jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# §1. Diagonal correctness
# ═══════════════════════════════════════════════════════════════════════

class TestPreconditionerDiagSign:
    """Dissipativity: every entry ≤ 0."""

    @pytest.mark.parametrize("T_gamma", [0.1, 0.5, 1.0, 2.0])
    def test_diagonal_non_positive(self, T_gamma):
        N_q = 12
        N_mu = 8
        q_nodes = jnp.linspace(0.5, 8.0, N_q)
        H_MeV = 1.0e-21    # rough order BBN epoch
        diag = compute_ap_preconditioner_diag(
            T_gamma, H_MeV, q_nodes, N_mu, n_species=3,
        )
        assert jnp.all(diag <= 0.0), (
            f"T={T_gamma}: ap_diag has positive entries (max={float(jnp.max(diag))})"
        )
        assert diag.shape == (3, N_mu * N_q)


class TestPreconditionerDiagMagnitude:
    """Order of magnitude matches gamma_over_H_jax at the same (T_γ, H)."""

    def test_matches_in_tree_gamma_over_h_in_order(self):
        from rabbit.jax.collision_rates_jax import gamma_over_H_jax

        T_gamma = 1.0   # MeV
        H_MeV = 1.0e-22

        N_q = 12
        N_mu = 4
        q_nodes = jnp.linspace(0.5, 8.0, N_q)

        ap_diag = compute_ap_preconditioner_diag(
            T_gamma, H_MeV, q_nodes, N_mu, n_species=3,
        )
        # Reference: scalar Γ_e/H at this T_γ
        ref_gamma_over_H = float(gamma_over_H_jax(T_gamma, H_MeV))
        ref_value = -ref_gamma_over_H

        # Per-bin shape factor (q / q_thermal_mean) centers near 1; the
        # mean of the diagonal across all bins of nue species should
        # therefore be order-of-magnitude consistent with -Γ_e/H.
        nue_diag = ap_diag[0]  # shape (N_mu * N_q,)
        mean = float(jnp.mean(nue_diag))
        # Allow within a factor of 5 (q_thermal_mean factor + tile-mean drift)
        ratio = mean / ref_value
        assert 0.2 < ratio < 5.0, (
            f"mean(ap_diag_nue) = {mean:.3e}, expected order "
            f"{ref_value:.3e} (ratio {ratio:.3f})"
        )


class TestPreconditionerLayoutInjection:
    """apply_to_jacobian_diag updates only the bank block."""

    def test_diagonal_updated_only_in_bank_slice(self):
        D = 50
        i_transport = 10
        n_transport = 30
        layout = dict(i_transport=i_transport, n_transport=n_transport)
        # base_diag is all 1's
        base_diag = jnp.ones(D)
        ap_diag = -2.0 * jnp.ones((3, n_transport // 3))
        new_diag = apply_to_jacobian_diag(base_diag, ap_diag, layout)
        assert new_diag.shape == base_diag.shape
        # Active region: 1 + (-2) = -1
        active = new_diag[i_transport:i_transport + n_transport]
        assert jnp.all(active == -1.0), f"active region has wrong values: {active}"
        # Inactive regions unchanged
        assert jnp.all(new_diag[:i_transport] == 1.0)
        assert jnp.all(new_diag[i_transport + n_transport:] == 1.0)

    def test_layout_size_mismatch_raises(self):
        D = 10
        layout = dict(i_transport=2, n_transport=6)
        base_diag = jnp.ones(D)
        # Wrong-size ap_diag: 3 species × 3 bins = 9 ≠ 6
        ap_diag = jnp.ones((3, 3))
        with pytest.raises(ValueError, match="ap_diag flat size"):
            apply_to_jacobian_diag(base_diag, ap_diag, layout)


class TestPreconditionedJacobianContractivity:
    """Numerical check: the preconditioned Jacobian has Re(eigvals) ≤ 0 + eps."""

    def test_dense_preconditioned_jacobian_max_real_eig_non_positive(self):
        # Construct a small test problem mimicking the AP-bank structure:
        # 3 species × 1 mu × 4 q = 12 bank entries + 4 thermo/network = 16 total
        n_species = 3
        N_mu = 1
        N_q = 4
        n_bank = n_species * N_mu * N_q
        n_other = 4
        D = n_bank + n_other

        # Base Jacobian: small skew-symmetric (eigenvalues purely imaginary)
        rng = np.random.default_rng(0)
        A = rng.standard_normal((D, D))
        J_full = jnp.asarray(0.5 * (A - A.T))   # skew-symmetric: pure imag eigs

        layout = dict(i_transport=0, n_transport=n_bank)
        T_gamma = 1.0
        H_MeV = 1.0e-22
        q_nodes = jnp.linspace(0.5, 4.0, N_q)
        ap_diag = compute_ap_preconditioner_diag(
            T_gamma, H_MeV, q_nodes, N_mu, n_species=n_species,
        )
        J_eff = materialize_preconditioned_jacobian(J_full, ap_diag, layout)

        eigs = jnp.linalg.eigvals(J_eff)
        max_real = float(jnp.max(eigs.real))
        # The base J has Re(eig)=0 (skew-symm); adding a strictly-negative
        # diagonal on the bank block must keep all real parts ≤ 0 + eps.
        assert max_real <= 1.0e-9, (
            f"max Re(eig) of preconditioned J = {max_real:.3e}; "
            f"preconditioner failed contractivity contract."
        )


# ═══════════════════════════════════════════════════════════════════════
# §2. Phase δ-2 placeholder
# ═══════════════════════════════════════════════════════════════════════

class TestApplyToLowRankFactors:
    """Phase δ-2: helper that injects AP preconditioner into LowRankJacobianFactors."""

    def test_base_jacobian_diagonal_updated(self):
        from rabbit.jax.solver_jax_rodas5p import LowRankJacobianFactors
        from rabbit.jax.collision_ap_preconditioner_jax import (
            apply_to_low_rank_factors, compute_ap_preconditioner_diag,
        )
        n_species = 3
        N_mu = 1
        N_q = 4
        n_bank = n_species * N_mu * N_q
        n_other = 4
        D = n_bank + n_other

        # Construct a representative low-rank factor structure
        base = jnp.eye(D) * 0.0   # zero diagonal so we see the addition
        L = jnp.zeros((D, 1))
        C = jnp.zeros((1, 1))
        R = jnp.zeros((1, D))
        factors = LowRankJacobianFactors(
            base_jacobian=base, left_factor=L, core_matrix=C, right_factor=R,
        )

        layout = dict(i_transport=0, n_transport=n_bank)
        T_gamma = 1.0
        H_MeV = 1.0e-22
        q_nodes = jnp.linspace(0.5, 4.0, N_q)
        ap_diag = compute_ap_preconditioner_diag(
            T_gamma, H_MeV, q_nodes, N_mu, n_species=n_species,
        )

        factors_eff = apply_to_low_rank_factors(factors, ap_diag, layout)

        # The L, C, R parts are unchanged
        assert jnp.array_equal(factors_eff.left_factor, factors.left_factor)
        assert jnp.array_equal(factors_eff.core_matrix, factors.core_matrix)
        assert jnp.array_equal(factors_eff.right_factor, factors.right_factor)

        # The base Jacobian got a non-zero diagonal in the bank slice only
        new_diag = jnp.diagonal(factors_eff.base_jacobian)
        assert jnp.all(new_diag[:n_bank] < 0.0), (
            "bank-slice diagonal should be negative after preconditioner"
        )
        assert jnp.all(new_diag[n_bank:] == 0.0), (
            "non-bank diagonal entries should be untouched"
        )

    def test_layout_size_mismatch_raises(self):
        from rabbit.jax.solver_jax_rodas5p import LowRankJacobianFactors
        from rabbit.jax.collision_ap_preconditioner_jax import (
            apply_to_low_rank_factors,
        )
        D = 8
        base = jnp.eye(D)
        factors = LowRankJacobianFactors(
            base_jacobian=base,
            left_factor=jnp.zeros((D, 1)),
            core_matrix=jnp.zeros((1, 1)),
            right_factor=jnp.zeros((1, D)),
        )
        layout = dict(i_transport=2, n_transport=6)
        # Wrong size: 3 species × 3 = 9 ≠ 6
        ap_diag = jnp.ones((3, 3))
        with pytest.raises(ValueError, match="ap_diag flat size"):
            apply_to_low_rank_factors(factors, ap_diag, layout)


@pytest.mark.skip(
    reason=(
        "Phase δ-3 deferred: collision_mode='ap_preconditioned_canonical' "
        "wire-up into driver_typeI_full_boltzmann.py and the FLRW "
        "Mangano gap < 5e-3 acceptance test land in the next round; the "
        "δ-2 helper apply_to_low_rank_factors is the integration point."
    ),
)
def test_ap_preconditioned_canonical_flrw_neff_under_budget():
    """When wired, asserts |N_eff_measured - 3.044| < 5e-3 with this collision_mode."""
    pass
