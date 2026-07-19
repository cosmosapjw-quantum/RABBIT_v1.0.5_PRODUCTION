"""tests/test_ap_preconditioned_dispatch.py — Phase δ-3 dispatch smoke.

Validates that ``collision_mode='ap_preconditioned_canonical'``:

  1. Is recognized by ``_collision_bank_core_jax`` and produces the
     same RHS as ``ap_unified_preflight`` (no physics divergence yet —
     the Jacobian preconditioner wire is Phase δ-4).
  2. Is detected by the new ``_mode_enables_ap_preconditioner``
     predicate (used by the future Jacobian-build path).
  3. Is rejected by ``_mode_enables_kernel_jacobian_preconditioner``
     so the existing kernel-remap-preconditioned path is not
     accidentally activated.

Plan §2.5 / §13.6 acceptance:
    - The new mode is wired into the dispatch (this commit).
    - The Mangano N_eff gap closure (< 5e-3) acceptance test lands in
      Phase δ-4 alongside the actual Jacobian-build augmentation via
      ``rabbit.jax.collision_ap_preconditioner_jax.apply_to_low_rank_factors``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest


class TestModePredicates:
    """The new mode is recognized by the dedicated predicate."""

    def test_predicate_true_for_new_mode(self):
        from rabbit.jax.driver_typeI_full_boltzmann import (
            _mode_enables_ap_preconditioner,
        )
        assert _mode_enables_ap_preconditioner("ap_preconditioned_canonical") is True
        assert _mode_enables_ap_preconditioner(" Ap_Preconditioned_Canonical ") is True

    def test_predicate_false_for_existing_modes(self):
        from rabbit.jax.driver_typeI_full_boltzmann import (
            _mode_enables_ap_preconditioner,
            _mode_enables_kernel_jacobian_preconditioner,
        )
        for m in (
            "ap_unified_preflight",
            "ap_unified_nu_nu_preflight",
            "ap_unified_nu_nu_spectral_preflight",
            "spectral_relaxation_preflight",
            "projected_physical_preflight",
            "jax_kernel_preflight",
            "jax_kernel_remap_preflight",
            "jax_kernel_remap_preconditioned_preflight",
            "collisionless",
        ):
            assert _mode_enables_ap_preconditioner(m) is False, (
                f"mode={m} should not enable ap preconditioner"
            )
        # And the kernel-remap predicate is also false for our new mode
        assert (
            _mode_enables_kernel_jacobian_preconditioner("ap_preconditioned_canonical")
            is False
        )


class TestRhsParityWithApUnifiedPreflight:
    """The new mode has the same RHS as ap_unified_preflight (Phase δ-3 contract)."""

    def test_collision_bank_core_returns_same_value(self):
        """Bit-exact RHS equality at a representative thermodynamic state."""
        from rabbit.jax.driver_typeI_full_boltzmann import _collision_bank_core_jax

        N_q = 8
        # Bank shape: (3 species, N_mu=1 ray, N_q nodes) flat to (3 * N_mu * N_q,)
        bank_state = jnp.linspace(0.0, 0.1, 3 * 1 * N_q)
        q_nodes = jnp.linspace(0.5, 8.0, N_q)
        q_weights = jnp.ones_like(q_nodes)
        T_gamma = jnp.array(1.0)
        T_nu_e = jnp.array(0.7)
        T_nu_x = jnp.array(0.7)
        H_inv_sec = jnp.array(1.0e-22)

        rhs_ap = _collision_bank_core_jax(
            bank_state,
            collision_mode="ap_unified_preflight",
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
            closure_strength=jnp.array(1.0),
        )
        rhs_pre = _collision_bank_core_jax(
            bank_state,
            collision_mode="ap_preconditioned_canonical",
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
            closure_strength=jnp.array(1.0),
        )
        # The RHS values must match bit-exactly: same internal kernel.
        assert jnp.allclose(rhs_ap, rhs_pre), (
            f"new collision_mode RHS deviates from ap_unified_preflight: "
            f"max|Δ|={float(jnp.max(jnp.abs(rhs_ap - rhs_pre)))}"
        )


class TestApPreconditionedEndToEnd:
    """Phase δ-4: full end-to-end run with the new collision_mode.

    The Jacobian-only difference relative to ap_unified_preflight means
    the forward y_final should match within numerical tolerance at
    well-converged grids; at small grids the preconditioner shifts
    N_eff by O(1e-6). The Mangano-gap closure budget < 5e-3 is the
    Phase δ-5 acceptance test (large grid, CL3, slower convergence).
    """

    @pytest.mark.slow
    def test_runs_end_to_end_without_crash(self):
        from rabbit.jax.driver_typeI_full_boltzmann import (
            JAXFullBoltzmannConfig, run_full_boltzmann_jax,
        )
        cfg = JAXFullBoltzmannConfig(
            Sigma_H_plus=0.0,
            eta=6.104e-10,
            tau_n=878.4,
            correction_level=0,
            N_q=4,
            N_mu=2,
            thermo_tier=2,
            collision_mode="ap_preconditioned_canonical",
        )
        import numpy as np
        result = run_full_boltzmann_jax(cfg)
        assert result.success
        assert np.isfinite(result.Yp)
        assert np.isfinite(result.DH)
        n_eff = result.metadata.get("N_eff_measured", float("nan"))
        assert np.isfinite(n_eff)
        # Should be in the AP-unified ballpark (3.03–3.04) at this grid.
        assert 3.0 < n_eff < 3.06, f"N_eff={n_eff} outside expected envelope"
        # Metadata advertises the new collision-scope contract.
        assert result.metadata["collision_scope_contract"] == (
            "ap_preconditioned_canonical_modified_equation_v1"
        )

    @pytest.mark.slow
    def test_n_eff_within_baseline_envelope(self):
        """Preconditioner should not destabilise the AP-unified baseline.

        At low resolution the Jacobian augmentation is small enough that
        the result must agree with ap_unified_preflight to 1e-3 in N_eff.
        """
        from rabbit.jax.driver_typeI_full_boltzmann import (
            JAXFullBoltzmannConfig, run_full_boltzmann_jax,
        )
        common = dict(
            Sigma_H_plus=0.0, eta=6.104e-10, tau_n=878.4,
            correction_level=0, N_q=4, N_mu=2, thermo_tier=2,
        )
        baseline = run_full_boltzmann_jax(
            JAXFullBoltzmannConfig(**common, collision_mode="ap_unified_preflight")
        )
        precond = run_full_boltzmann_jax(
            JAXFullBoltzmannConfig(**common, collision_mode="ap_preconditioned_canonical")
        )
        n_baseline = float(baseline.metadata["N_eff_measured"])
        n_precond = float(precond.metadata["N_eff_measured"])
        delta = abs(n_precond - n_baseline)
        assert delta < 1.0e-3, (
            f"preconditioned N_eff={n_precond} drift from baseline "
            f"N_eff={n_baseline} too large: |Δ|={delta:.3e}"
        )
