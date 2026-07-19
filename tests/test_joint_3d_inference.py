"""Host 3D likelihood diagnostics and the frozen B-05 sampler boundary."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════
# §1. Fast smoke gates (run in <2 minutes)
# ═══════════════════════════════════════════════════════════════════════

class TestLogPosterior3D:
    """Forward + boundary + gradient smoke."""

    @pytest.mark.slow  # ~12s per call: hits the SciPy canonical solver
    def test_log_posterior_at_flrw_finite(self):
        from rabbit.inference.joint_3d_inference import (
            JointInference3DConfig, make_log_posterior_3d,
        )
        cfg = JointInference3DConfig(
            backend="scipy", correction_level=2, N_q=20,
        )
        lp = make_log_posterior_3d(cfg)
        val = float(lp(jnp.array([0.0, 6.104, 878.4])))
        assert np.isfinite(val), f"log_post at FLRW = {val} (should be finite)"

    def test_log_posterior_below_boundary_is_minus_inf(self):
        """Σ_H < 0 must hit the half-line wall (-inf log-prior)."""
        from rabbit.inference.joint_3d_inference import (
            JointInference3DConfig, make_log_posterior_3d,
        )
        # Use a fast toy forward by overriding the solver to return a known
        # constant — we just want to verify the boundary penalty kicks in.
        cfg = JointInference3DConfig(
            backend="scipy", correction_level=0, N_q=8,  # cheapest forward
        )
        lp = make_log_posterior_3d(cfg)
        val = float(lp(jnp.array([-0.05, 6.104, 878.4])))
        assert val == float("-inf"), (
            f"log_post at Σ_H=-0.05 = {val} (should be -inf)"
        )

    @pytest.mark.slow
    def test_log_posterior_gradient_fails_closed_until_b05(self, monkeypatch):
        """The retired finite-difference VJP must not fabricate a gradient."""
        import rabbit.inference.bbn_inference as bbn_inference
        from rabbit.inference.observables import BBN_JAX_SAMPLER_UNAVAILABLE
        from rabbit.inference.joint_3d_inference import (
            JointInference3DConfig, make_log_posterior_3d,
        )
        monkeypatch.setattr(
            bbn_inference,
            "_scipy_forward_solve",
            lambda **_kwargs: {
                "Yp": 0.245,
                "DH": 2.5e-5,
                "success": True,
                "metadata": {},
            },
        )
        cfg = JointInference3DConfig(
            backend="scipy", correction_level=2, N_q=20,
        )
        lp = make_log_posterior_3d(cfg)
        params = jnp.array([0.0, 6.104, 878.4])
        with pytest.raises(RuntimeError) as exc_info:
            jax.value_and_grad(lp)(params)
        assert str(exc_info.value) == BBN_JAX_SAMPLER_UNAVAILABLE

    @pytest.mark.slow
    def test_log_posterior_disfavors_sigma_at_0p1(self):
        """Posterior at Σ_H=0.1 must be lower than at Σ_H=0 (FLRW)."""
        from rabbit.inference.joint_3d_inference import (
            JointInference3DConfig, make_log_posterior_3d,
        )
        cfg = JointInference3DConfig(
            backend="scipy", correction_level=2, N_q=20,
        )
        lp = make_log_posterior_3d(cfg)
        v0 = float(lp(jnp.array([0.0, 6.104, 878.4])))
        v01 = float(lp(jnp.array([0.1, 6.104, 878.4])))
        # The Half-Normal prior alone contributes -0.5*(0.1/0.3)^2 = -0.056
        # so at minimum the difference should be ~0.05 nat
        assert v0 - v01 > 0.04, (
            f"FLRW logpost {v0} not sufficiently above Σ=0.1 logpost {v01}; "
            f"prior should disfavor Σ=0.1."
        )


# ═══════════════════════════════════════════════════════════════════════
# §2. NUTS-3D end-to-end synthetic recovery (very slow gate)
# ═══════════════════════════════════════════════════════════════════════

class TestNutsThreeDSyntheticRecovery:
    """The former recovery entry points remain explicitly blocked."""

    @pytest.mark.slow
    @pytest.mark.production
    def test_flrw_truth_recovery_smoke(self):
        """Recovery cannot run through the non-traceable host forward."""
        from rabbit.inference.observables import BBN_JAX_SAMPLER_UNAVAILABLE
        from rabbit.inference.joint_3d_inference import (
            BBNNuts3DConfig, JointInference3DConfig, run_bbn_nuts_3d,
        )
        cfg = BBNNuts3DConfig(
            inference=JointInference3DConfig(
                backend="jax",
                correction_level=0,
                N_q=8,
            ),
            num_warmup=10, num_samples=20, max_tree_depth=3,
        )
        with pytest.raises(RuntimeError) as exc_info:
            run_bbn_nuts_3d(cfg, rng_key=jax.random.PRNGKey(0))
        assert str(exc_info.value) == BBN_JAX_SAMPLER_UNAVAILABLE

    @pytest.mark.expensive
    @pytest.mark.slow
    @pytest.mark.production
    def test_flrw_truth_recovery_full_chain(self):
        """The expensive recovery gate is blocked rather than skipped green."""
        from rabbit.inference.observables import BBN_JAX_SAMPLER_UNAVAILABLE
        from rabbit.inference.joint_3d_inference import (
            BBNNuts3DConfig, JointInference3DConfig, run_bbn_nuts_3d,
        )
        cfg = BBNNuts3DConfig(
            inference=JointInference3DConfig(
                backend="jax",
                correction_level=2,
                N_q=12,
            ),
            num_warmup=20, num_samples=50, max_tree_depth=4,
        )
        with pytest.raises(RuntimeError) as exc_info:
            run_bbn_nuts_3d(cfg, rng_key=jax.random.PRNGKey(0))
        assert str(exc_info.value) == BBN_JAX_SAMPLER_UNAVAILABLE
