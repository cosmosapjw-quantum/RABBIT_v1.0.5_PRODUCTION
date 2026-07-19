"""tests/test_joint_3d_native_ad.py — v3.1 Phase δ acceptance gates.

Plan §δ. Validates the native-AD ad_backend dispatch in
:mod:`rabbit.inference.joint_3d_inference`.

Acceptance:
  1. JointInference3DConfig accepts ad_backend='rosenbrock_native'
     without raising at construction
  2. ad_backend='diffrax_native' is rejected by make_log_posterior_3d
     until a real joint-3D dispatch exists
  3. Unknown ad_backend values raise informatively
  4. Calling make_log_posterior_3d with ad_backend='rosenbrock_native'
     raises NotImplementedError citing the documented refactor path
  5. ad_backend='fd_legacy' (default) continues to work — no v2.0
     regression
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


jax.config.update("jax_enable_x64", True)


class TestADBackendValidation:

    def test_default_is_fd_legacy(self):
        from rabbit.inference.joint_3d_inference import JointInference3DConfig
        cfg = JointInference3DConfig()
        assert cfg.ad_backend == "fd_legacy"

    def test_rosenbrock_native_accepted_at_construction(self):
        from rabbit.inference.joint_3d_inference import JointInference3DConfig
        cfg = JointInference3DConfig(ad_backend="rosenbrock_native")
        assert cfg.ad_backend == "rosenbrock_native"

    def test_diffrax_native_rejected_until_dispatch_exists(self):
        from rabbit.inference.joint_3d_inference import (
            JointInference3DConfig, make_log_posterior_3d,
        )
        cfg = JointInference3DConfig(ad_backend="diffrax_native")
        with pytest.raises(ValueError, match=r"diffrax_native.*not wired"):
            make_log_posterior_3d(cfg)

    def test_unknown_backend_raises(self):
        from rabbit.inference.joint_3d_inference import (
            JointInference3DConfig, make_log_posterior_3d,
        )
        cfg = JointInference3DConfig(ad_backend="bogus")
        with pytest.raises(ValueError, match=r"ad_backend"):
            make_log_posterior_3d(cfg)


class TestRosenbrockNativeDispatch:

    def test_rosenbrock_native_raises_documented_message(self):
        from rabbit.inference.joint_3d_inference import (
            JointInference3DConfig, make_log_posterior_3d,
        )
        cfg = JointInference3DConfig(ad_backend="rosenbrock_native")
        with pytest.raises(NotImplementedError, match=r"research-grade follow-on"):
            make_log_posterior_3d(cfg)


class TestFDLegacyStillWorks:

    @pytest.mark.slow
    def test_fd_legacy_log_posterior_finite(self):
        """v3.1 Phase δ: FD-legacy default still produces finite log-posterior."""
        from rabbit.inference.joint_3d_inference import (
            JointInference3DConfig, make_log_posterior_3d,
        )
        cfg = JointInference3DConfig(
            ad_backend="fd_legacy", backend="scipy",
            correction_level=2, N_q=20,
        )
        lp = make_log_posterior_3d(cfg)
        val = float(lp(jnp.array([0.0, 6.104, 878.4])))
        assert np.isfinite(val), f"FD-legacy logpost = {val}"
