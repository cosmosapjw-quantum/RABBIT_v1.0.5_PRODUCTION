import pytest

pytestmark = [pytest.mark.production, pytest.mark.release_smoke]


def test_scipy_rate_bounds_warning_and_error():
    from rabbit.network.abundances_standard import evaluate_nuclear_rates
    with pytest.warns(RuntimeWarning, match="outside the PRIMAT AC2024 tabulation window"):
        evaluate_nuclear_rates(1.0)
    with pytest.raises(ValueError, match="outside the PRIMAT AC2024 tabulation window"):
        evaluate_nuclear_rates(1.0, strict_bounds=True)


def test_tilted_v0_guard():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig
    with pytest.raises(ValueError, match=r"\|v0\| < 1"):
        TiltedBBNConfig(v0=1.0)


def test_tilted_axis_guard():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig
    with pytest.raises(ValueError, match="tilt_axis"):
        TiltedBBNConfig(tilt_axis=0)


def test_tilted_n_ell_guard():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig
    with pytest.raises(ValueError, match="n_ell=2 LRS or n_ell=3"):
        TiltedBBNConfig(n_ell=4)


def test_tilted_momentum_closure_guard():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig
    with pytest.raises(ValueError, match="momentum_constraint_closure"):
        TiltedBBNConfig(momentum_constraint_closure="instant_magic")


def test_tilted_hubble_closure_guard():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig
    with pytest.raises(ValueError, match="tilt_hubble_closure"):
        TiltedBBNConfig(tilt_hubble_closure="instant_magic")


def test_tilted_cl3_angular_kernel_requires_cl3():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig
    with pytest.raises(ValueError, match="tilt_cl3_angular_kernel requires correction_level=3"):
        TiltedBBNConfig(
            correction_level=2,
            tilt_weak_rate_boost=True,
            tilt_cl3_angular_kernel=True,
        )


def test_tilted_cl3_angular_kernel_requires_boosted_moments():
    # v3.0 Phase C: tilt_weak_rate_boost defaults to True, so the
    # validation only triggers when the user explicitly opts out.
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig
    with pytest.raises(ValueError, match="tilt_cl3_angular_kernel requires tilt_weak_rate_boost=True"):
        TiltedBBNConfig(
            correction_level=3,
            tilt_cl3_angular_kernel=True,
            tilt_weak_rate_boost=False,
        )


def test_tilted_runner_rejects_above_cl3_at_runtime():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn
    with pytest.raises(ValueError, match="Maximum is CL3"):
        run_tilted_bbn(TiltedBBNConfig(correction_level=4))


def test_python_typeI_runtime_guard_rejects_nonpositive_omega():
    from rabbit.drivers.full_coupled_typeI import _hubble_invsec
    with pytest.raises(ValueError, match="Silent boundary clamping is not allowed|Omega=.*<= 0"):
        _hubble_invsec(1.0, 1.0, 3.044, Sigma_sq=1.0)


def test_python_classA_runtime_guard_rejects_nonpositive_omega():
    from rabbit.drivers.classA_driver import _hubble_invsec
    with pytest.raises(ValueError, match="Silent boundary clamping is not allowed|Omega=.*<= 0"):
        _hubble_invsec(1.0, 1.0, 3.044, Sigma_sq=0.8, K=0.3)
