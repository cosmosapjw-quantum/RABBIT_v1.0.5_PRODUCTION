# tests/test_anisotropic_thermal_prerun.py
import numpy as np
import pytest

from rabbit.validation.anisotropic_thermal_prerun import (
    anisotropic_thermal_prerun_to_T0,
)
# NOTE (Phase 3 deflation): the isotropic-limit parity test against the FLRW prerun
# `augmented_continuous_ap65_rhs._phase1_thermo_prerun_to_T0` was removed with the AP65
# audit track. Re-anchor to a surviving FLRW prerun reference if that parity is wanted.


def test_isotropic_limit_returns_zero_shear_and_zero_A_modes():
    aniso = anisotropic_thermal_prerun_to_T0(
        T_start_MeV=3.0, T_end_MeV=1.0, n_species=9, q_count=4
    )
    assert aniso["Sigma_plus0_effective"] == 0.0
    assert aniso["Sigma_minus0_effective"] == 0.0
    modes = np.asarray(aniso["initial_A_modes"], dtype=float)
    assert modes.shape == (9, 3, 4)
    assert np.all(modes == 0.0)


def test_nonzero_shear_raises_not_implemented_until_stage_b():
    with pytest.raises(NotImplementedError):
        anisotropic_thermal_prerun_to_T0(
            T_start_MeV=3.0,
            T_end_MeV=1.0,
            n_species=9,
            q_count=4,
            sigma_plus0=1.0e-3,
        )


def test_nonzero_A_offset_raises_not_implemented_until_stage_b():
    with pytest.raises(NotImplementedError):
        anisotropic_thermal_prerun_to_T0(
            T_start_MeV=3.0,
            T_end_MeV=1.0,
            n_species=9,
            q_count=4,
            initial_A_monopole_offset=1.0e-5,
        )


def test_invalid_temperature_ordering_raises():
    with pytest.raises(ValueError):
        anisotropic_thermal_prerun_to_T0(
            T_start_MeV=1.0, T_end_MeV=3.0, n_species=9, q_count=4
        )
