import pytest

from rabbit.drivers.full_coupled_typeI import FullCoupledConfig


def test_characteristic_direct_driver_rejects_teff():
    with pytest.raises(ValueError):
        FullCoupledConfig(enable_teff=True)


def test_species_identical_guard_is_not_production_default():
    cfg = FullCoupledConfig(tier=2, enable_collisions=True)
    assert cfg.allow_species_identical_research is False
