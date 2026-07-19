import pytest

from rabbit.drivers.full_coupled_typeI import FullCoupledConfig


def test_legacy_shared_tier2_rejected_by_default():
    with pytest.raises(ValueError):
        FullCoupledConfig(
            tier=2,
            enable_collisions=True,
            characteristic_species_mode="legacy_shared",
            enable_teff=False,
        )


def test_legacy_shared_tier2_research_only_optin():
    cfg = FullCoupledConfig(
        tier=2,
        enable_collisions=True,
        characteristic_species_mode="legacy_shared",
        allow_legacy_shared_tier2_research=True,
        enable_teff=False,
    )
    assert cfg.requested_characteristic_species_mode == "legacy_shared"
    assert cfg.characteristic_species_mode == "legacy_shared"


def test_characteristic_species_mode_auto_promotes_with_requested_trace():
    cfg = FullCoupledConfig(
        tier=2,
        enable_collisions=True,
        characteristic_species_mode="auto",
        enable_teff=False,
    )
    assert cfg.requested_characteristic_species_mode == "auto"
    assert cfg.characteristic_species_mode == "per_species"


def test_characteristic_species_mode_rejects_invalid_literal():
    with pytest.raises(ValueError, match="characteristic_species_mode must be one of"):
        FullCoupledConfig(
            tier=2,
            enable_collisions=True,
            characteristic_species_mode="shared",
            enable_teff=False,
        )
