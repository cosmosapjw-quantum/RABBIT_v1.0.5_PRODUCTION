"""BD622-R4 (audit F-024): forbid explicit per_species at tier < 2.

At tier < 2 the state layout sets i_tne = i_tnx = -1 (unused sentinels), so
characteristic_species_mode="per_species" would make both nu-bank temperature
reads alias y[-1] = X_8 (last nuclear species) and live-3T thermo would run
silently on garbage. FullCoupledConfig.__post_init__ must reject the
combination at construction time.
"""
import pytest

from rabbit.drivers.full_coupled_typeI import FullCoupledConfig


def test_explicit_per_species_tier1_raises():
    """Explicit per_species + tier=1 must be rejected (F-024 sentinel alias)."""
    with pytest.raises(ValueError, match="F-024"):
        FullCoupledConfig(characteristic_species_mode="per_species", tier=1)


def test_per_species_tier2_collisions_constructs():
    """per_species is the supported production mode at tier 2 + collisions."""
    cfg = FullCoupledConfig(
        characteristic_species_mode="per_species",
        tier=2,
        enable_collisions=True,
    )
    assert cfg.characteristic_species_mode == "per_species"


def test_default_auto_tier1_constructs():
    """Default config (auto, tier=1) must remain constructible: auto only
    promotes to per_species at tier >= 2 with collisions enabled."""
    cfg = FullCoupledConfig()
    assert cfg.tier == 1
    assert cfg.characteristic_species_mode == "auto"
