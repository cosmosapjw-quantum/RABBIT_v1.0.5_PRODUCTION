from __future__ import annotations

import importlib

import pytest


def test_legacy_tier2_three_T_rhs_fails_loudly_to_active_3t_closure() -> None:
    from rabbit.thermo import incomplete_decoupling

    with pytest.raises(NotImplementedError, match="coupled_3T_rhs"):
        incomplete_decoupling.tier2_three_T_rhs(1.0, 0.9, 0.8, 1.0e-22)


def test_projected_operator_rate_constant_is_sourced_from_nudec_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rabbit.collisions import projected_operator
    from rabbit.thermo import nudec_tables

    original = nudec_tables._C_RATE
    try:
        monkeypatch.setattr(nudec_tables, "_C_RATE", 123.0)
        reloaded = importlib.reload(projected_operator)
        assert reloaded._C_RATE == pytest.approx(123.0, rel=0.0, abs=0.0)
    finally:
        monkeypatch.setattr(nudec_tables, "_C_RATE", original)
        importlib.reload(projected_operator)
