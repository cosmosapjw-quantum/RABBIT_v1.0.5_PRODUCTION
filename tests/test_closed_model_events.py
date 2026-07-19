"""Foundation F5 — Type IX recollapse + Mixmaster guard tests.

Verifies:
  * ``TerminalReason`` enum covers the four real terminal states + RUNNING.
  * ``classify_typeIX_state`` priority order: BBN-complete > recollapse >
    omega-out-of-range > mixmaster > running.
  * ``hubble_zero_event_factory`` builds a callable that crosses zero
    when H drops below the configured fraction of H_initial.
  * ``MixmasterTracker`` correctly counts axis-wise sign flips inside the
    sliding e-fold window, and ``trim`` drops stale history.
"""

from __future__ import annotations

import math

import pytest

from rabbit.geometry.closed_model_events import (
    IXEventConfig,
    MixmasterTracker,
    TerminalReason,
    classify_typeIX_state,
    hubble_zero_event_factory,
)


@pytest.mark.production
class TestTerminalReason:
    def test_values_are_unique(self):
        names = [r.value for r in TerminalReason]
        assert len(names) == len(set(names))

    def test_running_is_default_distinct(self):
        assert TerminalReason.RUNNING.value == "running"
        assert TerminalReason.RUNNING is not TerminalReason.BBN_COMPLETE


@pytest.mark.production
class TestClassifyTypeIXState:
    def test_running_when_all_thresholds_clear(self):
        r = classify_typeIX_state(
            H_over_H0=0.5, T_gamma_MeV=0.1,
            omega_budget=1.0, mixmaster_flip_count=0,
        )
        assert r is TerminalReason.RUNNING

    def test_bbn_complete_takes_priority_over_recollapse(self):
        # Even if H is low, finishing BBN is the canonical "done".
        r = classify_typeIX_state(
            H_over_H0=1e-5, T_gamma_MeV=1e-4,  # both fire individually
            omega_budget=1.0, mixmaster_flip_count=0,
        )
        assert r is TerminalReason.BBN_COMPLETE

    def test_recollapse_fires_when_only_H_low(self):
        r = classify_typeIX_state(
            H_over_H0=1e-5, T_gamma_MeV=0.05,  # T above BBN_END
            omega_budget=1.0, mixmaster_flip_count=0,
        )
        assert r is TerminalReason.RECOLLAPSED

    def test_omega_out_of_range_takes_priority_over_mixmaster(self):
        r = classify_typeIX_state(
            H_over_H0=0.5, T_gamma_MeV=0.05,
            omega_budget=2.0, mixmaster_flip_count=20,
        )
        assert r is TerminalReason.OMEGA_OUT_OF_RANGE

    def test_mixmaster_fires_when_flip_count_exceeds_threshold(self):
        r = classify_typeIX_state(
            H_over_H0=0.5, T_gamma_MeV=0.05,
            omega_budget=1.0, mixmaster_flip_count=11,
        )
        assert r is TerminalReason.MIXMASTER_DIVERGENT

    def test_mixmaster_threshold_is_strict_greater_than(self):
        # Exactly 10 flips => still RUNNING (default cap is 10).
        r = classify_typeIX_state(
            H_over_H0=0.5, T_gamma_MeV=0.05,
            omega_budget=1.0, mixmaster_flip_count=10,
        )
        assert r is TerminalReason.RUNNING

    def test_config_overrides_apply(self):
        cfg = IXEventConfig(T_BBN_END_MeV=0.05)
        r = classify_typeIX_state(
            H_over_H0=0.5, T_gamma_MeV=0.04,  # below custom T_END
            omega_budget=1.0, mixmaster_flip_count=0,
            config=cfg,
        )
        assert r is TerminalReason.BBN_COMPLETE


@pytest.mark.production
class TestHubbleZeroEventFactory:
    def test_returns_callable(self):
        cond = hubble_zero_event_factory(lambda t, y: y[0])
        assert callable(cond)

    def test_event_fires_at_threshold(self):
        cond = hubble_zero_event_factory(
            lambda t, y: y[0], H_threshold_fraction=1e-3, H_initial=1.0,
        )
        # H = 1.0 (well above) -> positive
        assert cond(0.0, [1.0]) > 0.0
        # H = 1e-3 (exactly threshold) -> zero
        assert math.isclose(cond(0.0, [1e-3]), 0.0, abs_tol=1e-15)
        # H = 5e-4 (below) -> negative
        assert cond(0.0, [5e-4]) < 0.0

    def test_threshold_scales_with_H_initial(self):
        cond = hubble_zero_event_factory(
            lambda t, y: y[0], H_threshold_fraction=1e-3, H_initial=10.0,
        )
        # Threshold = 1e-3 * 10 = 1e-2; H = 5e-3 fires (below).
        assert cond(0.0, [5e-3]) < 0.0
        # H = 1.5e-2 still positive (above).
        assert cond(0.0, [1.5e-2]) > 0.0


@pytest.mark.production
class TestMixmasterTracker:
    def test_empty_history_zero_flips(self):
        t = MixmasterTracker()
        assert t.flip_count_in_window == 0

    def test_one_observation_zero_flips(self):
        t = MixmasterTracker()
        t.observe(0.0, (1, 1, 1))
        assert t.flip_count_in_window == 0

    def test_single_axis_flip_counted(self):
        t = MixmasterTracker()
        t.observe(0.0, (+1, +1, +1))
        t.observe(0.1, (-1, +1, +1))   # one flip on axis 1
        assert t.flip_count_in_window == 1

    def test_multiple_axis_flips_counted(self):
        t = MixmasterTracker()
        t.observe(0.0, (+1, +1, +1))
        t.observe(0.1, (-1, -1, +1))   # two flips
        t.observe(0.2, (+1, +1, -1))   # three more flips => total 5
        assert t.flip_count_in_window == 5

    def test_zero_eigenvalue_not_a_flip(self):
        # A zero on either side does not count as a sign flip.
        t = MixmasterTracker()
        t.observe(0.0, (1, 1, 0))
        t.observe(0.1, (1, 1, +1))     # 0 -> +1 is not a flip
        assert t.flip_count_in_window == 0

    def test_window_drops_old_history(self):
        cfg = IXEventConfig(mixmaster_window_efolds=0.3)
        t = MixmasterTracker(config=cfg)
        t.observe(0.0, (+1, +1, +1))
        t.observe(0.1, (-1, +1, +1))   # flip 1
        t.observe(1.0, (+1, +1, +1))   # 1.0 - 0.3 = 0.7 cutoff; only this kept
        assert t.flip_count_in_window == 0  # nothing prior to 0.7

    def test_trim_clears_stale_history(self):
        cfg = IXEventConfig(mixmaster_window_efolds=0.5)
        t = MixmasterTracker(config=cfg)
        t.observe(0.0, (+1, +1, +1))
        t.observe(0.1, (-1, +1, +1))
        t.observe(2.0, (+1, +1, +1))
        t.trim()
        # Only the last entry remains (window is [1.5, 2.0]).
        assert len(t.history) == 1
        assert t.history[0][0] == 2.0

    def test_threshold_breach_only_via_classify(self):
        # Tracker counts; classifier decides the verdict.
        cfg = IXEventConfig(mixmaster_max_flips=2,
                             mixmaster_window_efolds=10.0)
        t = MixmasterTracker(config=cfg)
        t.observe(0.0, (+1, +1, +1))
        t.observe(0.1, (-1, -1, -1))    # 3 flips
        assert t.flip_count_in_window == 3
        r = classify_typeIX_state(
            H_over_H0=0.5, T_gamma_MeV=0.05,
            omega_budget=1.0,
            mixmaster_flip_count=t.flip_count_in_window,
            config=cfg,
        )
        assert r is TerminalReason.MIXMASTER_DIVERGENT


@pytest.mark.production
class TestIXEventConfig:
    def test_defaults_match_documented_thresholds(self):
        cfg = IXEventConfig()
        assert cfg.T_BBN_END_MeV == 0.01
        assert cfg.H_recollapse_eps == 1e-3
        assert cfg.omega_budget_max == 1.5
        assert cfg.mixmaster_window_efolds == 0.5
        assert cfg.mixmaster_max_flips == 10
        assert cfg.continue_through_bounce is False

    def test_config_is_frozen(self):
        cfg = IXEventConfig()
        with pytest.raises(Exception):
            cfg.T_BBN_END_MeV = 0.005  # type: ignore[misc]
