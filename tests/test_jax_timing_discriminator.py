from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_jax_tier2_timing_discriminator.py'
if not _SCRIPT.exists():
    pytest.skip(
        f"benchmark script not found: {_SCRIPT}",
        allow_module_level=True,
    )
_spec = importlib.util.spec_from_file_location('benchmark_jax_timing_discriminator', _SCRIPT)
if _spec is None or _spec.loader is None:
    pytest.skip("failed to create import spec for benchmark script", allow_module_level=True)
_mod = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_mod)
except Exception as _exc:
    pytest.skip(f"benchmark script import failed: {_exc}", allow_module_level=True)


def test_case_catalog_contains_expected_labels() -> None:
    assert set(_mod.CASE_SPECS) == {'finite_shear_exact', 'flrw_auto_locked'}
    assert _mod._canonical_case('finite-shear-exact') == 'finite_shear_exact'
    assert _mod._canonical_case('FLRW_AUTO_LOCKED') == 'flrw_auto_locked'


def test_probe_config_is_fixed_scope() -> None:
    cfg = _mod._probe_config('finite_shear_exact', 4.0)
    assert cfg['N_q'] == 6
    assert cfg['thermo_tier'] == 2
    assert cfg['use_live_weak_monopoles'] is True
    assert cfg['correction_level'] == 2
    assert cfg['max_steps'] == 1
    assert cfg['probe_budget_seconds'] == 4.0


def test_derived_case_diagnosis_flags_startup_dominated_timeout_path() -> None:
    case = {
        'same_process_pair': {
            'first_probe': {'status': 'timed_out', 'elapsed_seconds': 4.1},
            'second_probe': {'status': 'timed_out', 'elapsed_seconds': 4.0},
        },
        'fresh_subprocess_summary': {'wall_seconds': 6.5, 'payload': {'worker_mode': 'summary_only'}},
        'fresh_subprocess_probe': {'wall_seconds': 10.9, 'payload': {'worker_mode': 'probe'}},
    }
    out = _mod._derive_case_diagnosis(case)
    assert out['fresh_subprocess_incremental_probe_seconds'] == 4.4
    assert 'fresh_subprocess_startup_nontrivial' in out['evidence']
    assert 'fresh_subprocess_wall_startup_dominated' in out['evidence']
    assert 'no_clear_same_process_warm_speedup_within_probe_budget' in out['evidence']
    assert 'driver_probe_unresolved_within_budget' in out['evidence']


def test_derived_case_diagnosis_allows_warm_speedup_when_present() -> None:
    case = {
        'same_process_pair': {
            'first_probe': {'status': 'returned', 'elapsed_seconds': 6.0},
            'second_probe': {'status': 'returned', 'elapsed_seconds': 2.0},
        },
        'fresh_subprocess_summary': {'wall_seconds': 1.0, 'payload': {'worker_mode': 'summary_only'}},
        'fresh_subprocess_probe': {'wall_seconds': 4.0, 'payload': {'worker_mode': 'probe'}},
    }
    out = _mod._derive_case_diagnosis(case)
    assert out['same_process_warm_speedup_factor'] == 3.0
    assert 'no_clear_same_process_warm_speedup_within_probe_budget' not in out['evidence']
