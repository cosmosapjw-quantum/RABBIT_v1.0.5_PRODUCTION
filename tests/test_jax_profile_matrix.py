from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_jax_tier2_profile_matrix.py'
if not _SCRIPT.exists():
    pytest.skip(
        f"benchmark script not found: {_SCRIPT}",
        allow_module_level=True,
    )
_spec = importlib.util.spec_from_file_location('benchmark_jax_profile_matrix', _SCRIPT)
if _spec is None or _spec.loader is None:
    pytest.skip("failed to create import spec for benchmark script", allow_module_level=True)
_mod = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_mod)
except Exception as _exc:
    pytest.skip(f"benchmark script import failed: {_exc}", allow_module_level=True)


def test_case_catalog_and_probe_scope() -> None:
    assert set(_mod.CASE_SPECS) == {'finite_shear_exact', 'flrw_auto_locked'}
    cfg = _mod._probe_config('finite_shear_exact', 4.0)
    assert cfg['N_q'] == 6
    assert cfg['thermo_tier'] == 2
    assert cfg['max_steps'] == 1
    assert cfg['probe_budget_seconds'] == 4.0


def test_bucket_name_groups_expected_paths() -> None:
    assert _mod._bucket_name('/tmp/src/rabbit/jax/driver_typeI.py') == 'rabbit.jax'
    assert _mod._bucket_name('/tmp/src/rabbit/thermo/nudec_coupled.py') == 'rabbit.thermo'
    assert _mod._bucket_name('/usr/lib/python3.13/importlib/__init__.py') == 'python.importlib'
    assert _mod._bucket_name('/opt/pyvenv/lib/python3.13/site-packages/jax/_src/pjit.py') == 'third_party.jax'


def test_derive_case_profile_flags_startup_and_unresolved_probe() -> None:
    case = {
        'fresh_import_profile': {
            'wall_seconds': 6.0,
            'payload': {
                'profile': {
                    'elapsed_seconds': 4.0,
                    'top_buckets_by_self_seconds': [
                        {'bucket': 'python.importlib', 'self_seconds': 1.2},
                        {'bucket': 'third_party.jax', 'self_seconds': 0.8},
                    ],
                }
            },
        },
        'fresh_case_profile': {
            'wall_seconds': 9.0,
            'payload': {
                'summary_profile': {'elapsed_seconds': 1.5},
                'first_probe_profile': {'elapsed_seconds': 4.0, 'status': 'timed_out'},
                'second_probe_profile': {'elapsed_seconds': 4.0, 'status': 'timed_out'},
            },
        },
    }
    out = _mod._derive_case_profile(case)
    assert out['fresh_case_profile_start_residual_seconds'] == 3.5
    assert 'subprocess_startup_or_ipc_nontrivial' in out['evidence']
    assert 'importlib_self_time_is_top_import_bucket' in out['evidence']
    assert 'no_material_same_process_warm_speedup_under_bounded_probe' in out['evidence']
    assert 'bounded_probe_unresolved_twice' in out['evidence']
