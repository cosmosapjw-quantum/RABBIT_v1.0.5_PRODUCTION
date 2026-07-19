from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest


def _load_module():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_jax_tier2_auto_policy_smoke.py'
    if not script.exists():
        pytest.skip(f"benchmark script not found: {script}")
    spec = importlib.util.spec_from_file_location('benchmark_jax_tier2_auto_policy_smoke', script)
    if spec is None or spec.loader is None:
        pytest.skip("failed to create import spec for benchmark script")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"benchmark script import failed: {exc}")
    return module


def test_aggregate_two_case_partial_timeout_state(tmp_path):
    mod = _load_module()
    plan = {
        'contract': 'auto_policy_recording_plan_v1',
        'source_smoke_contract': 'auto_policy_end_to_end_smoke_v1',
        'cases': [
            {'label': 'finite_shear', 'recording_output_stem': 'auto_policy_smoke_finite_shear'},
            {'label': 'small_shear', 'recording_output_stem': 'auto_policy_smoke_small_shear'},
            {'label': 'FLRW', 'recording_output_stem': 'auto_policy_smoke_flrw'},
        ],
    }
    partial_dir = tmp_path / 'partials'
    partial_dir.mkdir()
    (partial_dir / 'auto_policy_smoke_finite_shear.json').write_text(json.dumps({
        'label': 'finite_shear', 'recording_status': 'timed_out', 'timeout_seconds': 5,
    }))
    (partial_dir / 'auto_policy_smoke_small_shear.json').write_text(json.dumps({
        'label': 'small_shear', 'recording_status': 'timed_out', 'timeout_seconds': 5,
    }))
    summary = mod._aggregate(plan, partial_dir)
    assert summary['recording_status'] == 'partially_recorded'
    assert summary['missing_cases'] == ['FLRW']
    assert summary['counts']['recorded'] == 0
    assert summary['counts']['timed_out'] == 2
    assert summary['counts']['failed'] == 0
    assert [c['label'] for c in summary['cases']] == ['finite_shear', 'small_shear']


def test_aggregate_all_cases_attempted_but_incomplete(tmp_path):
    mod = _load_module()
    plan = {
        'contract': 'auto_policy_recording_plan_v1',
        'source_smoke_contract': 'auto_policy_end_to_end_smoke_v1',
        'cases': [
            {'label': 'finite_shear', 'recording_output_stem': 'auto_policy_smoke_finite_shear'},
            {'label': 'small_shear', 'recording_output_stem': 'auto_policy_smoke_small_shear'},
            {'label': 'FLRW', 'recording_output_stem': 'auto_policy_smoke_flrw'},
        ],
    }
    partial_dir = tmp_path / 'partials'
    partial_dir.mkdir()
    (partial_dir / 'auto_policy_smoke_finite_shear.json').write_text(json.dumps({
        'label': 'finite_shear', 'recording_status': 'timed_out', 'timeout_seconds': 5,
    }))
    (partial_dir / 'auto_policy_smoke_small_shear.json').write_text(json.dumps({
        'label': 'small_shear', 'recording_status': 'timed_out', 'timeout_seconds': 5,
    }))
    (partial_dir / 'auto_policy_smoke_flrw.json').write_text(json.dumps({
        'label': 'FLRW', 'recording_status': 'timed_out', 'timeout_seconds': 5,
    }))
    summary = mod._aggregate(plan, partial_dir)
    assert summary['recording_status'] == 'all_cases_attempted_incomplete'
    assert summary['missing_cases'] == []
    assert summary['counts']['recorded'] == 0
    assert summary['counts']['timed_out'] == 3
    assert summary['counts']['failed'] == 0
    assert [c['label'] for c in summary['cases']] == ['finite_shear', 'small_shear', 'FLRW']


def test_merge_attempt_payload_preserves_history():
    mod = _load_module()
    existing = {
        'label': 'finite_shear',
        'recording_status': 'timed_out',
        'timeout_seconds': 5.0,
    }
    new_payload = {
        'label': 'finite_shear',
        'recording_status': 'timed_out',
        'timeout_seconds': 15.0,
    }
    merged = mod._merge_attempt_payload(existing, new_payload)
    assert merged['attempt_count'] == 2
    assert [a['timeout_seconds'] for a in merged['attempt_history']] == [5.0, 15.0]
    assert merged['latest_timeout_seconds'] == 15.0


def test_apply_escalation_uses_next_timeout_for_selected_case(tmp_path):
    mod = _load_module()
    plan = {
        'contract': 'auto_policy_recording_plan_v1',
        'source_smoke_contract': 'auto_policy_end_to_end_smoke_v1',
        'cases': [
            {'order': 1, 'label': 'finite_shear', 'recording_output_stem': 'auto_policy_smoke_finite_shear'},
            {'order': 2, 'label': 'small_shear', 'recording_output_stem': 'auto_policy_smoke_small_shear'},
            {'order': 3, 'label': 'FLRW', 'recording_output_stem': 'auto_policy_smoke_flrw'},
        ],
    }
    partial_dir = tmp_path / 'partials'
    partial_dir.mkdir()
    for stem, label in [('auto_policy_smoke_finite_shear', 'finite_shear'), ('auto_policy_smoke_small_shear', 'small_shear'), ('auto_policy_smoke_flrw', 'FLRW')]:
        (partial_dir / f'{stem}.json').write_text(json.dumps({'label': label, 'recording_status': 'timed_out', 'timeout_seconds': 5.0}))
    case = {'label': 'finite_shear'}
    resolved = mod._resolve_retry_timeout(case, 5.0, partial_dir, plan, apply_escalation=True)
    assert resolved == 15.0


def test_append_attempt_started_records_launch_history(tmp_path):
    mod = _load_module()
    partial = tmp_path / 'finite.json'
    case = {
        'label': 'finite_shear',
        'resolved_solver_jacobian_mode': 'exact_per_step',
        'auto_resolution_contract': 'auto_exact_fallback_outside_locked_scope',
        'resolved_solver_jacobian_refresh_stride_phase1': 1,
        'resolved_solver_jacobian_refresh_stride_phase2': 1,
        'common': {'shear_regime': 'finite_shear'},
        'in_locked_policy_scope': False,
    }
    merged = mod._append_attempt_started(partial, case, 60.0)
    assert merged['recording_status'] == 'attempt_started'
    assert merged['attempt_count'] == 1
    assert merged['latest_timeout_seconds'] == 60.0
    assert merged['attempt_history'][0]['recording_status'] == 'attempt_started'
