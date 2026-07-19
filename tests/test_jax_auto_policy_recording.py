from __future__ import annotations

import importlib.util
import json
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


def test_recording_status_classifier():
    mod = _load_module()
    assert mod._classify_recording_status(expected=3, recorded_cases=0, recorded_successes=0, started_cases=0) == 'plan_only'
    assert mod._classify_recording_status(expected=3, recorded_cases=1, recorded_successes=0, started_cases=0) == 'partially_recorded'
    assert mod._classify_recording_status(expected=3, recorded_cases=2, recorded_successes=1, started_cases=0) == 'partially_recorded'
    assert mod._classify_recording_status(expected=3, recorded_cases=3, recorded_successes=0, started_cases=0) == 'all_cases_attempted_incomplete'
    assert mod._classify_recording_status(expected=3, recorded_cases=3, recorded_successes=2, started_cases=0) == 'all_cases_attempted_incomplete'
    assert mod._classify_recording_status(expected=3, recorded_cases=3, recorded_successes=3, started_cases=0) == 'complete'
    assert mod._classify_recording_status(expected=3, recorded_cases=1, recorded_successes=0, started_cases=1) == 'attempts_started_unfinished'


def test_aggregate_counts_timeout_and_recorded(tmp_path):
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
        'label': 'finite_shear', 'recording_status': 'recorded', 'success': True,
    }))
    (partial_dir / 'auto_policy_smoke_small_shear.json').write_text(json.dumps({
        'label': 'small_shear', 'recording_status': 'timed_out', 'timeout_seconds': 5,
    }))
    summary = mod._aggregate(plan, partial_dir)
    assert summary['recording_status'] == 'partially_recorded'
    assert summary['missing_cases'] == ['FLRW']
    assert summary['counts']['recorded'] == 1
    assert summary['counts']['timed_out'] == 1
    assert summary['counts']['failed'] == 0
    assert summary['counts']['attempt_started'] == 0


def test_aggregate_attempt_started_state(tmp_path):
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
        'label': 'finite_shear', 'recording_status': 'attempt_started', 'timeout_seconds': 60.0,
    }))
    summary = mod._aggregate(plan, partial_dir)
    assert summary['recording_status'] == 'attempts_started_unfinished'
    assert summary['counts']['attempt_started'] == 1
    assert summary['counts']['timed_out'] == 0


def test_timeout_escalation_plan_from_all_attempted_incomplete(tmp_path):
    mod = _load_module()
    plan = {
        'contract': 'auto_policy_recording_plan_v1',
        'source_smoke_contract': 'auto_policy_end_to_end_smoke_v1',
        'cases': [
            {'order': 1, 'label': 'finite_shear', 'recording_output_stem': 'auto_policy_smoke_finite_shear', 'resolved_solver_jacobian_mode': 'exact_per_step', 'resolved_solver_jacobian_refresh_stride_phase1': 1, 'resolved_solver_jacobian_refresh_stride_phase2': 1},
            {'order': 2, 'label': 'small_shear', 'recording_output_stem': 'auto_policy_smoke_small_shear', 'resolved_solver_jacobian_mode': 'experimental_dependency_cadence', 'resolved_solver_jacobian_refresh_stride_phase1': 3, 'resolved_solver_jacobian_refresh_stride_phase2': 3},
            {'order': 3, 'label': 'FLRW', 'recording_output_stem': 'auto_policy_smoke_flrw', 'resolved_solver_jacobian_mode': 'experimental_dependency_cadence', 'resolved_solver_jacobian_refresh_stride_phase1': 4, 'resolved_solver_jacobian_refresh_stride_phase2': 2},
        ],
    }
    partial_dir = tmp_path / 'partials'
    partial_dir.mkdir()
    for stem, label in [('auto_policy_smoke_finite_shear', 'finite_shear'), ('auto_policy_smoke_small_shear', 'small_shear'), ('auto_policy_smoke_flrw', 'FLRW')]:
        (partial_dir / f'{stem}.json').write_text(json.dumps({'label': label, 'recording_status': 'timed_out', 'timeout_seconds': 5.0}))
    summary = mod._aggregate(plan, partial_dir)
    esc = mod.summarize_typeI_auto_policy_timeout_escalation(summary)
    assert esc['next_attempt_order'] == ['finite_shear', 'small_shear', 'FLRW']
    assert all(case['next_timeout_seconds'] == 15.0 for case in esc['cases'])
    assert all(case['should_retry'] for case in esc['cases'])
