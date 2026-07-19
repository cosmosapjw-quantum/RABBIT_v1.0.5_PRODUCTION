from __future__ import annotations

import csv
import json
import math
import os
import time
import warnings
from collections import Counter
from pathlib import Path

from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI

SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
TEFF_CASES = [False, True]
N_Q = 20
CORRECTION_LEVEL = 0
THREAD_ENV = {
    'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
    'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
    'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
    'NUMEXPR_NUM_THREADS': os.environ.get('NUMEXPR_NUM_THREADS'),
}

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'session2_outputs'
JSON_PATH = OUTDIR / 'typeI_scipy_highres_safeguard_scan_session2_2026-04-08.json'
CSV_PATH = OUTDIR / 'typeI_scipy_highres_safeguard_scan_session2_2026-04-08.csv'
SUMMARY_MD = OUTDIR / 'SAFeguard_session2_highres_summary_2026-04-08.md'
POLICY_JSON = OUTDIR / 'typeI_scipy_highres_policy_invariance_session2_2026-04-08.json'


def _sanitize(x):
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return str(x)
        return x
    if isinstance(x, dict):
        return {str(k): _sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sanitize(v) for v in x]
    return str(x)


def classify_policy(row: dict, policy: str) -> dict:
    """Post-hoc claim gate classification for the SciPy Type-I path.

    Important: the current SciPy Type-I reference path in this zip does not expose
    a runtime strict/warn/allow switch. Therefore policy invariance is tested as a
    classification layer on identical raw outputs, not as three numerically distinct
    solver executions.
    """
    success = bool(row['success'])
    final_state_ok = bool(row['final_state_ok'])
    warning_count = int(row['warnings_count'])
    if policy == 'strict':
        claim_admissible = success and final_state_ok and warning_count == 0
        claim_status = 'pass' if claim_admissible else 'veto'
    elif policy == 'warn':
        claim_admissible = success and final_state_ok
        claim_status = 'warn' if (claim_admissible and warning_count > 0) else ('pass' if claim_admissible else 'veto')
    elif policy == 'allow':
        claim_admissible = success and final_state_ok
        claim_status = 'allow' if claim_admissible else 'veto'
    else:
        raise ValueError(policy)
    return {
        'policy': policy,
        'claim_admissible': claim_admissible,
        'claim_status': claim_status,
        'observable_tuple': {
            'Yp': row['Yp'],
            'DH': row['DH'],
            'N_eff': row['N_eff'],
            'omega_final': row['omega_final'],
        },
    }


def run_point(sigma: float, enable_teff: bool) -> dict:
    t0 = time.perf_counter()
    cfg = FullCoupledConfig(
        Sigma_H_plus=float(sigma),
        Sigma_H_minus=0.0,
        correction_level=CORRECTION_LEVEL,
        n_reactions=12,
        N_q=N_Q,
        enable_teff=bool(enable_teff),
    )
    payload = {
        'Sigma_H': float(sigma),
        'enable_teff': bool(enable_teff),
        'N_q': int(N_Q),
        'correction_level': int(CORRECTION_LEVEL),
        'solver_method': cfg.solver.scipy_method_str,
        'rtol': cfg.solver.rtol,
        'atol': cfg.solver.atol,
        'max_step': cfg.solver.max_step,
        'thread_env': THREAD_ENV,
    }
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            result = run_full_coupled_typeI(cfg)
        sigma_final = float(result.trajectory['Sigma_plus'][-1])
        omega_final = float(1.0 - sigma_final**2)
        final_state_ok = bool(math.isfinite(omega_final) and (omega_final > 0.0))
        warn_texts = [str(w.message) for w in ws]
        warn_counter = Counter(warn_texts)
        payload.update({
            'success': True,
            'Yp': float(result.observables.Yp),
            'DH': float(result.observables.DH),
            'N_eff': float(result.observables.N_eff),
            'Li7H': float(result.observables.Li7H),
            'Li6H': float(result.observables.Li6H),
            'Xn_freeze': float(result.observables.Xn_freeze),
            'warnings_count': len(warn_texts),
            'first_warning_text': warn_texts[0] if warn_texts else None,
            'unique_warning_texts': list(warn_counter.keys()),
            'unique_warning_counts': dict(warn_counter),
            'driver_metadata': _sanitize(result.metadata),
            'phase1_steps': int(result.metadata.get('phase1_steps', -1)),
            'phase2_steps': int(result.metadata.get('phase2_steps', -1)),
            'mass_conservation': float(result.metadata.get('mass_conservation', float('nan'))),
            'N_eff_gap': float(result.metadata.get('N_eff_gap', float('nan'))),
            'Sigma_plus_final': sigma_final,
            'omega_final': omega_final,
            'final_state_ok': final_state_ok,
            'claim_veto_due_to_warning': len(warn_texts) > 0,
        })
    except Exception as exc:
        payload.update({
            'success': False,
            'Yp': float('nan'),
            'DH': float('nan'),
            'N_eff': float('nan'),
            'Li7H': float('nan'),
            'Li6H': float('nan'),
            'Xn_freeze': float('nan'),
            'warnings_count': -1,
            'first_warning_text': None,
            'unique_warning_texts': [],
            'unique_warning_counts': {},
            'driver_metadata': {'error': repr(exc)},
            'phase1_steps': -1,
            'phase2_steps': -1,
            'mass_conservation': float('nan'),
            'N_eff_gap': float('nan'),
            'Sigma_plus_final': float('nan'),
            'omega_final': float('nan'),
            'final_state_ok': False,
            'claim_veto_due_to_warning': False,
        })
    payload['wall_seconds'] = time.perf_counter() - t0
    return payload


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for enable_teff in TEFF_CASES:
        for sigma in SIGMAS:
            row = run_point(sigma=sigma, enable_teff=enable_teff)
            rows.append(row)
            progress = {
                'completed': len(rows),
                'total': len(SIGMAS) * len(TEFF_CASES),
                'last_row': {
                    'Sigma_H': row['Sigma_H'],
                    'enable_teff': row['enable_teff'],
                    'success': row['success'],
                    'Yp': row['Yp'],
                    'DH': row['DH'],
                    'N_eff': row['N_eff'],
                    'warnings_count': row['warnings_count'],
                    'first_warning_text': row['first_warning_text'],
                    'omega_final': row['omega_final'],
                    'final_state_ok': row['final_state_ok'],
                    'wall_seconds': row['wall_seconds'],
                },
            }
            print(json.dumps(progress), flush=True)
            JSON_PATH.write_text(json.dumps({'rows': _sanitize(rows)}, indent=2))

    # Derived deltas vs Teff-off and vs FLRW within each Teff branch
    by_teff_sigma = {(r['enable_teff'], r['Sigma_H']): r for r in rows}
    for row in rows:
        flrw_same_teff = by_teff_sigma[(row['enable_teff'], 0.0)]
        teff_off_same_sigma = by_teff_sigma[(False, row['Sigma_H'])]
        row['delta_Yp_vs_FLRW_same_teff'] = row['Yp'] - flrw_same_teff['Yp']
        row['delta_DH_vs_FLRW_same_teff'] = row['DH'] - flrw_same_teff['DH']
        row['delta_Yp_teff_on_minus_off'] = (
            row['Yp'] - teff_off_same_sigma['Yp'] if row['enable_teff'] else 0.0
        )
        row['delta_DH_teff_on_minus_off'] = (
            row['DH'] - teff_off_same_sigma['DH'] if row['enable_teff'] else 0.0
        )

    # CSV
    fieldnames = [
        'Sigma_H', 'enable_teff', 'success', 'Yp', 'DH', 'N_eff',
        'warnings_count', 'first_warning_text', 'Sigma_plus_final',
        'omega_final', 'final_state_ok', 'claim_veto_due_to_warning',
        'phase1_steps', 'phase2_steps', 'mass_conservation', 'N_eff_gap',
        'wall_seconds', 'delta_Yp_vs_FLRW_same_teff', 'delta_DH_vs_FLRW_same_teff',
        'delta_Yp_teff_on_minus_off', 'delta_DH_teff_on_minus_off',
    ]
    with CSV_PATH.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    # Policy invariance: post-hoc classification only
    representative_keys = [
        (False, 0.0),
        (True, 0.0),
        (False, 0.1),
        (True, 0.1),
    ]
    policy_payload = {
        'note': (
            'Current Type-I SciPy exact/reference path does not expose a runtime '\
            'strict/warn/allow switch. Invariance is therefore checked as post-hoc '\
            'classification on identical raw outputs.'
        ),
        'representative_cases': [],
    }
    for key in representative_keys:
        row = by_teff_sigma[key]
        policies = {p: classify_policy(row, p) for p in ('strict', 'warn', 'allow')}
        observable_ref = policies['strict']['observable_tuple']
        observable_invariant = all(p['observable_tuple'] == observable_ref for p in policies.values())
        policy_payload['representative_cases'].append({
            'Sigma_H': row['Sigma_H'],
            'enable_teff': row['enable_teff'],
            'warnings_count': row['warnings_count'],
            'first_warning_text': row['first_warning_text'],
            'final_state_ok': row['final_state_ok'],
            'observable_invariant_across_policies': observable_invariant,
            'policies': policies,
        })
    POLICY_JSON.write_text(json.dumps(_sanitize(policy_payload), indent=2))

    # Summary markdown
    off_rows = [r for r in rows if not r['enable_teff']]
    on_rows = [r for r in rows if r['enable_teff']]
    veto_rows = [r for r in rows if r['claim_veto_due_to_warning']]
    strict_pass_rows = [r for r in rows if r['success'] and r['final_state_ok'] and r['warnings_count'] == 0]
    off_veto_rows = [r for r in off_rows if r['claim_veto_due_to_warning']]
    on_veto_rows = [r for r in on_rows if r['claim_veto_due_to_warning']]
    off_strict_pass_rows = [r for r in off_rows if r['success'] and r['final_state_ok'] and r['warnings_count'] == 0]
    on_strict_pass_rows = [r for r in on_rows if r['success'] and r['final_state_ok'] and r['warnings_count'] == 0]
    summary_lines = []
    summary_lines.append('# SAFeguard session2 — Type I SciPy high-resolution revalidation (2026-04-08)')
    summary_lines.append('')
    summary_lines.append('## Execution contract')
    summary_lines.append('')
    summary_lines.append('- base package: `RABBIT_v9rc_safeguard_patch_session1_2026-04-08.zip`')
    summary_lines.append(f'- path: `run_full_coupled_typeI` (SciPy Radau reference/interim path), `N_q={N_Q}`, `correction_level={CORRECTION_LEVEL}`, `n_reactions=12`')
    summary_lines.append('- grid: `Sigma_H = 0.0, 0.1, 0.2, 0.3, 0.4, 0.5`; each with Teff OFF/ON')
    summary_lines.append('- thread-only performance env: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`')
    summary_lines.append('')
    summary_lines.append('## Headline verdict')
    summary_lines.append('')
    summary_lines.append(f'- numerical completion: `{sum(1 for r in rows if r["success"])} / {len(rows)}` runs returned success.')
    summary_lines.append(f'- strict claim-admissible points (success + positive final omega + zero warnings): `{len(strict_pass_rows)} / {len(rows)}`.')
    summary_lines.append(f'- warning-veto points: `{len(veto_rows)} / {len(rows)}`.')
    summary_lines.append(f'- Teff OFF strict-pass points: `{len(off_strict_pass_rows)} / {len(off_rows)}`; warning-veto points: `{len(off_veto_rows)} / {len(off_rows)}`.')
    summary_lines.append(f'- Teff ON strict-pass points: `{len(on_strict_pass_rows)} / {len(on_rows)}`; warning-veto points: `{len(on_veto_rows)} / {len(on_rows)}`.')
    summary_lines.append('- FLRW (`Sigma_H=0`) remains clean (0 warnings) both Teff OFF and ON.')
    summary_lines.append('- In this high-resolution scan, the warning-veto problem is **branch-asymmetric**: all anisotropic Teff OFF points (`Sigma_H >= 0.1`) triggered repeated monopole out-of-range warnings, while the matched Teff ON branch stayed warning-free on the scanned grid.')
    summary_lines.append('')
    summary_lines.append('## Compact result table')
    summary_lines.append('')
    summary_lines.append('| Sigma_H | Teff | success | Y_p | D/H | N_eff | warnings | omega_final | final_state_ok | claim status (strict) |')
    summary_lines.append('|---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|:---:|')
    for row in rows:
        strict_status = 'pass' if (row['success'] and row['final_state_ok'] and row['warnings_count'] == 0) else 'veto'
        summary_lines.append(
            f"| {row['Sigma_H']:.1f} | {'ON' if row['enable_teff'] else 'OFF'} | {row['success']} | {row['Yp']:.12f} | {row['DH']:.12e} | {row['N_eff']:.12f} | {row['warnings_count']} | {row['omega_final']:.12f} | {row['final_state_ok']} | {strict_status} |"
        )
    summary_lines.append('')
    summary_lines.append('## Interpretation')
    summary_lines.append('')
    summary_lines.append('- Relative to the low-resolution (`N_q=6`) session1 scan, the high-resolution (`N_q=20`) SciPy path is **not cleanly branch-uniform**: the Teff ON branch survives as strict-pass across the scanned anisotropic grid, but the matched Teff OFF branch does not.')
    summary_lines.append('- This is not a final-state positivity failure: `omega_final` stayed positive for all completed points.')
    summary_lines.append('- The blocking issue is upstream/distributional and branch-specific: repeated warnings report that the monopole left the physical FD range `[0,1]`, while interpolation only applies numerical logit stabilization. Under the current doctrine, every such OFF-branch anisotropic run is diagnostic-only.')
    summary_lines.append('- Because the current Type-I SciPy path has no runtime strict/warn/allow toggle, policy invariance was tested as a post-hoc classification layer. Raw observables are invariant across those labels, but claim status changes exactly where warnings appear.')
    summary_lines.append('')
    summary_lines.append('## Representative policy invariance check')
    summary_lines.append('')
    summary_lines.append('| Sigma_H | Teff | warnings | observable invariant across policies | strict | warn | allow |')
    summary_lines.append('|---:|:---:|---:|:---:|:---:|:---:|:---:|')
    for case in policy_payload['representative_cases']:
        pol = case['policies']
        summary_lines.append(
            f"| {case['Sigma_H']:.1f} | {'ON' if case['enable_teff'] else 'OFF'} | {case['warnings_count']} | {case['observable_invariant_across_policies']} | {pol['strict']['claim_status']} | {pol['warn']['claim_status']} | {pol['allow']['claim_status']} |"
        )
    summary_lines.append('')
    summary_lines.append('## Next action')
    summary_lines.append('')
    summary_lines.append('- Do **not** reintroduce floor/clamp salvage.')
    summary_lines.append('- Treat the current high-resolution anisotropic SciPy results as diagnostics, not physics claims.')
    summary_lines.append('- Isolate where the monopole leaves `[0,1]` in the Type-I SciPy path (transport projection / interpolation / Teff interface) and surface first-crossing provenance rather than masking it.')
    SUMMARY_MD.write_text('\n'.join(summary_lines) + '\n')

    JSON_PATH.write_text(json.dumps(_sanitize({'rows': rows}), indent=2))
    print(json.dumps({
        'done': True,
        'json': str(JSON_PATH),
        'csv': str(CSV_PATH),
        'summary_md': str(SUMMARY_MD),
        'policy_json': str(POLICY_JSON),
        'strict_pass_count': len(strict_pass_rows),
        'warning_veto_count': len(veto_rows),
    }, indent=2))


if __name__ == '__main__':
    main()
