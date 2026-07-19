from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SINGLE = ROOT / 'scripts' / 'audit_physics_p1_policy_drift.py'
SIGMAS_DEFAULT = [0.0, 5.0e-4, 1.0e-3, 1.5e-3, 2.0e-3]
FIELDS = ['Yp','DH','N_eff','Xn_freeze','phase1_handoff_T','phase1_handoff_lambda_np','phase1_handoff_lambda_pn']

def _f(x: Any):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def _run(path_label: str, sigma: float, nq: int, enable_teff: bool, rtol: float, atol: float) -> dict:
    cmd = [sys.executable, str(SINGLE), '--path', path_label, '--sigma-h', repr(float(sigma)), '--nq', str(int(nq)), '--rtol', repr(float(rtol)), '--atol', repr(float(atol))]
    if enable_teff:
        cmd.append('--enable-teff')
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    wall = float(time.perf_counter() - t0)
    if proc.returncode != 0:
        raise RuntimeError(f'path={path_label} sigma={sigma} rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}')
    payload = json.loads(proc.stdout)
    payload['launcher_wall_seconds'] = wall
    return payload

def _delta(c: dict, a: dict) -> dict:
    out = {}
    for k in FIELDS:
        av, cv = _f(a.get(k)), _f(c.get(k))
        out[k] = None if av is None or cv is None else float(cv - av)
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--nq', type=int, default=6)
    ap.add_argument('--enable-teff', action='store_true')
    ap.add_argument('--rtol', type=float, default=1e-8)
    ap.add_argument('--atol', type=float, default=1e-10)
    ap.add_argument('--output', required=True)
    ap.add_argument('--sigmas', type=float, nargs='*', default=SIGMAS_DEFAULT)
    args = ap.parse_args()

    records = []
    for sigma in args.sigmas:
        a = _run('A', sigma, args.nq, bool(args.enable_teff), args.rtol, args.atol)
        c = _run('C', sigma, args.nq, bool(args.enable_teff), args.rtol, args.atol)
        records.append({
            'Sigma_H': float(sigma),
            'A': a,
            'C': c,
            'delta_C_minus_A': _delta(c, a),
            'C_public_requested_mode': c.get('solver_jacobian_public_requested_mode') or c.get('solver_jacobian_requested_mode'),
            'C_dispatch_mode': c.get('solver_jacobian_dispatch_mode'),
            'C_resolved_mode': c.get('solver_jacobian_mode'),
            'C_auto_contract': c.get('solver_jacobian_auto_resolution_contract'),
        })
    adjacent = []
    for left, right in zip(records[:-1], records[1:]):
        obs_jump, drift_jump = {}, {}
        for k in FIELDS:
            lv, rv = _f(left['C'].get(k)), _f(right['C'].get(k))
            obs_jump[k] = None if lv is None or rv is None else float(rv - lv)
            ld, rd = _f(left['delta_C_minus_A'].get(k)), _f(right['delta_C_minus_A'].get(k))
            drift_jump[k] = None if ld is None or rd is None else float(rd - ld)
        adjacent.append({
            'left_sigma': left['Sigma_H'],
            'right_sigma': right['Sigma_H'],
            'left_dispatch': left['C_dispatch_mode'],
            'right_dispatch': right['C_dispatch_mode'],
            'left_resolved_mode': left['C_resolved_mode'],
            'right_resolved_mode': right['C_resolved_mode'],
            'C_observable_jump': obs_jump,
            'delta_jump': drift_jump,
        })
    out = {
        'contract': 'p3_boundary_honesty_v1',
        'nq': int(args.nq),
        'enable_teff': bool(args.enable_teff),
        'rtol': float(args.rtol),
        'atol': float(args.atol),
        'sigma_grid': [float(s) for s in args.sigmas],
        'records': records,
        'adjacent': adjacent,
        'notes': [
            'Boundary honesty asks whether the auto-policy dispatch transition coincides with a spike in C-A drift.',
            'Outside locked scope C should fall back to exact_per_step, shrinking drift to zero or near-zero.',
        ],
    }
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps({'ok': True, 'output': str(outp), 'records': len(records)}))

if __name__ == '__main__':
    main()
