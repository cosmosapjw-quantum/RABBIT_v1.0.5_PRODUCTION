from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rabbit.transport.stageAB_analysis import (
    compute_stageB_closure_residual,
    fit_power_law,
    integrate_stage_final_state,
    q_weighted_l2,
)
from rabbit.transport.stageAB_state import fermi_dirac

OUTDIR = ROOT / 'session9_outputs'
OUTDIR.mkdir(exist_ok=True)

SIGMA_GRID = [0.0, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 0.3, 0.5]
NQ_GRID = [20, 40]
SMALL_SIGMA_MAX = 0.1


def main() -> None:
    rows = []
    state_cache = {}
    for n_q in NQ_GRID:
        for sigma_h in SIGMA_GRID:
            for stage in ('A', 'B'):
                state = integrate_stage_final_state(stage, sigma_h, n_q)
                state_cache[(stage, n_q, sigma_h)] = state
                grid = state.grid
                fd = fermi_dirac(grid.nodes)
                row = {
                    'stage': stage,
                    'Sigma_H': sigma_h,
                    'N_q': n_q,
                    'F0_L2': q_weighted_l2(state.moment(0, 0), grid),
                    'deltaF0_L2': q_weighted_l2(state.moment(0, 0) - fd, grid),
                    'F2_L2': q_weighted_l2(state.moment(0, 2), grid),
                    'F2_min': float(state.moment(0, 2).min()),
                    'F2_max': float(state.moment(0, 2).max()),
                }
                if stage == 'B':
                    row['F4_L2'] = q_weighted_l2(state.moment(0, 4), grid)
                    resid = compute_stageB_closure_residual(state, sigma_h)
                    row.update({
                        'closure_keep_l2': resid.keep_l2,
                        'closure_omit_self_l2': resid.omit_self_l2,
                        'closure_omit_f4_l2': resid.omit_f4_l2,
                        'closure_omit_total_l2': resid.omit_total_l2,
                        'closure_ratio_self_to_keep': resid.ratio_self_to_keep,
                        'closure_ratio_f4_to_keep': resid.ratio_f4_to_keep,
                        'closure_ratio_total_to_keep': resid.ratio_total_to_keep,
                    })
                rows.append(row)

    comparisons = []
    for n_q in NQ_GRID:
        for sigma_h in SIGMA_GRID:
            state_a = state_cache[('A', n_q, sigma_h)]
            state_b = state_cache[('B', n_q, sigma_h)]
            grid = state_a.grid
            comparisons.append({
                'Sigma_H': sigma_h,
                'N_q': n_q,
                'A_minus_B_deltaF0_L2': q_weighted_l2(state_a.moment(0,0) - state_b.moment(0,0), grid),
                'A_minus_B_F2_L2': q_weighted_l2(state_a.moment(0,2) - state_b.moment(0,2), grid),
                'B_F4_L2': q_weighted_l2(state_b.moment(0,4), grid),
            })

    fits = []
    for n_q in NQ_GRID:
        small_comp = [r for r in comparisons if r['N_q'] == n_q and 0.0 < r['Sigma_H'] <= SMALL_SIGMA_MAX]
        xs = [r['Sigma_H'] for r in small_comp]
        fits.append({'N_q': n_q, 'metric': 'A_minus_B_deltaF0_L2', **fit_power_law(xs, [r['A_minus_B_deltaF0_L2'] for r in small_comp])})
        fits.append({'N_q': n_q, 'metric': 'A_minus_B_F2_L2', **fit_power_law(xs, [r['A_minus_B_F2_L2'] for r in small_comp])})
        fits.append({'N_q': n_q, 'metric': 'B_F4_L2', **fit_power_law(xs, [r['B_F4_L2'] for r in small_comp])})
        stageA_small = [r for r in rows if r['stage']=='A' and r['N_q']==n_q and 0.0 < r['Sigma_H'] <= SMALL_SIGMA_MAX]
        stageB_small = [r for r in rows if r['stage']=='B' and r['N_q']==n_q and 0.0 < r['Sigma_H'] <= SMALL_SIGMA_MAX]
        fits.append({'N_q': n_q, 'metric': 'A_F2_L2', **fit_power_law([r['Sigma_H'] for r in stageA_small], [r['F2_L2'] for r in stageA_small])})
        fits.append({'N_q': n_q, 'metric': 'B_F2_L2', **fit_power_law([r['Sigma_H'] for r in stageB_small], [r['F2_L2'] for r in stageB_small])})
        fits.append({'N_q': n_q, 'metric': 'B_deltaF0_L2', **fit_power_law([r['Sigma_H'] for r in stageB_small], [r['deltaF0_L2'] for r in stageB_small])})
        fits.append({'N_q': n_q, 'metric': 'B_closure_ratio_total_to_keep', **fit_power_law([r['Sigma_H'] for r in stageB_small], [r['closure_ratio_total_to_keep'] for r in stageB_small])})

    payload = {
        'sigma_grid': SIGMA_GRID,
        'nq_grid': NQ_GRID,
        'small_sigma_max': SMALL_SIGMA_MAX,
        'rows': rows,
        'comparisons': comparisons,
        'fits': fits,
    }
    (OUTDIR / 'stageAB_scaling_closure_pr4_2026-04-09.json').write_text(json.dumps(payload, indent=2))
    with (OUTDIR / 'stageAB_scaling_closure_pr4_rows_2026-04-09.csv').open('w', newline='') as f:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    with (OUTDIR / 'stageAB_scaling_closure_pr4_comparisons_2026-04-09.csv').open('w', newline='') as f:
        fieldnames = sorted({k for row in comparisons for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in comparisons:
            w.writerow(row)
    with (OUTDIR / 'stageAB_scaling_closure_pr4_fits_2026-04-09.csv').open('w', newline='') as f:
        fieldnames = ['N_q', 'metric', 'slope', 'intercept', 'n_used']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in fits:
            w.writerow(row)

if __name__ == '__main__':
    main()
