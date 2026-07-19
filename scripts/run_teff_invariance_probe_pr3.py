from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rabbit.config.grids import MomentumGrid
from rabbit.transport.state import fermi_dirac
from rabbit.transport.typeI_hierarchy import B_QUADRUPOLE_SOURCE
from rabbit.weak.teff_correction import (
    compute_teff_weak_correction,
    teff_quadrupole_theta_bounds,
    teff_tangency_diagnostic,
    fd_bounds_summary,
)

OUTDIR = ROOT / 'session8_outputs'
OUTDIR.mkdir(exist_ok=True)


def run_case(sigma_h: float, n_q: int, n_eval: int = 41, n_end: float = 0.5):
    grid = MomentumGrid(N_q=n_q)
    f0 = fermi_dirac(grid.nodes)
    N_values = np.linspace(0.0, n_end, n_eval)

    rows = []
    first_raw_violation_N = None
    first_teff_violation_N = None
    first_theta_nonpositive_N = None
    max_dres = 0.0
    max_abs_pi = 0.0
    warning_count = 0
    first_warning = None

    for N in N_values:
        # Legacy exact collisionless Type-I semantics:
        #   Psi0(q,N)=0,
        #   Psi2(q,N)= - (8/15) Sigma_H N  (q-independent)
        pi_nue = float(-B_QUADRUPOLE_SOURCE * sigma_h * N)
        pi_nuebar = pi_nue

        raw_nue = f0.copy()
        raw_nuebar = f0.copy()
        raw_summary_nue = fd_bounds_summary(raw_nue)
        raw_summary_nuebar = fd_bounds_summary(raw_nuebar)

        theta_min_nue, theta_max_nue = teff_quadrupole_theta_bounds(pi_nue)
        theta_min_nuebar, theta_max_nuebar = teff_quadrupole_theta_bounds(pi_nuebar)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            corr_nue, corr_nuebar, diag = compute_teff_weak_correction(
                raw_nue, raw_nuebar, grid.nodes,
                pi_tilde_nue=pi_nue,
                pi_tilde_nuebar=pi_nuebar,
                method='exact', N_mu=16,
            )
        if caught:
            warning_count += len(caught)
            if first_warning is None:
                first_warning = str(caught[0].message)

        corr_summary_nue = fd_bounds_summary(corr_nue)
        corr_summary_nuebar = fd_bounds_summary(corr_nuebar)

        dres_nue = float(teff_tangency_diagnostic(raw_nue, grid.nodes, pi_nue))
        dres_nuebar = float(teff_tangency_diagnostic(raw_nuebar, grid.nodes, pi_nuebar))
        max_dres = max(max_dres, dres_nue, dres_nuebar)
        max_abs_pi = max(max_abs_pi, abs(pi_nue), abs(pi_nuebar))

        raw_ok = raw_summary_nue['is_fd_admissible'] and raw_summary_nuebar['is_fd_admissible']
        teff_ok = corr_summary_nue['is_fd_admissible'] and corr_summary_nuebar['is_fd_admissible']
        theta_ok = (theta_min_nue > 0.0) and (theta_min_nuebar > 0.0)

        if first_raw_violation_N is None and not raw_ok:
            first_raw_violation_N = float(N)
        if first_teff_violation_N is None and not teff_ok:
            first_teff_violation_N = float(N)
        if first_theta_nonpositive_N is None and not theta_ok:
            first_theta_nonpositive_N = float(N)

        rows.append({
            'N': float(N),
            'Sigma_H': float(sigma_h),
            'N_q': int(n_q),
            'pi_tilde_nue': pi_nue,
            'pi_tilde_nuebar': pi_nuebar,
            'theta_min_nue': theta_min_nue,
            'theta_max_nue': theta_max_nue,
            'theta_min_nuebar': theta_min_nuebar,
            'theta_max_nuebar': theta_max_nuebar,
            'raw_nue_min': float(raw_summary_nue['min']),
            'raw_nue_max': float(raw_summary_nue['max']),
            'raw_nuebar_min': float(raw_summary_nuebar['min']),
            'raw_nuebar_max': float(raw_summary_nuebar['max']),
            'teff_nue_min': float(corr_summary_nue['min']),
            'teff_nue_max': float(corr_summary_nue['max']),
            'teff_nuebar_min': float(corr_summary_nuebar['min']),
            'teff_nuebar_max': float(corr_summary_nuebar['max']),
            'raw_fd_ok': bool(raw_ok),
            'teff_fd_ok': bool(teff_ok),
            'theta_positive': bool(theta_ok),
            'Dres_nue': dres_nue,
            'Dres_nuebar': dres_nuebar,
            'Sigma2_nue': float(diag['Sigma2_nue']),
            'Sigma2_nuebar': float(diag['Sigma2_nuebar']),
        })

    return {
        'success': True,
        'message': 'analytic legacy exact collisionless Type-I semantics',
        'Sigma_H': float(sigma_h),
        'N_q': int(n_q),
        'N_end': float(n_end),
        'first_raw_violation_N': first_raw_violation_N,
        'first_teff_violation_N': first_teff_violation_N,
        'first_theta_nonpositive_N': first_theta_nonpositive_N,
        'max_Dres': max_dres,
        'max_abs_pi_tilde': max_abs_pi,
        'warnings_count': warning_count,
        'first_warning': first_warning,
        'rows': rows,
    }


def main():
    cases = [run_case(sigma_h=s, n_q=nq) for nq in (20, 40) for s in (0.0, 0.3, 0.5)]
    out_json = OUTDIR / 'teff_invariance_probe_pr3_2026-04-09.json'
    out_json.write_text(json.dumps({'cases': cases}, indent=2))

    flat_rows = []
    for case in cases:
        flat_rows.append({
            'Sigma_H': case['Sigma_H'],
            'N_q': case['N_q'],
            'success': case['success'],
            'first_raw_violation_N': case['first_raw_violation_N'],
            'first_teff_violation_N': case['first_teff_violation_N'],
            'first_theta_nonpositive_N': case['first_theta_nonpositive_N'],
            'max_Dres': case['max_Dres'],
            'max_abs_pi_tilde': case['max_abs_pi_tilde'],
            'warnings_count': case['warnings_count'],
            'first_warning': case['first_warning'] or '',
        })
    out_csv = OUTDIR / 'teff_invariance_probe_pr3_2026-04-09.csv'
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader(); writer.writerows(flat_rows)
    print(f'Wrote {out_json}')
    print(f'Wrote {out_csv}')


if __name__ == '__main__':
    main()
