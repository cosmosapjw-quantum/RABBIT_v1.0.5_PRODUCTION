from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rabbit.transport.even_ladder_analysis import (  # noqa: E402
    compare_even_ladder_to_reference,
    fit_even_ladder_small_shear,
    integrate_even_ladder_final_state,
    summarize_even_ladder_state,
)


def main() -> None:
    outdir = ROOT / 'session10_outputs'
    outdir.mkdir(parents=True, exist_ok=True)

    sigmas = [0.0, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 0.3, 0.5]
    nqs = [20, 40]
    lmaxs = [2, 4, 6, 8]
    n_end = 0.5

    rows: list[dict[str, float | int]] = []
    references: dict[tuple[int, float], object] = {}

    for n_q in nqs:
        for sigma in sigmas:
            ref_state = integrate_even_ladder_final_state(8, sigma, n_q, n_end=n_end)
            references[(n_q, sigma)] = ref_state
            for lmax in lmaxs:
                state = ref_state if lmax == 8 else integrate_even_ladder_final_state(lmax, sigma, n_q, n_end=n_end)
                row = {
                    'lmax': lmax,
                    'N_q': n_q,
                    'Sigma_H': sigma,
                    'N_end': n_end,
                }
                row.update(summarize_even_ladder_state(state))
                row.update(compare_even_ladder_to_reference(state, ref_state))
                rows.append(row)

    fits: list[dict[str, float | int | str]] = []
    for n_q in nqs:
        for lmax in lmaxs:
            fits.extend(fit_even_ladder_small_shear(rows, n_q=n_q, lmax=lmax, sigma_max=0.1))

    payload = {
        'meta': {
            'sigmas': sigmas,
            'N_q_values': nqs,
            'lmax_values': lmaxs,
            'N_end': n_end,
            'reference_lmax': 8,
            'note': 'finite even-ell ladder with F_{lmax+2}=0 closure; lmax=2 here is not the ordered Stage-A closure',
        },
        'rows': rows,
        'fits': fits,
    }

    json_path = outdir / 'even_ladder_pr5_2026-04-09.json'
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    fieldnames = sorted({k for row in rows for k in row.keys()})
    csv_rows = outdir / 'even_ladder_pr5_rows_2026-04-09.csv'
    with csv_rows.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fieldnames_fits = ['N_q', 'lmax', 'metric', 'slope', 'intercept', 'n_used']
    csv_fits = outdir / 'even_ladder_pr5_fits_2026-04-09.csv'
    with csv_fits.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_fits)
        writer.writeheader()
        writer.writerows(fits)

    print(json.dumps({
        'json': str(json_path),
        'rows_csv': str(csv_rows),
        'fits_csv': str(csv_fits),
        'n_rows': len(rows),
        'n_fits': len(fits),
    }, indent=2))


if __name__ == '__main__':
    main()
