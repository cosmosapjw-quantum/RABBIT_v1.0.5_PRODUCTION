from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rabbit.transport.even_ladder_observable_bridge import (  # noqa: E402
    build_snapshot_row,
    compare_snapshot_to_reference,
    fit_small_shear_observable_diffs,
)

SIGMAS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 0.5]
LMAXS = [2, 4, 6, 8]
NQS = [20, 40]
TEMPS = [1.0, 0.8, 0.6]
N_END = 0.5


def main() -> None:
    out_dir = ROOT / 'session11_outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []
    references: dict[tuple[int, float, float], dict[str, float | int]] = {}
    for n_q in NQS:
        for sigma in SIGMAS:
            for temp in TEMPS:
                for lmax in LMAXS:
                    row = build_snapshot_row(lmax=lmax, sigma_h=sigma, n_q=n_q, T_gamma_MeV=temp, n_end=N_END)
                    rows.append(row)
                    if lmax == 8:
                        references[(n_q, sigma, temp)] = row

    comparisons: list[dict[str, float | int]] = []
    for row in rows:
        ref = references[(int(row['N_q']), float(row['Sigma_H']), float(row['T_gamma_MeV']))]
        comparison = dict(row)
        comparison.update(compare_snapshot_to_reference(row, ref))
        comparisons.append(comparison)

    fits: list[dict[str, float | int | str]] = []
    for n_q in NQS:
        for temp in TEMPS:
            for lmax in (2, 4, 6):
                fits.extend(fit_small_shear_observable_diffs(comparisons, n_q=n_q, T_gamma_MeV=temp, lmax=lmax, sigma_max=0.1))

    payload = {
        'meta': {
            'sigmas': SIGMAS,
            'lmaxs': LMAXS,
            'N_qs': NQS,
            'T_gamma_MeV': TEMPS,
            'N_end': N_END,
            'observable_bridge': 'handoff-level live weak-rate / N_eff-like / Xn_eq_proxy / Yp_proxy',
            'notes': [
                'This is a snapshot bridge at fixed thermal points, not a full phase-1+phase-2 BBN integration.',
                'D/H is intentionally omitted here because network-level propagation requires the full abundance ODE.',
                'The same F0(q) is used for nu_e and anti-nu_e to isolate lmax truncation effects in the monopole sector.',
            ],
        },
        'rows': rows,
        'comparisons': comparisons,
        'fits': fits,
    }

    json_path = out_dir / 'even_ladder_observable_bridge_pr6_2026-04-09.json'
    json_path.write_text(json.dumps(payload, indent=2))

    def write_csv(path: Path, data: list[dict]) -> None:
        if not data:
            return
        fieldnames = list(data[0].keys())
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    write_csv(out_dir / 'even_ladder_observable_bridge_pr6_rows_2026-04-09.csv', rows)
    write_csv(out_dir / 'even_ladder_observable_bridge_pr6_comparisons_2026-04-09.csv', comparisons)
    write_csv(out_dir / 'even_ladder_observable_bridge_pr6_fits_2026-04-09.csv', fits)

    print(f'wrote {json_path}')


if __name__ == '__main__':
    main()
