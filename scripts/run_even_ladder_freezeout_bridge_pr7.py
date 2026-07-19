from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import csv
import json
from pathlib import Path

from rabbit.transport.even_ladder_freezeout_bridge import (
    compare_freezeout_row_to_reference,
    fit_small_shear_freezeout_diffs,
    freezeout_bridge_row,
)


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    outdir = base / 'session12_outputs'
    outdir.mkdir(parents=True, exist_ok=True)

    sigmas = [0.0, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 0.3, 0.5]
    nqs = [20, 40]
    lmaxs = [2, 4, 6, 8]

    rows: list[dict] = []
    for nq in nqs:
        for sigma in sigmas:
            for lmax in lmaxs:
                row = freezeout_bridge_row(
                    lmax=lmax,
                    sigma_h=sigma,
                    n_q=nq,
                    T_start_MeV=2.0,
                    T_handoff_MeV=0.08,
                    T_decay_final_MeV=0.06,
                    n_T=96,
                )
                rows.append(row)

    comparisons: list[dict] = []
    for nq in nqs:
        for sigma in sigmas:
            ref = next(r for r in rows if r['N_q'] == nq and r['Sigma_H'] == sigma and r['lmax'] == 8)
            for lmax in [2, 4, 6]:
                row = next(r for r in rows if r['N_q'] == nq and r['Sigma_H'] == sigma and r['lmax'] == lmax)
                comparisons.append({**row, **compare_freezeout_row_to_reference(row, ref)})

    fits: list[dict] = []
    for nq in nqs:
        for lmax in [2, 4, 6]:
            fits.extend(fit_small_shear_freezeout_diffs(comparisons, n_q=nq, lmax=lmax))

    payload = {'rows': rows, 'comparisons': comparisons, 'fits': fits}
    json_path = outdir / 'even_ladder_freezeout_bridge_pr7_2026-04-09.json'
    json_path.write_text(json.dumps(payload, indent=2))

    def write_csv(path: Path, recs: list[dict]) -> None:
        if not recs:
            path.write_text('')
            return
        keys = list(recs[0].keys())
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(recs)

    write_csv(outdir / 'even_ladder_freezeout_bridge_pr7_rows_2026-04-09.csv', rows)
    write_csv(outdir / 'even_ladder_freezeout_bridge_pr7_comparisons_2026-04-09.csv', comparisons)
    write_csv(outdir / 'even_ladder_freezeout_bridge_pr7_fits_2026-04-09.csv', fits)
    print(f'wrote {json_path}')


if __name__ == '__main__':
    main()
