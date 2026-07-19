from __future__ import annotations

import json
from pathlib import Path
from math import isnan

from rabbit.transport.even_ladder_freezeout_bridge import (
    freezeout_bridge_row,
    handoff_metadata_from_freezeout_row,
)


def compare_meta_to_ref(meta: dict, ref: dict) -> dict:
    out = {
        'lmax': int(meta['lmax']),
        'Sigma_H': float(meta['Sigma_H']),
        'N_q': int(meta['N_q']),
    }
    for key in (
        'phase1_handoff_lambda_np',
        'phase1_handoff_lambda_pn',
        'phase1_handoff_Xn',
        'Xn_freeze',
        'N_eff_handoff_proxy',
        'Yp_handoff_proxy',
        'Yp_decay_proxy',
    ):
        out[f'{key}_diff_to_ref_abs'] = abs(float(meta[key]) - float(ref[key]))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text('')
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    lines = [','.join(keys)]
    for r in rows:
        vals = []
        for k in keys:
            v = r.get(k, '')
            if isinstance(v, float):
                vals.append(repr(v))
            else:
                vals.append(str(v))
        lines.append(','.join(vals))
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    outdir = Path('session13_outputs')
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    metadata_rows = []
    comparisons = []
    for n_q in (20, 40):
        for sigma in (0.0, 0.3, 0.5):
            ref_meta = None
            per_sigma = []
            for lmax in (2, 4, 6, 8):
                row = freezeout_bridge_row(
                    lmax=lmax,
                    sigma_h=sigma,
                    n_q=n_q,
                    T_start_MeV=2.0,
                    T_handoff_MeV=0.08,
                    T_decay_final_MeV=0.06,
                )
                meta = handoff_metadata_from_freezeout_row(row)
                rows.append(row)
                metadata_rows.append(meta)
                per_sigma.append(meta)
                if lmax == 8:
                    ref_meta = meta
            assert ref_meta is not None
            for meta in per_sigma:
                if meta['lmax'] == 8:
                    continue
                comparisons.append(compare_meta_to_ref(meta, ref_meta))

    payload = {
        'contract': 'pr8_even_ladder_handoff_metadata_bridge_v1',
        'scope': 'phase1_handoff_metadata_alignment',
        'rows': rows,
        'metadata_rows': metadata_rows,
        'comparisons': comparisons,
    }
    (outdir / 'even_ladder_handoff_metadata_pr8_2026-04-09.json').write_text(json.dumps(payload, indent=2))
    write_csv(outdir / 'even_ladder_handoff_metadata_pr8_rows_2026-04-09.csv', rows)
    write_csv(outdir / 'even_ladder_handoff_metadata_pr8_metadata_rows_2026-04-09.csv', metadata_rows)
    write_csv(outdir / 'even_ladder_handoff_metadata_pr8_comparisons_2026-04-09.csv', comparisons)
    print('Wrote PR8 handoff metadata rows/comparisons.')


if __name__ == '__main__':
    main()
