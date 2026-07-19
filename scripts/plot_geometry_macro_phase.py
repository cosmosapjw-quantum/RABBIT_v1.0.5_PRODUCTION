#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path, species: str) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = []
    for row in data['rows']:
        srow = row['species'][species]
        coeff = srow.get('coefficients', {})
        b_shared = coeff.get('b_shared', [])
        c_species = coeff.get('c_species', [])
        rows.append({
            'index': int(row['index']),
            'N': float(row['N']),
            'sigma_plus': float(row['sigma_plus']),
            'pi_plus_total': float(row['pi_plus_total']),
            'd_sigma_next': float(row['d_sigma_next']),
            'd_pi_next': float(row['d_pi_next']),
            'branch_label_local': str(row.get('branch_label_local', 'unknown')),
            'species': species,
            'a_fd2': float(coeff.get('a_fd2', [0.0])[0] if coeff.get('a_fd2') else 0.0),
            'b_shared1': float(b_shared[0] if len(b_shared) >= 1 else 0.0),
            'b_shared2': float(b_shared[1] if len(b_shared) >= 2 else 0.0),
            'c_species1': float(c_species[0] if len(c_species) >= 1 else 0.0),
            'qdot_raw': float(srow.get('raw_qdot', np.nan)),
            'qdot_proj': float(srow.get('proj_qdot', np.nan)),
            'qdot_orth': float(srow.get('orth_qdot', np.nan)),
        })
    return rows


def make_xy(rows: Sequence[Dict[str, Any]], xkey: str, ykey: str) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([r[xkey] for r in rows], dtype=np.float64)
    y = np.asarray([r[ykey] for r in rows], dtype=np.float64)
    return x, y


def color_values(rows: Sequence[Dict[str, Any]], color_by: str) -> np.ndarray:
    if color_by == 'branch':
        labels = [r['branch_label_local'] for r in rows]
        uniq = {lab: i for i, lab in enumerate(sorted(set(labels)))}
        return np.asarray([uniq[lab] for lab in labels], dtype=np.float64)
    if color_by == 'sign_mismatch':
        vals = []
        for r in rows:
            raw = r['qdot_raw']
            proj = r['qdot_proj']
            vals.append(1.0 if np.sign(raw) != np.sign(proj) else 0.0)
        return np.asarray(vals, dtype=np.float64)
    if color_by == 'index':
        return np.asarray([r['index'] for r in rows], dtype=np.float64)
    # default: by N
    return np.asarray([r['N'] for r in rows], dtype=np.float64)


def scatter_with_path(rows: Sequence[Dict[str, Any]], xkey: str, ykey: str, color_by: str, title: str, out: Path) -> None:
    x, y = make_xy(rows, xkey, ykey)
    c = color_values(rows, color_by)
    order = np.argsort([r['N'] for r in rows])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(x[order], y[order], alpha=0.45)
    sc = ax.scatter(x, y, c=c)
    ax.set_xlabel(xkey)
    ax.set_ylabel(ykey)
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=color_by)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description='Plot geometry-macro phase portraits.')
    ap.add_argument('inputs', nargs='+', help='geometry_macro JSON files')
    ap.add_argument('--species', nargs='*', default=['nue', 'nux'])
    ap.add_argument('--color-by', default='N', choices=['N', 'index', 'branch', 'sign_mismatch'])
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    combos = [
        ('sigma_plus', 'pi_plus_total'),
        ('sigma_plus', 'b_shared1'),
        ('pi_plus_total', 'b_shared1'),
        ('b_shared1', 'b_shared2'),
    ]

    manifest: List[Dict[str, Any]] = []
    for inp in args.inputs:
        stem = Path(inp).stem
        for sp in args.species:
            rows = load_rows(Path(inp), sp)
            for xkey, ykey in combos:
                out = outdir / f'{stem}_{sp}_{xkey}_vs_{ykey}.png'
                scatter_with_path(rows, xkey, ykey, args.color_by, f'{stem} [{sp}]', out)
                manifest.append({'input': inp, 'species': sp, 'plot': str(out), 'x': xkey, 'y': ykey})

    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'outdir': str(outdir.resolve()), 'n_plots': len(manifest)}, indent=2))


if __name__ == '__main__':
    main()
