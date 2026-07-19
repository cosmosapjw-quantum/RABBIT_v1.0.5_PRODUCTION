#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def to_py(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return [to_py(v) for v in x]
    if isinstance(x, dict):
        return {k: to_py(v) for k, v in x.items()}
    return x


def weighted_inner(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * a * b))


def weighted_norm(v: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, weighted_inner(v, v, w))))


def weighted_qr(vectors: Sequence[np.ndarray], w: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    basis: List[np.ndarray] = []
    for vec in vectors:
        vv = np.array(vec, dtype=np.float64, copy=True)
        for bb in basis:
            vv -= weighted_inner(vv, bb, w) * bb
        n = weighted_norm(vv, w)
        if n > eps:
            basis.append(vv / n)
    if not basis:
        return np.zeros((0, len(w)), dtype=np.float64)
    return np.vstack(basis)


def weighted_svd_modes(matrix: np.ndarray, row_weights: np.ndarray, quad_weights: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return np.zeros((0, matrix.shape[1]), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    rw = np.sqrt(np.asarray(row_weights, dtype=np.float64))[:, None]
    qw = np.sqrt(np.asarray(quad_weights, dtype=np.float64))[None, :]
    A = rw * matrix * qw
    _, S, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(rank, Vt.shape[0])
    modes = Vt[:k] / np.sqrt(np.asarray(quad_weights, dtype=np.float64))[None, :]
    modes = weighted_qr(modes, quad_weights)
    return modes, S[: len(modes)]


def load_geometry_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description='Build branch-conditioned macro banks from geometry-macro rows.')
    ap.add_argument('inputs', nargs='+', help='geometry_macro JSON files')
    ap.add_argument('--rank', type=int, default=2)
    ap.add_argument('--species', nargs='*', default=['nue', 'nux'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    q_nodes = None
    q_weights = None
    grouped: Dict[str, Dict[str, List[np.ndarray]]] = {}
    meta_rows: List[Dict[str, Any]] = []

    for inp in args.inputs:
        data = load_geometry_json(Path(inp))
        qn = np.asarray(data['q_nodes'], dtype=np.float64)
        qw = np.asarray(data['q_weights'], dtype=np.float64)
        if q_nodes is None:
            q_nodes = qn
            q_weights = qw
        else:
            if len(qn) != len(q_nodes) or np.max(np.abs(qn - q_nodes)) > 1e-12:
                raise ValueError('q grid mismatch across geometry_macro files')
            if len(qw) != len(q_weights) or np.max(np.abs(qw - q_weights)) > 1e-12:
                raise ValueError('q weights mismatch across geometry_macro files')

        for row in data['rows']:
            br = str(row.get('branch_label_local', 'unknown'))
            if br not in grouped:
                grouped[br] = {sp: [] for sp in args.species}
            for sp in args.species:
                srow = row['species'][sp]
                grouped[br][sp].append(np.asarray(srow['orth_profile'], dtype=np.float64))
            meta_rows.append({
                'source': str(inp),
                'index': int(row['index']),
                'branch_label_local': br,
                'sigma_plus': float(row['sigma_plus']),
                'pi_plus_total': float(row['pi_plus_total']),
            })

    if q_nodes is None or q_weights is None:
        raise RuntimeError('No geometry_macro rows loaded.')

    out_modes: Dict[str, Dict[str, Any]] = {}
    for br, payload in grouped.items():
        out_modes[br] = {}
        for sp, vecs in payload.items():
            if not vecs:
                out_modes[br][sp] = {'modes': [], 'singular_values': []}
                continue
            mat = np.vstack(vecs)
            row_weights = np.ones(mat.shape[0], dtype=np.float64)
            modes, sv = weighted_svd_modes(mat, row_weights, q_weights, args.rank)
            out_modes[br][sp] = {'modes': modes, 'singular_values': sv}

    out = {
        'q_nodes': q_nodes,
        'q_weights': q_weights,
        'branch_banks': out_modes,
        'rows': meta_rows,
        'config': {'rank': args.rank, 'species': args.species},
    }

    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(to_py(out), indent=2))
    print(json.dumps({'out': str(pp.resolve()), 'n_rows': len(meta_rows)}, indent=2))


if __name__ == '__main__':
    main()
