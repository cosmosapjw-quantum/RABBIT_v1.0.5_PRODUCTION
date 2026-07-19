#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


def pick(d: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in d:
            return d[name]
    return default


def require_array(d: Dict[str, Any], *names: str) -> np.ndarray:
    for name in names:
        if name in d:
            return np.asarray(d[name], dtype=np.float64)
    for container_key in ('profiles', 'arrays', 'state', 'payload'):
        c = d.get(container_key, {})
        if isinstance(c, dict):
            for name in names:
                if name in c:
                    return np.asarray(c[name], dtype=np.float64)
    raise KeyError(f'None of keys {names} found.')


def require_scalar(d: Dict[str, Any], *names: str, default: float = math.nan) -> float:
    for name in names:
        if name in d and d[name] is not None:
            return float(d[name])
    for container_key in ('scalars', 'state', 'payload'):
        c = d.get(container_key, {})
        if isinstance(c, dict):
            for name in names:
                if name in c and c[name] is not None:
                    return float(c[name])
    return float(default)


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


def project(v: np.ndarray, basis: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if basis.size == 0:
        return np.zeros_like(v), np.zeros((0,), dtype=np.float64)
    coeffs = np.asarray([weighted_inner(v, bb, w) for bb in basis], dtype=np.float64)
    recon = np.tensordot(coeffs, basis, axes=(0, 0))
    return recon, coeffs


def load_bank(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    modes = data['modes']
    return {
        'q_nodes': np.asarray(data['q_nodes'], dtype=np.float64),
        'q_weights': np.asarray(data['q_weights'], dtype=np.float64),
        'fd2': np.asarray(modes['fd2'], dtype=np.float64),
        'shared': np.asarray(modes.get('shared_windowed', []), dtype=np.float64),
        'species': {k: np.asarray(v, dtype=np.float64) for k, v in modes.get('species_windowed', {}).items()},
    }


def load_dump_states(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    for key in ('accepted_states', 'states', 'rows'):
        if key in data and isinstance(data[key], list):
            return list(data[key])
    if isinstance(data, list):
        return list(data)
    raise KeyError('Could not find accepted-state list in dump JSON.')


def enrich_state(state: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return {
        'index': idx,
        'N': require_scalar(state, 'N', 'n', 'efolds'),
        'sigma_plus': require_scalar(state, 'sigma_plus', 'Sigma_H', 'Sigma_H_plus', 'phase1_handoff_sigma_plus'),
        'sigma_minus': require_scalar(state, 'sigma_minus', 'Sigma_H_minus', 'phase1_handoff_sigma_minus'),
        'T_gamma': require_scalar(state, 'T_gamma', 'Tg', 'phase1_handoff_T_gamma', 'phase1_handoff_T'),
        'T_nu_e': require_scalar(state, 'T_nu_e', 'Tne', 'phase1_handoff_T_nu_e'),
        'T_nu_x': require_scalar(state, 'T_nu_x', 'Tnx', 'phase1_handoff_T_nu_x'),
        'Xn': require_scalar(state, 'Xn', 'phase1_handoff_Xn'),
        'pi_plus_total': require_scalar(state, 'pi_plus_total', 'phase1_handoff_pi_plus_total'),
        'raw_state': state,
    }


def load_micro_rows(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return []
    data = json.loads(path.read_text())
    for key in ('rows', 'states', 'accepted_states'):
        if key in data and isinstance(data[key], list):
            return list(data[key])
    if isinstance(data, list):
        return list(data)
    raise KeyError('Could not find row list in micro JSON.')


def species_payload(row: Dict[str, Any], species: str) -> Dict[str, Any]:
    if 'species' in row and isinstance(row['species'], dict) and species in row['species']:
        return row['species'][species]
    if species in row and isinstance(row[species], dict):
        return row[species]
    return row


def get_profile_and_scalars(row: Dict[str, Any], species: str) -> Dict[str, Any]:
    sp = species_payload(row, species)
    q_nodes = None
    q_weights = None
    for source in (sp, row):
        try:
            q_nodes = require_array(source, 'q_nodes', 'q_gl', 'q', 'momentum_nodes')
            q_weights = require_array(source, 'q_weights', 'q_wt', 'q_wgl', 'momentum_weights')
            break
        except Exception:
            continue
    if q_nodes is None or q_weights is None:
        raise KeyError(f'q grid not found for species={species}')

    orth = require_array(sp, 'orth_profile', 'orth', 'profile_orth')
    raw = require_array(sp, 'raw_profile', 'raw', 'profile_raw')
    proj = require_array(sp, 'proj_profile', 'proj', 'profile_proj')
    return {
        'q_nodes': q_nodes,
        'q_weights': q_weights,
        'raw': raw,
        'proj': proj,
        'orth': orth,
        'raw_qdot': require_scalar(sp, 'raw_qdot'),
        'proj_qdot': require_scalar(sp, 'proj_qdot'),
        'orth_qdot': require_scalar(sp, 'orth_qdot'),
        'pi_plus': require_scalar(sp, 'pi_plus', 'pi_plus_species'),
    }


def local_branch_label(sigma_plus: float, pi_plus_total: float, qdot_raw: float, qdot_proj: float) -> str:
    if np.isfinite(sigma_plus) and np.isfinite(pi_plus_total) and sigma_plus != 0.0 and pi_plus_total != 0.0:
        return 'pos' if sigma_plus * pi_plus_total > 0.0 else 'neg'
    if np.isfinite(qdot_raw) and np.isfinite(qdot_proj) and qdot_raw != 0.0 and qdot_proj != 0.0:
        return 'aligned' if qdot_raw * qdot_proj > 0.0 else 'mismatch'
    return 'unknown'


def main() -> None:
    ap = argparse.ArgumentParser(description='Extract geometry-macro coupling rows from replay dumps and/or micromacro rows.')
    ap.add_argument('--dump', required=True, help='Replay dump JSON with accepted states.')
    ap.add_argument('--bank', required=True, help='Mode bank JSON.')
    ap.add_argument('--micro', default=None, help='Optional micromacro JSON with per-state species profiles.')
    ap.add_argument('--species', nargs='*', default=['nue', 'nux'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    bank = load_bank(Path(args.bank))
    q_nodes_ref = bank['q_nodes']
    q_weights_ref = bank['q_weights']

    dump_states = [enrich_state(s, i) for i, s in enumerate(load_dump_states(Path(args.dump)))]
    micro_rows = load_micro_rows(Path(args.micro)) if args.micro else []
    n = min(len(dump_states), len(micro_rows)) if micro_rows else len(dump_states)
    if n < 2:
        raise RuntimeError('Need at least two states to compute next-step deltas.')

    rows_out: List[Dict[str, Any]] = []
    for i in range(n - 1):
        st = dump_states[i]
        st_next = dump_states[i + 1]
        row_micro = micro_rows[i] if micro_rows else {}
        species_rows: Dict[str, Any] = {}
        pi_sum = 0.0

        for sp in args.species:
            prof = get_profile_and_scalars(row_micro, sp) if micro_rows else None
            if prof is None:
                raise RuntimeError('Standalone dump-only mode is not implemented in this version; pass --micro.')

            q_nodes = prof['q_nodes']
            q_weights = prof['q_weights']
            if len(q_nodes) != len(q_nodes_ref) or np.max(np.abs(q_nodes - q_nodes_ref)) > 1e-12:
                raise ValueError(f'q grid mismatch for species={sp}, state={i}')
            if len(q_weights) != len(q_weights_ref) or np.max(np.abs(q_weights - q_weights_ref)) > 1e-12:
                raise ValueError(f'q weight mismatch for species={sp}, state={i}')

            fd2_basis = weighted_qr([bank['fd2']], q_weights_ref)
            shared_basis = weighted_qr(bank['shared'][:2], q_weights_ref) if bank['shared'].size else np.zeros((0, len(q_weights_ref)))
            species_basis = weighted_qr(bank['species'].get(sp, np.zeros((0, len(q_weights_ref))))[:2], q_weights_ref)

            recon_fd2, coeff_fd2 = project(prof['orth'], fd2_basis, q_weights_ref)
            recon_shared, coeff_shared = project(prof['orth'], shared_basis, q_weights_ref)
            recon_species, coeff_species = project(prof['orth'], species_basis, q_weights_ref)

            pi_plus = prof['pi_plus']
            if np.isfinite(pi_plus):
                pi_sum += pi_plus

            species_rows[sp] = {
                'raw_profile': prof['raw'],
                'proj_profile': prof['proj'],
                'orth_profile': prof['orth'],
                'raw_qdot': prof['raw_qdot'],
                'proj_qdot': prof['proj_qdot'],
                'orth_qdot': prof['orth_qdot'],
                'pi_plus': pi_plus,
                'coefficients': {
                    'a_fd2': coeff_fd2.tolist(),
                    'b_shared': coeff_shared.tolist(),
                    'c_species': coeff_species.tolist(),
                },
                'reconstruction_norms': {
                    'fd2': weighted_norm(recon_fd2, q_weights_ref),
                    'shared': weighted_norm(recon_shared, q_weights_ref),
                    'species': weighted_norm(recon_species, q_weights_ref),
                    'orth': weighted_norm(prof['orth'], q_weights_ref),
                },
            }

        pi_here = st['pi_plus_total'] if np.isfinite(st['pi_plus_total']) else pi_sum
        pi_next = dump_states[i + 1]['pi_plus_total']
        if not np.isfinite(pi_next) and i + 1 < len(micro_rows):
            pi_next = 0.0
            for sp in args.species:
                try:
                    pi_sp = get_profile_and_scalars(micro_rows[i + 1], sp)['pi_plus']
                    if np.isfinite(pi_sp):
                        pi_next += pi_sp
                except Exception:
                    pass

        row_out = {
            'index': i,
            'N': st['N'],
            'sigma_plus': st['sigma_plus'],
            'sigma_minus': st['sigma_minus'],
            'pi_plus_total': pi_here,
            'd_sigma_next': st_next['sigma_plus'] - st['sigma_plus'] if np.isfinite(st_next['sigma_plus']) and np.isfinite(st['sigma_plus']) else math.nan,
            'd_pi_next': pi_next - pi_here if np.isfinite(pi_next) and np.isfinite(pi_here) else math.nan,
            'T_gamma': st['T_gamma'],
            'T_nu_e': st['T_nu_e'],
            'T_nu_x': st['T_nu_x'],
            'Xn': st['Xn'],
            'species': species_rows,
            'branch_label_local': local_branch_label(
                st['sigma_plus'],
                pi_here,
                species_rows[args.species[0]]['raw_qdot'],
                species_rows[args.species[0]]['proj_qdot'],
            ),
        }
        rows_out.append(row_out)

    out = {
        'q_nodes': q_nodes_ref,
        'q_weights': q_weights_ref,
        'rows': rows_out,
    }

    pp = Path(args.out)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(to_py(out), indent=2))
    print(json.dumps({'out': str(pp.resolve()), 'n_rows': len(rows_out)}, indent=2))


if __name__ == '__main__':
    main()
