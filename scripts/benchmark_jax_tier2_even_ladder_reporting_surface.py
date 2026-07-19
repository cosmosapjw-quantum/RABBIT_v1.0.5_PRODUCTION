from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_reporting_surface(pr9: dict) -> dict:
    rows = []
    for row in pr9['rows']:
        rows.append({
            'Sigma_H': float(row['Sigma_H']),
            'lmax': int(row['lmax']),
            'N_q': int(row['N_q']),
            'phase1_handoff_N_synthetic': bool(row['phase1_handoff_N_synthetic']),
            'phase1_handoff_source': row['phase1_handoff_source'],
            'Y_p': float(row['Y_p_readout_thermal_plus_freezeout']),
            'N_eff': float(row['N_eff_readout_thermal_plus_freezeout']),
            'DH': float(row['DH_readout_thermal_plus_freezeout']),
            'phase1_handoff_lambda_np': float(row['phase1_handoff_lambda_np']),
            'phase1_handoff_lambda_pn': float(row['phase1_handoff_lambda_pn']),
            'phase1_handoff_Xn': float(row['phase1_handoff_Xn']),
            'honesty_note': row['honesty_note'],
            'deferred_features': list(row['deferred_features']),
            'benchmark_only': True,
            'promoted_to_canonical_dispatch': False,
        })
    return {
        'contract': 'jax_tier2_even_ladder_reporting_surface_snapshot_v1',
        'source_contract': pr9['contract'],
        'selection_contract': pr9['selection_contract'],
        'benchmark_only': True,
        'promoted_to_canonical_dispatch': False,
        'honesty_lock': {
            'phase1_handoff_N_synthetic_must_remain_true': True,
            'benchmark_only_must_remain_true': True,
            'promoted_to_canonical_dispatch_must_remain_false': True,
        },
        'recommended_paths': {
            'fast_path': 'ordered_stageA',
            'comparison_path': 'finite_even_ladder_L6',
            'reference_path': 'finite_even_ladder_L8',
        },
        'baseline': pr9['baseline'],
        'reduced_flrw_baseline': pr9['reduced_flrw_baseline'],
        'rows': rows,
        'comparisons': pr9['comparisons'],
        'note': (
            'Canonical reporting surface snapshot only. This repackages the PR9 reduced-control readout matrix into '
            'a stable reporting schema while preserving all honesty flags. It does not promote the proxy handoff to '
            'canonical inference dispatch.'
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    src = json.loads(Path(args.input).read_text())
    out = build_reporting_surface(src)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
