#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from rabbit.inference.bbn_inference import _scipy_forward_solve

YP_OBS = 0.2449
YP_ERR = 0.004
DH_OBS = 2.547e-5
DH_ERR = 0.029e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigma', type=float, required=True)
    ap.add_argument('--eta10', type=float, required=True)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    r = _scipy_forward_solve(args.sigma, args.eta10 * 1e-10, 878.4, 20, 0)
    out = {'sigma': args.sigma, 'eta10': args.eta10, 'success': bool(r['success'])}
    if r['success']:
        out['Yp'] = float(r['Yp'])
        out['DH'] = float(r['DH'])
        out['chi2'] = float(((r['Yp'] - YP_OBS)/YP_ERR)**2 + ((r['DH'] - DH_OBS)/DH_ERR)**2)
    else:
        out['chi2'] = 100.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    print(out_path)


if __name__ == '__main__':
    main()
