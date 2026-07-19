#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig
from rabbit.config.transport_mode import TransportMode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigma', type=float, required=True)
    ap.add_argument('--nq', type=int, default=20)
    ap.add_argument('--nmu', type=int, default=12)
    ap.add_argument('--mode', choices=['characteristic','linearized'], default='characteristic')
    ap.add_argument('--cl', type=int, default=0)
    ap.add_argument('--teff', action='store_true')
    ap.add_argument('--traj', action='store_true')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    tm = TransportMode.CHARACTERISTIC if args.mode == 'characteristic' else TransportMode.LINEARIZED_PSTF
    r = run_full_coupled_typeI(FullCoupledConfig(
        Sigma_H_plus=args.sigma,
        N_q=args.nq,
        N_mu=args.nmu,
        transport_mode=tm,
        correction_level=args.cl,
        enable_teff=args.teff,
    ))
    out = {
        'sigma': args.sigma,
        'nq': args.nq,
        'nmu': args.nmu,
        'mode': args.mode,
        'cl': args.cl,
        'teff': bool(args.teff),
        'Yp': float(r.observables.Yp),
        'DH': float(r.observables.DH),
        'N_eff_measured': float(r.metadata.get('N_eff_measured', float('nan'))),
    }
    if args.traj:
        out['N'] = np.asarray(r.trajectory['N']).tolist()
        out['Sigma_plus'] = np.asarray(r.trajectory['Sigma_plus']).tolist()
        out['T_gamma'] = np.asarray(r.trajectory['T_gamma']).tolist()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    print(out_path)


if __name__ == '__main__':
    main()
