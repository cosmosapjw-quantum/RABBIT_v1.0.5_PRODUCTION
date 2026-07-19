#!/usr/bin/env python3
"""scripts/run_cross_code.py — CLI front-end for rabbit.external.* wrappers.

Phase ζ deliverable (Plan §4.6). Invokes a single external BBN code at
the requested cosmological parameters and emits a JSON record on
stdout. Designed for piping into CI assertions.

Usage
-----
    python -m rabbit.external.run_cross_code --code nudec_bsm --eta 6.104e-10
    python -m rabbit.external.run_cross_code --code alterbbn --eta 6.104e-10 --tau-n 878.4

Exit codes
----------
    0  : success; stdout is JSON with {Y_p, DH, N_eff, code, ...}.
    2  : external code unavailable; stdout is JSON with {available: false, reason}.
    3  : external code crashed; stdout has {error: ...}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Force CPU before any jax import
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rabbit.external import ExternalCodeUnavailable


def _emit(obj: dict, exit_code: int = 0) -> int:
    print(json.dumps(obj, indent=2))
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code", required=True,
        choices=["nudec_bsm", "alterbbn"],
        help="External BBN backend",
    )
    parser.add_argument("--eta", type=float, default=6.104e-10,
                        help="Absolute baryon-to-photon ratio")
    parser.add_argument("--tau-n", type=float, default=878.4,
                        help="Neutron lifetime (seconds)")
    parser.add_argument("--n-eff", type=float, default=3.044,
                        help="Effective neutrino number target")
    args = parser.parse_args()

    try:
        if args.code == "nudec_bsm":
            # NUDEC_BSM v2 emits ANSI/progress chatter to stdout. Capture
            # it so our JSON record stays parseable on the consumer side.
            import contextlib
            import io
            from rabbit.external.nudec_bsm import run_nudec_bsm
            _trash = io.StringIO()
            with contextlib.redirect_stdout(_trash):
                res = run_nudec_bsm(eta=args.eta, tau_n=args.tau_n,
                                    n_eff_target=args.n_eff)
            return _emit({
                "code": "nudec_bsm",
                "available": True,
                "N_eff": res.N_eff,
                "T_gamma_final_MeV": res.T_gamma_final_MeV,
                "T_nu_e_final_MeV": res.T_nu_e_final_MeV,
                "T_nu_mu_final_MeV": res.T_nu_mu_final_MeV,
            })
        elif args.code == "alterbbn":
            from rabbit.external.alterbbn import run_alterbbn
            res = run_alterbbn(eta=args.eta, tau_n=args.tau_n,
                               n_eff=args.n_eff)
            return _emit({
                "code": "alterbbn",
                "available": True,
                "Y_p": res.Y_p,
                "DH": res.DH,
                "Li7H": res.Li7H,
            })
    except ExternalCodeUnavailable as exc:
        return _emit({
            "code": args.code,
            "available": False,
            "reason": exc.reason,
        }, exit_code=2)
    except Exception as exc:
        return _emit({
            "code": args.code,
            "available": True,
            "error": str(exc),
        }, exit_code=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
