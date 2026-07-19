"""CLI front-end for optional external BBN-code wrappers.

Run with ``python -m rabbit.external.run_cross_code``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from rabbit.external import ExternalCodeUnavailable


def _emit(obj: dict, exit_code: int = 0) -> int:
    print(json.dumps(obj, indent=2))
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code",
        required=True,
        choices=["nudec_bsm", "alterbbn"],
        help="External BBN backend",
    )
    parser.add_argument("--eta", type=float, default=6.104e-10)
    parser.add_argument("--tau-n", type=float, default=878.4)
    parser.add_argument("--n-eff", type=float, default=3.044)
    args = parser.parse_args(argv)

    try:
        if args.code == "nudec_bsm":
            from rabbit.external.nudec_bsm import run_nudec_bsm

            trash = io.StringIO()
            with contextlib.redirect_stdout(trash):
                res = run_nudec_bsm(
                    eta=args.eta,
                    tau_n=args.tau_n,
                    n_eff_target=args.n_eff,
                )
            return _emit({
                "code": "nudec_bsm",
                "available": True,
                "N_eff": res.N_eff,
                "T_gamma_final_MeV": res.T_gamma_final_MeV,
                "T_nu_e_final_MeV": res.T_nu_e_final_MeV,
                "T_nu_mu_final_MeV": res.T_nu_mu_final_MeV,
            })

        from rabbit.external.alterbbn import run_alterbbn

        res = run_alterbbn(eta=args.eta, tau_n=args.tau_n, n_eff=args.n_eff)
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


if __name__ == "__main__":
    sys.exit(main())
