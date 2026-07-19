#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

from rabbit.inference.forward_likelihood import canonical_forward_solver

FIELDS = [
    "phase1_handoff_N",
    "phase1_handoff_T",
    "phase1_handoff_Xn",
    "phase1_handoff_sigma_plus",
    "phase1_handoff_sigma_minus",
    "phase1_handoff_pi_plus",
    "phase1_handoff_lambda_np",
    "phase1_handoff_lambda_pn",
    "phase1_handoff_I0",
]

def packet(pred):
    m = pred.metadata
    out = {
        "Yp": float(pred.Yp),
        "DH": float(pred.DH),
        "backend": m.get("backend"),
        "driver": m.get("driver"),
    }
    for k in FIELDS:
        out[k] = m.get(k)
    out["phase1_handoff_monopole_probe"] = m.get("phase1_handoff_monopole_probe")
    return out

def delta(a, b):
    if a is None or b is None:
        return None
    try:
        aa = float(a); bb = float(b)
    except Exception:
        return None
    return {"abs": abs(aa - bb), "rel": abs(aa - bb) / max(abs(bb), 1.0e-300)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--cl", type=int, default=0)
    # NOTE: current JAX Type-I candidate is CL0-only by contract.
    ap.add_argument("--nq", type=int, default=20)
    ap.add_argument("--reactions", type=int, default=12)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    cfg = dict(
        Sigma_H=args.sigma,
        correction_level=args.cl,
        N_q=args.nq,
        n_reactions=args.reactions,
        enable_teff=False,
    )

    s = canonical_forward_solver(backend="scipy", **cfg)
    if args.cl != 0:
        raise SystemExit("Current JAX Type-I candidate is CL0-only; use --cl 0 for handoff probe.")
    j = canonical_forward_solver(backend="jax", **cfg)

    out = {"config": cfg, "scipy": packet(s), "jax": packet(j), "delta": {}}

    for k in ["Yp", "DH"] + FIELDS:
        out["delta"][k] = delta(out["jax"].get(k), out["scipy"].get(k))

    sm = out["scipy"].get("phase1_handoff_monopole_probe") or {}
    jm = out["jax"].get("phase1_handoff_monopole_probe") or {}
    out["delta"]["phase1_handoff_monopole_probe"] = {
        key: delta(jm.get(key), sm.get(key))
        for key in sorted(set(sm) | set(jm))
    }

    print(json.dumps(out, indent=2, sort_keys=True))

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"[saved] {path}")

if __name__ == "__main__":
    main()
