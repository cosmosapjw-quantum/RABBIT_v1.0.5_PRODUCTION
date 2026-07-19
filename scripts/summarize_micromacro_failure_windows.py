#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path

def _safe_float(x, default=0.0):
    return default if x is None else float(x)

def _safe_signflip(x):
    return -1 if x is None else int(x)

def get_species_rows(rows, sp):
    out = []
    for i, r in enumerate(rows):
        s = r["species"][sp]
        raw_signed = _safe_float(s.get("raw_qdot"))
        orth_signed = _safe_float(s.get("orth_qdot_Tmu"))
        proj_signed = _safe_float(s.get("proj_qdot_Tmu"))
        raw = abs(raw_signed)
        orth = abs(orth_signed)
        ratio = orth / raw if raw > 0 else (math.inf if orth > 0 else 0.0)
        out.append({
            "idx": i,
            "label": r.get("label", f"state_{i}"),
            "N": float(r["N"]),
            "sigma_plus": float(r["sigma_plus"]),
            "Xn": float(r["Xn"]),
            "T_gamma": float(r["T_gamma"]),
            "T_nu_e": float(r["T_nu_e"]),
            "T_nu_x": float(r["T_nu_x"]),
            "raw": raw,
            "orth": orth,
            "ratio": ratio,
            "C_norm": _safe_float(s.get("C_norm")),
            "deltaI_norm": _safe_float(s.get("deltaI_norm")),
            "signflip_raw": _safe_signflip(s.get("signflip_index_raw")),
            "signflip_orthTmu": _safe_signflip(s.get("signflip_index_orthTmu")),
            "proj_Tmu": proj_signed,
            "orth_Tmu_signed": orth_signed,
            "raw_signed": raw_signed,
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--raw-threshold", type=float, default=1.0,
                    help="Ignore tiny raw_qdot when defining onset.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    summary = {}
    for path in args.jsons:
        data = json.loads(Path(path).read_text())
        case_key = Path(path).stem
        rows = data["rows"]
        summary[case_key] = {}

        for sp in ["nue", "nux"]:
            rr = get_species_rows(rows, sp)

            onset = next(
                (x for x in rr if x["raw"] >= args.raw_threshold and x["orth"] > x["raw"]),
                None,
            )
            peak_raw = max(rr, key=lambda x: x["raw"])
            peak_orth = max(rr, key=lambda x: x["orth"])
            peak_C = max(rr, key=lambda x: x["C_norm"])
            peak_dI = max(rr, key=lambda x: x["deltaI_norm"])

            summary[case_key][sp] = {
                "onset": onset,
                "peak_raw": peak_raw,
                "peak_orth": peak_orth,
                "peak_Cnorm": peak_C,
                "peak_deltaI": peak_dI,
            }

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
