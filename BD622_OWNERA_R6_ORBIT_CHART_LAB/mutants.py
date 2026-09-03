"""Semantic mutant battery for the C-R6 orbit-chart lab.

Each mutant is a single-site source substitution applied to a scratch copy of
the lab; its preregistered check must FAIL for its preregistered reason.
A surviving mutant, a substitution that does not apply exactly once, or a
mechanical ERROR in a mutant run discards the package (exit 10/20).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

LAB_FILES = ("mechanism.py", "fixtures.py", "test_mechanism.py")

MUTATIONS = (
    ("MUT-1", "mechanism.py",
     "CLOSURE = ((MEMBER_P, QUOTIENT), (MEMBER_M, QUOTIENT))",
     "CLOSURE = ((MEMBER_P, QUOTIENT),)",
     "A3"),
    ("MUT-2", "mechanism.py",
     "    if form == \"gml_pppp_mm\":\n        return gain - loss",
     "    if form == \"gml_pppp_mm\":\n        return loss - gain",
     "A6"),
    ("MUT-3", "mechanism.py",
     "QUOTIENT = Fraction(1, 2)",
     "QUOTIENT = Fraction(1, 1)",
     "A4"),
    ("MUT-4", "mechanism.py",
     "CLOSURE = ((MEMBER_P, QUOTIENT), (MEMBER_M, QUOTIENT))",
     "CLOSURE = ((MEMBER_P, QUOTIENT), (MEMBER_P, QUOTIENT), (MEMBER_M, QUOTIENT))",
     "A3"),
    ("MUT-5", "mechanism.py",
     "    if CANONICAL_SORT_ENABLED:\n        addends.sort(key=lambda entry: entry[0])",
     "    if CANONICAL_SORT_ENABLED and False:\n        addends.sort(key=lambda entry: entry[0])",
     "A10-scramble"),
    ("MUT-6", "mechanism.py",
     "NATIVE_ROUNDTRIP_ENABLED = False",
     "NATIVE_ROUNDTRIP_ENABLED = True",
     "A6"),
    ("MUT-7", "mechanism.py",
     "CLOSURE = ((MEMBER_P, QUOTIENT), (MEMBER_M, QUOTIENT))",
     "CLOSURE = ((MEMBER_P, Fraction(1, 1)),)",
     "A3"),
    ("MUT-8", "mechanism.py",
     "LANES = (1, 1, -1, -1)",
     "LANES = (-1, -1, 1, 1)",
     "A6"),
    ("MUT-9A", "mechanism.py",
     "    return y1 * y1 * y2 * y2 * t * (1 - t)",
     "    return y1 * y1 * y1 * y2 * y2 * t * (1 - t)",
     "ANCHOR"),
    ("MUT-9B", "fixtures.py",
     "    (Fraction(1, 2), Fraction(-1, 16), Fraction(0)),",
     "    (Fraction(1, 2), Fraction(-1, 8), Fraction(0)),",
     "ANCHOR"),
)


def run_mutant(lab_dir, mutant, scratch_root):
    mut_id, target, old, new, kill_check = mutant
    work = os.path.join(scratch_root, mut_id)
    os.mkdir(work)
    for name in LAB_FILES:
        shutil.copy(os.path.join(lab_dir, name), os.path.join(work, name))
    path = os.path.join(work, target)
    with open(path) as fh:
        source = fh.read()
    count = source.count(old)
    if count != 1:
        return {"id": mut_id, "status": "ERROR",
                "reason": "substitution site found %d times" % count}
    with open(path, "w") as fh:
        fh.write(source.replace(old, new))
    try:
        proc = subprocess.run(
            [sys.executable, "test_mechanism.py", "--check", kill_check],
            cwd=work, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return {"id": mut_id, "status": "ERROR", "check": kill_check,
                "reason": "timeout"}
    out = proc.stdout + proc.stderr
    if proc.returncode == 0:
        return {"id": mut_id, "status": "SURVIVED", "check": kill_check}
    if proc.returncode == 10 and ("FAIL %s" % kill_check) in out:
        return {"id": mut_id, "status": "KILLED", "check": kill_check,
                "first_line": out.strip().splitlines()[0]}
    return {"id": mut_id, "status": "ERROR", "check": kill_check,
            "returncode": proc.returncode,
            "first_line": (out.strip().splitlines() or ["<no output>"])[0]}


def main(argv):
    out_path = None
    args = list(argv[1:])
    while args:
        arg = args.pop(0)
        if arg == "--out":
            out_path = args.pop(0)
        else:
            print("ERROR unknown argument %r" % arg)
            return 20
    lab_dir = os.path.dirname(os.path.abspath(__file__))
    rows = []
    with tempfile.TemporaryDirectory() as scratch_root:
        for mutant in MUTATIONS:
            row = run_mutant(lab_dir, mutant, scratch_root)
            rows.append(row)
            print("%s %s" % (row["status"], row["id"]))
    report = {"contract_id": "EC-OWNERA-R6-ORBIT-CHART-2026-07-27", "mutants": rows}
    if out_path is not None:
        with open(out_path, "w") as fh:
            json.dump(report, fh, sort_keys=True, indent=1)
            fh.write("\n")
    if any(r["status"] == "ERROR" for r in rows):
        return 20
    if any(r["status"] != "KILLED" for r in rows):
        return 10
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
