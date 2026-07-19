#!/usr/bin/env python3
"""Retired value grep-kill and document consistency patch.

Scans for RETIRED values and optionally patches them to CANONICAL.

CANONICAL (v1.0.0, Session 38):
  λ_slow = -0.687, λ_fast = -4.313, tr(M) = -5
  Re(λ_transport) = -1/2, ω = 1.023, σ ∝ a^{-5/2}
  Σ_H < 0.66 (95% CL, CL2), B₀₁ = 1.45
  Channel split: 98.6% / 1.4%

RETIRED (do NOT use):
  -0.617  (requires f_ν = 0.506, non-physical)
  -4.383  (paired with -0.617, wrong trace partition)
  σ ∝ a^{-3}  (free decay, ignores ν viscosity)
  Σ_H < 0.13, B₀₁ = 3.68, 90%/10%

Usage:
  python3 scripts/patch_retired_values.py --scan
  python3 scripts/patch_retired_values.py --patch [--dry]
"""
import argparse, re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Retired → Canonical replacements (TeX files)
REPLACEMENTS_TEX = [
    (r'\\lambda_\{\\rm slow\}\s*=\s*-0\.617',
     r'\\lambda_{\\rm slow} = -0.687',
     'eigenvalue: -0.617 → -0.687'),
    (r'\\lambda_\{\\rm fast\}\s*=\s*-4\.383',
     r'\\lambda_{\\rm fast} = -4.313',
     'eigenvalue: -4.383 → -4.313'),
]

# Strings that should NOT appear (except in "retired values" discussion)
RETIRED_STRINGS = [
    ('0.617', 'retired eigenvalue'),
    ('4.383', 'retired fast eigenvalue'),
    ('B_{01} = 3.7', 'retired Bayes factor'),
    ('3.68', 'retired Bayes factor'),
    ('< 0.13', 'retired constraint'),
]

SCAN_GLOBS = ['*.tex', 'src/**/*.py', 'scripts/**/*.py',
              'tests/**/*.py', 'docs/**/*.md']
SKIP_PATTERNS = ['retired', 'RETIRED', 'error', 'wrong', 'non-physical',
                 'should be', 'Retired Values', '__pycache__',
                 'Old Bayes', 'Old constraint', 'Eigenvalue &',
                 'REPLACEMENTS', 'RETIRED_STRINGS', '→',
                 'do NOT use', 'B₀₁']


def scan(path):
    """Scan file for retired values, ignoring legitimate mentions."""
    if not path.exists():
        return []
    text = path.read_text(errors='replace')
    findings = []
    for s, note in RETIRED_STRINGS:
        for i, line in enumerate(text.split('\n'), 1):
            if s in line and not any(sk in line for sk in SKIP_PATTERNS):
                findings.append((s, note, i, line.strip()[:80]))
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan', action='store_true')
    parser.add_argument('--patch', action='store_true')
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()
    if not args.scan and not args.patch:
        args.scan = True

    print("=" * 60)
    print("  Retired Value Grep-Kill (canonical: -0.687, -4.313)")
    print("=" * 60)

    all_files = []
    for g in SCAN_GLOBS:
        all_files.extend(REPO.glob(g))
    all_files = sorted(set(f for f in all_files if f.is_file()))

    total_found = 0
    for f in all_files:
        findings = scan(f)
        if findings:
            rel = f.relative_to(REPO)
            for s, note, line, ctx in findings:
                print(f"  {rel}:{line}: '{s}' ({note})")
                print(f"    {ctx}")
            total_found += len(findings)

    if total_found == 0:
        print("\n  ✓ Clean: no retired values found.")
    else:
        print(f"\n  Found {total_found} instance(s) of retired values.")

    if args.patch:
        for tex in REPO.glob('*.tex'):
            text = tex.read_text()
            changed = 0
            for pat, rep, note in REPLACEMENTS_TEX:
                text, n = re.subn(pat, rep, text)
                changed += n
            if changed and not args.dry:
                tex.write_text(text)
                print(f"  Patched {tex.name}: {changed} replacement(s)")
            elif changed:
                print(f"  [DRY] Would patch {tex.name}: {changed}")


if __name__ == "__main__":
    main()
