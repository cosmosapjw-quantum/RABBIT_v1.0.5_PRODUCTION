#!/usr/bin/env python3
"""Mechanical cost accounting for the RABBIT BBN anti-drift policy.

Emits the "Required PR Cost Line" defined by
`bbn_codex_anti_drift_cost_effective_policy.md` for a git revision range, in
the exact field order used by `docs/audit/*_adversarial_audit_*.md` section
"## 6. Cost line", so the block can be pasted straight into a report.

Everything countable is counted from git. The one field that is a human
judgement -- `blocker_movement_ratio` -- is never invented, inferred, or
defaulted: it is a required argument, and a non-empty `--blocker-justification`
is required alongside it and reproduced verbatim in the output.

Fails closed: no git, a bad revision, or an unparsable diff is an error with a
non-zero exit, never a zero-filled report. An empty range -- one whose diff
touches no file at all -- is refused the same way: it is reported as
`EMPTY_RANGE` with a non-zero exit rather than being absorbed into a normal
verdict, because a report measuring nothing must never be mistakable for a
report measuring a genuinely small change.

Usage:
    cost_report.py --since <rev> [--until <rev>] [--json] \\
        --blocker-movement <0.00..1.00> --blocker-justification <text> \\
        [--runtime-behavior-changed yes|no|harness-only] \\
        [--physics-behavior-changed yes|no] \\
        [--known-blocker-reduced yes|no] [--validation-strengthened yes|no] \\
        [--allow-empty-range]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

EXIT_ERROR = 2
# Distinct from EXIT_ERROR so a caller can tell "the range measured nothing"
# apart from "the measurement itself failed". Both are non-zero: neither is a
# passing cost report.
EXIT_EMPTY_RANGE = 3

POLICY_PATH = "bbn_codex_anti_drift_cost_effective_policy.md"

# Canonical field order of the policy's "Required PR Cost Line", matching
# `docs/audit/BD622_D065_remediation_adversarial_audit_2026-07-29.md` section 6.
COST_LINE_FIELDS = (
    "added_lines",
    "deleted_lines",
    "net_lines",
    "files_touched",
    "token_use_exact",
    "token_use_basis",
    "runtime_behavior_changed",
    "physics_behavior_changed",
    "known_blocker_reduced",
    "blocker_movement_ratio",
    "validation_strengthened",
    "cost_effectiveness_verdict",
)

TOKEN_USE_EXACT = "UNAVAILABLE"
TOKEN_USE_BASIS = (
    "harness/API exposes no reliable per-stage counter; the policy token rule "
    "forbids back-computing exact counts from transcript length"
)

# Area precedence. First matching anchored prefix wins, so the carve-outs
# (`run_evidence` out of `harness`) must precede the broader prefix.
# `rust` prefixes are discovered from the tree, not hardcoded (see rust_prefixes).
AREA_ORDER = (
    "production_source",
    "rust",
    "audit_scripts",
    "run_evidence",
    "harness",
    "tests",
    "docs",
    "other",
)
STATIC_AREA_PREFIXES = (
    ("production_source", ("src/",)),
    ("audit_scripts", ("scripts/audit/",)),
    ("run_evidence", (".agent-harness/runs/",)),
    ("harness", (".agent-harness/", ".codex/hooks/")),
    ("tests", ("tests/",)),
    ("docs", ("docs/",)),
)

VERDICTS = (
    "ACCEPT",
    "ACCEPT_WITH_LIMITS",
    "FAILURE_MODE_RELOCATION",
    "NO_PROGRESS",
    "DRIFT",
)
UNDETERMINED = "UNDETERMINED"
# Not a policy verdict: a marker that no cost was measured at all, so no verdict
# of any kind applies. It supersedes whatever the policy rules would derive from
# an all-zero cost line.
EMPTY_RANGE = "EMPTY_RANGE"

# Policy "## Blocker Movement Ratio" anchors, read as tiers.
MOVEMENT_TIERS = (
    ("none", "0.00 -- no measured movement; wrapper/doc/schema-only growth"),
    ("localization", "0.25 -- path/failure mode localized; endpoint/performance did not improve"),
    ("partial", "0.50 -- one known blocker partially moved"),
    ("substantial", "0.75 -- a hard blocker moved substantially"),
    ("endpoint", "1.00 -- previously impossible full e2e run reaches endpoint"),
)

# Policy "## Verdicts" entries reachable from each movement tier.
TIER_CANDIDATES = {
    "none": ("NO_PROGRESS", "DRIFT"),
    "localization": ("FAILURE_MODE_RELOCATION", "ACCEPT_WITH_LIMITS"),
    "partial": ("ACCEPT_WITH_LIMITS",),
    "substantial": ("ACCEPT",),
    "endpoint": ("ACCEPT",),
}


def die(message: str) -> NoReturn:
    sys.stderr.write(f"cost_report.py: error: {message}\n")
    raise SystemExit(EXIT_ERROR)


def run_git(repo: Path, args: list[str]) -> str:
    """Run git in `repo`. Any failure is fatal; callers never see partial data."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        die("git executable not found on PATH; cost accounting cannot be measured")
    except OSError as exc:
        die(f"could not execute git: {exc}")
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or f"exit {proc.returncode}"
        die(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout


def repo_root() -> Path:
    """Resolve the repo root via git.

    Deliberately not `_harness.root()`: that helper silently falls back to a
    path guess when git is missing, which would defeat the fail-closed rule.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        die("git executable not found on PATH; cost accounting cannot be measured")
    except OSError as exc:
        die(f"could not execute git: {exc}")
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or f"exit {proc.returncode}"
        die(f"not inside a git work tree: {detail}")
    top = proc.stdout.strip()
    if not top:
        die("git reported an empty work tree root")
    return Path(top).resolve()


def resolve_commit(repo: Path, rev: str) -> str:
    out = run_git(repo, ["rev-parse", "--verify", "--end-of-options", f"{rev}^{{commit}}"])
    sha = out.strip()
    if not sha:
        die(f"revision {rev!r} did not resolve to a commit")
    return sha


def rust_prefixes(repo: Path, until_commit: str) -> tuple[str, ...]:
    """Locate the Rust crate director(ies) from the tree instead of hardcoding."""
    out = run_git(repo, ["ls-tree", "-r", "--name-only", until_commit])
    found: set[str] = set()
    for line in out.splitlines():
        name = line.strip()
        if name.endswith("Cargo.toml"):
            parent = name[: -len("Cargo.toml")]
            if parent:
                found.add(parent)
    return tuple(sorted(found))


def area_prefixes(rust: tuple[str, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Ordered (area, prefixes) pairs; the first anchored match wins."""
    static = dict(STATIC_AREA_PREFIXES)
    ordered: list[tuple[str, tuple[str, ...]]] = []
    for area in AREA_ORDER:
        if area == "rust":
            if rust:
                ordered.append(("rust", rust))
        elif area in static:
            ordered.append((area, static[area]))
    return tuple(ordered)


def classify(path: str, prefixes: tuple[tuple[str, tuple[str, ...]], ...]) -> str:
    for area, pfxs in prefixes:
        if path.startswith(pfxs):
            return area
    return "other"


def parse_numstat(repo: Path, since: str, until: str) -> list[tuple[int, int, str, bool]]:
    """Parse `git diff --numstat -z <since>..<until>` into (added, deleted, path, binary)."""
    out = run_git(repo, ["diff", "--numstat", "-z", f"{since}..{until}"])
    tokens = out.split("\0")
    rows: list[tuple[int, int, str, bool]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token:
            i += 1
            continue
        parts = token.split("\t")
        if len(parts) != 3:
            die(f"unparsable numstat record: {token!r}")
        added_s, deleted_s, path = parts
        if path == "":
            # Rename/copy: the path field is empty and the source and destination
            # paths follow as two separate NUL-terminated fields.
            if i + 2 >= len(tokens):
                die("truncated numstat rename record")
            path = tokens[i + 2]
            i += 3
        else:
            i += 1
        binary = added_s == "-" or deleted_s == "-"
        try:
            added = 0 if binary else int(added_s)
            deleted = 0 if binary else int(deleted_s)
        except ValueError:
            die(f"non-numeric numstat counts in record: {token!r}")
        rows.append((added, deleted, path, binary))
    return rows


def line_budget_band(net: int) -> tuple[str, str]:
    """Policy '## Line Budget' band for net_lines."""
    if net <= 80:
        return "preferred", "net_lines <= 80"
    if net <= 200:
        return "acceptable_with_evidence", "80 < net_lines <= 200"
    if net <= 400:
        return "warning", "200 < net_lines <= 400"
    return "presumed_drift", "net_lines > 400"


def movement_tier(ratio: float) -> str:
    if ratio <= 0.0:
        return "none"
    if ratio < 0.50:
        return "localization"
    if ratio < 0.75:
        return "partial"
    if ratio < 1.00:
        return "substantial"
    return "endpoint"


def derive_verdict(
    net_lines: int,
    ratio: float,
    tier: str,
    validation_strengthened: str,
) -> tuple[str, list[str], list[str], str, str]:
    """Apply the policy's own rules; report candidates rather than guess.

    Returns (verdict, candidates, trace, band, band_desc).
    """
    band, band_desc = line_budget_band(net_lines)
    candidates = list(TIER_CANDIDATES[tier])
    trace = [
        f"blocker movement tier {tier!r} (ratio {ratio:.2f}) admits "
        f"{' | '.join(candidates)} from the policy verdict list"
    ]

    if band == "preferred":
        if tier == "none" and "DRIFT" in candidates and len(candidates) > 1:
            candidates = [c for c in candidates if c != "DRIFT"]
            trace.append(
                "line budget: net_lines <= 80 is the policy's preferred band, so cost is "
                "not disproportionate; DRIFT ('grew without proportional gain') eliminated"
            )
        else:
            trace.append("line budget: net_lines <= 80 (preferred) adds no cost penalty")

    elif band == "acceptable_with_evidence":
        evidence = validation_strengthened == "yes" or tier != "none"
        if evidence:
            trace.append(
                "line budget: 80 < net_lines <= 200 is acceptable and the required direct "
                "test/artifact evidence is on the record (blocker movement and/or "
                "validation_strengthened=yes)"
            )
        else:
            candidates = ["DRIFT"]
            trace.append(
                "line budget: 80 < net_lines <= 200 requires direct test/artifact evidence; "
                "no blocker movement and validation_strengthened != yes, so the band "
                "condition is unmet -> DRIFT"
            )

    elif band == "warning":
        if tier == "none":
            candidates = ["DRIFT"]
            trace.append(
                "line budget: 200 < net_lines <= 400 must include blocker movement or net "
                "deletion/consolidation; ratio 0.00 supplies neither -> DRIFT"
            )
        else:
            trace.append(
                "line budget: 200 < net_lines <= 400 is a warning band satisfied by the "
                "stated blocker movement"
            )

    else:  # presumed_drift
        if tier in ("none", "localization"):
            candidates = ["DRIFT"]
            trace.append(
                "line budget: net_lines > 400 is presumed drift unless a hard endpoint, "
                f"physics, runtime, or validation blocker moved; tier {tier!r} does not "
                "reach the policy's 'hard blocker moved substantially' anchor (0.75) -> DRIFT"
            )
        elif tier == "partial":
            candidates = sorted(set(candidates) | {"DRIFT"})
            trace.append(
                "line budget: net_lines > 400 is presumed drift unless a hard blocker "
                "moved; the policy calls 0.50 'one known blocker partially moved', which "
                "neither clearly meets nor clearly fails that exception -- ambiguous"
            )
        else:
            candidates = sorted(set(candidates) | {"DRIFT"})
            trace.append(
                "line budget: net_lines > 400 rebuts the drift presumption only if a hard "
                "blocker moved AND the record explains why the change could not be split; "
                "the ratio meets the first clause, the second is a human narrative this "
                "script cannot verify -- ambiguous"
            )

    if len(candidates) == 1:
        return candidates[0], candidates, trace, band, band_desc
    trace.append(
        "policy rules do not select a unique verdict for these inputs; reporting "
        "UNDETERMINED rather than guessing"
    )
    return UNDETERMINED, candidates, trace, band, band_desc


def ratio_arg(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"blocker movement ratio must be a number in 0.00..1.00, got {text!r}"
        )
    if value != value or value in (float("inf"), float("-inf")):
        raise argparse.ArgumentTypeError("blocker movement ratio must be finite")
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(
            f"blocker movement ratio must be within 0.00..1.00, got {value}"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cost_report.py",
        description=(
            "Emit the anti-drift Required PR Cost Line for a git range. Counted fields "
            "come from git; blocker_movement_ratio is a required human judgement and is "
            "never inferred or defaulted."
        ),
        epilog=(
            "Area precedence (first anchored prefix wins): production_source (src/), "
            "rust (discovered Cargo.toml directories), audit_scripts (scripts/audit/), "
            "run_evidence (.agent-harness/runs/), harness (.agent-harness/, "
            ".codex/hooks/), tests (tests/), docs (docs/), other."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--since", required=True, metavar="REV", help="range start revision")
    parser.add_argument(
        "--until", default="HEAD", metavar="REV", help="range end revision (default: HEAD)"
    )
    parser.add_argument("--json", action="store_true", help="emit machine JSON instead of the cost block")
    parser.add_argument(
        "--blocker-movement",
        required=True,
        type=ratio_arg,
        metavar="RATIO",
        help="REQUIRED human judgement, 0.00..1.00 (policy '## Blocker Movement Ratio')",
    )
    parser.add_argument(
        "--blocker-justification",
        required=True,
        metavar="TEXT",
        help="REQUIRED non-empty statement of what blocker moved; printed verbatim",
    )
    parser.add_argument(
        "--runtime-behavior-changed",
        choices=("yes", "no", "harness-only"),
        default=None,
        help="if omitted, derived from which areas changed (marked as derived)",
    )
    parser.add_argument(
        "--physics-behavior-changed",
        choices=("yes", "no"),
        default=None,
        help="if omitted, derived from production/rust source lines changed (marked as derived)",
    )
    parser.add_argument(
        "--known-blocker-reduced",
        choices=("yes", "no"),
        default=None,
        help="if omitted, derived from --blocker-movement (> 0.00 means yes)",
    )
    parser.add_argument(
        "--validation-strengthened",
        choices=("yes", "no"),
        default=None,
        help="if omitted, derived from added lines under tests/ and scripts/audit/",
    )
    parser.add_argument(
        "--allow-empty-range",
        action="store_true",
        help=(
            "permit a range that touches no file to exit 0. The EMPTY_RANGE marker is "
            "still emitted; only the exit code is relaxed. Without this flag an empty "
            f"range exits {EXIT_EMPTY_RANGE}, because a range that measured nothing "
            "must not read as a cheap change"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    justification = args.blocker_justification.strip()
    if not justification:
        die(
            "--blocker-justification must be non-empty: blocker_movement_ratio is a "
            "judgement and the record must keep the human's own words"
        )

    repo = repo_root()
    since_commit = resolve_commit(repo, args.since)
    until_commit = resolve_commit(repo, args.until)
    rust = rust_prefixes(repo, until_commit)
    prefixes = area_prefixes(rust)
    rows = parse_numstat(repo, args.since, args.until)

    areas = {a: {"added": 0, "deleted": 0, "net": 0, "files": 0} for a in AREA_ORDER}
    added_lines = deleted_lines = files_touched = 0
    binary_files: list[str] = []
    for added, deleted, path, binary in rows:
        area = classify(path, prefixes)
        bucket = areas[area]
        bucket["added"] += added
        bucket["deleted"] += deleted
        bucket["net"] += added - deleted
        bucket["files"] += 1
        added_lines += added
        deleted_lines += deleted
        files_touched += 1
        if binary:
            binary_files.append(path)

    net_lines = added_lines - deleted_lines
    # An empty range is a measurement failure, not a small change: `git diff`
    # reported no file at all between the endpoints. Zero net lines lands in the
    # policy's preferred band, so without this flag a mis-cited range would emit
    # a clean-looking cost report about no work whatsoever.
    empty_range = files_touched == 0
    production_source_lines_changed = areas["production_source"]["added"] + areas["production_source"]["deleted"]
    rust_lines_changed = areas["rust"]["added"] + areas["rust"]["deleted"]
    executable_nonproduction = sum(
        areas[a]["added"] + areas[a]["deleted"] for a in ("harness", "audit_scripts", "tests")
    )

    notes: list[str] = []

    # --- derived-unless-stated fields -------------------------------------
    if args.known_blocker_reduced is not None:
        known_blocker_reduced = args.known_blocker_reduced
        known_blocker_source = "stated"
    else:
        known_blocker_reduced = "yes" if args.blocker_movement > 0.0 else "no"
        known_blocker_source = "derived"
        notes.append(
            "known_blocker_reduced derived from --blocker-movement "
            "(policy: 0.00 means 'no measured movement')"
        )

    if args.validation_strengthened is not None:
        validation_strengthened = args.validation_strengthened
        validation_source = "stated"
    else:
        validation_added = areas["tests"]["added"] + areas["audit_scripts"]["added"]
        validation_strengthened = "yes" if validation_added > 0 else "no"
        validation_source = "derived"
        notes.append(
            f"validation_strengthened derived from added lines under tests/ and "
            f"scripts/audit/ ({validation_added}); pass --validation-strengthened to override"
        )

    if args.physics_behavior_changed is not None:
        physics_behavior_changed = args.physics_behavior_changed
        physics_source = "stated"
    elif production_source_lines_changed == 0 and rust_lines_changed == 0:
        physics_behavior_changed = "no"
        physics_source = "derived"
        notes.append(
            "physics_behavior_changed derived as 'no': no production source or Rust crate "
            "line changed in the range"
        )
    else:
        physics_behavior_changed = "UNSTATED"
        physics_source = "unstated"
        notes.append(
            "physics_behavior_changed is UNSTATED: production/Rust source lines changed, so "
            "it cannot be derived mechanically -- pass --physics-behavior-changed"
        )

    if args.runtime_behavior_changed is not None:
        runtime_behavior_changed = {
            "yes": "yes",
            "no": "no",
            "harness-only": "yes — harness only",
        }[args.runtime_behavior_changed]
        runtime_source = "stated"
    elif production_source_lines_changed == 0 and rust_lines_changed == 0:
        if executable_nonproduction > 0:
            runtime_behavior_changed = "yes — harness only"
            notes.append(
                "runtime_behavior_changed derived as 'yes — harness only': only harness, "
                "audit-script, or test executables changed"
            )
        else:
            runtime_behavior_changed = "no"
            notes.append(
                "runtime_behavior_changed derived as 'no': no executable surface changed in "
                "the range"
            )
        runtime_source = "derived"
    else:
        runtime_behavior_changed = "UNSTATED"
        runtime_source = "unstated"
        notes.append(
            "runtime_behavior_changed is UNSTATED: production/Rust source lines changed, so "
            "it cannot be derived mechanically -- pass --runtime-behavior-changed"
        )

    tier = movement_tier(args.blocker_movement)
    tier_desc = dict(MOVEMENT_TIERS)[tier]
    verdict, candidates, trace, band, band_desc = derive_verdict(
        net_lines, args.blocker_movement, tier, validation_strengthened
    )

    # --- empty range supersedes every derived verdict ----------------------
    if empty_range:
        trace.append(
            f"EMPTY_RANGE: {args.since}..{args.until} contains no changed file, so no cost "
            f"was measured; the derived verdict {verdict!r} is superseded -- an all-zero "
            "cost line is evidence of an unmeasured range, not of a cheap change"
        )
        verdict = EMPTY_RANGE
        candidates = [EMPTY_RANGE]
        notes.append(
            f"EMPTY_RANGE: git diff {args.since}..{args.until} "
            f"({since_commit[:8]}..{until_commit[:8]}) touches 0 files; check the cited "
            "range before treating this report as a cost accounting"
        )
        if args.allow_empty_range:
            notes.append(
                "--allow-empty-range was passed, so the empty range is reported and exits "
                "0; the EMPTY_RANGE marker stands and no verdict was derived"
            )
        else:
            notes.append(
                f"exiting {EXIT_EMPTY_RANGE}: an empty range is refused by default; pass "
                "--allow-empty-range only when reporting a genuinely empty range is the "
                "intent"
            )

    # --- consistency notes (never silently rewrite a stated field) ---------
    if physics_behavior_changed == "yes" and production_source_lines_changed == 0 and rust_lines_changed == 0:
        notes.append(
            "inconsistency: physics_behavior_changed=yes but 0 production source and 0 Rust "
            "lines changed in this range"
        )
    if physics_behavior_changed == "no" and production_source_lines_changed > 0:
        notes.append(
            f"check: physics_behavior_changed=no while {production_source_lines_changed} "
            "production source lines changed -- this claim needs test/artifact evidence"
        )
    if binary_files:
        notes.append(
            f"{len(binary_files)} binary file(s) counted in files_touched with 0 line delta "
            "(git reports no line counts for binaries)"
        )
    dirty = run_git(repo, ["status", "--porcelain"]).strip()
    if dirty:
        notes.append(
            f"working tree has {len(dirty.splitlines())} uncommitted path(s); they are NOT "
            "counted -- this report covers committed range content only"
        )

    verdict_line = verdict
    if verdict == UNDETERMINED:
        verdict_line = f"{UNDETERMINED} — candidates: {' | '.join(candidates)}"
    elif verdict == EMPTY_RANGE:
        verdict_line = (
            f"{EMPTY_RANGE} — range touches 0 files; no cost was measured and no verdict "
            "applies"
        )

    # Non-zero unless the caller explicitly asked to report an empty range.
    exit_code = EXIT_EMPTY_RANGE if empty_range and not args.allow_empty_range else 0

    cost_line = {
        "added_lines": added_lines,
        "deleted_lines": deleted_lines,
        "net_lines": net_lines,
        "files_touched": files_touched,
        "token_use_exact": TOKEN_USE_EXACT,
        "token_use_basis": TOKEN_USE_BASIS,
        "runtime_behavior_changed": runtime_behavior_changed,
        "physics_behavior_changed": physics_behavior_changed,
        "known_blocker_reduced": known_blocker_reduced,
        "blocker_movement_ratio": f"{args.blocker_movement:.2f}",
        "validation_strengthened": validation_strengthened,
        "cost_effectiveness_verdict": verdict_line,
    }

    if args.json:
        payload = {
            "schema": "rabbit.cost_report.v1",
            "policy": POLICY_PATH,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "range": {
                "since": args.since,
                "until": args.until,
                "since_commit": since_commit,
                "until_commit": until_commit,
            },
            "empty_range": empty_range,
            "empty_range_allowed": bool(args.allow_empty_range),
            "exit_code": exit_code,
            "added_lines": added_lines,
            "deleted_lines": deleted_lines,
            "net_lines": net_lines,
            "files_touched": files_touched,
            "production_source_lines_changed": production_source_lines_changed,
            "rust_lines_changed": rust_lines_changed,
            "areas": {a: areas[a] for a in AREA_ORDER},
            "rust_crate_prefixes": list(rust),
            "token_use_exact": TOKEN_USE_EXACT,
            "token_use_basis": TOKEN_USE_BASIS,
            "runtime_behavior_changed": runtime_behavior_changed,
            "runtime_behavior_changed_source": runtime_source,
            "physics_behavior_changed": physics_behavior_changed,
            "physics_behavior_changed_source": physics_source,
            "known_blocker_reduced": known_blocker_reduced,
            "known_blocker_reduced_source": known_blocker_source,
            "blocker_movement_ratio": args.blocker_movement,
            "blocker_movement_tier": tier,
            "blocker_movement_tier_basis": tier_desc,
            "blocker_justification": justification,
            "validation_strengthened": validation_strengthened,
            "validation_strengthened_source": validation_source,
            "line_budget_band": band,
            "line_budget_band_basis": band_desc,
            "cost_effectiveness_verdict": verdict,
            "verdict_candidates": candidates,
            "verdict_rule_trace": trace,
            "notes": notes,
        }
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return exit_code

    out: list[str] = []
    if empty_range:
        # First line of the report, before the pasteable block: the marker must
        # not be something a reader has to go looking for.
        out.append(
            f"{EMPTY_RANGE}: {args.since}..{args.until} "
            f"({since_commit[:8]}..{until_commit[:8]}) touches 0 files. Nothing was "
            "measured; this is not a cost accounting."
        )
        out.append("")
    out.append("```text")
    out.extend(f"{name}: {cost_line[name]}" for name in COST_LINE_FIELDS)
    out.append("```")
    out.append("")
    out.append("```text")
    out.append(f"range: {args.since}..{args.until} ({since_commit[:8]}..{until_commit[:8]})")
    out.append(f"production_source_lines_changed: {production_source_lines_changed}")
    out.append(f"{'area':<20}{'added':>9}{'deleted':>9}{'net':>9}{'files':>7}")
    for area in AREA_ORDER:
        bucket = areas[area]
        out.append(
            f"{area:<20}{bucket['added']:>9}{bucket['deleted']:>9}"
            f"{bucket['net']:>9}{bucket['files']:>7}"
        )
    out.append(f"{'TOTAL':<20}{added_lines:>9}{deleted_lines:>9}{net_lines:>9}{files_touched:>7}")
    out.append("```")
    out.append("")
    out.append("blocker_justification (verbatim, as stated by the human):")
    for line in justification.splitlines() or [""]:
        out.append(f"  {line}")
    out.append("")
    out.append("verdict derivation:")
    out.append(f"  line_budget_band: {band} ({band_desc})")
    out.append(f"  blocker_movement_tier: {tier} ({tier_desc})")
    for item in trace:
        out.append(f"  - {item}")
    out.append(f"  candidates: {' | '.join(candidates)}")
    out.append(f"  verdict: {verdict}")
    if notes:
        out.append("")
        out.append("notes:")
        out.extend(f"  - {note}" for note in notes)
    sys.stdout.write("\n".join(out) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
