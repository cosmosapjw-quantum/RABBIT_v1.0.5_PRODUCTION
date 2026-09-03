#!/usr/bin/env python3
"""Mechanical chronology checker for frozen-contract evidence (D-069 section 7).

D-065 finding F-D065-04 item 6 found evidence whose paperwork asserted an order
of events that could not have happened: an evidence run whose contract and
driver were modified roughly three hours *before* the adjudication that
supposedly authorised them, and adjudication envelopes carrying asserted
round-minute timestamps later than the commits that contained them. The frozen
D-069 contract requires that the invariant below be checked mechanically rather
than by eye:

    freeze_commit_time < report.started_at < report.completed_at
      < adjudication.started_at < adjudication.completed_at < containing_commit

Strictly increasing: two equal stamps are as impossible as two inverted ones.

Everything here fails closed. A commit that git cannot resolve *to exactly one
commit*, a stamp that is absent, unparseable, timezone-naive, or carries an
offset that cannot be unambiguously interpreted, and a git binary that is not on
PATH are all errors -- never a skipped check. An unrunnable chronology check is
not a passed one.

Timestamp frames and the round-minute rule
------------------------------------------
Every stamp is judged in two frames, and the two are deliberately not
conflated:

* **The frame it was typed in.** The round-minute rule is an assertion about a
  *human*: F-D065-04 caught someone writing ``23:05:00Z`` and ``23:20:00Z``. A
  hand-typed stamp is suspicious on the wall clock the hand was reading, so the
  round-minute test is asserted on the local wall clock exactly as written.
* **UTC.** Ordering is a statement about instants, so every comparison uses the
  UTC normalisation of the stamp, never its text.

The round-minute test additionally runs on that UTC ordering value. Under the
offset rule below the two frames always agree on the seconds and microseconds
fields, so this is redundancy rather than a second rule -- but it is written out
explicitly because testing a pre-normalisation value while ordering a
post-normalisation one is exactly how this check can be defeated: a stamp
spelled with a sub-minute offset reads as a non-round wall clock while the
instant used for ordering lands precisely on the minute.

Accordingly ``parse_stamp`` refuses any UTC offset that is not a whole number of
minutes, and any offset outside the real range of -14:00..+14:00. A stamp whose
offset cannot be unambiguously interpreted is an error; it never quietly skips
the round-minute test. A stamp carrying no seconds field at all asserts an exact
minute boundary just as plainly as ``:00`` does, so it is flagged too, and
fractional seconds are handled explicitly: ``:00.000`` is a round minute,
``:00.5`` is not.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from _harness import load_json, root

# The upper bound when --containing-commit is omitted. The commit that will
# contain this evidence does not exist yet (the normal case: you check before
# committing), so "now" is substituted and the substitution is reported. The
# constraint is never silently dropped.
PENDING_SOURCE = "PENDING (now)"

# Calendar date and wall clock, with everything after the wall clock captured
# whole so that a malformed offset is reported as a malformed offset rather than
# as an unrecognisable timestamp. The seconds field is optional here only so
# that its *absence* can be detected and flagged; see Stamp.has_seconds.
ISO_STAMP_RE = re.compile(
    r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"[Tt ]"
    r"(?P<hour>\d{2}):(?P<minute>\d{2})"
    r"(?::(?P<second>\d{2})(?:\.(?P<fraction>\d+))?)?"
    r"(?P<offset>.*)"
)

# ±hh, ±hhmm, ±hh:mm and -- only so that they can be rejected with a specific
# message instead of falling through to a generic "malformed" -- ±hh:mm:ss[.fff].
OFFSET_RE = re.compile(
    r"(?P<sign>[+-])(?P<hours>\d{2})"
    r"(?::?(?P<minutes>\d{2})"
    r"(?::(?P<seconds>\d{2})(?:\.(?P<subseconds>\d+))?)?"
    r")?"
)

# The real range of civil UTC offsets is -12:00..+14:00; the symmetric -14:00
# bound is used because it is the defensible outer limit and nothing legitimate
# in this repository sits near either edge.
MAX_UTC_OFFSET = timedelta(hours=14)

COMMIT_ID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


@dataclass(frozen=True)
class Stamp:
    """One timestamp, kept in both of the frames it has to be judged in.

    ``local`` is the wall clock exactly as written, carrying its declared
    offset: the frame a human typed in, and therefore the frame the round-minute
    rule is asserted in. ``utc`` is the same instant normalised, and is the only
    value used for ordering. ``has_seconds`` records whether the text carried a
    seconds field at all, because omitting it asserts a minute boundary.
    """

    raw: str
    local: datetime
    utc: datetime
    has_seconds: bool


def on_exact_minute(value: datetime) -> bool:
    return value.second == 0 and value.microsecond == 0


def parse_offset(
    text: str, raw: str, name: str, errors: list[str]
) -> timezone | None:
    """A UTC offset that can be interpreted with no ambiguity at all."""

    if text in ("Z", "z"):
        return timezone.utc
    match = OFFSET_RE.fullmatch(text)
    if match is None:
        errors.append(
            f"{name}: {raw!r} has a malformed UTC offset {text!r}; a stamp whose "
            "offset cannot be interpreted cannot be ordered or tested"
        )
        return None
    if match.group("seconds") is not None or match.group("subseconds") is not None:
        errors.append(
            f"{name}: {raw!r} declares the UTC offset {text!r}, which is not a "
            "whole number of minutes. Real UTC offsets are whole minutes, and a "
            "sub-minute offset moves the instant used for ordering without "
            "moving the wall clock that was typed -- so it is rejected, not "
            "interpreted"
        )
        return None
    minutes = int(match.group("minutes") or 0)
    if minutes > 59:
        errors.append(
            f"{name}: {raw!r} declares the UTC offset {text!r}, whose minutes "
            "field is not a real minute count"
        )
        return None
    delta = timedelta(hours=int(match.group("hours")), minutes=minutes)
    if delta > MAX_UTC_OFFSET:
        errors.append(
            f"{name}: {raw!r} declares the UTC offset {text!r}, outside the real "
            "range -14:00..+14:00"
        )
        return None
    if match.group("sign") == "-":
        delta = -delta
    return timezone(delta)


def parse_stamp(raw: object, name: str, errors: list[str]) -> Stamp | None:
    """Parse one ISO-8601 stamp, refusing anything that cannot be ordered.

    Parsing is done from an explicit grammar rather than by handing the text to
    ``datetime.fromisoformat``: the checker has to know whether a seconds field
    was present and whether the offset was a whole number of minutes, and it
    must not inherit whichever spellings the running interpreter happens to
    accept this release.
    """

    if not isinstance(raw, str) or not raw.strip():
        errors.append(f"{name}: missing or not a non-empty string (got {raw!r})")
        return None
    text = raw.strip()
    match = ISO_STAMP_RE.fullmatch(text)
    if match is None:
        errors.append(
            f"{name}: {raw!r} is not an ISO-8601 timestamp of the form "
            "YYYY-MM-DDThh:mm:ss<offset>"
        )
        return None

    offset_text = match.group("offset")
    if not offset_text:
        # Coercing a naive stamp to UTC would invent the very fact under audit.
        errors.append(
            f"{name}: {raw!r} is timezone-naive and cannot be ordered against "
            "git's timezone-aware commit times"
        )
        return None
    offset = parse_offset(offset_text, text, name, errors)
    if offset is None:
        return None

    fraction = match.group("fraction")
    if fraction is not None and len(fraction) > 6:
        errors.append(
            f"{name}: {raw!r} carries sub-microsecond precision that this "
            "checker would have to truncate to compare; truncating the value "
            "under audit is not allowed, so the stamp is rejected"
        )
        return None
    second_text = match.group("second")
    try:
        local = datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            int(second_text) if second_text is not None else 0,
            int(fraction.ljust(6, "0")) if fraction else 0,
            tzinfo=offset,
        )
    except ValueError as exc:
        errors.append(f"{name}: {raw!r} is not a real date and time ({exc})")
        return None
    return Stamp(
        raw=text,
        local=local,
        utc=local.astimezone(timezone.utc),
        has_seconds=second_text is not None,
    )


def round_minute_reasons(stamp: Stamp) -> list[str]:
    """Every way in which ``stamp`` asserts an exact minute boundary.

    Tested in both frames on purpose. The wall clock as written is the frame a
    human typed in and is the primary assertion; the UTC normalisation is the
    value that ordering actually uses. Under the whole-minute offset rule
    enforced by ``parse_offset`` the two always agree, but they are checked
    separately so that no future loosening of that rule can reopen the gap
    between "the value tested" and "the value used".
    """

    reasons: list[str] = []
    if not stamp.has_seconds:
        reasons.append(
            "it carries no seconds field at all, which asserts a minute boundary "
            "just as plainly as ':00' does"
        )
    if on_exact_minute(stamp.local):
        reasons.append(
            f"on the wall clock it was written against it reads "
            f"{stamp.local.isoformat()}"
        )
    if on_exact_minute(stamp.utc):
        reasons.append(
            f"normalised for ordering it is {stamp.utc.isoformat()}"
        )
    return reasons


def check_round_minute(
    stamp: Stamp,
    name: str,
    kind: str,
    allow: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Flag a stamp that landed exactly on a minute boundary.

    Blocking for *asserted* stamps -- the ``started_at``/``completed_at`` pairs
    in the report and adjudication envelopes. Those are hand-typeable, and
    F-D065-04 caught real asserted ones at 23:05:00Z and 23:20:00Z. A machine
    stamp from ``Reporter.stamp`` (``datetime.now(timezone.utc)`` at
    ``timespec="seconds"``) lands on an exact minute about once in sixty, so a
    round minute there is worth blocking on. ``--allow-round-minute`` is the
    documented escape hatch, and it scopes to asserted stamps only.

    Non-blocking for *git commit times*. Git generates them, so they land on
    ``:00`` about one time in sixty by construction, and the measured rate on
    this branch is consistent with chance: 3 of the 44 commits since the
    baseline have ``second == 00`` against 0.73 expected, but the modal
    second-value over the same 44 commits occurs 4 times and two other values
    also occur 3 times. Selecting ``00`` *because* it was flagged is a
    look-elsewhere artifact; 3 is unremarkable. Blocking on it would raise a
    false alarm roughly every 20 commits while detecting nothing.
    ``GIT_COMMITTER_DATE`` is settable, so the observation is still surfaced --
    as a warning that does not affect ``ok`` or the exit code.

    The PENDING "now" bound is checked at neither level: this checker generates
    it, nobody asserts it, and flagging it would fire on one honest run in
    sixty for no reason at all.
    """

    reasons = round_minute_reasons(stamp)
    if not reasons:
        return
    message = f"{name}: {stamp.raw!r} is an exact round minute ({'; '.join(reasons)})"
    if kind == "asserted":
        message += (
            " -- the signature of a hand-asserted timestamp rather than a machine stamp"
        )
        if not allow:
            errors.append(message)
            return
        warnings.append(message + " (downgraded by --allow-round-minute)")
        return
    warnings.append(
        message
        + "; git produces this about 1 time in 60 by chance, so it does not fail "
        "the check. Surfaced only because GIT_COMMITTER_DATE is settable"
    )


def run_git(
    repo: Path, argv: list[str], name: str, errors: list[str]
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *argv],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        # Fail closed: without git there is no independent clock to order the
        # asserted stamps against, so nothing here can be verified.
        errors.append(
            f"{name}: git is not available, so commit times cannot be read; "
            "the chronology check cannot run"
        )
    except OSError as exc:
        errors.append(f"{name}: git could not be invoked ({exc.__class__.__name__})")
    return None


def resolve_commit(repo: Path, rev: str, name: str, errors: list[str]) -> str | None:
    """The one full commit id ``rev`` names, or an error.

    ``rev-parse --verify`` refuses any revision that does not name exactly one
    object, so ranges (``a..b``), parent sets (``x^@``), pseudo-refs and
    ambiguous abbreviations are rejected here instead of silently collapsing to
    whichever commit git happened to print first. ``--end-of-options`` stops an
    argument that begins with ``-`` from being read as an option, and the
    ``^{commit}`` peel plus the explicit ``cat-file -t`` check make "this is a
    commit" a verified fact rather than an assumption.
    """

    completed = run_git(
        repo,
        ["rev-parse", "--verify", "--end-of-options", f"{rev}^{{commit}}"],
        name,
        errors,
    )
    if completed is None:
        return None
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        errors.append(
            f"{name}: git cannot resolve {rev!r} to exactly one commit "
            f"({detail[-1].strip() if detail else 'unknown revision'})"
        )
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1 or not COMMIT_ID_RE.fullmatch(lines[0]):
        errors.append(
            f"{name}: git resolved {rev!r} to {len(lines)} object id(s) rather "
            "than exactly one commit id; an ambiguous revision cannot anchor a "
            "chronology"
        )
        return None
    resolved = lines[0]

    kind = run_git(repo, ["cat-file", "-t", resolved], name, errors)
    if kind is None:
        return None
    object_type = (kind.stdout or "").strip()
    if kind.returncode != 0 or object_type != "commit":
        errors.append(
            f"{name}: {rev!r} resolved to {resolved}, whose object type is "
            f"{object_type or 'unreadable'!r} rather than a commit"
        )
        return None
    return resolved


def commit_time(
    repo: Path, rev: str, name: str, errors: list[str]
) -> tuple[Stamp | None, str | None]:
    """Committer time of the single commit ``rev`` names, or an error."""

    resolved = resolve_commit(repo, rev, name, errors)
    if resolved is None:
        return None, None
    completed = run_git(repo, ["show", "-s", "--format=%cI", resolved], name, errors)
    if completed is None:
        return None, None
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        errors.append(
            f"{name}: git cannot read the committer time of {resolved} "
            f"({detail[-1].strip() if detail else 'unknown error'})"
        )
        return None, None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        # Extra output is evidence that the question was ambiguous. Discarding
        # it and trusting line one is how a checker ends up certifying a
        # chronology it never actually read.
        errors.append(
            f"{name}: git returned {len(lines)} committer times for {resolved} "
            "where exactly one was required"
        )
        return None, None
    raw = lines[0]
    return parse_stamp(raw, name, errors), raw


def read_envelope(path: Path, label: str, errors: list[str]) -> dict | None:
    try:
        value = load_json(path)
    except OSError as exc:
        errors.append(f"{label}: cannot read {path} ({exc.__class__.__name__})")
        return None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        errors.append(f"{label}: {path} is not valid JSON ({exc})")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label}: {path} is not a JSON object")
        return None
    return value


def envelope_stamps(
    path: Path,
    label: str,
    allow_round_minute: bool,
    errors: list[str],
    warnings: list[str],
) -> list[tuple[str, datetime | None, str]]:
    """The (started_at, completed_at) pair of a report or adjudication."""

    envelope = read_envelope(path, label, errors)
    stamps: list[tuple[str, datetime | None, str]] = []
    for field in ("started_at", "completed_at"):
        name = f"{label}.{field}"
        if envelope is None:
            # The file itself already produced an error; record the slot so the
            # emitted chain still shows every link that was meant to be checked.
            stamps.append((name, None, "UNREADABLE"))
            continue
        raw = envelope.get(field)
        stamp = parse_stamp(raw, name, errors)
        if stamp is not None:
            check_round_minute(
                stamp, name, "asserted", allow_round_minute, errors, warnings
            )
        stamps.append(
            (
                name,
                stamp.utc if stamp is not None else None,
                raw if isinstance(raw, str) else repr(raw),
            )
        )
    return stamps


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the D-069 section 7 chronology invariant: freeze commit < "
            "report.started_at < report.completed_at < adjudication.started_at "
            "< adjudication.completed_at < containing commit."
        )
    )
    parser.add_argument(
        "--freeze-commit",
        required=True,
        help="Commit that froze the contract and driver, before any output.",
    )
    parser.add_argument(
        "--report",
        required=True,
        type=Path,
        help="Evidence report JSON carrying top-level started_at/completed_at.",
    )
    parser.add_argument(
        "--adjudication",
        required=True,
        type=Path,
        help="Harness result envelope for the adjudication.",
    )
    parser.add_argument(
        "--containing-commit",
        help=(
            "Commit containing the adjudication. Omit when it does not exist "
            f"yet; the bound then becomes the current time, reported as "
            f"{PENDING_SOURCE!r}."
        ),
    )
    parser.add_argument(
        "--allow-round-minute",
        action="store_true",
        help=(
            "Downgrade round-minute report/adjudication stamps from error to "
            "warning. The default is to fail: the D-065 audit found asserted "
            "round-minute stamps, so this is only for a documented legitimate "
            "coincidence. Git commit times already warn rather than fail and "
            "are unaffected by this flag."
        ),
    )
    args = parser.parse_args()

    repo = root()
    errors: list[str] = []
    warnings: list[str] = []
    chain: list[tuple[str, datetime | None, str]] = []

    freeze_stamp, freeze_raw = commit_time(
        repo, args.freeze_commit, "freeze_commit", errors
    )
    if freeze_stamp is not None:
        check_round_minute(
            freeze_stamp, "freeze_commit", "git", args.allow_round_minute, errors, warnings
        )
    chain.append(
        (
            "freeze_commit",
            freeze_stamp.utc if freeze_stamp is not None else None,
            freeze_raw or args.freeze_commit,
        )
    )

    chain.extend(
        envelope_stamps(args.report, "report", args.allow_round_minute, errors, warnings)
    )
    chain.extend(
        envelope_stamps(
            args.adjudication, "adjudication", args.allow_round_minute, errors, warnings
        )
    )

    if args.containing_commit:
        containing_source: str = args.containing_commit
        containing_stamp, containing_raw = commit_time(
            repo, args.containing_commit, "containing_commit", errors
        )
        if containing_raw is not None:
            containing_source = containing_raw
        containing_value = containing_stamp.utc if containing_stamp is not None else None
        if containing_stamp is not None:
            check_round_minute(
                containing_stamp,
                "containing_commit",
                "git",
                args.allow_round_minute,
                errors,
                warnings,
            )
    else:
        containing_source = PENDING_SOURCE
        containing_value = datetime.now(timezone.utc)
    chain.append(("containing_commit", containing_value, containing_source))

    # Strictly increasing, on the UTC normalisation of every stamp -- the value
    # the round-minute rule was also applied to. A slot that failed to parse is
    # already an error, so it is compared past rather than around: the last
    # stamp that did parse stays the predecessor, and an inversion spanning the
    # gap is still caught.
    previous_name: str | None = None
    previous_value: datetime | None = None
    for name, value, _source in chain:
        if value is None:
            continue
        if previous_value is not None and previous_value >= value:
            relation = "equals" if previous_value == value else "precedes"
            errors.append(
                f"chronology violation: {name} "
                f"({value.isoformat()}) {relation} "
                f"{previous_name} "
                f"({previous_value.isoformat()}); the "
                "chain must be strictly increasing"
            )
        previous_name, previous_value = name, value

    containing_report = (
        PENDING_SOURCE if not args.containing_commit else args.containing_commit
    )
    if errors:
        print(
            json.dumps(
                {
                    "ok": False,
                    "containing_commit": containing_report,
                    "errors": errors,
                    # Carried onto the failure path too: a non-blocking
                    # observation must not vanish because something else failed.
                    "warnings": warnings,
                },
                indent=2,
            )
        )
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "ok": True,
                "containing_commit": containing_report,
                "ordered": [
                    {
                        "name": name,
                        "timestamp": value.isoformat(),
                        "source": source,
                    }
                    for name, value, source in chain
                    if value is not None
                ],
                "warnings": warnings,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
