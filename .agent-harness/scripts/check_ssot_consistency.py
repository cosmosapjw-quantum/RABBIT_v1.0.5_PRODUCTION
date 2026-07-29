#!/usr/bin/env python3
"""Machine check that the single-source-of-truth surfaces agree with each other.

Why this exists
---------------
Adversarial-audit finding F-D065-05 recorded exactly the failure this script
prevents: the gate registry said PASS while the prose handoff surfaces said
FAIL, and the hook-fixture count was written four different ways across four
files. Both were repaired by hand. This is the permanent check so neither can
recur silently.

Authority model
---------------
``.agent-harness/context/GATE_REGISTRY.json`` and
``.agent-harness/context/CLAIM_REGISTRY.jsonl`` are AUTHORITATIVE. Prose is
checked against them, never the reverse. This script never reads a status out
of prose into a registry and never proposes a registry edit.

Design principle: under-detection must FAIL, not pass
----------------------------------------------------
The first version of this checker only walked assertions it happened to find
in prose, so anything it failed to parse -- a phrasing the extractor did not
recognise, a section the region rule did not reach, a gate nobody wrote down
at all -- was silently unchecked. Flipping both ``fail`` gates to ``pass`` and
deleting the two sentences that mentioned them produced
``{"ok": true, "status_assertions_checked": 0}``: F-D065-05 again, with the
polarity reversed. Every rule below is therefore built so that a gap in
coverage is loud:

* COVERAGE FLOOR. Every gate in GATE_REGISTRY.json must be asserted at least
  once in a live region, and ``status_assertions_checked == 0`` is an error
  unconditionally. A missed phrasing or an unreached section can no longer
  hide; it becomes a coverage failure naming the gate.
* LIVE BY DEFAULT. A section is scanned unless it is *provably* superseded.
  Undetermined status means live.
* NO VACUOUS TESTS. A declaration that is missing a field is a hard error, and
  no field is ever used as a substring needle without first proving it is
  non-empty (``"" in line`` is always True, which once disabled two checks).
* EXPLICIT, JUSTIFIED, SELF-EXPIRING EXCEPTIONS. The few places where prose
  legitimately disagrees with a registry are declared one by one in
  SSOT_FACTS.json with a reason, and each declaration is re-verified on every
  run: a declaration that no longer describes the file is an error, so an
  exception cannot outlive the situation that justified it.

Region rule (which prose is authoritative right now)
----------------------------------------------------
The handoff documents are append-only overlay records. A superseded section
legitimately still says ``G-HARNESS-INTEGRITY=PASS`` because that is what was
true when it was written; rewriting history to satisfy a linter would destroy
the audit trail. So superseded sections are excluded -- but only when their
supersession can be *proved*, never by document position:

* ``docs/harness/PROJECT_STATE.md`` and ``docs/harness/NEXT_SESSION_PROMPT.md``
  are split at ``## `` headings. A section is SUPERSEDED when its heading says
  ``superseded``, or when the heading carries an ISO date older than the
  newest dated heading in that file. Everything else -- the preamble, every
  undated standing section such as ``## Objective and authority boundary`` or
  ``## Next action``, and every dated section at the newest date -- is LIVE and
  is scanned. Undated standing sections used to fall outside the checked
  region entirely, which let a registry-contradicting status be planted in a
  genuinely live part of the handoff.
* ORDERING IS PARSED, NOT ASSUMED. The newest overlay is the one with the
  greatest parsed date, not the first one in the file, so appending a newer
  overlay at the end of an append-only record cannot leave a stale section
  reading as authoritative. Exactly one section per file may be marked
  controlling at the newest controlling date; a tie, an undated controlling
  heading, an unparseable date, or two dates in one heading is an error.
  A missing controlling marker is an error, not a skip.
* ``.agent-harness/context/SHARED_CONTEXT.md``: the ``- Current milestone:``
  bullet plus the ``## Known disputes and open questions`` Q-table. The frozen
  contract/evidence narrative between them is history -- it contains lines such
  as "At D-030, both ... were FAIL" -- and is not scanned.
* ``docs/harness/DECISION_LOG.md`` and ``docs/harness/VALIDATION_LEDGER.md``:
  the top data row of the leading table only. Every older row is the dated
  record of what was true then. Newest-first is *enforced*: the top row must
  carry the greatest date in that table, so a newer row appended at the bottom
  cannot hide behind the assumption that the top row is current.
* ``.agent-harness/context/FROZEN_DECISIONS.md``: frozen by definition, so it is
  never scanned for status assertions. It contributes only the decision
  inventory and one commit-pinned numeric fact (below).

What counts as an assertion
---------------------------
Two forms are recognised, and nothing else.

1. PROSE. Each line of a live region is cut into clauses at hard boundaries
   (table cell ``|``, sentence end, ``;``, ``--``, a leading list marker).
   Inside a clause, every registry id binds to the NEAREST uppercase status
   token (PASS, FAIL, VALIDATED, IMPLEMENTED, SPECIFIED, FORBIDDEN, DEPRECATED,
   PROPOSED, DERIVED) within ``ASSERTION_WINDOW`` characters, in either
   direction -- so both "``G-A`` and ``G-B`` remain PASS" and "FAIL for
   ``G-A``" are read, where the original left-to-right-only rule saw only the
   first. Negation is explicit: a negation cue between the id and its token, or
   immediately before the token, means the line asserts nothing, so "is no
   longer FAIL" can never be recorded as asserting FAIL. A bare mention of an
   id asserts nothing. An id that sits equidistant between two *conflicting*
   tokens is reported as unresolved rather than guessed at or skipped.

2. STATUS-BOARD TABLE. The ``|`` clause boundary is deliberate -- a status in a
   ledger's Result cell must never bind to an id in its Notes cell -- so a
   plain markdown row cannot assert across cells. A table may instead declare
   itself a status board by header: exactly one header cell named ``Gate``,
   ``Claim``, ``Gate ID``, ``Claim ID``, ``ID`` or ``Gate/Claim``, and exactly
   one named ``Status``. In such a table each data row binds the ids in its id
   cell to the single status token in its status cell. A board row whose id
   cell names an id but whose status cell holds no status token, or more than
   one, is reported as unresolved. No existing table in this repository
   declares a ``Status`` header, so this recognises only tables written to be
   recognised.

Fenced code blocks are blanked before scanning because they quote historical
command output.

Coverage floor
--------------
Every gate id in GATE_REGISTRY.json must be asserted at least once across the
live regions. There is no exemption list for gates: a gate the handoff never
states is a gate whose prose cannot be checked at all.

Claims are handled differently, and the reason is written down rather than
assumed. Most of the 21 registry claims are bounded findings sealed at their
decision (D-027..D-030, W3/W5, the B3-v2 design claim) or F-10C1/F-10C2
baseline claims; their status is carried by FROZEN_DECISIONS.md and the audit
report that established them, and the rolling handoff does not restate them.
Requiring all 21 in live prose would force ceremonial restatement, which is a
different kind of lie. So SSOT_FACTS.json declares
``coverage_policy.claims.exempt`` -- an explicit list, each entry carrying a
reason -- and every claim NOT on that list is REQUIRED to have live coverage.
Silence is never the permissive option: a claim added to CLAIM_REGISTRY.jsonl
tomorrow is required by default until someone writes down why it is not. An
exemption naming a claim that no longer exists is an error, so the list cannot
rot into a blanket pass.

Declared assertion exemptions
-----------------------------
``assertion_exemptions`` in SSOT_FACTS.json names individual lines in live
regions whose registry disagreement is narrative that the same section then
corrects -- for example ``## Next action`` in PROJECT_STATE.md recounts the
D-045 grant ("``G-F10C1-REGRESSION`` stays FAIL") before recording that
D-049..D-051 made it PASS. Each entry must name the file, a substring that
identifies exactly one line in exactly one live section, the id, the status
that line asserts, and a substring identifying a LATER line in the SAME section
that asserts the registry's current status. All of that is re-verified every
run: if the narrative line, the correcting line, or their order changes, the
exemption is an error rather than a silent pass. An exemption cannot be used to
excuse a fresh contradiction, because a fresh contradiction has no correction
after it.

Declared numeric facts
----------------------
``.agent-harness/context/SSOT_FACTS.json`` declares numbers that appear in prose
and must agree everywhere. A fact is a value plus the commit it holds at, not a
bare number, so a superseded value is legal exactly where it is dated:

* role ``current``    -- a live surface; must equal the declared current value.
* role ``frozen``     -- a frozen row pinned to a commit; must equal the value
                         measured at that commit, so "helpfully" refreshing a
                         frozen row is itself an error.
* role ``historical`` -- narrative; must equal the current value, or a prior
                         value whose ``as_of_commit`` is cited on the same line.
                         An undated superseded value is an error.

Then, per file: if one fact is asserted with two different values and neither
carries an as-of commit, that file contradicts itself at a single commit. That
is the F-D065-05 class and it fails whatever the roles are. Every declared
assertion must also still match something; a stale ``SSOT_FACTS.json`` is an
error rather than a silent pass.

The schema is validated strictly: unknown keys, missing keys, and empty commit
strings are hard errors. Dropping ``as_of_commit`` from a ``prior`` entry used
to leave an empty needle, and ``"" in line`` is always True, so one deleted
JSON key silently switched off both the undated-superseded-value rule and the
self-contradiction rule that depends on it.

Decision inventory
------------------
Every decision that has landed must be registered on both decision surfaces.
The inventory is the union of ids with a ``| D-NNN |`` row in
FROZEN_DECISIONS.md and ids with retained artifacts. Artifacts are looked for
wherever this project has actually put decision code, which is not only the
audit tree: a ``docs/audit/*_DNNN_*`` document, a ``dNNN_*`` filename, a
``D-NNN`` citation inside ``scripts/audit``, ``.agent-harness/scripts``,
``.agent-harness/tests``, ``.codex/hooks`` or ``tests``, or a
``run-...-dNNN-...`` run directory. The harness-script root matters: the
admission-binding and cost-discipline lanes shipped their code there and
nowhere else, so a scan limited to ``scripts/audit`` would have missed exactly
the "code but no SSOT row" case this check exists to catch.

Decision ids are matched by ONE shared pattern: ``D-`` followed by exactly three
digits, so the entire 000-999 range is covered. The earlier code hardcoded a
leading zero in five separate patterns, so the first decision numbered past 099
would have become invisible to the inventory -- silently unchecked rather than
loudly rejected, which is the same "silence is permissive" shape as an absent
coverage floor.

Note for anyone editing this file: the roots above are scanned for cited
decision ids, and this file is under one of them. Do not write a literal
decision id here for an id that has no FROZEN_DECISIONS.md row -- an
illustrative id in a comment would be inventoried as a real landed decision.
Write ``D-NNN`` or spell the number out instead.

Each inventoried id is required to have a FROZEN_DECISIONS.md row and a
DECISION_LOG.md entry, and both directions are reported. Three deliberate
bounds keep it honest:

* Ids below ``DECISION_ID_ERA_FLOOR`` are exempt, because DECISION_LOG.md rows
  older than D-027 are dated prose that carries no decision id at all and
  therefore cannot be matched to one.
* From ``STRICT_LOG_KEYING_FLOOR`` upward the log uses one convention,
  ``Record D-NNN:`` (or ``Record D-NNN/D-NNN:`` for a joint row), so a row must
  declare the id, not merely mention it. Below that floor only a mention is
  required: the older rows key their decisions inconsistently -- D-033 is named
  only in its consequences cell, D-043 mid-sentence, D-054/D-055/D-056 jointly
  -- and a stricter rule would be false-positive on the real corpus rather than
  on a hypothetical one.
* An id mentioned in DECISION_LOG.md with neither a row nor any artifact is a
  forward reference to a decision that has not happened yet. It is reported in
  the success payload, not failed.

Usage
-----
``venv/bin/python .agent-harness/scripts/check_ssot_consistency.py``

Exit 0 with ``{"ok": true, ...}``; exit 1 with ``{"ok": false, "errors": [...]}``.
Fail closed: an unreadable or unparseable input is an error, never a skip.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from _harness import load_json, root

GATE_REGISTRY = ".agent-harness/context/GATE_REGISTRY.json"
CLAIM_REGISTRY = ".agent-harness/context/CLAIM_REGISTRY.jsonl"
FACTS_FILE = ".agent-harness/context/SSOT_FACTS.json"
FROZEN_DECISIONS = ".agent-harness/context/FROZEN_DECISIONS.md"
DECISION_LOG = "docs/harness/DECISION_LOG.md"
SHARED_CONTEXT = ".agent-harness/context/SHARED_CONTEXT.md"

# Prose surfaces split into dated/undated sections and scanned live-by-default.
OVERLAY_SURFACES = (
    "docs/harness/PROJECT_STATE.md",
    "docs/harness/NEXT_SESSION_PROMPT.md",
)
LEDGER_SURFACES = (
    "docs/harness/DECISION_LOG.md",
    "docs/harness/VALIDATION_LEDGER.md",
)

STATUS_TOKENS = (
    "PASS",
    "FAIL",
    "VALIDATED",
    "IMPLEMENTED",
    "SPECIFIED",
    "FORBIDDEN",
    "DEPRECATED",
    "PROPOSED",
    "DERIVED",
)
STATUS_RE = re.compile(r"(?<![A-Za-z0-9_-])(" + "|".join(STATUS_TOKENS) + r")(?![A-Za-z0-9_-])")
GATE_STATUSES = {"PASS", "FAIL"}
CLAIM_STATUSES = set(STATUS_TOKENS) - {"PASS", "FAIL"}

# How far an id may sit from a status token and still be its subject.
ASSERTION_WINDOW = 200
# Hard boundaries: a status token never binds across a table cell or a sentence.
BOUNDARY_RE = re.compile(r"\|| [-]{2} |\. |\.$|; |\? |! |^\s*[-*] ")
# A cue between an id and its token (or just before the token) means the line
# denies the status rather than asserting it: "is no longer FAIL" asserts nothing.
NEGATION_RE = re.compile(
    r"(?<![A-Za-z])(no longer|not|never|n't|rather than|instead of|neither|nor|no more)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)

# One pattern for every decision id: `D-` plus exactly three digits. Five
# separate hardcoded `D-0\d\d` patterns meant that the first decision numbered
# past 099 would silently stop being inventoried.
DECISION_ID_RE = re.compile(r"(?<![A-Za-z0-9])D-(\d{3})(?![0-9])")
FROZEN_ROW_RE = re.compile(r"^\|\s*(D-\d{3})\s*\|")
RECORD_DECL_RE = re.compile(r"Record ((?:D-\d{3}/)*D-\d{3})")
# Filename forms: `..._D067_...` for audit docs, `d067_...` for scripts.
FILENAME_ID_RE = re.compile(r"(?:^|[^A-Za-z0-9])[Dd](\d{3})(?![0-9])")
RUN_DIR_ID_RE = re.compile(r"-d(\d{3})(?![0-9])")
# DECISION_LOG rows older than this carry no decision id at all; see docstring.
DECISION_ID_ERA_FLOOR = 27
# From here up, a log row must declare its decision as `Record D-NNN`, not just
# mention it, so deleting a row cannot be masked by another row's cross-reference.
STRICT_LOG_KEYING_FLOOR = 45

# Where decision code actually lands. `docs/audit` is matched on filename; the
# rest are matched on filename and on cited ids in file text.
ARTIFACT_DOC_ROOT = "docs/audit"
ARTIFACT_RUN_ROOT = ".agent-harness/runs"
ARTIFACT_CODE_ROOTS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("scripts/audit", ("*.py", "*.json"), False),
    (".agent-harness/scripts", ("*.py",), False),
    (".agent-harness/tests", ("*.py",), False),
    (".codex/hooks", ("*.py",), False),
    ("tests", ("*.py",), True),
)

ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
LOOSE_DATE_RE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}")
SUPERSEDED_RE = re.compile(r"superseded", re.IGNORECASE)
CONTROLLING_RE = re.compile(r"controlling", re.IGNORECASE)


def decision_number(identifier: str) -> int:
    """`D-067` -> 67. Ids are fixed-width, so this also orders them."""
    return int(identifier.split("-", 1)[1])


def read_text(repo: Path, rel: str, errors: list[str]) -> str | None:
    """Read a required text file. Unreadable is an error, never a skip."""
    path = repo / rel
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"{rel}: unreadable ({exc.__class__.__name__}); cannot be checked.")
        return None


def blank_fenced_blocks(text: str) -> list[str]:
    """Return lines with fenced-code content blanked, preserving line numbers."""
    lines = text.splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        if line.lstrip().startswith("```"):
            inside = not inside
            out.append("")
            continue
        out.append("" if inside else line)
    return out


def load_gates(repo: Path, errors: list[str]) -> dict[str, str]:
    try:
        registry = load_json(repo / GATE_REGISTRY)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{GATE_REGISTRY}: unreadable or invalid JSON ({exc.__class__.__name__}).")
        return {}
    gates: dict[str, str] = {}
    entries = registry.get("gates") if isinstance(registry, dict) else None
    if not isinstance(entries, list):
        errors.append(f"{GATE_REGISTRY}: no 'gates' array; the authority surface is unusable.")
        return {}
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append(f"{GATE_REGISTRY}: gate entry is not an object.")
            continue
        gate_id = str(entry.get("gate_id") or "")
        status = str(entry.get("status") or "").upper()
        if not gate_id:
            errors.append(f"{GATE_REGISTRY}: gate entry without a gate_id.")
            continue
        if status not in GATE_STATUSES:
            errors.append(f"{GATE_REGISTRY}: {gate_id} has status {status or '(empty)'!r}.")
            continue
        if gate_id in gates:
            errors.append(f"{GATE_REGISTRY}: duplicate gate_id {gate_id}.")
        gates[gate_id] = status
    return gates


def load_claims(repo: Path, errors: list[str]) -> dict[str, str]:
    text = read_text(repo, CLAIM_REGISTRY, errors)
    if text is None:
        return {}
    claims: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"{CLAIM_REGISTRY}:{number}: malformed JSON line.")
            continue
        if not isinstance(row, dict):
            errors.append(f"{CLAIM_REGISTRY}:{number}: line is not an object.")
            continue
        claim_id = str(row.get("claim_id") or "")
        status = str(row.get("status") or "").upper()
        if not claim_id:
            errors.append(f"{CLAIM_REGISTRY}:{number}: no claim_id.")
            continue
        if status not in CLAIM_STATUSES:
            errors.append(f"{CLAIM_REGISTRY}:{number}: {claim_id} has status {status or '(empty)'!r}.")
            continue
        if claim_id in claims:
            errors.append(f"{CLAIM_REGISTRY}: duplicate claim_id {claim_id}.")
        claims[claim_id] = status
    return claims


def build_id_regex(ids: list[str]) -> re.Pattern[str]:
    ordered = sorted(ids, key=len, reverse=True)
    joined = "|".join(re.escape(value) for value in ordered)
    return re.compile(r"(?<![A-Za-z0-9-])(" + joined + r")(?![A-Za-z0-9-])")


# --------------------------------------------------------------------------
# Live-region extraction. A region is (label, [(line_number, line_text)]).
# --------------------------------------------------------------------------


class Region:
    """A contiguous run of lines that is authoritative right now."""

    def __init__(self, rel: str, label: str, lines: list[tuple[int, str]]) -> None:
        self.rel = rel
        self.label = label
        self.lines = lines


def parse_heading_date(heading: str, rel: str, errors: list[str]) -> date | None:
    """The ISO date in a heading. Ambiguous or unparseable is an error."""
    strict = ISO_DATE_RE.findall(heading)
    loose = LOOSE_DATE_RE.findall(heading)
    if len(loose) > len(strict):
        errors.append(
            f"{rel}: heading {heading.strip()!r} carries a date that is not ISO "
            "YYYY-MM-DD; overlay ordering cannot be determined."
        )
        return None
    if not strict:
        return None
    if len({*strict}) > 1:
        errors.append(
            f"{rel}: heading {heading.strip()!r} carries more than one date "
            f"({', '.join(sorted(set(strict)))}); overlay ordering is ambiguous."
        )
        return None
    try:
        return date.fromisoformat(strict[0])
    except ValueError:
        errors.append(
            f"{rel}: heading {heading.strip()!r} carries an unparseable date "
            f"{strict[0]!r}; overlay ordering cannot be determined."
        )
        return None


def overlay_regions(lines: list[str], rel: str, errors: list[str]) -> list[Region]:
    """Every section of an overlay surface that is not provably superseded.

    Live by default. A section is excluded only when its heading says
    ``superseded`` or when its parsed date is older than the newest dated
    heading in the file. Position in the file decides nothing.
    """
    starts = [index for index, line in enumerate(lines) if line.startswith("## ")]
    blocks: list[tuple[int, int, str]] = []
    if not starts or starts[0] > 0:
        blocks.append((0, starts[0] if starts else len(lines), "(preamble)"))
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        blocks.append((start, end, lines[start]))

    dates: dict[int, date | None] = {}
    controlling: list[int] = []
    for index, (start, _end, heading) in enumerate(blocks):
        if heading == "(preamble)":
            dates[index] = None
            continue
        dates[index] = parse_heading_date(heading, rel, errors)
        if CONTROLLING_RE.search(heading) and not SUPERSEDED_RE.search(heading):
            controlling.append(index)

    if not controlling:
        errors.append(
            f"{rel}: no '## ...controlling...' section heading; the controlling "
            "overlay cannot be identified, so the surface cannot be trusted."
        )
    undated_controlling = [index for index in controlling if dates[index] is None]
    for index in undated_controlling:
        errors.append(
            f"{rel}: section {blocks[index][2].strip()!r} is marked controlling but "
            "carries no parseable date, so it cannot be ordered against the other "
            "overlays."
        )
    dated_controlling = [
        (index, stamp) for index in controlling for stamp in [dates[index]] if stamp is not None
    ]
    if dated_controlling:
        newest_controlling = max(stamp for _index, stamp in dated_controlling)
        tied = [index for index, stamp in dated_controlling if stamp == newest_controlling]
        if len(tied) > 1:
            headings = "; ".join(blocks[index][2].strip() for index in tied)
            errors.append(
                f"{rel}: {len(tied)} sections claim to be the newest controlling "
                f"overlay at {newest_controlling.isoformat()} ({headings}). Exactly "
                "one section can be authoritative."
            )

    known_dates = [value for value in dates.values() if value is not None]
    newest = max(known_dates) if known_dates else None

    regions: list[Region] = []
    for index, (start, end, heading) in enumerate(blocks):
        if heading != "(preamble)" and SUPERSEDED_RE.search(heading):
            continue
        stamp = dates[index]
        if stamp is not None and newest is not None and stamp < newest:
            continue
        regions.append(
            Region(
                rel,
                heading.strip() if heading != "(preamble)" else "(preamble)",
                [(number + 1, lines[number]) for number in range(start, end)],
            )
        )
    return regions


def shared_context_region(lines: list[str], errors: list[str]) -> list[Region]:
    """The 'Current milestone' bullet plus the open-questions Q-table."""
    collected: list[tuple[int, str]] = []
    milestone = [
        (number + 1, line)
        for number, line in enumerate(lines)
        if line.lstrip().startswith("- Current milestone:")
    ]
    if not milestone:
        errors.append(
            f"{SHARED_CONTEXT}: no '- Current milestone:' bullet; the controlling "
            "region cannot be located."
        )
    collected.extend(milestone)

    start: int | None = None
    for index, line in enumerate(lines):
        if line.startswith("## ") and "open questions" in line.lower():
            start = index
            break
    if start is None:
        errors.append(
            f"{SHARED_CONTEXT}: no '## Known disputes and open questions' section; "
            "the Q-table cannot be located."
        )
    else:
        end = len(lines)
        for index in range(start + 1, len(lines)):
            if lines[index].startswith("## "):
                end = index
                break
        collected.extend((number + 1, lines[number]) for number in range(start, end))
    return [Region(SHARED_CONTEXT, "current milestone + open questions", collected)]


def top_table_row(lines: list[str], rel: str, errors: list[str]) -> list[Region]:
    """The top data row of the leading table, with newest-first enforced.

    The top row is only authoritative if it really is the newest row, so the
    date column of every data row is parsed and compared. A newer row appended
    at the bottom is an error, not an invisible supersession.
    """
    separator: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and set(stripped) <= set("|-: "):
            separator = index
            break
    if separator is None:
        errors.append(f"{rel}: no markdown table separator; the top row cannot be located.")
        return []

    rows: list[tuple[int, str, date | None]] = []
    for index in range(separator + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped.startswith("|"):
            if stripped:
                break
            continue
        first_cell = stripped.strip("|").split("|", 1)[0].strip()
        match = ISO_DATE_RE.search(first_cell)
        stamp: date | None = None
        if match:
            try:
                stamp = date.fromisoformat(match.group(0))
            except ValueError:
                stamp = None
        if match and stamp is None:
            errors.append(
                f"{rel}:{index + 1}: row date {match.group(0)!r} is unparseable; "
                "newest-first ordering cannot be verified."
            )
        rows.append((index + 1, lines[index], stamp))

    if not rows:
        errors.append(f"{rel}: leading table has no data row; the top row cannot be located.")
        return []

    top_number, top_line, top_date = rows[0]
    if top_date is None:
        errors.append(
            f"{rel}:{top_number}: the top data row carries no ISO date, so it cannot "
            "be shown to be the newest row."
        )
    else:
        newer = [
            (number, stamp)
            for number, _text, stamp in rows[1:]
            if stamp is not None and stamp > top_date
        ]
        if newer:
            number, stamp = max(newer, key=lambda item: item[1])
            errors.append(
                f"{rel}:{number}: a row dated {stamp.isoformat()} sits below the top "
                f"row dated {top_date.isoformat()}. This table is read newest-first, "
                "so a later row appended at the bottom would never be checked."
            )
    return [Region(rel, "leading table top row", [(top_number, top_line)])]


# --------------------------------------------------------------------------
# Status assertions
# --------------------------------------------------------------------------


BOARD_ID_HEADERS = {
    "gate",
    "gates",
    "claim",
    "claims",
    "id",
    "gate id",
    "claim id",
    "gate/claim",
    "gate or claim",
}
BOARD_STATUS_HEADERS = {"status", "registry status", "current status"}


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _is_table_row(line: str) -> bool:
    return line.strip().startswith("|")


def _is_separator_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and set(stripped) <= set("|-: ")


def _normalise_header(cell: str) -> str:
    return re.sub(r"\s+", " ", cell.strip().strip("`*_ ").lower())


def find_board_assertions(
    region: Region, id_regex: re.Pattern[str]
) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
    """Read tables that declare themselves status boards by header.

    A plain markdown row cannot assert across cells, because the ``|`` boundary
    is what stops a ledger Result cell binding to an id in its Notes cell. A
    table that names one id column and one ``Status`` column is unambiguous, so
    it is read row by row.
    """
    found: list[tuple[int, str, str]] = []
    unresolved: list[tuple[int, str, str]] = []
    lines = region.lines
    index = 0
    while index + 1 < len(lines):
        header_number, header_text = lines[index]
        _separator_number, separator_text = lines[index + 1]
        if not (_is_table_row(header_text) and _is_separator_row(separator_text)):
            index += 1
            continue
        headers = [_normalise_header(cell) for cell in _cells(header_text)]
        id_columns = [i for i, name in enumerate(headers) if name in BOARD_ID_HEADERS]
        status_columns = [i for i, name in enumerate(headers) if name in BOARD_STATUS_HEADERS]
        del header_number
        if len(id_columns) != 1 or len(status_columns) != 1:
            index += 1
            continue
        id_column, status_column = id_columns[0], status_columns[0]
        row = index + 2
        while row < len(lines) and _is_table_row(lines[row][1]):
            number, text = lines[row]
            cells = _cells(text)
            row += 1
            if max(id_column, status_column) >= len(cells):
                continue
            identifiers = [m.group(1) for m in id_regex.finditer(cells[id_column])]
            if not identifiers:
                continue
            statuses = {m.group(1) for m in STATUS_RE.finditer(cells[status_column])}
            if len(statuses) != 1:
                why = (
                    "its Status cell holds no status token"
                    if not statuses
                    else "its Status cell holds " + "/".join(sorted(statuses))
                )
                for identifier in identifiers:
                    unresolved.append((number, identifier, why))
                continue
            status = statuses.pop()
            for identifier in identifiers:
                found.append((number, identifier, status))
        index = row
    return found, unresolved


def split_clauses(line: str) -> list[tuple[int, str]]:
    """Cut a line at hard boundaries; return (offset_in_line, clause_text)."""
    clauses: list[tuple[int, str]] = []
    position = 0
    for boundary in BOUNDARY_RE.finditer(line):
        if boundary.start() > position:
            clauses.append((position, line[position : boundary.start()]))
        position = max(position, boundary.end())
    if position < len(line):
        clauses.append((position, line[position:]))
    return clauses


def _negated(clause: str, id_span: tuple[int, int], token_span: tuple[int, int]) -> bool:
    """True when a cue between (or just before) the pair denies the status."""
    low = min(id_span[1], token_span[1])
    high = max(id_span[0], token_span[0])
    between = clause[low:high] if high > low else ""
    lead = clause[max(0, token_span[0] - 30) : token_span[0]]
    return bool(NEGATION_RE.search(between) or NEGATION_RE.search(lead))


def find_assertions(
    region: Region, id_regex: re.Pattern[str]
) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
    """Return (assertions, unresolved) for one region.

    ``assertions`` is (line_number, id, status). ``unresolved`` is
    (line_number, id, why) for ids that sit between conflicting status tokens:
    those are reported rather than guessed at, because a silent skip is exactly
    how an unparsed line becomes an unchecked line.
    """
    found, unresolved = find_board_assertions(region, id_regex)
    for number, line in region.lines:
        for _offset, clause in split_clauses(line):
            ids = [(m.start(), m.end(), m.group(1)) for m in id_regex.finditer(clause)]
            tokens = [(m.start(), m.end(), m.group(1)) for m in STATUS_RE.finditer(clause)]
            if not ids or not tokens:
                continue
            for id_start, id_end, identifier in ids:
                scored = []
                for token_start, token_end, status in tokens:
                    if token_start >= id_end:
                        gap = token_start - id_end
                    elif token_end <= id_start:
                        gap = id_start - token_end
                    else:
                        gap = 0
                    scored.append((gap, token_start, token_end, status))
                scored.sort(key=lambda item: item[0])
                gap, token_start, token_end, status = scored[0]
                if gap > ASSERTION_WINDOW:
                    continue
                rivals = {item[3] for item in scored if item[0] == gap}
                if len(rivals) > 1:
                    unresolved.append(
                        (
                            number,
                            identifier,
                            "sits equidistant between conflicting status tokens "
                            + "/".join(sorted(rivals)),
                        )
                    )
                    continue
                if _negated(clause, (id_start, id_end), (token_start, token_end)):
                    continue
                found.append((number, identifier, status))
    # A board row whose cells also read as prose would otherwise be counted
    # twice; the assertion is the same either way.
    return _dedupe(found), _dedupe(unresolved)


def _dedupe(rows: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    seen: set[tuple[int, str, str]] = set()
    out: list[tuple[int, str, str]] = []
    for row in rows:
        if row in seen:
            continue
        seen.add(row)
        out.append(row)
    return out


def check_status_assertions(
    region: Region,
    gates: dict[str, str],
    claims: dict[str, str],
    id_regex: re.Pattern[str],
    exemptions: list[dict[str, Any]],
    errors: list[str],
) -> tuple[int, set[str]]:
    """Compare one live region against the registries. Returns (checked, covered)."""
    rel = region.rel
    checked = 0
    covered: set[str] = set()
    assertions, unresolved = find_assertions(region, id_regex)
    for number, identifier, why in unresolved:
        errors.append(
            f"{rel}:{number}: cannot resolve what status is asserted for {identifier}: "
            f"it {why}. Rewrite the line so one status binds to one id."
        )
    for number, identifier, status in assertions:
        checked += 1
        expected = gates.get(identifier, claims.get(identifier))
        if expected is None:
            errors.append(
                f"{rel}:{number}: live surface asserts {status} for {identifier}, "
                "which has no row in the gate or claim registry."
            )
            continue
        if status == expected:
            covered.add(identifier)
            continue
        excused = _matching_exemption(
            exemptions, rel, region, number, identifier, status, expected, errors
        )
        if excused:
            continue
        kind = "gate" if identifier in gates else "claim"
        errors.append(
            f"{rel}:{number}: live surface says {identifier}={status} but the "
            f"{kind} registry says {expected}."
        )
    return checked, covered


def _matching_exemption(
    exemptions: list[dict[str, Any]],
    rel: str,
    region: Region,
    number: int,
    identifier: str,
    status: str,
    expected: str,
    errors: list[str],
) -> bool:
    """True when a declared, still-valid narrative exemption covers this line."""
    for entry in exemptions:
        if entry["file"] != rel or entry["id"] != identifier:
            continue
        if entry["asserted_status"] != status:
            continue
        if entry["line_contains"] not in dict(region.lines).get(number, ""):
            continue
        # The section must correct itself after the narrative line.
        later = [
            (line_number, text)
            for line_number, text in region.lines
            if line_number > number and entry["superseded_by_line_contains"] in text
        ]
        if not later:
            errors.append(
                f"{FACTS_FILE}: the assertion exemption for {rel}:{number} ({identifier}"
                f"={status}) claims the section later says "
                f"{entry['superseded_by_line_contains']!r}, but no line after "
                f"{number} in {region.label!r} does. The exemption no longer "
                "describes the file."
            )
            return False
        correcting = Region(rel, region.label, later)
        corrected, _ = find_assertions(correcting, build_id_regex([identifier]))
        if not any(found_status == expected for _line, _id, found_status in corrected):
            errors.append(
                f"{FACTS_FILE}: the assertion exemption for {rel}:{number} ({identifier}"
                f"={status}) points at line {later[0][0]} as the correction, but that "
                f"line does not assert {identifier}={expected}. The exemption no longer "
                "describes the file."
            )
            return False
        return True
    return False


# --------------------------------------------------------------------------
# Declared facts, coverage policy, and assertion exemptions
# --------------------------------------------------------------------------

FACT_KEYS = {
    "fact_id",
    "description",
    "measurement",
    "value",
    "as_of_commit",
    "prior",
    "assertions",
}
FACT_REQUIRED = {"fact_id", "value", "as_of_commit", "assertions"}
PRIOR_KEYS = {"value", "as_of_commit", "note"}
PRIOR_REQUIRED = {"value", "as_of_commit"}
ASSERTION_KEYS = {"file", "line_contains", "value_regex", "role", "pinned_to_commit", "note"}
ASSERTION_REQUIRED = {"file", "value_regex", "role"}
EXEMPTION_KEYS = {
    "file",
    "line_contains",
    "id",
    "asserted_status",
    "superseded_by_line_contains",
    "reason",
}
DOCUMENT_KEYS = {"schema_version", "purpose", "facts", "coverage_policy", "assertion_exemptions"}


def _require_keys(
    where: str, entry: dict[str, Any], allowed: set[str], required: set[str], errors: list[str]
) -> bool:
    """Strict schema: unknown keys and missing/empty required keys are errors."""
    ok = True
    unknown = sorted(set(entry) - allowed)
    if unknown:
        errors.append(f"{FACTS_FILE}: {where} has unknown key(s) {', '.join(unknown)}.")
        ok = False
    for key in sorted(required):
        if key not in entry:
            errors.append(f"{FACTS_FILE}: {where} is missing required key {key!r}.")
            ok = False
        elif isinstance(entry[key], str) and not entry[key].strip():
            errors.append(f"{FACTS_FILE}: {where} has an empty {key!r}.")
            ok = False
    return ok


def load_facts_document(repo: Path, errors: list[str]) -> dict[str, Any]:
    try:
        document = load_json(repo / FACTS_FILE)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{FACTS_FILE}: unreadable or invalid JSON ({exc.__class__.__name__}).")
        return {}
    if not isinstance(document, dict):
        errors.append(f"{FACTS_FILE}: top level is not an object.")
        return {}
    unknown = sorted(set(document) - DOCUMENT_KEYS)
    if unknown:
        errors.append(f"{FACTS_FILE}: unknown top-level key(s) {', '.join(unknown)}.")
    return document


def load_coverage_policy(
    document: dict[str, Any], claims: dict[str, str], errors: list[str]
) -> set[str]:
    """Claim ids explicitly excused from needing live prose coverage.

    Anything not named here is required, so forgetting to write a policy entry
    fails the run instead of quietly widening what passes.
    """
    policy = document.get("coverage_policy")
    if policy is None:
        errors.append(
            f"{FACTS_FILE}: no 'coverage_policy'. Every claim would then be required "
            "to appear in live prose; declare the exemptions explicitly instead."
        )
        return set()
    if not isinstance(policy, dict):
        errors.append(f"{FACTS_FILE}: 'coverage_policy' is not an object.")
        return set()
    unknown = sorted(set(policy) - {"note", "claims"})
    if unknown:
        errors.append(f"{FACTS_FILE}: coverage_policy has unknown key(s) {', '.join(unknown)}.")
    claim_policy = policy.get("claims")
    if not isinstance(claim_policy, dict):
        errors.append(f"{FACTS_FILE}: coverage_policy.claims is missing or not an object.")
        return set()
    unknown = sorted(set(claim_policy) - {"exempt"})
    if unknown:
        errors.append(
            f"{FACTS_FILE}: coverage_policy.claims has unknown key(s) {', '.join(unknown)}."
        )
    entries = claim_policy.get("exempt")
    if not isinstance(entries, list):
        errors.append(f"{FACTS_FILE}: coverage_policy.claims.exempt is missing or not a list.")
        return set()
    exempt: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append(f"{FACTS_FILE}: coverage_policy.claims.exempt entry is not an object.")
            continue
        if not _require_keys(
            "coverage_policy.claims.exempt entry",
            entry,
            {"claim_id", "reason"},
            {"claim_id", "reason"},
            errors,
        ):
            continue
        claim_id = str(entry["claim_id"])
        if claim_id not in claims:
            errors.append(
                f"{FACTS_FILE}: coverage_policy exempts {claim_id}, which has no row in "
                f"{CLAIM_REGISTRY}. A stale exemption cannot be allowed to stand in for "
                "a live one."
            )
            continue
        if claim_id in exempt:
            errors.append(f"{FACTS_FILE}: coverage_policy exempts {claim_id} twice.")
        exempt.add(claim_id)
    return exempt


def load_assertion_exemptions(
    document: dict[str, Any],
    gates: dict[str, str],
    claims: dict[str, str],
    errors: list[str],
) -> list[dict[str, Any]]:
    entries = document.get("assertion_exemptions", [])
    if not isinstance(entries, list):
        errors.append(f"{FACTS_FILE}: 'assertion_exemptions' is not a list.")
        return []
    valid: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append(f"{FACTS_FILE}: assertion_exemptions entry is not an object.")
            continue
        if not _require_keys(
            "assertion_exemptions entry", entry, EXEMPTION_KEYS, EXEMPTION_KEYS, errors
        ):
            continue
        identifier = str(entry["id"])
        if identifier not in gates and identifier not in claims:
            errors.append(
                f"{FACTS_FILE}: assertion_exemptions names {identifier}, which has no "
                "row in the gate or claim registry."
            )
            continue
        status = str(entry["asserted_status"]).upper()
        if status not in STATUS_TOKENS:
            errors.append(
                f"{FACTS_FILE}: assertion_exemptions for {identifier} names an unknown "
                f"status {entry['asserted_status']!r}."
            )
            continue
        valid.append({**entry, "id": identifier, "asserted_status": status})
    return valid


def check_exemptions_used(
    exemptions: list[dict[str, Any]], used: set[int], errors: list[str]
) -> None:
    for index, entry in enumerate(exemptions):
        if index in used:
            continue
        errors.append(
            f"{FACTS_FILE}: the assertion exemption for {entry['file']} "
            f"({entry['id']}={entry['asserted_status']}, line containing "
            f"{entry['line_contains']!r}) matched nothing on this run. Either the line "
            "is gone or it is no longer in a live region; a dead exemption is removed, "
            "not kept."
        )


def check_facts(repo: Path, document: dict[str, Any], errors: list[str]) -> int:
    facts = document.get("facts")
    if not isinstance(facts, list) or not facts:
        errors.append(f"{FACTS_FILE}: no 'facts' array.")
        return 0

    checked = 0
    for fact in facts:
        if not isinstance(fact, dict):
            errors.append(f"{FACTS_FILE}: fact entry is not an object.")
            continue
        fact_id = str(fact.get("fact_id") or "(unnamed)")
        if not _require_keys(f"fact {fact_id}", fact, FACT_KEYS, FACT_REQUIRED, errors):
            continue
        current = fact.get("value")
        if not isinstance(current, int) or isinstance(current, bool):
            errors.append(f"{FACTS_FILE}: {fact_id} has no integer 'value'.")
            continue
        current_commit = str(fact["as_of_commit"]).strip()

        priors: dict[int, str] = {}
        prior_entries = fact.get("prior", [])
        if not isinstance(prior_entries, list):
            errors.append(f"{FACTS_FILE}: {fact_id} has a 'prior' that is not a list.")
            continue
        broken = False
        for entry in prior_entries:
            if not isinstance(entry, dict):
                errors.append(f"{FACTS_FILE}: {fact_id} has a 'prior' entry that is not an object.")
                broken = True
                continue
            if not _require_keys(
                f"fact {fact_id} prior entry", entry, PRIOR_KEYS, PRIOR_REQUIRED, errors
            ):
                broken = True
                continue
            value = entry["value"]
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(
                    f"{FACTS_FILE}: {fact_id} has a 'prior' value that is not an integer."
                )
                broken = True
                continue
            commit = str(entry["as_of_commit"]).strip()
            if value == current:
                errors.append(
                    f"{FACTS_FILE}: {fact_id} lists {value} as a prior, but that is also "
                    "the current value."
                )
                broken = True
                continue
            if value in priors:
                errors.append(f"{FACTS_FILE}: {fact_id} lists prior value {value} twice.")
                broken = True
                continue
            priors[value] = commit
        if broken:
            # A malformed declaration must never narrow what is enforced.
            continue
        known = {current: current_commit} | priors

        assertions = fact.get("assertions")
        if not isinstance(assertions, list) or not assertions:
            errors.append(f"{FACTS_FILE}: {fact_id} declares no assertion sites.")
            continue

        # Values asserted without an as-of commit, per file: more than one
        # distinct value means that file contradicts itself at one commit.
        undated: dict[str, dict[int, str]] = {}
        for assertion in assertions:
            if not isinstance(assertion, dict):
                errors.append(
                    f"{FACTS_FILE}: {fact_id} has an assertion entry that is not an object."
                )
                continue
            if not _require_keys(
                f"fact {fact_id} assertion", assertion, ASSERTION_KEYS, ASSERTION_REQUIRED, errors
            ):
                continue
            rel = str(assertion["file"])
            needle = str(assertion.get("line_contains") or "")
            pattern = str(assertion["value_regex"])
            role = str(assertion["role"])
            if role not in {"current", "frozen", "historical"}:
                errors.append(f"{FACTS_FILE}: {fact_id} assertion for {rel} has role {role!r}.")
                continue
            if role == "frozen" and not str(assertion.get("pinned_to_commit") or "").strip():
                errors.append(
                    f"{FACTS_FILE}: {fact_id} declares a frozen assertion in {rel} with no "
                    "'pinned_to_commit'; a frozen row without its commit cannot be checked."
                )
                continue
            try:
                value_regex = re.compile(pattern)
            except re.error as exc:
                errors.append(f"{FACTS_FILE}: {fact_id} has an invalid value_regex ({exc}).")
                continue

            text = read_text(repo, rel, errors)
            if text is None:
                continue
            matches = 0
            for number, line in enumerate(text.splitlines(), start=1):
                if needle and needle not in line:
                    continue
                for match in value_regex.finditer(line):
                    matches += 1
                    checked += 1
                    try:
                        value = int(match.group(1))
                    except (IndexError, ValueError):
                        errors.append(
                            f"{FACTS_FILE}: {fact_id} value_regex for {rel} does not "
                            "capture an integer."
                        )
                        continue
                    quoted = " ".join(match.group(0).split())
                    # `priors[value]` is guaranteed non-empty above, so this is a
                    # real substring test and never the vacuous `"" in line`.
                    dated = value != current and value in priors and priors[value] in line
                    if not dated:
                        undated.setdefault(rel, {}).setdefault(value, f"{rel}:{number}")
                    if role == "current":
                        if value != current:
                            errors.append(
                                f"{rel}:{number}: live surface asserts "
                                f"{fact_id}={value} ({quoted!r}), but the declared value "
                                f"is {current} (as of {current_commit})."
                            )
                    elif role == "frozen":
                        pinned = str(assertion["pinned_to_commit"]).strip()
                        expected = [
                            candidate
                            for candidate, commit in known.items()
                            if commit and commit == pinned
                        ]
                        if not expected:
                            errors.append(
                                f"{FACTS_FILE}: {fact_id} pins {rel} to commit "
                                f"{pinned}, which declares no value."
                            )
                        elif value != expected[0]:
                            errors.append(
                                f"{rel}:{number}: frozen row asserts {fact_id}={value} "
                                f"({quoted!r}), but the value measured at its pinned "
                                f"commit {pinned} is {expected[0]}. A frozen row is not "
                                "refreshed."
                            )
                    elif value != current:
                        if value not in priors:
                            errors.append(
                                f"{rel}:{number}: asserts {fact_id}={value} ({quoted!r}), "
                                "which was never measured at any recorded commit."
                            )
                        elif not dated:
                            errors.append(
                                f"{rel}:{number}: asserts the superseded value "
                                f"{fact_id}={value} ({quoted!r}) without citing its "
                                f"commit {priors[value]}; current is {current}."
                            )
            if matches == 0:
                errors.append(
                    f"{FACTS_FILE}: {fact_id} declares an assertion in {rel} "
                    f"(/{pattern}/) that no longer matches; the facts file is stale."
                )

        for rel, values in sorted(undated.items()):
            if len(values) > 1:
                detail = ", ".join(
                    f"{value} at {where}" for value, where in sorted(values.items())
                )
                errors.append(
                    f"{rel}: asserts {fact_id} with conflicting undated values ({detail}). "
                    "A surface that contradicts itself at one commit cannot be resolved "
                    "by precedence."
                )
    return checked


# --------------------------------------------------------------------------
# Decision inventory
# --------------------------------------------------------------------------


def landed_decision_ids(repo: Path, errors: list[str]) -> dict[str, list[str]]:
    """Decision ids that have already produced retained repository artifacts."""
    landed: dict[str, set[str]] = {}

    def record(identifier: str, evidence: str) -> None:
        landed.setdefault(identifier, set()).add(evidence)

    audit_docs = repo / ARTIFACT_DOC_ROOT
    if not audit_docs.is_dir():
        errors.append(f"{ARTIFACT_DOC_ROOT}: missing; decision artifacts cannot be inventoried.")
    else:
        for path in sorted(audit_docs.iterdir()):
            for match in FILENAME_ID_RE.finditer(path.name):
                record(f"D-{match.group(1)}", f"{ARTIFACT_DOC_ROOT}/{path.name}")

    for rel_root, patterns, recursive in ARTIFACT_CODE_ROOTS:
        code_root = repo / rel_root
        if not code_root.is_dir():
            errors.append(f"{rel_root}: missing; decision artifacts cannot be inventoried.")
            continue
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(
                code_root.rglob(pattern) if recursive else code_root.glob(pattern)
            )
        for path in sorted(set(paths)):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(repo).as_posix()
            for match in FILENAME_ID_RE.finditer(path.name):
                record(f"D-{match.group(1)}", rel)
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                errors.append(
                    f"{rel}: unreadable ({exc.__class__.__name__}); decision artifacts "
                    "cannot be inventoried."
                )
                continue
            for match in DECISION_ID_RE.finditer(text):
                record(f"D-{match.group(1)}", rel)

    runs = repo / ARTIFACT_RUN_ROOT
    if not runs.is_dir():
        errors.append(f"{ARTIFACT_RUN_ROOT}: missing; decision artifacts cannot be inventoried.")
    else:
        for path in sorted(runs.iterdir()):
            for match in RUN_DIR_ID_RE.finditer(path.name):
                record(f"D-{match.group(1)}", f"{ARTIFACT_RUN_ROOT}/{path.name}")

    return {key: sorted(value) for key, value in landed.items()}


def check_decision_inventory(repo: Path, errors: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "frozen_rows": 0,
        "decision_log_ids": 0,
        "landed_with_artifacts": 0,
        "forward_references": [],
    }

    frozen_text = read_text(repo, FROZEN_DECISIONS, errors)
    log_text = read_text(repo, DECISION_LOG, errors)
    landed = landed_decision_ids(repo, errors)
    if frozen_text is None or log_text is None:
        return summary

    frozen_ids: set[str] = set()
    for number, line in enumerate(frozen_text.splitlines(), start=1):
        match = FROZEN_ROW_RE.match(line)
        if match:
            identifier = match.group(1)
            if identifier in frozen_ids:
                errors.append(f"{FROZEN_DECISIONS}:{number}: duplicate row for {identifier}.")
            frozen_ids.add(identifier)

    log_ids: set[str] = set()
    log_declared: set[str] = set()
    for line in log_text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        for match in DECISION_ID_RE.finditer(line):
            log_ids.add(f"D-{match.group(1)}")
        for match in RECORD_DECL_RE.finditer(line):
            log_declared.update(match.group(1).split("/"))

    landed_ids = set(landed)
    inventory = sorted(
        identifier
        for identifier in frozen_ids | landed_ids
        if decision_number(identifier) >= DECISION_ID_ERA_FLOOR
    )
    for identifier in inventory:
        if identifier not in frozen_ids:
            evidence = ", ".join(landed.get(identifier, [])[:3])
            errors.append(
                f"{FROZEN_DECISIONS}: no row for {identifier}, which has landed "
                f"artifacts ({evidence}). A decision without a row did not happen "
                "as far as any later auditor is concerned."
            )
        strict = decision_number(identifier) >= STRICT_LOG_KEYING_FLOOR
        logged = identifier in log_declared if strict else identifier in log_ids
        if not logged:
            evidence = ", ".join(landed.get(identifier, [])[:3]) or "a frozen row"
            how = (
                f"no row declaring `Record {identifier}:` (that exact form is required "
                f"from D-{STRICT_LOG_KEYING_FLOOR:03d} upward; a passing mention of "
                f"{identifier} in another row does not count)"
                if strict
                else f"no mention of {identifier}"
            )
            errors.append(f"{DECISION_LOG}: {how}, but {identifier} has {evidence}.")

    summary["frozen_rows"] = len(frozen_ids)
    summary["decision_log_ids"] = len(log_ids)
    summary["landed_with_artifacts"] = len(landed_ids)
    summary["forward_references"] = sorted(
        identifier
        for identifier in log_ids
        if decision_number(identifier) >= DECISION_ID_ERA_FLOOR
        and identifier not in frozen_ids
        and identifier not in landed_ids
    )
    return summary


# --------------------------------------------------------------------------


def collect_regions(repo: Path, errors: list[str]) -> list[Region]:
    regions: list[Region] = []
    for rel in OVERLAY_SURFACES:
        text = read_text(repo, rel, errors)
        if text is None:
            continue
        regions.extend(overlay_regions(blank_fenced_blocks(text), rel, errors))

    text = read_text(repo, SHARED_CONTEXT, errors)
    if text is not None:
        regions.extend(shared_context_region(blank_fenced_blocks(text), errors))

    for rel in LEDGER_SURFACES:
        text = read_text(repo, rel, errors)
        if text is None:
            continue
        regions.extend(top_table_row(blank_fenced_blocks(text), rel, errors))
    return regions


def main() -> None:
    repo = root()
    errors: list[str] = []

    gates = load_gates(repo, errors)
    claims = load_claims(repo, errors)
    document = load_facts_document(repo, errors)
    exempt_claims = load_coverage_policy(document, claims, errors)
    exemptions = load_assertion_exemptions(document, gates, claims, errors)

    assertions_checked = 0
    covered: set[str] = set()
    regions = collect_regions(repo, errors)
    if gates or claims:
        id_regex = build_id_regex(list(gates) + list(claims))
        for region in regions:
            checked, region_covered = check_status_assertions(
                region, gates, claims, id_regex, exemptions, errors
            )
            assertions_checked += checked
            covered |= region_covered
        # An exemption whose line no longer sits in a live region is dead and is
        # reported, so exceptions cannot quietly accumulate past their occasion.
        used_exemptions = {
            index
            for index, entry in enumerate(exemptions)
            for region in regions
            if region.rel == entry["file"]
            and any(entry["line_contains"] in text for _number, text in region.lines)
        }
        check_exemptions_used(exemptions, used_exemptions, errors)

        # ---- coverage floor -------------------------------------------------
        if assertions_checked == 0:
            errors.append(
                "no status assertion was found in any live region. The prose surfaces "
                "cannot be checked against the registries at all, which is the "
                "condition F-D065-05 describes."
            )
        for gate_id in sorted(gates):
            if gate_id not in covered:
                errors.append(
                    f"{GATE_REGISTRY}: {gate_id}={gates[gate_id]} is asserted nowhere in "
                    "any live region of the handoff surfaces, so no prose can be checked "
                    "against it. Every gate must be stated where it is current."
                )
        for claim_id in sorted(claims):
            if claim_id in covered or claim_id in exempt_claims:
                continue
            errors.append(
                f"{CLAIM_REGISTRY}: {claim_id}={claims[claim_id]} is asserted nowhere in "
                "any live region and is not listed in coverage_policy.claims.exempt. "
                "State it in a live region or declare, with a reason, why it does not "
                "need to be stated."
            )

    facts_checked = check_facts(repo, document, errors)
    decisions = check_decision_inventory(repo, errors)

    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "ok": True,
                "gates": len(gates),
                "claims": len(claims),
                "live_regions": len(regions),
                "status_assertions_checked": assertions_checked,
                "gates_covered": len([g for g in gates if g in covered]),
                "claims_covered": len([c for c in claims if c in covered]),
                "claims_coverage_exempt": len(exempt_claims),
                "fact_assertions_checked": facts_checked,
                "decisions": decisions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
