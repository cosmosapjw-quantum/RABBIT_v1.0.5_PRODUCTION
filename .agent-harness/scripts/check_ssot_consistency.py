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

Measured, not asserted
----------------------
Every fact carries a ``measurement``: the mechanical derivation of its number
from the repository. That field used to be prose. It named the exact command
that produced the value, it was quoted in reviews as if it had been run, and it
was never executed by anything -- it appeared once in this file, as a schema
key. The declared hook-fixture count therefore sat two behind the real one
while this checker printed ``{"ok": true}``, because "all the surfaces agree
with the declaration" was being checked and "the declaration agrees with the
repository" was not. That is the F-D065-05 shape -- a stated fact diverging
from reality with no machine check -- recurring inside the tool built to close
F-D065-05. So the measurement is now RUN, and the fact's ``value`` is compared
against what it returns.

EXECUTION MODEL: declarative, never a shell. A measurement is a small typed
object with a closed vocabulary of kinds --

* ``{"kind": "count_lines_matching", "path": ..., "pattern": ...}``
  lines of a repository file matching a Python regular expression;
* ``{"kind": "json_array_length", "path": ..., "json_path": [...]}``
  the length of the JSON array at a path of object keys.

-- and this checker executes those two operations itself. It never passes any
string from ``SSOT_FACTS.json`` to a shell, a subprocess, ``eval`` or an
import. The alternative was to keep the shell string and run it under
``subprocess`` with ``shell=True``. It was rejected on four counts, and the
usual defence of it is the weakest of the four:

1. It would make this checker a NEW code-execution path for repository data.
   The same-OS-user residual (recorded against the admission mechanism, and
   restated in the frozen decision rows) does mean that whoever can edit
   ``SSOT_FACTS.json`` can already run code some other way -- but "it was
   already possible" argues for not widening it, not for widening it. This
   script is run by hooks, by CI, and by reviewers pointing it at a branch they
   are auditing; in every one of those contexts it is otherwise a pure reader,
   and a pure reader is a thing you can safely run against untrusted bytes.
   That property is worth more than the convenience of arbitrary commands.
2. A shell string is not a measurement, it is a program, and its result depends
   on things nothing here pins: ``PATH``, the shell dialect, the ``grep``
   implementation and whether the pattern is BRE or ERE, the locale, and the
   working directory. The declarative form resolves paths from the repository
   root and compiles the pattern with Python's own engine, so it returns the
   same number wherever it runs.
3. ``grep -c`` exits 1 and prints ``0`` when nothing matches. A mistyped pattern
   would therefore report the number zero rather than an error -- silence
   presenting as data, which is the failure mode this whole file exists to
   refuse.
4. The vocabulary is closed, so every way of getting it wrong is enumerable and
   is an error: an unknown ``kind``, a missing or unknown key, a path that is
   absolute or escapes the repository, an uncompilable pattern, an unreadable
   file, malformed JSON, a ``json_path`` that does not exist, a target that is
   not an array. FAIL CLOSED: a measurement that cannot be run is an error and
   never a skip, because a skipped measurement leaves the fact asserted, which
   is precisely the state this section removes.

The cost is real and is accepted: a new kind of fact needs a new kind here.
That is the point. Adding a kind is a reviewed change to this file; adding a
shell string is an unreviewed change to a data file.

NO EXEMPTION. ``measurement`` is required on every fact -- there is no "this one
cannot be measured" key, in the JSON or in the code. A number that cannot be
derived mechanically from the repository is not a fact this file can keep
honest; it is a claim, and claims belong in CLAIM_REGISTRY.jsonl with a status.
An opt-out key would be reached for the first time it was inconvenient and
would recreate exactly the unchecked-declaration state that this section
closes. If a future number genuinely needs a different derivation, the change
is a new ``kind`` with its own validation and its own fixture, which is a
strictly smaller and more reviewable thing than a blanket escape hatch.

WHAT MEASUREMENT MEANS FOR A FROZEN FACT: nothing, deliberately. The
measurement runs exactly once per fact, against the working tree, and is
compared against ONE number: the fact's head-of-chain ``value``. It is never
compared against a ``prior`` entry and never against a ``frozen`` assertion.
A frozen row records what was true at a commit that is not this working tree;
"measuring" it here would either be meaningless or -- far worse -- would
"correct" ``35 @ ed7bc49`` to today's count, which is the exact refresh the
frozen role exists to forbid. Prior values and frozen rows keep the
commit-pinned checks above and are not re-derived. Re-deriving them would mean
reading historical blobs out of git, which trades a real property (this script
reads the working tree and nothing else) for a check that the pin rule already
provides.

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
import subprocess
import unicodedata
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from _harness import load_json, loads_strict, root

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
# The REFUSAL scan is case-insensitive; the two legal FORMS are not. Round 12
# attacked its own round-11 fix and found `` `G-X` is pass. `` and `` `G-X=pass` ``
# both invisible: STATUS_RE is upper-case only, so neither the bare-token scan
# nor the structural form saw them, while a human reads both as a status. The
# asymmetry is deliberate and is closed in the safe direction -- a DECLARATION
# must still be written `G-X=PASS`, because a declaration is machine-read and
# should look like one, but anything that a reader would take for a status is
# refused whatever its case. Measured cost on the live corpus: zero segments.
# Characters that render as nothing, or reorder what follows them. Round 12:
# `G-HARNESS<U+200B>-INTEGRITY is PASS` is not a registry id, so it was invisible
# to every rule here while displaying as the real gate id. There is no way to
# read past that by improving the id match -- the fix is to refuse the character
# class outright in a live region, which no honest handoff line needs. Measured
# cost on the live corpus: zero lines.
GATE_STATUSES = {"PASS", "FAIL"}
CLAIM_STATUSES = set(STATUS_TOKENS) - {"PASS", "FAIL"}

# A table cell. Nothing crosses it, ever -- this is what stops a ledger Result
# cell binding to an id in its Notes cell.
# NO sentence tier. Round 11 first scoped the refusal below to the sentence and
# the round-10 abbreviation fixture killed it within the hour: `. ` fires inside
# `cf.`, so ``G-BETA` (r10, cf. the appendix) is PASS.` split into a half with
# the id and a half with the token, and neither half broke the rule. Every
# repair for that -- an abbreviation list, a capital-letter test -- is a way of
# reading English, and reading English is what lost this argument four times.
#
# The scope is therefore the HARD segment, with no subdivision. Nothing is
# gained by splitting it: a structural pair is masked together with its own id,
# so a segment written correctly leaves nothing behind to trip the rule. That
# was measured at zero refusals across the live corpus.
# Markup a Status cell may carry around its token and remain a bare status.

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
            row = loads_strict(line)
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

    # Round 9's F-CONTROLLING-DATE-GAP. Liveness is decided by comparing each
    # section's date against `newest`, which is computed over EVERY dated
    # heading -- but the tie/undated checks above only ever looked at headings
    # that say "controlling". Nothing required the controlling section to
    # actually BE the newest. Adding one heading dated later, claiming neither
    # "controlling" nor "superseded", therefore pushed the real controlling
    # section below `newest` and dropped it out of every live region silently:
    # the surface still named its controlling overlay, and that overlay was no
    # longer being read. A section that outranks the controlling one is either a
    # mistake or an attack, and either way it is not a thing to resolve quietly.
    if dated_controlling and newest is not None:
        newest_controlling = max(stamp for _index, stamp in dated_controlling)
        if newest_controlling < newest:
            usurpers = "; ".join(
                sorted(
                    blocks[index][2].strip()
                    for index, stamp in dates.items()
                    if stamp == newest and index not in controlling
                )
            )
            errors.append(
                f"{rel}: the controlling overlay is dated "
                f"{newest_controlling.isoformat()} but the file's newest dated section "
                f"is {newest.isoformat()} ({usurpers}). A later section that claims "
                "neither 'controlling' nor 'superseded' silently demotes the "
                "controlling overlay out of every live region. Mark it superseded, or "
                "make it the controlling overlay."
            )

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



def is_git_repo(repo: Path) -> bool:
    """True when this tree is a git repository at all.

    Tracking is a property of git repositories, so the retention check below is
    gated on this rather than firing on every synthetic fixture corpus. It is
    not a bypass: `validate_harness.py` refuses to run at all when git is
    unavailable, so the real repository can never take this branch.
    """
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--git-dir"], cwd=repo,
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return False
    return done.returncode == 0

def is_tracked(repo: Path, rel: str) -> bool:
    """True when git tracks this path. Fail-closed: unknown counts as untracked.

    The generated board is the only authority for current status, so a board a
    fresh checkout would not receive is not an authority at all. `.gitignore`
    already excludes whole trees in this repository and round 13 proved a
    directory can vanish while the checker stays green, so presence on this disk
    is not evidence of retention.
    """
    try:
        done = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", rel],
            cwd=repo, capture_output=True, text=True, check=False,
        )
    except OSError:
        return False
    return done.returncode == 0

BOARD_ARTIFACT = ".agent-harness/generated/STATUS_BOARD.md"
# Files that show the board inline. A tuple so a second is possible, shipping
# with none: every host multiplies the byte-equality surface and reopens the
# question of which host is live. D-073 commit 2 adds PROJECT_STATE.md.
BOARD_HOSTS: tuple[str, ...] = ("docs/harness/PROJECT_STATE.md",)
BOARD_BEGIN = "<!-- BEGIN GENERATED STATUS BOARD -->"
BOARD_END = "<!-- END GENERATED STATUS BOARD -->"


def render_board(gates: dict[str, str], claims: dict[str, str]) -> str:
    """The board, as a pure function of the two registries.

    Sorted by id so the bytes do not depend on registry ordering, and carrying
    no date, no timestamp and no ``## `` heading -- see the module docstring of
    ``build_status_board.py`` for why both of those are load-bearing rather than
    stylistic.

    Status is rendered UPPERCASE regardless of how the registry spells it. The
    registry stores ``"fail"`` and the board shows ``FAIL``; normalising here
    means a registry casing change cannot silently alter the board's bytes.
    """
    lines = [BOARD_BEGIN, "", "| Gate | Status |", "|---|---|"]
    lines += [f"| `{gate}` | {status.upper()} |" for gate, status in sorted(gates.items())]
    lines += ["", "| Claim | Status |", "|---|---|"]
    lines += [f"| `{claim}` | {status.upper()} |" for claim, status in sorted(claims.items())]
    lines += ["", BOARD_END]
    return "\n".join(lines) + "\n"


def extract_board(text: str, rel: str, errors: list[str]) -> str | None:
    """The sentinel span of one file, or None with an error explaining why not.

    Exactly one BEGIN and one END, in that order. Two blocks is how a second and
    false board would be smuggled into a file that already carries a true one,
    so the count is checked rather than the first match taken.
    """
    begins, ends = text.count(BOARD_BEGIN), text.count(BOARD_END)
    if begins == 0 and ends == 0:
        errors.append(
            f"{rel}: no generated status board. This file is declared a board host, so it must "
            f"carry the block between {BOARD_BEGIN} and {BOARD_END}."
        )
        return None
    if begins != 1 or ends != 1:
        errors.append(
            f"{rel}: found {begins} board BEGIN and {ends} board END markers, expected one of each. "
            "A second block is how a false board is smuggled in beside a true one."
        )
        return None
    start = text.index(BOARD_BEGIN)
    stop = text.index(BOARD_END)
    if stop < start:
        errors.append(f"{rel}: the board END marker precedes its BEGIN marker.")
        return None
    return text[start : stop + len(BOARD_END)] + "\n"


def check_no_status_beside_an_id(
    repo: Path,
    regions: list[Region],
    id_regex: re.Pattern[str],
    errors: list[str],
) -> None:
    """A live line outside the board may not name a registry id AND a status.

    This is the one piece of the old machinery that survives, and it survives
    because it is not the old machinery: **it decides nothing.** Every historical
    defeat -- comma adjacency, decoy shadowing, negation cues, letter case,
    invisible characters, homoglyphs -- was a defeat of BINDING, of working out
    which status attaches to which id. This never binds. It refuses the
    co-occurrence and says so, which is a question with one answer.

    Measured cost on the migrated corpus: zero lines. Ordinary prose does not
    need to put a gate id and a status word on one line now that the board
    states every status, and the board's own lines are excluded by construction
    because the sentinels bound them.

    What this does NOT close, stated because the re-audit reproduced it: a false
    SEMANTIC claim carrying no registry id -- "all obligations have been
    discharged" -- is not detected, here or anywhere. No parser closes that. The
    board is the authority; prose is explicitly not.
    """
    board_spans: dict[str, tuple[int, int]] = {}
    for rel in BOARD_HOSTS + (BOARD_ARTIFACT,):
        text = read_text(repo, rel, [])
        if text is None or BOARD_BEGIN not in text or BOARD_END not in text:
            continue
        first = text[: text.index(BOARD_BEGIN)].count("\n") + 1
        last = text[: text.index(BOARD_END)].count("\n") + 1
        board_spans[rel] = (first, last)

    for region in regions:
        span = board_spans.get(region.rel)
        for number, line in region.lines:
            if span and span[0] <= number <= span[1]:
                continue
            if not STATUS_RE.search(line):
                continue
            for match in id_regex.finditer(line):
                errors.append(
                    f"{region.rel}:{number}: {match.group(1)} appears on a live line that also "
                    "carries a status token. Status is stated by the generated board and nowhere "
                    "else; this line is refused rather than read, because every version of the "
                    "reader that tried has been defeated."
                )


def check_generated_board(
    repo: Path,
    gates: dict[str, str],
    claims: dict[str, str],
    regions: list[Region],
    errors: list[str],
) -> int:
    """Require the rendered board to appear verbatim in the artifact and every host.

    Returns the row count, which is the mechanical replacement for the old
    coverage floor: if the renderer ever drops a row, the count stops matching
    the registries and this fails, without anyone having to notice a missing
    line in a table.
    """
    expected = render_board(gates, claims)
    rows = len(gates) + len(claims)

    artifact = read_text(repo, BOARD_ARTIFACT, errors)
    if artifact is None:
        return 0
    if is_git_repo(repo) and not is_tracked(repo, BOARD_ARTIFACT):
        errors.append(
            f"{BOARD_ARTIFACT}: the generated board is not tracked by git, so a fresh checkout "
            "would not receive the only authority for current status."
        )
    found = extract_board(artifact, BOARD_ARTIFACT, errors)
    if found is not None and found != expected:
        errors.append(
            f"{BOARD_ARTIFACT}: the generated board does not match what the registries render. "
            "Run .agent-harness/scripts/build_status_board.py. Verification is byte equality "
            "against a fresh render, not a stored hash, so there is nothing to refresh and "
            "nothing that can go stale."
        )

    live = live_line_index(regions)
    for rel in BOARD_HOSTS:
        text = read_text(repo, rel, errors)
        if text is None:
            continue
        shown = extract_board(text, rel, errors)
        if shown is None:
            continue
        if shown != expected:
            errors.append(
                f"{rel}: the board shown here is not what the registries render. It was edited by "
                "hand, or the registries moved and the host was not rebuilt. Either way the file "
                "states a status the authority does not."
            )
            continue
        # The block must sit where somebody reads it. Without this it could
        # verify byte-perfect while parked in a superseded section, leaving the
        # controlling overlay with no board at all and the checker green -- which
        # is precisely the F-D065-05 condition this module exists to prevent.
        begin_line = text[: text.index(BOARD_BEGIN)].count("\n") + 1
        if (rel, begin_line) not in live:
            errors.append(
                f"{rel}:{begin_line}: the generated status board sits outside a live region. A "
                "board nobody reads is not a board; move it into the controlling section."
            )
    return rows

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
FACT_REQUIRED = {"fact_id", "measurement", "value", "as_of_commit", "assertions"}
# Closed vocabulary of measurement kinds -> the keys each one takes. Every key
# is required; nothing is optional, so a half-written measurement is an error
# rather than a measurement with a default silently filled in.
MEASUREMENT_KEYS: dict[str, set[str]] = {
    "count_lines_matching": {"kind", "path", "pattern"},
    "json_array_length": {"kind", "path", "json_path"},
}
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
# `coverage_policy` and `assertion_exemptions` are GONE, and removing them from this
# whitelist matters as much as removing the code that read them: a key left here
# would be accepted and unread, which is worse than an error because it reads as
# supported. Coverage stopped being a policy when the board began rendering every
# registry entry, and an assertion exemption has nothing left to exempt.
DOCUMENT_KEYS = {"schema_version", "purpose", "facts"}


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


def describe_measurement(spec: Any) -> str:
    """Render a measurement as the operation it is, for an error message.

    Rendered from the declaration itself rather than from a second, prose copy
    of it: a human-readable duplicate stored alongside the machine-readable one
    is one more stated fact that can drift from reality unchecked, which is the
    thing this file is for.
    """
    if not isinstance(spec, dict):
        return f"(not a measurement: {spec!r})"
    kind = spec.get("kind")
    if kind == "count_lines_matching":
        return f"count_lines_matching(path={spec.get('path')!r}, pattern={spec.get('pattern')!r})"
    if kind == "json_array_length":
        keys = spec.get("json_path")
        rendered = ".".join(keys) if isinstance(keys, list) and all(
            isinstance(key, str) for key in keys
        ) else repr(keys)
        return f"json_array_length(path={spec.get('path')!r}, json_path={rendered!r})"
    return f"(unrunnable measurement of kind {kind!r})"


def _measured_path(repo: Path, spec: dict[str, Any], where: str, errors: list[str]) -> Path | None:
    """Resolve a measurement's target inside the repository, or error.

    A measurement reads repository files and nothing else, so an absolute path
    or one containing ``..`` is refused rather than followed.
    """
    rel = spec.get("path")
    if not isinstance(rel, str) or not rel.strip():
        errors.append(f"{FACTS_FILE}: {where} has no usable 'path'.")
        return None
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts:
        errors.append(
            f"{FACTS_FILE}: {where} names {rel!r}, which is absolute or escapes the "
            "repository. A measurement reads repository files and nothing else."
        )
        return None
    target = repo / rel
    # The spelling check above is about the STRING; containment is about the
    # FILE. Round 9's F-SYMLINK-ESCAPE separated the two: a syntactically clean
    # relative path can name an in-tree symlink whose target is outside the
    # repository, and `read_text` follows it, so the guarantee in this
    # function's own docstring was true of the path and false of the read.
    # Resolve both sides and compare the real locations.
    # Deliberately NON-strict. Resolving symlinks is all this needs; whether the
    # file exists and can be read is diagnosed by `_read_measured_file`, whose
    # message names the real problem. A strict resolve here would swallow
    # "unreadable" and report "does not resolve" for a merely missing file --
    # still fail-closed, but a worse answer to the question the operator asked.
    try:
        real = target.resolve()
        root = repo.resolve()
    except (OSError, RuntimeError) as exc:
        errors.append(
            f"{FACTS_FILE}: {where} cannot be run: {rel} does not resolve "
            f"({exc.__class__.__name__}). A measurement that cannot be run is an "
            "error, never a skip."
        )
        return None
    if real != root and root not in real.parents:
        errors.append(
            f"{FACTS_FILE}: {where} names {rel!r}, which resolves to {real} — outside "
            "the repository. A measurement reads repository files and nothing else, "
            "and a symlink does not change that."
        )
        return None
    return target


def _read_measured_file(path: Path, rel: str, where: str, errors: list[str]) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(
            f"{FACTS_FILE}: {where} cannot be run: {rel} is unreadable "
            f"({exc.__class__.__name__}). A measurement that cannot be run is an error, "
            "never a skip -- a skipped measurement leaves the fact merely asserted."
        )
        return None


def _count_lines_matching(
    path: Path, spec: dict[str, Any], where: str, errors: list[str]
) -> int | None:
    pattern = spec.get("pattern")
    if not isinstance(pattern, str):
        errors.append(f"{FACTS_FILE}: {where} has a 'pattern' that is not a string.")
        return None
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        errors.append(
            f"{FACTS_FILE}: {where} has an invalid 'pattern' ({exc}), so it cannot be run."
        )
        return None
    text = _read_measured_file(path, str(spec["path"]), where, errors)
    if text is None:
        return None
    count = sum(1 for line in text.splitlines() if regex.search(line))
    if count == 0:
        # Round 9's F-REGEX-SILENT-ZERO. Reason 3 in this module's docstring
        # rejects `grep -c` because it "would report the number zero rather than
        # an error -- silence presenting as data, which is the failure mode this
        # whole file exists to refuse." Moving the match into Python did not by
        # itself fix that: a pattern that compiles but matches nothing was
        # returning the integer 0, indistinguishable from a legitimate zero.
        # A counting measurement that finds nothing is a broken measurement.
        # There is no legitimate zero here to protect -- nobody declares a fact
        # to assert that a file contains none of something -- so this is an
        # error, not a warning. (`json_array_length` is deliberately different:
        # resolving the json_path already proves the array exists, so an empty
        # array there is measured data, not silence.)
        errors.append(
            f"{FACTS_FILE}: {where} matched no lines. {describe_measurement(spec)} "
            "compiled but found nothing, which is a broken measurement and not the "
            "value 0. Fix the pattern, or the file it reads."
        )
        return None
    return count


def _json_array_length(
    path: Path, spec: dict[str, Any], where: str, errors: list[str]
) -> int | None:
    keys = spec.get("json_path")
    if (
        not isinstance(keys, list)
        or not keys
        or not all(isinstance(key, str) and key.strip() for key in keys)
    ):
        errors.append(
            f"{FACTS_FILE}: {where} needs a non-empty 'json_path' of object key names."
        )
        return None
    rel = str(spec["path"])
    text = _read_measured_file(path, rel, where, errors)
    if text is None:
        return None
    try:
        value: Any = loads_strict(text)
    except json.JSONDecodeError as exc:
        errors.append(
            f"{FACTS_FILE}: {where} cannot be run: {rel} is not valid JSON "
            f"({exc.__class__.__name__})."
        )
        return None
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(value, dict) or key not in value:
            errors.append(
                f"{FACTS_FILE}: {where} cannot be run: {rel} has no {'.'.join(walked)}."
            )
            return None
        value = value[key]
    if not isinstance(value, list):
        errors.append(
            f"{FACTS_FILE}: {where} cannot be run: {rel} {'.'.join(keys)} is "
            f"{type(value).__name__}, not an array."
        )
        return None
    return len(value)


def check_description_names_target(
    fact_id: str, description: Any, spec: Any, errors: list[str]
) -> None:
    """Require a fact's prose to name the file its measurement actually reads.

    Round 9's F-DESC-MEASURE-DECOUPLE: ``description`` was inert prose. Nothing
    tied it to ``measurement``, so a fact whose description talked about
    ``test_hooks.py`` could measure ``LICENSE`` and pass, as long as ``value``
    matched what was really read. The description is what a human reads to
    decide whether a number means what they think it means, so an unchecked
    description is not documentation -- it is a second, unverified claim sitting
    next to a verified one, which is the exact shape this file exists to refuse.

    The check is deliberately weak and mechanical: the measured path must appear
    somewhere in the description. It cannot prove the prose is *right*, only
    that it is about the same file. That is enough to stop the substitution,
    and it stays checkable without parsing English.
    """
    if not isinstance(spec, dict):
        return
    rel = spec.get("path")
    if not isinstance(rel, str) or not rel.strip():
        return
    if not isinstance(description, str) or not description.strip():
        errors.append(
            f"{FACTS_FILE}: fact {fact_id} has no 'description', so nothing states "
            f"in prose what {rel} is being counted for. Add one that names the file."
        )
        return
    if rel not in description:
        errors.append(
            f"{FACTS_FILE}: fact {fact_id} measures {rel!r} but its description never "
            "names that file, so the prose and the measurement can describe different "
            "things. Name the measured path in the description."
        )


def measure_fact(
    repo: Path, fact_id: str, spec: Any, errors: list[str]
) -> tuple[int | None, str]:
    """Run one fact's declared measurement against the working tree.

    Returns ``(value, description)``; ``value`` is None exactly when the
    measurement could not be run, and in that case an error has been recorded.
    Nothing here is handed to a shell, a subprocess, ``eval`` or an import: the
    two supported kinds are performed by this function. See the module
    docstring for why the executable-string form was rejected.
    """
    where = f"fact {fact_id} measurement"
    description = describe_measurement(spec)
    if isinstance(spec, str):
        errors.append(
            f"{FACTS_FILE}: {where} is a command string ({spec!r}). This checker never "
            f"hands anything in {FACTS_FILE} to a shell; declare the measurement "
            f"structurally instead, using one of: {', '.join(sorted(MEASUREMENT_KEYS))}."
        )
        return None, description
    if not isinstance(spec, dict):
        errors.append(
            f"{FACTS_FILE}: {where} is not an object. Every fact must declare how it is "
            "measured; there is no exemption."
        )
        return None, description
    kind = spec.get("kind")
    if not isinstance(kind, str) or kind not in MEASUREMENT_KEYS:
        errors.append(
            f"{FACTS_FILE}: {where} has kind {kind!r}, which this checker cannot run. "
            f"Known kinds: {', '.join(sorted(MEASUREMENT_KEYS))}. An unknown kind is an "
            "error, never a skip."
        )
        return None, description
    allowed = MEASUREMENT_KEYS[kind]
    if not _require_keys(where, spec, allowed, allowed, errors):
        return None, description
    path = _measured_path(repo, spec, where, errors)
    if path is None:
        return None, description
    if kind == "count_lines_matching":
        return _count_lines_matching(path, spec, where, errors), description
    return _json_array_length(path, spec, where, errors), description


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

def live_line_index(regions: list[Region]) -> set[tuple[str, int]]:
    """Every ``(file, line)`` this run treats as authoritative right now."""
    return {(region.rel, number) for region in regions for number, _line in region.lines}


def check_facts(
    repo: Path,
    document: dict[str, Any],
    errors: list[str],
    live_lines: set[tuple[str, int]] | None = None,
) -> tuple[int, dict[str, int]]:
    facts = document.get("facts")
    if not isinstance(facts, list) or not facts:
        errors.append(f"{FACTS_FILE}: no 'facts' array.")
        return 0, {}

    checked = 0
    measurements: dict[str, int] = {}
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

        # The declaration is compared against the repository, not merely against
        # the other places it is written down. Only the head-of-chain `value` is
        # measured: `prior` entries and `frozen` assertions are pinned to commits
        # that are not this working tree, and re-deriving them here would refresh
        # exactly what the frozen role forbids refreshing.
        check_description_names_target(
            fact_id, fact.get("description"), fact.get("measurement"), errors
        )
        measured, measurement = measure_fact(repo, fact_id, fact.get("measurement"), errors)
        if measured is not None:
            measurements[fact_id] = measured
            if measured != current:
                errors.append(
                    f"{FACTS_FILE}: {fact_id} declares value {current} (as of "
                    f"{current_commit}) but measures {measured} in the working tree. "
                    f"Measurement: {measurement}. The declaration has drifted from the "
                    "repository -- set 'value' to the measured number, move the old "
                    "value into 'prior' with the commit it held at, and update every "
                    "role 'current' assertion site; or repair the repository. Do not "
                    "touch the 'frozen' or 'historical' rows."
                )

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
                    if (
                        role != "current"
                        and live_lines is not None
                        and (rel, number) in live_lines
                    ):
                        # Round 9's F-FROZEN-ROLE-SMUGGLE. `role` was a
                        # free-standing label: nothing tied a row calling itself
                        # `frozen` to actually sitting somewhere this checker
                        # treats as no longer authoritative, so a wildly wrong
                        # number could sit in present-tense prose in a document's
                        # lead section and be excused by its own label. A frozen
                        # row's whole justification is that it is pinned to a past
                        # commit and never refreshed, which is a claim about WHERE
                        # the line is; the label now has to be true of that.
                        #
                        # `historical` is covered too, and the narrowing that
                        # exempted it is WITHDRAWN. It was justified in the record
                        # by the claim that the `dated` test below already
                        # constrains such rows, "earning its licence by content
                        # rather than by position". Round 10 disproved that: the
                        # `dated` test is a bare substring search for the commit
                        # ANYWHERE on the line, so "There are 35 hook tests
                        # actually (ed7bc49)" -- present-tense prose in an
                        # always-live region, using only a legitimately declared
                        # prior -- passed the unmutated checker at exit 0. A
                        # parenthesised hash does not make a sentence read as
                        # history to anyone.
                        #
                        # There is no vocabulary that separates a past-tense
                        # citation from a present-tense claim reliably, and this
                        # file has now lost that argument three times. So the rule
                        # is positional again, which is checkable: a superseded
                        # number is cited where superseded things live.
                        errors.append(
                            f"{rel}:{number}: {fact_id} is declared with role {role!r} "
                            f"but this line is inside a LIVE region ({quoted!r}). A "
                            "frozen or historical assertion must sit where it is no "
                            "longer authoritative; the role is a claim about the line's "
                            "position, not a licence to ignore it."
                        )
                        continue
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
    return checked, measurements


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

    board_rows = 0
    regions = collect_regions(repo, errors)
    if gates or claims:
        id_regex = build_id_regex(list(gates) + list(claims))
        board_rows = check_generated_board(repo, gates, claims, regions, errors)
        check_no_status_beside_an_id(repo, regions, id_regex, errors)

    facts_checked, fact_measurements = check_facts(
        repo, document, errors, live_lines=live_line_index(regions)
    )
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
                "board_rows": board_rows,
                "fact_assertions_checked": facts_checked,
                "fact_measurements": fact_measurements,
                "decisions": decisions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
