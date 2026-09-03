#!/usr/bin/env python3
"""Replace raw admission tokens in retained evidence with their digest.

An admission token is a single-use secret that travels in the spawn prompt and
comes back in the agent's final message. Any captured SubagentStop event JSON
therefore contains it in plaintext, and a capture of a *blocked* stop is the
dangerous one, because that receipt is still open. Run directories are retained
and force-added as evidence, so the token would be committed.

FIELD-SCRUB (original behaviour, unchanged): every ``admission_proof`` value
found -- structurally, walking JSON, or textually via regex for non-JSON and
escaped evidence -- is rewritten to ``sha256:<hex>``, exactly what SubagentStop
compares against. The evidence stays verifiable while the reusable secret is
gone. Idempotent: an already-scrubbed value is left alone.

    python3 .agent-harness/scripts/scrub_admission_proof.py <path> [<path> ...]
    python3 .agent-harness/scripts/scrub_admission_proof.py --run <run-id>

ANYWHERE-SWEEP (additive, closing the 2026-07-29 R6-regression-live-token
incident's stated residual): the field-scrub above only ever looked at the
``admission_proof`` key. It would not have caught -- and did not catch -- a
raw token pasted into an arbitrary prose field, because a review agent, while
writing up the fact that its own stop had been BLOCKED, did exactly that,
twice, into a persistent field of its result envelope, while its receipt was
still open (see .agent-harness/incidents/2026-07-29-r6-regression-live-token.json).

The gap is closable exactly, not heuristically: a digest cannot be reversed,
but it CAN be recomputed and compared. Every receipt under
``.agent-harness/admissions/<run>/`` carries ``token_digest``. So: extract
every token-shaped substring from the target text (the alphabet
``secrets.token_urlsafe`` uses is ``[A-Za-z0-9_-]``; the exact length is
derived at runtime from ``admit_agent.TOKEN_BYTES`` -- the real minting
configuration -- rather than hardcoded, and a small window is scanned around
it so a token embedded inside a longer run of the same alphabet, such as a
Python ``b'...'`` bytes literal, still yields the real token as a candidate at
some offset), hash each candidate, and compare against every known
``token_digest``. A hash match is certain, not probable, and it names exactly
which receipt leaked.

    python3 .agent-harness/scripts/scrub_admission_proof.py --verify-result <path>
    python3 .agent-harness/scripts/scrub_admission_proof.py --sweep <path> [<path> ...]
    python3 .agent-harness/scripts/scrub_admission_proof.py --sweep-run <run-id>

``--verify-result`` is the pre-admission gate the incident needed: run it on a
result artifact BEFORE dispatching Stop; it exits non-zero if any match is
still bound to an OPEN receipt (live). ``--sweep``/``--sweep-run`` run the same
read-only check over one or more files, directories (walked recursively), or
a whole run, and report every match found -- live (open receipt, urgent) or
spent (consumed receipt, hygiene) -- with the file, offset, and the exact
run/assignment the leaked receipt belongs to. None of the three ever writes
anything.

LEDGER-PINNED BYTES: the field-scrub (and only the field-scrub -- the sweep
modes never write) refuses to rewrite a result artifact whose current bytes
are pinned by a consumed row in that run's ``ADMISSIONS.jsonl``
(``.agent-harness/runs/<run>/results/<assignment_id>.json`` whose sha256 equals
the ledger's last ``consumed`` row for that assignment). Rewriting those bytes
would silently invalidate the very digest validate_harness.py already trusts.
This is not hypothetical: one round-2 token
(``run-20260729-f10-d067-review-r2/results/A-D067-REVIEW-R2.json``) is exactly
such a file today and is deliberately left byte-intact. The refusal is loud,
never silent, and names the real remedy -- a declared supersession via
``admit_agent.py --reopen --reason '...'``, never a silent rewrite.

Fail-closed throughout: an unreadable or malformed file, receipt, or ledger is
always reported as an error, never silently treated as clean or skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _harness import dump_json_atomic, root, token_digest
from admit_agent import TOKEN_BYTES

ALREADY_SCRUBBED = re.compile(r"sha256:[0-9a-f]{64}\Z")

# --- anywhere-sweep: candidate extraction -----------------------------------

# secrets.token_urlsafe(nbytes) always emits the same LENGTH for a fixed
# nbytes regardless of the random content (base64 without padding), so calling
# it once derives the real length from the real minting configuration in
# admit_agent.py instead of hardcoding a guessed number here. If TOKEN_BYTES
# ever changes, this changes with it automatically.
TOKEN_CANDIDATE_LENGTH = len(secrets.token_urlsafe(TOKEN_BYTES))
# A small window around the derived length, not to guess at historical
# TOKEN_BYTES values (there has only ever been one), but so an off-by-a-little
# surrounding context cannot slide the true boundary out of the scanned set.
# The mechanism that actually finds a token embedded in a longer alphabet run
# (e.g. inside a Python `b'...'` literal) is sliding the START offset across
# every position in the run, done unconditionally below for every length here.
TOKEN_CANDIDATE_PAD = 2
TOKEN_CANDIDATE_LENGTHS = tuple(
    length
    for length in range(
        TOKEN_CANDIDATE_LENGTH - TOKEN_CANDIDATE_PAD,
        TOKEN_CANDIDATE_LENGTH + TOKEN_CANDIDATE_PAD + 1,
    )
    if length > 0
)
TOKEN_ALPHABET_RUN_RE = re.compile(r"[A-Za-z0-9_-]+")


def find_candidate_tokens(text: str) -> list[tuple[int, str]]:
    """Every token-shaped substring of ``text`` worth hashing, as (offset, str).

    Every maximal run of the ``secrets.token_urlsafe`` alphabet is windowed at
    every length in ``TOKEN_CANDIDATE_LENGTHS`` and at every start offset
    inside the run -- this is what makes a token found mid-run (a longer
    surrounding alphabet run, e.g. the ``b`` of a ``b'<token>'`` literal
    glued onto the front) still appear as a candidate at its true boundary,
    without needing to know where that boundary is in advance.

    Linear in ``len(text)``: each run of length L contributes at most
    ``L * len(TOKEN_CANDIDATE_LENGTHS)`` candidates, and both
    ``TOKEN_CANDIDATE_LENGTHS`` and its constant factor are fixed, independent
    of file size -- not quadratic.
    """
    candidates: list[tuple[int, str]] = []
    for run in TOKEN_ALPHABET_RUN_RE.finditer(text):
        run_text = run.group(0)
        run_start = run.start()
        run_len = len(run_text)
        for length in TOKEN_CANDIDATE_LENGTHS:
            if length > run_len:
                continue
            for start in range(0, run_len - length + 1):
                candidates.append((run_start + start, run_text[start : start + length]))
    return candidates


# --- anywhere-sweep: known digests ------------------------------------------


@dataclass(frozen=True)
class ReceiptDigest:
    """One admission receipt's token_digest, labelled for reporting a match."""

    run_id: str
    assignment_id: str
    state: str
    digest: str  # verbatim token_digest string as read from the receipt file
    receipt_path: str  # repo-relative, for citation in a report

    @property
    def spent(self) -> bool:
        # Fail closed: only an explicit "consumed" state is trusted as spent.
        # Anything else -- "open", or any unrecognised value -- must read as
        # live so an ambiguous receipt can never be reported as safe.
        return self.state == "consumed"


@dataclass(frozen=True)
class TokenMatch:
    offset: int
    line: int
    length: int
    receipts: tuple[ReceiptDigest, ...]

    @property
    def live(self) -> bool:
        return any(not r.spent for r in self.receipts)


def fail(message: str) -> None:
    raise SystemExit(f"scrub_admission_proof: {message}")


def read_utf8(path: Path) -> tuple[bytes, str] | None:
    """(raw_bytes, decoded_text) for ``path``, or None if unreadable as UTF-8.

    Collapses OSError and UnicodeDecodeError to one outcome so every caller
    has exactly one thing to check, and always has the exact original bytes
    (not a decode/re-encode round trip) available for hashing.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return raw, text


def repo_relative(repo: Path, path: Path) -> str | None:
    try:
        return path.resolve().relative_to(repo).as_posix()
    except ValueError:
        return None


def file_sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def collect_known_token_digests(repo: Path) -> dict[str, list[ReceiptDigest]]:
    """Every ``token_digest`` carried by a receipt under
    ``.agent-harness/admissions/``, across every run.

    This is the trust root a digest-match verdict rests on, so it fails closed
    exactly like ``admit_agent.scan_run_receipts``: a listing failure, a
    symlink, or one corrupt receipt anywhere aborts the whole scan rather than
    silently answering "no known tokens", which would make every file in the
    repo look clean by construction. No admissions directory at all is not an
    error -- it means no receipt has ever been minted.
    """
    admissions_dir = repo / ".agent-harness" / "admissions"
    known: dict[str, list[ReceiptDigest]] = {}
    if not admissions_dir.exists():
        return known
    if admissions_dir.is_symlink():
        fail(f"admissions directory path is a symlink: {admissions_dir}")
        return known
    if not admissions_dir.is_dir():
        fail(f"admissions path exists but is not a directory: {admissions_dir}")
        return known
    try:
        run_dirs = sorted(p for p in admissions_dir.iterdir())
    except OSError as exc:
        fail(f"admissions directory could not be listed ({exc.__class__.__name__})")
        return known
    for run_dir in run_dirs:
        if run_dir.is_symlink():
            fail(f"admissions run directory is a symlink: {run_dir}")
            return known
        if not run_dir.is_dir():
            continue
        try:
            entries = sorted(run_dir.iterdir())
        except OSError as exc:
            fail(
                "admission receipts directory could not be listed "
                f"({exc.__class__.__name__}): {run_dir}"
            )
            return known
        for entry in entries:
            # Mirrors admit_agent.scan_run_receipts exactly: dot-prefixed
            # atomic-write temps and the agent-lock directory, and any
            # non-".json" suffix (Stop's O_EXCL ".claim" files), are the only
            # deliberate exclusions -- never a silent skip of anything else.
            if entry.name.startswith("."):
                continue
            if entry.suffix != ".json":
                continue
            if entry.is_symlink():
                fail(
                    "admission receipt path is a symlink; refusing to follow "
                    f"it: {entry}"
                )
                return known
            if not entry.is_file():
                continue
            decoded = read_utf8(entry)
            if decoded is None:
                fail(f"admission receipt is unreadable: {entry}")
                return known
            _, raw_text = decoded
            try:
                receipt = json.loads(raw_text)
            except json.JSONDecodeError:
                fail(f"admission receipt is not valid JSON: {entry}")
                return known
            if not isinstance(receipt, dict):
                fail(f"admission receipt is not a JSON object: {entry}")
                return known
            digest = receipt.get("token_digest")
            if not isinstance(digest, str) or not digest:
                fail(f"admission receipt has no token_digest: {entry}")
                return known
            state = receipt.get("state")
            if not isinstance(state, str) or not state:
                fail(f"admission receipt has no state: {entry}")
                return known
            run_id = receipt.get("run_id")
            assignment_id = receipt.get("assignment_id")
            ref = ReceiptDigest(
                run_id=str(run_id) if run_id else run_dir.name,
                assignment_id=str(assignment_id) if assignment_id else entry.stem,
                state=state,
                digest=digest,
                receipt_path=repo_relative(repo, entry) or str(entry),
            )
            known.setdefault(digest, []).append(ref)
    return known


def sweep_text(text: str, known: dict[str, list[ReceiptDigest]]) -> list[TokenMatch]:
    """Every candidate substring of ``text`` whose digest matches a known one.

    A dict keyed by digest makes the per-candidate lookup O(1) rather than
    O(known-receipt-count) -- since the receipt count is bounded and
    independent of file size, this keeps the whole sweep linear in file size,
    but the dict alone is only the fast *filter*. The actual digest-equality
    verdict is ``hmac.compare_digest`` against the receipt's own stored
    digest string (a different object than the one just recomputed here),
    matching the same comparison convention ``subagent_stop_validate.py`` uses
    for ``admission_proof`` rather than trusting bare ``==`` in one place and
    ``compare_digest`` in another.
    """
    if not known:
        return []
    matches: list[TokenMatch] = []
    for offset, candidate in find_candidate_tokens(text):
        computed = token_digest(candidate)
        receipts = known.get(computed)
        if not receipts:
            continue
        if not hmac.compare_digest(computed, receipts[0].digest):
            continue
        line = text.count("\n", 0, offset) + 1
        matches.append(TokenMatch(offset=offset, line=line, length=len(candidate), receipts=tuple(receipts)))
    return matches


def iter_sweep_targets(paths: list[Path]) -> list[Path]:
    """Every file to scan for ``--sweep``/``--sweep-run``.

    A plain directory is walked recursively (mirrors the existing ``--run``
    mode's own ``rglob("*")`` plus an ``is_file()`` filter, for consistency);
    anything else -- a file, or a symlink given directly -- is passed through
    as-is so the caller's own ``is_symlink()`` check can report and skip it
    rather than this function silently resolving through it. Like the
    pre-existing ``--run`` path, this does not specially guard against a
    symlinked *directory* discovered inside the tree.
    """
    files: list[Path] = []
    for given in paths:
        if given.is_dir() and not given.is_symlink():
            files.extend(sorted(p for p in given.rglob("*") if p.is_file()))
        else:
            files.append(given)
    return files


# --- ledger-pinned bytes: refuse to scrub -----------------------------------


@dataclass(frozen=True)
class LedgerPin:
    run_id: str
    assignment_id: str
    consumed_at: str
    consumed_by_agent_id: str


RESULT_PATH_RE = re.compile(r"\A\.agent-harness/runs/([^/]+)/results/([^/]+)\.json\Z")


def ledger_pin_for(repo: Path, rel_path: str, current_sha256: str) -> LedgerPin | None:
    """If ``rel_path`` is a run's ``results/<assignment_id>.json`` whose
    CURRENT bytes are pinned by that run's admission ledger, the pin; else
    None.

    Mirrors the exact rule ``validate_harness.collect_result_attribution``
    enforces: the LAST ``consumed`` row for an assignment is a permanent
    statement about specific bytes -- once a result is admitted, its bytes may
    never differ from that row's ``result_sha256``. This tool must never
    itself create the mismatch validate_harness.py exists to catch.

    Fails closed like every other ledger reader in this codebase: a ledger
    that exists for a matching run but cannot be read or parsed is an error,
    not "not pinned" -- an unprovable negative must not default to permission
    to write. A run directory with no ledger at all (legacy, pre-admission,
    or simply not this file's run) is not pinned by this rule; it may still be
    pinned by ``LEGACY_RESULTS_MANIFEST.json``, which this check does not
    consult (out of the stated scope: "a consumed ledger row").
    """
    match = RESULT_PATH_RE.match(rel_path)
    if match is None:
        return None
    run_id, assignment_id = match.group(1), match.group(2)
    ledger_path = repo / ".agent-harness" / "runs" / run_id / "ADMISSIONS.jsonl"
    if not ledger_path.is_file():
        return None
    decoded = read_utf8(ledger_path)
    if decoded is None:
        fail(f"admission ledger is unreadable: {ledger_path}")
        return None
    _, raw = decoded
    last_consumed: dict[str, Any] | None = None
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            fail(f"admission ledger line {number} is malformed: {ledger_path}")
            return None
        if not isinstance(row, dict):
            fail(f"admission ledger line {number} is not an object: {ledger_path}")
            return None
        if row.get("event") == "consumed" and str(row.get("assignment_id")) == assignment_id:
            last_consumed = row
    if last_consumed is None:
        return None
    if str(last_consumed.get("result_sha256")) != current_sha256:
        return None
    return LedgerPin(
        run_id=run_id,
        assignment_id=assignment_id,
        consumed_at=str(last_consumed.get("at") or ""),
        consumed_by_agent_id=str(last_consumed.get("agent_id") or ""),
    )


def warn_refusal(path: Path, pin: LedgerPin) -> None:
    print(
        f"REFUSED: {path} matched, but its bytes are pinned by a consumed "
        f"admission-ledger row (run={pin.run_id!r} assignment={pin.assignment_id!r}, "
        f"consumed_at={pin.consumed_at!r}, consumed_by_agent_id="
        f"{pin.consumed_by_agent_id!r}). Scrubbing would change these bytes and "
        "invalidate the digest validate_harness.py already trusts for this "
        "result. NOT scrubbed. The correct remedy is a declared supersession -- "
        f"mint a fresh receipt with `admit_agent.py --assignment-id "
        f"{pin.assignment_id} --reopen --reason '...'`, produce a newly "
        "scrubbed result, and let it be re-consumed -- never a silent rewrite "
        "of already-admitted bytes.",
        file=sys.stderr,
    )


# --- field-scrub (original behaviour) ---------------------------------------


def scrub(value: Any) -> tuple[Any, int]:
    """Return (scrubbed, count) for any nested JSON structure."""

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        count = 0
        for key, item in value.items():
            if key == "admission_proof" and isinstance(item, str) and item:
                if ALREADY_SCRUBBED.fullmatch(item):
                    out[key] = item
                else:
                    out[key] = token_digest(item)
                    count += 1
            else:
                out[key], sub = scrub(item)
                count += sub
        return out, count
    if isinstance(value, list):
        items = [scrub(item) for item in value]
        return [item for item, _ in items], sum(count for _, count in items)
    if isinstance(value, str):
        # HARNESS_RESULT markers are embedded as text inside last_assistant_message.
        changed = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            token = match.group(2)
            if ALREADY_SCRUBBED.fullmatch(token):
                return match.group(0)
            changed += 1
            return f"{match.group(1)}{token_digest(token)}"

        # re.subn counts every match, including the ones `replace` leaves alone,
        # so count the real substitutions instead or the pass is never idempotent.
        # Tolerate escaped quoting: a HARNESS_RESULT marker embedded inside
        # another JSON string appears as \"admission_proof\": \"token\", and a
        # regex anchored on bare quotes misses it (F-R3-12).
        scrubbed = re.sub(
            r'(admission_proof\\?"\s*:\s*\\?")([^"\\]+)', replace, value
        )
        return scrubbed, changed
    return value, 0


# --- reporting for the anywhere-sweep ---------------------------------------


def print_match(path: Path, match: TokenMatch) -> None:
    for receipt in match.receipts:
        verdict = (
            "spent (consumed receipt -- hygiene)"
            if receipt.spent
            else "LIVE (open receipt -- URGENT)"
        )
        print(
            f"TOKEN MATCH: {path} char {match.offset} line {match.line} "
            f"length {match.length}: receipt run={receipt.run_id!r} "
            f"assignment={receipt.assignment_id!r} state={receipt.state!r} "
            f"({receipt.receipt_path}) -> {verdict}"
        )


def print_pin_note(target: Path, pin: LedgerPin | None) -> None:
    if pin is None:
        return
    print(
        f"  note: {target} is pinned by a consumed admission-ledger row "
        f"(run={pin.run_id!r} assignment={pin.assignment_id!r}, consumed_at="
        f"{pin.consumed_at!r}); do not scrub this file directly -- a declared "
        "supersession (admit_agent.py --reopen) is required first."
    )


def cmd_verify_result(repo: Path, path: Path) -> int:
    """Read-only pre-admission gate for a single result artifact.

    Exits 0 when the file carries no live token match (clean, or only spent
    matches); exits 1 when at least one match is still bound to an open
    receipt. Never writes anything -- catching the leak here means it never
    reaches retained evidence at all, which is what the incident needed.
    """
    if path.is_symlink():
        fail(f"result artifact path is a symlink; refusing to follow it: {path}")
        return 2
    if not path.exists():
        fail(f"no such file: {path}")
        return 2
    if not path.is_file():
        fail(f"not a regular file: {path}")
        return 2
    decoded = read_utf8(path)
    if decoded is None:
        fail(f"result artifact is not readable UTF-8 text: {path}")
        return 2
    _, text = decoded
    known = collect_known_token_digests(repo)
    matches = sweep_text(text, known)
    for match in matches:
        print_match(path, match)
    rel = repo_relative(repo, path)
    if rel is not None and matches:
        print_pin_note(path, ledger_pin_for(repo, rel, file_sha256(decoded[0])))
    live = [m for m in matches if m.live]
    if live:
        print(
            f"VERIFY-RESULT: FAIL -- {path} carries {len(live)} live admission "
            "token match(es) (see above). Do not dispatch Stop for this result "
            "until every occurrence is removed or rewritten; a live token "
            "pasted anywhere in retained evidence is a usable secret "
            "(2026-07-29 R6-regression-live-token incident).",
            file=sys.stderr,
        )
        return 1
    print(f"VERIFY-RESULT: PASS -- {path} carries no live admission token.")
    return 0


def cmd_sweep(repo: Path, given_paths: list[Path]) -> int:
    """Read-only anywhere-sweep over one or more files/directories."""
    for given in given_paths:
        if not given.exists() and not given.is_symlink():
            fail(f"no such path: {given}")
            return 2
    known = collect_known_token_digests(repo)
    targets = iter_sweep_targets(given_paths)
    unreadable: list[Path] = []
    symlinks: list[Path] = []
    scanned = 0
    matches_by_target: dict[Path, list[TokenMatch]] = {}
    for target in targets:
        if target.is_symlink():
            symlinks.append(target)
            continue
        decoded = read_utf8(target)
        if decoded is None:
            unreadable.append(target)
            continue
        scanned += 1
        _, text = decoded
        found = sweep_text(text, known)
        if found:
            matches_by_target[target] = found

    live_count = 0
    spent_count = 0
    for target in sorted(matches_by_target):
        found = matches_by_target[target]
        for match in found:
            print_match(target, match)
            if match.live:
                live_count += 1
            else:
                spent_count += 1
        rel = repo_relative(repo, target)
        if rel is not None:
            raw_bytes = target.read_bytes()  # already proven readable above
            print_pin_note(target, ledger_pin_for(repo, rel, file_sha256(raw_bytes)))

    print(
        f"sweep: scanned {scanned} file(s) against {len(known)} known admission "
        f"token(s); {live_count} LIVE match(es), {spent_count} spent match(es)."
    )
    if symlinks:
        print(f"SWEEP WARNING: {len(symlinks)} symlink(s) not followed:", file=sys.stderr)
        for s in symlinks:
            print(f"  {s}", file=sys.stderr)
    if unreadable:
        print(f"NOT SCANNED (unreadable): {len(unreadable)} file(s)", file=sys.stderr)
        for u in unreadable:
            print(f"  {u}", file=sys.stderr)
    if live_count or symlinks or unreadable:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--run", default="", help="Scrub every JSON under this run.")
    parser.add_argument(
        "--verify-result",
        type=Path,
        default=None,
        metavar="PATH",
        help="Read-only pre-admission gate: sweep PATH for any token-shaped "
        "substring whose SHA-256 matches a known receipt's token_digest, "
        "anywhere in the file, not only in an admission_proof field. Exit 0 "
        "if clean or only spent matches are found; exit 1 if any match is "
        "still bound to an open receipt. Never writes anything.",
    )
    parser.add_argument(
        "--sweep",
        nargs="+",
        type=Path,
        default=None,
        metavar="PATH",
        help="Read-only: the same anywhere/digest-match sweep as "
        "--verify-result, over one or more files or directories (directories "
        "are walked recursively). Reports every match; exit 1 if any is live.",
    )
    parser.add_argument(
        "--sweep-run",
        default="",
        metavar="RUN_ID",
        help="Read-only: --sweep over .agent-harness/runs/<RUN_ID> in full.",
    )
    args = parser.parse_args()

    new_modes = (bool(args.verify_result), bool(args.sweep), bool(args.sweep_run))
    if sum(new_modes) > 1:
        parser.error("pass at most one of --verify-result, --sweep, --sweep-run")
    if any(new_modes) and (args.paths or args.run):
        parser.error(
            "--verify-result/--sweep/--sweep-run cannot be combined with the "
            "scrub paths or --run"
        )

    repo = root()

    if args.verify_result is not None:
        return cmd_verify_result(repo, args.verify_result)
    if args.sweep is not None:
        return cmd_sweep(repo, list(args.sweep))
    if args.sweep_run:
        run_dir = repo / ".agent-harness" / "runs" / args.sweep_run
        if not run_dir.is_dir():
            print(f"scrub_admission_proof: no such run: {args.sweep_run}", file=sys.stderr)
            return 2
        return cmd_sweep(repo, [run_dir])

    targets: list[Path] = list(args.paths)
    if args.run:
        run_dir = repo / ".agent-harness" / "runs" / args.run
        if not run_dir.is_dir():
            print(f"scrub: no such run: {args.run}", file=sys.stderr)
            return 2
        # Every file, not just *.json: tokens also reach .log and .md evidence
        # (D-067 round-3 review F-R3-12).
        targets.extend(sorted(p for p in run_dir.rglob("*") if p.is_file()))
    if not targets:
        parser.error("give at least one path or --run")

    total = 0
    unreadable: list[Path] = []
    refused: list[tuple[Path, LedgerPin]] = []
    for path in targets:
        decoded = read_utf8(path)
        if decoded is None:
            unreadable.append(path)
            continue
        raw_bytes, text = decoded
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            # Not JSON: scrub the raw text, which also catches the escaped
            # payloads a structural walk misses.
            scrubbed_text, count = scrub(text)
            if count:
                rel = repo_relative(repo, path)
                pin = ledger_pin_for(repo, rel, file_sha256(raw_bytes)) if rel else None
                if pin is not None:
                    refused.append((path, pin))
                    warn_refusal(path, pin)
                    continue
                path.write_text(scrubbed_text, encoding="utf-8")
                print(f"scrubbed {count} token(s) as text: {path}")
                total += count
            continue
        scrubbed, count = scrub(value)
        if count:
            rel = repo_relative(repo, path)
            pin = ledger_pin_for(repo, rel, file_sha256(raw_bytes)) if rel else None
            if pin is not None:
                refused.append((path, pin))
                warn_refusal(path, pin)
                continue
            dump_json_atomic(path, scrubbed)
            print(f"scrubbed {count} token(s): {path}")
            total += count
    if unreadable:
        # Silence here would read as "clean" (F-R3-12).
        print(f"NOT SCANNED (unreadable): {len(unreadable)} file(s)", file=sys.stderr)
        for path in unreadable:
            print(f"  {path}", file=sys.stderr)
    if refused:
        print(
            f"REFUSED (ledger-pinned, not scrubbed): {len(refused)} file(s)",
            file=sys.stderr,
        )
        for path, pin in refused:
            print(f"  {path} (run={pin.run_id}, assignment={pin.assignment_id})", file=sys.stderr)
    print(f"total tokens scrubbed: {total}")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
