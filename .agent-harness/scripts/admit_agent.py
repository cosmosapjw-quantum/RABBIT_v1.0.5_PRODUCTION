#!/usr/bin/env python3
"""Mint a single-use admission receipt binding one spawned agent to one assignment.

Why this exists (D-067, discharging D-065 finding F-D065-01): the SubagentStart
hook event carries only ``agent_id`` and ``agent_type``. It never carries the
spawn prompt, so no hook can infer which assignment an agent was launched for.
The D-058 lease therefore sealed *every* assignment registered in the run, and
SubagentStop admitted whichever one the stopping agent declared for itself. A
controlled same-run substitution passed.

The fix moves the binding to the only party that knows it: the parent. Before
spawning, the parent runs this script, which stores ``sha256(token)`` in the
receipt and prints ``token`` exactly once for pasting into the spawn prompt.
SubagentStop admits the agent only if the ``admission_proof`` in its
HARNESS_RESULT envelope hashes to the stored digest, then consumes the receipt,
recording the writing ``agent_id`` and the result SHA-256 as the
write-attribution record.

TOKEN HANDLING -- this script never writes the raw token to disk, but the token
does not stay secret by itself. It is pasted into a spawn prompt, echoed in the
agent's final message, and therefore appears in any transcript or captured
SubagentStop event JSON. A captured *blocked* stop is the dangerous case,
because the receipt is still open. Two consequences, both load-bearing:

* Always pass ``--expect-agent-id`` when the parent chooses the agent id (it
  does under the recorded VS Code LOCAL-ADAPT). A leaked token is then useless
  to any other agent.
* Scrub ``admission_proof`` before retaining a stop-event JSON as run evidence;
  ``scrub_admission_proof.py`` does this, replacing the value with its digest.

An already-consumed receipt cannot be re-admitted regardless.

THREAT MODEL -- read this before claiming more than it delivers. The harness and
its subagents run as the same OS user, so an agent that deliberately writes
outside its declared result path can forge any repo-local artifact, including a
receipt, a lease, and the ledger. What this mechanism removes is *confusion*:
prompt-level substitution, same-run assignment mix-ups, races between
overlapping mains, and unattributable writes. Deliberate forgery is not
prevented.

Usage:
    python3 .agent-harness/scripts/admit_agent.py --assignment-id A-XXXX
    python3 .agent-harness/scripts/admit_agent.py --assignment-id A-XXXX \\
        --expect-agent-id canary-01
    python3 .agent-harness/scripts/admit_agent.py --assignment-id A-XXXX \\
        --reopen --reason "first agent died before stopping"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path

from _harness import (
    ADMISSION_KEY_RE,
    ASSIGNMENT_ID_RE,
    active_run_id,
    admission_path,
    assignment_runtime_agent_type,
    dump_json_atomic,
    load_json,
    root,
    token_digest,
    utc_now,
)

TOKEN_BYTES = 32


def fail(message: str) -> None:
    raise SystemExit(f"admit_agent: {message}")


def last_consumed_result_sha256(ledger_path: Path, assignment_id: str) -> str:
    """The digest of the bytes the ledger last admitted for this assignment.

    A reopen must *declare* which admitted digest it supersedes (D-067 round-4
    review, defect 2). The declaration is taken from the append-only ledger
    rather than from the receipt, because the receipt directory is gitignored
    working state and may have been replaced or corrupted -- the ledger is the
    record that survives into committed evidence. Fails closed: an existing
    ledger that cannot be read or parsed aborts the mint rather than emitting a
    reopen row that declares nothing.
    """
    if not ledger_path.exists():
        return ""
    try:
        raw = ledger_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        fail(
            f"admission ledger could not be read ({exc.__class__.__name__}); a "
            "reopen must declare the digest it supersedes"
        )
        return ""
    digest = ""
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            fail(
                f"admission ledger line {number} is malformed; a reopen must "
                "declare the digest it supersedes"
            )
            return ""
        if not isinstance(row, dict):
            fail(
                f"admission ledger line {number} is not an object; a reopen must "
                "declare the digest it supersedes"
            )
            return ""
        if (
            row.get("event") == "consumed"
            and str(row.get("assignment_id")) == assignment_id
        ):
            digest = str(row.get("result_sha256") or "")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Defaults to ACTIVE_RUN; must equal it (admission is for the live run).",
    )
    parser.add_argument(
        "--expect-agent-id",
        default="",
        help="Bind the receipt to this agent_id as well as to the token. "
        "SubagentStop then rejects any other agent even if it holds the token. "
        "Required unless --agent-id-unknown is passed.",
    )
    parser.add_argument(
        "--agent-id-unknown",
        action="store_true",
        help="Mint without an agent_id binding, for runtimes that assign the id "
        "themselves and do not expose it before the spawn. The receipt is then "
        "bound to the token only, which is weaker: record why in the run.",
    )
    parser.add_argument(
        "--reopen",
        action="store_true",
        help="Replace an existing receipt for this assignment. Requires --reason. "
        "Appends a declared `reopened` ledger row naming the result digest it "
        "supersedes, which is what licenses the second consume row that follows: "
        "validate_harness errors on an undeclared second consume. Until the "
        "reopened assignment is consumed again, it also reports any result whose "
        "bytes differ from the last consume row.",
    )
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    repo = root()
    harness = repo / ".agent-harness"

    assignment_id = str(args.assignment_id)
    if not ASSIGNMENT_ID_RE.fullmatch(assignment_id):
        fail(f"assignment_id {assignment_id!r} has an invalid form")
    expect_agent_id = str(args.expect_agent_id or "")
    if expect_agent_id and not ADMISSION_KEY_RE.fullmatch(expect_agent_id):
        fail(f"expect-agent-id {expect_agent_id!r} has an invalid form")
    if not expect_agent_id and not args.agent_id_unknown:
        # The D-065 obligation is an agent-to-assignment binding. Leaving the
        # agent half to convention is how it quietly degrades to token-only
        # (D-067 round-3 review F-R3-15).
        fail(
            "--expect-agent-id is required so the receipt binds an agent, not "
            "just a token. Pass --agent-id-unknown only when the runtime assigns "
            "the agent id itself and does not expose it before the spawn."
        )

    active = active_run_id(repo)
    run_id = str(args.run_id or active)
    if run_id != active:
        fail(
            f"run_id {run_id!r} is not the active run {active!r}; admission is only "
            "minted for the live run"
        )

    assignment_file = harness / "runs" / run_id / "assignments" / f"{assignment_id}.json"
    try:
        assignment_raw = assignment_file.read_bytes()
    except OSError:
        fail(f"registered assignment does not exist: {assignment_file.relative_to(repo)}")
        return
    assignment = load_json(assignment_file)
    if not isinstance(assignment, dict):
        fail("registered assignment is not a JSON object")
        return
    if str(assignment.get("run_id")) != run_id:
        fail("registered assignment run_id does not match the active run")

    index = load_json(harness / "context" / "CONTEXT_INDEX.json")
    context_version = str(index.get("context_version", "UNBUILT"))
    if str(assignment.get("context_version")) != context_version:
        fail(
            f"registered assignment is stale for context {context_version!r}; rebuild "
            "the pack and reissue the assignment before admitting an agent"
        )

    target = admission_path(harness, run_id, assignment_id)
    if target is None:
        fail("run_id or assignment_id is not a safe admission path key")
        return

    prior = None
    if target.exists():
        try:
            prior = load_json(target)
        except (OSError, json.JSONDecodeError):
            prior = {"state": "unreadable", "created_at": "", "consumed_by_agent_id": ""}
        if not isinstance(prior, dict):
            prior = {"state": "malformed", "created_at": "", "consumed_by_agent_id": ""}
    if prior is not None:
        if not args.reopen:
            fail(
                f"an admission receipt already exists for {assignment_id!r} in state "
                f"{str(prior.get('state'))!r}. Receipts are single-use; pass --reopen "
                "--reason '...' to replace it deliberately."
            )
        if not args.reason.strip():
            fail("--reopen requires a non-empty --reason")

    token = secrets.token_urlsafe(TOKEN_BYTES)
    receipt = {
        "schema_version": 1,
        "run_id": run_id,
        "assignment_id": assignment_id,
        "assignment_sha256": "sha256:" + hashlib.sha256(assignment_raw).hexdigest(),
        "runtime_agent_type": assignment_runtime_agent_type(assignment),
        "context_version": context_version,
        "token_digest": token_digest(token),
        "expected_agent_id": expect_agent_id,
        "state": "open",
        "created_at": utc_now(),
    }
    if prior is not None:
        receipt["reopened_from"] = {
            "state": str(prior.get("state")),
            "created_at": str(prior.get("created_at")),
            "consumed_at": str(prior.get("consumed_at") or ""),
            "consumed_by_agent_id": str(prior.get("consumed_by_agent_id") or ""),
            "consumed_by_agent_type": str(prior.get("consumed_by_agent_type") or ""),
            "result_sha256": str(prior.get("result_sha256") or ""),
            "reason": args.reason.strip(),
        }

    # A reopen must also release the O_EXCL claim, or the new agent cannot
    # consume the receipt it was just admitted for.
    claim = target.with_name(target.name + ".claim")
    minted_at = utc_now()
    ledger_path = harness / "runs" / run_id / "ADMISSIONS.jsonl"

    # A reopen is a *declared* supersession, not merely a second mint. Before
    # this the reopen row was indistinguishable from a first mint (`"event":
    # "minted"`), so a second consume row could appear with nothing on the
    # append-only record saying that superseding the first was intended -- which
    # is how an edited artifact was laundered back to green (D-067 round-4
    # review, defect 2). The distinct event, the superseded digest, and the
    # mandatory reason are what validate_harness requires before it will accept
    # more than one consume row for an assignment.
    row: dict[str, object] = {
        "event": "reopened" if prior is not None else "minted",
        "run_id": run_id,
        "assignment_id": assignment_id,
        "assignment_sha256": receipt["assignment_sha256"],
        "token_digest": receipt["token_digest"],
        "expected_agent_id": expect_agent_id,
        "reopened": bool(prior is not None),
        "at": minted_at,
    }
    if prior is not None:
        superseded = last_consumed_result_sha256(ledger_path, assignment_id)
        if not superseded:
            # No consume row yet: the receipt being replaced was never admitted
            # (a dead agent, a failed stop). Fall back to whatever the receipt
            # itself recorded so the row is still explicit about what it found.
            superseded = str(prior.get("result_sha256") or "")
        row["superseded_result_sha256"] = superseded
        row["superseded_state"] = str(prior.get("state"))
        row["superseded_consumed_by_agent_id"] = str(
            prior.get("consumed_by_agent_id") or ""
        )
        row["reason"] = args.reason.strip()

    # Ledger first, receipt second. If the ledger append fails the receipt was
    # never created, so no open receipt survives whose token was never printed
    # (D-067 round-2 review F-13).
    try:
        with ledger_path.open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError as exc:
        fail(f"admission ledger could not be appended ({exc.__class__.__name__})")
    try:
        dump_json_atomic(target, receipt)
        claim.unlink(missing_ok=True)
    except OSError as exc:
        fail(f"admission receipt could not be written ({exc.__class__.__name__})")

    print(f"RUN_ID={run_id}")
    print(f"ASSIGNMENT_ID={assignment_id}")
    print(f"CONTEXT_VERSION={context_version}")
    print(f"INDEPENDENCE_MODE={assignment.get('independence_mode')}")
    print(f"ADMISSION_TOKEN={token}")
    print(f"receipt: {target.relative_to(repo)}")


if __name__ == "__main__":
    main()
