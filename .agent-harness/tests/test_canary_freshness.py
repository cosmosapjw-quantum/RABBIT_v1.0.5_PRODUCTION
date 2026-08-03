"""Regression cases for check_canary_freshness.py (D-070 Part B10, F-R9-004).

Four canaries died of an attestation outliving the code it attested, and every
remedy was a written rule that the next canary then broke. These fixtures exist
so the rule is measured instead. Each one must die if its guard is removed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_canary_freshness as C  # noqa: E402

START_HOOK = ".codex/hooks/subagent_start_context.py"
STOP_HOOK = ".codex/hooks/subagent_stop_validate.py"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    write(tmp_path / START_HOOK, "# start hook v1\n")
    write(tmp_path / STOP_HOOK, "# stop hook v1\n")
    return tmp_path


def attest(repo: Path, run: str, document: dict[str, object], *, backed: bool = True) -> Path:
    """Write an attestation; by default also forge the receipt trail it needs.

    Round 10 defeated the detector with a file nobody dispatched, so an
    attestation now has to point at a consumed admission-ledger row and the
    result that row pins. `backed=False` writes the bare file the attack used.
    """
    run_dir = repo / ".agent-harness/runs" / run
    path = run_dir / "artifacts/ATTESTATION.json"
    body = {"artifact_class": "CANARY_ATTESTATION", **document}
    if backed:
        assignment = str(body.setdefault("consumed_assignment_id", "A-CANARY"))
        agent = str(body.setdefault("consumed_by_agent_id", "canary-agent"))
        body.setdefault("canary_lease_run_id", run)
        result = run_dir / "results" / f"{assignment}.json"
        write(result, json.dumps({"status": "pass"}, indent=1))
        digest = "sha256:" + hashlib.sha256(result.read_bytes()).hexdigest()
        body.setdefault("result_sha256", digest)
        write(
            run_dir / "ADMISSIONS.jsonl",
            json.dumps(
                {
                    "event": "consumed",
                    "assignment_id": assignment,
                    "agent_id": agent,
                    "result_sha256": digest,
                }
            )
            + "\n",
        )
    write(path, json.dumps(body, indent=1))
    return path


def fresh(repo: Path, canary: str = "C7", **extra: object) -> dict[str, object]:
    return {
        "canary": canary,
        "start_hook_sha256": digest((repo / START_HOOK).read_text(encoding="utf-8")),
        "stop_hook_sha256": digest((repo / STOP_HOOK).read_text(encoding="utf-8")),
        **extra,
    }


def test_a_fresh_canary_passes(repo: Path) -> None:
    attest(repo, "run-canary-c7", fresh(repo))
    assert C.check(repo) == []


def test_editing_an_attested_hook_makes_the_canary_stale(repo: Path) -> None:
    """The exact class that killed C5 and C6, now measured rather than noticed."""
    attest(repo, "run-canary-c7", fresh(repo))
    write(repo / START_HOOK, "# start hook v2 -- one line changed\n")
    errors = C.check(repo)
    assert any("C7 is STALE" in m and START_HOOK in m for m in errors), errors


def test_a_stale_canary_superseded_by_a_fresh_one_is_retired(repo: Path) -> None:
    """Retirement is by replacement; retained attestations are never rewritten."""
    stale = fresh(repo, canary="C6")
    write(repo / START_HOOK, "# start hook v2\n")
    attest(repo, "run-canary-c6", stale)
    attest(repo, "run-canary-c7", fresh(repo, supersedes_canary=["C6"]))
    assert C.check(repo) == []


def test_superseding_the_wrong_canary_does_not_retire_the_stale_one(repo: Path) -> None:
    stale = fresh(repo, canary="C6")
    write(repo / START_HOOK, "# start hook v2\n")
    attest(repo, "run-canary-c6", stale)
    attest(repo, "run-canary-c7", fresh(repo, supersedes_canary=["C4"]))
    assert any("C6 is STALE" in m for m in C.check(repo))


def test_id_is_inferred_from_the_run_directory_when_the_field_is_absent(repo: Path) -> None:
    """C5 predates the `canary` field and its bytes are evidence, not editable."""
    document = fresh(repo)
    del document["canary"]
    attest(repo, "run-20260729-f10-d070-canary-c5", document)
    write(repo / STOP_HOOK, "# stop hook v2\n")
    assert any("C5 is STALE" in m for m in C.check(repo))


def test_an_attestation_with_no_resolvable_id_is_an_error_not_a_skip(repo: Path) -> None:
    document = fresh(repo)
    del document["canary"]
    attest(repo, "run-something-else", document)
    assert any("cannot be superseded by anything" in m for m in C.check(repo))


def test_a_canary_attesting_no_hook_digest_is_an_error(repo: Path) -> None:
    """A canary that cannot go stale cannot be evidence that anything holds."""
    attest(repo, "run-canary-c7", {"canary": "C7", "method": "words only"})
    assert any("attests no hook digest" in m for m in C.check(repo))


def test_an_unmappable_digest_field_is_an_error_not_a_skip(repo: Path) -> None:
    attest(repo, "run-canary-c7", fresh(repo, merge_hook_sha256="0" * 64))
    assert any("cannot map to a file" in m for m in C.check(repo))


def test_an_attested_file_that_does_not_exist_is_an_error(repo: Path) -> None:
    attest(repo, "run-canary-c7", fresh(repo))
    (repo / START_HOOK).unlink()
    assert any("does not exist" in m for m in C.check(repo))


def test_non_file_digest_fields_are_not_treated_as_attestations(repo: Path) -> None:
    """`assignment_sha256` names an envelope, not a source file, and must not error.

    `result_sha256` used to be the example here; it is now load-bearing, because
    receipt backing re-derives it from the pinned result. The exclusion list
    still has to keep genuinely non-file digests out of the attested set.
    """
    attest(repo, "run-canary-c7", fresh(repo, assignment_sha256="sha256:" + "0" * 64))
    assert C.check(repo) == []


# --------------------------------------------------------------------------
# Round 10 (D-070 Part B13). The detector was defeated by a text editor.
# --------------------------------------------------------------------------


def test_f_r10_a_file_nobody_dispatched_cannot_retire_a_real_canary(repo: Path) -> None:
    """The decisive round-10 finding, as the attack was run.

    One typed JSON file with its digests set to the current bytes retired the
    real canary and returned ok:true while the attested hook carried a hostile
    edit. Only a canary backed by a consumed receipt may retire another.
    """
    attest(repo, "run-canary-c7", fresh(repo, canary="C7"))
    write(repo / START_HOOK, "# hostile edit\n")
    forged = fresh(repo, canary="C999", supersedes_canary=["C7"])
    attest(repo, "run-forged", forged, backed=False)
    errors = C.check(repo)
    assert any("not itself backed by a consumed admission receipt" in m for m in errors), errors
    assert any("C7 is STALE" in m for m in errors), errors


def test_f_r10_a_canary_cannot_supersede_itself(repo: Path) -> None:
    attest(repo, "run-canary-c7", fresh(repo, canary="C7", supersedes_canary=["C7"]))
    write(repo / START_HOOK, "# hostile edit\n")
    errors = C.check(repo)
    assert any("names ITSELF" in m for m in errors), errors
    assert any("C7 is STALE" in m for m in errors), errors


def test_f_r10_two_canaries_cannot_retire_each_other(repo: Path) -> None:
    """A cycle would leave both unreachable by the freshness check."""
    attest(repo, "run-canary-c6", fresh(repo, canary="C6", supersedes_canary=["C7"]))
    attest(repo, "run-canary-c7", fresh(repo, canary="C7", supersedes_canary=["C6"]))
    write(repo / START_HOOK, "# hostile edit\n")
    errors = C.check(repo)
    assert any("supersession cycle" in m for m in errors), errors


def test_f_r10_an_attestation_must_name_a_real_consumed_row(repo: Path) -> None:
    document = fresh(repo, canary="C7")
    document["consumed_by_agent_id"] = "somebody-else"
    attest(repo, "run-canary-c7", document, backed=False)
    errors = C.check(repo)
    assert any("names no receipt" in m or "no such consumed row exists" in m for m in errors), errors


def test_f_r10_a_pinned_result_digest_is_re_derived_not_trusted(repo: Path) -> None:
    attest(repo, "run-canary-c7", fresh(repo, canary="C7"))
    result = repo / ".agent-harness/runs/run-canary-c7/results/A-CANARY.json"
    write(result, json.dumps({"status": "tampered"}, indent=1))
    errors = C.check(repo)
    assert any("hashes to" in m and "disagree" in m for m in errors), errors


def test_f_r10_an_attestation_hidden_deeper_is_still_found(repo: Path) -> None:
    """The single-level glob was a hiding place, not a filter."""
    attest(repo, "run-canary-c7", fresh(repo, canary="C7"))
    deep = repo / ".agent-harness/runs/run-hidden/artifacts/sub/deep.json"
    write(
        deep,
        json.dumps(
            {"artifact_class": "CANARY_ATTESTATION", "canary": "C888",
             "start_hook_sha256": "0" * 64, "stop_hook_sha256": "0" * 64},
            indent=1,
        ),
    )
    assert any("C888" in m for m in C.check(repo))


def test_f_r10_zero_live_canaries_is_an_error_where_one_is_declared(repo: Path) -> None:
    """runs/ is gitignored wholesale, so a missing attestation looked like a pass."""
    write(repo / C.POLICY_FILE, json.dumps({"schema_version": 1, "min_live": 1}))
    assert any("live retained canary attestation" in m for m in C.check(repo))


def test_f_r10_a_tree_declaring_no_canary_policy_needs_no_canary(repo: Path) -> None:
    """A synthetic fixture holds no canary evidence and must not be forced to
    invent some. Absence of the declaration is the permission, not a bypass."""
    assert C.check(repo) == []


def test_f_r10_deleting_the_policy_cannot_lower_the_floor(repo: Path) -> None:
    """If git tracks the declaration and the tree has lost it, that is an error."""
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    write(repo / C.POLICY_FILE, json.dumps({"schema_version": 1, "min_live": 1}))
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "policy"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / C.POLICY_FILE).unlink()
    assert any("cannot be lowered by deleting" in m for m in C.check(repo))


# --------------------------------------------------------------------------
# F-R11 -- the four canary defects an external audit reported and a local
# reproduction confirmed (D-070 Part B16).
#
# Every one of them made the checker report `ok: true` on a tree whose canary
# evidence was absent, duplicated, or mutually retired into invisibility. The
# three-node cycle is the worst of them: the checker printed `live: [C4]` and
# exited 0 while C1, C2 and C3 had retired each other out of the record.
# --------------------------------------------------------------------------


def git_repo(repo: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    write(repo / ".gitignore", "/.agent-harness/runs/\n")


def commit_all(repo: Path, *force: Path) -> None:
    import subprocess

    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    for path in force:
        subprocess.run(["git", "add", "-f", str(path.relative_to(repo))],
                       cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "x"], cwd=repo, check=True, capture_output=True)


def retained(repo: Path, run: str) -> list[Path]:
    """Every file a canary needs force-added, since runs/ is gitignored."""
    base = repo / ".agent-harness/runs" / run
    return [base / "artifacts/ATTESTATION.json", base / "ADMISSIONS.jsonl",
            base / "results/A-CANARY.json"]


def test_f_r11_three_node_supersession_cycle_is_refused(repo: Path) -> None:
    """C1 -> C2 -> C3 -> C1 passed, reporting all three as superseded.

    The independent C4 matters: without it the floor would fail anyway and the
    case would prove nothing about the graph. Round 9 lost three fixtures to
    exactly that mistake, so the discriminator is built in.
    """
    attest(repo, "run-a-canary-c1", fresh(repo, canary="C1", supersedes_canary=["C2"]))
    attest(repo, "run-b-canary-c2", fresh(repo, canary="C2", supersedes_canary=["C3"]))
    attest(repo, "run-c-canary-c3", fresh(repo, canary="C3", supersedes_canary=["C1"]))
    attest(repo, "run-d-canary-c4", fresh(repo, canary="C4"))
    errors = C.check(repo)
    assert any("supersession cycle" in m for m in errors), errors
    assert any("C1" in m and "C2" in m and "C3" in m for m in errors), errors


def test_f_r11_each_distinct_cycle_is_reported_exactly_once(repo: Path) -> None:
    """One error per cycle -- not one per rotation, and not one for two cycles.

    Two INDEPENDENT cycles are used rather than one, because a single cycle
    cannot distinguish "reported once" from "reported once per graph": the walk
    visits each node at most once, so one cycle is structurally incapable of
    being reported twice. A fixture built on one cycle passed with the
    de-duplication removed, which is how the dead `seen` set was found.
    """
    attest(repo, "run-a-canary-c1", fresh(repo, canary="C1", supersedes_canary=["C2"]))
    attest(repo, "run-b-canary-c2", fresh(repo, canary="C2", supersedes_canary=["C1"]))
    attest(repo, "run-c-canary-c3", fresh(repo, canary="C3", supersedes_canary=["C4"]))
    attest(repo, "run-d-canary-c4", fresh(repo, canary="C4", supersedes_canary=["C3"]))
    attest(repo, "run-e-canary-c9", fresh(repo, canary="C9"))
    reported = [m for m in C.check(repo) if "supersession cycle" in m]
    assert len(reported) == 2, reported
    assert any("C1 -> C2 -> C1" in m for m in reported), reported
    assert any("C3 -> C4 -> C3" in m for m in reported), reported


def test_f_r11_an_acyclic_chain_is_still_legal(repo: Path) -> None:
    """The control the cycle fix must not break: C3 -> C2 -> C1 is a normal
    replacement history and retires two canaries without complaint."""
    write(repo / START_HOOK, "# start hook v2\n")
    attest(repo, "run-a-canary-c1", fresh(repo, canary="C1"))
    attest(repo, "run-b-canary-c2", fresh(repo, canary="C2", supersedes_canary=["C1"]))
    attest(repo, "run-c-canary-c3", fresh(repo, canary="C3", supersedes_canary=["C2"]))
    assert C.check(repo) == []


def test_f_r11_two_files_claiming_one_canary_id_do_not_count_twice(repo: Path) -> None:
    """`live: [C1, C1]` satisfied a declared floor of two and reported clean."""
    git_repo(repo)
    write(repo / C.POLICY_FILE, json.dumps({"schema_version": 1, "min_live": 2}))
    attest(repo, "run-a-canary-c1", fresh(repo, canary="C1"))
    attest(repo, "run-b-canary-c1", fresh(repo, canary="C1"))
    commit_all(repo, *retained(repo, "run-a-canary-c1"), *retained(repo, "run-b-canary-c1"))
    errors = C.check(repo)
    assert any("is attested by 2 live files" in m for m in errors), errors
    assert any("live retained canary attestation" in m for m in errors), errors


def test_f_r11_untracked_canary_evidence_does_not_satisfy_the_floor(repo: Path) -> None:
    """runs/ is gitignored, so an attestation nobody force-added exists on ONE
    disk. Counting it reproduces the fail-open default the floor was written
    to close, one level up."""
    git_repo(repo)
    write(repo / C.POLICY_FILE, json.dumps({"schema_version": 1, "min_live": 1}))
    attest(repo, "run-a-canary-c1", fresh(repo, canary="C1"))
    commit_all(repo)  # deliberately NOT force-adding the run directory
    errors = C.check(repo)
    assert any("does not count toward the live floor" in m for m in errors), errors
    assert any("live retained canary attestation" in m for m in errors), errors


def test_f_r11_tracked_canary_evidence_does_satisfy_the_floor(repo: Path) -> None:
    """The control: the same canary, force-added, is clean. Without this the
    test above would pass even if the floor rejected everything."""
    git_repo(repo)
    write(repo / C.POLICY_FILE, json.dumps({"schema_version": 1, "min_live": 1}))
    attest(repo, "run-a-canary-c1", fresh(repo, canary="C1"))
    commit_all(repo, *retained(repo, "run-a-canary-c1"))
    assert C.check(repo) == []


def test_f_r11_an_untracked_policy_cannot_replace_the_tracked_one(repo: Path) -> None:
    """The tracked-deletion guard ran only on the ABSENT branch, so removing the
    declaration from the index and leaving an untracked `min_live: 0` in its
    place lowered the floor to zero with no error at all."""
    import subprocess

    git_repo(repo)
    write(repo / C.POLICY_FILE, json.dumps({"schema_version": 1, "min_live": 1}))
    commit_all(repo)
    subprocess.run(["git", "rm", "--cached", "-q", C.POLICY_FILE], cwd=repo,
                   check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "untrack"], cwd=repo, check=True, capture_output=True)
    write(repo / C.POLICY_FILE, json.dumps({"schema_version": 1, "min_live": 0}))
    errors = C.check(repo)
    assert any("git does not track it" in m for m in errors), errors
