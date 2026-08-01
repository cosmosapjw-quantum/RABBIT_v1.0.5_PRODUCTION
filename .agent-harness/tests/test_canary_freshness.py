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


def attest(repo: Path, run: str, document: dict[str, object]) -> Path:
    path = repo / ".agent-harness/runs" / run / "artifacts/ATTESTATION.json"
    write(path, json.dumps({"artifact_class": "CANARY_ATTESTATION", **document}, indent=1))
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
    """`result_sha256` names an envelope, not a source file, and must not error."""
    attest(repo, "run-canary-c7", fresh(repo, result_sha256="sha256:" + "0" * 64))
    assert C.check(repo) == []
