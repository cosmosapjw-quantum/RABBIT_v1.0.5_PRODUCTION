"""Run the mutation battery as part of the suite, not from a scratch directory.

WHY THIS EXISTS (D-070 Part B22).

This project's stated standard of proof is that *a guard whose deletion breaks
no test is not verified*. Rounds 9 through 13 all applied it -- and applied it
by hand, in ``/tmp``, from throwaway scripts that were deleted with the session.
Every ledger row reading "N of N guards mutation-verified" therefore cites
evidence that no longer exists and cannot be re-derived by anyone, which is the
exact failure this chain keeps finding in other people's records: the claim
outlived the thing that justified it.

Worse, hand-running is what let the misses through. Round 12's battery reported
a dead ``seen`` set as verified until a mutation was written for it; round 13's
first per-document-authority fixture passed under both the fixed and the broken
code because its case could not distinguish them. Both were caught by a battery
that happened to be run that day, by a writer who happened to think of it.

So the battery lives here, in the repository, and the suite fails when a guard
stops being killed by the fixture that is supposed to kill it.

WHAT THIS DOES NOT CLAIM. It is not full mutation coverage: the two checkers
hold well over a hundred error-raising sites and this manifest names a fraction
of them. That gap is DECLARED in the manifest rather than left implicit, and
``test_guard_site_count_has_not_grown_silently`` makes new guards force a
decision instead of arriving uncovered and unnoticed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / ".agent-harness" / "context" / "MUTATION_MANIFEST.json"
GUARD_SITE_RE = re.compile(r"^\s*(?:errors|unresolved)\.append\(", re.MULTILINE)


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


MUTATIONS = load_manifest()["mutations"]


@pytest.fixture(scope="module")
def tree() -> Path:
    """One pristine copy of the tracked tree, reused by every mutation.

    ``git archive`` rather than a directory copy: the incident that filled a
    915 GB disk was a recursive copy of this repository, and the tracked tree is
    24 MB.
    """
    with tempfile.TemporaryDirectory(prefix="mutation-gate-") as tmp:
        target = Path(tmp) / "tree"
        target.mkdir(parents=True)
        archive = Path(tmp) / "tree.tar"
        with archive.open("wb") as handle:
            subprocess.run(["git", "archive", "HEAD"], cwd=REPO, check=True, stdout=handle)
        subprocess.run(["tar", "-x", "-f", str(archive), "-C", str(target)], check=True)
        # Working-tree edits, so the gate measures what is about to be committed
        # rather than what was committed last time. A battery that only ever saw
        # HEAD would pass on exactly the commit that introduced an unverified
        # guard.
        changed = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"], cwd=REPO,
            capture_output=True, text=True, check=True,
        ).stdout.split()
        for rel in changed:
            source = REPO / rel
            if not source.is_file():
                continue
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / rel)
        yield target


def run_tests(tree: Path, node_ids: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header", *node_ids],
        cwd=tree, capture_output=True, text=True,
    )


@pytest.mark.parametrize("entry", MUTATIONS, ids=[m["id"] for m in MUTATIONS])
def test_each_declared_guard_is_killed_by_its_named_fixture(entry: dict, tree: Path) -> None:
    """Remove one guard; the fixtures that name it must go red.

    Three distinct failures are separated deliberately, because they mean
    different things and the previous hand-run batteries conflated the first two:

    * the anchor no longer matching is an ERROR, not a skip -- a refactor that
      moves a guard must not silently retire its own verification;
    * the named fixtures failing BEFORE the mutation means the manifest is
      pointing at something already broken;
    * the fixtures passing WITH the guard removed is the real finding: the guard
      is not verified by them, whatever the ledger says.
    """
    target = tree / entry["target"]
    source = target.read_text(encoding="utf-8")
    occurrences = source.count(entry["anchor"])
    assert occurrences == 1, (
        f"{entry['id']}: anchor matched {occurrences} times in {entry['target']}, expected exactly 1. "
        "A guard that moved must have its manifest entry updated, not silently skipped."
    )

    before = run_tests(tree, entry["kills"])
    assert before.returncode == 0, (
        f"{entry['id']}: the named fixtures already fail before the mutation.\n{before.stdout[-1500:]}"
    )

    target.write_text(source.replace(entry["anchor"], entry["replacement"], 1), encoding="utf-8")
    try:
        after = run_tests(tree, entry["kills"])
    finally:
        target.write_text(source, encoding="utf-8")

    assert after.returncode != 0, (
        f"{entry['id']}: the guard SURVIVED its own removal -- {', '.join(entry['kills'])} "
        f"still pass with it deleted, so they do not verify it.\nWhy it matters: {entry['why']}"
    )


def test_guard_site_count_has_not_grown_silently() -> None:
    """New guards must force a decision instead of arriving uncovered.

    This is a ratchet, not a coverage claim. The manifest names a fraction of
    the error-raising sites in these files, and says so. What it refuses is
    growth: adding a guard without either covering it or restating the declared
    total is exactly how six guards in this chain shipped with no fixture at all.
    """
    manifest = load_manifest()
    declared = manifest["declared_guard_sites"]
    measured = {
        rel: len(GUARD_SITE_RE.findall((REPO / rel).read_text(encoding="utf-8")))
        for rel in sorted(declared)
    }
    assert measured == declared, (
        "guard-site counts have drifted from the declaration.\n"
        f"  declared: {declared}\n  measured: {measured}\n"
        "Cover the new guard with a MUTATION_MANIFEST entry, or update "
        "'declared_guard_sites' and say in 'coverage_gap' why it is uncovered."
    )


def test_manifest_targets_and_fixtures_exist() -> None:
    """A manifest entry naming a file or a test that is gone is an error.

    Same reason the facts file re-verifies every declaration on every run: a
    dead entry that is silently ignored accumulates until the battery is mostly
    decoration.
    """
    problems: list[str] = []
    for entry in MUTATIONS:
        if not (REPO / entry["target"]).is_file():
            problems.append(f"{entry['id']}: target {entry['target']} does not exist")
        for node in entry["kills"]:
            rel, _, name = node.partition("::")
            path = REPO / rel
            if not path.is_file():
                problems.append(f"{entry['id']}: fixture file {rel} does not exist")
            elif f"def {name}(" not in path.read_text(encoding="utf-8"):
                problems.append(f"{entry['id']}: fixture {name} is not defined in {rel}")
    assert not problems, "\n".join(problems)
