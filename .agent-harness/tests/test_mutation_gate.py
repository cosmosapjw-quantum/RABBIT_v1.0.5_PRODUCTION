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
        # Working-tree state, so the gate measures what is about to be committed
        # rather than what was committed last time. A battery that only ever saw
        # HEAD would pass on exactly the commit that introduced an unverified
        # guard.
        #
        # `git status --porcelain` rather than `git diff --name-only HEAD`: the
        # latter misses UNTRACKED files, so a commit introducing a brand-new
        # script left that script out of the scratch tree and every fixture
        # depending on it failed there for a reason that had nothing to do with
        # any mutation. Found while adding build_status_board.py, which is
        # exactly the case -- a new file arriving with the guards that need it.
        changed = [
            line[3:].strip().strip('"')
            for line in subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO,
                capture_output=True, text=True, check=True,
            ).stdout.splitlines()
            if line[:2] != "D " and line[:2] != " D"
        ]
        for rel in changed:
            source = REPO / rel
            if not source.is_file():
                continue
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / rel)
        yield target


def run_tests(tree: Path, node_ids: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        # -rf prints one "FAILED <nodeid>" line per failing test, and --tb=no keeps
        # the output small. Both are required by the FAILED-line check below: -q
        # alone collapses outcomes into dots and makes a kill indistinguishable
        # from a collection error.
        [sys.executable, "-m", "pytest", "-q", "-rf", "--tb=no",
         "-p", "no:cacheprovider", "--no-header", *node_ids],
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

    mutated = source.replace(entry["anchor"], entry["replacement"], 1)

    # F10-DESIGN-002, first half. A mutation that does not COMPILE is a broken
    # mutation, not a killed guard. The third-party auditor reproduced this: a
    # replacement of `if status == expected or:` made the checker unimportable,
    # every named fixture failed for that reason, and the gate recorded a
    # successful kill. The guard it claimed to verify was never exercised at all.
    try:
        compile(mutated, str(target), "exec")
    except SyntaxError as exc:
        raise AssertionError(
            f"{entry['id']}: the MUTATION is broken, not the guard -- the mutated "
            f"{entry['target']} does not compile ({exc.msg} at line {exc.lineno}). "
            "A replacement must be valid code that removes one guard, or the fixtures "
            "fail for a reason that has nothing to do with what is being verified."
        ) from None

    target.write_text(mutated, encoding="utf-8")
    try:
        after = run_tests(tree, entry["kills"])
    finally:
        target.write_text(source, encoding="utf-8")

    assert after.returncode != 0, (
        f"{entry['id']}: the guard SURVIVED its own removal -- {', '.join(entry['kills'])} "
        f"still pass with it deleted, so they do not verify it.\nWhy it matters: {entry['why']}"
    )

    # F10-DESIGN-002, second half, and the part that actually closes it. A
    # non-zero exit says only "something went wrong". pytest reports a fixture
    # that FAILED an assertion differently from one that ERRORed during
    # collection or setup, so requiring every named test to appear as FAILED is
    # what distinguishes "the guard was missed" from "the run fell over". The
    # compile check above cannot cover this on its own: a mutation can be
    # perfectly valid Python and still break an import, a fixture, or a
    # module-level constant that the named tests need before they ever assert.
    missing = [node for node in entry["kills"] if f"FAILED {node}" not in after.stdout]
    assert not missing, (
        f"{entry['id']}: the guard's removal did not make {', '.join(missing)} FAIL an "
        "assertion -- the run went non-zero for some other reason (collection error, "
        "import error, or fixture error). That is a broken mutation reported as a kill, "
        f"which is what the mutation gate exists to prevent.\n{after.stdout[-1500:]}"
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


# --------------------------------------------------------------------------
# F10-DESIGN-003 (third-party design re-audit). The manifest had no schema and
# no coverage identity, so `schema_version: 999` plus an unknown top-level key
# was accepted silently, and deleting an entry simply reduced parametrisation
# from 19 cases to 18 with the suite still green. A battery that shrinks without
# complaining is a battery that can be quietly emptied.
# --------------------------------------------------------------------------

MANIFEST_KEYS = {
    "schema_version", "purpose", "declared_guard_sites", "coverage_gap",
    "required_ids", "mutations",
}
ENTRY_KEYS = {"id", "target", "anchor", "replacement", "kills", "why"}


def test_manifest_matches_a_closed_schema() -> None:
    """Unknown keys are an error, not decoration.

    Every declaration file in this harness is closed the same way, for the same
    reason: a key nothing reads is a key that can carry a false statement
    indefinitely. `SSOT_FACTS.json` has FACT_KEYS, `check_canary_freshness.py`
    has ATTESTED_FILES, and this is the same rule for the battery's own manifest.
    """
    manifest = load_manifest()
    problems: list[str] = []
    if manifest.get("schema_version") != 1:
        problems.append(f"schema_version is {manifest.get('schema_version')!r}, expected 1")
    unknown = set(manifest) - MANIFEST_KEYS
    if unknown:
        problems.append(f"unknown top-level keys: {sorted(unknown)}")
    for key in MANIFEST_KEYS:
        if key not in manifest:
            problems.append(f"missing required top-level key {key!r}")
    seen: set[str] = set()
    for index, entry in enumerate(manifest.get("mutations", [])):
        where = entry.get("id", f"entry[{index}]")
        extra = set(entry) - ENTRY_KEYS
        if extra:
            problems.append(f"{where}: unknown keys {sorted(extra)}")
        for key in ENTRY_KEYS:
            if key not in entry:
                problems.append(f"{where}: missing required key {key!r}")
        if not entry.get("kills"):
            problems.append(f"{where}: 'kills' is empty -- a mutation that names no fixture "
                            "verifies nothing")
        if entry.get("id") in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(entry.get("id"))
    assert not problems, "\n".join(problems)


def test_every_required_guard_is_still_covered() -> None:
    """A pinned floor, so the battery cannot be emptied one entry at a time.

    The declared coverage is partial and says so, which means a plain count is
    not a floor -- 18 of 131 is as 'partial' as 19 of 131. What makes it a floor
    is naming the guards that may never lose their mutation: the ones that can
    change canonical gate state, admit or substitute evidence, authorise a
    retirement, or bind provenance. Everything else may come and go with the
    code it guards.
    """
    manifest = load_manifest()
    required = set(manifest["required_ids"])
    present = {entry["id"] for entry in manifest["mutations"]}
    missing = required - present
    assert not missing, (
        f"required mutation entries are gone: {sorted(missing)}. These cover guards that can "
        "change a gate, admit or substitute evidence, authorise a retirement, or bind "
        "provenance. Removing one is a deliberate reduction in what this battery proves and "
        "must be argued in the manifest, not achieved by deleting a line."
    )
