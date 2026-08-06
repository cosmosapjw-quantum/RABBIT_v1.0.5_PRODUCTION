"""Regression fixtures for .agent-harness/scripts/check_ssot_consistency.py.

Each test below pins one defect that an adversarial panel reproduced against an
earlier version of the checker, or one behaviour that already worked and must
not be lost. This project has twice had a fix round introduce new defects, so
both directions are pinned: the attacks must fail, and the four rules that
already worked -- registry-vs-prose divergence, the frozen-value rule, the
per-file self-contradiction rule, and the correct exclusion of genuinely
superseded overlays -- must keep working.

The F-SSOT-10 section pins the newest of those defects: each fact declared the
command that measured it, nothing ever ran that command, and the declared
hook-fixture count therefore drifted two behind the file while the checker
reported ok. Those fixtures require that the measurement is run, that it is run
declaratively rather than through a shell, that a measurement which cannot be
run is an error rather than a skip, and that it is never applied to a frozen or
prior value.

Every test builds a complete miniature SSOT corpus in a temporary git
repository and runs the installed checker against it with ``cwd`` set there, so
the real repository is never read or written.

NOTE FOR EDITORS: this file lives under an artifact-scan root, so a literal
decision id written here would be inventoried as a landed decision of the REAL
repository. Fixture ids are built by ``decision_id()`` at run time and must
never be inlined as literals.
"""

from __future__ import annotations

import json
import subprocess
import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CHECKER = REPO / ".agent-harness" / "scripts" / "check_ssot_consistency.py"
sys.path.insert(0, str(REPO / ".agent-harness" / "scripts"))

# The round-11 refusal (D-070 Part B15). A live hard segment that names a
# registry id and also carries a bare status token is rejected without being
# read. The scope is the segment, not the sentence: see the checker's SENTENCE
# tier note for the abbreviation fixture that killed the sentence-scoped one. Tests
# match on this fragment rather than the whole message so the wording can be
# improved without silently loosening what is being asserted.
REFUSED = "appears beside the bare status"


def decision_id(number: int) -> str:
    """Build a fixture decision id without writing one as a literal."""
    return "D-%03d" % number


# An era-floor-exempt id, a strict-keying id, and one past the old three-digit
# ceiling that began with zero.
EARLY = decision_id(30)
STRICT = decision_id(50)
PAST_NINETY_NINE = decision_id(100)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def gate_registry(alpha: str = "pass", beta: str = "fail") -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "gates": [
                # Both carry a legacy basis and no package: the fixture corpus's
                # statuses predate the evidence-package requirement, exactly as
                # the real registry's eight do. The rule is on MOVEMENT.
                {"gate_id": "G-ALPHA", "status": alpha,
                 "status_package": None, "status_basis_legacy": "fixture standing status"},
                {"gate_id": "G-BETA", "status": beta,
                 "status_package": None, "status_basis_legacy": "fixture standing status"},
            ],
        },
        indent=2,
    )


def claim_registry() -> str:
    rows = [
        {"claim_id": "C-LIVE", "status": "VALIDATED"},
        {"claim_id": "C-SEALED", "status": "DEPRECATED"},
    ]
    return "".join(json.dumps(row) + "\n" for row in rows)


# The fixture corpus mirrors the real one: the measured fact counts `def test_`
# definitions in a hook fixture file that the fixture repository actually
# contains, so a measurement is genuinely run on every test below.
HOOK_FIXTURE_FILE = ".agent-harness/tests/test_hooks.py"
BASELINE_HOOK_COUNT = 39
HOOK_MEASUREMENT: dict[str, object] = {
    "kind": "count_lines_matching",
    "path": HOOK_FIXTURE_FILE,
    "pattern": "^def test_",
}
MANIFEST_FILE = ".agent-harness/context/LEGACY_RESULTS_MANIFEST.json"

_DEFAULT = object()


def hook_fixture_source(count: int) -> str:
    """A stand-in hook fixture file holding exactly `count` definitions.

    NOTE FOR EDITORS: this lands under an artifact-scan root, so nothing here
    may contain a literal decision id.
    """
    lines = ['"""Stand-in for the measured hook fixture file."""', ""]
    for index in range(count):
        lines.append("def test_fixture_%03d() -> None:" % index)
        lines.append("    assert True")
        lines.append("")
    return "\n".join(lines)


def manifest_source(entries: int, declared: int | None = None) -> str:
    """A manifest whose self-reported count may disagree with its array."""
    return json.dumps(
        {
            "entry_count": entries if declared is None else declared,
            "entries": [{"path": "results/r%03d.json" % index} for index in range(entries)],
        },
        indent=2,
    )


def manifest_fact(value: int, json_path: list[str] | None = None) -> dict[str, object]:
    return {
        "fact_id": "manifest_entry_count",
        # F-DESC-MEASURE-DECOUPLE (round 9): prose must name the measured file.
        "description": f"Result artifacts pinned by digest in {MANIFEST_FILE}.",
        "measurement": {
            "kind": "json_array_length",
            "path": MANIFEST_FILE,
            "json_path": json_path if json_path is not None else ["entries"],
        },
        "value": value,
        "as_of_commit": "a1cdd8a",
        "assertions": [
            {
                "file": MANIFEST_FILE,
                "line_contains": "entry_count",
                "value_regex": r"\"entry_count\": (\d+)",
                "role": "current",
            }
        ],
    }


def facts_document(
    *,
    drop_prior_commit: bool = False,
    extra_assertions: list[dict[str, object]] | None = None,
    exemptions: list[dict[str, object]] | None = None,
    exempt_claims: list[dict[str, str]] | None = None,
    measurement: object = _DEFAULT,
    drop_measurement: bool = False,
    value: int = BASELINE_HOOK_COUNT,
    extra_facts: list[dict[str, object]] | None = None,
) -> str:
    prior_35: dict[str, object] = {"value": 35, "as_of_commit": "ed7bc49"}
    if drop_prior_commit:
        del prior_35["as_of_commit"]
    fact: dict[str, object] = {
        "fact_id": "hook_fixture_count",
        # F-DESC-MEASURE-DECOUPLE (round 9): the description must name the file
        # the measurement actually reads, so prose and measurement cannot
        # describe different things.
        "description": (
            "Focused hook-lifecycle fixtures in "
            f"{HOOK_MEASUREMENT['path']}."
        ),
        "measurement": dict(HOOK_MEASUREMENT) if measurement is _DEFAULT else measurement,
        "value": value,
        "as_of_commit": "07e3507",
        "prior": [prior_35, {"value": 12, "as_of_commit": "b28ea0b"}],
        "assertions": [
            {
                "file": ".agent-harness/context/SHARED_CONTEXT.md",
                "line_contains": "Q-HOOK-01 remediation",
                "value_regex": r"(\d+) hook tests",
                "role": "current",
            },
            {
                "file": ".agent-harness/context/FROZEN_DECISIONS.md",
                "line_contains": "| %s |" % EARLY,
                "value_regex": r"(\d+) hook fixtures",
                "role": "frozen",
                "pinned_to_commit": "ed7bc49",
            },
            {
                "file": "docs/harness/VALIDATION_LEDGER.md",
                "line_contains": "seal back-reference",
                "value_regex": r"(\d+) at the seal",
                "role": "historical",
            },
        ]
        + (extra_assertions or []),
    }
    if drop_measurement:
        del fact["measurement"]
    # `coverage_policy` and `assertion_exemptions` were removed from the document
    # schema at D-073 commit 5: the generated board states every registry entry,
    # so there is nothing to exempt from coverage and prose can no longer assert
    # a status for an exemption to excuse. The parameters are kept and ignored so
    # the many call sites that pass them do not all have to change in the same
    # commit as the deletion; they go with the fixtures that use them.
    document = {
        "schema_version": 1,
        "purpose": "fixture",
        "facts": [fact] + (extra_facts or []),
    }
    return json.dumps(document, indent=2)


PROJECT_STATE = """# Project State

## Current status

The programme narrative carries no gate id.

## 2026-07-29 {early} overlay (controlling)

- The milestone narrative says what the work is, and states no status: the
  generated board below is the only place a status appears.

{board}

## 2026-07-28 {early} predecessor overlay (superseded by the overlay above)

- `G-BETA` was PASS when this was written.

## 2026-07-27 older dated overlay

- `G-ALPHA` was FAIL then.

## Standing boundary

This undated standing section is live and is scanned.
"""

NEXT_SESSION_PROMPT = """# Next Session Prompt

## 2026-07-29 {strict} controlling overlay

Nothing has moved.

## 2026-07-26 {early} controlling overlay

- `G-ALPHA` was FAIL under this older overlay.

## Objective and authority boundary

This undated standing section is live and is scanned.
"""

SHARED_CONTEXT = """# Shared Context

## Project objective

- Current milestone: the board is unchanged.

## Frozen narrative

At an earlier decision `G-ALPHA` was FAIL.

## Known disputes and open questions

| Question ID | Question | Owner |
|---|---|---|
| Q-HOOK-01 remediation | 39 hook tests pass. | owner |
"""

DECISION_LOG = """# Decision Log

| Date | Decision | Consequences |
|---|---|---|
| 2026-07-29 | Record {strict}: the strict-keying row. | none |
| 2026-07-28 | An early row that mentions {early} in prose. | none |
"""

VALIDATION_LEDGER = """# Validation Ledger

| Date | Command/check | Result | Notes |
|---|---|---|---|
| 2026-07-29 | the current row | recorded | none |
| 2026-07-28 | seal back-reference: 35 at the seal `ed7bc49` | recorded | none |
"""

FROZEN_DECISIONS = """# Frozen Decisions

| Decision ID | Decision | Scope |
|---|---|---|
| {early} | Sealed with 35 hook fixtures at its seal. | harness |
| {strict} | A strictly keyed decision. | harness |
"""


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A miniature SSOT corpus in its own git repository."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True)
    names = {"early": EARLY, "strict": STRICT, "board": BOARD_PLACEHOLDER}
    write(tmp_path / ".agent-harness/context/GATE_REGISTRY.json", gate_registry())
    write(tmp_path / ".agent-harness/context/CLAIM_REGISTRY.jsonl", claim_registry())
    write(tmp_path / ".agent-harness/context/SSOT_FACTS.json", facts_document())
    write(tmp_path / ".agent-harness/context/FROZEN_DECISIONS.md", FROZEN_DECISIONS.format(**names))
    write(tmp_path / ".agent-harness/context/SHARED_CONTEXT.md", SHARED_CONTEXT)
    write(tmp_path / "docs/harness/PROJECT_STATE.md", PROJECT_STATE.format(**names))
    write(tmp_path / "docs/harness/NEXT_SESSION_PROMPT.md", NEXT_SESSION_PROMPT.format(**names))
    write(tmp_path / "docs/harness/DECISION_LOG.md", DECISION_LOG.format(**names))
    write(tmp_path / "docs/harness/VALIDATION_LEDGER.md", VALIDATION_LEDGER)
    write(tmp_path / HOOK_FIXTURE_FILE, hook_fixture_source(BASELINE_HOOK_COUNT))
    for directory in (
        "docs/audit",
        "scripts/audit",
        ".agent-harness/scripts",
        ".agent-harness/tests",
        ".agent-harness/runs",
        ".codex/hooks",
        "tests",
    ):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    # The generated board is the status authority (D-073), so a corpus without
    # one has no status surface at all and the checker says so. The fixture
    # therefore CARRIES a board rather than the check being weakened to tolerate
    # its absence -- the same call made when the validator fixture had to carry
    # its sibling checkers.
    subprocess.run(
        [sys.executable, str(REPO / ".agent-harness/scripts/build_status_board.py")],
        cwd=tmp_path, check=True, capture_output=True,
    )
    # ...and stage it. The fixture corpus is a git repository, so the board is
    # subject to the same retention rule as the real one: an untracked authority
    # is one a fresh checkout would not receive.
    subprocess.run(
        ["git", "add", "-f", ".agent-harness/generated/STATUS_BOARD.md"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    # ...and transclude it into the declared host, inside the controlling
    # overlay. `PROJECT_STATE.md` is a BOARD_HOST, so a corpus without the block
    # is a corpus whose controlling section states no status at all -- which the
    # checker is right to refuse. The block is rendered from THIS corpus's
    # synthetic registries, not the real ones.
    artifact = (tmp_path / ".agent-harness/generated/STATUS_BOARD.md").read_text(encoding="utf-8")
    block = artifact[artifact.index(BOARD_BEGIN) : artifact.index(BOARD_END) + len(BOARD_END)]
    state = tmp_path / "docs/harness/PROJECT_STATE.md"
    text = state.read_text(encoding="utf-8")
    assert BOARD_PLACEHOLDER in text, text
    write(state, text.replace(BOARD_PLACEHOLDER, block, 1))
    return tmp_path


def run_checker(repo: Path) -> tuple[int, dict[str, object]]:
    completed = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.stdout, completed.stderr
    return completed.returncode, json.loads(completed.stdout)


def errors_of(repo: Path) -> list[str]:
    code, payload = run_checker(repo)
    assert code == 1, payload
    assert payload["ok"] is False
    return [str(item) for item in payload["errors"]]


def assert_clean(repo: Path) -> dict[str, object]:
    code, payload = run_checker(repo)
    assert code == 0, payload
    return payload


def edit(repo: Path, rel: str, old: str, new: str) -> None:
    path = repo / rel
    text = path.read_text(encoding="utf-8")
    assert old in text, f"fixture drift: {old!r} not in {rel}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------


def test_baseline_fixture_is_clean(repo: Path) -> None:
    """The corpus is clean and the board accounts for every registry entry.

    `gates_covered`, `claims_covered`, `claims_coverage_exempt` and
    `status_assertions_checked` are GONE from the payload. They measured how much
    of the corpus the prose parser happened to find, and nothing is parsed now:
    the board renders every gate and every claim, so the only meaningful number
    is that the rows account for the registries exactly.
    """
    payload = assert_clean(repo)
    assert payload["gates"] == 2
    assert payload["claims"] == 2
    assert payload["board_rows"] == payload["gates"] + payload["claims"]



# --------------------------------------------------------------------------
# F-SSOT-01 -- coverage floor. Under-detection must fail, not pass.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# F-SSOT-02 -- live standing sections are inside the checked region.
# --------------------------------------------------------------------------


def test_f_ssot_02_undated_standing_section_is_scanned(repo: Path) -> None:
    """A status planted far below the controlling overlay must still be caught.

    The earlier region rule stopped at the first `## ` heading after the
    controlling marker, so every undated standing section below it -- the
    authority boundary, the accepted state, the stop condition -- was outside
    the checked region entirely.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "This undated standing section is live and is scanned.",
        "For the avoidance of doubt `G-BETA` is PASS`.",
    )
    messages = errors_of(repo)
    assert any("PROJECT_STATE.md" in x and "G-BETA" in x for x in errors_of(repo))


def test_f_ssot_02_standing_section_of_the_prompt_is_scanned(repo: Path) -> None:
    edit(
        repo,
        "docs/harness/NEXT_SESSION_PROMPT.md",
        "This undated standing section is live and is scanned.",
        "The gate `G-ALPHA` is FAIL.",
    )
    messages = errors_of(repo)
    assert any("NEXT_SESSION_PROMPT.md" in m and "G-ALPHA" in m for m in messages)


def test_f_ssot_02_preamble_is_scanned(repo: Path) -> None:
    edit(repo, "docs/harness/PROJECT_STATE.md", "# Project State\n",
         "# Project State\n\n`G-BETA` is PASS.\n")
    assert any("G-BETA" in m for m in errors_of(repo))


# --------------------------------------------------------------------------
# F-SSOT-03 -- no vacuous truth. A missing field never widens what passes.
# --------------------------------------------------------------------------


def test_f_ssot_03_prior_without_a_commit_is_a_hard_error(repo: Path) -> None:
    """`priors[value] = ""` made `priors[value] in line` always True.

    Deleting one JSON key switched off both the undated-superseded-value check
    and the self-contradiction check that depends on it.
    """
    write(repo / ".agent-harness/context/SSOT_FACTS.json", facts_document(drop_prior_commit=True))
    messages = errors_of(repo)
    assert any("prior entry" in m and "as_of_commit" in m for m in messages)


def test_f_ssot_03_undated_superseded_value_still_fails(repo: Path) -> None:
    """The rule the vacuous truth disabled must itself still work."""
    edit(
        repo,
        "docs/harness/VALIDATION_LEDGER.md",
        "35 at the seal `ed7bc49`",
        "35 at the seal",
    )
    messages = errors_of(repo)
    assert any("without citing its commit" in m for m in messages)


def test_f_ssot_03_unknown_key_in_the_facts_file_is_an_error(repo: Path) -> None:
    document = json.loads((repo / ".agent-harness/context/SSOT_FACTS.json").read_text())
    document["facts"][0]["as_of_comit"] = "07e3507"
    write(repo / ".agent-harness/context/SSOT_FACTS.json", json.dumps(document, indent=2))
    assert any("unknown key" in m for m in errors_of(repo))


def test_f_ssot_03_frozen_assertion_without_its_pin_is_an_error(repo: Path) -> None:
    document = json.loads((repo / ".agent-harness/context/SSOT_FACTS.json").read_text())
    for assertion in document["facts"][0]["assertions"]:
        assertion.pop("pinned_to_commit", None)
    write(repo / ".agent-harness/context/SSOT_FACTS.json", json.dumps(document, indent=2))
    assert any("no 'pinned_to_commit'" in m for m in errors_of(repo))


# --------------------------------------------------------------------------
# F-SSOT-04 -- extractor: negation, direction, and table boards.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# F-SSOT-07 -- artifacts are looked for where the code actually lives.
# --------------------------------------------------------------------------


def test_f_ssot_07_audit_document_filename_is_still_inventoried(repo: Path) -> None:
    write(repo / ("docs/audit/BD622_%s_report.md" % PAST_NINETY_NINE.replace("-", "")), "report\n")
    assert any("no row for %s" % PAST_NINETY_NINE in m for m in errors_of(repo))


def test_f_ssot_07_run_directory_is_still_inventoried(repo: Path) -> None:
    slug = PAST_NINETY_NINE.replace("-", "").lower()  # the run-dir convention: `d100`
    (repo / ".agent-harness/runs" / ("run-20260729-f10-%s-lane" % slug)).mkdir()
    assert any("no row for %s" % PAST_NINETY_NINE in m for m in errors_of(repo))


def test_f_ssot_07_missing_artifact_root_is_an_error_not_a_skip(repo: Path) -> None:
    (repo / ".codex/hooks").rmdir()
    assert any(".codex/hooks: missing" in m for m in errors_of(repo))


# --------------------------------------------------------------------------
# F-SSOT-08 -- newest-first is parsed, never assumed.
# --------------------------------------------------------------------------


def test_f_ssot_08_newer_overlay_appended_at_the_end_is_authoritative(repo: Path) -> None:
    """The earlier rule took the FIRST heading marked controlling.

    These files are append-only overlay records, so a newer overlay appended at
    the end left a stale section reading as authoritative and the new one
    unscanned.
    """
    path = repo / "docs/harness/PROJECT_STATE.md"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "\n## 2026-07-31 newest overlay (controlling)\n\n- `G-BETA` is PASS.\n",
        encoding="utf-8",
    )
    messages = errors_of(repo)
    assert any("G-BETA" in x for x in errors_of(repo))


def test_f_ssot_08_two_sections_claiming_the_newest_controlling_date_is_an_error(repo: Path) -> None:
    path = repo / "docs/harness/PROJECT_STATE.md"
    path.write_text(
        path.read_text(encoding="utf-8") + "\n## 2026-07-29 rival overlay (controlling)\n\nnothing\n",
        encoding="utf-8",
    )
    assert any("claim to be the newest controlling overlay" in m for m in errors_of(repo))


def test_f_ssot_08_undated_controlling_heading_is_an_error(repo: Path) -> None:
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "## 2026-07-29 %s overlay (controlling)" % EARLY,
        "## %s overlay (controlling)" % EARLY,
    )
    assert any("marked controlling but carries no parseable date" in m for m in errors_of(repo))


def test_f_ssot_08_missing_controlling_marker_is_an_error(repo: Path) -> None:
    for rel in ("docs/harness/PROJECT_STATE.md", "docs/harness/NEXT_SESSION_PROMPT.md"):
        path = repo / rel
        path.write_text(path.read_text(encoding="utf-8").replace("(controlling)", "").replace(
            "controlling overlay", "overlay"
        ), encoding="utf-8")
    messages = errors_of(repo)
    assert any("no '## ...controlling...' section heading" in m for m in messages)


def test_f_ssot_08_unparseable_heading_date_is_an_error(repo: Path) -> None:
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "## 2026-07-27 older dated overlay",
        "## 2026-13-45 older dated overlay",
    )
    assert any("unparseable date" in m for m in errors_of(repo))


def test_f_ssot_08_newer_ledger_row_below_the_top_row_is_an_error(repo: Path) -> None:
    edit(
        repo,
        "docs/harness/VALIDATION_LEDGER.md",
        "| 2026-07-28 | seal back-reference",
        "| 2026-07-30 | seal back-reference",
    )
    messages = errors_of(repo)
    assert any("sits below the top row" in m for m in messages)


# --------------------------------------------------------------------------
# F-SSOT-09 -- one decision-id pattern covering the whole 000-999 range.
# --------------------------------------------------------------------------


def test_f_ssot_09_decision_past_ninety_nine_is_inventoried(repo: Path) -> None:
    """Five hardcoded `D-0\\d\\d` patterns made ids past 099 invisible.

    The failure mode was the dangerous direction: the decision became
    unchecked rather than rejected.
    """
    write(repo / "scripts/audit/lane.py", '"""Implements %s."""\n' % PAST_NINETY_NINE)
    messages = errors_of(repo)
    assert any("no row for %s" % PAST_NINETY_NINE in m for m in messages)
    assert any("Record %s:" % PAST_NINETY_NINE in m for m in messages)


def test_f_ssot_09_decision_past_ninety_nine_with_both_rows_is_clean(repo: Path) -> None:
    write(repo / "scripts/audit/lane.py", '"""Implements %s."""\n' % PAST_NINETY_NINE)
    edit(
        repo,
        ".agent-harness/context/FROZEN_DECISIONS.md",
        "| %s | A strictly keyed decision. | harness |" % STRICT,
        "| %s | A strictly keyed decision. | harness |\n| %s | A later decision. | harness |"
        % (STRICT, PAST_NINETY_NINE),
    )
    edit(
        repo,
        "docs/harness/DECISION_LOG.md",
        "| 2026-07-29 | Record %s: the strict-keying row. | none |" % STRICT,
        "| 2026-07-29 | Record %s: a later row. | none |\n| 2026-07-29 | Record %s: the strict-keying row. | none |"
        % (PAST_NINETY_NINE, STRICT),
    )
    assert_clean(repo)


def test_f_ssot_09_strict_keying_error_names_the_required_form(repo: Path) -> None:
    write(repo / "scripts/audit/lane.py", '"""Implements %s."""\n' % PAST_NINETY_NINE)
    edit(
        repo,
        ".agent-harness/context/FROZEN_DECISIONS.md",
        "| %s | A strictly keyed decision. | harness |" % STRICT,
        "| %s | A strictly keyed decision. | harness |\n| %s | A later decision. | harness |"
        % (STRICT, PAST_NINETY_NINE),
    )
    messages = errors_of(repo)
    assert any(
        "Record %s:" % PAST_NINETY_NINE in m and "that exact form is required" in m for m in messages
    )


def test_f_ssot_09_mention_is_not_enough_above_the_strict_floor(repo: Path) -> None:
    edit(
        repo,
        "docs/harness/DECISION_LOG.md",
        "Record %s: the strict-keying row." % STRICT,
        "a row that only mentions %s in passing." % STRICT,
    )
    assert any("Record %s:" % STRICT in m for m in errors_of(repo))


# --------------------------------------------------------------------------
# Behaviour that already worked and must not regress.
# --------------------------------------------------------------------------

def test_no_regress_frozen_row_must_not_be_refreshed(repo: Path) -> None:
    """35 is pinned at ed7bc49 and stays 35 even though current is 39."""
    edit(
        repo,
        ".agent-harness/context/FROZEN_DECISIONS.md",
        "Sealed with 35 hook fixtures",
        "Sealed with 39 hook fixtures",
    )
    messages = errors_of(repo)
    assert any("A frozen row is not refreshed" in m for m in messages)


def test_no_regress_frozen_row_at_its_pinned_value_is_accepted(repo: Path) -> None:
    assert_clean(repo)


def test_no_regress_self_contradiction_in_one_file(repo: Path) -> None:
    """One file asserting two undated values for one fact cannot be resolved.

    Both citations sit in OLDER ledger rows, outside the live top-row region.
    A `historical` assertion may no longer sit in a live region at all, so this
    check is exercised where superseded numbers actually belong.
    """
    write(
        repo / ".agent-harness/context/SSOT_FACTS.json",
        facts_document(
            extra_assertions=[
                {
                    "file": "docs/harness/VALIDATION_LEDGER.md",
                    "line_contains": "first back-reference",
                    "value_regex": r"(\d+) hook fixtures",
                    "role": "historical",
                },
                {
                    "file": "docs/harness/VALIDATION_LEDGER.md",
                    "line_contains": "second back-reference",
                    "value_regex": r"(\d+) hook fixtures",
                    "role": "historical",
                },
            ]
        ),
    )
    path = repo / "docs/harness/VALIDATION_LEDGER.md"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "| 2026-07-27 | first back-reference: 35 hook fixtures | recorded | none |\n"
        + "| 2026-07-26 | second back-reference: 12 hook fixtures | recorded | none |\n",
        encoding="utf-8",
    )
    messages = errors_of(repo)
    assert any("conflicting undated values" in m for m in messages), messages


def test_no_regress_stale_fact_declaration_is_an_error(repo: Path) -> None:
    edit(repo, ".agent-harness/context/SHARED_CONTEXT.md", "39 hook tests", "thirty-nine hook tests")
    assert any("no longer matches; the facts file is stale" in m for m in errors_of(repo))


def test_no_regress_superseded_overlay_may_contradict_the_registry(repo: Path) -> None:
    """A section marked superseded, and an older dated overlay, are history."""
    assert_clean(repo)
    text = (repo / "docs/harness/PROJECT_STATE.md").read_text(encoding="utf-8")
    assert "`G-BETA` was PASS when this was written." in text
    assert "`G-ALPHA` was FAIL then." in text


def test_no_regress_older_controlling_overlay_of_the_prompt_is_excluded(repo: Path) -> None:
    """Seven older sections of the real prompt are still headed 'controlling'.

    They are excluded by parsed date, not by position, and must stay excluded.
    """
    assert_clean(repo)
    text = (repo / "docs/harness/NEXT_SESSION_PROMPT.md").read_text(encoding="utf-8")
    assert "`G-ALPHA` was FAIL under this older overlay." in text


def test_no_regress_shared_context_frozen_narrative_is_not_scanned(repo: Path) -> None:
    assert_clean(repo)
    text = (repo / ".agent-harness/context/SHARED_CONTEXT.md").read_text(encoding="utf-8")
    assert "At an earlier decision `G-ALPHA` was FAIL." in text


def test_no_regress_forward_reference_is_reported_not_failed(repo: Path) -> None:
    edit(
        repo,
        "docs/harness/DECISION_LOG.md",
        "| 2026-07-29 | Record %s: the strict-keying row. | none |" % STRICT,
        "| 2026-07-29 | Record %s: the strict-keying row. Later %s is planned. | none |"
        % (STRICT, PAST_NINETY_NINE),
    )
    payload = assert_clean(repo)
    assert payload["decisions"]["forward_references"] == [PAST_NINETY_NINE]  # type: ignore[index]


def test_no_regress_unreadable_input_is_an_error(repo: Path) -> None:
    (repo / "docs/harness/PROJECT_STATE.md").unlink()
    assert any("PROJECT_STATE.md: unreadable" in m for m in errors_of(repo))


# --------------------------------------------------------------------------
# Declared assertion exemptions: narrow, verified, and self-expiring.
# --------------------------------------------------------------------------


EXEMPTION = {
    "file": "docs/harness/PROJECT_STATE.md",
    "line_contains": "the grant left `G-BETA=PASS`",
    "id": "G-BETA",
    "asserted_status": "PASS",
    "superseded_by_line_contains": "the later review restored it",
    "reason": "fixture narrative followed by its own correction",
}


def _plant_narrative(
    repo: Path,
    correction: str = "- Then the later review restored it: `G-BETA=FAIL`.",
) -> None:
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "This undated standing section is live and is scanned.",
        "- At the time the grant left `G-BETA=PASS`.\n%s" % correction,
    )

# --------------------------------------------------------------------------
# F-SSOT-10 -- declared facts are MEASURED, not asserted.
#
# `measurement` named the exact command that produced each number and was never
# run: it appeared once in the checker, as a schema key. So the declared
# hook-fixture count sat two behind the file while the checker printed
# `{"ok": true}` -- the F-D065-05 shape (a stated fact diverging from reality
# with no machine check) recurring inside the tool built to close F-D065-05.
# The measurement is now executed declaratively, never through a shell.
# --------------------------------------------------------------------------


def facts_and_measurement(repo: Path, **kwargs: object) -> None:
    write(repo / ".agent-harness/context/SSOT_FACTS.json", facts_document(**kwargs))  # type: ignore[arg-type]


def test_f_ssot_10_declared_value_that_drifted_from_the_repository_is_caught(repo: Path) -> None:
    """The live defect: two fixtures were added and the declaration was not."""
    write(repo / HOOK_FIXTURE_FILE, hook_fixture_source(BASELINE_HOOK_COUNT + 2))
    messages = errors_of(repo)
    assert any(
        "hook_fixture_count declares value 39" in m
        and "measures 41" in m
        and "count_lines_matching" in m
        and HOOK_FIXTURE_FILE in m
        for m in messages
    ), messages


def test_f_ssot_10_drift_is_caught_when_the_repository_shrinks(repo: Path) -> None:
    write(repo / HOOK_FIXTURE_FILE, hook_fixture_source(BASELINE_HOOK_COUNT - 1))
    assert any("declares value 39" in m and "measures 38" in m for m in errors_of(repo))

def test_f_ssot_10_measurement_is_required_on_every_fact(repo: Path) -> None:
    """No exemption key, in the JSON or in the code: an unmeasured fact fails."""
    facts_and_measurement(repo, drop_measurement=True)
    assert any("missing required key 'measurement'" in m for m in errors_of(repo))


def test_f_ssot_10_a_shell_command_string_is_refused(repo: Path) -> None:
    """The old prose form must not become an execution path."""
    facts_and_measurement(repo, measurement="grep -c '^def test_' %s" % HOOK_FIXTURE_FILE)
    messages = errors_of(repo)
    assert any("is a command string" in m and "never hands anything" in m for m in messages), messages


def test_f_ssot_10_unknown_measurement_kind_is_an_error(repo: Path) -> None:
    facts_and_measurement(repo, measurement={"kind": "run_shell", "path": "x", "pattern": "y"})
    messages = errors_of(repo)
    assert any("has kind 'run_shell'" in m and "never a skip" in m for m in messages), messages


def test_f_ssot_10_unknown_measurement_key_is_an_error(repo: Path) -> None:
    facts_and_measurement(repo, measurement=dict(HOOK_MEASUREMENT, shell=True))
    assert any("measurement has unknown key(s) shell" in m for m in errors_of(repo))


def test_f_ssot_10_missing_measurement_key_is_an_error(repo: Path) -> None:
    spec = {key: value for key, value in HOOK_MEASUREMENT.items() if key != "pattern"}
    facts_and_measurement(repo, measurement=spec)
    assert any("measurement is missing required key 'pattern'" in m for m in errors_of(repo))


def test_f_ssot_10_unreadable_measurement_target_is_an_error_not_a_skip(repo: Path) -> None:
    """Fail closed: a measurement that cannot be run leaves the fact asserted."""
    (repo / HOOK_FIXTURE_FILE).unlink()
    messages = errors_of(repo)
    assert any(
        "cannot be run" in m and "unreadable" in m and "never a skip" in m for m in messages
    ), messages


def test_f_ssot_10_measurement_path_escaping_the_repository_is_refused(repo: Path) -> None:
    facts_and_measurement(repo, measurement=dict(HOOK_MEASUREMENT, path="../outside.py"))
    assert any("escapes the repository" in m for m in errors_of(repo))


def test_f_ssot_10_absolute_measurement_path_is_refused(repo: Path) -> None:
    facts_and_measurement(repo, measurement=dict(HOOK_MEASUREMENT, path="/etc/hostname"))
    assert any("absolute or escapes the repository" in m for m in errors_of(repo))


def test_f_ssot_10_invalid_measurement_pattern_is_an_error(repo: Path) -> None:
    facts_and_measurement(repo, measurement=dict(HOOK_MEASUREMENT, pattern="[unclosed"))
    assert any("invalid 'pattern'" in m and "cannot be run" in m for m in errors_of(repo))


# ---- what measurement means for a frozen fact: nothing, deliberately -------


def test_f_ssot_10_frozen_row_is_not_measured_against_current_reality(repo: Path) -> None:
    """The frozen row stays 35 while the measured reality moves to 40."""
    write(repo / HOOK_FIXTURE_FILE, hook_fixture_source(40))
    facts_and_measurement(repo, value=40)
    edit(repo, ".agent-harness/context/SHARED_CONTEXT.md", "39 hook tests", "40 hook tests")
    payload = assert_clean(repo)
    assert payload["fact_measurements"] == {"hook_fixture_count": 40}
    frozen = (repo / ".agent-harness/context/FROZEN_DECISIONS.md").read_text(encoding="utf-8")
    assert "Sealed with 35 hook fixtures" in frozen


def test_f_ssot_10_measurement_does_not_legitimise_a_refreshed_frozen_row(repo: Path) -> None:
    """A frozen row edited to today's measured value is still an error."""
    edit(
        repo,
        ".agent-harness/context/FROZEN_DECISIONS.md",
        "Sealed with 35 hook fixtures",
        "Sealed with %d hook fixtures" % BASELINE_HOOK_COUNT,
    )
    assert any("A frozen row is not refreshed" in m for m in errors_of(repo))


def test_f_ssot_10_prior_values_are_not_measured_against_the_working_tree(repo: Path) -> None:
    """Priors 35 and 12 differ from the measured 39 and that is not a defect."""
    payload = assert_clean(repo)
    assert payload["fact_measurements"]["hook_fixture_count"] == BASELINE_HOOK_COUNT  # type: ignore[index]
    ledger = (repo / "docs/harness/VALIDATION_LEDGER.md").read_text(encoding="utf-8")
    assert "35 at the seal `ed7bc49`" in ledger


# ---- the JSON kind ---------------------------------------------------------


def test_f_ssot_10_json_array_length_measurement_is_clean_when_it_agrees(repo: Path) -> None:
    write(repo / MANIFEST_FILE, manifest_source(3))
    facts_and_measurement(repo, extra_facts=[manifest_fact(3)])
    payload = assert_clean(repo)
    assert payload["fact_measurements"]["manifest_entry_count"] == 3  # type: ignore[index]


def test_f_ssot_10_json_array_length_measures_the_array_not_its_self_report(repo: Path) -> None:
    """A manifest that miscounts itself is caught even though every surface agrees.

    The old measurement read `entry_count` back out of the same file that
    declares it, so the number proved only that the file equalled itself.
    """
    write(repo / MANIFEST_FILE, manifest_source(3, declared=2))
    facts_and_measurement(repo, extra_facts=[manifest_fact(2)])
    messages = errors_of(repo)
    assert any(
        "manifest_entry_count declares value 2" in m
        and "measures 3" in m
        and "json_array_length" in m
        for m in messages
    ), messages


def test_f_ssot_10_json_path_that_does_not_exist_is_an_error(repo: Path) -> None:
    write(repo / MANIFEST_FILE, manifest_source(3))
    facts_and_measurement(repo, extra_facts=[manifest_fact(3, json_path=["records"])])
    assert any("cannot be run" in m and "has no records" in m for m in errors_of(repo))


def test_f_ssot_10_json_path_pointing_at_a_non_array_is_an_error(repo: Path) -> None:
    write(repo / MANIFEST_FILE, manifest_source(3))
    facts_and_measurement(repo, extra_facts=[manifest_fact(3, json_path=["entry_count"])])
    assert any("is int, not an array" in m for m in errors_of(repo))


def test_f_ssot_10_malformed_json_measurement_target_is_an_error(repo: Path) -> None:
    write(repo / MANIFEST_FILE, "{not json")
    facts_and_measurement(repo, extra_facts=[manifest_fact(3)])
    assert any("is not valid JSON" in m for m in errors_of(repo))


def test_f_ssot_10_empty_json_path_is_an_error(repo: Path) -> None:
    write(repo / MANIFEST_FILE, manifest_source(3))
    facts_and_measurement(repo, extra_facts=[manifest_fact(3, json_path=[])])
    assert any("non-empty 'json_path'" in m for m in errors_of(repo))


# --------------------------------------------------------------------------
# Round 9 (D-070 Part B9). Six confirmed defects, one negative fixture each.
# Every one of these must die if its guard is reverted; that is the only thing
# that makes the guard verified rather than merely present.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Round 10 (D-070 Part B12). The panel found the B9 binder broken in the paths
# B9 never covered. One negative fixture per repair; each must die on revert.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# F-R11 -- the bypass family that retired the prose parser (D-070 Part B15).
#
# An external audit reported ONE of these (`is not only PASS`). Reproducing it
# locally found five more, in two mechanisms that no vocabulary distinguishes:
#
#   decoy shadowing   a TRUE token earlier in the segment took the id's single
#                     binding slot, and the false token after it was never read
#   affirmative "not" `not only/merely/just PASS` AFFIRMS PASS, and every
#                     denylist of ways to say "no" reads it as a denial
#
# Each string below exited 0 against the round-10 checker with the fixture
# registry saying G-BETA=FAIL. They are parametrised rather than written out
# as separate tests so that adding a seventh string is one line, not one
# function -- the previous rounds' habit of hand-writing each case is part of
# why three fixtures shipped without discriminating.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# F-R12 -- round 12 attacked round 11's own fix and found three more ways for a
# false gate status to be visible to a reader and invisible to this file.
#
# All three exploit the same seam: the checker's vocabulary of what a status
# LOOKS like is narrower than a reader's. Each is closed structurally rather
# than by widening a match, and each cost zero on the live corpus.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# D-073 -- the generated status board. Status stops being written by hand.
#
# These fixtures pin the properties that make generation an authority rather
# than a convenience: the artifact must equal a fresh render, it must be
# tracked, it may carry exactly one block, and every row must be present.
# --------------------------------------------------------------------------


def git_repo_for_board(repo: Path) -> None:
    """Give the fixture repo a git identity, so tracking checks can run."""
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)


def commit_board(repo: Path) -> None:
    import subprocess

    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "board"], cwd=repo, check=True, capture_output=True)


BOARD = ".agent-harness/generated/STATUS_BOARD.md"
from pathlib import Path as pathlib_Path  # noqa: E402
from check_ssot_consistency import manifest_digest as C_manifest_digest  # noqa: E402


def rebuild_board(repo: Path) -> None:
    """Re-render the board after a registry edit, as the generator would."""
    import subprocess

    subprocess.run([sys.executable, str(REPO / ".agent-harness/scripts/build_status_board.py")],
                   cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "-f", ".agent-harness/generated/STATUS_BOARD.md"],
                   cwd=repo, check=True, capture_output=True)
    artifact = (repo / ".agent-harness/generated/STATUS_BOARD.md").read_text(encoding="utf-8")
    block = artifact[artifact.index(BOARD_BEGIN): artifact.index(BOARD_END) + len(BOARD_END)]
    state = repo / "docs/harness/PROJECT_STATE.md"
    text = state.read_text(encoding="utf-8")
    write(state, text[:text.index(BOARD_BEGIN)] + block + text[text.index(BOARD_END) + len(BOARD_END):])


BOARD_PLACEHOLDER = "<!-- BOARD GOES HERE -->"
BOARD_BEGIN = "<!-- BEGIN GENERATED STATUS BOARD -->"
BOARD_END = "<!-- END GENERATED STATUS BOARD -->"


def build_board(repo: Path) -> None:
    """Render the board into the fixture repository, as the generator would."""
    import subprocess

    subprocess.run(
        [sys.executable, str(REPO / ".agent-harness/scripts/build_status_board.py")],
        cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(["git", "add", "-f", BOARD], cwd=repo, check=True, capture_output=True)


def test_d073_a_freshly_built_board_is_clean(repo: Path) -> None:
    """The control. Everything below must fail for its own reason, not because
    the board is broken in general."""
    git_repo_for_board(repo)
    build_board(repo)
    commit_board(repo)
    assert_clean(repo)


def test_d073_a_hand_edited_board_is_refused(repo: Path) -> None:
    """The whole point: a status edited into the artifact does not survive.

    This is the case the prose parser could never close, because a hand-written
    board was the authority. Now the registries are, and the artifact is checked
    against what they render.
    """
    git_repo_for_board(repo)
    build_board(repo)
    commit_board(repo)
    path = repo / BOARD
    text = path.read_text(encoding="utf-8")
    assert "| `G-BETA` | FAIL |" in text, text
    write(path, text.replace("| `G-BETA` | FAIL |", "| `G-BETA` | PASS |", 1))
    assert any("does not match what the registries render" in m for m in errors_of(repo)), errors_of(repo)


def test_d073_a_second_block_is_refused(repo: Path) -> None:
    """Two blocks is how a false board is smuggled in beside a true one."""
    git_repo_for_board(repo)
    build_board(repo)
    commit_board(repo)
    path = repo / BOARD
    text = path.read_text(encoding="utf-8")
    write(path, text + "\n" + text[text.index("<!-- BEGIN"):])
    assert any("expected one of each" in m for m in errors_of(repo)), errors_of(repo)


def test_d073_an_untracked_board_is_refused(repo: Path) -> None:
    """A board a fresh checkout would not receive is not an authority.

    The discriminator is that the CONTENT is correct: only the tracking check
    can fail this case, so it cannot pass for the wrong reason.
    """
    import subprocess

    git_repo_for_board(repo)
    build_board(repo)
    commit_board(repo)
    # Untrack it and commit that removal WITHOUT `git add -A`, which would
    # simply re-add the file and make this fixture prove nothing.
    subprocess.run(["git", "rm", "--cached", "-q", BOARD], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "untrack the board"], cwd=repo,
                   check=True, capture_output=True)
    errors = errors_of(repo)
    assert any("not tracked by git" in m for m in errors), errors
    assert not any("does not match what the registries render" in m for m in errors), (
        "content must be correct, or this proves nothing", errors)


def test_d073_a_dropped_row_is_refused(repo: Path) -> None:
    """The mechanical replacement for the coverage floor.

    A renderer that silently drops a row would previously have been caught, if
    at all, by a human noticing a missing line. Byte equality catches it without
    anyone having to look.
    """
    git_repo_for_board(repo)
    build_board(repo)
    commit_board(repo)
    path = repo / BOARD
    lines = [l for l in path.read_text(encoding="utf-8").split("\n") if "`G-ALPHA`" not in l]
    write(path, "\n".join(lines))
    assert any("does not match what the registries render" in m for m in errors_of(repo)), errors_of(repo)


def test_d073_coverage_is_an_identity_not_a_floor(repo: Path) -> None:
    """What replaced the coverage floor, and why nine fixtures were retired.

    The floor existed because live status was hand-written and a gate could go
    unmentioned. Nine fixtures pinned it: that a deleted mention was caught, that
    a misread denial did not satisfy it, that a past-tense copula did not, and so
    on. Every one of those is now unreachable, because the board renders EVERY
    registry entry and a gate cannot be stated nowhere.

    They are retired rather than left passing vacuously, because a fixture that
    can no longer fail is a fixture that no longer says anything -- and this
    project has already found six guards that shipped in exactly that condition.
    What protects the property now is the row-count identity below, plus
    `test_d073_a_dropped_row_is_refused` for the case where the renderer breaks.
    """
    payload = assert_clean(repo)
    assert payload["board_rows"] == payload["gates"] + payload["claims"], payload


# --------------------------------------------------------------------------
# D-073 commits 7-8 -- evidence packages and the gate-movement ratchet.
#
# The generated board made a status impossible to MISSTATE. It did not make one
# impossible to CHANGE without evidence. A package is the closed record a change
# has to arrive with; the ratchet is what refuses a change without one.
# --------------------------------------------------------------------------


def evidence_package(repo: Path, **over: object) -> pathlib_Path:
    """A well-formed package for G-BETA, with every relation satisfied."""
    import subprocess

    inputs = [{"path": HOOK_FIXTURE_FILE,
               "sha256": hashlib.sha256((repo / HOOK_FIXTURE_FILE).read_bytes()).hexdigest()}]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()
    doc = {
        "schema_version": 1, "artifact_class": "EVIDENCE_PACKAGE", "package_id": "P-BETA",
        "input_commit": head, "input_files": inputs,
        "input_tree_sha256": C_manifest_digest(inputs),
        "contract_path": "docs/audit/contract.md", "contract_sha256": "0" * 64,
        "contract_frozen_at_commit": head,
        "commands": [{"argv": ["true"], "cwd": ".", "exit_code": 0,
                      "started_utc": "2026-08-04T00:00:00Z", "finished_utc": "2026-08-04T00:00:01Z"}],
        "results": [], "gate_id": "G-BETA", "implies_status": "FAIL",
        "implication": "the measurement did not clear the gate",
        "process_class": "self_measured", "reviewers": [],
        "limitations": ["one host, one trial"], "sealed_commit": "PENDING_COMMIT",
    }
    doc.update(over)
    path = repo / ".agent-harness/evidence" / f"{doc['package_id']}.json"
    write(path, json.dumps(doc, indent=1))
    subprocess.run(["git", "add", "-f", str(path.relative_to(repo))], cwd=repo,
                   check=True, capture_output=True)
    return path


def test_d073_a_gate_may_not_move_without_a_package(repo: Path) -> None:
    """The rule the packages exist for: a status change needs a record."""
    git_repo_for_board(repo)
    commit_board(repo)
    write(repo / ".agent-harness/context/GATE_REGISTRY.json", gate_registry("pass", "pass"))
    rebuild_board(repo)
    errors = errors_of(repo)
    assert any("moved from fail to pass with no status_package" in m for m in errors), errors


def test_d073_a_gate_that_does_not_move_needs_no_package(repo: Path) -> None:
    """The control, and the bootstrap: eight standing statuses predate the rule.

    Back-filling packages for statuses set weeks ago would be fabricating
    records, which is the class this chain keeps catching. The rule is on
    MOVEMENT; standing state carries an explicit legacy basis instead.
    """
    git_repo_for_board(repo)
    commit_board(repo)
    assert_clean(repo)


def test_d073_evidence_backing_is_a_ratchet(repo: Path) -> None:
    """Once a gate has a package, it may never go back to having none."""
    import subprocess

    git_repo_for_board(repo)
    commit_board(repo)          # a package names a commit, so one must exist
    evidence_package(repo)
    registry = json.loads((repo / ".agent-harness/context/GATE_REGISTRY.json").read_text())
    for g in registry["gates"]:
        if g["gate_id"] == "G-BETA":
            g["status_package"] = "P-BETA"
    write(repo / ".agent-harness/context/GATE_REGISTRY.json", json.dumps(registry, indent=2))
    rebuild_board(repo)
    commit_board(repo)
    for g in registry["gates"]:
        if g["gate_id"] == "G-BETA":
            g["status_package"] = None
    write(repo / ".agent-harness/context/GATE_REGISTRY.json", json.dumps(registry, indent=2))
    rebuild_board(repo)
    assert any("does not come off" in m for m in errors_of(repo)), errors_of(repo)


def test_d073_a_contract_frozen_after_its_inputs_is_refused(repo: Path) -> None:
    """The anti-refitting rule, mechanised.

    D-069 required a contract frozen before any output byte, and that has rested
    on prose and a timestamp in a report ever since. A contract sealed after its
    own measurement cannot constrain it, and ancestry says so.
    """
    import subprocess

    git_repo_for_board(repo)
    commit_board(repo)
    first = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                           text=True, check=True).stdout.strip()
    write(repo / "later.txt", "a later commit\n")
    commit_board(repo)
    later = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                           text=True, check=True).stdout.strip()
    evidence_package(repo, input_commit=first, contract_frozen_at_commit=later)
    assert any("not frozen before the inputs" in m for m in errors_of(repo)), errors_of(repo)


def test_d073_a_package_with_no_limitations_is_refused(repo: Path) -> None:
    """A package with no stated limits is a claim, not evidence."""
    git_repo_for_board(repo)
    commit_board(repo)
    evidence_package(repo, limitations=[])
    assert any("limitations" in m for m in errors_of(repo)), errors_of(repo)


# --------------------------------------------------------------------------
# BD623 R3 -- the Basis column may not restructure the table it sits in.
# --------------------------------------------------------------------------


def test_bd623_r3_a_pipe_in_the_basis_cannot_add_columns_to_the_board() -> None:
    """Measured before the fix: a `|` in `status_basis_legacy` rendered a
    five-cell row in a three-column table, `build_status_board.py` exited 0, and
    the extra cells landed INSIDE the board span -- which
    `check_no_status_beside_an_id` excludes by construction. A status token
    beside a gate id was therefore exempt from the rule written to catch it.
    """
    from check_ssot_consistency import render_board

    body = render_board(
        {"G-ALPHA": "fail"},
        {},
        {"G-ALPHA": "benign prose | PASS | all obligations discharged"},
    )
    import re

    row = next(line for line in body.splitlines() if "G-ALPHA" in line)
    separators = re.findall(r"(?<!\\)\|", row)
    assert len(separators) == 4, row  # four separators bound exactly three cells
    assert "\\|" in row, row
    assert "all obligations discharged" in row, row


def test_bd623_r3_a_newline_in_the_basis_cannot_break_the_row() -> None:
    """The same argument for the other character that ends a table row."""
    from check_ssot_consistency import render_board

    body = render_board({"G-ALPHA": "fail"}, {}, {"G-ALPHA": "first line\nsecond line"})
    rows = [line for line in body.splitlines() if "G-ALPHA" in line]
    assert len(rows) == 1, rows
    assert "second line" in rows[0], rows[0]


def test_bd623_r3_an_ordinary_basis_is_unchanged() -> None:
    """The escape must be a no-op on every basis the registry holds today, or
    the board's bytes would move and every host copy would need rewriting."""
    from check_ssot_consistency import basis_cell

    for text in [
        "nine rows execute with rowwise invariants at module bytes 760a7c04",
        "D-065 upholds it only on the frozen matched family and module bytes",
        "",
    ]:
        assert basis_cell(text) == text
