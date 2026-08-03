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
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CHECKER = REPO / ".agent-harness" / "scripts" / "check_ssot_consistency.py"

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
                {"gate_id": "G-ALPHA", "status": alpha},
                {"gate_id": "G-BETA", "status": beta},
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
    document = {
        "schema_version": 1,
        "purpose": "fixture",
        "coverage_policy": {
            "note": "fixture",
            "claims": {
                "exempt": (
                    exempt_claims
                    if exempt_claims is not None
                    else [{"claim_id": "C-SEALED", "reason": "sealed at its decision"}]
                )
            },
        },
        "assertion_exemptions": exemptions or [],
        "facts": [fact] + (extra_facts or []),
    }
    return json.dumps(document, indent=2)


PROJECT_STATE = """# Project State

## Current status

The programme narrative carries no gate id.

## 2026-07-29 {early} overlay (controlling)

- `G-ALPHA=PASS` remains and `C-LIVE=VALIDATED`.
- `G-BETA=FAIL` remains.

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
    names = {"early": EARLY, "strict": STRICT}
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
    payload = assert_clean(repo)
    assert payload["gates_covered"] == 2
    assert payload["claims_covered"] == 1
    assert payload["claims_coverage_exempt"] == 1
    assert payload["status_assertions_checked"] > 0
    # The declared value was measured, not taken on trust.
    assert payload["fact_measurements"] == {"hook_fixture_count": BASELINE_HOOK_COUNT}


# --------------------------------------------------------------------------
# F-SSOT-01 -- coverage floor. Under-detection must fail, not pass.
# --------------------------------------------------------------------------


def test_f_ssot_01_flipping_gates_and_deleting_the_prose_is_caught(repo: Path) -> None:
    """The original attack: flip both gates and neutralise what mentioned them.

    The earlier checker walked only assertions it found in prose, so deleting
    the prose left nothing to walk and it reported ok with zero assertions
    checked.
    """
    write(repo / ".agent-harness/context/GATE_REGISTRY.json", gate_registry("pass", "pass"))
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-ALPHA=PASS` remains and `C-LIVE=VALIDATED`.\n- `G-BETA=FAIL` remains.",
        "- The board is unchanged.",
    )
    messages = errors_of(repo)
    assert any("G-ALPHA" in m and "asserted nowhere" in m for m in messages)
    assert any("G-BETA" in m and "asserted nowhere" in m for m in messages)


def test_f_ssot_01_zero_assertions_anywhere_is_an_error(repo: Path) -> None:
    for rel, old, new in (
        (
            "docs/harness/PROJECT_STATE.md",
            "- `G-ALPHA=PASS` remains and `C-LIVE=VALIDATED`.\n- `G-BETA=FAIL` remains.",
            "- The board is unchanged.",
        ),
    ):
        edit(repo, rel, old, new)
    messages = errors_of(repo)
    assert any("no status assertion was found in any live region" in m for m in messages)


def test_f_ssot_01_gate_with_no_coverage_is_named(repo: Path) -> None:
    edit(repo, "docs/harness/PROJECT_STATE.md", "- `G-BETA=FAIL` remains.", "- Nothing here.")
    messages = errors_of(repo)
    assert any("G-BETA=FAIL is asserted nowhere" in m for m in messages)
    assert not any("G-ALPHA" in m for m in messages)


def test_f_ssot_01_claim_not_on_the_exempt_list_requires_coverage(repo: Path) -> None:
    """Silence is never the permissive option: an unlisted claim is required."""
    write(repo / ".agent-harness/context/SSOT_FACTS.json", facts_document(exempt_claims=[]))
    messages = errors_of(repo)
    assert any("C-SEALED=DEPRECATED is asserted nowhere" in m for m in messages)


def test_f_ssot_01_coverage_policy_is_mandatory(repo: Path) -> None:
    document = json.loads((repo / ".agent-harness/context/SSOT_FACTS.json").read_text())
    del document["coverage_policy"]
    write(repo / ".agent-harness/context/SSOT_FACTS.json", json.dumps(document, indent=2))
    assert any("no 'coverage_policy'" in m for m in errors_of(repo))


def test_f_ssot_01_exemption_for_a_claim_that_does_not_exist_is_an_error(repo: Path) -> None:
    write(
        repo / ".agent-harness/context/SSOT_FACTS.json",
        facts_document(
            exempt_claims=[
                {"claim_id": "C-SEALED", "reason": "sealed"},
                {"claim_id": "C-GONE", "reason": "stale"},
            ]
        ),
    )
    assert any("exempts C-GONE" in m for m in errors_of(repo))


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
        "For the avoidance of doubt `G-BETA=PASS`.",
    )
    messages = errors_of(repo)
    assert any("PROJECT_STATE.md" in m and "G-BETA=PASS" in m and "registry says FAIL" in m for m in messages)


def test_f_ssot_02_standing_section_of_the_prompt_is_scanned(repo: Path) -> None:
    edit(
        repo,
        "docs/harness/NEXT_SESSION_PROMPT.md",
        "This undated standing section is live and is scanned.",
        "The gate `G-ALPHA=FAIL`.",
    )
    messages = errors_of(repo)
    assert any("NEXT_SESSION_PROMPT.md" in m and "G-ALPHA=FAIL" in m for m in messages)


def test_f_ssot_02_preamble_is_scanned(repo: Path) -> None:
    edit(repo, "docs/harness/PROJECT_STATE.md", "# Project State\n", "# Project State\n\n`G-BETA=PASS`.\n")
    assert any("G-BETA=PASS" in m for m in errors_of(repo))


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


def test_f_ssot_04_negation_is_refused_rather_than_interpreted(repo: Path) -> None:
    """"is no longer PASS" is no longer READ at all -- it is refused.

    The old rule was "a denial must not be recorded as asserting the status",
    and it needed a denylist of ways to say no. Round 11 withdrew the denylist:
    the sentence is rejected for putting a bare status token beside a gate id,
    without deciding what it means. The two things that must both hold are that
    the line is reported, and that it is NOT recorded as an assertion of PASS.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-BETA=FAIL` remains.",
        "- `G-BETA` is no longer PASS.\n- `G-BETA=FAIL` remains.",
    )
    messages = errors_of(repo)
    assert any("G-BETA" in m and REFUSED in m for m in messages), messages
    assert not any("G-BETA=PASS but the gate registry" in m for m in messages), messages


def test_f_ssot_04_negated_mention_does_not_satisfy_coverage(repo: Path) -> None:
    edit(repo, "docs/harness/PROJECT_STATE.md", "- `G-BETA=FAIL` remains.", "- `G-BETA` is no longer FAIL.")
    messages = errors_of(repo)
    assert any("G-BETA=FAIL is asserted nowhere" in m for m in messages)


def test_f_ssot_04_status_before_the_id_is_refused(repo: Path) -> None:
    """Backward order was a binding rule; now it is simply not a legal form."""
    edit(repo, "docs/harness/PROJECT_STATE.md", "- `G-BETA=FAIL` remains.", "- Still PASS for `G-BETA`.")
    messages = errors_of(repo)
    assert any("G-BETA" in m and REFUSED in m for m in messages), messages


def test_f_ssot_04_status_board_table_is_read(repo: Path) -> None:
    """A table that declares a Status header binds across cells; others do not."""
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-ALPHA=PASS` remains and `C-LIVE=VALIDATED`.\n- `G-BETA=FAIL` remains.",
        "| Gate | Status |\n|---|---|\n| `G-ALPHA` | PASS |\n| `G-BETA` | PASS |\n| `C-LIVE` | VALIDATED |\n",
    )
    messages = errors_of(repo)
    assert any("G-BETA=PASS" in m and "registry says FAIL" in m for m in messages)


def test_f_ssot_04_plain_table_row_does_not_bind_across_cells(repo: Path) -> None:
    """The pipe boundary is deliberate: a Result cell must not bind a Notes id."""
    edit(
        repo,
        "docs/harness/VALIDATION_LEDGER.md",
        "| 2026-07-29 | the current row | recorded | none |",
        "| 2026-07-29 | the current row | PASS | `G-BETA` was the subject |",
    )
    assert_clean(repo)


def test_f_ssot_04_conflicting_equidistant_tokens_are_reported_not_skipped(repo: Path) -> None:
    edit(repo, "docs/harness/PROJECT_STATE.md", "- `G-BETA=FAIL` remains.", "- FAIL `G-BETA` PASS\n- `G-BETA=FAIL` remains.")
    messages = errors_of(repo)
    assert any("G-BETA" in m and REFUSED in m for m in messages), messages


# --------------------------------------------------------------------------
# F-SSOT-07 -- artifacts are looked for where the code actually lives.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel",
    [
        ".agent-harness/scripts/lane_tool.py",
        ".agent-harness/tests/test_lane.py",
        ".codex/hooks/lane_hook.py",
        "tests/test_lane_surface.py",
        "scripts/audit/lane_probe.py",
    ],
)
def test_f_ssot_07_code_outside_the_audit_tree_is_inventoried(repo: Path, rel: str) -> None:
    """The admission-binding and cost-discipline lanes shipped code here.

    A scan limited to `scripts/audit` would have missed exactly the "code but
    no SSOT row" case the inventory check exists to catch.
    """
    write(repo / rel, '"""Implements %s."""\n' % PAST_NINETY_NINE)
    messages = errors_of(repo)
    assert any("no row for %s" % PAST_NINETY_NINE in m and rel in m for m in messages)


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
        + "\n## 2026-07-31 newest overlay (controlling)\n\n- `G-BETA=PASS`.\n",
        encoding="utf-8",
    )
    messages = errors_of(repo)
    assert any("G-BETA=PASS" in m and "registry says FAIL" in m for m in messages)


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


def test_no_regress_registry_versus_declaration_divergence_is_detected(repo: Path) -> None:
    """A false STRUCTURAL statement is still a contradiction, not a refusal."""
    edit(repo, "docs/harness/PROJECT_STATE.md", "- `G-BETA=FAIL` remains.", "- `G-BETA=PASS` remains.")
    messages = errors_of(repo)
    assert any("G-BETA=PASS but the gate registry says FAIL" in m for m in messages), messages


def test_no_regress_registry_versus_prose_divergence_is_refused(repo: Path) -> None:
    """The same lie in prose is refused instead. Either way it cannot survive."""
    edit(repo, "docs/harness/PROJECT_STATE.md", "- `G-BETA=FAIL` remains.", "- `G-BETA` remains PASS.")
    messages = errors_of(repo)
    assert any("G-BETA" in m and REFUSED in m for m in messages), messages


def test_no_regress_claim_divergence_is_detected(repo: Path) -> None:
    edit(repo, "docs/harness/PROJECT_STATE.md", "`C-LIVE=VALIDATED`", "`C-LIVE=PROPOSED`")
    messages = errors_of(repo)
    assert any("C-LIVE=PROPOSED but the claim registry says VALIDATED" in m for m in messages), messages


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


def test_exemption_excuses_narrative_that_the_section_corrects(repo: Path) -> None:
    _plant_narrative(repo)
    write(repo / ".agent-harness/context/SSOT_FACTS.json", facts_document(exemptions=[EXEMPTION]))
    assert_clean(repo)


def test_exemption_cannot_excuse_an_uncorrected_contradiction(repo: Path) -> None:
    _plant_narrative(repo, correction="- No correction follows.")
    write(repo / ".agent-harness/context/SSOT_FACTS.json", facts_document(exemptions=[EXEMPTION]))
    messages = errors_of(repo)
    assert any("no line after" in m for m in messages)


def test_exemption_whose_correction_asserts_the_wrong_status_is_an_error(repo: Path) -> None:
    _plant_narrative(repo, correction="- Then the later review restored it: `G-BETA=VALIDATED`.")
    write(repo / ".agent-harness/context/SSOT_FACTS.json", facts_document(exemptions=[EXEMPTION]))
    messages = errors_of(repo)
    assert any("does not assert G-BETA=FAIL" in m for m in messages)


def test_dead_exemption_is_reported(repo: Path) -> None:
    write(repo / ".agent-harness/context/SSOT_FACTS.json", facts_document(exemptions=[EXEMPTION]))
    messages = errors_of(repo)
    assert any("matched nothing on this run" in m for m in messages)


def test_exemption_naming_an_unknown_id_is_an_error(repo: Path) -> None:
    entry = dict(EXEMPTION, id="G-GONE")
    write(repo / ".agent-harness/context/SSOT_FACTS.json", facts_document(exemptions=[entry]))
    messages = errors_of(repo)
    assert any("names G-GONE, which has no row" in m for m in messages)


def test_exemption_missing_a_field_is_an_error(repo: Path) -> None:
    entry = {key: value for key, value in EXEMPTION.items() if key != "reason"}
    write(repo / ".agent-harness/context/SSOT_FACTS.json", facts_document(exemptions=[entry]))
    messages = errors_of(repo)
    assert any("missing required key 'reason'" in m for m in messages)


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


def test_f_ssot_10_prose_agreeing_with_a_wrong_declaration_is_not_enough(repo: Path) -> None:
    """Every surface agrees; only the repository disagrees.

    This is the state the checker used to call `ok: true`: it proved the
    surfaces agreed with the declaration and never that the declaration agreed
    with the repository.
    """
    facts_and_measurement(repo, value=BASELINE_HOOK_COUNT + 1)
    edit(repo, ".agent-harness/context/SHARED_CONTEXT.md", "39 hook tests", "40 hook tests")
    messages = errors_of(repo)
    assert messages == [
        message for message in messages if "measures 39 in the working tree" in message
    ], messages
    assert len(messages) == 1, messages


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


def test_f_r9_comma_adjacency_lie_binds_to_its_own_id(repo: Path) -> None:
    """The round-9 critical finding, as the attack was actually written.

    `ID1` is S1, `ID2` is S2 -- the separator before ID2 is three characters and
    the one after it is five, so distance-ranking gave ID2 its NEIGHBOUR's
    status and a plainly false line passed silently.

    The two statuses MUST differ, and the neighbour's must equal ID2's TRUE
    status. Otherwise mis-binding still lands on a value that contradicts the
    registry, the checker errors either way, and the fixture proves nothing --
    which is exactly how the first draft of this test, and the first manual
    reproduction of the finding, both failed to discriminate. Here both gates
    are really FAIL, so binding `G-BETA` to its neighbour's FAIL would agree
    with the registry and pass in silence.
    """
    write(repo / ".agent-harness/context/GATE_REGISTRY.json", gate_registry("fail", "fail"))
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-ALPHA=PASS` remains and `C-LIVE=VALIDATED`.",
        "- `C-LIVE=VALIDATED`.\n- `G-ALPHA` is FAIL, `G-BETA=PASS`.",
    )
    edit(repo, "docs/harness/PROJECT_STATE.md", "- `G-BETA=FAIL` remains.", "")
    assert any("G-BETA=PASS but the gate registry says FAIL" in m for m in errors_of(repo))


def test_f_r9_ordinary_two_clause_declaration_stays_clean(repo: Path) -> None:
    """Two true statements on one line, in the structural form, stay clean.

    F-COMMA-ADJACENCY was the round-9 defect where `ID1 is S1, ID2 is S2` let
    ID2 take its NEIGHBOUR's status because the separator widths decided it.
    ``ID=STATUS`` removes the question: there is no separator to measure.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-BETA=FAIL` remains.",
        "- `G-ALPHA=PASS` remains and `G-BETA=FAIL` also.",
    )
    assert_clean(repo)


def test_f_r9_ordinary_two_clause_prose_is_now_refused(repo: Path) -> None:
    """The measured cost of the round-11 inversion, pinned rather than implied.

    This input is TRUE in both clauses and used to pass. It is now an error,
    because the checker no longer decides what a sentence means. Refusing four
    honest sentences corpus-wide was the price of the six-string bypass family
    dying; if that price ever rises, this fixture is where it shows up.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-BETA=FAIL` remains.",
        "- `G-ALPHA` remains PASS and `G-BETA` is FAIL.",
    )
    messages = errors_of(repo)
    assert any(REFUSED in m for m in messages), messages


def test_f_r9_shared_subject_list_lie_is_still_caught(repo: Path) -> None:
    """`A` and `B` are both PASS is a lie about B, and must not survive.

    Under the binder this needed a subject-group rule so one token could reach
    two ids. It now needs no rule at all: the sentence names a gate beside a
    bare token, which is refused before anyone asks who the token belongs to.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-ALPHA=PASS` remains and `C-LIVE=VALIDATED`.",
        "- `G-ALPHA` and `G-BETA` are both PASS.\n- `C-LIVE=VALIDATED`.",
    )
    messages = errors_of(repo)
    assert any("G-BETA" in m and REFUSED in m for m in messages), messages


def test_f_r9_regex_matching_nothing_is_an_error_not_the_value_zero(repo: Path) -> None:
    """`grep -c` prints 0 on no match; moving to Python did not fix that alone."""
    facts_and_measurement(
        repo, measurement=dict(HOOK_MEASUREMENT, pattern="^def NOSUCHTHING_"), value=0
    )
    messages = errors_of(repo)
    assert any("matched no lines" in m and "not the value 0" in m for m in messages), messages


def test_f_r9_in_tree_symlink_pointing_outside_the_repo_is_refused(repo: Path) -> None:
    """The path check was about the STRING; containment is about the FILE."""
    outside = repo.parent / "outside_target.py"
    write(outside, "def test_planted() -> None:\n    pass\n")
    planted = repo / ".agent-harness/tests/borrowed.py"
    planted.symlink_to(outside)
    facts_and_measurement(repo, measurement=dict(HOOK_MEASUREMENT, path=str(planted.relative_to(repo))), value=1)
    messages = errors_of(repo)
    assert any("outside the repository" in m for m in messages), messages


def test_f_r9_description_must_name_the_file_the_measurement_reads(repo: Path) -> None:
    """A fact described as one file may not quietly measure another."""
    facts_and_measurement(repo, measurement=dict(HOOK_MEASUREMENT, path="LICENSE"))
    messages = errors_of(repo)
    assert any("never names that file" in m for m in messages), messages


def test_f_r9_frozen_role_cannot_sit_in_a_live_region(repo: Path) -> None:
    """`role: frozen` is a claim about WHERE the line is, and must be true."""
    facts_and_measurement(
        repo,
        extra_assertions=[
            {
                "file": "docs/harness/VALIDATION_LEDGER.md",
                "line_contains": "the current row",
                "value_regex": r"(\d+) at the seal",
                "role": "frozen",
                "pinned_to_commit": "ed7bc49",
            }
        ],
    )
    edit(
        repo,
        "docs/harness/VALIDATION_LEDGER.md",
        "| 2026-07-29 | the current row | recorded | none |",
        "| 2026-07-29 | the current row: 35 at the seal `ed7bc49` | recorded | none |",
    )
    messages = errors_of(repo)
    assert any("inside a LIVE region" in m for m in messages), messages


def test_f_r9_historical_role_in_a_live_row_stays_legal(repo: Path) -> None:
    """The control for the guard above.

    A live row citing a superseded number ALONGSIDE the commit it held at is
    ordinary and correct, and is already constrained by the same-line commit
    test. Only `frozen`, whose entire licence is position, is restricted.
    """
    assert_clean(repo)


def test_f_r9_a_later_undeclared_section_cannot_demote_the_controlling_overlay(
    repo: Path,
) -> None:
    """Liveness used `newest` over all headings; the tie checks saw only
    'controlling' ones, so a later plain section silently emptied the overlay."""
    state = (repo / "docs/harness/PROJECT_STATE.md").read_text(encoding="utf-8")
    write(
        repo / "docs/harness/PROJECT_STATE.md",
        state + "\n## 2099-01-01 appendix\n\nNothing to see here.\n",
    )
    messages = errors_of(repo)
    assert any("silently demotes the controlling overlay" in m for m in messages), messages


def test_f_r9_denial_that_agrees_by_accident_does_not_satisfy_coverage(repo: Path) -> None:
    """NEGATION_RE is a denylist and can never be finished.

    "has no bearing on whether ... VALIDATED" reads to a human as an explicit
    denial and to the denylist as an assertion -- and because the status it
    lands on happens to match the registry, it produced no contradiction and
    silently satisfied the coverage floor for a claim stated nowhere real.
    Coverage now requires an AFFIRMATIVE connector, so the floor fails loudly
    instead. The contradiction path deliberately keeps the loose window: failing
    to catch a lie is the expensive error, and this is the cheap one.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-ALPHA=PASS` remains and `C-LIVE=VALIDATED`.",
        "- `G-ALPHA` remains PASS.\n"
        "- `C-LIVE` has no bearing on whether the lane is VALIDATED.",
    )
    messages = errors_of(repo)
    assert any(
        "C-LIVE=VALIDATED is asserted nowhere in any live region" in m for m in messages
    ), messages


# --------------------------------------------------------------------------
# Round 10 (D-070 Part B12). The panel found the B9 binder broken in the paths
# B9 never covered. One negative fixture per repair; each must die on revert.
# --------------------------------------------------------------------------


def test_f_r10_board_status_cell_must_be_a_bare_status(repo: Path) -> None:
    """The panel's first critical finding.

    Board rows were injected marked affirmative by construction, so they were
    the one path running through NEITHER `_negated` NOR `_affirmative`. A
    denial in the Status cell gave exit 0, ok:true, and still counted the gate
    as covered. A Status COLUMN holds a status, not a sentence about one.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-ALPHA=PASS` remains and `C-LIVE=VALIDATED`.\n- `G-BETA=FAIL` remains.",
        "| Gate | Status |\n|---|---|\n| `G-ALPHA` | PASS |\n"
        "| `G-BETA` | no longer FAIL |\n| `C-LIVE` | VALIDATED |\n",
    )
    messages = errors_of(repo)
    assert any("rather than the bare status" in m for m in messages), messages


def test_f_r10_board_bare_status_still_passes(repo: Path) -> None:
    """Control: markup around the token is still a bare status."""
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-ALPHA=PASS` remains and `C-LIVE=VALIDATED`.\n- `G-BETA=FAIL` remains.",
        "| Gate | Status |\n|---|---|\n| `G-ALPHA` | **PASS** |\n"
        "| `G-BETA` | `FAIL` |\n| `C-LIVE` | VALIDATED |\n",
    )
    assert_clean(repo)


def test_f_r10_negation_rebinds_instead_of_dropping_the_id(repo: Path) -> None:
    """`is no longer FAIL but PASS` left PASS orphaned and never tested.

    G-BETA is FAIL in the fixture registry, so the trailing PASS is a live
    false statement and must be reported. Dropping the id at the negated first
    candidate exited 0.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-BETA=FAIL` remains.",
        "- `G-BETA` is no longer PASS but PASS.",
    )
    messages = errors_of(repo)
    assert any("G-BETA" in m and REFUSED in m for m in messages), messages


def test_f_r10_a_true_denial_is_refused_not_accepted(repo: Path) -> None:
    """`G-BETA is not PASS` is TRUE, and is refused anyway.

    That is the point of the inversion. Deciding this sentence is honest means
    deciding that `not` denies -- and `is not only PASS`, `is not merely PASS`
    and `is not just PASS` are the same three characters AFFIRMING. No reader
    of `not` can tell those apart; a reader of structure never has to.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-BETA=FAIL` remains.",
        "- `G-BETA` is not PASS.\n- `G-BETA=FAIL` remains.",
    )
    messages = errors_of(repo)
    assert any("G-BETA" in m and REFUSED in m for m in messages), messages


def test_f_r10_an_abbreviation_cannot_sever_a_lie_from_its_detection(repo: Path) -> None:
    """`. ` fires inside `cf.`, `Sec.` and `e.g.`.

    Under one boundary tier the false split left a clause holding ids and no
    tokens, which was skipped in SILENCE -- every abbreviation in a technical
    handoff was a detector-disabling device.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-BETA=FAIL` remains.",
        "- `G-BETA` (r10, cf. the appendix) is PASS.",
    )
    messages = errors_of(repo)
    # Reported, not silently skipped, and the id gets no coverage either -- both
    # halves matter, because a silent skip is how the lie used to travel. The
    # abbreviation splits the sentence; a narrower scope still forbids what the
    # wider one forbade, so no abbreviation list is needed to close this.
    assert any("G-BETA" in m and REFUSED in m for m in messages), messages
    assert any("G-BETA" in m and "asserted nowhere" in m for m in messages), messages


def test_f_r10_a_sentence_end_still_breaks_the_subject_group(repo: Path) -> None:
    """Soft boundaries must keep separating subjects, or the abbreviation fix
    would turn two sentences into one shared subject."""
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-ALPHA=PASS` remains and `C-LIVE=VALIDATED`.",
        "- We reviewed `G-ALPHA`. Separately, `C-LIVE=VALIDATED`.\n- `G-ALPHA=PASS`.",
    )
    assert_clean(repo)


def test_f_r10_an_unrelated_token_beside_an_id_is_refused_not_bound(repo: Path) -> None:
    """The backward fallback is gone, and so is the false contradiction it made.

    The round-10 defect was that `G-ALPHA` reached back across a sentence end
    and took `VALIDATED` from a subject that was not a registry id, reporting a
    contradiction that did not exist. Nothing binds by distance now, so no
    false contradiction is possible -- but the segment is still REFUSED, and
    that is the deliberate trade. A bare status token beside a gate id is not
    interpreted in either direction; it is sent back to be written properly.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-ALPHA=PASS` remains and `C-LIVE=VALIDATED`.",
        "- Something unrelated is VALIDATED. We also reviewed `G-ALPHA`.\n"
        "- `G-ALPHA=PASS`.\n- `C-LIVE=VALIDATED`.",
    )
    messages = errors_of(repo)
    assert any("G-ALPHA" in m and REFUSED in m for m in messages), messages
    # ...and specifically NOT the round-10 false contradiction.
    assert not any("G-ALPHA=VALIDATED" in m for m in messages), messages


def test_f_r10_a_separate_bullet_keeps_an_unrelated_token_clean(repo: Path) -> None:
    """The escape hatch, so the refusal is not a demand for silence.

    A bullet is a HARD boundary, so an unrelated status word costs nothing as
    long as it does not share a segment with a registry id. This is what a
    writer does instead of arguing with the checker.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-ALPHA=PASS` remains and `C-LIVE=VALIDATED`.",
        "- Something unrelated is VALIDATED.\n- We also reviewed `G-ALPHA`.\n"
        "- `G-ALPHA=PASS`.\n- `C-LIVE=VALIDATED`.",
    )
    assert_clean(repo)


def test_f_r10_affirmative_requires_a_present_tense_copula(repo: Path) -> None:
    """The flat allowlist accepted `was FAIL` -- past tense is not a claim
    about now -- so a superseded status could satisfy the coverage floor."""
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-ALPHA=PASS` remains and `C-LIVE=VALIDATED`.",
        "- `G-ALPHA` remains PASS.\n- `C-LIVE` was VALIDATED.",
    )
    messages = errors_of(repo)
    assert any(
        "C-LIVE=VALIDATED is asserted nowhere in any live region" in m for m in messages
    ), messages


def test_f_r10_present_tense_hedges_are_refused_with_everything_else(repo: Path) -> None:
    """The copula rule is withdrawn; tense no longer decides anything.

    It was introduced so `is currently PASS` would grant coverage while
    `was PASS` would not, and it cost two rounds of argument about which verbs
    count. Coverage now comes only from a board row or ``ID=STATUS``, so there
    is no tense to get right.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-ALPHA=PASS` remains and `C-LIVE=VALIDATED`.",
        "- `G-ALPHA` is currently PASS.\n- `C-LIVE` continues to be VALIDATED.",
    )
    messages = errors_of(repo)
    assert any("G-ALPHA" in m and REFUSED in m for m in messages), messages
    assert any("C-LIVE" in m and REFUSED in m for m in messages), messages


def test_f_r10_historical_role_cannot_sit_in_a_live_region_either(repo: Path) -> None:
    """Round 10 disproved the published justification for exempting it.

    The `dated` test is a bare substring search for the commit ANYWHERE on the
    line, so a present-tense sentence with a parenthesised hash -- using only a
    legitimately declared prior -- passed the unmutated checker at exit 0. A
    parenthesised hash does not make a sentence read as history to anyone, so
    the rule is positional again, which is checkable.
    """
    write(
        repo / ".agent-harness/context/SSOT_FACTS.json",
        facts_document(
            extra_assertions=[
                {
                    "file": ".agent-harness/context/SHARED_CONTEXT.md",
                    "line_contains": "actually",
                    "value_regex": r"(\d+) hook fixtures",
                    "role": "historical",
                }
            ]
        ),
    )
    path = repo / ".agent-harness/context/SHARED_CONTEXT.md"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "\nThere are 35 hook fixtures actually (ed7bc49).\n",
        encoding="utf-8",
    )
    messages = errors_of(repo)
    assert any(
        "declared with role 'historical'" in m and "inside a LIVE region" in m
        for m in messages
    ), messages


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


R11_BYPASS_STRINGS = [
    pytest.param("- `G-BETA` was FAIL but is now PASS.", id="decoy-was-is-now"),
    pytest.param("- `G-BETA` is no longer FAIL; it is PASS.", id="decoy-semicolon"),
    pytest.param("- `G-BETA` is far from FAIL and is PASS.", id="decoy-far-from"),
    pytest.param("- `G-BETA` is cleared; it is PASS.", id="decoy-pronoun"),
    pytest.param("- `G-BETA` is not only PASS but also recorded.", id="affirmative-not-only"),
    pytest.param("- `G-BETA` is not merely PASS but fully closed.", id="affirmative-not-merely"),
    pytest.param("- `G-BETA` is not just PASS but exemplary.", id="affirmative-not-just"),
]


@pytest.mark.parametrize("sentence", R11_BYPASS_STRINGS)
def test_f_r11_bypass_string_cannot_reach_a_live_surface(repo: Path, sentence: str) -> None:
    """Every measured bypass must fail the checker, and none may be read as PASS."""
    edit(repo, "docs/harness/PROJECT_STATE.md", "- `G-BETA=FAIL` remains.", sentence)
    messages = errors_of(repo)
    assert any("G-BETA" in m and REFUSED in m for m in messages), messages


@pytest.mark.parametrize("sentence", R11_BYPASS_STRINGS)
def test_f_r11_bypass_string_never_grants_coverage(repo: Path, sentence: str) -> None:
    """The other half: a refused segment must not satisfy the coverage floor.

    Round 9's F-NEGATION-CLOSED-VOCAB was exactly this -- a misread denial that
    counted as the one live statement a gate needed. Reporting the line while
    still counting it as coverage would leave that hole open.
    """
    edit(repo, "docs/harness/PROJECT_STATE.md", "- `G-BETA=FAIL` remains.", sentence)
    messages = errors_of(repo)
    assert any("G-BETA=FAIL is asserted nowhere" in m for m in messages), messages


def test_f_r11_the_string_an_auditor_committed_to_this_repository(repo: Path) -> None:
    """The exact bytes an external auditor left in a commit on this branch.

    Recorded verbatim because it was a real write to the real branch ref, not a
    hypothetical: it is the only bypass in this family that was ever committed.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-BETA=FAIL` remains.",
        "Current correction: `G-BETA` was **FAIL** but is now **PASS**.",
    )
    messages = errors_of(repo)
    assert any("G-BETA" in m and REFUSED in m for m in messages), messages


def test_f_r11_a_false_structural_declaration_is_still_a_contradiction(repo: Path) -> None:
    """Refusal must not become the only outcome: a legal FORM stating a false
    status is reported as the contradiction it is, naming both values."""
    edit(repo, "docs/harness/PROJECT_STATE.md", "- `G-BETA=FAIL` remains.", "- `G-BETA=PASS` now.")
    messages = errors_of(repo)
    assert any("G-BETA=PASS but the gate registry says FAIL" in m for m in messages), messages


@pytest.mark.parametrize(
    "form",
    [
        pytest.param("`G-BETA=FAIL`", id="backticked"),
        pytest.param("**G-BETA=FAIL**", id="bold"),
        pytest.param("G-BETA=FAIL", id="bare"),
        pytest.param("`G-BETA` = `FAIL`", id="spaced-and-split-markup"),
    ],
)
def test_f_r11_markup_around_the_equals_is_not_grammar(repo: Path, form: str) -> None:
    """The corpus writes the structural form several ways; all must be read.

    If markup decided whether a declaration counted, then markup would be a
    detector-disabling device -- which is precisely what `**FAIL**` was under
    the old board-cell rule before round 10 forced the cell to reduce to a
    bare token.
    """
    edit(repo, "docs/harness/PROJECT_STATE.md", "- `G-BETA=FAIL` remains.", f"- {form} remains.")
    assert_clean(repo)


def test_f_r11_a_misspelt_id_does_not_satisfy_the_floor_for_the_real_one(repo: Path) -> None:
    """An id the registries do not contain is INVISIBLE, and that is stated.

    Both regexes are built from the registry, so `G-BET=PASS` matches neither
    the structural form nor the bare-id scan: it is not reported, and it never
    was. Writing a test that asserted otherwise would have been a fixture that
    documents a guard nobody wrote.

    What actually protects the corpus here is the coverage floor. A misspelling
    cannot stand in for the gate it resembles, because the real gate still has
    to be asserted somewhere live, and this pins that -- the floor is the guard,
    so the floor is what gets the fixture.
    """
    edit(repo, "docs/harness/PROJECT_STATE.md", "- `G-BETA=FAIL` remains.", "- `G-BET=FAIL` remains.")
    messages = errors_of(repo)
    assert any("G-BETA=FAIL is asserted nowhere" in m for m in messages), messages
    assert not any("G-BET=" in m and "G-BETA" not in m for m in messages), messages


# --------------------------------------------------------------------------
# F-R12 -- round 12 attacked round 11's own fix and found three more ways for a
# false gate status to be visible to a reader and invisible to this file.
#
# All three exploit the same seam: the checker's vocabulary of what a status
# LOOKS like is narrower than a reader's. Each is closed structurally rather
# than by widening a match, and each cost zero on the live corpus.
# --------------------------------------------------------------------------


def test_f_r12_a_lower_case_status_beside_an_id_is_refused(repo: Path) -> None:
    """`G-BETA` is pass. -- STATUS_RE is upper-case only, so this was invisible."""
    edit(repo, "docs/harness/PROJECT_STATE.md", "- `G-BETA=FAIL` remains.", "- `G-BETA` is pass.")
    assert any("G-BETA" in m and REFUSED in m for m in errors_of(repo)), errors_of(repo)


def test_f_r12_a_lower_case_declaration_is_refused(repo: Path) -> None:
    """`G-BETA=pass` matched neither the structural form nor the bare-token scan."""
    edit(repo, "docs/harness/PROJECT_STATE.md", "- `G-BETA=FAIL` remains.", "- `G-BETA=pass`.")
    assert any("G-BETA" in m and REFUSED in m for m in errors_of(repo)), errors_of(repo)


def test_f_r12_an_ordinary_word_that_contains_a_status_is_not_a_status(repo: Path) -> None:
    """The control the case-insensitive scan must not break.

    `passes` and `failed` contain a status token and are ordinary English. If
    the word boundary were dropped to catch lower case, every sentence in the
    corpus that mentions a test suite would become an error.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-BETA=FAIL` remains.",
        "- The suite passes and `G-BETA` was reviewed; nothing failed.\n- `G-BETA=FAIL` remains.",
    )
    assert_clean(repo)


def test_f_r12_an_invisible_character_in_a_live_line_is_refused(repo: Path) -> None:
    """`G-HARNESS<ZWSP>-INTEGRITY` is not a registry id and displays as one.

    There is no way to read past this by improving the id match, because the
    string genuinely is not the id. The character class is refused instead.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-BETA=FAIL` remains.",
        "- `G-BE​TA` is PASS.\n- `G-BETA=FAIL` remains.",
    )
    assert any("zero-width or bidirectional" in m for m in errors_of(repo)), errors_of(repo)


def test_f_r12_a_bidi_override_in_a_live_line_is_refused(repo: Path) -> None:
    """Same class: U+202E reorders what follows it when displayed."""
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-BETA=FAIL` remains.",
        "- ‮G-BETA is PASS.\n- `G-BETA=FAIL` remains.",
    )
    assert any("zero-width or bidirectional" in m for m in errors_of(repo)), errors_of(repo)


def test_f_r12_a_fenced_block_may_not_hold_an_id_beside_a_status(repo: Path) -> None:
    """Fences are blanked before anything reads them -- correctly, since this
    corpus quotes attack strings. Blanking decides what the CHECKER sees; it
    does not decide what a PERSON sees, and that gap is this module's subject."""
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-BETA=FAIL` remains.",
        "- `G-BETA=FAIL` remains.\n\n```\nG-BETA=PASS\n```\n",
    )
    assert any("fenced block sits in a live region" in m for m in errors_of(repo)), errors_of(repo)


def test_f_r12_a_fenced_block_without_a_registry_id_is_fine(repo: Path) -> None:
    """The control: ordinary command samples mentioning PASS stay legal."""
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-BETA=FAIL` remains.",
        "- `G-BETA=FAIL` remains.\n\n```\nvenv/bin/python check.py   # expect PASS\n```\n",
    )
    assert_clean(repo)


def test_f_r13_a_homoglyph_status_is_refused(repo: Path) -> None:
    """Round 13, registered reviewer, CRITICAL.

    Round 12 closed characters that render as NOTHING and stopped there. The
    reviewer used the other half: characters that render as SOMETHING ELSE.
    Cyrillic ER, A, DZE, DZE spells a perfect `PASS` that matched nothing here,
    and produced output byte-identical to a clean run -- not even a refusal,
    because the refusal only fires once a literal token has been found.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-BETA=FAIL` remains.",
        "- `G-BETA=РАЅЅ` is recorded.\n- `G-BETA=FAIL` remains.",
    )
    assert any("non-ASCII letters" in m for m in errors_of(repo)), errors_of(repo)


def test_f_r13_a_homoglyph_in_prose_beside_an_id_is_refused(repo: Path) -> None:
    """The same substitution in the prose form, which is refused unread anyway --
    except that it was NOT, because the refusal needs a literal token first."""
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-BETA=FAIL` remains.",
        "- `G-BETA` is РАЅЅ.\n- `G-BETA=FAIL` remains.",
    )
    assert any("non-ASCII letters" in m for m in errors_of(repo)), errors_of(repo)


def test_f_r13_ordinary_non_ascii_punctuation_stays_legal(repo: Path) -> None:
    """The control, and the reason the rule is LETTERS and not all non-ASCII.

    The live corpus legitimately carries em dashes, arrows and section signs.
    Refusing every non-ASCII byte would have cost fifteen honest characters to
    buy the same protection that costs zero when scoped to letters.
    """
    edit(
        repo,
        "docs/harness/PROJECT_STATE.md",
        "- `G-BETA=FAIL` remains.",
        "- `G-BETA=FAIL` remains — see § 4 → the appendix.",
    )
    assert_clean(repo)
