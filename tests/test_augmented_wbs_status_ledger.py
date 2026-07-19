from __future__ import annotations

import re
from pathlib import Path


WBS_PATH = Path("docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md")
PR_CATALOG_PATH = Path("docs/ROADMAP_PR_CATALOG.md")


def _ap_statuses() -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in WBS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| AP"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[0].startswith("AP") and cells[0][2:].isdigit():
            statuses[cells[0]] = cells[1]
    return statuses


def test_augmented_wbs_distinguishes_stage_landed_rows_from_program_partial_rows() -> None:
    statuses = _ap_statuses()

    for idx in (4,):
        assert statuses[f"AP{idx}"] == "partial"

    for idx in (5, 6, 7, 8, 9, *range(10, 51)):
        assert statuses[f"AP{idx}"] == "stage-recorded (historical; not current capability)"


def test_augmented_wbs_completion_queue_rows_are_stage_recorded() -> None:
    statuses = _ap_statuses()

    for idx in (51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80):
        assert statuses[f"AP{idx}"] == "stage-recorded (historical; not current capability)"


def test_augmented_pr_catalog_covers_planned_completion_queue_boundaries() -> None:
    catalog = PR_CATALOG_PATH.read_text(encoding="utf-8")

    assert "### AP51-FULL-SPAN-OUTCOME-POLICY" in catalog
    assert "### AP76-FINAL-PUBLICATION-READINESS-AUDIT" in catalog
    assert "### AP77-COUPLED-WEAK-RATE-GATE" in catalog
    assert "### AP78-PUBLICATION-MATRIX-SAME-CL3-WEAK-RATE-CONTROL" in catalog
    assert "### AP79-READINESS-AUDIT-COUPLED-WEAK-RATE-LINK" in catalog
    assert "### AP80-COUPLED-WEAK-RATE-CONVERGENCE" in catalog
    assert "AP51-AP75 evidence" in catalog


def _planned_catalog_sections() -> dict[int, str]:
    catalog = PR_CATALOG_PATH.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^### AP(\d+)-", catalog, flags=re.MULTILINE))
    sections: dict[int, str] = {}
    for pos, match in enumerate(matches):
        idx = int(match.group(1))
        if idx < 51 or idx > 80:
            continue
        end = matches[pos + 1].start() if pos + 1 < len(matches) else len(catalog)
        sections[idx] = catalog[match.start():end]
    return sections


def test_augmented_pr_catalog_has_completion_entries_with_status_boundaries() -> None:
    sections = _planned_catalog_sections()

    assert set(sections) == set(range(51, 81))
    for idx, section in sections.items():
        assert "- **Status:** stage-recorded (historical; not current capability)" in section, f"AP{idx} has the wrong status"
        assert "- **Scope boundary:**" in section, f"AP{idx} lacks a scope boundary"
        assert "- **Exit gate:**" in section, f"AP{idx} lacks an exit gate"


def test_augmented_completion_plan_locks_reuse_first_contract() -> None:
    wbs = WBS_PATH.read_text(encoding="utf-8")
    catalog = PR_CATALOG_PATH.read_text(encoding="utf-8")
    combined = f"{wbs}\n{catalog}"

    for required in (
        "abundance_rhs_phase2",
        "AugmentedWeakInputs",
        "WeakAngularCorrectionMetadata",
        "AngularWeakRateMomentInputs",
        "nonlinear_transport.py",
        "ForwardModel",
        "BBNLikelihood",
        "run_nonlrs_typeI_blackjax_adaptive_smc.py",
        "jax/smc_pipeline.py",
        "generate_paper_figures.py",
        "figure_registry.py",
        "figure_cache_schema.py",
        "Schramm",
    ):
        assert required in combined


def test_augmented_completion_plan_preserves_no_public_dispatch_boundary() -> None:
    catalog = PR_CATALOG_PATH.read_text(encoding="utf-8")
    catalog_words = " ".join(catalog.split())

    assert "PROMOTED-DIAGNOSTIC" not in catalog
    assert "Promoted diagnostic" not in catalog
    assert "no accidental `canonical_forward_solver` promotion" in catalog
    assert "without canonical/public default dispatch" in catalog_words
    assert "no public canonical dispatch" in catalog_words
    assert "no QKE" in catalog
