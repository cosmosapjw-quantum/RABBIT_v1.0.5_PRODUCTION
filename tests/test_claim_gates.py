"""Foundation F13 — claim-gate coverage and structural integrity tests.

Every line under the SUPPORTED_CAPABILITIES.md "Forbidden Claims" section
must have a corresponding ClaimGate registered in
``rabbit.config.claim_gates``.  This test fails CI if a Forbidden line is
added without a gate, or if the gate's stored ``forbidden_text`` drifts
from the markdown.

This is the wording-drift firewall: claims cannot silently move from
Forbidden to Permitted without going through the promotion script that
runs the gate's required tests.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from rabbit.config.claim_gates import (
    ALL_GATES,
    ClaimGate,
    forbidden_lines_set,
    gates_by_id,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SC_PATH = REPO_ROOT / "SUPPORTED_CAPABILITIES.md"

CLAIMS_BLOCK_RE = re.compile(
    r"<!--\s*BEGIN:CLAIMS\s*-->(?P<body>.*?)<!--\s*END:CLAIMS\s*-->",
    re.DOTALL,
)
FORBIDDEN_LINE_RE = re.compile(r"^\-\s*(?P<text>\".+)\s*$", re.MULTILINE)


def _read_forbidden_block() -> str:
    text = SC_PATH.read_text(encoding="utf-8")
    match = CLAIMS_BLOCK_RE.search(text)
    if not match:
        raise AssertionError("SUPPORTED_CAPABILITIES.md missing CLAIMS block markers.")
    body = match.group("body")
    # Take everything between "## Forbidden Claims" and "## Permitted Claims".
    forbidden_start = body.index("## Forbidden Claims")
    permitted_start = body.index("## Permitted Claims")
    return body[forbidden_start:permitted_start]


def _extract_forbidden_lines() -> list[str]:
    block = _read_forbidden_block()
    return [m.group("text").strip() for m in FORBIDDEN_LINE_RE.finditer(block)]


@pytest.mark.production
@pytest.mark.release_smoke
class TestClaimGateCoverage:
    def test_every_forbidden_line_has_a_gate(self):
        registered = forbidden_lines_set()
        on_disk = set(_extract_forbidden_lines())
        missing = on_disk - registered
        assert missing == set(), (
            "Forbidden lines without a ClaimGate (add an entry to "
            "src/rabbit/config/claim_gates.py for each):\n"
            + "\n".join(sorted(missing))
        )

    def test_no_gate_references_a_line_not_on_disk(self):
        registered = forbidden_lines_set()
        on_disk = set(_extract_forbidden_lines())
        stale = registered - on_disk
        assert stale == set(), (
            "ClaimGate.forbidden_text values not present in SUPPORTED_CAPABILITIES.md "
            "(either delete the gate or restore the line):\n"
            + "\n".join(sorted(stale))
        )

    def test_gate_ids_unique(self):
        ids = [g.claim_id for g in ALL_GATES]
        assert len(ids) == len(set(ids)), f"Duplicate claim_id values: {ids}"

    def test_promotable_gates_have_test_node_ids(self):
        """A gate with empty required_test_node_ids and empty permitted_text
        is a documentation-only Forbidden (e.g. architectural decisions).
        Otherwise both must be non-empty."""
        for g in ALL_GATES:
            if not g.required_test_node_ids and not g.permitted_text:
                continue  # documentation-only gate
            assert g.required_test_node_ids, (
                f"Gate {g.claim_id!r} is promotable but has no required_test_node_ids"
            )
            assert g.permitted_text, (
                f"Gate {g.claim_id!r} is promotable but has empty permitted_text"
            )

    def test_test_node_ids_are_well_formed(self):
        """Defensive: catch typos like missing '::' separator before they hit CI."""
        for g in ALL_GATES:
            for node in g.required_test_node_ids:
                assert node.startswith("tests/"), (
                    f"Gate {g.claim_id!r}: node {node!r} must start with 'tests/'"
                )
                # Either a file-only reference (entire file's tests) or file::Class::method.
                if "::" in node:
                    parts = node.split("::")
                    assert len(parts) in (2, 3), (
                        f"Gate {g.claim_id!r}: node {node!r} has unexpected '::' depth"
                    )


@pytest.mark.production
@pytest.mark.release_smoke
class TestPermittedTextHygiene:
    def test_permitted_lines_are_distinct(self):
        seen: set[str] = set()
        for g in ALL_GATES:
            if not g.permitted_text:
                continue
            assert g.permitted_text not in seen, (
                f"Two gates emit identical permitted_text: {g.permitted_text!r}"
            )
            seen.add(g.permitted_text)

    def test_permitted_text_does_not_contain_forbidden_keywords(self):
        """No promoted line should contain 'forbidden', 'cannot', 'not yet'."""
        bad = ("forbidden", "cannot", "not yet", "TBD", "TODO")
        for g in ALL_GATES:
            low = g.permitted_text.lower()
            for word in bad:
                assert word.lower() not in low, (
                    f"Gate {g.claim_id!r} permitted_text contains forbidden keyword "
                    f"{word!r}: {g.permitted_text!r}"
                )


@pytest.mark.production
@pytest.mark.release_smoke
class TestSourceClaimLanguageHygiene:
    """Regression firewall for the 2026-06-30 external-audit claim downscope (PR#20).

    The auditor flagged maturity-overstatement strings ("publication-grade",
    "PUBLIC TIER", "tier-3 eliminates residual gap") on UN-gated production
    registry / driver / substrate sites -- language that exceeds actual path
    support. Each was qualified. This test fails CI if a bare maturity claim
    reappears on a remediated site, while explicitly NOT clobbering
    (a) the GATED ``permitted_text`` in claim_gates.py (which legitimately reads
    "Publication-grade" but unlocks ONLY when real external/SMC test nodes pass),
    nor (b) the pre-existing honest hedges. A curated site list is used on
    purpose -- a blanket grep for "publication-grade" would false-positive on the
    gated/hedged occurrences.
    """

    # (relative path, forbidden bare substring, required qualifier substring)
    _REMEDIATED = (
        ("src/rabbit/config/backend_capabilities.py",
         "This is the publication-grade JAX Bianchi I BBN path.",
         "no external anisotropic anchor"),
        ("src/rabbit/config/backend_capabilities.py",
         "residual gap is eliminated by the full-collision tier-3 upgrade",
         "tier-3 closure"),
        ("src/rabbit/jax/driver_typeI_char.py",
         "This is the publication-grade JAX path",
         "FLRW-anchored"),
        ("src/rabbit/jax/driver_typeI.py",
         "MATURITY: PUBLIC TIER-1/TIER-2 SURFACES WIRED THROUGH CL0--CL3",
         "FLRW-anchored"),
        ("src/rabbit/jax/driver_typeI.py",
         "Characteristic-ray dispatch (publication-grade, LRS, tier-1 or tier-2)",
         "canonical, LRS"),
        # Test-file docstrings/comments that described the path/parity as "publication-grade"
        # (audit-flagged + found by the PR#20 review). Firewalled so a revert can't slip back.
        ("tests/test_jax_typeI_characteristic_parity.py",
         "This test establishes the publication-grade JAX Bianchi I BBN path.",
         "canonical, FLRW-anchored JAX tier-1 characteristic"),
        ("tests/test_jax_typeI_characteristic_parity.py",
         "tolerances (tight, publication-grade)",
         "the canonical tier-1 parity bar"),
        ("tests/test_cross_backend_regression.py",
         "parity becomes publication-grade",
         "the canonical tier-1 bar"),
        ("tests/test_jax_typeI_characteristic_tier2.py",
         "pins the post-fix parity at publication grade",
         "the canonical tier-1 bar"),
    )
    # F06 hard-retired these endpoint and endpoint-test surfaces.  Absence is
    # now the stronger claim-hygiene contract: restoring any one of them must
    # also restore the curated qualifier check above deliberately.
    _HARD_RETIRED = frozenset({
        "src/rabbit/jax/driver_typeI.py",
        "src/rabbit/jax/driver_typeI_char.py",
        "tests/test_jax_typeI_characteristic_parity.py",
        "tests/test_cross_backend_regression.py",
        "tests/test_jax_typeI_characteristic_tier2.py",
    })

    _PSTF_MODULES = (
        "pstf_lrs_moment_kernel", "pstf_collision_operator", "pstf_x_integral",
        "pstf_trilinear_kernel", "pstf_reduced_kernel", "pstf_body_frame",
        "pstf_matrix_element",
    )
    _PSTF_FORBIDDEN = "publication-quality nonperturbative Bianchi-I programme"
    _PSTF_REQUIRED = "nonperturbative Bianchi-I substrate"

    # Exemptions: legitimate occurrences that MUST survive the downscope.
    _EXEMPT_PRESENT = (
        ("src/rabbit/config/claim_gates.py", "Publication-grade"),       # gated permitted_text
        ("src/rabbit/config/feature_capabilities.py", "not publication-grade"),
        ("src/rabbit/drivers/classA_driver.py", "not publication-grade"),
    )

    def _read(self, rel: str) -> str:
        return (REPO_ROOT / rel).read_text(encoding="utf-8")

    def test_remediated_sites_drop_bare_maturity_claims(self):
        for rel, forbidden, required in self._REMEDIATED:
            if rel in self._HARD_RETIRED:
                continue
            text = self._read(rel)
            assert forbidden not in text, (
                f"{rel}: bare maturity claim reappeared: {forbidden!r}")
            assert required in text, (
                f"{rel}: expected qualifier {required!r} missing -- the downscope "
                "must QUALIFY the claim, not merely delete it")

    def test_hard_retired_claim_surfaces_remain_absent(self):
        restored = sorted(rel for rel in self._HARD_RETIRED if (REPO_ROOT / rel).exists())
        assert not restored, (
            "F06-retired JAX endpoint claim surfaces were restored without an "
            "authority decision and renewed source-claim audit: " + ", ".join(restored)
        )

    def test_pstf_substrate_modules_not_called_publication_quality(self):
        for mod in self._PSTF_MODULES:
            rel = f"src/rabbit/collisions/{mod}.py"
            text = self._read(rel)
            assert self._PSTF_FORBIDDEN not in text, (
                f"{rel}: PSTF substrate still claims {self._PSTF_FORBIDDEN!r}")
            assert self._PSTF_REQUIRED in text, (
                f"{rel}: expected substrate framing {self._PSTF_REQUIRED!r}")

    def test_honest_hedges_and_gated_claims_survive(self):
        for rel, present in self._EXEMPT_PRESENT:
            text = self._read(rel)
            assert present in text, (
                f"{rel}: legitimate text {present!r} was clobbered by the downscope")


@pytest.mark.production
@pytest.mark.release_smoke
class TestRepoWideMaturityClaimScan:
    """Repo-wide firewall (external audit BD601 AUDIT2-3). The PR#20
    ``TestSourceClaimLanguageHygiene`` is curated-site; this scans ALL of ``src/rabbit`` for bare
    maturity over-claims so a NEW un-gated site fails CI, not only the curated ones. Allowlist: the
    ``claim_gates.py`` registry (its job is to NAME these claims, gated behind required tests) and
    honest 'not publication-...' hedges."""

    _LEXICON = (
        "publication-grade", "publication grade",
        "publication-quality", "publication quality",
        "public tier",
    )
    # Unicode hyphen/dash variants normalized to ASCII '-' so "publication‑grade" (U+2011 etc.)
    # cannot bypass the ASCII-hyphen lexicon.
    _HYPHENS = "‐‑‒–—−"

    def _norm(self, line: str) -> str:
        out = line.lower()
        for h in self._HYPHENS:
            out = out.replace(h, "-")
        return out

    def _is_hedge(self, line_norm: str) -> bool:
        return any(f"not {term}" in line_norm for term in self._LEXICON)

    def test_no_ungated_maturity_overclaim_in_src(self):
        src_root = REPO_ROOT / "src" / "rabbit"
        allowed = {src_root / "config" / "claim_gates.py"}
        violations: list[str] = []
        for path in src_root.rglob("*.py"):
            if path in allowed:
                continue
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                low = self._norm(line)
                if any(term in low for term in self._LEXICON) and not self._is_hedge(low):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
        assert not violations, (
            "Ungated maturity over-claim(s) in src/rabbit — gate the claim via "
            "config/claim_gates.py (a Forbidden→Permitted ClaimGate) or qualify it as an honest "
            "hedge ('not publication-grade'):\n" + "\n".join(violations))


@pytest.mark.production
@pytest.mark.release_smoke
class TestNoValidatedCollisionPhysicsClaim:
    """The B4 collision twins are parity-locked to a CALIBRATED RTA model — parity is engineering
    fidelity, NOT physical validation of the collision integral (external audit BD601 AUDIT2-3).
    Pins (a) the twin modules carry that caveat, and (b) nothing on the B4 surface claims the
    collision PHYSICS is validated."""

    _MODULES = (
        "src/rabbit/jax/collision_operator_jax.py",
        "src/rabbit/jax/teff_collision_bridge_jax.py",
    )
    _B4_RETIRED_SURFACE = (
        "src/rabbit/jax/driver_typeI_char.py",
        "tests/test_b4_jax_char_collision_wiring.py",
    )
    _B4_SURFACE = _MODULES + (
        "tests/test_physical_collision_operator_reference.py",
        "tests/test_jax_collision_operator_parity.py",
        "tests/test_jax_teff_collision_bridge_parity.py",
        "tests/test_teff_bridge_tangency_degenerate.py",
        # B4-PR4: the live JAX characteristic collisional wiring surface. (The
        # meta honesty test ``test_b4_collisional_char_reference_unsound.py`` is
        # deliberately NOT scanned here -- like ``test_claim_gates.py`` itself, it
        # reasons ABOUT the "validated collision physics" phrase and would
        # self-trip; it carries its own scoped scan with negation handling.)
        "docs/audit/B4_PR4_collisional_char_reference_unsound_2026-07-01.md",
    )
    # An AFFIRMATIVE claim co-locating "validat*" with "collision" and "physic*" on one line --
    # robust to word order, hyphenation, and verb/adjective forms ("validated collision physics",
    # "validation of the collision physics", "validate collision-physics", ...). Negation caveats
    # are excluded so the honest "NOT a physical validation of the collision physics" passes.
    _VALIDATE_RE = re.compile(r"validat\w*", re.IGNORECASE)
    _NEGATIONS = ("not a physical validation", "not physically validated", "not validated",
                  "parity is not", "not a validation", "not physics validation",
                  "not a physics-validated", "not physics-validated", "not externally validated")

    def _claims_validated_collision_physics(self, line: str) -> bool:
        low = line.lower()
        if "collision" not in low or "physic" not in low:
            return False
        if not self._VALIDATE_RE.search(low):
            return False
        return not any(neg in low for neg in self._NEGATIONS)

    def test_b4_twins_carry_the_calibrated_rta_caveat(self):
        for rel in self._MODULES:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8").lower()
            assert "calibrated" in text and "not a physical validation" in text, (
                f"{rel}: missing the calibrated-RTA / 'not a physical validation' caveat — parity to "
                "the calibrated RTA model must never be presented as validated collision physics")

    def test_no_validated_collision_physics_claim_on_b4_surface(self):
        violations: list[str] = []
        for rel in self._B4_SURFACE:
            for i, line in enumerate((REPO_ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
                if self._claims_validated_collision_physics(line):
                    violations.append(f"{rel}:{i}: {line.strip()}")
        assert not violations, (
            "B4 surface claims validated collision PHYSICS (it is only parity to a calibrated RTA "
            "model):\n" + "\n".join(violations))

    def test_retired_b4_endpoint_surfaces_remain_absent(self):
        restored = sorted(rel for rel in self._B4_RETIRED_SURFACE if (REPO_ROOT / rel).exists())
        assert not restored, (
            "F06-retired B4 endpoint surfaces were restored without renewed collision-claim "
            "hygiene review: " + ", ".join(restored)
        )


@pytest.mark.production
class TestGateLookup:
    def test_gates_by_id_round_trip(self):
        d = gates_by_id()
        assert isinstance(d, dict)
        for g in ALL_GATES:
            assert d[g.claim_id] is g

    def test_claim_gate_is_frozen(self):
        g = ALL_GATES[0]
        with pytest.raises(Exception):
            g.claim_id = "different"  # type: ignore[misc]
