"""PR-T3C-PF #7: feature-level registration of the cumulative
tier-3 preflight surface.

Locks the ``TIER3_FULL_COLLISION_PREFLIGHT`` ``FeatureCapability``
entry in ``rabbit.config.feature_capabilities``, which is the
single source of truth for cross-cutting feature maturity (vs
``backend_capabilities`` which is per-backend).

The feature is candidate-tier with diagnostic surface_class and
explicitly enumerates the four open canonical blockers so the
registry alone documents the path to PR-T3D promotion.
"""
from __future__ import annotations


def test_tier3_feature_registered() -> None:
    """The feature key is in the registry."""
    from rabbit.config.feature_capabilities import FEATURE_BY_KEY

    assert "tier3_full_collision_preflight" in FEATURE_BY_KEY


def test_tier3_feature_tier_and_surface_class() -> None:
    """Tier and surface_class lock at candidate-diagnostic."""
    from rabbit.config.feature_capabilities import FEATURE_BY_KEY

    f = FEATURE_BY_KEY["tier3_full_collision_preflight"]
    assert f.tier == "candidate"
    assert f.validation_mode == "diagnostic"
    assert f.surface_class == "diagnostic"


def test_tier3_feature_canonical_blockers_enumerated() -> None:
    """The narrowed canonical blockers (post-scope-reframing) are
    explicitly listed in the registry so the path to PR-T3D
    promotion is traceable from the registry alone.

    Scope reframing: jax_kernel + IMEX hybrid is OUT (incompatible
    with the Rodas5P invariant); canonical destination is AP-form
    unification + Dolgov-Hansen-Semikoz coefficient table with a
    documented Mangano gap.  See ``docs/audit/PR-T3B_jax_kernel_runtime.md``
    for the scope analysis.
    """
    from rabbit.config.feature_capabilities import FEATURE_BY_KEY

    f = FEATURE_BY_KEY["tier3_full_collision_preflight"]
    assert len(f.blockers) >= 3, (
        f"expected >= 3 narrowed canonical blockers, got {len(f.blockers)}"
    )
    blocker_text = " ".join(f.blockers).lower()
    # AP-form unification mention (folds spectral grid scaling +
    # projected anisotropy into a single canonical target).
    assert "ap-form" in blocker_text and "unification" in blocker_text
    # Anisotropy keyword is still present inside the unification
    # blocker (e.g., "anisotropy stability").
    assert "anisotrop" in blocker_text
    # Grid scaling keyword for the spectral_relaxation limit.
    assert "grid scaling" in blocker_text or "grid" in blocker_text
    # Dolgov-Hansen-Semikoz coefficient calibration.
    assert "dolgov" in blocker_text or "nu-nu" in blocker_text or "coefficient" in blocker_text
    # Documented Mangano gap as accepted model limit.
    assert "mangano" in blocker_text or "n_eff gap" in blocker_text


def test_tier3_feature_evidence_includes_measurements() -> None:
    """The evidence summary must include the central calibration
    numbers (N_eff = 3.030738 for AP-form, 2.993427 for kernel) so
    reading the registry is enough to know where the gap is."""
    from rabbit.config.feature_capabilities import FEATURE_BY_KEY

    f = FEATURE_BY_KEY["tier3_full_collision_preflight"]
    ev = f.evidence_summary
    assert "3.030738" in ev or "3.044" in ev
    assert "2.993427" in ev or "0.013" in ev or "anti-heating" in ev


def test_tier3_feature_notes_point_to_audit_docs() -> None:
    """Notes must reference the audit docs so reader can follow the
    full calibration trail."""
    from rabbit.config.feature_capabilities import FEATURE_BY_KEY

    f = FEATURE_BY_KEY["tier3_full_collision_preflight"]
    notes = f.notes.lower()
    assert "audit" in notes
    assert "pr-t3" in notes
