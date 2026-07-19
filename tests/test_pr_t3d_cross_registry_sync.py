"""PR-T3D-PF #7: cross-registry consistency between
``backend_capabilities`` and ``feature_capabilities`` for the
tier-3 preflight surface.

The tier-3 preflight surface is registered at two registry
levels:

* ``rabbit.config.backend_capabilities.JAX_TYPEI_FULL_BOLTZMANN_TIER3_PREFLIGHT``
  (key: ``"jax_typeI_full_boltzmann_tier3_preflight"``) —
  per-backend dispatch identity.

* ``rabbit.config.feature_capabilities.TIER3_FULL_COLLISION_PREFLIGHT``
  (key: ``"tier3_full_collision_preflight"``) — cross-cutting
  feature maturity tag.

This module locks the consistency contract between the two so a
silent drift in one registry surfaces in the other.
"""
from __future__ import annotations


def test_backend_and_feature_both_registered() -> None:
    """Both registries carry the tier-3 preflight surface entries.

    F06 retains only the full-Boltzmann component/preflight metadata.
    The retired AP-unified endpoint candidate is deliberately absent.
    """
    from rabbit.config.backend_capabilities import CAPABILITY_BY_KEY
    from rabbit.config.feature_capabilities import FEATURE_BY_KEY

    assert "jax_typeI_full_boltzmann_tier3_preflight" in CAPABILITY_BY_KEY
    assert "jax_typeI_ap_unified_tier3_candidate" not in CAPABILITY_BY_KEY
    assert "tier3_full_collision_preflight" in FEATURE_BY_KEY


def test_full_boltzmann_component_metadata_is_noncanonical() -> None:
    """The retained full-Boltzmann entry is component evidence only."""
    from rabbit.config.backend_capabilities import CAPABILITY_BY_KEY

    cap = CAPABILITY_BY_KEY["jax_typeI_full_boltzmann_tier3_preflight"]
    assert cap.tier == "candidate"
    assert cap.surface_class == "candidate"
    assert cap.collision_scope_contract == "jax_kernel_plus_diagonal_nu_nu_skeleton_preflight_v1"
    notes = cap.notes.lower()
    assert "not promoted to canonical" in notes
    assert "no entry in capability_by_backend" in notes


def test_tier_alignment_across_registries() -> None:
    """Both registries declare the surface as ``candidate`` tier."""
    from rabbit.config.backend_capabilities import CAPABILITY_BY_KEY
    from rabbit.config.feature_capabilities import FEATURE_BY_KEY

    bk = CAPABILITY_BY_KEY["jax_typeI_full_boltzmann_tier3_preflight"]
    ft = FEATURE_BY_KEY["tier3_full_collision_preflight"]

    assert bk.tier == "candidate", (
        f"backend tier-3 preflight tier drifted: {bk.tier}"
    )
    assert ft.tier == "candidate", (
        f"feature tier-3 preflight tier drifted: {ft.tier}"
    )


def test_neither_in_canonical_dispatch() -> None:
    """Neither registry's tier-3 preflight entry is exposed via the
    public ``CAPABILITY_BY_BACKEND`` dispatch table.  This is the
    explicit no-public-canonical-promotion contract: the surface
    is reachable only through the private ``collision_mode``
    runtime knob on the JAX full-Boltzmann driver."""
    from rabbit.config.backend_capabilities import (
        CAPABILITY_BY_BACKEND,
        CAPABILITY_BY_KEY,
    )

    bk = CAPABILITY_BY_KEY["jax_typeI_full_boltzmann_tier3_preflight"]
    # The dispatch table maps backend strings -> capability objects;
    # neither the key nor the capability object should appear there.
    assert bk.key not in [c.key for c in CAPABILITY_BY_BACKEND.values()], (
        f"tier-3 preflight backend appeared in CAPABILITY_BY_BACKEND "
        f"dispatch table -- silent canonical promotion?"
    )


def test_feature_blockers_reference_canonical_targets() -> None:
    """The feature's ``blockers`` field enumerates the **narrowed**
    canonical-PR-T3{B,C,D} target labels (post-scope-reframing) so
    a registry reader knows exactly what canonical work is missing.

    Scope reframing: jax_kernel + IMEX hybrid is OUT (incompatible
    with the Rodas5P invariant); canonical destination is AP-form
    unification + Dolgov-Hansen-Semikoz coefficient table with a
    documented Mangano gap.  Stiff/IMEX/remap keywords are
    deliberately ABSENT — that path is closed."""
    from rabbit.config.feature_capabilities import FEATURE_BY_KEY

    ft = FEATURE_BY_KEY["tier3_full_collision_preflight"]
    blockers_text = " ".join(ft.blockers).lower()

    # The three narrowed canonical blockers are explicitly mentioned:
    # 1. AP-form unification (folds spectral grid + projected
    #    anisotropy into a single canonical target).
    assert "ap-form" in blockers_text and "unification" in blockers_text
    # 2. Anisotropy keyword inside the unification blocker.
    assert "anisotrop" in blockers_text
    # 3. Grid scaling for spectral_relaxation.
    assert "scale" in blockers_text or "grid" in blockers_text
    # 4. Dolgov-Hansen-Semikoz coefficient calibration.
    assert "dolgov" in blockers_text or "coefficient" in blockers_text
    # 5. Mangano gap accepted as documented model limit.
    assert "mangano" in blockers_text or "n_eff" in blockers_text

    # Stiff/IMEX/remap path is OUT OF SCOPE — these keywords must NOT
    # appear in blockers (they have been moved to the canonical-NOT-
    # PURSUED section of the notes field instead).
    assert "imex" not in blockers_text, (
        "IMEX should not appear in blockers post-scope-reframing"
    )
    assert "remap" not in blockers_text, (
        "q-remap should not appear in blockers post-scope-reframing"
    )


def test_evidence_summary_attribution_per_mode() -> None:
    """The feature's ``evidence_summary`` attributes findings to
    specific collision modes (spectral_relaxation,
    projected_physical, jax_kernel) so a reader of the registry
    can map the open work to the right preflight mode without
    reading audit docs."""
    from rabbit.config.feature_capabilities import FEATURE_BY_KEY

    ft = FEATURE_BY_KEY["tier3_full_collision_preflight"]
    ev = ft.evidence_summary.lower()
    assert "spectral_relaxation" in ev or "spectral relaxation" in ev
    assert "projected_physical" in ev or "projected physical" in ev
    assert "jax_kernel" in ev or "jax-kernel" in ev or "kernel" in ev
