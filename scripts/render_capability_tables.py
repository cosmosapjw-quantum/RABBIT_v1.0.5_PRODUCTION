#!/usr/bin/env python3
"""Generate capability tables from registries and apply to docs.

Usage:
    python scripts/render_capability_tables.py          # print only
    python scripts/render_capability_tables.py --apply  # update all docs

Generated blocks (8 total across 4 documents):
    README.md:                    BACKEND_TABLE, CORE_SUMMARY
    SUPPORTED_CAPABILITIES.md:    BACKEND_TABLE, FEATURE_TABLE
    STATUS.md:                    TIER_SUMMARY
    PROMOTION_GATES.md:           PROMOTION_STATUS
    docs/BACKEND_CAPABILITY_MATRIX.md: full overwrite
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
sys.path.insert(0, "src")

from rabbit.config.backend_capabilities import (
    ACTIVE_CANONICAL_BACKENDS,
    CAPABILITY_BY_BACKEND,
    CAPABILITY_BY_KEY,
    QUARANTINED_BACKENDS,
)
from rabbit.config.claim_gates import ALL_GATES
from rabbit.config.feature_capabilities import FEATURE_BY_KEY, FEATURE_TIERS


def _current_status_counts():
    """Read the sync-count block so render --apply remains a no-op after sync."""
    fallback = {
        "total": 1390,
        "gold": 37,
        "smoke": 107,
        "production_total": 144,
        "production_not_slow": 144,
        "production_slow": 0,
    }
    try:
        text = open("STATUS.md").read()
    except FileNotFoundError:
        return fallback
    m = re.search(
        r"## Tests \((\d+) collected; overlapping marker subsets: "
        r"\*\*(\d+) gold\*\*, (\d+) smoke; `@production` (\d+) total, "
        r"(\d+) production-and-not-slow\)",
        text,
    )
    if m:
        total, gold, smoke, production_total, production_not_slow = (
            int(v) for v in m.groups()
        )
        return {
            "total": total,
            "gold": gold,
            "smoke": smoke,
            "production_total": production_total,
            "production_not_slow": production_not_slow,
            "production_slow": production_total - production_not_slow,
        }

    # One-cycle migration support for documents generated before the count
    # contract distinguished the required fast lane.  sync_test_counts.py will
    # replace this provisional split with live collection counts.
    legacy = re.search(
        r"## Tests \((\d+) collected; \*\*(\d+) gold\*\* BBN physics gates, "
        r"(\d+) smoke, (\d+) production total\)",
        text,
    )
    if legacy:
        total, gold, smoke, production_total = (int(v) for v in legacy.groups())
        return {
            "total": total,
            "gold": gold,
            "smoke": smoke,
            "production_total": production_total,
            "production_not_slow": production_total,
            "production_slow": 0,
        }
    return fallback


def _replace_block(filepath, begin, end, content):
    text = open(filepath).read()
    begin_esc = re.escape(begin.rstrip(" ->"))
    pattern = re.compile(rf'({begin_esc}[^\n]*?)\n(.*?)\n({re.escape(end)})', re.DOTALL)
    m = pattern.search(text)
    if not m:
        return None
    updated = text[:m.start()] + m.group(1) + "\n" + content + "\n" + m.group(3) + text[m.end():]
    if updated == text:
        return False
    open(filepath, "w").write(updated)
    return True


# ═══════════════════════════════════════════════════
# Renderers — each produces markdown from registries
# ═══════════════════════════════════════════════════

def render_readme_backend_table():
    """README backend table — fully registry-driven."""
    dispatch_order = [
        "auto", "scipy", "jax", "jax_characteristic",
        "jax_characteristic_tier2", "jax_characteristic_nonlrs", "jax_advanced",
        "jax_ap_unified_tier3",
        "jax_classA", "jax_classB", "jax_tilted", "jax_tilted_full_coupled",
    ]
    lines = ["| Backend | Tier | CL | Surface class |", "|---|---|---|---|"]
    for name in dispatch_order:
        if name in CAPABILITY_BY_BACKEND:
            c = CAPABILITY_BY_BACKEND[name]
            historical = name in QUARANTINED_BACKENDS
            bold_tier = (
                f"**{c.tier}** (historical)"
                if c.tier == "canonical" and historical
                else f"**{c.tier}**" if c.tier == "canonical" else c.tier
            )
            sc = "frozen oracle" if historical else c.effective_surface_class.replace("_", "-")
            cl = f"0–{c.max_correction_level}" if c.max_correction_level > 0 else "0"
            lines.append(f"| `{name}` | {bold_tier} | {cl} | {sc} |")
    return "\n".join(lines)


def render_supported_backend_table():
    """SUPPORTED backend table — fully registry-driven."""
    dispatch_order = [
        "auto", "scipy", "jax", "jax_characteristic",
        "jax_characteristic_tier2", "jax_characteristic_nonlrs", "jax_advanced",
        "jax_ap_unified_tier3",
        "jax_classA", "jax_classB", "jax_tilted", "jax_tilted_full_coupled",
    ]
    lines = ["| Backend | Tier | Surface class | Description |", "|---|---|---|---|"]
    for name in dispatch_order:
        if name in CAPABILITY_BY_BACKEND:
            c = CAPABILITY_BY_BACKEND[name]
            historical = name in QUARANTINED_BACKENDS
            bold_tier = (
                f"**{c.tier}** (historical)"
                if c.tier == "canonical" and historical
                else f"**{c.tier}**" if c.tier == "canonical" else c.tier
            )
            sc = "frozen oracle" if historical else c.effective_surface_class.replace("_", "-")
            # Description from registry notes (first sentence)
            desc = c.notes.split(". ")[0] if c.notes else c.physics_scope
            lines.append(f"| `{name}` | {bold_tier} | {sc} | {desc} |")
    return "\n".join(lines)


def render_feature_table():
    groups = {}
    for key, feat in FEATURE_BY_KEY.items():
        groups.setdefault(feat.surface_class, []).append(feat)
    lines = []
    for sc in ["canonical", "candidate_strong", "candidate_layered", "diagnostic", "exploratory"]:
        if sc not in groups:
            continue
        label = sc.replace("_", "-")
        lines.append(f"### {label.title()}")
        lines.append("")
        lines.append("| Feature | Evidence | Blockers |")
        lines.append("|---|---|---|")
        for f in groups[sc]:
            blockers = "; ".join(f.blockers) if f.blockers else "—"
            ev = f.evidence_summary[:70] + ("..." if len(f.evidence_summary) > 70 else "")
            lines.append(f"| {f.name} | {ev} | {blockers} |")
        lines.append("")
    return "\n".join(lines)


def render_tier_summary():
    counts = {}
    for k, c in CAPABILITY_BY_KEY.items():
        counts[c.tier] = counts.get(c.tier, 0) + 1
    labels = {
        "canonical": "Historical registry maturity; active runtime authority is separate",
        "candidate": "End-to-end functional, parity-checked, opt-in",
        "substrate": "Module-level validated, not wired end-to-end",
    }
    lines = ["| Tier | Meaning | Count |", "|---|---|---|"]
    for tier in ["canonical", "candidate", "substrate"]:
        lines.append(f"| **{tier}** | {labels[tier]} | {counts.get(tier, 0)} |")
    return "\n".join(lines)


def render_canonical_core():
    """Canonical Core summary table from feature registry."""
    canonical = [f for f in FEATURE_BY_KEY.values()
                 if f.tier == "canonical" or (f.validation_mode == "full" and f.surface_class == "canonical")]

    lines = [
        "## Active Runtime Core and Frozen Oracles (NOT publication validation)",
        "",
        "| Capability | Backend | Regime |",
        "|---|---|---|",
    ]
    # Map features to presentation rows
    for f in canonical:
        if "flrw" in f.key:
            lines.append("| FLRW BBN baseline | SciPy Radau | CL0-CL3 |")
        elif "typeI" in f.key:
            lines.append("| Type I anisotropic BBN | SciPy reference | historical regression envelope; publication validation open |")
        elif "weak" in f.key:
            lines.append("| Weak rates (Born-Sirlin) | Python | CL0-CL2 |")
        elif "nuclear" in f.key:
            lines.append("| 9-species network | PRIMAT AC2024 | full + backbone (31/12) |")

    # Preserve the JAX evidence surface without presenting it as active authority.
    jax = CAPABILITY_BY_BACKEND.get("jax")
    if jax and jax.tier == "canonical":
        lines.append("| Frozen JAX Type I oracle (live weak) | JAX Rodas5P | historical scoped parity only; no future/runtime authority |")

    lines.extend([
        "",
        "## Feature Registry (by surface class)",
        "",
        "Generated from `feature_capabilities.py`.",
    ])
    return "\n".join(lines)


def render_sc_header():
    """SUPPORTED_CAPABILITIES header: SSOT, governance scope, maturity tiers."""
    # Count render targets dynamically
    n_targets = len([line for line in text_of_apply_targets if "BEGIN:" in line or "full overwrite" in line]) if False else 12

    tiers = {
        "canonical": ("Runtime-regression reference path", "Runtime gold; publication gate separate"),
        "candidate": ("Functional, partially validated", "Component + dispatch + partial BBN"),
        "substrate": ("Module-level validated, not wired end-to-end", "Import/component tests only"),
    }

    lines = [
        "# RABBIT v1.0.0: Supported Capabilities",
        "",
        "**Single sources of truth** (two registries):",
        "- **Backend dispatch**: `src/rabbit/config/backend_capabilities.py`",
        "- **Feature maturity**: `src/rabbit/config/feature_capabilities.py`",
        "",
        "All maturity claims in this file MUST match the `tier` field in code.",
        "",
        "**Generated scope**: This entire document is registry-generated via "
        "`render_capability_tables.py --apply` and self-heals on each release. "
        "If content disagrees with code, the code is correct and this file is wrong.",
        "",
        "## Maturity Tiers",
        "",
        "| Tier | Meaning | Gate requirement |",
        "|---|---|---|",
    ]
    for tier, (meaning, gate) in tiers.items():
        lines.append(f"| **{tier}** | {meaning} | {gate} |")

    lines.append("")
    lines.append("## Backend Maturity (from `backend_capabilities.py`)")
    return "\n".join(lines)


def render_claims():
    """Forbidden and Permitted Claims from feature registry."""
    lines = ["## Forbidden Claims", ""]

    for gate in ALL_GATES:
        lines.append(f"- {gate.forbidden_text}")

    lines.extend([
        "",
        "## Permitted Claims",
        "",
        "Only the following runtime-scoped statements are permitted here. None "
        "is publication validation or public-production authority.",
        "",
    ])

    # Canonical core
    lines.append('- "Type I runtime regression surface (finite-shear publication domain NOT VALIDATED; B-01--B-06 open)"')
    lines.append('- "PRIMAT-backed network and FLRW gold are runtime regressions, not matched publication anchors"')
    lines.append('- "JAX Type I parity is frozen historical regression evidence; PUB-02C grants no repeated-run authority; G-01R remains open"')

    # Feature-specific permitted: production-locked features with scope
    for key, feat in FEATURE_BY_KEY.items():
        if feat.validation_mode in ("full", "diagnostic", "candidate") and not feat.blockers:
            if feat.surface_class in ("canonical",):
                continue  # already covered above
            lines.append(
                f'- "{feat.name} documented runtime/diagnostic scope '
                f'({feat.short_summary}); not publication evidence"'
            )

    # Exploratory explicitly
    lines.append('- "Bayes factor / evidence computation remains exploratory"')

    return "\n".join(lines)


def render_readme_header():
    """README title + tagline."""
    return """# RABBIT v1.0.0

Type I BBN research runtime with publication validation in progress."""


def render_readme_backends_header():
    """Section header before backend table."""
    return "## Backends (from `backend_capabilities.py`)"


def render_readme_quickstart():
    """Quick start code examples from canonical backends."""
    lines = [
        "## Quick start",
        "",
        "```python",
        "from rabbit.inference.forward_likelihood import canonical_forward_solver",
        "",
        "# Standard BBN (FLRW baseline)",
        'pred = canonical_forward_solver(Sigma_H=0.0, backend="auto")',
        'print(f"Yp = {pred.Yp:.6f}, D/H = {pred.DH:.2e}")',
        "",
        "# Type I anisotropic BBN",
        'pred = canonical_forward_solver(Sigma_H=0.05, backend="auto")',
        "",
        "# JAX endpoint dispatch names are retired. Frozen low-level JAX",
        "# component oracles remain non-dispatchable metadata only.",
        "```",
    ]
    return "\n".join(lines)


def render_readme_footer():
    """README build-env note."""
    return """**Note**: Counts above are **build-environment metadata**, not portable invariants.
Optional dependencies (JAX, BlackJAX) affect test collection.
Run `make sync-counts` to update for your environment.
See `docs/RENDER_PROVENANCE.json` for exact build provenance."""


def render_identity():
    """README identity paragraph from registries."""
    canonical_backends = [c for c in CAPABILITY_BY_KEY.values() if c.tier == "canonical"]
    n_canonical = len(canonical_backends)
    # Find parity info from JAX canonical
    jax = CAPABILITY_BY_BACKEND.get("jax")
    parity = "0.0006%" if jax and jax.tier == "canonical" else "candidate"
    n_candidate = sum(1 for c in CAPABILITY_BY_KEY.values() if c.tier == "candidate")
    return (
        "RABBIT provides a **canonical Type I BBN core** in the registry-runtime "
        "sense only, plus a candidate/substrate perimeter. Registry maturity, gold regression, "
        f"and the historical SciPy/JAX {parity} parity result are not publication "
        "validation; the publication PR programme remains authoritative."
    )


def render_core_summary():
    """README canonical core + candidate perimeter summary."""
    # Canonical features
    canonical = [f for f in FEATURE_TIERS.get("canonical", [])]
    # Group candidates by surface_class
    groups = {}
    for f in FEATURE_TIERS.get("candidate", []):
        groups.setdefault(f.surface_class, []).append(f)

    lines = [
        "## Active Runtime Core (NOT publication validation)",
        "",
    ]
    for f in canonical:
        lines.append(f"- **{f.name}**: {f.short_summary}")
    # Add canonical backends
    canon_backends = [
        f"`{n}`" for n, c in CAPABILITY_BY_BACKEND.items()
        if c.tier == "canonical" and n in ACTIVE_CANONICAL_BACKENDS and n != "auto"
    ]
    lines.append(f"- **Backends**: {' and '.join(canon_backends)}")
    teff = FEATURE_BY_KEY["teff_spectral"]
    lines.extend([
        "",
        "## Frozen Legacy Diagnostics",
        "",
        f"- **{teff.name}**: {teff.short_summary}",
    ])
    lines.append("")
    lines.append("## Candidate Features (code works, partial validation, NOT publication-locked)")
    lines.append("")

    sc_labels = {
        "candidate_strong": ("Candidate-strong", "BBN-verified in documented regime"),
        "candidate_layered": ("Candidate-layered", "multi-layer validation, partial BBN"),
        "diagnostic": ("Diagnostic", "component-level only"),
        "exploratory": ("Exploratory", "component-level only"),
    }
    for sc in ["candidate_strong", "candidate_layered"]:
        if sc in groups:
            label, desc = sc_labels[sc]
            lines.append(f"**{label}** ({desc}):")
            for f in groups[sc]:
                if f.blockers:
                    lines.append(f"- {f.name} ({f.blockers[0]})")
                else:
                    lines.append(f"- {f.name} (documented-scope: {f.short_summary})")
            lines.append("")

    # Combine diagnostic + exploratory
    diag_exp = groups.get("diagnostic", []) + groups.get("exploratory", [])
    if diag_exp:
        lines.append("**Diagnostic / Exploratory** (component-level only):")
        for f in diag_exp:
            mode = f.validation_mode
            if f.blockers:
                lines.append(f"- {f.name} ({mode}; {f.blockers[0]})")
            else:
                lines.append(f"- {f.name} ({mode}: {f.short_summary})")
    return "\n".join(lines)


def render_promotion_status():
    """PROMOTION_GATES current status from registries."""
    # Backend canonical entries
    lines = [
        "| Feature | Code tier | Surface class | Next blocker |",
        "|---|---|---|---|",
    ]
    # SciPy reference
    lines.append("| SciPy Type I | **canonical** | canonical | — (reference) |")
    # JAX retains its historical tier but is frozen and quarantined.
    jax = CAPABILITY_BY_BACKEND.get("jax")
    if jax and jax.tier == "canonical":
        lines.append(
            f"| JAX Type I (`{jax.key}`) | frozen oracle | historical canonical | "
            "Quarantined compatibility/parity only; PUB-02C grants no repeated-run authority |"
        )
    # Feature candidates
    for key, feat in FEATURE_BY_KEY.items():
        if feat.tier == "candidate":
            sc = feat.surface_class.replace("_", "-")
            blocker = feat.blockers[0] if feat.blockers else "—"
            lines.append(f"| {feat.name} | candidate | {sc} | {blocker} |")
    return "\n".join(lines)


def render_pg_header():
    """PROMOTION_GATES header: authority, governance scope, gate requirements."""
    return """# RABBIT Feature Promotion Gates

**Authority**: Two registries are the single sources of truth.
- `backend_capabilities.py` — backend dispatch tiers
- `feature_capabilities.py` — feature maturity tiers

This document is registry-generated via `render_capability_tables.py --apply`.

**Generated scope**: This entire document is registry-generated and self-heals on each release.
A feature is "promoted" ONLY when its tier changes in the authoritative registry.
Registry promotion does not establish a publication claim; the publication PR
programme and claim ledger are separate authorities.

## Gate Requirements (canonical promotion)

To move from `candidate` → `canonical`, ALL of these must pass:

1. **End-to-end BBN gold lock** — actual Yp match, not just no-TypeError
2. **Limit recovery** — zero/null/isotropic at BBN level
3. **Cross-backend parity** — same physics gives <5% Yp agreement
4. **Convergence envelope** — N_q, tolerance sensitivity documented
5. **Single source of truth** — all 5 documents agree on tier
6. **Registry tier = "canonical"**

## Current Promotion Status"""


def render_pg_body():
    """PROMOTION_GATES body: gate progress scale, rules, decision records."""
    # JAX runtime decision record from backend registry
    jax = CAPABILITY_BY_BACKEND.get("jax")
    jax_key = jax.key if jax is not None else "UNREGISTERED"
    jax_status = (
        "Frozen compatibility/parity oracle; historical registry tier retained."
        if jax and jax.tier == "canonical"
        else "Frozen unregistered oracle."
    )

    return f"""## What "gate progress" means

- **6/6**: canonical-ready (all gates pass with BBN evidence)
- **5/6**: near-canonical (one blocker remains)
- **4/6**: strong candidate (BBN run succeeded, some gates missing)
- **3/6**: candidate (component-level + dispatch verified)
- **2/6**: early candidate (config + docs only)
- **1/6**: substrate scaffold

## Rules

1. **Code tier is truth** — registries are authoritative; documents must not claim higher
2. **No "PROMOTED" label** until registry tier changes
3. **Production gates require BBN output**, not just no-TypeError
4. **README claims must lag** behind code by one validation cycle

## JAX Type I Runtime Decision Record

- **Status**: {jax_status}
- **Current explicit key**: `{jax_key}`; `jax_typeI_liveweak_cl0` remains candidate
- **Evidence ceiling**: historical parity and BBN gold are runtime regressions only
- **Governance**: PUB-02C grants no repeated-run authority; Rust is SPECIFIED only; G-01R remains open

## Next Promotion Queue"""


def render_pg_footer():
    """PROMOTION_GATES trailing sentence."""
    return "No candidate should be promoted without explicit gate criteria and decision record."


def render_next_queue():
    """Render historical candidate inventory without authorizing new work."""
    candidates = [(k, f) for k, f in FEATURE_BY_KEY.items() if f.tier == "candidate"]
    # Sort by surface_class priority
    priority = {"candidate_strong": 0, "candidate_layered": 1, "diagnostic": 2, "exploratory": 3}
    candidates.sort(key=lambda x: priority.get(x[1].surface_class, 99))

    lines = [
        "Historical candidate inventory (not an active implementation queue):",
        "",
        "JAX-backed and legacy entries are frozen. Active work order comes only "
        "from the Rust-first publication plan.",
        "",
        "| # | Feature | Surface class | Historical blocker |",
        "|---|---|---|---|",
    ]
    for i, (key, feat) in enumerate(candidates, 1):
        sc = feat.surface_class.replace("_", "-")
        blocker = "; ".join(feat.blockers) if feat.blockers else "—"
        lines.append(f"| {i} | {feat.name} | {sc} | {blocker} |")
    return "\n".join(lines)


def render_classB_layers():
    """Class B layered validation table from registry."""
    f = FEATURE_BY_KEY.get("classB_bbn")
    if not f or not f.layered_scope:
        return "No Class B layered scope in registry."
    ls = f.layered_scope
    layers = [
        ("Geometry substrate", ls["geometry"], ", ".join(ls["geometry_types"]), "Unit tests"),
        ("Family envelope", ls["family_envelope"], ", ".join(ls["envelope_types"]), "Representative candidate envelope"),
        ("Full-BBN smoke", ls["bbn_smoke"], ", ".join(ls["smoke_types"]), "Physical range + success"),
    ]
    if ls.get("gold_locked", 0) > 0:
        gold_type = ls.get("gold_type", "TYPE_UNKNOWN").replace("TYPE_", "") + " only"
        evidence = ls.get("gold_evidence")
        if not evidence:
            gold_yp = ls.get("gold_Yp", "n/a")
            gold_dyp = ls.get("gold_DYp_A", 0.0)
            evidence = f"Yp={gold_yp}, ΔYp(A)=+{gold_dyp:.1e}"
        layers.append(("BBN-verified gold", ls["gold_locked"], gold_type, evidence))
    else:
        layers.append(("BBN-verified gold", 0, "none", "retired after geometry/initial-data audit"))
    lines = [
        "### Class B validation layers (registry-generated)",
        "",
        "| Layer | Count | Types | Evidence |",
        "|---|---|---|---|",
    ]
    for name, count, types, evidence in layers:
        lines.append(f"| {name} | {count} types | {types} | {evidence} |")
    return "\n".join(lines)


def render_status_header():
    """STATUS.md header."""
    return """# RABBIT — Status (v1.0.0)

Single source of truth for runtime maturity claims, not publication validation.
Two registries: `backend_capabilities.py` (dispatch) + `feature_capabilities.py` (features).
This document is registry-generated via `render_capability_tables.py --apply`.

## Capability tiers"""


def render_status_detail():
    """STATUS body: dispatch, runtime regressions, candidate scope, and lanes."""
    lines = []

    # Dispatch table
    lines.append("## Dispatch table")
    lines.append("")
    lines.append("```")
    lines.append("canonical_forward_solver(backend=...)")
    dispatch = {
        "auto": (CAPABILITY_BY_BACKEND["auto"].key, "bounded canonical default"),
        "scipy": ("scipy_typeI_reference", "reference / fallback"),
    }
    # Add from backend registry
    for name, cap in CAPABILITY_BY_BACKEND.items():
        if name in ("auto", "scipy"):
            continue
        dispatch[name] = (cap.key, cap.physics_scope)
    for name, (key, note) in dispatch.items():
        pad = max(1, 16 - len(f'  "{name}"  '))
        note_str = f"({note})" if note else ""
        lines.append(f'  "{name}"{" "*pad}→ {key}  {note_str}')
    lines.append("```")
    lines.append("")

    # Transport
    lines.append("## Transport")
    lines.append("")
    auto_capability = CAPABILITY_BY_BACKEND["auto"]
    lines.append(
        "**Canonical default**: `backend=\"auto\"` resolves to "
        f"`{auto_capability.key}` ({auto_capability.backend}, "
        f"{auto_capability.transport_scope_contract})."
    )
    lines.append("**JAX endpoint dispatch**: retired at F06; frozen low-level component oracles remain non-dispatchable metadata only.")
    lines.append("**Candidate**: Extended 1+3 PSTF ℓ_max=8. Complex eigenvalue λ = −1/2 ± iω confirmed.")
    lines.append("")

    # Feature detail sections from registry
    for key, feat in FEATURE_BY_KEY.items():
        if feat.surface_class == "canonical":
            continue  # covered by TIER_SUMMARY + STATUS_BACKENDS
        sc = feat.surface_class.replace("_", "-")
        lines.append(f"## {feat.name} ({sc})")
        lines.append("")
        lines.append(f"**Evidence**: {feat.evidence_summary}")
        if feat.short_summary:
            lines.append(f"**Summary**: {feat.short_summary}")
        if feat.notes:
            lines.append(f"**Notes**: {feat.notes}")
        if feat.blockers:
            lines.append(f"**Blockers**: {'; '.join(feat.blockers)}")
        if feat.layered_scope:
            ls = feat.layered_scope
            lines.append("")
            lines.append("**Validation layers**:")
            lines.append(f"- Geometry substrate: {ls['geometry']} types ({', '.join(ls['geometry_types'])})")
            lines.append(f"- Family envelope: {ls['family_envelope']} types ({', '.join(ls['envelope_types'])})")
            lines.append(f"- Full-BBN smoke: {ls['bbn_smoke']} types ({', '.join(ls['smoke_types'])})")
            if ls.get('gold_locked', 0) > 0:
                lines.append(f"- BBN-verified gold: {ls['gold_locked']} type ({ls.get('gold_type', 'TYPE_UNKNOWN')})")
            else:
                lines.append("- BBN-verified gold: none (retired after geometry/initial-data audit)")
        lines.append("")

    # Test summary (kept in sync by scripts/sync_test_counts.py)
    counts = _current_status_counts()
    lines.append(
        f"## Tests ({counts['total']} collected; overlapping marker subsets: "
        f"**{counts['gold']} gold**, {counts['smoke']} smoke; "
        f"`@production` {counts['production_total']} total, "
        f"{counts['production_not_slow']} production-and-not-slow)"
    )
    lines.append("")

    # Runtime-regression summary.  These checks are not publication evidence.
    lines.append("## Runtime regression coverage (NOT publication validation)")
    lines.append("")
    lines.append("1. FLRW baseline gold tables are regression locks only.")
    lines.append("2. Historical Type-I shear, transport, eigenvalue, and channel checks are diagnostic regressions only.")
    lines.append("3. CL0–CL3 weak-correction tests cover runtime behavior, not matched publication physics.")
    lines.append("4. Finite-shear publication validation and every finite-shear inference constraint remain OPEN (B-03/B-05/B-06).")
    lines.append("")

    # Production gate hierarchy
    lines.append("## Production gate hierarchy")
    lines.append("")
    lines.append(
        f'1. Required fast release lane: `-m "production and not slow"` '
        f"({counts['production_not_slow']} collected)"
    )
    lines.append(
        f"2. `@production` is the total marker family "
        f"({counts['production_total']} collected; {counts['production_slow']} also marked slow)"
    )
    lines.append(
        f"3. `@gold` ({counts['gold']}) and `@release_smoke` "
        f"({counts['smoke']}) are overlapping subsets, not a partition or sum"
    )
    lines.append("4. `@build_env_only`: exact count sync (packaging only)")
    lines.append("5. `@slow`: opt-in runtime classification; excluded from the required fast lane")

    return "\n".join(lines)


def render_status_backends():
    """STATUS.md canonical/candidate/substrate backend tables from registry."""
    lines = []
    for tier_label in ["Canonical", "Candidate", "Substrate"]:
        tier_key = tier_label.lower()
        backends = [(k, c) for k, c in CAPABILITY_BY_KEY.items() if c.tier == tier_key]
        if not backends:
            continue
        lines.append(f"### {tier_label}")
        lines.append("")
        if tier_key == "substrate":
            lines.append("| Key | Scope | Status |")
            lines.append("|---|---|---|")
            for k, c in backends:
                lines.append(f"| `{k}` | {c.physics_scope} | Scaffold only |")
        else:
            lines.append("| Key | Backend | Scope | CL | Teff | Thermo |")
            lines.append("|---|---|---|---|---|---|")
            for k, c in backends:
                be = "SciPy" if "scipy" in k else "JAX (frozen oracle)"
                cl = f"0–{c.max_correction_level}" if c.max_correction_level > 0 else "0"
                if "geometry" in k or "solver" in k or "pstf" in k:
                    cl = "—"
                teff = "yes" if c.supports_teff else "no"
                if "geometry" in k or "solver" in k or "pstf" in k:
                    teff = "—"
                thermo = f"tier {c.thermo_tier}" if hasattr(c, "thermo_tier") and c.thermo_tier else "—"
                lines.append(f"| `{k}` | {be} | {c.physics_scope} | {cl} | {teff} | {thermo} |")
        lines.append("")
    return "\n".join(lines)


def render_full_matrix():
    lines = [
        "# Backend Capability Matrix (v1.0.0)", "",
        "**Auto-generated from `backend_capabilities.py`. Do not edit manually.**", "",
        "Historical tier labels preserve scoped evidence. Only `auto`/`scipy` have active runtime authority; JAX endpoint dispatch names are retired and retained JAX entries are non-dispatchable component metadata.", "",
        "## Dispatch table", "",
        "| `backend=` | Resolves to | Tier | Weak mode | Notes |",
        "|---|---|---|---|---|",
    ]
    for name in ("auto", "scipy"):
        cap = CAPABILITY_BY_BACKEND[name]
        desc = cap.notes.split(".")[0] if cap.notes else cap.physics_scope
        lines.append(f'| `"{name}"` | `{cap.key}` | **{cap.tier}** | {cap.weak_mode} | {desc} |')
    for tier in ["canonical", "candidate", "substrate"]:
        items = [(k, c) for k, c in CAPABILITY_BY_KEY.items() if c.tier == tier]
        lines.extend(["", f"## {tier.title()} ({len(items)})", ""])
        lines.append("| Key | Backend | CL | Weak mode | Teff | Notes |")
        lines.append("|---|---|---|---|---|---|")
        for k, c in items:
            teff = "yes" if c.supports_teff else "no"
            notes = c.notes.split(".")[0][:60] if c.notes else ""
            lines.append(f"| `{k}` | {c.backend} | {c.max_correction_level} | {c.weak_mode} | {teff} | {notes} |")
    return "\n".join(lines) + "\n"


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _build_render_provenance(targets, full_overwrites):
    """Return content-addressed provenance with no wall-clock state."""
    source_paths = [
        "scripts/render_capability_tables.py",
        "scripts/sync_test_counts.py",
        "src/rabbit/config/backend_capabilities.py",
        "src/rabbit/config/feature_capabilities.py",
        "src/rabbit/config/claim_gates.py",
    ]
    source_sha256 = {
        path: _sha256_bytes(Path(path).read_bytes())
        for path in source_paths
    }
    generated_target_sha256 = {
        f"{filepath}#{begin.split(':', 1)[1].split()[0]}": _sha256_bytes(content.encode("utf-8"))
        for filepath, begin, _end, content in targets
    }
    generated_target_sha256.update(
        {
            filepath: _sha256_bytes(content.encode("utf-8"))
            for filepath, content in full_overwrites.items()
        }
    )
    provenance_body = {
        "render_script": "scripts/render_capability_tables.py",
        "source_sha256": source_sha256,
        "generated_target_sha256": generated_target_sha256,
        "n_backends": len(CAPABILITY_BY_KEY),
        "n_features": len(FEATURE_BY_KEY),
        "n_canonical_backends": sum(
            1 for capability in CAPABILITY_BY_KEY.values()
            if capability.tier == "canonical"
        ),
        "n_canonical_features": len(FEATURE_TIERS.get("canonical", [])),
        "targets_rendered": len(generated_target_sha256),
        "documents_rendered": len(
            {filepath for filepath, _begin, _end, _content in targets}
            | set(full_overwrites)
        ),
        "surface_classes_backend": sorted(
            {capability.effective_surface_class for capability in CAPABILITY_BY_KEY.values()}
        ),
        "surface_classes_feature": sorted(
            {feature.surface_class for feature in FEATURE_BY_KEY.values()}
        ),
    }
    fingerprint_payload = json.dumps(
        provenance_body,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "provenance_schema": "content-addressed-v2",
        "render_fingerprint": _sha256_bytes(fingerprint_payload),
        **provenance_body,
    }


if __name__ == "__main__":
    apply = "--apply" in sys.argv
    print("═" * 60)
    print("Registry-driven document generation")
    print("═" * 60)

    blocks = {
        "README backend table": render_readme_backend_table(),
        "README core summary": render_core_summary(),
        "SUPPORTED backend table": render_supported_backend_table(),
        "SUPPORTED feature table": render_feature_table(),
        "STATUS tier summary": render_tier_summary(),
        "PROMOTION status": render_promotion_status(),
        "BACKEND_MATRIX full": render_full_matrix(),
        "PG header": render_pg_header(),
        "PG body": render_pg_body(),
        "PG footer": render_pg_footer(),
        "PROMOTION next queue": render_next_queue(),
        "CLASSB layers": render_classB_layers(),
        "STATUS backends": render_status_backends(),
        "STATUS header": render_status_header(),
        "STATUS detail": render_status_detail(),
        "README header": render_readme_header(),
        "README backends header": render_readme_backends_header(),
        "README quickstart": render_readme_quickstart(),
        "README footer": render_readme_footer(),
        "README identity": render_identity(),
        "SC header": render_sc_header(),
        "SC canonical core": render_canonical_core(),
        "CLAIMS": render_claims(),
    }

    for name, content in blocks.items():
        print(f"\n--- {name} ---")
        print(content[:200] + "..." if len(content) > 200 else content)

    if apply:
        rendered = []
        changed_documents = set()

        targets = [
            ("README.md", "<!-- BEGIN:README_BACKENDS_HEADER", "<!-- END:README_BACKENDS_HEADER -->", blocks["README backends header"]),
            ("README.md", "<!-- BEGIN:BACKEND_TABLE", "<!-- END:BACKEND_TABLE -->", blocks["README backend table"]),
            ("README.md", "<!-- BEGIN:README_HEADER", "<!-- END:README_HEADER -->", blocks["README header"]),
            ("README.md", "<!-- BEGIN:IDENTITY", "<!-- END:IDENTITY -->", blocks["README identity"]),
            ("README.md", "<!-- BEGIN:CORE_SUMMARY", "<!-- END:CORE_SUMMARY -->", blocks["README core summary"]),
            ("README.md", "<!-- BEGIN:README_QUICKSTART", "<!-- END:README_QUICKSTART -->", blocks["README quickstart"]),
            ("README.md", "<!-- BEGIN:README_FOOTER", "<!-- END:README_FOOTER -->", blocks["README footer"]),
            ("SUPPORTED_CAPABILITIES.md", "<!-- BEGIN:SC_HEADER", "<!-- END:SC_HEADER -->", blocks["SC header"]),
            ("SUPPORTED_CAPABILITIES.md", "<!-- BEGIN:BACKEND_TABLE", "<!-- END:BACKEND_TABLE -->", blocks["SUPPORTED backend table"]),
            ("SUPPORTED_CAPABILITIES.md", "<!-- BEGIN:SC_CANONICAL_CORE", "<!-- END:SC_CANONICAL_CORE -->", blocks["SC canonical core"]),
            ("SUPPORTED_CAPABILITIES.md", "<!-- BEGIN:FEATURE_TABLE", "<!-- END:FEATURE_TABLE -->", blocks["SUPPORTED feature table"]),
            ("STATUS.md", "<!-- BEGIN:STATUS_HEADER", "<!-- END:STATUS_HEADER -->", blocks["STATUS header"]),
            ("STATUS.md", "<!-- BEGIN:TIER_SUMMARY", "<!-- END:TIER_SUMMARY -->", blocks["STATUS tier summary"]),
            ("PROMOTION_GATES.md", "<!-- BEGIN:PG_HEADER", "<!-- END:PG_HEADER -->", blocks["PG header"]),
            ("PROMOTION_GATES.md", "<!-- BEGIN:PROMOTION_STATUS", "<!-- END:PROMOTION_STATUS -->", blocks["PROMOTION status"]),
            ("PROMOTION_GATES.md", "<!-- BEGIN:PG_BODY", "<!-- END:PG_BODY -->", blocks["PG body"]),
            ("PROMOTION_GATES.md", "<!-- BEGIN:NEXT_QUEUE", "<!-- END:NEXT_QUEUE -->", blocks["PROMOTION next queue"]),
            ("PROMOTION_GATES.md", "<!-- BEGIN:PG_FOOTER", "<!-- END:PG_FOOTER -->", blocks["PG footer"]),
            ("SUPPORTED_CAPABILITIES.md", "<!-- BEGIN:CLAIMS", "<!-- END:CLAIMS -->", blocks["CLAIMS"]),
            ("SUPPORTED_CAPABILITIES.md", "<!-- BEGIN:CLASSB_LAYERS", "<!-- END:CLASSB_LAYERS -->", blocks["CLASSB layers"]),
            ("STATUS.md", "<!-- BEGIN:STATUS_DETAIL", "<!-- END:STATUS_DETAIL -->", blocks["STATUS detail"]),
            ("STATUS.md", "<!-- BEGIN:STATUS_BACKENDS", "<!-- END:STATUS_BACKENDS -->", blocks["STATUS backends"]),
        ]

        for filepath, begin, end, content in targets:
            marker = begin.split(':')[1].split()[0]
            result = _replace_block(filepath, begin, end, content)
            if result is None:
                print(f"  ⚠ {filepath}: markers not found for {begin}")
                continue
            rendered.append(f"{filepath} ({marker})")
            if result:
                changed_documents.add(filepath)
                print(f"  ✅ {filepath} {marker} (updated)")
            else:
                print(f"  ✓ {filepath} {marker} (unchanged)")

        # Full overwrite
        full_overwrites = {
            "docs/BACKEND_CAPABILITY_MATRIX.md": blocks["BACKEND_MATRIX full"],
        }
        for filepath, content in full_overwrites.items():
            path = Path(filepath)
            if not path.exists() or path.read_text() != content:
                path.write_text(content)
                changed_documents.add(filepath)
                print(f"  ✅ {filepath} (updated full render)")
            else:
                print(f"  ✓ {filepath} (unchanged full render)")
            rendered.append(f"{filepath} (full)")

        # Content-addressed provenance: no wall-clock timestamp and no rewrite
        # when sources and generated payloads are unchanged.
        provenance = _build_render_provenance(targets, full_overwrites)
        prov_path = "docs/RENDER_PROVENANCE.json"
        os.makedirs("docs", exist_ok=True)
        provenance_text = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        provenance_file = Path(prov_path)
        if not provenance_file.exists() or provenance_file.read_text() != provenance_text:
            provenance_file.write_text(provenance_text)
            changed_documents.add(prov_path)
            print(f"  ✅ {prov_path} (updated content-addressed provenance)")
        else:
            print(f"  ✓ {prov_path} (unchanged content-addressed provenance)")

        print(
            f"\n  Rendered {len(rendered)} targets across "
            f"{provenance['documents_rendered']} documents; "
            f"changed {len(changed_documents)} documents."
        )
        print(f"  Provenance fingerprint: {provenance['render_fingerprint']}")
    else:
        print("\n[Use --apply to update documents]")
