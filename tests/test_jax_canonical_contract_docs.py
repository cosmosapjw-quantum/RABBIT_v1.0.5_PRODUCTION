from pathlib import Path

from rabbit.config.backend_capabilities import (
    ACTIVE_CANONICAL_BACKENDS,
    CAPABILITY_BY_BACKEND,
    JAX_RODAS5P_SOLVER,
)


ARCH_DOC = Path("docs/JAX_CANONICAL_ARCHITECTURE.md")
ROADMAP_DOC = Path("docs/JAX_MAIN_CANONICAL_PROMOTION_ROADMAP_2026-04-18.md")
RUST_FIRST_PLAN = Path("docs/TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md")
RENDERER = Path("scripts/render_capability_tables.py")
README_DOC = Path("README.md")
STATUS_DOC = Path("STATUS.md")
SUPPORTED_DOC = Path("SUPPORTED_CAPABILITIES.md")
PROMOTION_DOC = Path("PROMOTION_GATES.md")
BACKEND_MATRIX_DOC = Path("docs/BACKEND_CAPABILITY_MATRIX.md")
RENDER_PROVENANCE_DOC = Path("docs/RENDER_PROVENANCE.json")
GENERATED_CAPABILITY_DOCS = (
    README_DOC,
    STATUS_DOC,
    SUPPORTED_DOC,
    PROMOTION_DOC,
    BACKEND_MATRIX_DOC,
    RENDER_PROVENANCE_DOC,
)
STALE_PUB02_OPEN = ("PUB-02/G-01 open", "PUB-02 and G-01 remain open")
REPORT_SECTIONS = [
    Path("docs/RABBIT_report/sections/00_frontmatter.tex"),
    Path("docs/RABBIT_report/sections/04_eigenvalue_structure.tex"),
    Path("docs/RABBIT_report/sections/18_results.tex"),
    Path("docs/RABBIT_report/sections/19_inference_and_gradient_bridge.tex"),
    Path("docs/RABBIT_report/sections/20_validation.tex"),
    Path("docs/RABBIT_report/sections/22_conclusions.tex"),
]


def test_jax_architecture_doc_exists_and_is_frozen():
    assert ARCH_DOC.exists(), "JAX architecture provenance doc missing"
    text = ARCH_DOC.read_text()
    assert "DEPRECATED as a promotion roadmap" in text
    assert "JAX repeated-run/backend promotion: FORBIDDEN" in text
    assert "Rust active implementation target: SPECIFIED" in text
    assert '`backend="auto"` resolves to the SciPy Type-I reference' in text
    assert "does not grant active runtime" in text
    assert '`backend="jax"`' in text
    assert '`backend="jax_advanced"`' in text


def test_jax_main_promotion_roadmap_is_historical_only():
    assert ROADMAP_DOC.exists(), "JAX main canonical promotion roadmap missing"
    assert RUST_FIRST_PLAN.exists(), "Rust-first normative plan missing"
    text = ROADMAP_DOC.read_text()
    for needle in ["PR-A", "PR-B", "PR-C", "PR-D", "PR-E", "dispatch flip"]:
        assert needle in text, f"Roadmap missing {needle}"
    normative = RUST_FIRST_PLAN.read_text()
    assert "Historical" in normative
    assert "backend recommendation notes are provenance only" in normative
    assert "The first implementation after PORT-00 MUST be R-01" in normative


def test_generated_docs_close_pub02c_without_jax_runtime_authority():
    sources = (RENDERER,) + GENERATED_CAPABILITY_DOCS
    rendered = {path: path.read_text(encoding="utf-8") for path in sources}

    for path, text in rendered.items():
        for stale in STALE_PUB02_OPEN:
            assert stale not in text, f"{path} retains stale claim {stale!r}"

    promotion_text = rendered[PROMOTION_DOC]
    assert "jax_typeI_liveweak_cl3_tier1" in promotion_text
    assert "frozen oracle" in promotion_text
    assert "PUB-02C grants no repeated-run authority" in promotion_text
    assert "G-01R remains open" in promotion_text
    assert "jax_typeI_liveweak_cl0.tier` → `\"canonical\"" not in promotion_text


def test_pub02_does_not_promote_rodas5p_dispatch_or_default_tier():
    assert JAX_RODAS5P_SOLVER.tier == "candidate"
    assert JAX_RODAS5P_SOLVER.surface_class == "candidate"
    assert JAX_RODAS5P_SOLVER.validated_default is False
    assert JAX_RODAS5P_SOLVER.key not in {
        capability.key for capability in CAPABILITY_BY_BACKEND.values()
    }
    assert ACTIVE_CANONICAL_BACKENDS == frozenset({"scipy", "auto"})


def test_current_sources_withdraw_finite_shear_and_sampler_claims():
    """Historical outputs must not reappear as current publication evidence."""
    sources = REPORT_SECTIONS + [
        Path("docs/ROADMAP_INDEX.md"),
        Path("src/rabbit/inference/forward_likelihood.py"),
        Path("src/rabbit/inference/model_comparison.py"),
        Path("scripts/plot_fig20_fig21_standalone.py"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    forbidden = (
        r"\boxed{\Sigma_H < 0.66",
        "profile-likelihood bound is",
        "canonical production inference backend is BlackJAX",
        "SIGMA_VALIDATED_MAX",
        "production-facing visualisation",
        "operative production constraint",
        "validated SciPy range",
        "Canonical physics reference:",
    )
    for phrase in forbidden:
        assert phrase not in combined, f"stale publication claim remains: {phrase}"

    assert "B-03/B-05" in combined
    assert "historical diagnostic" in combined.lower()
    assert "must not be cited as a current science result" in combined.lower()
