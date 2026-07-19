"""rabbit.config.claim_gates — machine-checkable Forbidden -> Permitted gates.

Every entry in the SUPPORTED_CAPABILITIES.md "Forbidden Claims" block must
have a corresponding ClaimGate here.  A claim flips from Forbidden to
Permitted only when every test node referenced in `required_test_node_ids`
passes in CI.  No auto-mutation of docs or registry: scripts/promotion_check.py
emits a PR-ready diff for human review.

Foundation block F13 of the v1.0.5 -> v2.0.0 upgrade plan.

Lifecycle:
    1. A new test exists (created by a foundation or cell PR).
    2. The test passes.
    3. ``promotion_check.py`` collects the test node IDs and verifies success.
    4. On green, the script emits a diff that:
       (a) removes the Forbidden line from SUPPORTED_CAPABILITIES.md
           between the BEGIN:CLAIMS / END:CLAIMS markers,
       (b) appends `permitted_text` to the Permitted block,
       (c) optionally bumps the relevant FeatureCapability.tier.
    5. A human reviews and merges the resulting PR.

Forward references are allowed: a gate may name test node IDs that don't
yet exist on disk.  ``promotion_check.py`` will treat missing tests as
"gate red"; ``tests/test_claim_gates.py`` only enforces that every Forbidden
line has a gate, not that the named tests exist today.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from rabbit.config.bianchi_cells import (
    full_bianchi_gate_node_ids,
    tilted_bianchi_gate_node_ids,
)


@dataclass(frozen=True)
class ClaimGate:
    """A single Forbidden -> Permitted promotion gate."""
    claim_id: str
    forbidden_text: str
    required_test_node_ids: Tuple[str, ...]
    permitted_text: str
    feature_capability_key: str = ""  # Optional: tier-flip target in feature_capabilities.py
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════
# Gate registry — one entry per Forbidden line in SUPPORTED_CAPABILITIES.md
# ═══════════════════════════════════════════════════════════════════════

GATE_FULL_BIANCHI_BBN = ClaimGate(
    claim_id="full_bianchi_bbn",
    forbidden_text='"full Bianchi-BBN package" (only Type I canonical)',
    required_test_node_ids=full_bianchi_gate_node_ids(),
    permitted_text='"Full Bianchi-BBN package (canonical, all 11 types orthogonal+tilted, with constraint-residual gates)"',
    notes=(
        "The eleven-family gate intentionally treats VI_{-1/9} as an "
        "exceptional VI_h validation slice rather than a twelfth Bianchi family."
    ),
)

GATE_FULL_DIFFERENTIABLE_BBN = ClaimGate(
    claim_id="full_differentiable_bbn_solver",
    forbidden_text='"full differentiable BBN solver" (AD is diagnostic custom_vjp/FD bridge only, not native reverse-mode)',
    required_test_node_ids=(
        "tests/test_native_ad_parity.py::TestNativeADParity::test_yp_diffrax_vs_rodas5p",
        "tests/test_native_ad_parity.py::TestNativeADParity::test_value_and_grad_vs_richardson_fd",
        "tests/test_native_ad_parity.py::TestNativeADParity::test_jacrev_through_full_solve",
    ),
    permitted_text='"Full differentiable BBN solver (Diffrax/Kvaerno5 native reverse-mode AD; jacrev/jacfwd through the production ODE solve; Richardson-FD parity to 1e-4 rel)"',
)

GATE_BAYES_FACTOR_HEADLINE = ClaimGate(
    claim_id="bayes_factor_headline",
    forbidden_text='"Bayes factor headline" (no sampler ran on real BBN likelihood; grid-scan only)',
    # BD612-F4: forward-reference — test file not yet on disk; gate stays RED-missing (fail-closed) by design
    required_test_node_ids=(
        "tests/test_sampler_end_to_end_synthetic.py::TestEvidenceCalibration::test_smc_log_evidence_within_target",
        "tests/test_sampler_end_to_end_synthetic.py::TestEvidenceCalibration::test_savage_dickey_consistent_with_smc",
        "tests/test_sampler_end_to_end_synthetic.py::TestNullRecovery::test_flrw_truth_bf_favours_null",
    ),
    permitted_text='"Publication-grade Bayes factor (SMC log-evidence on real BBN forward model, Savage-Dickey consistency, null-recovery calibration)"',
)

GATE_GRADIENT_BASED_INFERENCE = ClaimGate(
    claim_id="gradient_based_inference_ready",
    forbidden_text='"gradient-based inference ready" (AD is diagnostic-tier, not production HMC/NUTS)',
    # BD612-F4: forward-reference — test file not yet on disk; gate stays RED-missing (fail-closed) by design
    required_test_node_ids=(
        "tests/test_sampler_end_to_end_synthetic.py::TestNUTSEndToEnd::test_nuts_recovers_synthetic_truth",
        "tests/test_fisher.py::TestFIM::test_bbn_cond_number_below_budget",
        "tests/test_native_ad_parity.py::TestNativeADParity::test_value_and_grad_vs_richardson_fd",
    ),
    permitted_text='"Gradient-based inference (BlackJAX NUTS through Diffrax-AD on real BBN forward model; FIM cond < 1e6 on (Sigma_H, eta, tau_n))"',
)

GATE_TEFF_ON_CANONICAL_SCIPY = ClaimGate(
    claim_id="teff_public_runtime_deprecated",
    forbidden_text='"Teff is active on public forward-solver paths" (deprecated legacy; characteristic/full transport supersedes it)',
    required_test_node_ids=(),
    permitted_text="",
    notes=(
        "Documentation-only gate: Teff is deprecated and removed from public "
        "forward-solver promotion. Low-level kernels remain diagnostic only."
    ),
)

GATE_SIGMA_H_TO_0P95 = ClaimGate(
    claim_id="sigma_h_validated_to_0p95",
    forbidden_text=(
        '"Type I validated to Sigma_H = 0.95" (the 0.75 cutoff is only a legacy '
        "guarded runtime/implementation range; the publication/science shear domain "
        "is NOT VALIDATED pending B-03/B-05)"
    ),
    required_test_node_ids=(
        "tests/test_typeI_extended_range.py::TestExtendedRangeSweep::test_sigma_h_up_to_0p95_constraint_residual",
        "tests/test_typeI_extended_range.py::TestExtendedRangeSweep::test_sigma_h_up_to_0p95_yp_regression",
        # External-audit 2026-06-30 (B8): the two nodes above are INTERNAL (constraint residual +
        # Yp regression vs self-consistency gold). A finite-shear science claim must not promote on
        # self-consistency alone, so the claim ALSO requires an external cross-code anchor. That node
        # FAILS CLOSED until an independent benchmark is supplied (tests/test_finite_shear_external_anchor.py),
        # so this gate stays red even once the two internal nodes are written.
        "tests/test_finite_shear_external_anchor.py::test_finite_shear_matches_external_oracle",
    ),
    permitted_text=(
        '"Legacy Type I finite-shear implementation checks through Sigma_H = 0.95 '
        "(internal evolved mass-fraction closure + Yp regression AND external cross-code "
        "parity); this does not validate a publication/science shear domain, which remains "
        "NOT VALIDATED pending B-03/B-05)"
    ),
    feature_capability_key="typeI_anisotropic",
    notes=(
        "The legacy implementation checks require BOTH internal self-consistency AND an external "
        "finite-shear cross-code anchor. The internal nodes check the EVOLVED mass-fraction closure "
        "|sum X_i - 1| + a Yp "
        "regression lock (the Friedmann residual |1-Omega-Sigma^2| is identically 0 by construction, "
        "so it carries no validation weight); the external anchor has no comparable external code "
        "today (plan B8), so the gate is fail-closed red by construction. The 0.75 cutoff is a "
        "legacy runtime guard, not evidence of a validated shear interval; B-03 and B-05 remain the "
        "publication/science-domain authorities."
    ),
)

GATE_CL3_FULLY_CANONICAL = ClaimGate(
    claim_id="cl3_fully_canonical",
    forbidden_text='"CL3 is fully canonical or publication validated" (historical JAX tiers are frozen)',
    # BD612-F4: forward-reference — test file not yet on disk; gate stays RED-missing (fail-closed) by design
    required_test_node_ids=(
        "tests/test_cl3_canonical_promotion.py::TestCL3Canonical::test_cl3_scipy_jax_parity",
        "tests/test_cl3_canonical_promotion.py::TestCL3Canonical::test_cl3_external_parity_primat",
        "tests/test_cl3_canonical_promotion.py::TestCL3Canonical::test_cl3_finite_mass_regression",
    ),
    permitted_text='"Historical CL3 SciPy/JAX parity and finite-mass/Coulomb/Sirlin checks are bounded runtime regressions only"',
    feature_capability_key="weak_rates",
)

GATE_PUBLICATION_GRADE_EVIDENCE = ClaimGate(
    claim_id="publication_grade_model_comparison",
    forbidden_text='"publication-grade model comparison / evidence" (inference is exploratory PE framework only)',
    # BD612-F4: forward-reference — test file not yet on disk; gate stays RED-missing (fail-closed) by design
    required_test_node_ids=(
        "tests/test_sampler_end_to_end_synthetic.py::TestEvidenceCalibration::test_smc_log_evidence_within_target",
        "tests/test_sampler_end_to_end_synthetic.py::TestEvidenceCalibration::test_dynesty_logz_smc_consistent",
        "tests/test_cross_code_cluster.py::TestCrossCodeCluster::test_evidence_consistent_across_codes",
    ),
    permitted_text='"Publication-grade model comparison (SMC + nested-sampling evidence on real BBN, multi-code cross-validation)"',
)

GATE_FULL_NUDEC_SELFCONSISTENT = ClaimGate(
    claim_id="full_neutrino_decoupling_selfconsistent",
    forbidden_text='"full neutrino-decoupling self-consistent solver" (canonical default is tier-1 thermo)',
    # BD612-F4: forward-reference — test file not yet on disk; gate stays RED-missing (fail-closed) by design
    required_test_node_ids=(
        "tests/test_tier3_decoupled_thermo.py::TestTier3::test_neff_matches_mangano_within_1e_minus_3",
        "tests/test_tier3_decoupled_thermo.py::TestTier3::test_T_nu_e_T_nu_x_split",
        "tests/test_flrw_external_parity.py::TestNeffParity::test_neff_vs_mangano_at_thermo_tier3",
    ),
    permitted_text='"Full self-consistent neutrino-decoupling (Tier-3 decoupled thermo on Hannestad-Madsen kernels; N_eff vs Mangano 2005 within 1e-3)"',
    feature_capability_key="flrw_baseline",
)

GATE_AUGMENTED_TYPEI_PUBLIC_PRODUCTION = ClaimGate(
    claim_id="augmented_typei_public_production",
    forbidden_text=(
        '"augmented Type I PSTF no-QKE publication-ready / public production" '
        '(AP51-AP81 are only direct-wrapper outcome, solver-matrix, '
        'source-policy, convergence, source-budget, evidence-bundle, sanity-matrix, angular input-model, and LRS/non-LRS angular-rate diagnostics '
        'plus rate-only and AP65-coupled weak-rate candidate gates, nonlinear transport RHS/solve scaffolds, nonlinear 3T weak/network candidate coupling, opt-in nonlinear collision-feedback wiring, a publication-candidate convergence matrix, a known-limit validation atlas, a guarded inference adapter, an augmented SMC likelihood schema, a smoke tempered-SMC runner, AP71 runtime/cache controls, AP72 synthetic SMC validation, AP73 figure-ready artifact tables, AP74 diagnostic plots, AP75 reproducibility-bundle packaging, AP76/AP79 readiness audit, AP77 smoke coupled weak-rate gate, AP78 same-CL3 weak-rate controls, AP80 profile-level weak-rate convergence diagnostics, FB-20 optional production-candidate evidence gate, and AP81 six-monomial collision statistical-factor plus staged pairwise diagonal-nu-nu source wiring; no public dispatch or production '
        'SMC validation yet)'
    ),
    required_test_node_ids=(),
    permitted_text="",
    feature_capability_key="typeI_augmented_pstf_noqke_staging",
    notes=(
        "Documentation-only boundary after BD497: the publication/readiness/SMC/"
        "figure evidence web was quarantined under out_of_scope and no longer "
        "defines a promotion path for the active AP65 no-QKE solver."
    ),
)

GATE_TEFF_OUTSIDE_DOCUMENTED_REGIME = ClaimGate(
    claim_id="teff_promotion_target_deprecated",
    forbidden_text='"Teff is a promotion target" (deprecated legacy; retained only as low-level kernel diagnostics)',
    required_test_node_ids=(),
    permitted_text="",
    feature_capability_key="teff_spectral",
    notes=(
        "Documentation-only gate: Teff no longer competes with the Bianchi "
        "transport roadmap."
    ),
)

GATE_TILTED_OUTSIDE_TYPE_I_SCALAR = ClaimGate(
    claim_id="tilted_production_outside_type_i_scalar",
    forbidden_text=('"Tilted production-ready outside Type I scalar slice" (documented candidate scope '
                    'only: Type I, scalar, v0 in [0, 1e-3])'),
    # BD612-F4: forward-reference — test file not yet on disk; gate stays RED-missing (fail-closed) by design
    required_test_node_ids=(
        *tilted_bianchi_gate_node_ids(include_type_i=False),
        "tests/test_tilt_vector_scalar_parity.py::TestTiltVectorScalarParity::test_scalar_recovers_from_vector_at_v1_v2_zero",
    ),
    permitted_text='"Tilted BBN canonical across all 11 Bianchi types (full 3-vector tilt, momentum constraint G^0_i < 1e-10, gold-locked)"',
    feature_capability_key="tilted_scalar",
)

GATE_CLASS_A_EXACT_CURVED_PSTF = ClaimGate(
    claim_id="class_a_exact_curved_pstf_production",
    forbidden_text=('"Class A exact curved PSTF production-ready" (exact hierarchy remains substrate; only '
                    'reduced kappa-cascade is a documented candidate path)'),
    # BD612-F4: forward-reference — test file not yet on disk; gate stays RED-missing (fail-closed) by design
    required_test_node_ids=(
        "tests/test_curved_pstf_convergence.py::TestCurvedPSTFConvergence::test_lmax_4_vs_6_vs_8_drift",
        "tests/test_curved_pstf_convergence.py::TestCurvedPSTFConvergence::test_flat_limit_PQ_zero_bit_stable",
        "tests/gold/test_typeVI0_orthogonal_bbn.py",
        "tests/gold/test_typeVII0_orthogonal_bbn.py",
        "tests/gold/test_typeVIII_orthogonal_bbn.py",
    ),
    permitted_text='"Class A exact curved PSTF (full ell->ell+/-1 m-decomposed transport; canonical for VI_0, VII_0, VIII; ell_max=8 convergence-envelope documented)"',
    feature_capability_key="classA_curved_transport",
)

GATE_CLASS_B_BEYOND_DOCUMENTED = ClaimGate(
    claim_id="class_b_beyond_documented_slices",
    forbidden_text=('"Class B production-ready beyond documented candidate slices" (all 6 reduced-mask labels '
                    'have representative gold cells; continuous h-family, anisotropic K_+/K_-, frame-vector '
                    'momentum sources, and full Class B remain unpromoted)'),
    # BD612-F4: forward-reference — test file not yet on disk; gate stays RED-missing (fail-closed) by design
    required_test_node_ids=(
        "tests/gold/test_typeV_orthogonal_bbn.py",
        "tests/gold/test_typeIV_orthogonal_bbn.py",
        "tests/gold/test_typeIII_orthogonal_bbn.py",
        "tests/gold/test_typeVIIh_orthogonal_bbn.py",
        "tests/gold/test_typeVIh_orthogonal_bbn.py",
        "tests/gold/test_typeVI_m19_orthogonal_bbn.py",
        "tests/test_curvature_full.py::TestAnisotropicCurvature::test_K_plus_minus_reduce_to_scalar_K_for_V_IV",
        "tests/test_h_family_continuity.py::THFamily::test_VIIh_h_to_zero_recovers_VII0",
    ),
    permitted_text='"Class B BBN canonical for all 6 types (V, IV, III, VI_h, VII_h, VI_{-1/9}); h-parameter family encoded; anisotropic K_+/K_- and frame-vector S_a integrated)"',
    feature_capability_key="classB_bbn",
)


ALL_GATES: Tuple[ClaimGate, ...] = (
    GATE_FULL_BIANCHI_BBN,
    GATE_FULL_DIFFERENTIABLE_BBN,
    GATE_BAYES_FACTOR_HEADLINE,
    GATE_GRADIENT_BASED_INFERENCE,
    GATE_TEFF_ON_CANONICAL_SCIPY,
    GATE_SIGMA_H_TO_0P95,
    GATE_CL3_FULLY_CANONICAL,
    GATE_PUBLICATION_GRADE_EVIDENCE,
    GATE_FULL_NUDEC_SELFCONSISTENT,
    GATE_AUGMENTED_TYPEI_PUBLIC_PRODUCTION,
    GATE_TEFF_OUTSIDE_DOCUMENTED_REGIME,
    GATE_TILTED_OUTSIDE_TYPE_I_SCALAR,
    GATE_CLASS_A_EXACT_CURVED_PSTF,
    GATE_CLASS_B_BEYOND_DOCUMENTED,
)


def gates_by_id() -> dict:
    return {g.claim_id: g for g in ALL_GATES}


def forbidden_lines_set() -> set:
    """Set of forbidden_text values currently registered."""
    return {g.forbidden_text for g in ALL_GATES}


__all__ = [
    "ClaimGate",
    "ALL_GATES",
    "gates_by_id",
    "forbidden_lines_set",
    "GATE_FULL_BIANCHI_BBN",
    "GATE_FULL_DIFFERENTIABLE_BBN",
    "GATE_BAYES_FACTOR_HEADLINE",
    "GATE_GRADIENT_BASED_INFERENCE",
    "GATE_TEFF_ON_CANONICAL_SCIPY",
    "GATE_SIGMA_H_TO_0P95",
    "GATE_CL3_FULLY_CANONICAL",
    "GATE_PUBLICATION_GRADE_EVIDENCE",
    "GATE_FULL_NUDEC_SELFCONSISTENT",
    "GATE_TEFF_OUTSIDE_DOCUMENTED_REGIME",
    "GATE_TILTED_OUTSIDE_TYPE_I_SCALAR",
    "GATE_CLASS_A_EXACT_CURVED_PSTF",
    "GATE_CLASS_B_BEYOND_DOCUMENTED",
]
