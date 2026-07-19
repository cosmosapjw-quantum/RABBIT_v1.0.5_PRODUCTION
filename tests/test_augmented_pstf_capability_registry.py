"""Registry locks for the augmented-PSTF no-QKE staging surface."""

from __future__ import annotations

from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND, CAPABILITY_BY_KEY
from rabbit.config.feature_capabilities import FEATURE_BY_KEY


AUGMENTED_BACKEND_KEY = "jax_typeI_augmented_pstf_noqke_staging"
AUGMENTED_FEATURE_KEY = "typeI_augmented_pstf_noqke_staging"


def test_augmented_pstf_backend_capability_is_catalog_only():
    assert AUGMENTED_BACKEND_KEY in CAPABILITY_BY_KEY

    cap = CAPABILITY_BY_KEY[AUGMENTED_BACKEND_KEY]
    assert cap.tier == "substrate"
    assert cap.effective_surface_class == "diagnostic"
    assert cap.validation_mode == "diagnostic"
    assert cap.physics_scope == "TypeI_augmented_pstf_noqke_staging"
    assert cap.weak_mode == "live_monopole_lrs_cl3_moment_input_nonlrs_s2_moment_input"
    assert cap.max_correction_level == 3
    assert cap.max_thermo_tier == 0
    assert not cap.validated_default

    dispatch_keys = {cap.key for cap in CAPABILITY_BY_BACKEND.values()}
    assert AUGMENTED_BACKEND_KEY not in dispatch_keys


def test_augmented_pstf_backend_scope_contracts_are_honest():
    cap = CAPABILITY_BY_KEY[AUGMENTED_BACKEND_KEY]
    assert cap.readiness_scope_contract == "substrate_typeI_augmented_pstf_noqke_staging_v1"
    assert cap.transport_scope_contract == "lrs_collisionless_nonlrs_source_and_nonlinear_staging_v1"
    assert cap.thermo_scope_contract == "not_full_bbn_thermo_v1"
    assert cap.collision_scope_contract == "deterministic_moment_feedback_and_nodal_projection_staging_v1"

    notes = cap.notes.lower()
    first_sentence = notes.split(".")[0]
    assert "catalog-only" in first_sentence
    assert "no public dispatch" in first_sentence
    assert "no public dispatch" in notes
    # BD612-F2: notes must state the AP0-AP81 surface is historical/deleted, not
    # current capability, before enumerating the (mostly deleted) stage detail.
    assert "historical" in notes
    assert "deleted in pr-d1..d3" in notes
    assert "not current" in notes
    assert "no default/promoted full-span collision-coupled runtime" in notes
    assert "promotion-grade coupled-solve weak-rate convergence" in notes
    assert "six-monomial pauli statistical-factor" in notes
    assert "ap4/ap65 combined-source full-span candidate gate" in notes
    assert "charge-neutral n_span=1e-4 evolved charge-asymmetry handoff evidence" in notes
    assert "fb-08 electron-bath/scalar-qed chained cross-product rows" in notes
    assert "scalar-qed one-hot diagnostics" in notes
    assert "fb-09 chained n_q/n_mu/n_phi resolution ladder rows" in notes
    assert "fb-20 optional production-candidate gate wiring" in notes
    assert "fb-20 is only an optional production-candidate evidence gate" in notes
    assert "charge-neutral electron chemical-potential forwarding" in notes
    assert "terminal electron-mu/charge-asymmetry observables" in notes
    assert "local pstf six-monomial angular contraction table" in notes
    assert "g_mumu" in notes
    assert "pi_ij" in notes
    assert "radial p4 contraction" in notes
    assert "radial channel kernel-grid assembly" in notes
    assert "pstf collision-kernel solver/runtime coupling" in notes
    assert "qke out of scope" in notes
    assert "rodas5p" in notes


def test_augmented_pstf_feature_registry_labels_staging_scope():
    assert AUGMENTED_FEATURE_KEY in FEATURE_BY_KEY

    feat = FEATURE_BY_KEY[AUGMENTED_FEATURE_KEY]
    assert feat.tier == "substrate"
    assert feat.validation_mode == "diagnostic"
    assert feat.surface_class == "diagnostic"
    assert "AP0-AP81" in feat.evidence_summary
    # BD612-F2: evidence_summary must LEAD with the honest deflation (rendered
    # surface truncates to the first 70 chars) — deleted capability must not read
    # as current.
    assert feat.evidence_summary.startswith("HISTORICAL AP0-AP81 stage-record")
    assert "deleted in " in feat.evidence_summary
    assert "NOT current capability" in feat.evidence_summary
    assert "characteristic_rays_nonlrs_jax" in feat.evidence_summary
    assert "ell_max" in feat.evidence_summary
    assert "deterministic JSON artifact runner" in feat.evidence_summary
    assert "collision-moment thermo feedback hook" in feat.evidence_summary
    assert "diagonal no-QKE nu-nu thermo source factory" in feat.evidence_summary
    assert "monopole nu-e plus pair-process thermo source factory" in feat.evidence_summary
    assert "combined source factory" in feat.evidence_summary
    assert "collision-feedback artifact runner" in feat.evidence_summary
    assert "standard-relative observable deltas" in feat.evidence_summary
    assert "stage-scoped landed" in feat.evidence_summary
    assert "bounded LRS CL3 weak-rate multiplier" in feat.evidence_summary
    assert "per-q angular weak-rate input objects" in feat.evidence_summary
    assert "moment-input angular weak-rate mode" in feat.evidence_summary
    assert "non-LRS S2 plus/minus basis" in feat.evidence_summary
    assert "deterministic weak-rate candidate/convergence gate" in feat.evidence_summary
    assert "non-LRS S2 nonlinear transport logit-RHS operator" in feat.evidence_summary
    assert "non-LRS nonlinear transport solve shell" in feat.evidence_summary
    assert "nonlinear non-LRS 3T weak/network candidate shell" in feat.evidence_summary
    assert "opt-in nonlinear angular collision-feedback 3T wrapper" in feat.evidence_summary
    assert "radial `C_modes` hierarchy payloads" in feat.evidence_summary
    assert "AP6 `pstf_radial` process-source budget case" in feat.evidence_summary
    assert "LRS charge-neutral AP6 `pstf_radial` source-budget case" in feat.evidence_summary
    assert "non-LRS AP6 `pstf_radial` process-source budget case" in feat.evidence_summary
    assert "non-LRS charge-neutral AP6 `pstf_radial` source-budget case" in feat.evidence_summary
    assert "AP56 AP55 LRS/non-LRS fixed and charge-neutral radial budget summaries" in feat.evidence_summary
    assert "AP57 AP6 radial source-budget sanity case" in feat.evidence_summary
    assert "AP57 LRS charge-neutral AP6 radial source-budget sanity case" in feat.evidence_summary
    assert "AP57 non-LRS AP6 radial source-budget sanity case" in feat.evidence_summary
    assert "AP57 non-LRS charge-neutral AP6 radial source-budget sanity case" in feat.evidence_summary
    assert "nonlinear non-LRS `pstf_radial` collision-feedback 3T wrapper" in feat.evidence_summary
    assert "opt-in AP65 combined angular+pstf_radial nonlinear collision wrapper/artifact" in feat.evidence_summary
    assert "AP4/AP65 combined-source full-span candidate gate" in feat.evidence_summary
    assert "real `live_rhs` two-span and warm-preset `N_span=1e-8` budgeted evidence" in feat.evidence_summary
    assert "named `piecewise_physical_preview` preset" in feat.evidence_summary
    assert "piecewise source-refresh refinement artifact" in feat.evidence_summary
    assert "`piecewise_frozen` source-refresh gate with nonuniform state handoff through `N_span=1e-3`" in feat.evidence_summary
    assert "charge-neutral `N_span=1e-4` evolved charge-asymmetry handoff evidence" in feat.evidence_summary
    assert "fixed/charge-neutral electron chemical-potential forwarding" in feat.evidence_summary
    assert "terminal electron-mu/charge-asymmetry observables" in feat.evidence_summary
    assert "combined-source source-policy span-profile artifact/CLI" in feat.evidence_summary
    assert "publication-candidate convergence matrix" in feat.evidence_summary
    assert "source-model" in feat.evidence_summary
    assert "component `combined_*_dA_abs_max_final` diagnostics" in feat.evidence_summary
    assert "`pstf_radial` source-budget observables" in feat.evidence_summary
    assert "fixed/charge-neutral electron-bath controls" in feat.evidence_summary
    assert "terminal electron-mu/charge-asymmetry observables" in feat.evidence_summary
    assert "collision_dA_abs_max_final" in feat.evidence_summary
    assert "known-limit validation atlas" in feat.evidence_summary
    assert "AP66 source-model evidence links" in feat.evidence_summary
    assert "AP67 radial momentum-delta forwarding into AP65/AP66" in feat.evidence_summary
    assert "fixed/charge-neutral electron-bath controls" in feat.evidence_summary
    assert "fail-closed radial source-budget rows" in feat.evidence_summary
    assert "guarded inference adapter" in feat.evidence_summary
    assert "selectable angular-only or combined angular+`pstf_radial` source models" in feat.evidence_summary
    assert "electron-bath control forwarding" in feat.evidence_summary
    assert "radial-control forwarding" in feat.evidence_summary
    assert "augmented SMC likelihood schema" in feat.evidence_summary
    assert "source-model/radial/electron-bath solver controls" in feat.evidence_summary
    assert "radial momentum-delta" in feat.evidence_summary
    assert "combined-source charge-neutral smoke likelihood evidence" in feat.evidence_summary
    assert "smoke tempered SMC runner" in feat.evidence_summary
    assert "AP69 source-model/source-refresh/radial/electron-bath control metadata preservation" in feat.evidence_summary
    assert "scalar-QED control metadata preservation" in feat.evidence_summary
    assert "runtime/cache controls" in feat.evidence_summary
    assert "source-model/source-refresh/radial/electron-bath-control-aware cache keys" in feat.evidence_summary
    assert "scalar-QED-aware cache context" in feat.evidence_summary
    assert "AP69 source/source-refresh/radial/electron-bath dry-run controls" in feat.evidence_summary
    assert "N_span-end, scalar-QED, radial momentum-delta, and piecewise_frozen subspan controls" in feat.evidence_summary
    assert "last successful prediction metadata preservation" in feat.evidence_summary
    assert "synthetic null/injection SMC validation" in feat.evidence_summary
    assert "source-model/source-refresh/N_span/radial/electron-bath-control provenance" in feat.evidence_summary
    assert "scalar-QED provenance" in feat.evidence_summary
    assert "radial momentum-delta provenance" in feat.evidence_summary
    assert "figure-ready publication artifact tables" in feat.evidence_summary
    assert "source-model/source-refresh/N_span/radial-momentum-delta/electron-bath-control row context" in feat.evidence_summary
    assert "diagnostic augmented publication plots" in feat.evidence_summary
    assert "source-model/source-refresh/N_span/radial-momentum-delta/electron-bath plot provenance" in feat.evidence_summary
    assert "reproducibility bundle" in feat.evidence_summary
    assert "source-model/source-refresh/N_span/radial-momentum-delta/electron-bath bundle provenance" in feat.evidence_summary
    assert "final readiness audit" in feat.evidence_summary
    assert "not_promoted" in feat.evidence_summary
    assert "source-model/source-refresh/N_span/radial-momentum-delta/electron-bath readiness-ledger provenance" in feat.evidence_summary
    assert "coupled weak-rate gate" in feat.evidence_summary
    assert "AP65 3T solve" in feat.evidence_summary
    assert "electron-bath control forwarding" in feat.evidence_summary
    assert "same CL3 weak kernel" in feat.evidence_summary
    assert "required input to the readiness audit" in feat.notes
    assert "AP77 electron-bath control provenance" in feat.notes
    assert "profile-level coupled weak-rate convergence artifact" in feat.evidence_summary
    assert "FB-20 adds an optional production-candidate gate" in feat.evidence_summary
    assert "AP79 full-chain readiness evidence and AP80 extended weak-rate profiles" in feat.evidence_summary
    assert "FB-08 electron-bath/scalar-QED chained cross-product rows" in feat.evidence_summary
    assert "evolved charge-asymmetry and scalar-QED one-hot diagnostics" in feat.evidence_summary
    assert "FB-09 chained N_q/N_mu/N_phi resolution ladder rows" in feat.evidence_summary
    assert "terminal BBN observable deltas and source-budget metadata" in feat.evidence_summary
    assert "quartic-cancelled six-monomial Pauli statistical factor" in feat.evidence_summary
    assert "local PSTF six-monomial angular contraction" in feat.evidence_summary
    assert "G_mumu" in feat.evidence_summary
    assert "channel `K` assembly" in feat.evidence_summary
    assert "radial-grid p2/p3 contraction" in feat.evidence_summary
    assert "radial channel kernel-grid assembly" in feat.evidence_summary
    assert "unit-direction momentum-delta" in feat.evidence_summary
    assert "p-dependent radial" in feat.evidence_summary
    assert "static angular-geometry reuse" in feat.evidence_summary
    assert "p4" in feat.evidence_summary
    assert "concrete moment and kinetic-source amplitude deltas" in feat.evidence_summary
    assert "charge-neutrality radial sources preserving `dA_modes` hierarchy payloads" in feat.evidence_summary
    assert "collision_dA_abs_max` kinetic-source amplitude rows and deltas" in feat.evidence_summary
    assert "final `collision_dA_abs_max_final` kinetic-source observables" in feat.evidence_summary
    assert "combined angular+`pstf_radial` component `dA`" in feat.notes
    assert "terminal electron-bath observables" in feat.notes
    assert "source-model provenance" in feat.notes
    assert "radial momentum-delta forwarding into AP65/AP66" in feat.notes
    assert "selectable angular-only or combined angular+`pstf_radial` source models" in feat.notes
    assert "fixed source-model/radial/electron-bath solver controls" in feat.notes
    assert "radial momentum-delta" in feat.notes
    assert "AP69 source-model/source-refresh/radial/electron-bath control metadata preservation" in feat.notes
    assert "scalar-QED control metadata preservation" in feat.notes
    assert "source-model/source-refresh/radial/electron-bath-control-aware cache keys" in feat.notes
    assert "scalar-QED-aware cache context" in feat.notes
    assert "last successful prediction metadata preservation" in feat.notes
    assert "source-model/source-refresh/N_span/radial/electron-bath-control provenance" in feat.notes
    assert "scalar-QED provenance" in feat.notes
    assert "radial momentum-delta provenance" in feat.notes
    assert "source-model/source-refresh/N_span/radial-momentum-delta/electron-bath-control row context" in feat.notes
    assert "source-model/source-refresh/N_span/radial-momentum-delta/electron-bath plot provenance" in feat.notes
    assert "source-model/source-refresh/N_span/radial-momentum-delta/electron-bath bundle provenance" in feat.notes
    assert "source-model/source-refresh/N_span/radial-momentum-delta/electron-bath readiness-ledger provenance" in feat.notes
    assert "diagonal no-QKE 2-to-2 collision kernels" in feat.evidence_summary
    assert "staged NumPy AP19/AP33/AP35/AP41 diagonal-nu-nu source bridge" in feat.evidence_summary
    assert "all-nine ordered pairwise bank coverage" in feat.evidence_summary
    assert "identical-bank self-scattering diagnostics" in feat.evidence_summary
    assert "weighted-energy closure projection" in feat.notes
    assert "all nine ordered bank pairs" in feat.notes
    assert "same-bank number/energy-neutral projection" in feat.notes
    assert "off-diagonal number-neutral projection" in feat.notes
    assert "all-nine radial number conservation" in feat.notes
    assert "unordered-pair energy-neutral closure" in feat.notes
    assert "preserves relative energy exchange" in feat.notes
    assert "fixed-point redistribution helper remains legacy comparison" in feat.notes
    assert "six-monomial polynomial" in feat.notes
    assert "K34" in feat.notes
    assert "G_mu" in feat.notes
    assert "Pi_ij Pi_kl" in feat.notes
    assert "p4 = E1 + E2 - E3" in feat.notes
    assert "radial `C_modes` to `dA_modes` hierarchy payload mapping" in feat.notes
    assert "concrete moment and kinetic-source amplitude deltas" in feat.notes
    assert "charge-neutrality radial sources preserving `dA_modes` hierarchy payloads" in feat.notes
    assert "finite-mu-scaled isotropic QED" in feat.notes
    assert "exact_finite_mu_scalar" in feat.notes
    assert "signed-`mu_e` `nu-e` scattering and pair-process Pauli blocking" in feat.evidence_summary
    assert "AP18/AP40 callback forwarding" in feat.evidence_summary
    assert "signed-`mu_e` `nu-e` scattering and pair-process Pauli blocking" in feat.notes
    assert "AP18/AP40 callback forwarding" in feat.notes
    assert "collision_dA_abs_max` kinetic-source amplitude rows and deltas" in feat.notes
    assert "collision_dA_abs_max_final" in feat.notes
    assert "invalid-kinematic zeroing" in feat.notes
    assert "momentum-delta weights" in feat.notes
    assert "unit-direction momentum-delta" in feat.notes
    assert "p-dependent radial Gaussian" in feat.notes
    assert "q=(3,4,5)" in feat.notes
    assert "FB-20 adds an optional production-candidate gate artifact" in feat.notes
    assert "still leaves public dispatch, production SMC validation, and promotion decision disabled" in feat.notes
    assert "AP6 `pstf_radial` source moments plus kinetic `dA` amplitude" in feat.notes
    assert "weak-rate candidate sub-gate" in feat.evidence_summary
    assert "collision-feedback candidate sub-gate" in feat.evidence_summary
    assert "AP36 LRS source-policy routing" in feat.evidence_summary
    assert "span-ladder candidate gate" in feat.evidence_summary
    assert "collision-feedback 3T convergence runners" in feat.evidence_summary
    assert "collision source-moment control-variate reports" in feat.evidence_summary
    assert "LRS angular-node nu-e scattering projection bridge" in feat.evidence_summary
    assert "elastic-scattering number closure" in feat.evidence_summary
    assert "LRS angular projection path to the electromagnetic pair-process" in feat.evidence_summary
    assert "number-closed scattering" in feat.evidence_summary
    assert "AP81 pairwise diagonal no-QKE nu-nu 2-to-2 reference" in feat.evidence_summary
    assert "generic angular-node v2 contracts" in feat.evidence_summary
    assert "explicit AP18 3T thermo-source callback" in feat.evidence_summary
    assert "LRS angular-collision 3T direct wrapper" in feat.evidence_summary
    assert "LRS direct wrapper in the collision-feedback artifact" in feat.evidence_summary
    assert "source-update policy" in feat.evidence_summary
    assert "source-only non-LRS coevolution shell" in feat.evidence_summary
    assert "source-only non-LRS weak/network shell" in feat.evidence_summary
    assert "source-only non-LRS 3T thermo/Hubble shell" in feat.evidence_summary
    assert "non-LRS collision-moment thermo feedback hook" in feat.evidence_summary
    assert "optional projected collision `dA_modes` hierarchy term" in feat.evidence_summary
    assert "non-LRS S2 angular collision thermo-source factory" in feat.evidence_summary
    assert "non-LRS collision-feedback artifact runner" in feat.evidence_summary
    assert "non-LRS collision-feedback convergence runners" in feat.evidence_summary
    assert "non-LRS collision-feedback candidate gate" in feat.evidence_summary
    assert "source-update policy artifact" in feat.evidence_summary
    assert "source-update policy candidate gate" in feat.evidence_summary
    assert "angular collision-feedback 3T solve wrapper" in feat.evidence_summary
    assert "artifact reuse of that wrapper" in feat.evidence_summary
    assert "direct wrapper artifact runner" in feat.evidence_summary
    assert "direct wrapper candidate gate" in feat.evidence_summary
    assert "direct wrapper outcome policy/report runner" in feat.evidence_summary
    assert "pre-budget direct solver-matrix artifact" in feat.evidence_summary
    assert "source-update policy promotion study" in feat.evidence_summary
    assert "pre-budget direct-wrapper convergence artifact" in feat.evidence_summary
    assert "direct non-LRS `pstf_radial` collision-feedback 3T wrapper" in feat.evidence_summary
    assert "AP4/AP65 combined full-span candidate gate observables" in feat.notes
    assert "AP4/AP65 radial closure observables" in feat.notes
    assert "real `live_rhs` two-span ladder" in feat.notes
    assert "deterministic warm-preset `N_span=1e-8` ladder" in feat.notes
    assert "frozen-source Radau physical-preview preset" in feat.notes
    assert "isolated `N_span=1e-3` diagnostic evidence" in feat.notes
    assert "named `piecewise_physical_preview` preset" in feat.notes
    assert "piecewise refinement artifact" in feat.notes
    assert "`piecewise_frozen` source-refresh gate" in feat.notes
    assert "evolved electron charge-asymmetry state through a real `N_span=1e-4` smoke row" in feat.notes
    assert "shared AP6 radial-grid cache reuse" in feat.notes
    assert "FB-08 runs the fixed/charge-neutral electron-bath and finite/exact scalar-QED cross-product over chained rows" in feat.notes
    assert "scalar-QED one-hot diagnostics in every row" in feat.notes
    assert "FB-09 runs smoke chained N_q/N_mu/N_phi ladders" in feat.notes
    assert "source-evaluation and replay-budget metadata for every ladder point" in feat.notes
    assert "charge-neutral e-/e+ bath forwarding" in feat.notes
    assert "terminal electron-mu/charge-asymmetry maxima" in feat.notes
    assert "companion AP4/AP65 source-policy span profile" in feat.notes
    assert "collision-source budget report" in feat.evidence_summary
    assert "candidate evidence bundle" in feat.evidence_summary
    assert "physical sanity matrix" in feat.evidence_summary
    assert "QKE out of scope" in feat.evidence_summary
    assert "AP6 radial number/pair-energy closure observables" in feat.evidence_summary
    assert "frozen-source Radau physical-preview short-span evidence" in feat.evidence_summary
    assert "piecewise_frozen nonuniform N_span=1e-3 source-refresh evidence" in feat.evidence_summary
    assert "angular and pstf_radial collision sources carry optional kinetic dA" in feat.short_summary
    assert "not public dispatch" in feat.short_summary
    assert "AP4/AP65 combined full-span gate/profile with live-RHS two-span, warm-preset, frozen-source physical-preview short-span evidence, fixed/charge-neutral/exact-scalar-QED named piecewise_physical_preview source-refresh evidence including cross-control row and piecewise refinement, piecewise_frozen nonuniform N_span=1e-3 source-refresh evidence" in feat.short_summary
    assert "charge-neutral evolved charge-asymmetry handoff" in feat.short_summary
    assert "FB-08 electron-bath/scalar-QED chained cross-product rows" in feat.short_summary
    assert "FB-09 chained N_q/N_mu/N_phi resolution ladder rows" in feat.short_summary
    assert "FB-20 optional production-candidate evidence gate" in feat.short_summary

    blockers = " ".join(feat.blockers).lower()
    assert "default/promoted full-span angular kinetic+thermo collision feedback solve" in blockers
    assert "coupled-solve weak-rate convergence beyond the ap80/fb-07 diagnostic gates" in blockers
    assert "pstf collision-kernel solver/runtime coupling" in blockers
    assert "anisotropic/tensor qed response" in blockers
    assert "promotion-grade exact-scalar-qed full-span coupled-solver validation" in blockers
    assert "fb-08 chained scalar-qed control cross-product rows" in blockers
    assert "process families outside the supported no-qke hm" in blockers
    assert "all-nine diagonal-nu-nu catalog" in blockers
    assert "promotion-grade coupled-solver use of that catalog" in blockers
    assert "physical full-bbn span" in blockers
    assert "ap4/ap65 combined full-span candidate gate/source-policy profile" in blockers
    assert "promotion tolerances beyond fb-09 smoke chained ladders" in blockers
    assert "production smc" in blockers
    assert "ap76/ap79 readiness audit retained diagnostic staging" in blockers
    assert "smc" in blockers
