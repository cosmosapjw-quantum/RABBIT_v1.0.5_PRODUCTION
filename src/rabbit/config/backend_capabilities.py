"""Backend capability matrix for honesty locks and dispatch decisions.

Three historical maturity tiers:
  canonical   — regression-locked reference or bounded promoted surface
  candidate   — end-to-end functional, parity-checked, explicit opt-in only
  substrate   — module-level validated, not wired end-to-end

The dispatch table CAPABILITY_BY_BACKEND controls public forward authority.
F06 removes the retired JAX endpoint drivers from that table: only SciPy and
its ``auto`` alias remain dispatchable while Rust AOT is the SPECIFIED
migration target.  JAX entries retained in CAPABILITY_BY_KEY are frozen local
component/parity oracles or non-dispatchable geometry metadata; they grant no
endpoint, runtime-policy, inference, or public-production authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BackendCapability:
    key: str
    backend: str
    tier: str           # "canonical" | "candidate" | "substrate"
    physics_scope: str
    weak_mode: str
    max_correction_level: int
    supports_teff: bool
    max_thermo_tier: int
    validated_default: bool
    notes: str
    supports_live_weak_opt_in: bool = False
    # Teff diagnostic fields (Channel 2 spectral hardening)
    teff_kernel_validated: bool = False   # kernel parity vs NumPy < 1e-12
    teff_blocking_reason: str = ""        # why supports_teff=False (empty = not applicable)
    live_weak_species: Tuple[str, ...] = ()
    # Validation mode: "full" (canonical), "candidate" (default), "diagnostic" (AD), "exploratory" (inference)
    validation_mode: str = "candidate"
    # Surface presentation class: drives README/STATUS/PROMOTION grouping
    surface_class: str = ""  # auto-filled from tier if empty
    # Scope honesty contracts: machine-readable boundary descriptors exposed in
    # prediction metadata so users can distinguish core references from bounded
    # promoted surfaces and candidate perimeters.
    readiness_scope_contract: str = ""
    transport_scope_contract: str = ""
    thermo_scope_contract: str = ""
    collision_scope_contract: str = ""
    # backward compat alias
    @property
    def maturity(self) -> str:
        return self.tier

    @property
    def effective_surface_class(self) -> str:
        """Resolved surface_class (falls back to tier if empty)."""
        if self.surface_class:
            return self.surface_class
        return self.tier


# ═══════════════════════════════════════════════════════════════
# §1. SciPy — canonical reference
# ═══════════════════════════════════════════════════════════════

SCIPY_TYPEI_REFERENCE = BackendCapability(
    key="scipy_typeI_reference",
    backend="scipy",
    tier="canonical",
    surface_class="canonical",
    physics_scope="TypeI",
    weak_mode="live_f0_cl0_cl2",
    max_correction_level=3,
    supports_teff=False,
    teff_kernel_validated=True,
    max_thermo_tier=2,
    validated_default=True,
    validation_mode="full",
    readiness_scope_contract="typeI_canonical_core_reference_v1",
    transport_scope_contract="characteristic_typeI_reference_v1",
    thermo_scope_contract="tier1_baseline_thermo_v1",
    collision_scope_contract="collisionless_or_shared_characteristic_v1",
    notes=(
        "Regression-locked SciPy Type-I reference path. "
        "Gold table: CL0 Y_p=0.24238, CL2 Y_p=0.24810. "
        "Characteristic transport supersedes the legacy Teff closure on this path. "
        "Used as the conservative reference / fallback path."
    ),
)

SCIPY_TYPEI_TIER2_PER_SPECIES = BackendCapability(
    key="scipy_typeI_tier2_per_species",
    backend="scipy",
    tier="candidate",
    surface_class="candidate",
    physics_scope="TypeI_tier2_per_species",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=False,
    teff_kernel_validated=True,
    max_thermo_tier=2,
    validated_default=False,
    validation_mode="candidate",
    readiness_scope_contract="transitional_scipy_tier2_decoupling_backbone_v1",
    transport_scope_contract="characteristic_per_species_transport_v1",
    thermo_scope_contract="tier2_isotropic_decoupling_backbone_v1",
    collision_scope_contract="anisotropic_residual_relaxation_v1",
    notes=(
        "SciPy Type-I characteristic tier-2 per-species collision path. "
        "The current runtime uses an isotropic momentum-grid decoupling backbone "
        "for thermo and weak-background evolution, while the characteristic side "
        "uses a lightweight anisotropic residual relaxation closure. "
        "This is fenced against legacy shared mode by default and surfaced via "
        "transport_species_mode='per_species'. "
        "It remains a transitional candidate surface while the residual "
        "characteristic/collision split is being completed and relocked. "
        "It remains separate from the coarse auto/scipy dispatch key, which still "
        "points at the canonical tier-1 reference surface."
    ),
)

SCIPY_TYPEI_TIER3_WEAK_BUDGET = BackendCapability(
    key="scipy_typeI_tier3_weak_budget",
    backend="scipy",
    tier="candidate",
    surface_class="candidate",
    physics_scope="TypeI_tier3_weak_budget",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=False,
    teff_kernel_validated=True,
    max_thermo_tier=2,
    validated_default=False,
    validation_mode="candidate",
    readiness_scope_contract="transitional_scipy_tier3_decoupling_backbone_v1",
    transport_scope_contract="characteristic_per_species_transport_v1",
    thermo_scope_contract="tier2_isotropic_decoupling_backbone_v1",
    collision_scope_contract="anisotropic_residual_relaxation_v1",
    notes=(
        "SciPy Type-I CL3 weak-budget surface on top of the transitional "
        "isotropic decoupling backbone plus anisotropic residual relaxation "
        "(Born + Coulomb + Sirlin + recoil/weak-magnetism). "
        "Metadata remains reported via weak_budget_mode, but the residual "
        "collision closure is still a transitional candidate pending the full "
        "anisotropic-residual refactor."
    ),
)


# ═══════════════════════════════════════════════════════════════
# §2. JAX — frozen historical tiers (explicit compatibility/oracle use only)
# ═══════════════════════════════════════════════════════════════

JAX_TYPEI_REDUCED_CL0 = BackendCapability(
    key="jax_typeI_reduced_cl0",
    backend="jax",
    tier="candidate",
    surface_class="candidate",
    physics_scope="TypeI",
    weak_mode="equilibrium_fd_cl0",
    max_correction_level=0,
    supports_teff=False,
    max_thermo_tier=1,
    validated_default=False,
    readiness_scope_contract="candidate_jax_reduced_cl0_v1",
    transport_scope_contract="linearized_pstf_lmax2_v1",
    thermo_scope_contract="tier1_single_T_nu_v1",
    collision_scope_contract="no_collision_coupled_transport_v1",
    notes=(
        "JAX Type-I with equilibrium-FD CL0 weak rates, "
        "tier-1 thermo, ℓ_max=2 linearized PSTF. "
        "Opt-in via use_live_weak_monopoles=False. "
        "7.2% Yp offset vs SciPy due to missing spectral distortions."
    ),
)

JAX_TYPEI_LIVEWEAK_CL0 = BackendCapability(
    key="jax_typeI_liveweak_cl0",
    backend="jax",
    tier="candidate",
    surface_class="candidate",
    physics_scope="TypeI",
    weak_mode="live_f0_cl0",
    max_correction_level=0,
    supports_teff=False,
    max_thermo_tier=1,
    validated_default=False,
    validation_mode="full",
    readiness_scope_contract="candidate_jax_liveweak_cl0_v1",
    transport_scope_contract="linearized_pstf_lmax2_v1",
    thermo_scope_contract="tier1_single_T_nu_v1",
    collision_scope_contract="no_collision_coupled_transport_v1",
    notes=(
        "Frozen JAX Type-I live-weak compatibility/parity path. "
        "Its CPU-first exact-execution behavior is retained for bounded regression "
        "checks only; no new performance or runtime-development work is authorized."
    ),
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar"),
)

JAX_TYPEI_CHARACTERISTIC_TIER1 = BackendCapability(
    key="jax_typeI_characteristic_tier1",
    backend="jax",
    tier="canonical",
    surface_class="canonical",
    physics_scope="TypeI",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=1,
    validated_default=False,
    validation_mode="full",
    readiness_scope_contract="bounded_jax_characteristic_tier1_lrs_cpu_first_v1",
    transport_scope_contract="characteristic_lrs_exact_v1",
    thermo_scope_contract="tier1_single_T_nu_v1",
    collision_scope_contract="no_collision_coupled_transport_v1",
    notes=(
        "Frozen JAX Type-I characteristic-ray tier-1 parity-oracle surface "
        "(historical registry tier: canonical; no active runtime authority). "
        "Exact nonperturbative collisionless transport via the method of "
        "characteristics (Paper §6), live weak rates from the transported "
        "monopole f̃₀(q), PRIMAT-AC2024 backbone/standard network. "
        "Parity with the SciPy characteristic reference driver is |ΔY_p| ≲ 5e-8 "
        "across Σ_H ∈ [0, 0.5] and CL0-CL2 at N_μ=12, N_q=20. "
        "This historical bounded path is FLRW-anchored "
        "against the SciPy reference and PRIMAT/Mangano; the finite-shear "
        "anisotropic response is internally regression-locked only — there is "
        "no external anisotropic anchor (no comparable external Bianchi-I BBN "
        "code exists)."
    ),
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar"),
)

JAX_TYPEI_CHARACTERISTIC_TIER2 = BackendCapability(
    key="jax_typeI_characteristic_tier2",
    backend="jax",
    tier="canonical",
    surface_class="canonical",
    physics_scope="TypeI_tier2_3T",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=2,
    validated_default=False,
    validation_mode="full",
    readiness_scope_contract="bounded_jax_characteristic_tier2_3T_lrs_cpu_first_v1",
    transport_scope_contract="characteristic_lrs_exact_v1",
    thermo_scope_contract="tier2_3T_bounded_v1",
    collision_scope_contract="momentum_averaged_nu_e_energy_transfer_v1",
    notes=(
        "Frozen JAX Type-I characteristic-ray tier-2 parity-oracle surface "
        "(historical registry tier: canonical; no active runtime authority). "
        "Exact nonperturbative collisionless characteristic transport combined "
        "with the coupled three-temperature plasma–neutrino thermodynamics "
        "(Paper §11.3, §11.6).  The neutrino reheating source is the "
        "momentum-averaged Mangano-style ν–e energy transfer "
        "(T_γ − T_ν) · T_γ⁴ T_ν⁴ with the F(m_e/T) suppression; weak rates "
        "are evaluated from the transported monopole at T_νₑ.  Landing "
        "N_eff ≈ 3.034 at FLRW (within ~0.3% of the SM benchmark 3.044; the "
        "residual ~0.3% gap is documented, NOT closed — tier-3 closure (the "
        "full-collision upgrade) is deferred: the thermo-tier-3 transport path "
        "was unimplemented in the now-retired forward driver, so the gap is not "
        "yet demonstrated to close)."
    ),
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar"),
)

JAX_TYPEI_CHARACTERISTIC_NONLRS_TIER1 = BackendCapability(
    key="jax_typeI_characteristic_nonlrs_tier1",
    backend="jax",
    tier="candidate",
    surface_class="candidate",
    physics_scope="TypeI_nonlrs_characteristic",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=2,
    validated_default=False,
    validation_mode="candidate",
    readiness_scope_contract="candidate_jax_characteristic_nonlrs_tier1_cpu_first_v1",
    transport_scope_contract="characteristic_nonlrs_exact_v1",
    thermo_scope_contract="tier1_single_Tnu_or_tier2_3T_nonlrs_candidate_v1",
    collision_scope_contract="no_collision_coupled_transport_v1",
    notes=(
        "Frozen JAX Type-I non-LRS characteristic-ray tier-1/tier-2 oracle surface. "
        "Generic orthogonal Type I (Sigma_plus, Sigma_minus) on a tensor-product "
        "S^2 ray grid with exact diagonal-stretch direction map and analytic "
        "non-LRS intensity/Jacobian reconstruction from accumulated shear "
        "integrals. Tier-2 3T thermodynamics is opt-in via jax_thermo_tier=2; "
        "collision-coupled non-LRS public dispatch remains out of scope. A private "
        "diagnostic helper, run_nonlrs_tier2_residual_state_jax(...), now carries "
        "explicit S^2 per-species residual R_I/R_J state for anisotropic residual "
        "relaxation smoke tests without promoting enable_collisions=True. The "
        "explicit backend name remains compatibility-only."
    ),
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar"),
)

JAX_TYPEI_LIVEWEAK_CL3_TIER1 = BackendCapability(
    key="jax_typeI_liveweak_cl3_tier1",
    backend="jax",
    tier="canonical",
    surface_class="canonical",
    physics_scope="TypeI",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=1,
    validated_default=False,
    validation_mode="full",
    readiness_scope_contract="bounded_jax_tier1_cpu_first_v1",
    transport_scope_contract="linearized_pstf_lmax2_v1",
    thermo_scope_contract="tier1_single_T_nu_v1",
    collision_scope_contract="no_collision_coupled_transport_v1",
    notes=(
        "Frozen JAX Type-I tier-1 live-weak parity-oracle surface "
        "(historical registry tier: canonical; no active runtime authority) "
        "(Born + Coulomb + Sirlin + recoil/WM with single-T_nu thermo). "
        "FLRW parity versus SciPy is locked at the few-ppm Y_p level across CL0-CL3, "
        "within its frozen regression scope. "
        "This compatibility surface requires explicit backend selection or the "
        "RABBIT_AUTO_PREFER_JAX opt-in; backend='auto' defaults to the SciPy "
        "Type-I reference. "
        "NONZERO-SHEAR CAVEAT (BD599 S2-R1): this surface uses the linearized_pstf "
        "(ell=2) closure, which captures only the linear-order shear response and "
        "substantially under-predicts the Y_p shear signal at nonzero Sigma_H. Only "
        "FLRW (Sigma_H=0) parity is locked; nonzero-shear yields are NOT shear-faithful. "
        "Use backend='jax_characteristic' (method of characteristics, no ell-truncation) "
        "for the shear-faithful path."
    ),
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar"),
)

JAX_TYPEI_LIVEWEAK_CL3_TIER2 = BackendCapability(
    key="jax_typeI_liveweak_cl3_tier2",
    backend="jax",
    tier="canonical",
    surface_class="canonical",
    physics_scope="TypeI_tier2_3T",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=2,
    validated_default=False,
    validation_mode="full",
    readiness_scope_contract="bounded_jax_tier2_3T_compatibility_v1",
    transport_scope_contract="linearized_pstf_lmax2_v1",
    thermo_scope_contract="tier2_3T_bounded_v1",
    collision_scope_contract="no_collision_coupled_transport_v1",
    notes=(
        "Frozen JAX Type-I tier-2 3T live-weak compatibility/oracle surface "
        "(Born + Coulomb + Sirlin + recoil/WM, Teff off). "
        "The historical explicit no-Teff `backend='jax_advanced'` dispatch resolves "
        "to the tier-3 weak-budget surface across CL0-CL3; this key remains frozen "
        "in the catalog for compatibility with older reports and fixtures."
    ),
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar"),
)

JAX_TYPEI_TIER3_WEAK_BUDGET = BackendCapability(
    key="jax_typeI_tier3_weak_budget",
    backend="jax",
    tier="canonical",
    surface_class="canonical",
    physics_scope="TypeI_tier3_weak_budget",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=2,
    validated_default=False,
    validation_mode="full",
    readiness_scope_contract="bounded_jax_tier3_no_teff_cpu_first_v1",
    transport_scope_contract="linearized_pstf_lmax2_v1",
    thermo_scope_contract="tier2_3T_bounded_v1",
    collision_scope_contract="no_collision_coupled_transport_v1",
    notes=(
        "Frozen JAX Type-I tier-3 weak-budget compatibility/oracle surface on the "
        "3T live-weak path (historical registry tier: canonical) "
        "(Teff off) with CL0-CL3 support. "
        "The active weak sub-budget is reported via weak_budget_mode="
        "'born'/'born+coulomb'/'born+coulomb+sirlin'/'born+coulomb+sirlin+finite_mass'. "
        "The explicit no-Teff `backend='jax_advanced'` surface is retained only for "
        "bounded CPU parity cross-checks and historical fixtures."
    ),
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar"),
)

JAX_TYPEI_AP_UNIFIED_TIER3_CANDIDATE = BackendCapability(
    key="jax_typeI_ap_unified_tier3_candidate",
    backend="jax",
    tier="candidate",
    surface_class="candidate",
    physics_scope="TypeI_tier3_ap_unified_canonical_track",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=2,
    validated_default=False,
    readiness_scope_contract="canonical_track_jax_tier3_ap_unified_v1",
    transport_scope_contract="full_phase_space_ray_v1",
    thermo_scope_contract="private_tier2_collision_moment_3T_v1",
    collision_scope_contract="ap_unified_preflight_v1",
    notes=(
        "Frozen JAX tier-3 incomplete-decoupling AP-form compatibility surface "
        "with historical PR-T3B/PR-T3D evidence. The explicit "
        "canonical_forward_solver(backend='jax_ap_unified_tier3', ...) name remains "
        "callable for bounded regression only. It combines spectral_relaxation's "
        "full-FD psi_target + damping projection (anisotropy "
        "stability) with projected_physical's grid scaling and a "
        "sign-safe soft total-rate enforcement clip ([0.1, 5] vs "
        "spectral's [0, 100]) that avoids the ill-conditioning breaking spectral "
        "above (4, 6).  **Full CL0-CL3 ladder supported** (Born / "
        "+Coulomb+Sirlin / +finite-mass weak corrections; PR-T3 full "
        "CL envelope locked in tests/test_pr_t3_full_cl_envelope.py): "
        "Y_p increases monotonically with CL (CL0=0.2414, CL1=0.2453, "
        "CL2=0.2470, CL3=0.2475 at FLRW), and N_eff is CL-invariant "
        "at 3.0345 across the ladder.  Canonical milestone metrics: "
        "FLRW N_eff = 3.0345 (gap +0.0095 to Mangano 3.044, the "
        "documented AP-form model-approximation limit, surfaced via "
        "BBNPrediction.metadata.flrw_mangano_gap_documented); "
        "the phase-space driver now uses the canonical 3T Mangano "
        "prefactor directly and production tests lock bidirectional "
        "heating/cooling energy-moment parity with the 3T target rates; "
        "the public dispatch exposes explicit diagonal no-QKE nu-nu "
        "options via jax_tier3_nu_nu='3t' or 'spectral' plus an "
        "accuracy-candidate mode jax_tier3_nu_nu='accuracy' that combines "
        "the spectral bank with a calibrated AP energy-transfer lift while keeping "
        "the PR-T3D baseline as the default 'off' mode; the 3T and "
        "spectral options share the same leading-order diagonal nu-nu "
        "T^5 rate prefactor and are public-forward compared in the "
        "overlap regime; the spectral option also damps same-sector "
        "spectral shape distortions with an explicit number/energy-neutral "
        "projection, so equal-temperature non-FD common shapes are not "
        "mistaken for an inactive source; the accuracy candidate raises the "
        "bounded FLRW smoke-grid N_eff to the Mangano/NUDEC target envelope "
        "without claiming QKE flavour physics; "
        "public CL3 FLRW dispatch is cross-code locked against "
        "PRIMAT-AC2024 Y_p at 5e-4 and the standard BBN D/H window in "
        "tests/test_flrw_external_parity_public.py; "
        "grid-converged across (4,6)/(8,12)/(12,20) with spread "
        "<1e-4 for every CL; anisotropy spread <1e-3 across "
        "Sigma_H ∈ {0, 0.05, 0.10} for every CL, PASSING the "
        "canonical PR-T3D §5 stability gate.  Compatible with the "
        "historically load-bearing Rodas5P invariant + JAX/GPU-friendly "
        "architecture; no IMEX or operator splitting.  The remaining "
        "0.0095 Mangano gap is the AP-form residual; the historical Option-E "
        "Option E (in-RHS analytic relaxation pre-conditioner) may "
        "close it toward 5e-3.  See "
        "docs/audit/PR-T3B_jax_kernel_runtime.md and "
        "docs/research/PR-T3B_option_E_canonical_post_enhancement.md."
    ),
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar", "nux", "nuxbar"),
)


JAX_TYPEI_FULL_BOLTZMANN_TIER3_PREFLIGHT = BackendCapability(
    key="jax_typeI_full_boltzmann_tier3_preflight",
    backend="jax",
    tier="candidate",
    surface_class="candidate",
    physics_scope="TypeI_tier3_full_collision_preflight",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=2,
    validated_default=False,
    readiness_scope_contract="bounded_jax_tier3_full_boltzmann_preflight_v1",
    transport_scope_contract="full_phase_space_ray_v1",
    thermo_scope_contract="private_tier2_collision_moment_3T_v1",
    collision_scope_contract="jax_kernel_plus_diagonal_nu_nu_skeleton_preflight_v1",
    notes=(
        "JAX tier-3 full-Boltzmann incomplete-decoupling preflight surface. "
        "Aggregates the prior PR-T3A/T3B/T3C preflight slices: per-ray per-momentum "
        "transport (driver_typeI_full_boltzmann), separate ν_e/ν̄_e live-f0 "
        "weak-rate coupling in the full RHS and low-rank Jacobian "
        "(including the collisionless factorized Jacobian dense-AD parity gate; "
        "tests/test_full_boltzmann_separate_weak_coupling.py and "
        "tests/test_pr_t3a_collisionless_driver.py), "
        "temperature^4 and heavy-flavour degeneracy weighted species "
        "stress coupling in the full RHS and low-rank Jacobian "
        "(tests/test_full_boltzmann_species_stress_weighting.py), pure-JAX nu-e elastic + pair "
        "operators (collisions_jax) wired through collision_mode='jax_kernel_preflight', "
        "temperature-frame remapped kernel diagnostics via "
        "collision_mode='jax_kernel_remap_preflight', and a remapped-kernel "
        "Jacobian-preconditioned full-ODE diagnostic via "
        "collision_mode='jax_kernel_remap_preconditioned_preflight' "
        "(same remapped RHS as jax_kernel_remap_preflight; modified "
        "collision-bank Jacobian only; bounded FLRW smoke N_eff ≈ 3.0759, "
        "production-to-tight tolerance gap ≈ 0.0015 in N_eff, but "
        "N_q=6/8/10 grid spread ≈ 0.0205 so not canonical), "
        "plus a remapped-kernel "
        "spectral-shape + Mangano-moment projected ODE path via "
        "collision_mode='jax_kernel_projected_preflight' "
        "(tests/test_jax_kernel_temperature_remap.py and "
        "tests/test_jax_kernel_projected_preflight.py, plus "
        "tests/test_jax_kernel_remap_preconditioned.py), "
        "a structural diagonal nu-nu skeleton (nu_nu_scattering_jax), and "
        "collision_mode='ap_unified_nu_nu_preflight' with an "
        "energy-conserving diagonal nu-nu 3T equilibration source wired "
        "into the full ODE (tests/test_nu_nu_3t_equilibration.py; "
        "bounded FLRW smoke N_eff ≈ 3.0357 and reduced final "
        "T_nue/T_nux split relative to ap_unified_preflight), and "
        "collision_mode='ap_unified_nu_nu_spectral_preflight' with the "
        "diagonal nu-nu spectral bank shape wired into the runtime and "
        "projected to the energy-conserving 3T moment, plus a "
        "number/energy-neutral same-sector self-thermalization shape "
        "damping term "
        "(tests/test_nu_nu_spectral_bank_coupling.py; bounded FLRW smoke "
        "N_eff ≈ 3.03572). "
        "NOT promoted to canonical: the Dolgov-Hansen-Semikoz coefficient table is "
        "not yet calibrated for the full spectral runtime kernel, the "
        "off-grid f_beta(y_4) interpolation residual bounds the structural "
        "kernel energy-conservation lock, and there is no "
        "FLRW |N_eff - 3.044| lock against Mangano 2005 / LASAGNA / FortEPiaNO. "
        "Registered in CAPABILITY_BY_KEY for introspection only; no entry in "
        "CAPABILITY_BY_BACKEND or canonical_forward_solver dispatch."
    ),
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar", "nux", "nuxbar"),
)


JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE = BackendCapability(
    key="jax_typeI_liveweak_cl3_tier2_teff_candidate",
    backend="jax",
    tier="substrate",
    surface_class="diagnostic",
    physics_scope="TypeI_tier2_3T_teff_legacy",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=False,
    teff_kernel_validated=True,
    teff_blocking_reason="deprecated legacy closure; public forward-solver runtime rejects enable_teff=True",
    max_thermo_tier=2,
    validated_default=False,
    readiness_scope_contract="deprecated_teff_kernel_diagnostic_only_v1",
    transport_scope_contract="linearized_pstf_lmax2_v1",
    thermo_scope_contract="tier2_3T_teff_legacy_diagnostic_v1",
    collision_scope_contract="teff_legacy_kernel_only_v1",
    notes=(
        "Deprecated legacy Teff closure. Kernel parity is retained for diagnostics, "
        "but no public forward-solver backend advertises active Teff support."
    ),
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar"),
)


# ═══════════════════════════════════════════════════════════════
# §3. JAX — substrate tiers (module-level, not end-to-end)
# ═══════════════════════════════════════════════════════════════

JAX_TYPEI_AUGMENTED_PSTF_NOQKE_STAGING = BackendCapability(
    key="jax_typeI_augmented_pstf_noqke_staging",
    backend="jax",
    tier="substrate",
    surface_class="diagnostic",
    physics_scope="TypeI_augmented_pstf_noqke_staging",
    weak_mode="live_monopole_lrs_cl3_moment_input_nonlrs_s2_moment_input",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=0,
    validated_default=False,
    validation_mode="diagnostic",
    readiness_scope_contract="substrate_typeI_augmented_pstf_noqke_staging_v1",
    transport_scope_contract="lrs_collisionless_nonlrs_source_and_nonlinear_staging_v1",
    thermo_scope_contract="not_full_bbn_thermo_v1",
    collision_scope_contract="deterministic_moment_feedback_and_nodal_projection_staging_v1",
    notes=(
        "Catalog-only augmented-PSTF Type-I no-QKE staging with no public "
        "dispatch.  HISTORICAL: the AP0-AP81 surface below is a stage-record; "
        "most backing code was deleted in PR-D1..D3 and is NOT current "
        "capability (surviving substrate: augmented_pstf_distribution, "
        "augmented_nonlrs_transport, augmented_typeI_nonlrs_collisionless, "
        "augmented_typeI_observables, jax/characteristic_rays_nonlrs_jax).  "
        "The staged AP0-AP81 surface included SciPy/JAX fixed-grid "
        "augmented distribution reconstruction and projection, SciPy LRS "
        "collisionless and source-only non-LRS projection, coevolution, and convergence gates, "
        "QMC deterministic replay validation for scalar controls, nodal df/dN "
        "sources, and collision source-moment vector reports, a JAX "
        "nodal projection bridge, and a CPU-JAX LRS projected-source solve "
        "through the existing Rodas5P Rosenbrock core, plus a SciPy live "
        "weak/network RHS bridge, narrow fixed-thermo solve_ivp shell, and "
        "SciPy 3T thermo/Hubble solve shell, 3T ell/q/angular convergence "
        "runners, a deterministic JSON artifact runner for smoke/extended "
        "3T convergence ladders, and an opt-in collision-moment thermo "
        "feedback hook for the 3T shell, plus a deterministic diagonal no-QKE "
        "nu-nu thermo source factory and a deterministic monopole nu-e plus "
        "pair-process thermo source factory from current augmented monopoles, "
        "an explicit combined source factory summing both moment sources, "
        "a deterministic collision-feedback variant artifact runner, "
        "standard-relative artifact deltas, and stage-scoped WBS status "
        "normalization for AP10-AP23, AP25 bounded LRS Sigma_+ K_2 "
        "CL3 weak-rate multiplier wiring, and an AP26 smoke-scale "
        "weak-rate candidate sub-gate for that multiplier, plus an AP27 "
        "collision-feedback source-variant candidate sub-gate with AP36 LRS source-policy routing, AP28 "
        "staged 3T span-ladder candidate gate over explicit N_span values with the same LRS source-policy routing, "
        "AP29 collision-feedback 3T ell/q/angular convergence runners with AP36 LRS source-policy routing, "
        "AP30 QMC collision source-moment control-variate reports, "
        "AP31 live LRS angular-node nu-e scattering projection into PSTF modes "
        "with elastic-scattering number closure, "
        "AP32 live LRS angular-node electromagnetic pair-process projection "
        "with number-closed scattering component diagnostics, "
        "AP33 live LRS angular-node diagonal no-QKE nu-nu projection, "
        "AP34 generic angular-node v2 contracts with non-LRS S2 projection coverage, "
        "an AP35 opt-in angular collision thermo-source callback and LRS 3T direct wrapper for the AP18 hook, "
        "AP36 angular source-variant artifact/gate/convergence plumbing routed through the AP35 LRS direct wrapper, and "
        "AP37 source-only non-LRS plus/minus coevolution shell wiring, and "
        "AP38 source-only non-LRS fixed-thermo weak/network shell wiring, and "
        "AP39 source-only non-LRS 3T thermo/Hubble shell wiring, and "
        "AP40 opt-in non-LRS collision-moment thermo feedback hook wiring, and "
        "AP41 non-LRS S2 angular collision thermo-source factory wiring, and "
        "AP42 non-LRS collision-feedback artifact wiring for angular and AP6 "
        "`pstf_radial` source variants with a smoke-default frozen-initial-state "
        "source-moment policy plus explicit live_rhs option, and "
        "AP43 non-LRS collision-feedback q/N_mu/N_phi convergence runners, and "
        "AP44 non-LRS collision-feedback candidate-gate wiring over the AP42 artifact "
        "including the AP6 `pstf_radial` source-evaluation budget, and "
        "AP45 source-update policy artifact wiring for very short-span live_rhs smoke comparison "
        "with AP6 `pstf_radial` budget forwarding plus a dedicated non-LRS radial "
        "source-policy artifact/writer, and "
        "AP46 source-update policy candidate-gate wiring with radial budget observables, AP47 opt-in non-LRS "
        "angular collision-feedback 3T solve wrapper wiring, and AP48 artifact "
        "routing through that wrapper, and AP49 direct wrapper JSON artifact "
        "runner wiring, AP50 direct wrapper candidate-gate wiring, AP51 "
        "direct wrapper outcome policy/report wiring, AP52 pre-budget "
        "direct solver-matrix artifact wiring, AP53 source-update "
        "policy promotion-study wiring over AP42/AP51 evidence including the AP42 "
        "`pstf_radial` budgeted surface, AP54 "
        "pre-budget direct-wrapper convergence ladder wiring, and AP55 "
        "collision-source budget artifact wiring for FLRW quiet, LRS sign/energy "
        "redistribution, component-sum, and non-LRS S2 source-context checks, and AP56 "
        "candidate evidence bundle wiring over AP42/AP49/AP51-AP55 summaries, AP57 "
        "physical sanity-matrix wiring over AP56/AP55 evidence, and AP58 "
        "explicit LRS/non-LRS per-q angular weak-rate input objects with "
        "fail-closed angular-mode resolution, and AP59 opt-in LRS CL3 "
        "moment-input angular weak-rate factors for lambda_np/lambda_pn, and AP60 "
        "non-LRS S2 plus/minus moment-input angular weak-rate factors now used by the "
        "staged correction-level-3 non-LRS default with explicit metadata-only controls, and AP61 "
        "deterministic weak-rate candidate/convergence gate wiring over those LRS/non-LRS "
        "rate factors, and AP62 non-LRS S2 nonlinear transport logit-RHS operator wiring "
        "with LRS-oracle reduction, and AP63 nonlinear transport solve-shell wiring "
        "without weak/network or collision feedback, and AP64 nonlinear non-LRS "
        "3T weak/network candidate shell wiring with explicit source-only/nonlinear "
        "transport routing, AP65 opt-in nonlinear angular collision-feedback "
        "3T wrapper wiring with live/frozen source-update policy handling plus an "
        "AP65 combined angular+pstf_radial nonlinear wrapper/artifact, an AP4/AP65 "
        "combined-source full-span candidate gate over explicit span ladders including real live-RHS two-span, warm-preset N_span=1e-8 budget evidence, frozen-source physical-preview short-span evidence, piecewise_frozen nonuniform N_span=1e-3 source-refresh evidence with state handoff, a fixed/charge-neutral/exact-scalar-QED named piecewise_physical_preview preset with routine N_span=1e-4 plus isolated N_span=1e-3 evidence including charge-neutral plus exact scalar-QED cross-control evidence, a piecewise source-refresh refinement artifact, charge-neutral N_span=1e-4 evolved charge-asymmetry handoff evidence, JSON-safe terminal-state restart payloads for chained diagnostics and CPU-JAX/Rodas5P replay handoff, an FB-02 chained-window runner with restart/resume equivalence and CPU-JAX/Rodas5P replay vectors, FB-03 drift-budgeted adaptive source-refresh scheduling with uniform compatibility and per-window driver diagnostics, FB-04 CPU-JAX/Rodas5P pretabulated window-map replay solves with pass/error metadata plus an optional live-source RHS sidecar that evaluates nonlinear non-LRS S2 logit transport, 3T/Hubble, live weak monopoles, and PRIMAT network blocks inside Rodas5P, plus terminal and sidecar Yp/DH/N_eff_3T/Sigma_H/Schramm readouts, sidecar-vs-terminal BBN deltas, restart handoff kwargs, and a smoke live-source RHS chain runner/CLI with optional full-chain diagnostic comparison, frozen terminal collision payloads or dynamic restart-state payload rebuilds with supplied/applied/dynamic payload counts plus provenance fingerprints, opt-in staged repeated-run readout source, FB-21 live-source repeated-run diagnostic gate rows with finite BBN/comparison/collision-payload provenance checks, an FB-24 multi-row live-source repeated-run profile gate over tiny chained span layouts, optional FB-25 passive profile handoff into downstream evidence-chain manifests, FB-26 opt-in dynamic restart-state collision-payload refresh at live-source chain window boundaries, FB-27 increasing-span dynamic live-source profile artifacts plus generated diagnostic BBN/shear/payload plots, optional FB-28 passive FB-27 profile/plot handoff into downstream evidence-chain manifests, an FB-29 smoke/extended preset diagnostic bundle for the FB-27 profile plus plots, an FB-30 diagnostic-long preset bundle with fail-closed BBN observable-bound metadata, FB-31 chain-local AP6 radial-grid cache reuse for dynamic collision payload refresh, FB-32 deterministic collision-reference vectorization for AP41 `nu-e`, pair, and diagonal `nu-nu` contractions, FB-33 batched AP41 angular collision-reference dispatch, FB-34 dynamic AP41/AP6 source-factory cache reuse, FB-35 dynamic cache-hit overhead reduction, and FB-36 dynamic live-source E2E BBN full-chain/AP68 surface wiring, FB-05 live-RHS micro-window comparator rows over chained restart states, FB-06 per-window electromagnetic and all-nine diagonal nu-nu closure ledger rows over chained states, FB-07 paired chained weak-rate convergence rows with terminal lambda/CL3 metadata, FB-08 electron-bath/scalar-QED chained cross-product rows with evolved charge-asymmetry and scalar-QED one-hot diagnostics, FB-09 chained N_q/N_mu/N_phi resolution ladder rows with terminal BBN observable deltas and source-budget metadata, fixed/charge-neutral electron chemical-potential forwarding, scalar QED-model routing including exact_finite_mu_scalar smoke evidence, terminal electron-mu/charge-asymmetry observables, shared AP6 radial-grid cache reuse across span ladders, plus a companion frozen/live source-policy span-profile artifact with the same shared cache, and AP66 "
        "publication-candidate convergence matrix wiring over N_q/N_mu/N_phi, span, "
        "tolerance, source-model, source-policy, weak-rate-mode, electron-bath control, and scalar-QED model rows, including "
        "opt-in combined angular+pstf_radial component diagnostics, AP4-backed piecewise_frozen combined-source rows with explicit subspan-end/source-refresh observables, optional FB-09 chained full-BBN resolution artifact links that preserve no-QKE/not-public-dispatch scope, plus terminal electron-mu/charge-asymmetry observables and exact_finite_mu_scalar row markers, AP67 known-limit "
        "validation-atlas wiring over FLRW/null, LRS, non-LRS, injected, and stress cases with fixed/charge-neutral "
        "electron-bath controls plus terminal electron-mu/charge-asymmetry observables, exact_finite_mu_scalar scalar-QED atlas markers, AP66 source-update policy/subspan evidence-link provenance for piecewise_frozen rows, and AP66/FB-09 full-chain evidence-link readiness metadata, AP68 "
        "guarded inference-adapter wiring around the AP65 candidate solve through the existing "
        "ForwardModel/BBNLikelihood interface with radial momentum-delta, AP4-style piecewise_frozen combined-source subspan refresh, FB-12 full-chain mode over the FB-04 chained runner or cached chained artifacts, optional FB-21 live-source repeated-run gate metadata plus FB-36 dynamic payload-refresh counters, electron-bath, and scalar-QED control forwarding plus terminal electron-mu/charge-asymmetry, scalar-QED, and full-chain metadata, and AP69 augmented SMC likelihood-schema wiring "
        "with priors, vector adapters, fixed solver/radial momentum-delta/electron-bath/scalar-QED controls plus AP68 piecewise_frozen source-refresh subspan controls and FB-13 full-chain execution/window/cache/source-refresh/replay/restart controls, AP68 radial momentum-delta/electron-bath/scalar-QED/source-refresh/full-chain provenance, and failure-aware log-likelihoods, AP70 "
        "smoke tempered-SMC runner wiring with replayable seeds, ESS/resampling, optional rejuvenation, "
        "failure accounting, and source-model/source-refresh/radial/electron-bath/scalar-QED/full-chain metadata preservation, and AP71 SMC runtime/cache wiring with checkpoint/resume, "
        "duplicate-call avoidance, source-model/source-refresh/radial/electron-bath-control-aware cache keys plus scalar-QED-aware cache context and full-chain-aware cache context, batched likelihood hooks, incompatible-manifest rejection, "
        "portable run manifests, CLI dry-run control forwarding including N_span end and piecewise_frozen subspan ends, last successful prediction metadata preservation, and failure-metadata preservation, plus AP72 synthetic "
        "null/injection SMC validation artifacts with posterior summaries, ESS/acceptance, "
        "logZ-error proxy, forward-success accounting, and source-model/source-refresh/N_span/radial/electron-bath/scalar-QED provenance, plus an opt-in AP72 full-chain physical smoke row with AP68 terminal BBN observables, CPU-JAX/Rodas5P replay status, and optional live-source repeated-run BBN readout plus FB-21 gate provenance, and AP73 figure-ready "
        "publication artifact tables with schema versions, provenance, units, registry keys, "
        "source-model/source-refresh/N_span/radial-momentum-delta/electron-bath row context plus scalar-QED row context, stale/mixed-contract rejection, and convergence/validation/SMC/Schramm rows, plus AP72 full-chain physical-smoke Schramm rows carrying AP68 terminal BBN observables, CPU-JAX/Rodas5P replay status, and optional live-source repeated-run BBN readout plus FB-21 gate provenance, and AP74 "
        "diagnostic augmented publication plots with PNG output, plot hashes, manifest metadata, "
        "source-model/source-refresh/N_span/radial-momentum-delta/electron-bath plot provenance plus scalar-QED plot provenance, AP72 full-chain physical-smoke Schramm plot provenance with completed-window, CPU-JAX/Rodas5P replay status, and optional live-source repeated-run BBN readout plus FB-21 gate contract/claim/window/payload/delta provenance, registry linkage, and report/paper regeneration hooks, and AP75 "
        "diagnostic reproducibility-bundle packaging with required artifact provenance, "
        "source-model/source-refresh/N_span/radial-momentum-delta/electron-bath bundle provenance plus scalar-QED bundle provenance, plot hash verification, copied artifact/figure manifests, environment metadata, "
        "command records, claim-boundary notes, AP72 full-chain physical-smoke bundle summaries with finite BBN observables, CPU-JAX/Rodas5P replay status, and optional live-source repeated-run BBN readout plus FB-21 gate contract/claim/window/payload/delta provenance, and AP76 final readiness-audit "
        "wiring that retains diagnostic staging with a not_promoted decision, source-model/source-refresh/N_span/radial-momentum-delta/electron-bath readiness-ledger provenance plus scalar-QED readiness-ledger provenance, and AP72 full-chain physical-smoke readiness-ledger provenance that accepts live-source repeated-run BBN readout only with matching FB-21 gate contract/claim/window/payload/delta provenance, plus AP77 "
        "coupled weak-rate gate wiring that exercises the AP60 non-LRS S2 angular weak-rate "
        "mode inside the AP65 3T solve against same-CL3 metadata-only control rows with electron-bath control forwarding, AP78 "
        "same-CL3 weak-rate control hardening for the AP66 publication matrix, AP79 "
        "readiness-audit linkage that requires AP77 coupled gate evidence with electron-bath control provenance, AP80 "
        "profile-level coupled weak-rate convergence artifact wiring over AP77 "
        "smoke/extended q-ladder reports plus FB-07 paired chained-row weak-rate "
        "diagnostics, FB-20 optional production-candidate gate wiring over AP79 "
        "full-chain readiness evidence and AP80 extended weak-rate profiles, "
        "FB-08 electron-bath/scalar-QED chained cross-product rows, FB-09 "
        "chained N_q/N_mu/N_phi resolution ladder rows, and private FB20-FB89 "
        "diagnostic evidence categories covering live-source repeated-run, "
        "payload/provenance handoff, vectorization/cache profiling, current-artifact "
        "figure/SMC/readiness wrappers, full-BBN freedom/residual/weak-rate "
        "diagnostics, and continuous-AP65 RHS/span/positivity/failure probes.  "
        "FB-20 is only an optional production-candidate evidence gate and FB-21 "
        "is only a live-source repeated-run diagnostic gate; none of these FB "
        "witnesses are promotion decisions, AP81 "
        "six-monomial Pauli statistical-factor wiring for diagonal no-QKE 2-to-2 collision kernels across deterministic nu-e/pair references, the staged NumPy AP19/AP33/AP35/AP41 diagonal-nu-nu source bridge with all-nine ordered pairwise bank coverage including identical-bank self-scattering diagnostics, legacy SciPy nu-e/pair operators, and JAX nu-e/pair/diagonal-nu-nu kernels, and AP6 local PSTF six-monomial angular contraction table plus universal `G0`, `G_mu`, and `G_mumu` geometric kernel tables, channel `K` assembly from HM-style `Pi_ij`/`Pi_ij Pi_kl` descriptors plus `m_e^4` terms, radial p4 contraction via radial-grid `p2,p3` summation with linear `p4 = E1 + E2 - E3` interpolation, radial channel kernel-grid assembly with invariant prefactors, normalized unit-direction momentum-delta weights, and an opt-in p-dependent radial Gaussian momentum-delta closure for `K34`, `K12`, `K123`, `K124`, `K134`, and `K234` mode products, UR physical HM process descriptors, charge-split finite-mass HM elastic nu-e_minus/nu-e_plus descriptors, finite-mass HM pair-annihilation descriptors, plus a default supported-species `{nue,nuebar,nux}` descriptor catalog whose electromagnetic entries use finite-mass HM terms while all nine ordered pairwise diagonal no-QKE nu-nu channels, including identical-bank self-scattering entries with Fierz factor 2, are in the supported default with same-bank number/energy-neutral projection and off-diagonal number-neutral projection plus unordered-pair energy-neutral closure before thermo/hierarchy feedback, descriptor-to-mode-label mapping, fixed, dynamic ultra-relativistic FD, dynamic finite-mass FD electron/positron bath mode support with zero-chemical-potential default, signed-`mu_e` `nu-e` scattering and pair-process Pauli blocking in the staged electromagnetic source bridge with AP18/AP40 callback forwarding, explicit fixed-`mu_e` route-level `e_minus`/`e_plus` bath splitting, direct fixed-`mu_e` and charge-neutrality radial source artifacts with concrete moment deltas and finite-mass signed-`mu_e` e-/e+ energy/pressure diagnostics, corrected single-charge number-density normalization, opt-in fixed-`mu_e` and charge-neutrality finite-mass e-/e+ EOS feedback through the LRS, non-LRS source-only, and non-LRS nonlinear 3T Hubble/RHS shells with recorded `mu_e` histories, evolved LRS/source-only non-LRS/nonlinear non-LRS charge-neutral electron charge-asymmetry density states and histories, network-derivative electron charge-asymmetry energy feedback, fast finite-mass charge-susceptibility `mu_e` solves, finite-mu-scaled isotropic QED feedback in the signed-mu plasma EOS, and opt-in exact_finite_mu_scalar QED pressure/energy corrections through the scalar 3T thermo entrypoints plus the LRS, source-only non-LRS, and nonlinear non-LRS 3T Hubble/RHS solve shells with recorded QED-model metadata, descriptor-label-aware total-energy radial grids with neutrino `p=q` and electron/positron `p=sqrt(max(q^2-(m_e/T_nu_e0)^2,0))`, static radial-grid precomputation with in-builder static angular-geometry reuse, fast internal channel-grid assembly, precomputed radial quadrature weights, optional NPZ pretabulated radial-grid reuse in the live radial provider, scalar contraction hot-path reuse of validated radial grids, and mode-thresholded process-batched radial contraction within an explicit memory budget, a descriptor-driven radial source evaluator returning concrete `C_modes`, raw-quadrature number/energy moment extraction from those source modes, a live augmented-state radial moment provider, default-catalog LRS `pstf_radial` artifact routing with 18 radial moment sources and momentum-delta diagnostics, finite-mass elastic/pair and diagonal nu-nu source-count diagnostics, plus all-nine diagonal nu-nu number- and pair-energy-projection diagnostics, an opt-in AP18 3T thermo-source bridge mapping those moments into `dQ_nue_pair_N`/`dQ_nux_bank_N` with an opt-in `standard_3t_plasma` electromagnetic energy-closure mode that matches the canonical 3T plasma-transfer table while preserving number neutrality, LRS/non-LRS AP18 source-evaluator acceptance for that live bridge with nonzero radial-source diagnostics, an explicit LRS `pstf_radial` collision-feedback artifact/candidate/full-span source variant that routes the AP6 radial provider through the AP18 3T callback with frozen-initial-state default and budgeted live_rhs execution under `max_pstf_radial_source_evaluations` including charge-neutrality `mu_e` payload diagnostics plus candidate/full-span source-evaluation budget observables, a non-LRS AP42/AP44/AP45/AP46/AP53 `pstf_radial` diagnostic route that forwards the same explicit source-evaluation budget and standard-3T radial energy-normalization mode through artifact, candidate-gate, source-policy, and AP42 promotion-study surfaces with a dedicated non-LRS radial live-vs-frozen artifact/writer, direct AP6/LRS/non-LRS artifact CLIs that expose the `standard_3t_plasma` radial energy-normalization mode and explicit 3T initial temperatures with direct-wrapper closure-target summary values, a direct standard-3T radial span-profile artifact/CLI over explicit `N_span_end` ladders, a direct standard-3T radial source-policy span-profile artifact/CLI comparing frozen-initial-state and live-RHS updates over matched span ladders with source-evaluation budgets plus structured live-RHS source-budget failure rows, an AP65 combined angular+pstf_radial nonlinear collision wrapper/artifact that evaluates AP41 angular diagnostics and forwards AP6 finite-mass radial thermo moments plus `dA_modes` as the effective no-double-count nonlinear RHS payload while reporting terminal electron-mu plus charge-asymmetry observables, an AP4/AP65 combined-source full-span candidate gate that records terminal thermo/network/source, kinetic `dA`, component source, radial budget observables, AP6 radial number/pair-energy closure observables, fixed/charge-neutral electron chemical-potential controls, shared radial-grid cache entries, an opt-in frozen-source Radau physical-preview preset with stable routine short-span evidence plus isolated N_span=1e-3 diagnostic evidence, and a `piecewise_frozen` nonuniform state-handoff source-refresh path over explicit subspan ends including a passing real live-RHS two-span ladder, a passing real warm-preset N_span=1e-8 ladder, a passing physical-preview short-span routine gate, a passing piecewise_frozen N_span=1e-3 row, and a passing charge-neutral N_span=1e-4 row with evolved charge-asymmetry handoff plus electron-mu/charge-asymmetry maxima, a companion AP4/AP65 combined-source frozen/live source-policy span-profile artifact/CLI with live-minus-frozen observable deltas, source-evaluation counts, charge-neutral forwarding, and shared radial-grid cache reuse, a tiny-span budgeted live-RHS radial diagnostic artifact/writer that executes the same AP6 provider inside the 3T RHS under an explicit source-evaluation budget with standard-3T energy-closure support, and an LRS live-vs-frozen radial source-policy artifact/writer that records matched-policy observable/source-diagnostic deltas with matched standard-3T energy-closure support.  The bridge now uses the pairwise diagonal-nu-nu reference by default over all nine ordered bank pairs, including identical-bank self-scattering with Fierz factor 2, same-bank number/energy-neutral projection, explicit per-bank number closure, and weighted-energy closure; AP6 radial diagonal-nu-nu sources now add off-diagonal number-neutral projection for all-nine radial number conservation plus unordered-pair energy-neutral closure that preserves relative energy exchange, and the older fixed-point redistribution helper remains legacy comparison plumbing.  The weak/network "
        "pieces compute weak rates from current augmented monopoles and feed "
        "the PRIMAT network derivative.  The AP18-AP24 thermo-feedback path "
        "plus AP35 angular source factory is opt-in and is not a built-in physical angular collision kernel.  There "
        "is no CAPABILITY_BY_BACKEND entry and no canonical_forward_solver "
        "route. There is no default/promoted full-span collision-coupled runtime "
        "using the angular source.  The FB20-FB89 diagnostic witnesses remain "
        "private provenance, profiling, span, figure, SMC, readiness, and "
        "failure-analysis evidence only; they are not promotion decisions, no full "
        "promotion-grade coupled-solve weak-rate convergence beyond the AP80/FB-07 diagnostic gates, no promotion-grade PSTF collision-kernel solver/runtime coupling beyond the LRS diagnostic `pstf_radial` artifact/candidate/full-span route, the non-LRS AP42/AP44/AP45/AP46/AP53 diagnostic `pstf_radial` route with frozen default, explicit budgeted live_rhs smoke execution, live-vs-frozen radial source-policy surfaces, the AP65 combined angular+pstf_radial nonlinear diagnostic wrapper/artifact, and the AP4/AP65 combined-source full-span candidate gate/source-policy profile, no anisotropic/tensor QED response or promotion-grade exact-scalar-QED full-span coupled-solver validation beyond diagnostic exact_finite_mu_scalar scalar 3T routing/gate smoke and FB-08 chained scalar-QED control cross-product rows, no support for process families outside the supported no-QKE HM finite-mass electromagnetic plus all-nine diagonal-nu-nu catalog, "
        "no real-data/production SMC evidence or AP76 promotion approval, "
        "no promotion-grade physical full-BBN span or promotion-tolerance chained "
        "resolution ladder beyond FB-09 smoke diagnostics and private diagnostic "
        "FB50-FB89 evidence, and QKE out of scope."
    ),
)

JAX_CLASSA_GEOMETRY = BackendCapability(
    key="jax_classA_geometry",
    backend="jax",
    tier="candidate",
    surface_class="candidate",
    physics_scope="ClassA_6types_geometry",
    weak_mode="not_applicable",
    max_correction_level=0,
    supports_teff=False,
    max_thermo_tier=0,
    validated_default=False,
    readiness_scope_contract="candidate_classA_geometry_v1",
    transport_scope_contract="geometry_only_v1",
    thermo_scope_contract="not_applicable_v1",
    collision_scope_contract="not_applicable_v1",
    notes=(
        "JAX Class A geometry for all 6 types (WH formalism). "
        "5 official gates: G1 Type I reduction (1e-12), "
        "G2 Collins-Stewart equilibrium (dΣ/dN<1e-12), "
        "G3 Type II representative (finite K, physical Ω), "
        "G4 Friedmann constraint (6 types, <1e-12), "
        "G5 JAX↔NumPy parity (6 types, <1e-12)."
    ),
)

JAX_CLASSA_DRIVER = BackendCapability(
    key="jax_classA_driver",
    backend="jax",
    tier="substrate",
    surface_class="substrate",
    physics_scope="ClassA_6types",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=2,
    validated_default=False,
    readiness_scope_contract="substrate_classA_kappa_cascade_pstf_v1",
    transport_scope_contract="classA_kappa_cascade_reduced_v1",
    thermo_scope_contract="tier2_3T_candidate_v1",
    collision_scope_contract="no_collision_coupled_transport_v1",
    notes=(
        "JAX Class-A driver via κ-cascade reduced ℓ_max=2 PSTF. "
        "Demoted candidate→substrate at v3.0 Phase I (Plan §2.2): "
        "linearised PSTF cannot capture the 27-36% nonlinear correction "
        "documented in RABBIT_report §06 at Σ_H ∈ [0.2, 0.5]. "
        "Production-grade Class-A anisotropic Y_p requires the "
        "characteristic-method path (JAX_CLASSA_CHARACTERISTIC). "
        "FLRW-parity diagnostic only; must not claim Σ_+ > 0 production "
        "yields. Geometry gates (G1-G5) at <1e-12 still pass and are "
        "preserved. Dispatch: backend='jax_classA' (explicit opt-in only)."
    ),
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar"),
)

JAX_CLASSA_CHARACTERISTIC = BackendCapability(
    key="jax_classA_characteristic",
    backend="jax",
    tier="candidate",
    surface_class="candidate",
    physics_scope="ClassA_6types",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=2,
    validated_default=False,
    readiness_scope_contract="candidate_classA_characteristic_v1",
    transport_scope_contract="classA_characteristic_v1",
    thermo_scope_contract="tier2_3T_candidate_v1",
    collision_scope_contract="no_collision_coupled_transport_v1",
    notes=(
        "v3.0 Phase I (Plan §2.2): characteristic-method Class-A driver. "
        "Generalises the Type-I exact characteristic f(u, μ_j, N) = "
        "f_FD(u·e^{2 I_j(N)}) to the 6 orthogonal Class-A types via "
        "per-type integrating factors I_j^{(type)}(N). Phase I ships "
        "TYPE_I (full implementation via delegation to canonical Type-I "
        "characteristic backend). Types II, VI₀, VII₀, VIII, IX raise "
        "NotImplementedError until per-type Maartens-Bassett 1998 §3.3 "
        "geodesic derivations land. The same 27-36% nonlinear correction "
        "envelope from RABBIT_report §06 carries through to Type I "
        "trivially; per-type validation is research-grade follow-on. "
        "Historically replaced the demoted PSTF JAX_CLASSA_DRIVER as a "
        "candidate; PORT-00 freezes both paths as provenance/oracle code."
    ),
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar"),
)

JAX_EXTENDED_PSTF = BackendCapability(
    key="jax_extended_pstf",
    backend="jax",
    tier="substrate",
    surface_class="substrate",
    physics_scope="TypeI_transport",
    weak_mode="born_or_live",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=1,
    validated_default=False,
    readiness_scope_contract="substrate_extended_pstf_v1",
    transport_scope_contract="extended_pstf_lmax8_v1",
    thermo_scope_contract="tier1_single_T_nu_v1",
    collision_scope_contract="no_collision_coupled_transport_v1",
    notes=(
        "Extended 1+3 PSTF hierarchy ℓ_max=8. FLRW parity 7.8e-7 vs legacy. "
        "Demoted candidate→substrate per RABBIT_report §03/§06: linearised "
        "PSTF Ψ_ℓ are momentum-independent and cannot capture the exact "
        "f_FD(u·e^{2I_j}) spectral distortion. Production-grade anisotropic "
        "yields require the characteristic method (§06: 27-36% nonlinear "
        "correction at Σ_H ∈ [0.2, 0.5], 70-80% of total anisotropic shift). "
        "ℓ_max → ∞ does not close this gap; the linearisation itself is the "
        "ceiling. Retained as FLRW-parity diagnostic only — must not be used "
        "to claim production yields at Σ_+ > 0."
    ),
)

JAX_WEAK_CL3_KERNEL = BackendCapability(
    key="jax_weak_cl3_kernel",
    backend="jax",
    tier="candidate",
    surface_class="candidate",
    physics_scope="weak_corrections",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=True,
    teff_kernel_validated=True,
    max_thermo_tier=1,
    validated_default=False,
    readiness_scope_contract="candidate_jax_weak_cl3_kernel_v1",
    transport_scope_contract="not_applicable_v1",
    thermo_scope_contract="tier1_single_T_nu_v1",
    collision_scope_contract="not_applicable_v1",
    notes=(
        "CL3 weak rate kernels (recoil + weak magnetism + K_{r,ℓ}). "
        "32 parity tests pass. I₀(CL3)=+4.87% vs Born. "
        "Driver-wired via tier-2 liveweak path at correction_level=3. "
        "CL2→CL3 shift: +5.7e-4 Y_p at FLRW."
    ),
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar"),
)

JAX_CURVED_HIERARCHY = BackendCapability(
    key="jax_curved_hierarchy",
    backend="jax",
    tier="substrate",
    surface_class="substrate",
    physics_scope="transport_curved",
    weak_mode="not_applicable",
    max_correction_level=0,
    supports_teff=False,
    max_thermo_tier=0,
    validated_default=False,
    readiness_scope_contract="substrate_curved_hierarchy_v1",
    transport_scope_contract="curved_pstf_scaffold_v1",
    thermo_scope_contract="not_applicable_v1",
    collision_scope_contract="not_applicable_v1",
    notes=(
        "Exact m-decomposed curved PSTF coupling coefficients. "
        "Scaffold only — coupling matrix computed, not integrated. "
        "Note (RABBIT_report §06): Plan §2.3's promotion path to ℓ_max=4 "
        "production was mooted by the same linearisation ceiling that "
        "demoted JAX_EXTENDED_PSTF. Class-A anisotropic yields would require "
        "a characteristic / fully-nonlinear construction, "
        "not extended curved PSTF. This substrate is retained only as a "
        "documentation artifact for the curved coupling matrix derivation."
    ),
)

JAX_RODAS5P_SOLVER = BackendCapability(
    key="jax_rodas5p_solver",
    backend="jax",
    tier="candidate",
    surface_class="candidate",
    physics_scope="solver",
    weak_mode="not_applicable",
    max_correction_level=0,
    supports_teff=False,
    max_thermo_tier=0,
    validated_default=False,
    readiness_scope_contract="candidate_jax_solver_v1",
    transport_scope_contract="not_applicable_v1",
    thermo_scope_contract="not_applicable_v1",
    collision_scope_contract="not_applicable_v1",
    notes=(
        "JAX-native Rodas5P (8-stage, order 5(4), SciML convention). "
        "LU-factorize-once optimization. Jacobian reuse with stride control. "
        "PUB-02 validates numerical solver code paths: nonautonomous order, "
        "bounded failure, NumPy/JAX twin agreement, and replay agreement. "
        "PORT-00 freezes this as a numerical-method reference only; BDF remains "
        "the number-of-record until Rust authority gates pass. No repeated-run, "
        "endpoint, physics-validation, or public-production authority is granted."
    ),
)

JAX_CLASSB_DRIVER = BackendCapability(
    key="jax_classB_driver",
    backend="jax",
    tier="candidate",
    surface_class="candidate",
    physics_scope="ClassB_types",
    weak_mode="born",
    max_correction_level=0,
    supports_teff=False,
    max_thermo_tier=1,
    validated_default=False,
    readiness_scope_contract="candidate_classB_reduced_family_v1",
    transport_scope_contract="classB_reduced_mask_transport_v1",
    thermo_scope_contract="tier1_single_T_nu_v1",
    collision_scope_contract="no_collision_coupled_transport_v1",
    notes=(
        "JAX Class B BBN driver (candidate). "
        "Reduced-mask Class B path with production smoke on all 6 labels: V, IV, canonical III, h-locked VI_h/VII_h representatives, and canonical VI₋₁/₉. "
        "Phase-1 + Phase-2 full BBN validated for Type V (A_init≤1e-4, orthogonal/tilted gold gates) "
        "Type IV (A_init≤1e-5, N1_init≤1e-3, orthogonal/tilted gold gates), "
        "Type III (A_init≤1e-5, canonical N2=-N3=1e-4, orthogonal/tilted gold gates), "
        "Type VI_h (h=-2, N3=h*N2, orthogonal/tilted gold gates), "
        "Type VII_h (h=0.5, N3=h*N2, orthogonal/tilted gold gates), "
        "and Type VI₋₁/₉ (A_init≤1e-5, canonical N3=-N2/9 with c=15/4, orthogonal/tilted gold gates). "
        "Tier-1 Born-only; CL>0 raises ValueError (guard, not silent ignore). "
        "FLRW reduction: Type V (A→0) matches FLRW baseline. "
        "Dispatch: backend='jax_classB' (explicit opt-in only)."
    ),
)

JAX_TILTED_FULL_COUPLED = BackendCapability(
    key="jax_tilted_full_coupled",
    backend="jax",
    tier="candidate",
    surface_class="candidate",
    physics_scope="tilted_all_types_full_coupled",
    weak_mode="cl0_cl3_equilibrium_fd_optional_boosted_l012",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=1,
    validated_default=False,
    readiness_scope_contract="candidate_tilted_full_coupled_v1",
    transport_scope_contract="tilted_reduced_lrs_v1",
    thermo_scope_contract="tier1_single_T_nu_v1",
    collision_scope_contract="no_collision_coupled_transport_v1",
    notes=(
        "Phase γ-3 (Plan §2.2) additive variant of jax_tilted: same code "
        "path but with tilt_stress_feedback, tilt_weak_rate_boost, and "
        "(at CL3) tilt_cl3_angular_kernel auto-flipped to True so the "
        "tilted scalar BBN slice exercises its full physics chain by "
        "default. The original jax_tilted backend is preserved unchanged "
        "for backward compatibility with existing gold tests; users who "
        "want the audit-honest full-coupled physics opt in to "
        "backend='jax_tilted_full_coupled' explicitly. Same v0 ≤ 1e-3 "
        "guard, same Bianchi-type coverage."
    ),
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar"),
)


JAX_TILTED_BBN = BackendCapability(
    key="jax_tilted_bbn",
    backend="jax",
    tier="candidate",
    surface_class="candidate",
    physics_scope="tilted_all_types",
    weak_mode="cl0_cl3_equilibrium_fd_optional_boosted_l012",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=1,
    validated_default=False,
    readiness_scope_contract="candidate_tilted_reduced_v1",
    transport_scope_contract="tilted_reduced_lrs_v1",
    thermo_scope_contract="tier1_single_T_nu_v1",
    collision_scope_contract="no_collision_coupled_transport_v1",
    notes=(
        "JAX tilted BBN runner (candidate). "
        "Principal-axis scalar tilt for all Bianchi types + FLRW alias. "
        "Legacy H/√(1-v²) plus opt-in normal-frame T00 Hubble closure, "
        "with stress and G0i enthalpy normalization tied to the selected "
        "closure. Opt-in tilted perfect-fluid anisotropic stress, algebraic "
        "G0i heat-flux closure, plasma-frame boosted-FD weak monopole "
        "ratios, and opt-in CL3 principal-axis K_l angular weak-kernel "
        "ratios are wired. "
        "The tilted public path auto-selects n_ell=3 diagonal non-LRS "
        "quadrupole feedback when Sigma_minus is nonzero or the principal "
        "tilt axis is 1/2, activating the reduced Sigma_minus -> Psi_minus "
        "-> Pi_minus transport channel. "
        "Tier-1 thermo; CL0-CL3 weak corrections supported. CL3 defaults to "
        "scalar finite-mass/recoil/weak-magnetism on the live monopole; the "
        "direct l<=2 angular K_l projection is available only when explicitly "
        "enabled with boosted FD moments. "
        "Tilt grows during radiation (eigenvalue +1); v₀≲1e-7 for BBN. "
        "Dispatch: backend='jax_tilted' (explicit opt-in only)."
    ),
)


# ═══════════════════════════════════════════════════════════════
# §4. Dispatch tables
# ═══════════════════════════════════════════════════════════════

# Public forward authority.  F06 hard-retires the JAX endpoint names rather
# than preserving compatibility dispatch.  Component-oracle and alternate-
# geometry metadata remain available only through CAPABILITY_BY_KEY below.
CAPABILITY_BY_BACKEND = {
    "scipy": SCIPY_TYPEI_REFERENCE,
    "auto": SCIPY_TYPEI_REFERENCE,
}


# ═══════════════════════════════════════════════════════════════════════
# Retired forward names
# ═══════════════════════════════════════════════════════════════════════
#
# F06 replaces the former callable quarantine with hard retirement: retired
# names are absent from the public dispatch registry.  Alternate-geometry
# physics/source metadata remains preserved below, but is not a backend API.
QUARANTINED_BACKENDS = frozenset()

#: Backends on the active-canonical default surface after BD619/BD620.
ACTIVE_CANONICAL_BACKENDS = frozenset({"scipy", "auto"})


def is_quarantined(backend: str) -> bool:
    """Return quarantine membership (empty after F06 hard retirement)."""
    return backend in QUARANTINED_BACKENDS

# Full catalog for introspection.
CAPABILITY_BY_KEY = {cap.key: cap for cap in [
    SCIPY_TYPEI_REFERENCE,
    SCIPY_TYPEI_TIER2_PER_SPECIES,
    SCIPY_TYPEI_TIER3_WEAK_BUDGET,
    JAX_TYPEI_FULL_BOLTZMANN_TIER3_PREFLIGHT,
    JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE,
    JAX_TYPEI_AUGMENTED_PSTF_NOQKE_STAGING,
    JAX_CLASSA_GEOMETRY,
    JAX_CLASSA_DRIVER,
    JAX_CLASSA_CHARACTERISTIC,
    JAX_CLASSB_DRIVER,
    JAX_TILTED_BBN,
    JAX_TILTED_FULL_COUPLED,
    JAX_WEAK_CL3_KERNEL,
    JAX_CURVED_HIERARCHY,
    JAX_RODAS5P_SOLVER,
]}

# Component and alternate-geometry aliases retained for non-dispatch metadata.
JAX_TYPEI_CL3_TIER2_TEFF_CANDIDATE = JAX_TYPEI_LIVEWEAK_CL3_TIER2_TEFF_CANDIDATE
JAX_CLASSA_GEOMETRY_SUBSTRATE = JAX_CLASSA_GEOMETRY
JAX_CLASSA_TRANSPORT_SUBSTRATE = JAX_CLASSA_DRIVER
JAX_WEAK_CORRECTION_SUBSTRATE = JAX_WEAK_CL3_KERNEL
