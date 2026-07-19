<!-- BEGIN:PG_HEADER -->
# RABBIT Feature Promotion Gates

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

## Current Promotion Status
<!-- END:PG_HEADER -->

<!-- BEGIN:PROMOTION_STATUS -->
| Feature | Code tier | Surface class | Next blocker |
|---|---|---|---|
| SciPy Type I | **canonical** | canonical | — (reference) |
| Tilted scalar BBN | candidate | candidate-strong | — |
| Class A curved transport | candidate | candidate-layered | — |
| Class B BBN | candidate | candidate-layered | — |
| AD diagnostic gradients | candidate | diagnostic | — |
| Inference / PE framework / model comparison | candidate | exploratory | — |
| Tier-3 full-collision preflight surface (classical Boltzmann; no QKE) | candidate | diagnostic | AP-form unification (combine spectral_relaxation anisotropy stability with projected_physical grid scaling): RESOLVED in PR-T3B canonical milestone via collision_mode='ap_unified_preflight' (anisotropy spread ~7e-5, grid spread <1e-4) |
<!-- END:PROMOTION_STATUS -->

<!-- BEGIN:PG_BODY -->
## What "gate progress" means

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

- **Status**: Frozen unregistered oracle.
- **Current explicit key**: `UNREGISTERED`; `jax_typeI_liveweak_cl0` remains candidate
- **Evidence ceiling**: historical parity and BBN gold are runtime regressions only
- **Governance**: PUB-02C grants no repeated-run authority; Rust is SPECIFIED only; G-01R remains open

## Next Promotion Queue
<!-- END:PG_BODY -->

<!-- BEGIN:NEXT_QUEUE -->
Historical candidate inventory (not an active implementation queue):

JAX-backed and legacy entries are frozen. Active work order comes only from the Rust-first publication plan.

| # | Feature | Surface class | Historical blocker |
|---|---|---|---|
| 1 | Tilted scalar BBN | candidate-strong | — |
| 2 | Class A curved transport | candidate-layered | — |
| 3 | Class B BBN | candidate-layered | — |
| 4 | AD diagnostic gradients | diagnostic | — |
| 5 | Tier-3 full-collision preflight surface (classical Boltzmann; no QKE) | diagnostic | AP-form unification (combine spectral_relaxation anisotropy stability with projected_physical grid scaling): RESOLVED in PR-T3B canonical milestone via collision_mode='ap_unified_preflight' (anisotropy spread ~7e-5, grid spread <1e-4); Dolgov-Hansen-Semikoz nu-nu coefficient calibration: PARTIAL (total_rate_nu_nu_diagonal_jax helper landed; energy-conserving 3T nu-nu source wired in ap_unified_nu_nu_preflight; moment-projected spectral bank source plus number/energy-neutral shape damping wired in ap_unified_nu_nu_spectral_preflight; calibrated no-QKE AP energy-transfer accuracy candidate wired in ap_unified_nu_nu_spectral_accuracy_preflight; full DH-S coefficient-table runtime kernel remains unpromoted); Tier-3 AP energy-moment contract: LOCKED for the no-QKE surface (single Mangano C-rate source, sign-safe positive scaling, bidirectional heating/cooling AP moment parity with 3T rates, bidirectional nu-nu spectral conservation tests, and number/energy-neutral spectral self-thermalization tests); Public diagonal nu-nu dispatch: LOCKED as explicit opt-in for jax_ap_unified_tier3 (off default; 3t/spectral/accuracy modes routed to full forward collision modes; spectral public solve reduces flavour-temperature split; accuracy candidate closes the bounded smoke-grid FLRW N_eff gap to <5e-3 without claiming QKE; public 3T-vs-spectral same-limit comparison locked); FLRW N_eff gap to Mangano 2005 (~0.0095) remains the default AP-form baseline limitation; jax_tier3_nu_nu='accuracy' is a no-QKE calibrated candidate path, not a full DH-S/QKE promotion |
| 6 | Inference / PE framework / model comparison | exploratory | — |
<!-- END:NEXT_QUEUE -->

<!-- BEGIN:PG_FOOTER -->
No candidate should be promoted without explicit gate criteria and decision record.
<!-- END:PG_FOOTER -->
