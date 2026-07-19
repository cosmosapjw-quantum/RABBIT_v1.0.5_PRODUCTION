# BD598 PR-B2 — N_eff convention resolution

Date: 2026-06-28. Method: 3-agent workflow (numerical derivation → literature/CRAG
→ adversarial verification). Confidence: high.

## Question

Tier-1 reports N_eff ≈ 3.011 (entropy-ratio of the parametric T_ν); the Hubble
seeds ρ_ν with config.N_eff = 3.044. Which is correct, and does
`_hubble_invsec(N_eff=config.N_eff, T_ν=parametric)` double-count neutrino heating?

## Verdict: reporting-only; ρ_ν is physically correct (to ~0.36%)

1. **No large/epoch-dependent double-count.** An initial derivation flagged a
   "+7% at 0.07 MeV → +2.8e-3 in Yp" overcount, but adversarial verification
   showed this is a **strawman**: it compared the code's live parametric T_ν
   against a FROZEN (4/11)^{1/3} ratio, which is itself unphysical during e±
   annihilation (it makes H wrong by +24% at 0.5 MeV). Real codes
   (PRIMAT/NUDEC_BSM/PArthENoPE) solve a live T_ν(a); the
   `N_eff·(7/8)·(4/11)^{4/3}·ρ_γ` form is an ASYMPTOTIC definition, not a
   per-epoch prescription. The verifier proved
   ρ_ν_code / ρ_ν(N=3, live T_ν) = constant **1.01467 = 3.044/3.0** at every
   epoch — i.e. the code applies the SM N_eff correction consistently with the
   physical T_ν. **No spurious +2.8e-3 Yp.**

2. **Genuine but tiny asymptotic double-count (~0.36%).** Because
   `g_s(T_dec=2.0 MeV) = 5.485 < 5.5`, the parametric asymptote settles at
   T_ν/T_γ = 0.71440, 0.089% above the canonical (4/11)^{1/3} = 0.71377. Since
   3.044 is *defined* with the canonical ratio, multiplying it by the slightly
   hot parametric T_ν re-counts that sliver: effective N_eff → 3.0549 at T_final
   (+0.36% in ρ_ν, +0.011 in N_eff, ~1.5e-4 in Yp). This is within the code's
   stated fidelity and far below observational Yp precision (~1e-3).

3. **Reporting.** The SM-comparable value is N_eff = **3.044** (the input;
   Mangano 2005: 3.046; Bennett/Froustey/Akita 2020: 3.044, used by PRIMAT and
   NUDEC_BSM). The entropy-ratio number 3.011 is convention-inconsistent (pairs
   flavour-count 3.0 with the parametric T_ν and omits the QED/non-instantaneous
   heating that 3.044 packages); it must NOT be presented as "the" N_eff. At
   Tier 1, N_eff is an INPUT, not a measurement; a DERIVED N_eff requires Tier ≥ 2.
   PR-B's `Observables.N_eff_input` (=3.044) and `N_eff_is_derived=False` already
   encode this.

## Action taken (PR-B2, reporting-only)

- Resolution recorded here; PR-B spec OPEN→RESOLVED.
- Schema docstring clarified: the Tier-1 `N_eff` field is a convention-inconsistent
  entropy-ratio diagnostic; `N_eff_input` (3.044) is the SM-comparable value.
- No ρ_ν change, no value change, no gold churn (verified: the reported N_eff is
  decoupled from the Hubble baseline — changing config.N_eff left it unchanged).

## Deferred (separate decision — baseline-shifting)

The genuine ~0.36% asymptotic double-count can be removed by either raising
T_dec so `g_s(T_dec) → 5.5` (parametric asymptote → exact (4/11)^{1/3}) or
multiplying ρ_ν by 1/1.003566. Either shifts the baseline Yp by ~1.5e-4 →
breaks `flrw_gold Yp_abs=1e-6` and requires a full gold regen. Deferred as its
own decision (it changes the published baseline), tracked as PR-B3.

## Sources (CRAG)

- Mangano et al. 2005, Nucl.Phys.B 729:221, arXiv:astro-ph/0506164 (N_eff=3.046)
- Bennett, Buldgen, Drewes, Wong 2020, arXiv:2001.04466 (N_eff=3.044)
- Escudero 2020 / NUDEC_BSM, arXiv:1812.05605: N_eff=(7/8)(11/4)^{4/3}(ρ_ν/ρ_γ)
- Pitrou, Coc, Uzan, Vangioni 2018 / PRIMAT, arXiv:1801.08023
- Froustey/Pitrou/Volpe 2020 (2008.01074); Akita & Yamaguchi 2020 (2005.07047)
- Lesgourgues & Pastor 2006, arXiv:astro-ph/0603494 (ρ_r=[1+(7/8)(4/11)^{4/3}N_eff]ρ_γ)
