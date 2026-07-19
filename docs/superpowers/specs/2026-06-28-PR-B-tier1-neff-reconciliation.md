# PR-B Spec — Reconcile Tier-1 N_eff reporting with the Hubble-driving N_eff

Audit refs: BD598 A-R1 (P2), A-R3 (P3). Source: `docs/audit/BD598_internal_hostile_journal_audit_2026-06-28.md`.

## Problem (verified)

In the default Tier-1 path of `run_full_coupled_typeI`:

- The expansion rate uses `config.N_eff` (`_N_EFF = 3.044`) to set the neutrino
  energy density: `rho_nu = N_eff * (7/8) * prefactor * T_nu^4`
  (`full_coupled_typeI.py:544-546`).
- The value **reported** as the `N_eff` observable is the entropy-ratio
  *diagnostic* `N_eff_from_T_ratio(T_nu, T_gamma)` = `3.0 * (T_nu/T_nu_std)^4`
  (`full_coupled_typeI.py:1391-1393`, `incomplete_decoupling.py:300-310`).
  Because Tier-1 `T_nu_from_T_gamma_tier1` is *instantaneous* decoupling
  (`T_nu/T_gamma -> (4/11)^{1/3}` exactly), this ratio recovers ≈ **3.011**,
  not the 3.044 that drives the dynamics.
- The self-consistency guard only warns when `|config.N_eff - N_eff_meas| > 0.05`
  (`:1399-1405`); the actual gap is 0.0333, so it passes **silently**.
- Net effect: the neutrino-sector observable reported to the user / gold table
  (`tests/fixtures/flrw_gold_v861.json` N_eff = 3.0107) is **decoupled from the
  expansion physics it is supposed to summarize.**

`incomplete_decoupling.py:9-11` labels Tier-1 "Smoke test only / FORBIDDEN in
the canonical path", yet Tier-1 is the dataclass default in drivers and
inference (A-R3).

## Physics resolution (REVISED decision — honesty layer, no value change)

Initial plan was "report `config.N_eff` (3.044) as the dynamical value". On
investigation this was rejected as too aggressive and not provably correct:

1. **Codebase-wide convention.** The 3.011 value is locked in three fixtures
   (`flrw_gold_v861.json`, `jax_gold_v861.json`, `tier3_cross_code.json`) and
   asserted across several tests. `test_primat_parity.py:8-21` *documents* it:
   "N_eff≈3.011 from tier-1 … gap (3.011 vs 3.044) accounts for ΔY_p≈+0.0015".
2. **Possible double-count.** The RHS calls `_hubble_invsec(T_gamma, T_nu, N_eff)`
   with `N_eff=config.N_eff=3.044` AND the parametric `T_nu` (whose ratio gives
   `N_eff_from_T_ratio≈3.011`, i.e. ~0.36% above instantaneous). The standard-
   convention effective N_eff is then `config.N_eff·(T_nu/T_nu_std)^4 ≈ 3.055`,
   not 3.044 and not 3.011. So *neither* current number is unambiguously the
   "true" dynamical N_eff; there is a real convention/double-count question.
3. **AGENTS.md non-negotiable:** "Do not silently change physical conventions."
   Changing the reported N_eff value is exactly that.

**This PR therefore implements only a safe honesty layer (value unchanged):**

1. `Observables.N_eff_input` — exposes the Hubble-seed `config.N_eff`, so the gap
   between reported N_eff and the expansion seed is inspectable, not hidden.
2. `Observables.N_eff_is_derived` — False at Tier 1 (entropy-ratio DIAGNOSTIC,
   SPECIFIED), True at Tier ≥ 2 / backbone (DERIVED). Any derived-N_eff claim
   must use Tier ≥ 2.
3. Same provenance mirrored into the JAX driver `metadata`
   (`N_eff_input`, `N_eff_is_derived`) so backends agree on what N_eff *means*.
4. Guard message clarified (diagnostic vs derived); no fixture/value churn.

## RESOLVED (PR-B2) — see docs/audit/BD598_PR-B2_neff_convention_resolution_2026-06-28.md

A 3-agent physics-math-audit + literature(CRAG) + adversarial-verification pass
(confidence: high) concluded **ρ_ν is physically correct (reporting-only fix)**:

- The "~3.055 / +2.8e-3 Yp double-count" was a strawman (it froze the (4/11)^{1/3}
  ratio, unphysical during e± annihilation). ρ_ν_code / ρ_ν(N=3, live T_ν) is a
  constant 3.044/3.0 at every epoch — the SM N_eff correction is applied
  consistently with the physical live T_ν. No spurious Yp shift.
- SM-comparable value = **3.044** (input); the entropy-ratio 3.011 is
  convention-inconsistent and must not be cited as "the" N_eff. PR-B's
  `N_eff_input`/`N_eff_is_derived` already encode this.
- A genuine but tiny ~0.36% asymptotic double-count remains (g_s(T_dec=2.0)=5.485
  <5.5; ~1.5e-4 in Yp); removing it shifts the baseline → full gold regen, so it
  is deferred as PR-B3 (separate baseline-shifting decision).

## Scope / files

- `src/rabbit/drivers/full_coupled_typeI.py` — N_eff_meas branch (`:1385-1396`),
  guard (`:1398-1405`), `Observables` (add `N_eff_entropy_ratio` field, default
  None so non-Tier-1 callers unaffected).
- `src/rabbit/thermo/incomplete_decoupling.py` — no logic change; ensure
  `N_eff_from_T_ratio` docstring states it is a diagnostic, not the dynamical
  N_eff.
- `tests/fixtures/flrw_gold_v861.json` — update locked `N_eff` 3.0107 → 3.044
  with a provenance note (dynamical/input-echo, SPECIFIED); keep `N_eff_abs` tol.
- Tests: new `tests/test_tier1_neff_reporting_consistency.py`.

Anti-drift: net target +20/-6. No new readiness/manifest/figure gate. Reuses
existing `config.N_eff`, `_hubble_invsec`, and fixture infrastructure.

## TDD plan (RED first)

1. `test_tier1_reported_neff_equals_hubble_driving_neff`: run FLRW Tier-1; assert
   `obs.N_eff == config.N_eff` within 1e-9 (currently fails: 3.0107 != 3.044).
2. `test_tier1_neff_is_marked_specified_not_derived`: assert the diagnostic field
   `obs.N_eff_entropy_ratio` is populated (≈3.011) and distinct from `obs.N_eff`.
3. `test_neff_claim_requires_tier2`: a helper/guard rejects treating Tier-1
   N_eff as derived (assert the documented contract — e.g. a `is_derived` flag or
   a raise/warn when a derived-N_eff is requested at tier 1).
4. Regenerate gold fixture value; rerun `pytest -m "gold and not slow"` +
   `test_flrw_external_parity_public.py` (parity should improve).

## Acceptance gate

- New test file green.
- `obs.N_eff` == `_hubble_invsec` N_eff within 1e-3 (here exact).
- Gold + external-parity suites green with updated fixture.
- `/review` (local subagent) clean of Critical/Warning.
