# PR-C Spec — Σ_H identifiability + shrinkage diagnostic/gate

Audit refs: BD598 D-R1, D-R2 (both P1, load-bearing inference honesty).

## Problem (verified)

- `joint_3d_inference.py` does not import `fisher.py` and the NUTS run output
  carries no rank/condition/shrinkage flag (D-R1). Because Y_p enters as Σ_H²,
  the likelihood is rank-2 at the FLRW null (Σ_H direction degenerate), so the
  reported Σ_H "posterior" is essentially the half-normal prior — but nothing at
  runtime says so.
- Recovery tests (`test_synthetic_recovery_full_3d.py`,
  `test_joint_3d_inference.py`) accept bands (`sigma_marg<=0.31` vs prior `0.30`)
  satisfied by the prior alone; no `posterior_sigma << prior_sigma` assertion
  exists (D-R2). A prior-dominated number can be read as data-driven.

The physics is already documented in `fisher.py:354-371` (rank-2 at FLRW is
*expected*; cond budgets; `MIN_RANK_AT_FLRW=2`) and the machinery exists
(`bbn_fisher`, `gaussian_prior_fisher`, `fisher_diagnostics`) — it is simply not
wired into the inference path or asserted.

## Design (reuse, minimal)

1. `sigma_h_identifiability(config=None, *, sigma_H_fid=0.0) -> SigmaHIdentifiability`
   in `joint_3d_inference.py`. Reuses `fisher.bbn_fisher` (likelihood Fisher +
   diagnostics) and `fisher.gaussian_prior_fisher` (prior Fisher in
   (Σ_H, log η, log τ_n) space), then `fisher.fisher_diagnostics(F_lik, prior_F)`
   for the posterior marginal. Returns: `rank_likelihood`, `cond_likelihood`,
   `cond_posterior`, `prior_sigma`, `posterior_sigma`, `shrinkage`
   (= posterior/prior), `informative` (bool), `fiducial_sigma_H`.
2. `informative = (rank_likelihood >= 3) and (shrinkage < SHRINKAGE_INFORMATIVE)`,
   `SHRINKAGE_INFORMATIVE = 0.9`. At the FLRW null rank is 2 ⇒ informative=False
   (honest "BBN data uninformative on Σ_H at the null; bound is prior-conditional").
3. Surface it in `run_bbn_nuts_3d` output under key `identifiability` (cheap vs
   NUTS), closing D-R1's runtime-surfacing gap.

## Revised finding (after running the real likelihood)

Two empirical facts forced a more honest design than first planned:

1. **D-R5 confirmed.** The central-FD marginal for Σ_H at the null is ε-noise:
   `posterior_sigma ≈ 1.7e-7` (spuriously tight) because FD of the even Σ_H²
   response is numerically unstable. So the FD shrinkage MUST NOT be the basis of
   an "informative" claim.
2. **BBN does not independently identify Σ_H.** Even at Σ_H=0.25 the likelihood
   stays rank-2 (cond ≈ 5e18): η,τ_n are pinned so stiffly (σ_DH=3e-7) that the
   Σ_H direction never clears the rank threshold under default BBN obs errors.

Therefore the robust, FD-noise-immune signal is the **likelihood rank /
condition number**. The flag is rank-gated:
`informative = (rank ≥ 3) and (shrinkage < 0.9)`; `degenerate = rank < 3`.
`posterior_sigma`/`shrinkage` are reported for transparency but documented as
untrustworthy when `degenerate`.

## TDD (as built)

- `test_informative_requires_rank3_and_shrinkage` (fast, no solve): unit-tests
  the `_sigma_h_informative` flag logic — positive branch (rank3+shrink→True),
  rank-deficient→False, weak-shrink→False.
- `test_flrw_null_likelihood_is_rank_deficient` / `..._is_not_informative`:
  real null likelihood is rank-2, degenerate, `informative False`, cond>1e8.
- `test_bbn_does_not_independently_identify_sigma_h_off_null` (slow): honest
  lock that even at Σ_H=0.25, BBN alone keeps Σ_H degenerate → flag stays False.
- `run_bbn_nuts_3d` output now carries the `identifiability` read-out (D-R1).

## Deferred

Demonstrating an *informative* off-null regime requires a Fisher robust to the
even Σ_H² response (analytic/2nd-order or profile likelihood) — the D-R5 fix —
tracked separately. PR-C delivers the honest negative result + runtime flag.

## Acceptance gate

- New tests green. Reuses fisher.py (no duplicate Fisher code). No new readiness
  gate, no Bayes-factor/evidence claim added. `/review` clean of Critical/Warning.
