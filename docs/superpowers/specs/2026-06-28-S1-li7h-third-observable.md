# S1 Spec — Li7/H as an opt-in 3rd observable (break the Σ_H rank ceiling)

Audit refs: BD599 INF-1 (structural 2-observable rank ceiling). First structural
blocker from the Wave-2 "remaining maturity" list.

## Feasibility (measured, decisive)

The BD599 INF-1 ceiling is: `make_bbn_predict_callable` returns only [Y_p, D/H]
(2 observables) for 3 parameters (Σ_H, log η, log τ_n), so the likelihood Fisher
is rank ≤ 2 at every fiducial and a data-identified Σ_H is impossible.

Measured (central-FD Fisher, SciPy canonical, N_q=20):

| Σ_H | 2-obs rank / cond | +Li7H rank / cond |
|---|---|---|
| 0.0 | 2 / 4.8e19 | 3 / 1.2e7 |
| 0.2 | 2 / 2.1e18 | 3 / 1.4e5 |

Adding Li7/H lifts the Fisher to **rank 3 off-null** with a well-conditioned
matrix (cond 1.4e5) — i.e. Σ_H becomes formally identifiable off-null. (At the
exact null the rank-3 is FD-noise on the even Σ_H² response — D-R5 — and Σ_H is
genuinely degenerate there; that is physical, not a defect.)

## CRITICAL caveat — the lithium problem (honesty-gating)

BBN predicts Li7/H ≈ 5e-10 but the Spite-plateau observation is ≈ 1.6e-10 (a
factor ~3 discrepancy, unresolved). Therefore Li7/H MUST NOT be wired into the
default likelihood at face value — a naive Li7 term would be dominated by the
lithium problem and would pull η/Σ_H to absorb a non-BBN systematic. Li7/H is
added strictly as an **opt-in identifiability observable** (it restores Fisher
rank / makes the identifiability structure honest), NOT as a default data
constraint. Any constraint derived with Li7/H must carry the lithium-problem
caveat and is EXPLORATORY.

## Design (opt-in, default behavior unchanged)

1. `fisher.make_bbn_predict_callable(..., include_li7h=False)`: when True, return
   [Y_p, D/H, Li7/H] (Li7/H from the prediction). Default False → unchanged.
2. `fisher.bbn_fisher(..., include_li7h=False, li7h_sigma=...)`: 3rd σ when on.
3. `joint_3d_inference.sigma_h_identifiability(..., include_li7h=False)`: when
   True, build the 3-observable Fisher so `informative` can be True off-null
   (rank 3 + shrinkage). Default False preserves the honest rank-2 verdict.
4. Constants: `LI7H_SIGMA_DEFAULT` documented (Spite-plateau scatter ~0.04 dex on
   ~1.6e-10 ⇒ a representative σ ~ a few×1e-11), with the lithium-problem note.

## TDD

- `test_li7h_lifts_fisher_rank_offnull`: with include_li7h, rank==3 and cond ≪ 1e8
  at Σ_H=0.2 (vs rank 2 without).
- `test_default_remains_two_observable_rank2`: include_li7h=False → rank 2
  (the BD599 INF-1 ceiling, unchanged).
- `test_identifiability_informative_reachable_with_li7h_offnull`: with Li7H,
  `sigma_h_identifiability(sigma_H_fid=0.2, include_li7h=True).informative is True`.

## Out of scope (follow-on / other structural blockers)

- A face-value Li7 *constraint* in the default likelihood (blocked by the lithium
  problem). - The robust non-FD Fisher (needs a JAX-traceable forward). - Real-data
  PPC. These remain separate, decision-gated structural items.
