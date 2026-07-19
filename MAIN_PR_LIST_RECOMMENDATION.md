# Main PR List Recommendation

> **Historical recommendation (PUB-00, 2026-07-12).**  This bounded five-PR
> recommendation records an earlier branch state.  Open PR-N3/R1/R2 entries are
> not current instructions.  The sole active order is in
> `docs/TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md`.

This file answers the seven required decisions and gives an ordered, <=5-PR
recommendation. Constraints enforced throughout: QKE out of scope; no public/
publication-ready claims; CPU-JAX + in-tree Rodas5P/AP65 backend; preserve raw
negative/nonfinite/failed evidence; no new readiness/manifest/hash/figure/claim
gate; no default-on optimization before parity/floor/FLRW evidence; whole-rewrite
rejected unless a >50-70% stable residual kernel is proven post-fix; anti-drift
net-line budget (prefer deletion/reuse; net <=80 preferred, >400 presumed drift).

## 2026-06-17 Status Addendum

BD491 now supplies the q4 thermal-start controlled LRS/non-LRS collision-on
pair/floor evidence that this document originally requested as PR-N1/PR-N2:

- Artifact:
  `diagnostic_outputs/bd491_pr_b_thermal_collision_on_split_current_head/bd491_q4_thermal_start_lrs_nonlrs_collision_on_parity_current_head.json`
- Pair status:
  `default_on_blocker_status=passed_pr_b_neff_floor_and_lrs_nonlrs_parity`
- LRS endpoint:
  `N_eff_3T=3.0348008780946367`, `Yp=0.24200023566442583`,
  `D/H=2.49293604970139e-05`, `Sigma_H=4.570206882454147e-31`
- Non-LRS endpoint:
  `N_eff_3T=3.0348087179727026`, `Yp=0.24201652194550552`,
  `D/H=2.493028169465174e-05`, `Sigma_H=3.3286755172789884e-31`
- Pair delta:
  `delta.N_eff_3T=7.839878065851735e-06`

This does **not** validate the solver globally and does **not** flip any
optimization default. It narrows the next work: use BD491 as the current
thermal-start q4 baseline, then pursue nonzero-shear/ell convergence,
component-wall residual attribution, and post-BD491 performance remeasurement.
Also note that the BD491 top-level artifact has `passed=false`; the controlled
PR-B pair/floor object passes, but full ladder/readiness/convergence closure is
not claimed here.

## The Seven Decisions

**Q1. Did the thermally prerun restart (PR-N1) need to precede optimization?**
**YES — and BD491 is now the current q4 evidence baseline.** PR-N1/thermal-start
was the right direction for the earlier `N_eff_3T` floor blocker; BD491 shows a
thermal-start controlled pair in band and parity. BD490/BD491 are not a
single-knob ablation, so do not claim that thermal start alone is the isolated
cause. Performance work should now be measured against the BD491-style run, not
against the old equal-temperature start.

**Q2. Does `N_eff_dist` come before or after the endpoint A/B?**
**AFTER the endpoint A/B, in parallel as a diagnostic.** The endpoint A/B evidence
now exists in BD490/BD491. `N_eff_dist` remains useful for attributing the `dA`
channel and checking the 3T proxy, but it is not a blocker on the already-run
q4 thermal-start controlled pair.

**Q3. Does step-base payload reuse stay opt-in?**
**YES — opt-in only.** It is a strong performance/memory lever (wall -18.4%, RSS
-62%) but it did **not** by itself move the physics blocker. BD491 removes the
specific q4 thermal-start pair/floor blocker, but default-on status still
requires post-BD491 component-wall remeasurement and the relevant broader
convergence/parity evidence for the setting being changed.

**Q4. Which phase-2 changes are real endpoint candidates?**
**NONE for the `N_eff` blocker; all are performance-only.** The cold endpoint is
collision/phase-2-independent (all `dQ=0` at N=4.8; Topics 01/03/07). Phase-2
work (reduce Newton iterations; delete the 391 s per-substep Python overhead) is
a *performance* candidate after BD491-style remeasurement, never a
physics-endpoint candidate. Performance and physics must be tracked on separate
ledgers.

**Q5. Is an AP65/Rodas5P architecture split needed?**
**NO — the split already exists.** The JAX-native Rodas5P background solver and
the host-side Python phase-2 corrector are already distinct layers. The dense LU
is correct for the N=17 background (0.13% wall). The real issue is host-side
phase-2 overhead and residual-bound Newton, addressed by PR-R1/PR-R2, not by a
new architectural split. (A *maintainability* extraction of the AP65 god modules
is a separate Topic-06 cleanup, behavior-preserving, not an algorithmic split.)

**Q6. What is the overengineering deletion order?**
Per Topic 06 / BD408: (1) consolidate/delete publication/readiness/SMC wrappers
(out-of-scope, low physics risk); (2) delete thin AP65 span wrapper scripts after
a reference scan; (3) consolidate duplicate JSON/hash helpers; (4) split AP65 RHS
tests by contract (count-locks -> invariants); (5) extract AP65 RHS config/
telemetry/artifact builders (behavior-preserving); (6) extract span-ladder case-
spec/replay; then freeze historical planning/docs and trim phase-2 instrumentation
(the latter overlaps PR-R1, after BD491 remeasurement). Never delete active AP65 physics or
raw-state paths.

**Q7. Any breakthrough beyond the 5-PR plan?**
**No breakthrough required, and none claimed.** The whole-rewrite bar (a proven
>50-70% stable residual kernel dominating wall after the physics fix and parity)
is **not** met: costs are distributed across phase-2, payload, and Python
orchestration in the available profiles. BD491 resolves the q4 thermal-start
pair/floor object but does not prove a rewrite target or full artifact
readiness.

## Ordered PR Plan (<=5 blocker-relevant PRs)

| # | PR | Type | Gate | Expected effect | Net-line budget |
|---:|---|---|---|---|---|
| 1 | **PR-N1** thermal-start restart (compute `T_nu0` by integrating the 3T closure from ~3 MeV to the run `T_gamma0`; never hardcode 0.8 MeV or the ratio; FLRW-scoped with a Bianchi shear-modified-Hubble extension point) | physics | — (first) | `N_eff_3T` endpoint 3.115 -> ~3.035 (in band) | Status: evidenced by BD491 q4 thermal-start controlled pair |
| 2 | **PR-N2** endpoint A/B (old equal-T vs thermal start; collision off+on; LRS+non-LRS) | validation | after N1 | confirms band movement + parity + raw-state | Status: BD490/BD491 provide equal-T vs thermal-start controlled pair evidence |
| 3 | **PR-N3** `N_eff_dist` diagnostic (distribution-energy `N_eff`, no gate, no clipping) | diagnostic | with/after N2 | attributes `dA` channel; confirms proxy | Still open; diagnostic-only |
| 4 | **PR-R1** delete per-substep phase-2 Python orchestration / reduce residual unattributed wall | performance | after BD491 re-measure | recover a fraction of measured endpoint wall without changing raw physics | Open; requires post-BD491 component attribution |
| 5 | **PR-R2** reduce Newton iterations (better AB2 initial guess / residual reuse) | performance | after BD491 and PR-R1 measurement | cut residual-bound Newton solve only if still dominant | Open; opt-in until broader parity and convergence evidence |

Cleanup PRs (Topic 06 items 1-3, net-negative wrapper/dup deletion) proceed on a
**separate track**, independent of and not blocking the physics fix, each
reporting `added/deleted/net_lines` and a blocker-movement ratio per the BD408
cost gate. PR-N4 (full-trajectory `dQ` audit) is an optional diagnostic after N3.

## What Not To Do

- Do not default-on payload reuse or any optimization from BD491 alone; require
  the relevant post-BD491 component-wall remeasurement plus the broader
  convergence/parity evidence for the setting being changed.
- Do not hardcode the 0.8 MeV start or the neutrino ratio (0.705 is wrong; 0.994
  is FLRW-only).
- Do not present any phase-2 change as fixing `N_eff`.
- Do not add a new readiness/manifest/hash/figure/claim gate.
- Do not delete active AP65 physics, raw-state, or evidence paths.
- Do not claim QKE, public-production, publication-ready, or Bianchi-extension
  validity.
