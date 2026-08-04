# Blocker-resolution brainstorm — evidence, hypothesis space, adversarial review, verification design

**Date:** 2026-08-04
**Status:** PROPOSAL for owner decision. Nothing here moves a gate, edits a registry, spends a
contract, or touches frozen machinery. Both gates keep their recorded statuses; the trajectory
lane stays closed on its preserved measurement per D-071; the harness freeze (D-074) stays in force.
**Method:** physmath-research-harness pipeline (evidence-acquisition → hypothesis-space →
adversarial-review → verification-design), run with a 6-agent web-evidence workflow and two
independent adversarial reviewer agents. Reviewer independence is computational only — the writer
spawned and prompted them. That recorded limit applies to this document itself.

---

## 1. New evidence this exercise produced

### 1.1 From the retained bytes (reviewer findings, checkable locally)

- **F-R1 — the "unexplained drop" is explained.** The r4 progress line is printed from inside the
  RHS (`scripts/audit/_trajectory_core.py`, every 50 evals), so it prints **trial** points, not
  accepted state. The 0.1813 "peak" (eval 751) was never accepted state; the −0.0184 "drop" at
  eval 951 is the rejection fallback to just past the last accepted state (eval-701 vicinity).
  D-071 §2.3's "unexplained DROP" reading is over-mystified; rejections are already inside the
  measured creep window, which makes the 4.58-yr projection *more* defensible, and removes the drop
  as a stand-alone anomaly.
- **F-R2 — the creep is FD-Jacobian refresh/rejection cycling.** Domain-phase state dimension is
  3×60+2 = 182; scipy BDF's finite-difference Jacobian costs ~182 RHS evals per refresh. The
  post-drop window's 10,100 evals ≈ 55.5 × 182: the creep's wall is almost entirely Jacobian
  refreshes. The trace shows 150–250-eval plateaus at single T_cm values and repeated
  forward/backward trial retreats — a discrete reject-refresh limit cycle, not smooth honest tiny
  steps. A second, *escaped* stall episode exists at holdout startup (evals 1–301, ΔN=0.0007).
- **F-R3 — the three-order-of-magnitude excess is a local fact.** The same instrument at 48/24
  completed in 3,694 evals; 60/30 projects 3.27e7. No web trust needed.
- **F-R4 — nothing of the stalled phase survives.** `r4_trajectory_report.json` has
  `domain_holdout: null`. What *is* retained: base-phase `checkpoint_states`, endpoint spectrum,
  per-eval rejection counters, `max_roundoff`.
- **F-R5 — instrumentation is wrapper-feasible.** The solver loop (`solve_ivp(..., method="BDF",
  rtol=1e-6, atol=1e-9)`) lives in `scripts/audit/_trajectory_core.py`, not in the frozen module.
  A diagnostic driver can step-drive scipy BDF against the frozen module's public API with zero
  edits to `_independent_noqke.py`.

### 1.2 From the literature (6 axes, each verified against primary sources by its agent)

- **Practice check:** every production code on the same or harder physics (FortEPiaNO Ny 20–100;
  Bennett et al. Ny 60–80, y_max 20–30; NEVO N=80 with an analytic O(N³) collision Jacobian;
  nudec_BSM) integrates with LSODA/BDF at tolerances 1e-6..1e-7 and completes full decoupling
  evolutions in minutes-to-hours on desktops — lifetime evaluation budgets of order 1e4–1e5. No
  published analogue of a 1e7+-step requirement exists for this physics. The one published blow-up
  mechanism (adaptive solver vs unresolved sharp features, FortEPiaNO App. B) is consistent with
  the F-R2 signature.
- **BDF stall taxonomy:** creep near quasi-equilibrium is one of four documented, separable
  mechanisms — A(α) stability-wedge limitation from weakly damped oscillatory modes (STALD detects
  this from step history at 2–7% overhead); error-estimator noise floor (acute here: the collision
  term is a near-cancellation of gain and loss terms); Newton-failure cycling; genuinely fast
  physics visible in one dense eigendecomposition. Signatures are disjoint-ish and standard
  counters discriminate them.
- **Asymptotic route precedent:** Esposito et al. 2000 integrate delta-f in 4 Fermi-weighted
  orthonormal polynomials (17 stiff ODEs; the method line behind classic N_eff results).
  AP/exponential/BGK-penalized reformulations carry uniform-in-stiffness proofs including
  Fermi-Dirac equilibria. Caveat: every proven step-count win is vs explicit/semi-implicit
  baselines, never vs an already-implicit BDF in creep.
- **Rigorous surrogate route:** off-the-shelf machinery is dead at this scale (validated
  integrators top out ~2–12 nonlinear states; ROM bounds are exponential-in-time or mere
  estimates). One live mechanism: defect + logarithmic-norm certificate (Söderlind 2006; Wirtz et
  al. 2014, effectivity ~10 on dissipative problems). Decisive unknown: the one-sided Lipschitz
  constant μ of the collision+expansion Jacobian in a weighted norm — physics pushes μ→0 exactly
  where certification is needed.
- **Domain reduction:** y_max 30→20 is analytically certified for *equilibrium* content
  (incomplete-gamma bound, ≤3.4e-6 relative in the energy moment at y_max=20; 5e-10 at 30; below
  20 measured-invalid). CLASS completes tails analytically in production.
- **Harness independence:** every surveyed framework decomposes independence into EXECUTION /
  ATTESTATION / JUDGMENT. Execution and attestation are purchasable (hosted CI + attestations;
  Rekor anchoring makes backdating impossible; private repos do not write to the public log and
  must anchor separately). JUDGMENT is irreducibly a human or organization outside the writer's
  control (IEEE 1012 managerial independence; ACM "person or team other than the authors";
  CODECHECK; registered reports). No framework treats reviewer multiplicity under one controller
  as independence.

## 2. Candidate families and their adversarial-review outcomes

Two reviewer agents (read-only, no candidate rewriting, no promote/kill authority) reviewed the
ten families. Both reviews are quoted from their returned reports; full texts retained in the
session task outputs.

### Trajectory lane

| Family | Class | Reviewer role | Load-bearing amendment |
|---|---|---|---|
| H-T1 diagnosis-first instrumented replica | (instrument) | DEFENDED | pre-register the mechanism prediction set before running; log the kinematic-domain rejection counter live |
| H-T2 asymptotic delta-f reformulation | (a) | NEEDS_MORE_EVIDENCE | mechanism as written was wrong: plain delta-f keeps the fast spectrum and fixes the *cancellation noise floor*; only micro-macro/AP closure removes fast modes; fidelity budget ~1% of a ~0.04 distortion ≈ 4e-4 vs the 2e-4 band — a convergence ladder is mandatory |
| H-T3 confound-separating discriminator + certified tail | (c) | TRADEOFF | my first-node claim was **false**: at fixed y_max 24, order 60 gives first node ≈ 0.0147×(48/60)² ≈ 0.0094 — *closer* to zero than the stalled config's 0.0118. The run discriminates drivers; it does not presumptively complete. The analytic leg must bound three channels (equilibrium storage + distortion tail + operator feedback), not one. Avoid order 64 in any ladder or carry an explicit lane-scope adjudication |
| H-T4 defect + log-norm certificate | (b) | TRADEOFF | kill test runs on base-phase `checkpoint_states` (the stalled phase retains no state); "rigorous" needs tube-level control, else label as scientific-computing-grade; pre-register numeric tripwires |
| H-T5 instrument-pathology interpretation | (frame) | NEEDS_MORE_EVIDENCE | partially pre-confirmed by F-R2, so the registered prediction must be sharpened to "pathology signature **and** a driver absent from at least one allowed alternative construction", else confirmation is vacuous |

Reviewer's structural conclusion: **the evidence graph is a DAG with H-T1 at the root.** H-T2,
H-T3, H-T5 are H-T1-conditional; H-T4 is the only H-T1-independent candidate and carries its own
cheap self-kill test. One trap stated explicitly: if the diagnosis returns a controller/tolerance
pathology, the obvious cure (order cap, atol floor, analytic Jacobian) is **explicitly not a
reopen** — those are the disposition's disqualified "generic optimization" items. H-T1's value is
to select among reformulation-class candidates and kill the wrong ones, not to license tuning.

### Harness gate

| Family | Reviewer role | Load-bearing amendment |
|---|---|---|
| H-H1 external execution + attestation | TRADEOFF | counterexample: **run-picking** — writer runs N times, discloses the green one; every cryptographic property survives. Fix: trigger on every push to the audited branch + anchored monotone run index, plus a published conjunct-coverage map (live hook-runtime conjuncts are unreachable from CI) |
| H-H2 external human judgment + frozen commitment | DEFENDED (design) | two mandatory amendments: freeze the **adjudicator identity** (or a deterministic recruitment procedure) before results, not merely the criteria — else reviewer-shopping defeats it; add a conjunct-scope annex (own re-execution vs judging retained bytes, per conjunct). Also: the signed verdict is a tamper-evident *input* to a writer-authored package, not the package |
| H-H3 second local UID | REJECTED_SHORTCUT | `sudo` crosses it; implementation breaches the freeze; C13 depends on the shared user |
| H-H4 transparency anchoring of the existing corpus | DEFENDED (narrow) | anchor record must carry the negative scope statement ("attests existence-at-time only; nothing about content honesty or implies_status"); worth doing only paired with a live H-H1/H-H2 process, else orphaned ceremony |
| H-H5 accept the permanent FAIL as terminal | TRADEOFF | genuinely competitive, not a strawman; but "zero drift" is false (nothing enforces limitation-text propagation) and "the gate never moves" is a choice presented as necessity. Cheap decisive test: attempt H-H2 recruitment; measured failure upgrades H-H5 to the defended terminal state |

Reviewer's structural conclusion: **no combination without H-H2 can move the harness gate** —
of D-075 §4's three independently sufficient reasons, only H-H2 touches judgment independence.
H-H1/H-H3/H-H4 are ingredients of an H-H2-bearing package or components of the H-H5 terminal
state, never freestanding routes.

## 3. Verification design — the smallest decisive test per surviving candidate

Nine fields per the verification-design discipline. Ordered by the priority rule: cheapest test
that discriminates the most candidates first.

### V1 — instrumented stall diagnostic (executes H-T1; discriminates H-T2/H-T3/H-T5)

1. **Testable claim:** the 60/30 domain-holdout creep is one (or a set) of: A(α)-wedge stability
   limitation; error-estimator noise floor; Newton-failure cycling; RHS kinematic-boundary
   discontinuity; genuine fast modes.
2. **Required inputs:** wrapper driver around scipy BDF via the frozen module's public API
   (F-R5); logging of h, order, error norm, error-test vs Newton-failure counters, per-eval
   kinematic-domain rejection counter, RHS gain/loss magnitudes separately (noise-floor probe);
   one dense 182-dim Jacobian eigendecomposition at the stall state; ~2–13 h wall.
3. **Expected result if the pathology frame is correct:** a discrete signature — rejection
   cycling tied to an identifiable driver (near-origin components, high-y bins, kinematic
   boundary, or noise floor) — sharpened per the reviewer: a driver absent from at least one
   allowed alternative construction.
4. **Expected under the strongest competitor (intrinsic cost):** well-conditioned Newton, few
   rejections, error tests honestly passed at tiny h, eigenvalues demanding that h.
5. **Pass criterion:** the trace assigns ≥1 mechanism with a quantitative signature (e.g.
   >50% of wall in Jacobian refreshes following rejections; or gain/loss cancellation ratio
   putting the RHS noise floor above atol=1e-9/rtol=1e-6 demands).
6. **Kill criterion (for the pathology frame):** the intrinsic-cost signature of field 4.
7. **Ambiguity condition:** mixed signature with no dominant driver, or non-reproduction of the
   stall.
8. **Cheapest next action:** bank the pre-registered prediction set (dated, before any run
   byte), then run the diagnostic. No contract, no gates, no frozen edits, no band touched.
9. **Escalation path:** driver identified → draft the matching reopen contract (V3 if
   construction-specific driver; H-T2's AP sub-route if genuine fast modes; V2's certificate
   regardless); intrinsic-cost outcome → only H-T2 (with its fidelity ladder) and V2 remain.

**Compliance note:** uncontracted diagnostic ⇒ post-hoc selection surface. Mitigation is field
8's pre-registration, consistent with the parallel-run grant's "all axes gated and reported, no
post-hoc selection".

### V2 — μ kill test for the defect+log-norm certificate (executes H-T4's gate; H-T1-independent)

1. **Testable claim:** the collision+expansion Jacobian has a usefully negative one-sided
   Lipschitz constant μ(t) in some weighted norm along the epoch.
2. **Required inputs:** retained base-phase `checkpoint_states` (verified present, F-R4); dense
   Jacobian evaluations at ~1e3 checkpoints (~1–2 h at 4.42 s/eval); a small weighted-norm search.
3. **Expected if H-T4 viable:** μ(t) ≤ μ* < 0 across the epoch in some tested norm, with the
   projected certificate ≤ 1e-4 in the ΔN_eff functional.
4. **Expected under competitor (certificate uselessness):** μ → 0⁻ or sign-indefinite as rates
   die; or projected bound > 2e-4.
5. **Pass criterion (pre-registered tripwire):** projected certificate ≤ 1e-4 at measured μ and
   estimated defect scale.
6. **Kill criterion (pre-registered tripwire):** μ not usefully negative in any tested norm, or
   projected certificate > 1e-4 — stop before any norm-optimization rabbit hole.
7. **Ambiguity condition:** μ marginally negative with projected bound in (1e-4, 2e-4): formally
   decisive-capable but with no margin — treat as kill for contract purposes.
8. **Cheapest next action:** the μ computation itself; hours, no new physics code.
9. **Escalation path:** pass → design the tube-level (rigor-labeled) certificate + surrogate
   tail ansatz for y∈(24,30] with its defect bounded, then a class-(b) reopen contract; kill →
   record as an informative negative in the graph, H-T4 closed.

### V3 — confound-separating discriminator (executes H-T3; conditional on V1)

1. **Testable claim:** T13's stall driver is specific to the domain-extension axis of the 60/30
   construction, and the representation-convergence question L7 actually gates can be measured by
   a density-only companion plus a three-channel analytic bound on domain extension.
2. **Required inputs:** V1's driver identification; corrected first-node arithmetic (order 60 at
   y_max 24 ⇒ first node ≈ 0.0094 — *closer* to the origin; the run discriminates near-origin vs
   high-y drivers rather than presumptively completing); drafted three-channel tail bound
   (equilibrium storage via incomplete gamma + distortion tail + operator feedback) in the ΔN_eff
   norm, presented before any output byte; a 1–2 h prefix probe covering the stalled early-N
   phase as the reviewed discriminator required by reopen condition 3.
3. **Expected if correct:** prefix probe of the chosen alternative construction traverses the
   stalled N-window at a rate projecting completion inside the 18 h budget with margin.
4. **Expected under competitor:** the alternative construction creeps identically (driver is
   order-conditioning, not domain extension) — projection fails.
5. **Pass criterion:** projected completion ≤ 50% of the frozen wall budget, from the probe's
   measured rate over the previously-stalled window.
6. **Kill criterion:** probe rate projects > 100% of budget, or the three-channel bound cannot
   be stated at ≤ 2e-4-compatible looseness.
7. **Ambiguity condition:** probe projects 50–100% of budget — margin requirement unmet;
   redesign, do not contract.
8. **Cheapest next action:** the tail-bound draft (paper exercise, zero runtime), since its
   failure kills the family without any run.
9. **Escalation path:** pass → prospectively sealed class-(c) contract (avoiding order 64 or
   carrying the lane-scope adjudication), reviewed before implementation per the disposition;
   kill → H-T2/V2 remain.
10. **Self-destruct clause (reviewer-mandated):** if the analytic leg fails, this family
    degenerates into "same instrument, easier config" = disqualified generic optimization, and
    must be recorded as such rather than run anyway.

### V4 — adjudicator-recruitment probe (executes H-H2's bottleneck; simultaneously decides H-H5)

1. **Testable claim:** at least one qualified external human, outside the writer's control, will
   accept a bounded adjudication engagement over the harness gate's eight pass_condition
   conjuncts with a reproduction bundle.
2. **Required inputs:** owner outreach (CODECHECK codechecker community is the concrete named
   route; a colleague qualifies); a one-page scope statement; NO repo changes.
3. **Expected if viable:** a named adjudicator (or deterministic recruitment procedure) that can
   be frozen into a prospective commitment *before* any verdict work begins.
4. **Expected under competitor (H-H5):** recruitment measurably fails — no qualified acceptor
   within the owner's chosen window.
5. **Pass criterion:** named adjudicator + agreed scope, recorded before any adjudication byte.
6. **Kill criterion:** no acceptor within the window ⇒ H-H5 becomes the defended terminal state
   (with its two amendments: anchor the terminal corpus; hold the limitation text as a
   registry-referenced artifact cited by id).
7. **Ambiguity condition:** an acceptor who declines verdict authority (pure CODECHECK-style
   execution witness) — valuable, but covers execution only; judgment stays open and the record
   must say so.
8. **Cheapest next action:** the outreach message. Zero repo cost.
9. **Escalation path:** pass → assemble the reproduction bundle + conjunct-scope annex + frozen
   commitment (OSF registration or Rekor-anchored contract), then the adjudication itself, whose
   signed verdict enters a writer-authored evidence package as a tamper-evident input with the
   last-mile mediation stated in `limitations`; components H-H4 (scope-capped anchor) and H-H1
   (anti-run-picking CI, conjunct-coverage map) attach here as ingredients.

### V5 — components, only-if-paired

- **H-H4 anchor:** one-time hash of the sealed evidence tree + registries anchored externally,
  carrying the negative scope statement verbatim. Execute only once V4 has a live outcome either
  way (pass: anchors the bundle the adjudicator receives; kill: anchors the terminal corpus).
- **H-H1 CI:** greenfield `.github/workflows/` re-running the deterministic suite on every push
  to the audited branch with an anchored monotone run index and a published conjunct-coverage
  map. An owner decision on scope is required despite being outside the frozen dirs, because
  D-074's spirit ("continuing to add machinery…") applies even where its letter does not.

## 4. Recommended sequencing (owner decisions, not commitments)

```
now ──► V1 diagnostic (2–13 h, pre-registered)     ──► selects among H-T2-AP / V3 / kills frames
    ──► V2 μ kill test (hours, tripwired)          ──► H-T4 lives or dies cheaply
    ──► V4 recruitment probe (outreach, zero repo) ──► H-H2 proceeds or H-H5 becomes terminal
then, and only with the above in hand:
    V3 tail-bound draft → prefix probe → sealed class-(c) contract   (if V1 supports it)
    H-T2 fidelity ladder → AP sub-route contract                     (if V1 shows fast modes)
    V5 components attach to whichever harness outcome V4 produced
```

The three head items are independent, cheap, and each has a pre-registered kill. None requires
unfreezing anything, none touches a contract, none moves a gate. Every reopen-class follow-on
requires its own prospectively sealed contract reviewed before implementation, per the
disposition — this document does not draft those contracts.

## 5. What this document does not do

No gate movement (no accepted evidence package exists). No harness unfreeze. No contract. No
claim that the trajectory question is "actually easy" — the reviewer's metacognitive probe stands:
the pathology frame is evidence-based but outcome-flattering, and its decisive test is V1, not
this prose. The two errors the adversarial pass caught in the writer's own hypothesis graph
(H-T3's first-node direction; H-T4's nonexistent retained trajectory) are preserved in §2's table
as evidence that the review loop is load-bearing, not ceremonial.
