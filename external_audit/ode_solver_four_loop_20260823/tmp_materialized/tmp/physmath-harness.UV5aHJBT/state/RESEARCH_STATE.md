# RESEARCH_STATE

PROJECT: rec_bianchi ODE Physics-Specific Remedy Research Loop
VERSION: 0.1
CURRENT_PHASE: closeout_complete
LAST_UPDATED: 2026-08-23

## Primary question

PRIMARY_RQ: Which of the previously proposed physics-specific remedies for the current rec_bianchi ODE/DAE blockers survive source binding, primary-literature audit, independent adversarial review, and physics/mathematics validation, and what minimal decisive verification should govern implementation priority?

## Subquestions

1. Which seed remedies address a missing physical formulation or information rank, and which merely mitigate a numerical symptom?
2. Under exactly which frames, gauges, variables, conservation laws, and regimes does primary literature support each seed?
3. Which at-most-six substantively distinct remedy families cover the seed space and its strongest null/systematic alternatives?
4. Which hidden assumptions, limiting cases, or mathematical obstructions kill or narrow each serious candidate?
5. Which smallest calculation or code experiment most decisively separates each survivor from its strongest competitor?

## Scope

IN_SCOPE:

- Current checkout `main@5a09f3797210284f83a1a1adb0e0092d1ac48475` and its ODE/DAE, frame, background, collision, transport, interface, event, history, and validation surfaces.
- The prior physics-specific report only as a source of candidate seeds, never as evidence.
- Primary papers, authoritative monographs/reviews when needed, current source, analytical derivations, limiting cases, and bounded read-only calculations.
- At most six distinct hypothesis families and at most two serious survivors.

OUT_OF_SCOPE:

- Repository or test edits, dependency installation, production/long trajectory execution, scientific promotion, paper formalization, external publication, push/merge/reseal, and user-state mutation.
- Treating a contract witness, static receipt, prior narrative, or passing smoke test as implementation or trajectory authority.

## Conventions and evidence standard

- Metric signature `(-,+,+,+)` unless a cited source explicitly differs; keep `c`, `hbar`, `k_B`, and `G` explicit when dimensions matter.
- Distinguish normal-frame and hydrogen-frame tetrads, physical time `t` and `eta=ln(a)`, scalar distribution functions and frame-dependent ordinates/measures.
- Claim labels: `SUPPORTED`, `PARTIALLY_SUPPORTED`, `CONTESTED`, `UNSUPPORTED`, `MISATTRIBUTED`, `INFERENCE_ONLY`, `HYPOTHESIS`.
- Highest evidence: current executable source and bounded counterexample for implementation behavior; primary derivation/paper for method physics; independent analytic limit for conservation/frame/sign claims.
- A citation proves only its stated assumptions and scope. Absence of evidence is a gap, not a negative result.

## Hypothesis status

ACTIVE_HYPOTHESES: H-001 and H-002 are the only serious candidates, each tracked by atomic conjunct; H-003/H-004 are dependent enabling hypotheses; narrowed H-005 is an admission gate; H-006 is the rejected strong null/negative control

PROMOTED: H-001 to restricted implementation-design consideration and H-002 to specification consideration only; neither is implemented or trajectory-validated

ON_HOLD: H-003 local DAE/events and H-004 preconditioning/PTC/AP until the physical operator/reduced limit exists

REJECTED: H-006 strong harness-only/solver-only cure; exact dual-digest necessity; overbroad exact-hierarchy, universal collision-invariant, finite-spike-width, universal two-moment-remap, finite-global-DAE, and generic-IX citation claims

## Blockers

KEY_BLOCKERS: research loop has no unresolved in-scope finding; implementation, completed-residual, trajectory, and scientific blockers remain intentionally unclosed and no promotion is authorized

MISSING_EVIDENCE: compiler collision-clock identity; hydrogen spatial-axis convention; transformed quadrature; full angular residual/JVP; independent material four-force; executable E1C residual; native support geometry; family-specific providers beyond orthogonal BII; continuous-event/AP/endpoint evidence

## Approval boundary

Read, search, analyze, and update only the isolated harness workspace and scratch record. Stop before repository edits, external sharing, convention/tolerance changes, production runs, or material scope expansion.

## Gate

CURRENT_COMPLETION_BAR: all ten seeds, all prior omitted surfaces, and all fifteen current bindings have evidence dispositions and future discriminators; two serious candidates have physics/math audits, one independent gate, and one completed reconciliation

NEXT_GATE: a new separately authorized implementation-design work unit, beginning with atomic A1 frame/clock convention and A3 orthogonal-BII admission definitions

NEXT_MINIMAL_ACTION: if implementation is authorized, freeze the exact numerical storage tetrad, spatial-axis transport and collision-clock manifest before changing source
