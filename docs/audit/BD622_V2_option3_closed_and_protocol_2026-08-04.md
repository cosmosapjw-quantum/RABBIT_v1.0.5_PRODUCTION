# Option 3 is closed with no route, and V2 is sealed as a pathology probe

**Date:** 2026-08-04
**Part A** records the outcome of the owner's option 3 — develop the tail-slaving reformulation on
paper, zero compute — which is **NO ROUTE**, on independently reproduced grounds, and which also
retires the one positive result V1C had preserved.
**Part B** seals the V2 protocol, before the instrument for it exists, under the owner's standing
instruction that if option 3 found no route the work moves to option 2 **as deeper pathology
exploration, not as an attempt to reopen the instrument**.

Board unchanged: **6 PASS / 2 FAIL**. Nothing here moves a gate or meets a reopen condition.

---

# Part A — option 3: no route

Four specialists derived the closure, the solver/Jacobian axis, the fidelity budget and the
discriminator. Two independent reviewers then attacked the package. Both returned **NO_ROUTE**, and
all four specialist verdicts were negative. Everything below was reproduced by at least two agents
from the retained matrices; the items marked *(also verified here)* were recomputed directly.

## A1. The separation the slaving route needs does not exist

Tikhonov/Fenichel reduction requires the fast spectrum to dominate the retained spectrum. The
governing ratio is

```text
rho(y*) = min |Re sigma(Jdd)| / max |Re sigma(S)|
```

Measured over every admissible momentum cut on the grid:

| cut y* | dim(D) | min abs Re sigma(Jdd) | max abs Re sigma(S) | rho |
|---|---|---|---|---|
| 6.57 | 123 | 491 | 1513 | 0.324 |
| 15.39 | 87 | 2302 | 3865 | **0.596 — the maximum anywhere** |
| 19.97 | 69 | 3045 | 5806 | 0.525 |
| 24.07 | 51 | 3284 | 9873 | 0.333 |
| 29.85 | 6 | 4009 | 4.38e5 | 0.009 |

`rho` never approaches 1. The expansion parameter is `1/rho`, i.e. **1.7 to 2.7 — not small, at any
cut.** Concretely the spectra interleave: `sigma(S)` spans [−8774.8, −6.43] while `sigma(Jdd)` spans
[−3.54e8, −3273.5], so **29 of the 126 "slow" modes are faster than the slowest "fast" mode.**

**The "factor of 504" cited in V1C §5 is the wrong quantity.** It compares the slowest *fast* mode
(−3273) with the slowest *slow* mode (−6.43). That is a stiffness ratio and is silent on
separability. Both reviewers identified this independently. **V1C §5's "what survives, and is
stronger after the correction" is therefore withdrawn**, and this document supersedes it.

## A2. Three further refutations, each sufficient on its own

1. **The control has the same structure.** The 48/24 configuration, which *completes*, exhibits the
   same failing ratio (`rho` ≈ 0.35–0.58), the same cut-invariant Schur abscissa and the same
   endpoint-localised fastest modes. Whatever distinguishes stalling from completing, it is not this.
2. **The rigorous version buys nothing and the useful version is not rigorous.** The only genuinely
   gapped subsystem has **dimension 3** (consecutive-eigenvalue ratio 9284), localised at
   y = 29.55–29.94, one mode per species. Slaving it removes 3 of 182 Jacobian columns.
3. **The justification dies in the epoch that decides the answer.** `absc(Jdd) = −3273` and
   `absc(S) = −6.43` are *collision rates* measured at N = 0.1258, deep in the collision-dominated
   regime. Neutrino decoupling is precisely the process by which those rates fall to zero. The
   fast/slow structure the closure relies on therefore disappears over the 97.9% of the trajectory
   that was never attempted.

Two mechanical results also close the practical route: the explicit/affine tail closure is refuted
because `Jdd` is not diagonal (row-wise off-diagonal to diagonal ratio, median 19.7, maximum 878), so
the closure is a **54-equation nonlinear solve whose every residual costs a full collision
right-hand side**; and the cost model gives at most **1.25x** at one inner iteration and is *slower*
at two or more, against a miss of **~8,305x**.

## A3. The owner's Jacobian question, answered

**The noun is right and the adjective is wrong.** The Jacobian is the bottleneck — but as an
*accuracy* defect, not a construction cost.

- Construction cost is 68.9% (domain) and 61.0% (base) of all right-hand-side calls. **Deleting
  Jacobian construction entirely still misses the wall budget by 77x to 694x.** It is not the answer.
- The historically recorded dead ends stand and were not re-proposed without explanation. Rodas5P
  remains 2-3x slower than BDF on the production endpoint because of its 8-stage-per-step multiplier
  against an expensive right-hand side, which no linear-algebra backend addresses. Lagged/frozen
  Jacobians remain counterproductive at rtol ≤ 1e-6. Krylov, banded preconditioning and AD remain
  inapplicable or unprofitable for a dense collision Jacobian.
- The one thing that *is* new is that the accuracy defect is very likely the proximate cause of the
  stall — and **fixing it is disqualified**: pinning the finite-difference step or supplying an
  analytic Jacobian is neither an asymptotic reformulation, a rigorous surrogate bound, nor an
  analytically constrained domain reduction, and V0 §7 names instrument tuning in response to a
  diagnosed pathology as out of bounds. **The axis is a diagnosis, not a remedy.**

## A4. One recorded fact was computed on the defective matrix

The "operator feedback = 3.1582e-4, 1034x the direct channel, the binding one" figure — carried into
V1C §5 from the meta-audit — reproduces exactly on the **stored, ratchet-defective** Jacobian and
does not carry over unchanged to the corrected one. It is withdrawn as a quantitative claim; the
qualitative point that feedback dominates the direct tail channel is retained as unquantified.

## A5. What option 3 leaves standing

Nothing that reopens anything. What it produced is a clean negative: **the 60/30 operator has no
usable fast/slow splitting at any momentum cut, the property is shared with the configuration that
completes, and the structure it does have dissolves as decoupling proceeds.** The stall mechanism
remains unmeasured, and every account of it — mine, the audit's, and the specialists' — is inference
from a window that ends before the creep begins.

---

# Part B — V2 protocol, sealed before the instrument exists

**Purpose: observe the creep directly.** V1 stopped at raw call 907; the r4 creep runs from
evaluation 951 to 11,051. Every mechanism claim so far concerns a regime *before* the phenomenon.
V2 exists to end that.

**V2 is not an attempt to reopen the gate.** Per the owner's instruction it is a pathology probe.
Its claim ceiling is identical to V0's: it cannot satisfy any D-071 reopen conjunct, cannot move a
gate, is not an evidence package, `implies_status: null`.

## B1. The governance fix, which is the reason to trust this run more than the last

V1's decision was wrong for three reasons that had nothing to do with physics: the sealed predicates
were **re-typed** into the analysis code and silently weakened, and nothing compared the two. The
instrument mechanically bound *borrowed* code to its pinned original and left the writer's *own*
sealed criteria unbound.

**V2 seals its decision criteria as machine-readable JSON inside this document, and the analysis
code must load them from here.** The analyser will refuse to run unless:

- the criteria block parses and its sha256 matches the value recorded at seal time;
- every mechanism id in the block has an implemented predicate, and every implemented predicate has
  an id in the block (a two-way cover check, so neither side can drift);
- every conjunct listed for a mechanism is evaluated and reported individually, with the decision
  formed from the recorded conjunction rather than from a hand-written summary;
- the decision function consults **all** sealed triggers, including "a cap reached first".

A predicate the code cannot evaluate is a `FAIL_PROTOCOL`, not a silently dropped conjunct.

## B2. Pins

```text
input_commit_relation      pinned commit must be an ancestor of HEAD with no change under
                           src/, scripts/, .agent-harness/, .codex/ and those paths clean
frozen_module_sha256       760a7c044081e507fae9d5695b301bd44f6466d96322c46f53b77161e32b558a
python / numpy / scipy     3.12.3 / 2.4.2 / 1.17.0
scipy_bdf_source_sha256    a5b75a2ae8aca2e66cc35ed268af51d8a9685878209d5cf77a44c8ccef3b76e6
configuration              order 60, y_max 30.0, rtol 1e-6, atol 1e-9, t_start 10.0,
                           t_gamma_end 0.005, incoming_polar_order 4, final_polar_order 4,
                           electron_radial_order 24
```

## B3. The one design decision that matters

The integration must reproduce r4 **exactly**, so `num_jac` is left completely untouched — the
ratcheting stays, because it is part of the phenomenon under study. Accurate Jacobians are obtained
**as observations only**, at checkpoints, computed with a pinned converged step and **never fed back
into the solver**. The trajectory is therefore faithful and the analysis is accurate at the same
time.

The datum V1 could not supply and V2 must: **`num_jac`'s per-component `factor` vector, logged at
every refresh.** That is the direct evidence for or against the ratchet hypothesis, and it costs
nothing.

## B4. Caps

```text
integration raw RHS calls   4000          (V1 reached 907; r4's creep begins at 951)
wall                        6.5 hours, enforced by a monotonic deadline across ALL phases
checkpoint Jacobians        at most 4, each <= 200 raw calls, pinned step sqrt(EPS), observation only
cycle cap                   NONE -- V1's cycle cap is what stopped it before the phenomenon
```

Breaching a cap ends the phase and is reported. Extending one because the answer is inconvenient is
forbidden.

## B5. Sealed decision criteria

The analyser loads this block by its sha256 and refuses to run on any mismatch. Every conjunct is
evaluated and reported separately.

```json
{
  "schema": "bd622-v2-criteria-1",
  "mechanisms": {
    "P1_ratchet_loop": {
      "question": "does num_jac's step factor ratchet during the creep, and do Newton failures follow it?",
      "conjuncts": [
        "max over components of jac_factor at the last refresh exceeds 100x its initial sqrt(EPS) value",
        "the components whose factor ratcheted are the same components carrying the largest relative deviation between the ratcheted and the pinned-step checkpoint Jacobian",
        "at least one Newton failure occurs with jac_was_current true after the factor has ratcheted"
      ]
    },
    "P2_step_collapse": {
      "question": "does the accepted step size collapse during the creep, and is the error test or Newton the binding constraint?",
      "conjuncts": [
        "median accepted h over the last 500 integration calls is below one tenth of the median over the first 500",
        "error-test failures per accepted step over the whole phase exceed 0.5"
      ]
    },
    "P3_newton_limited": {
      "question": "is the creep Newton-limited rather than accuracy-limited?",
      "conjuncts": [
        "error-test failures per accepted step is below 0.1",
        "Newton non-convergences per accepted step exceeds 1.0"
      ]
    },
    "P4_throughput": {
      "question": "what is the measured creep throughput, in accepted steps and in N per raw call?",
      "conjuncts": [
        "at least 5 accepted steps occur after raw call 951",
        "N advance per raw call after call 951 is below one hundredth of the value before call 351"
      ]
    },
    "P0_no_creep": {
      "question": "null: the run does not reproduce the creep at all",
      "conjuncts": [
        "N advance per raw call after call 951 is at least one tenth of the value before call 351"
      ]
    }
  },
  "decision_rule": {
    "FAIL_PROTOCOL": "a pin mismatch, a criteria-digest mismatch, a two-way cover failure, an unevaluated conjunct, or lost raw failure data",
    "INCONCLUSIVE": "the phase ends before raw call 951, or P0_no_creep holds, or no mechanism has all conjuncts true",
    "MECHANISM_IDENTIFIED": "exactly one of P1..P3 has all conjuncts true",
    "MULTI_MECHANISM": "more than one of P1..P3 has all conjuncts true"
  },
  "claim_ceiling": "diagnostic evidence only; cannot satisfy any D-071 reopen conjunct; implies_status null"
}
```

## B6. Outputs and preservation

Line-buffered from the first byte — V1 held three of six streams entirely in memory, which would
have lost 100% of them on an abnormal exit. Every failure, exception and cap breach is retained
verbatim, and the output root is never deleted or reused between attempts, so a `FAIL_PROTOCOL`
payload survives (V1's did not).

## B7. What V2 cannot do

It cannot reopen the gate, cannot license tuning, and cannot answer the gated physical question. If
it identifies the mechanism, the mechanism will in all likelihood be one whose repair D-071
disqualifies. That is understood and accepted in advance: the owner asked for pathology exploration,
and knowing why the instrument fails is worth having even when the knowledge authorises nothing.

```text
added_lines: this document
runtime_behavior_changed: no
physics_behavior_changed: no
known_blocker_reduced: no -- option 3 closed negative; V2 sealed
blocker_movement_ratio: 0
cost_effectiveness_verdict: RECORD_ONLY
```
