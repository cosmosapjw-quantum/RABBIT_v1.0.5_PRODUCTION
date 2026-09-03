# D-075 — Owner decisions: revise D-065 obligation 2's wording, and correct the harness gate's pass condition

**Date:** 2026-08-04
**Authority:** owner decision, taken on the two items D-074 left open and explicitly reserved.
**Gate movement:** **none.** `G-HARNESS-INTEGRITY` stays FAIL. §4 states why, and why that is not a
formality.

D-074 froze harness *development*. It did not freeze the registries, and an owner decision on a
reserved item is not development. No checker, guard, hook, or fixture changes here.

---

## 1. What is NOT edited

`docs/audit/BD622_D065_remediation_adversarial_audit_2026-07-29.md` is a **sealed audit report and
retained evidence**. Its text is not touched. Obligation 2 is superseded here the same way a stale
canary is retired — by replacement in a later decision that names it — because rewriting a finding to
match the implementation is the exact move this project forbids everywhere else.

The original wording, quoted so the change is legible:

> Minimum prospective remedy:
> 1. atomically bind one spawned `agent_id` to one run, assignment, and digest;
> **2. make receipt/lease creation failure a hard Start failure;**
> 3. add same-run substitution and receipt-write-failure negative fixtures;
> 4. pass a replacement live overlapping-run canary with unique write attribution.

---

## 2. Why the wording is revised rather than the code

**The defect F-D065-01 actually named is fixed.** Read in its own context, obligation 2 exists because
of this sentence in the finding:

> Start also catches a lease-write `OSError` and **permits Stop to fall back to mutable `ACTIVE_RUN`**,
> restoring the authority path that D-057 rejected.

The protective property at stake is *no fallback to the mutable pointer, and no admissible result from
an agent whose lease or receipt failed*. That property is implemented and measured. Canary C9's
obligation-2 measurement dispatched a Stop with a real receipt, a real assignment, a contract-valid
result and a correct admission proof, with a missing lease as the only defect, and Stop refused:

> `No Start-time run lease exists for this agent, so its run identity cannot be established. Refusing
> to fall back to the mutable ACTIVE_RUN pointer.`

**"Hard Start failure" named a mechanism, and the implementation chose a different mechanism for a
recorded reason.** `subagent_start_context.py` deliberately does not exit non-zero, and says so in its
own docstring:

> absence of a receipt for this `agent_id` is reported but is NOT fatal, because not every spawned
> agent is a registered assignment; the parent runs plenty of unregistered helper agents that were
> never meant to hold one, and Start hard-failing on all of them would make the common case
> indistinguishable from the dangerous one.

A universal non-zero exit would therefore break the ordinary case to guard the rare one. What the hook
does instead is report `Hook preflight: FAIL` in-band with the specific cause, and the lease-write
handler's own comment records the intent: *"a lease that cannot be written is a hard Start failure.
There is no ACTIVE_RUN fallback any more — SubagentStop blocks a leaseless agent."*

**The third-party auditor's finding stands and is not disputed.** It searched for
`sys.exit|os._exit|SystemExit` in the Start hook, found nothing, and concluded the literal wording was
unmet. That is correct. What was unmet is the *mechanism named in the remedy*, not the *property the
finding demanded*.

---

## 3. The revised obligation, and what it concedes

**D-065 obligation 2, as revised by this decision:**

> Receipt or lease creation failure must make an admissible result impossible, with **no fallback to
> the mutable `ACTIVE_RUN` pointer** under any failure path. `SubagentStart` must detect the failure
> and report it in-band — `Hook preflight: FAIL` with the specific cause, in the context injected into
> the agent — and `SubagentStop` must refuse the agent. `SubagentStart` is **not** required to exit
> non-zero: not every spawned agent is a registered assignment, so a universal hard exit would make
> the common case indistinguishable from the dangerous one.

**Residual this concedes, stated because a revision that hides its cost is a refit.** An agent whose
lease or receipt could not be written **still runs**. Only its *admission* is refused, at Stop, which
may be hours later. A non-zero exit at Start would have prevented the work entirely. What is given up
is therefore wasted effort and any side effects that agent has before stopping — not attribution, and
not admission. F-D065-01 is a finding about *attribution being bypassable*, and attribution is intact;
side-effects-before-refusal is a different and narrower concern, and it is now on the record as an
accepted residual rather than an unstated one.

**Status under the revised wording: DISCHARGED**, on the evidence already retained (canary C9's
obligation-2 measurement, `run-20260803-f10-d070-canary-c9/raw_logs/ob2_*`), which was adjudicated by
the round-13 registered panel and is not re-adjudicated here.

---

## 4. Why this does not move the gate

All four D-065 obligations are now discharged. **`G-HARNESS-INTEGRITY` stays FAIL**, for three
independent reasons, any one of which is sufficient:

1. **The obligations are not the pass condition.** The gate's `pass_condition` has eight conjuncts,
   of which the obligations touch perhaps two. It also requires pre-spawn non-run hashes, verified
   role-file and result-template bytes, matching runtime and assignment agent types, a blocked invalid
   first Stop with an automatically accepted corrected v2 result, external tools confined outside the
   repository, and post-run hashes proving result-only subagent writes. None of that is adjudicated.
2. **Operator and decision independence is absent**, as D-074 records. Every adjudication in this
   chain — including the round-13 panel that discharged obligations 1, 3 and 4 — was performed by
   agents the writer under review spawned, prompted, admitted and recorded. That is computational
   diversity, not a governance authority, and no receipt makes a judgment independent.
3. **Under D-073, a gate may only move with an accepted evidence package.** There is none, and
   manufacturing one for a status the writer wants would be precisely the fabrication this chain has
   repeatedly caught.

Discharging four obligations is real progress and it is not a gate. The record says both.

---

## 5. The pass-condition correction

`G-HARNESS-INTEGRITY.pass_condition` required:

> main records a current registered assignment, **exact four-line prompt**, and pre-spawn non-run hashes

while `AGENTS.md:114-116` mandates:

> Every spawned subagent MUST receive a spawn header containing **all five fields**: `RUN_ID`,
> `ASSIGNMENT_ID`, `CONTEXT_VERSION`, `INDEPENDENCE_MODE`, and `ADMISSION_TOKEN`. `SubagentStart`
> never receives the spawn prompt, so the token is the only thing that binds one agent to one
> assignment.

The registry text predates D-067, which added the fifth field. It is corrected to the five-field
contract.

**This raises the bar rather than lowering it.** The fifth field is the admission token — the entire
D-067 binding mechanism. A pass condition that asked for four lines could be satisfied by a spawn
header carrying no token at all, which is the pre-D-067 state that D-065 rejected. Correcting it means
PASS now requires the mechanism the gate exists to test.

---

## 6. Cost

```
added_lines: this document, one registry field edit, three record rows
deleted_lines: 0
runtime_behavior_changed: no
physics_behavior_changed: no
known_blocker_reduced: no -- obligation 2 is discharged under revised wording, the gate does not move
blocker_movement_ratio: 0
cost_effectiveness_verdict: RECORD_ONLY
```

Board remains **6 PASS / 2 FAIL**.
