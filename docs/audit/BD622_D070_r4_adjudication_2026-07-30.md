# BD622 D-070 Part B7 — adjudication of the D-069 r4 execution

Date: 2026-07-30
Lane: owner-granted 2026-07-29 blocker-clearance chain
Contract adjudicated: `docs/audit/BD622_D069_independent_trajectory_r4_contract_2026-07-29.md`, frozen `0fb94f2`
Evidence banked: `ce737a7`
Status: **ADJUDICATED — BOTH AGENTS RETURNED `fail` — NO GATE MOVEMENT — BOARD STAYS 6 PASS / 2 FAIL**

## 1. The outcome

The frozen driver exited 20 with `verdict: ERROR`, `error: "wall budget: frozen
wall budget exceeded"`, `wall_seconds` 64801.9 against the frozen 64800 s (18 h)
budget. Contract §7 pre-declared this class: *"a breach yields no `checks` block
at all — phase front-loading preserves the evidence, not the verdict."*

The adjudicator established the exit path from the bytes rather than from the
coordinator's account. `Deadline.check` (`_trajectory_core.py:71-73`) is the only
`raise TimeoutError` in the driver tree and its message is exactly `frozen wall
budget exceeded`; `d069_independent_trajectory_r4.py:586-589` is the only handler
that formats `f"wall budget: {exc}"`; the report's `error` matches
character-for-character. `evaluate()` (`:609`) and the verdict conjunction
`all(c["ok"] for c in checks.values())` (`:614`) sit **below** the try/except and
are unreachable on that path. **Zero of the fourteen T-predicates was
constructed.** An occupation excursion would have written a different string
(`IndependentNoQkeError: occupation left (0, 1)`), so the class is identified,
not inferred.

The exit is mechanical. Its *cause* is numerical, and that distinction is the
substance of this adjudication.

## 2. No gate moves, in either direction

`G-F10-INDEPENDENT-FLRW` cannot reach PASS: there is no verdict, T12 and T13 are
structurally unevaluable because their phases produced no report block at all,
and F-D065-04 items 4 and 5 are undischarged.

It also cannot be **re-failed** by r4. The gate is already FAIL on F-D065-04, and
r4 supplies no new grounds — it reproduces the preserved D-063 payload bitwise
and contradicts nothing. Any record implying r4 re-established the FAIL would be
wrong. `C-F10-INDEPENDENT` moves in neither direction for the same reasons.

What r4 *does* falsify is a feasibility assumption inside the D-069 contract
itself.

## 3. The T13 instrument is measured non-viable

Not budget-starved — non-viable by three to four orders of magnitude.

Over the domain phase's logged span `N` moved 0.1628 → 0.1653 across 10700
evaluations and 13.145 h: a rate of **2.34e-7 N/eval**. The terminal condition is
`N` = 7.9367. At the observed rate the remaining 7.7714 needs ~3.3e7 further
evaluations ≈ **4.7 years** of wall time on this host — ~2270× the whole 18 h
budget and **~8450×** the contract's 3938-evaluation projection.

Observed `N` peaked at 0.1813 = **2.284 %** of the target and stood at 0.1653 =
2.083 % at the last logged evaluation, oscillating 0.1628 / 0.1813 / 0.1629.
That is the signature of repeated BDF step rejection.

**The cost model is not what failed.** Eval-to-eval cost measured **4.4211
s/eval** at 60/30 against the contract's asserted 4.754 — conservative by 7 %.
Base measured 2.6579 against 2.807. Base wall 2.727 h and mutants 1.678 h landed
inside their 2.9 h and 1.6 h projections. What failed is the **step-count**
assumption.

**The retained bytes cannot diagnose the stall.** The domain phase emitted only
`eval / N / T_cm / T_gamma / ratio` every 50 evaluations. There is no step-size
trace, no rejected-step counter, no error norms, no Jacobian statistics, and —
because the report carries no `domain_holdout` block — not one occupation value
from that integration survives. Which state components drive the rejection is
unknown and unrecoverable. Any replacement configuration would therefore be a
guess, and this report does not offer one.

## 4. What this does to F-D065-04

Item 5 reads: *"The holdout varies `rtol` only; it supplies no radial, angular,
domain, or tail uncertainty."* The D-069 answer was T13, and T13 was already only
a partial answer by the project's own record — the frozen D-069 row and both
ledger rows state that the angular and collision-radial parts remained
undischarged.

r4 removes the part that was on offer. **Item 5 is undischarged in full** —
radial, angular, domain and tail — and item 4 (evolved beyond-domain enclosure),
which shared the same instrument, falls with it.

**`G-F10-INDEPENDENT-FLRW` remains FAIL, and on current measurement it has no
available route out of that state.** The adjudicator states that as the stable outcome rather than proposing a
cheaper configuration that merely fits the budget, and this report does the same.

## 5. What survives, and at what scope

Reproduction, not verdict. Recorded so the surviving evidence is not hidden, and
labelled so it cannot be read as a partial PASS.

- Base endpoint `N_eff` 3.034054308076679, `N_end` 7.936698865363719, `t_end_s`
  52678.731916604454, 3694 evals.
- **T14 verified by the adjudicator at a strictly stronger scope than the driver
  ran it.** Rather than trusting the reported `W_nu_rel`, it re-derived the
  occupations from the stored `cloglog` via `-expm1(-exp(c))` — all **4464**
  stored values (31 × 3 × 48) reproduce exactly — and reconstructed `T_cm` from
  `N` alone. All six re-derived quantities agree bitwise. **F-D065-04 item 3
  (recomputable checkpoints) is genuinely closed.**
- Both mutants killed, but **not at equal evidential quality**, and the asymmetry
  travels with the numbers: `M2_qem_sign` is a **physics kill** (`R_transfer`
  2.0076 / 2.0009 / 2.0113 at N = 2.05 / 2.10 / 2.15, all active, against
  `KILL_R_TRANSFER` = 1.0) on 373 evaluations; `M1_pair_sign` is a
  **solver-failure kill** on 1853 evaluations with `max_R_transfer` **absent from
  the block** because no residual was ever computed. The sign flip destabilised
  BDF before any physics predicate saw it.
- T8c in-support sup 1.5881626368027338e-3; full-domain sup
  **2.0800641747316935e-3**, non-gating, against the ~2.080e-3 that §6
  pre-committed *before any output byte* — which is why it cannot read as an
  excuse invented afterwards.

**No measurement recorded outside a `checks` block can move a gate.** The
contract says so prospectively, the verdict is a fourteen-way conjunction, and a
partial conjunction is not a verdict.

## 6. Corrections to the coordinator's own record

Three of the last four defects in this chain were in the record rather than in
the code. The panel was pointed at the record deliberately, and found five more,
plus two advisories (the last table row and the paragraph below it).

| Coordinator wrote | Correct statement |
|---|---|
| "a fourth independent execution" | One deterministic computation re-run four times on one host. Reproducibility and driver equivalence, **not** four independent measurements. Withdrawn. |
| "T13 is a negative result… preserved as one" | There is **no** T13 result. §6's "preserved FAIL" covers a run that *reaches* its terminal condition and lands outside band; this is §7's mechanical class. The row called one event both mechanical and a T13 verdict. Withdrawn. |
| "4.275 s/eval, ~11 % conservative" (2026-07-29 Part B1 row and the operator artifact) | A **mid-run partial** over evals 1→151, at 34.7 % of the phase, stated with no as-of qualifier. The phase's eval-to-eval rate is **4.4211 s/eval**; the real margin is ~7 %. |
| "from the stored 146-component state alone" | Describes what the **adjudicator** did, not what the driver did. See §7. |
| "N reached 0.1653 … 2.08 %" as the headline | Observed max is **0.1813 = 2.284 %**; 0.1653 = 2.083 % is the last logged value. Both are now stated. |
| "first output byte 13 s later" *(advisory)* | Retained stamps give **14 s** (freeze 03:49:12Z, `started_at` 03:49:26Z). |

One further precision defect, raised by the refit hunter: **11051 is the last
*logged* evaluation, not the final count.** The `TimeoutError` unwound before
`report["domain_holdout"]` was ever assigned, so the report carries no
domain-phase eval counter; the log samples every 50; and the retained 83 s to the
breach implies ~19 more evaluations ran unlogged. The true count is ~11070. It
changes no conclusion, and stating 11051 as an exact fact was a precision the
bytes do not support.

## 7. A defect in the frozen artifact pair

Contract §6 requires T14 to re-derive `W_nu` **and** `W_tot` from the stored
state. `recompute_from_stored` (`d069:297-349`) re-derives `W_nu` only; `W_tot`
appears nowhere in the function or in `recompute.checks`. The second clause it
does carry, `t_cm_rel = abs(t_cm/(setup.t_start*exp(-n_k)) - 1)` (`d069:332`),
compares the stored field against the very expression that **wrote** it
(`_trajectory_core.py:333`), so it is 0 by construction and tests nothing — yet
it is folded into `worst_relative_deviation`, the statistic the record quotes.

The contract and driver are frozen and are **not** edited here. This is recorded
as a defect of the frozen pair and is an obligation of any reissue.

## 8. The adversarial hunt found no refit — and caught the coordinator anyway

`A-D070-R4-REFIT-HUNTER` (Sonnet, `blind-results`, instructed to refute rather
than confirm) cleared every integrity question:

- `git diff 0fb94f2..ce737a7` over the driver, the shared core, the anchors and
  the contract is **empty**; no commit in that range touches them; the working
  tree is clean; file mtimes predate the freeze commit; the frozen module hash
  matches; the freeze-time BLAS-pin fix is present and correctly ordered.
- The operator artifact's embedded trace is a **verified strict prefix** of the
  banked stdout log, and filesystem mtimes on the banked report and log match the
  report's internal `completed_at` and the log's own elapsed arithmetic to the
  second — independent evidence against post-hoc editing.
- The bitwise-identity claim is **true and broader than the coordinator claimed**:
  r4's base agrees with r3 and r2 across the whole `endpoint`, `blocks`,
  `residual_rows`, spectra, `rejections`, `window`, `max_R_total`, `max_roundoff`
  and `tail_last_node_fraction`, and matches D-056 on every shared field.

It nevertheless returned **`fail`**, on two verified precision defects of its own
finding: F-003, that `11051` is stated at a precision the bytes do not support
(see §6), and F-005, that the operator artifact's own `4.275 s/eval` note does
not reconstruct from the artifact's own embedded trace (~4.451 s/eval over evals
1→1501). Neither amounts to refit, fabrication, post-hoc selection or gate
manipulation, and the agent says so explicitly.

It left one question open (`inconclusive`): the retained evidence cannot settle
whether the 60/30 companion is genuinely non-convergent or merely very slow. The
"stopped, not slow" reading is an extrapolation from an ~11000-evaluation sample.
It is stated here as such. The 1.9 s overshoot is *not* a suspicious near-miss —
`Deadline.check()` runs once at the top of each RHS call, so breach-detection
latency is bounded by one evaluation, and 1.9 s is well under the phase's ~4.4 s.

## 8a. The harness caught a defect the record had already absorbed

The refit hunter's **first** stop was **BLOCKED** by the real Stop hook:

```text
{"decision": "block", "reason": "Result artifact violates RESULT_ENVELOPE:
 status=pass requires every finding verdict to be pass"}
```

Its envelope declared `status: pass` while carrying two `fail` findings and one
`inconclusive`. The receipt stayed **open and unconsumed** — exactly the state the
`scrub_admission_proof.py` docstring calls the dangerous one, which is why the
retained stop event was scrubbed and the anywhere-sweep re-run.

The coordinator had meanwhile written "REFIT-HUNT PASS" into the ledger, taken
from the agent's own summary message rather than from an admitted artifact. **The
hook caught that; the record did not.** It is the same class this whole chain has
been failing on — a surface asserting a status that nothing had verified — and it
recurred one row after the row that documents it.

**A retention defect in the coordinator's own procedure, recorded rather than
papered over.** The retry wrote to the *same* `stop_event_` and `stop_out_`
paths as the first attempt, so the retained bytes of the blocked capture no
longer exist. The block itself is not in dispute — it is why the receipt stayed
open, why the agent was sent back, and why the consume row is stamped 07:56:12Z
against a corrected `completed_at` of 07:55:10Z, roughly 30 minutes after that
receipt was minted versus ~19 for the adjudicator whose first stop succeeded —
and the agent's own `errors[]` entry independently quotes the same reason
string. But the primary artifact is gone, and a coordinator transcription is
weaker evidence than a hook capture. It is retained as
`artifacts/COORDINATOR_RECORD_blocked_first_stop.json`, labelled
`COORDINATOR_RECORD_NOT_A_HOOK_CAPTURE`. **Rule this establishes:** a retried
hook dispatch must write to a new filename. Overwriting is how a fail-closed
refusal disappears from the record while the record still says it happened.

The agent was sent back to correct its own envelope. It was explicitly **not**
told which status to choose, and was explicitly forbidden from deleting or
softening a finding to make `pass` legal. It chose `fail`, and changed exactly
three things: `status`, `completed_at`, and a new `errors[]` entry recording the
blocked attempt. All seven findings — statements, evidence, verdicts, severities
— are byte-identical to the first submission. Both receipts are now consumed with
their result digests pinned in `ADMISSIONS.jsonl`.

## 9. Disposition

**A reissue is not recommended and no execution authority is granted.** The
reason is arithmetic, not caution: an r5 restricted to what is measurable would
gate T1–T12 and T14, discharging items 1, 3 and 6 and leaving items 4 and 5 open,
so **even a clean r5 PASS leaves the gate at FAIL** — at a cost of ~7.6 h to
convert numbers already reproduced in this adjudication into a contract-produced
verdict that moves nothing.

Owner disposition is **STOP / PRESERVE**. Any trajectory slice requires an
explicit owner decision. If one is elected, the reissue may change **exactly
two** items:

1. remove T13 from the gated battery, retuning no band, mask or proxy, and carry
   F-D065-04 items 4 and 5 forward as explicitly undischarged;
2. implement the T14 the contract already specifies (`W_tot` included, the
   tautological clause removed).

Budget stays 64800 s. **Refused by the adjudicator:** a smaller or reshaped
companion, a `y_max`-only companion, any budget raise, any node mask, widening
the T8 band toward 2.0800641747316935e-3, and any tail-proxy substitution.

## 10. Verification

- Both agents admitted through `admit_agent.py` with `--expect-agent-id`,
  SubagentStart dispatched with real event JSON (VS Code LOCAL-ADAPT, hook output
  retained beside the event), Stop validated by the real hook, both receipts
  consumed in `ADMISSIONS.jsonl`.
- `scrub_admission_proof.py --verify-result` run on **both** result artifacts
  before dispatching Stop; retained stop events scrubbed afterwards; anywhere-sweep
  over the run reports 0 live and 0 spent matches against 24 known tokens.
- `check_chronology.py` PASS on both envelopes: freeze `0fb94f2` 03:49:12Z <
  report 03:49:26Z < 21:49:28Z < adjudication 07:27:20Z / 07:27:32Z < containing
  commit. No round-minute stamps.
- `validate_harness.py` ok; `check_ssot_consistency.py` ok; module sha256 stays
  `760a7c04…`.

## 11. Cost

```text
production_source_lines_changed: 0
runtime_behavior_changed: no
physics_behavior_changed: no
frozen_contract_or_driver_edited: no
known_blocker_reduced: no -- and the reduction available from this lane got
  SMALLER: F-D065-04 item 5 moved from partially-addressed to undischarged in
  full, and item 4 fell with it
blocker_movement_ratio: 0.0
reviews: 2 registered agents, mixed-model (Opus adjudicator, Sonnet refit
  hunter), 2/2 fail, 10 fail findings, all acted on; one stop BLOCKED by the
  real hook on an envelope-contract violation and corrected by its own author
tests: unchanged
```
