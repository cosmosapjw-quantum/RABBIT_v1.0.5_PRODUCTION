# BD622 D-067 — exact agent-to-assignment admission binding

Date: 2026-07-29
Lane: D-065 remedy lane 1 (harness), owner-granted 2026-07-29
Runs: `run-20260729-f10-d067-admission-canary`, `run-20260729-f10-d067-admission-decoy`,
`run-20260729-f10-d067-review-r2`

Status: **IMPLEMENTED AND REVIEWED — NO GATE FLIP — ADJUDICATION DEFERRED TO D-070**

## 1. What D-065 required and what this changes

`G-HARNESS-INTEGRITY` failed under D-065 finding F-D065-01 with a four-item
prospective remedy: (1) atomically bind one spawned `agent_id` to one run,
assignment, and digest; (2) make receipt/lease creation failure a hard Start
failure; (3) add same-run substitution and receipt-write-failure negative
fixtures; (4) pass a replacement live overlapping-run canary with unique write
attribution.

The binding could not be built where D-058 tried to build it. The
`SubagentStart` event carries only `agent_id` and `agent_type` — never the spawn
prompt (frozen rows D-031/D-032; verified again here against the captured event
payloads). So no hook can *infer* which assignment an agent was launched for,
and D-058's lease could only seal every assignment in the run.

D-067 moves the binding to the one party that knows it, the parent:

- `admit_agent.py` mints a single-use receipt per assignment, storing
  `sha256(token)` and optionally `expected_agent_id`, and prints the token once
  for the spawn prompt. The mint is appended to the run's `ADMISSIONS.jsonl`
  *before* the receipt is written, so a failed mint leaves no orphan receipt.
- The spawn header is now five fields, ending in `ADMISSION_TOKEN`, and
  `HARNESS_RESULT` carries `admission_proof`.
- `SubagentStop` requires the lease (no `ACTIVE_RUN` fallback), then requires an
  `open` receipt for the claimed assignment whose `assignment_sha256` matches
  current bytes and whose `token_digest` matches `sha256(admission_proof)`, with
  `expected_agent_id` enforced when present. Acceptance takes an `O_EXCL` claim,
  writes the consumed receipt, and appends the attribution row.
- `SubagentStart` no longer swallows a lease-write `OSError`; it emits
  `Hook preflight: FAIL` and records no lease, which `SubagentStop` then blocks.
- `validate_harness.py` reconciles results against the run's append-only
  `ADMISSIONS.jsonl` in both directions: every result in the active run must
  carry a consumed receipt whose `result_sha256` still matches the bytes on
  disk, and every consume row in the ledger pins the bytes it admitted. The
  ledger direction is the load-bearing one, because the receipt-driven check
  alone is cleared by `admit_agent.py --reopen`, which replaces a consumed
  receipt with an open one. A result whose receipt is still open is reported as
  `pending_results` rather than an error, so a live run is not wedged; a missing
  git fails validation instead of being skipped.

  **As sealed, both halves of that sentence were narrower than written** — the
  forward check covered only the active run, and the ledger was last-row-wins so
  a later consume row silently superseded an earlier one. Both were corrected at
  D-070; see the round-4 correction under §2. Read the two claims as scoped to
  what the code did, not to what this paragraph said.
- `scrub_admission_proof.py` replaces raw tokens with their digest in retained
  event JSON, because a captured *blocked* stop would otherwise deposit a still
  valid single-use secret into committed run evidence.

## 2. Threat model — read before claiming more than this delivers

The harness and its subagents run as **the same OS user**. An agent that
deliberately writes outside its declared result path can forge any repo-local
artifact: a receipt, a lease, the ledger. Two independent blind reviewers
reproduced exactly that, and it is not fixed.

What D-067 removes is *confusion*, which is what actually failed:

| Failure mode | Before | After |
|---|---|---|
| Prompt-level substitution (the D-065 negative) | accepted | blocked |
| Same-run assignment mix-up | accepted | blocked |
| Overlapping-main pointer move | D-058 handled | still handled |
| Lease-write failure | fell back to `ACTIVE_RUN` | hard Start failure, Stop blocks |
| `ACTIVE_RUN` deleted/blanked | validation switched off | blocked while the harness dir exists |
| Concurrent double consume | silent overwrite | `O_EXCL`, one attribution row |
| Fabricated or post-hoc edited result | invisible | validator error — but see the D-070 correction below: as sealed here this held for the ACTIVE run only |
| Deliberate forgery by a writing agent | undetected | **still possible**, recorded as a residual |

### D-070 round-4 correction — two rows above were overstated

A fourth adversarial review round (mixed-model, 2 FAIL / 1 PASS) found that two
claims in this report were **false as written at the time they were written**, and
both were reproduced end to end:

- **"Fabricated or post-hoc edited result → validator error"** held for the ACTIVE
  run only. The forward `result → receipt` check sat inside two `if active:`
  blocks, so it covered **1 of 83 run directories**; the ledger loop walked every
  run but only in the `ledger → result` direction and never asked whether a result
  file had a row. A wholly fabricated result — `agent_id` that never existed, no
  assignment, no receipt, no Stop event — dropped into any closed-out run left the
  validator reporting `ok: true` in silence. Compounding it, `/.agent-harness/runs/`
  is gitignored in full, so 60 of 146 result files were untracked and the
  tracked-evidence check could never have seen the file either.
- **"every consume row permanently pins the bytes it admitted"** (§1) was
  last-row-wins: `validate_harness.py` built `ledger_consumed[assignment_id] = row`
  while scanning, discarding earlier rows, and `admit_agent.py --reopen` wrote an
  `"event": "minted"` row indistinguishable from a first mint. Two laundering
  routes followed, neither needing the stated same-OS-user residual — `--reopen`
  after an edit, and deleting the gitignored `.claim` file so the Stop hook took
  the fresh-acceptance path without ever comparing the new bytes.

Both are now fixed: a forward sweep covers every run directory, with pre-admission
history pinned by digest in `.agent-harness/context/LEGACY_RESULTS_MANIFEST.json`
rather than exempted; and a second consume row is admissible only when a declared
`"event": "reopened"` row carrying the superseded digest and a non-empty reason
sits between them. Start now also checks the admission receipt, closing the half
of obligation 2 that was never implemented. A fresh overlapping-run canary (C4)
was run on the post-`96089f0` bytes, but round 5 then rejected its attestation —
see the round-5 correction below. Hook fixtures went 39 → 102. Round 6 then failed again on all four obligations bar the
canary; its fixes and the retraction above are recorded in the round-6 correction.

The lesson is recorded here rather than only in the fix: **this report asserted as
settled two properties the code did not have, after three review rounds had already
passed over it.** That is the D-064 failure mode — the record getting ahead of the
mechanism — reproduced at smaller scale inside the remedy for it.

### D-070 round-5 correction — the C4 canary attestation was defective

Round 5 (mixed-model, both reviewers FAIL) rejected C4 on its own artifact.
`ACTIVE_RUN_ATTESTATION_AT_STOP.json` is named for, and self-describes as,
capture "at the moment of the stop dispatch". It was captured **286 seconds
early** — immediately after the pointer theft, while the canary agent still had
five minutes of work left. Attestation `08:41:06Z`, ledger consume `08:45:52Z`.
Had the pointer been restored in between, C4 would have proved nothing. That is
the same defect class round 4 rejected C1 for, reproduced inside the artifact
built to close it.

**C5 replaces it**, run against frozen bytes at commit `d94270f` rather than
against bytes that could move underneath it:

- `ACTIVE_RUN` read immediately before **and** immediately after the Stop
  dispatch, in one shell command with no intervening work — a **39 ms** bracket,
  recorded in `artifacts/ACTIVE_RUN_BRACKET_AROUND_CONSUME.json`;
- the pointer named the decoy at **both** ends, so it cannot have been restored
  around the consume;
- accepted under the Start-time lease (`run-...-canary-c5`) while the pointer
  named `run-...-canary-c5-decoy`, with `expected_agent_id` enforced,
  `consumed_by_agent_id=d070-canary-c5`, and the result digest pinned in the
  append-only ledger.

Two honest limits are recorded in the artifact rather than glossed. The receipt
writes `consumed_at` with `timespec="seconds"`, so a naive containment test
against nanosecond bounds returns a spurious `False`; the truncated stamp denotes
an interval, and the correct statement is that bracket and consume fall in the
same wall second. Second-resolution stamps cannot prove sub-second ordering, and
the attestation does not claim to — it establishes the *pointer state across the
dispatch*, which is the property obligation 4 actually asks about.

C5's own subject matter was the two round-5 fixes, verified against the real
imported functions rather than reimplementations: the open-receipt carve-out
(six conditions, all AND-ed; non-active runs never excused; 24 h ceiling strict
at the boundary; missing, naive and future `created_at` all refused) and the
legacy-manifest pin (three independent surfaces; appending caught by digest and
count, a same-count entry swap caught by digest alone; deletion caught twice and
cascading to ~75 `not attributed` errors). C5 returned `pass` on both.

### Obligation 2 — what Start can and cannot do, stated rather than argued

Round 5 judged obligation 2 ("make receipt/lease creation failure a hard Start
failure") only partially discharged, and the reasoning is structural rather than
a defect to patch. It is recorded here instead of being closed by assertion.

The **lease** half is fully met: a lease-write `OSError` yields
`Hook preflight: FAIL`, no lease is recorded, and Stop then blocks. Fixture- and
canary-attested.

> **RETRACTED at round 6.** The paragraph below was wrong, and the error was
> mine. It is kept, struck, rather than deleted, because this record's failure
> mode has three times been a claim that outran the code, and quietly rewriting
> the claim would hide the fourth instance.
>
> ~~The **receipt** half cannot be fully met at Start, because a mint failure and
> a legitimately unregistered agent are indistinguishable from inside the hook.
> Start receives only `agent_id` and `agent_type` — never the spawn prompt, and
> never any statement of the parent's intent (frozen rows D-031/D-032).~~
>
> **Why it is false.** `admit_agent.py` writes **ledger first, receipt second**
> (deliberately, so a failed mint leaves no orphan receipt). A `minted` row in
> `.agent-harness/runs/<run>/ADMISSIONS.jsonl` carries `expected_agent_id` —
> verified against a live row, whose keys are `assignment_id`,
> `assignment_sha256`, `at`, `event`, `expected_agent_id`, `reopened`, `run_id`,
> `token_digest`. So when a receipt write fails, the ledger retains a durable,
> machine-readable statement of the parent's intent to admit that exact
> `agent_id`, sitting in the run directory that Start can read. The two states
> are distinguishable, and the ordering that makes them distinguishable was
> introduced by this very slice.
>
> Round 6 reproduced the consequence: `chmod 0o500` on `admissions/<run>/` →
> `admit_agent.py` exits 1, no receipt on disk, orphan `minted` row → Start
> reported `Admission receipt: none found` and **`Hook preflight: PASS`**. That
> is exactly the case obligation 2 names, and Start passed it. No assumption
> about dispatch mode is needed, so the deployment-specific argument recorded
> below does not rescue it either.

Start now distinguishes **three** states rather than two:

- **no ledger row naming this `agent_id`** → "none found", reported but NOT
  fatal. Unregistered helper agents genuinely exist, and this is the case the
  non-fatal rule was always right about.
- **a `minted` row names this `agent_id` but no usable receipt exists** → the
  parent intended to admit this agent and the mint did not complete. **Hard
  FAIL.** This is the obligation-2 case, and it is now enforced rather than
  argued away.
- **receipt present** → unchanged: `open` and correctly bound passes;
  `consumed`, wrong run, malformed, or ambiguous is a hard FAIL.

Two compensating controls carry the rest, and they are where the obligation is
actually satisfied:

- `admit_agent.py` **exits non-zero** on any mint failure, so creation failure is
  a hard failure at the moment of creation, in the process that knows the intent.
- `SubagentStop` refuses to admit any result without a matching open receipt, so
  an agent that was never minted cannot produce an admissible artifact regardless
  of what Start said.

Round 5 also observed that under the recorded LOCAL-ADAPT — VS Code does not
dispatch project Start/Stop hooks, so the parent invokes them explicitly — the
parent always mints, and "none found" would then unambiguously mean "the parent
failed to mint". That is correct for this deployment and does not generalise:
the hook must remain correct where the IDE does dispatch it and unregistered
agents exist. The narrower deployment-specific tightening is available and is
deliberately not taken, because a rule that is only sound under one dispatch mode
is the kind of thing this record has been burned by before.

The table's last row remains the honest limit. It is stated in `admit_agent.py`,
in `.agent-harness/README.md`, and here. Nothing in this slice should be read as a
claim that a hostile agent cannot mis-attribute a result.

## 3. Review record — four rounds, all FAIL, all acted on

Reviews ran through the new flow itself (registered assignments, minted
receipts, real Stop-hook validation).

Round 1, two independent blind reviewers, both `fail`:
`A-D067-REVIEW-CORRECTNESS` (11 findings) and `A-D067-REVIEW-BYPASS`
(12 findings, 6 succeeding attacks). Round 2, `A-D067-REVIEW-R2`, `fail`,
16 findings, judgement "obligation NOT DISCHARGED". Round 3,
`A-D067-REVIEW-R3`, `fail`, judged against the committed post-fix bytes: it
verified all nine round-2 findings genuinely fixed, found three new defects
introduced by that fix pass, and returned "PARTIALLY DISCHARGED" with the
recommendation not to flip the gate on those bytes. Those three are fixed
below.

Findings fixed after round 1: livelock on resumed agents (acceptance consumed
the lease, so a re-stop could never succeed — a regression this slice
introduced); non-atomic single-use; `ACTIVE_RUN`-deletion bypass; missing
`agent_id` binding; gitignored attribution; validator blind to fabricated
results and fail-open on missing git; symlinked result path; four-vs-five field
contract divergence across `AGENTS.md`, `.agent-harness/README.md`, the skill,
the session hook, and `new_assignment.py`.

Findings fixed after round 3 (all three were introduced by the round-2 fix
pass, and all are confusion-class, not the stated residual): `--reopen` cleared
the post-hoc-edit detector by demoting an edited artifact to `pending_results`,
now closed by the reverse ledger reconciliation; the stop hook wrote the receipt
before the ledger, so a failed append left a consumed receipt with no
attribution row and the retry was then accepted, now ledger-first with the claim
released on failure; `repo_root()` degrading to cwd could make the harness look
uninstalled and switch every check off, now also checked next to the hook file;
plus the agent binding is required rather than conventional, and the scrubber
covers non-JSON and escaped payloads and reports what it could not read.

Findings fixed after round 2: the idempotent re-stop guard defaulted a missing
`result_sha256` to the current one, so planting a claim file yielded an accepted
result with **no attribution record at all**; orphan receipt on ledger-append
failure; result artifact read twice (validated bytes ≠ digested bytes); ledger
never validated and un-force-added; a live run wedged by the result↔receipt rule
before its Stop was dispatched; `??` porcelain entries misreported as tracked
modifications; two false claims in the `admit_agent.py` docstring about token
secrecy; and a concurrency test that ran nothing concurrently because
`communicate()` in a loop serialises the children.

Tests: **102 pass** (was 12), including same-run substitution, receipt-write
failure, lease-write failure, planted claim, replayed receipt, changed-bytes
re-stop refusal, genuine four-process concurrency, ledger cross-check, scrubber
idempotence and escaped-payload coverage, `--reopen` failing to clear the tamper
detector, a consume-side ledger failure leaving the receipt open, and the
required agent binding.

Three review rounds have now each found real defects, two of them regressions
introduced by the previous round's fixes. That is the honest headline: this
mechanism took four passes to stabilise, and the D-070 adjudicator should weigh
that rather than the final green suite alone.

## 4. Live canaries

| Canary | Design | Outcome |
|---|---|---|
| C1 overlapping run | Real subagent admitted for `A-D067-CANARY-C1`; a decoy run stole `ACTIVE_RUN` mid-flight; Stop dispatched afterwards | **ACCEPT** under the lease; receipt consumed with `expected_agent_id` enforced and `result_sha256` recorded |
| C2 substitution | Real subagent admitted for `A-D067-CANARY-C2` produced a byte-valid envelope for the sibling `A-D067-CANARY-C2-VICTIM` (zero contract errors against its own assignment); that envelope was fed to Stop with C2's token | **BLOCK**: "admission_proof does not match the receipt". Victim receipt still `open`. C2's own assignment then accepted |
| C3 lease-write failure | `leases/` made read-only, then Start and Stop dispatched | Start: `Run lease: NOT RECORDED (PermissionError)`, `Hook preflight: FAIL`. Stop: **BLOCK** |

Negative control on the validator: editing a consumed result artifact produced
`Result artifact changed after it was admitted`, and restoring it cleared the
error. This is the D-066 incident class, now detected.

The substitution probe artifact is retained at
`runs/run-20260729-f10-d067-admission-canary/artifacts/CONTROLLED_NEGATIVE_A-D067-CANARY-C2-VICTIM.json`
rather than under `results/`, because it was never admitted.

## 5. What is deliberately not claimed

- **No gate flips here.** `G-HARNESS-INTEGRITY` stays FAIL and
  `C-HARNESS-INTEGRITY` stays `IMPLEMENTED`. Whether the D-065 obligation is
  discharged is an adjudication question for D-070, and round 2 answered "not
  discharged" against the pre-fix bytes. The single writer must not grade its
  own remedy — that was the D-064 error.
- `Q-HOOK-01` moves to **REMEDIATED (pending adjudication)**, not RESOLVED.
- One raw token survives in evidence, inside
  `run-20260729-f10-d067-review-r2/results/A-D067-REVIEW-R2.json`, where the
  round-2 reviewer quoted it in prose as proof of the leak. Its receipt is
  already `consumed`, so the token is spent; the artifact is left byte-intact
  because scrubbing it would invalidate its own attribution digest.
- The `A-D067-CANARY-C2-VICTIM` receipt remains permanently `open` by design.
- Single platform, same-model reviewers, and the recorded LOCAL-ADAPT that hooks
  are invoked explicitly rather than dispatched by the IDE.

## 6. Cost

```text
runtime_behavior_changed: yes -- harness only
physics_behavior_changed: no
production_source_lines_changed: 0
tests: 12 -> 35 at the seal `ed7bc49`; -> 39 at `07e3507`, which added four
  regression fixtures for the three defects the round-3 review found. The
  frozen D-067 row records 35 because that was the true count at sealing.
reviews: 4 reviewer runs across 3 rounds (2 blind round-1 axes, 1 adversarial
  round-2, 1 post-fix round-3), all FAIL, all acted on
```
