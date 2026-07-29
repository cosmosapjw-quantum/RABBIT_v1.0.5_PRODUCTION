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
- `validate_harness.py` requires every result in the active run to carry a
  consumed receipt whose `result_sha256` still matches the bytes on disk, and
  cross-checks each consumed receipt against the committed ledger. A missing git
  now fails validation instead of being skipped.
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
| Fabricated or post-hoc edited result | invisible | validator error |
| Deliberate forgery by a writing agent | undetected | **still possible**, recorded as a residual |

The last row is the honest limit. It is stated in `admit_agent.py`, in
`.agent-harness/README.md`, and here. Nothing in this slice should be read as a
claim that a hostile agent cannot mis-attribute a result.

## 3. Review record — two rounds, both FAIL, both acted on

Reviews ran through the new flow itself (registered assignments, minted
receipts, real Stop-hook validation).

Round 1, two independent blind reviewers, both `fail`:
`A-D067-REVIEW-CORRECTNESS` (11 findings) and `A-D067-REVIEW-BYPASS`
(12 findings, 6 succeeding attacks). Round 2, `A-D067-REVIEW-R2`, `fail`,
16 findings, judgement "obligation NOT DISCHARGED".

Findings fixed after round 1: livelock on resumed agents (acceptance consumed
the lease, so a re-stop could never succeed — a regression this slice
introduced); non-atomic single-use; `ACTIVE_RUN`-deletion bypass; missing
`agent_id` binding; gitignored attribution; validator blind to fabricated
results and fail-open on missing git; symlinked result path; four-vs-five field
contract divergence across `AGENTS.md`, `.agent-harness/README.md`, the skill,
the session hook, and `new_assignment.py`.

Findings fixed after round 2: the idempotent re-stop guard defaulted a missing
`result_sha256` to the current one, so planting a claim file yielded an accepted
result with **no attribution record at all**; orphan receipt on ledger-append
failure; result artifact read twice (validated bytes ≠ digested bytes); ledger
never validated and un-force-added; a live run wedged by the result↔receipt rule
before its Stop was dispatched; `??` porcelain entries misreported as tracked
modifications; two false claims in the `admit_agent.py` docstring about token
secrecy; and a concurrency test that ran nothing concurrently because
`communicate()` in a loop serialises the children.

Tests: **35 pass** (was 12), including same-run substitution, receipt-write
failure, lease-write failure, planted claim, replayed receipt, changed-bytes
re-stop refusal, genuine four-process concurrency, ledger cross-check, and
scrubber idempotence.

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
tests: 12 -> 35
reviews: 3 (2 blind round-1 axes + 1 adversarial round-2), all FAIL, all acted on
```
