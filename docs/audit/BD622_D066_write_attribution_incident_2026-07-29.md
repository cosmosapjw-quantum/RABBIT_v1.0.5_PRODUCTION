# BD622 D-066 — write-attribution incident on a tracked D-057 evidence artifact

Date: 2026-07-29
Affected path:
`.agent-harness/runs/run-20260728-f10-d057-adversarial-adjudication-r5/results/A-D057R5-CROSS-REJECT.json`

Status: **RECORDED — AUTHENTICITY UNRESOLVED — BOTH BYTE-SETS PRESERVED**

## 1. What happened

A tracked historical evidence artifact from the D-057 adversarial adjudication was
overwritten in the working tree during the D-065 audit window. The write was not
performed by the D-065 main writer and no registered D-065 assignment authorised
it. D-065 observed the drift, declined to revert it, and explicitly excluded the
file from its own evidence and line accounting (D-065 §5, "Write-attribution
incident").

D-066 does not adjudicate which byte-set is authentic. It preserves both, restores
the committed bytes so the tracked tree is self-consistent, and records the open
question.

## 2. The two byte-sets

| Byte-set | SHA-256 | `status` | `completed_at` | Where it now lives |
|---|---|---|---|---|
| Committed (authority for the tracked tree) | `12d17edef9382b76959e1cf0d7f189ffae932d82d7b5c436a6f37e082ed1f486` | `fail` | `2026-07-28T05:40:32Z` | the tracked path, restored |
| Working-tree overwrite (preserved) | `1ae3dd1534f0b5646c7cdd1bdd00a72e0ee4dfbe9133f32a4d26dbd2c679d62b` | `error` | `2026-07-28T05:42:47Z` | `.agent-harness/incidents/2026-07-29-A-D057R5-CROSS-REJECT.withdrawal.json` |

Both declare the same `started_at` (`2026-07-28T05:39:25Z`) and the same
`spawn_contract` block.

The committed byte-set carries six findings: `F-D057R5-COV-PRESERVE-001` (pass),
`F-D057R5-COV-REJECT-002` (fail/critical), an endpoint-preserve finding,
`F-D057R5-ENDPOINT-REJECT-004` (fail/critical), `F-D057R5-HARNESS-PRESERVE-005`
(pass), and `F-D057R5-HARNESS-REJECT-006` (fail/critical).

The overwrite replaces all six with a single fail-closed withdrawal,
`F-D057R5-EVIDENCE-DRIFT-001` (fail/critical, confidence 1.0), on the ground that
the authorised D-057 report changed mid-review — it records
`initial_observed_sha256 f5c16c421a7434689e9f14229bc51c69fbe65c663d20c7eca55d0bb742abadba`
and
`later_observed_sha256 d3ecd54d2a2c7cb6b012f43e64bdf9b52fc5ca32101a6858bbfbcb6dbeb4574a`
for `docs/audit/BD622_D057_blocker_resolution_adversarial_audit_2026-07-28.md`,
and states: *"The substantive draft combined reasoning from the earlier bytes with
a later current artifact and is therefore withdrawn."* Its `errors` array is
non-empty. The committed final bytes of that D-057 report are
`0061a0f5b1d0e58397a64158fca187c8fd65f53fe0db1ca82667e339c26e20eb`, i.e. a third
value — the report was still being authored while the adjudicator read it.

## 3. Timeline

All times UTC.

| Time | Event |
|---|---|
| 2026-07-28T05:39:25 | `A-D057R5-CROSS-REJECT` `started_at` (both byte-sets) |
| 2026-07-28T05:40:32 | committed byte-set `completed_at` (`status: fail`) |
| 2026-07-28T05:42:47 | overwrite byte-set `completed_at` (`status: error`) |
| 2026-07-28T06:56:16 | commit `2875883` records the **committed** byte-set |
| 2026-07-29T00:15:38 / 00:15:52 | D-065 runs `…-remediation-adversarial-audit` / `-r2` initialized |
| 2026-07-29T00:17:27 | lease `019fab3b-b5a5-…` written (`default`, r2 run, 4 sealed assignments) |
| 2026-07-29T00:17:33 | lease `019fab3b-d04e-…` written (`default`, r2 run, 4 sealed assignments) |
| **2026-07-29T00:18:33** | **the tracked artifact is overwritten** (file mtime) |
| 2026-07-29T00:21:30 | first D-065 result written (`A-D065-PROVENANCE-HARNESS`) |

The overwrite lands 60 s after the second subagent Start and ~3 min before any
D-065 agent produced its own result. Eight leases from the D-065 window remain
unconsumed, which is expected under the recorded LOCAL-ADAPT (the IDE does not
dispatch project Stop hooks automatically), so lease residue does **not** identify
a writer. The available parent-side evidence is therefore circumstantial: it
establishes that a write occurred inside the D-065 window from outside the D-065
main writer, and nothing more.

## 4. Why this is not adjudicated

Two mutually exclusive readings both fit the evidence:

1. The overwrite is the authentic terminal write of the D-057 R5 adjudicator —
   later `completed_at`, coherent withdrawal rationale, and a real observed drift
   in the artifact it was authorised to review — that reached the working tree
   after commit `2875883` had already captured the earlier draft.
2. The overwrite is a stray write by a D-065-window process re-deriving that
   assignment, in which case its 2026-07-28 timestamps are reconstructed rather
   than measured.

Nothing in the harness distinguishes them, because the harness had no write
attribution at the time — which is precisely finding F-D065-01. Attributing the
file either way would be an unfounded claim, so D-066 records it as open question
`Q-ATTRIB-01` and leaves it open.

## 5. Impact assessment

If reading 1 is correct, all six D-057 R5 cross-reject findings are withdrawn and
the D-057 report's cross-reject stage carries no admitted verdict. The practical
impact on the current gate board is **nil**: D-065 re-derived the same two FAIL
gates independently, through different assignments (`A-D065-HARNESS`,
`A-D065-TRAJECTORY`, `A-D065-SOFTWARE-COST`) and a separate adjudication
(`A-D065-ADJUDICATION-R2`), and the D-058--D-064 remediation work that D-057
prompted is preserved intact with its own PASS/FAIL artifacts. No gate status,
claim status, or evidence key depends on the disputed file alone.

This paragraph is an impact statement, not a resolution. `Q-ATTRIB-01` stays open.

## 6. Actions taken

1. The overwrite bytes were copied, mtime preserved, to
   `.agent-harness/incidents/2026-07-29-A-D057R5-CROSS-REJECT.withdrawal.json`
   and are tracked from this commit onward.
2. The tracked path was restored to its committed bytes with `git checkout --`.
3. `Q-ATTRIB-01` was registered in `.agent-harness/context/SHARED_CONTEXT.md`.
4. D-067 adds the mechanism that makes a recurrence attributable: a
   parent-authenticated admission consumed at Stop with the writing `agent_id` and
   the result SHA-256 recorded, plus a `validate_harness.py` check that no tracked
   file under `.agent-harness/runs/` is modified.

## 7. Verification

```text
sha256sum .agent-harness/runs/run-20260728-f10-d057-adversarial-adjudication-r5/results/A-D057R5-CROSS-REJECT.json
# 12d17edef9382b76959e1cf0d7f189ffae932d82d7b5c436a6f37e082ed1f486

sha256sum .agent-harness/incidents/2026-07-29-A-D057R5-CROSS-REJECT.withdrawal.json
# 1ae3dd1534f0b5646c7cdd1bdd00a72e0ee4dfbe9133f32a4d26dbd2c679d62b

git show HEAD:.agent-harness/runs/run-20260728-f10-d057-adversarial-adjudication-r5/results/A-D057R5-CROSS-REJECT.json | sha256sum
# 12d17edef9382b76959e1cf0d7f189ffae932d82d7b5c436a6f37e082ed1f486
```
