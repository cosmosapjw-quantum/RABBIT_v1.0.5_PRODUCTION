# D-074 — Freeze harness development, record the permanent limits, and stop

**Date:** 2026-08-04
**Decision:** `FREEZE_HARNESS_FIX_CRITICAL_ONLY`.
**Gate movement:** none. The board is **6 PASS / 2 FAIL** and this decision authorises nothing.

---

## 1. What is frozen, and what is not

Harness *development* stops. From this decision, changes to `.agent-harness/` and `.codex/hooks/` are
limited to:

- a reproduced path from non-authoritative input to generated authoritative state;
- a generated artifact that does not match what its inputs render to;
- an accepted evidence package with wrong bytes or ancestry;
- a required mutation surviving its own removal;
- a hook or runtime contract change that breaks the admission path.

Anything else — a new guard, a new checker, a new declaration file, another enumeration of what a
status might look like — is **out of scope and stays out** unless one of the above forces it.

Not frozen: the registries themselves, which is where status now lives, and which move only through
an accepted evidence package.

## 2. Why here

The third-party design re-audit recommended freezing after one bounded, net-deleting migration. That
migration is done (D-073). The measured case for stopping is the one the re-audit made and this
session confirmed twice more:

**Roughly half the defects found late in this chain were introduced by the immediately preceding fix.**
This session alone: two of the three findings closed at D-072 were defects in the meta-fixes landed
hours earlier at `b62343e`; the provenance guard then found the same wrong-commit citation in two
further canaries that the same diagnosis had failed to re-check; and three commits landed with the
aggregate validator red.

Continuing to add machinery to a system whose dominant defect source is the machinery is not
diligence.

## 3. The permanent limits, recorded rather than engineered against

These are not open work items. They are properties of the arrangement, and no amount of
repository-local code removes them.

| Limit | Why it cannot be closed here |
|---|---|
| **Shared OS user** | Harness and subagents run as one identity, so any repo-local artifact is forgeable. Canary C13 *depends* on this to construct its input. Raising the cost is the claim; closing it needs a second UID. |
| **Operator and decision independence is ABSENT** | Every reviewer in this chain was spawned, prompted, admitted and recorded by the writer under review. Multiple local agents are computational diversity, not a governance authority. No receipt makes a judgment independent. |
| **One host, one filesystem** | Every concurrency result is one trial on one local filesystem. NFS and overlay lock semantics are untested. |
| **Canary evidence is bounded to its bytes and host** | A canary attests the implementation it ran against, nothing wider. |
| **False semantic prose is not detected** | A sentence carrying no registry id and no status token — "all obligations have been discharged" — is not caught, here or anywhere. No parser closes it; the board is the authority and prose is explicitly not. Accepted at D-073. |
| **Cited evidence is not proved to support its claim** | F-R13-04 remains open. A package binds bytes and ancestry; it does not prove the conclusion drawn from them. `implies_status` is asserted by whoever wrote the package. |

## 4. Obligation state at the freeze

Adjudicated by a round-13 registered panel, not by the writer:

| D-065 obligation | State |
|---|---|
| 1. Atomically bind one agent to one run/assignment/digest | DISCHARGED, within the cooperative/local-filesystem threat model |
| 2. Hard Start failure on lease/receipt creation failure | **WORDING_FALSIFIED** — `subagent_start_context.py` has no non-zero exit path for any input. The EFFECT is fail-closed (Stop refuses); the literal wording is unmet. |
| 3. Same-run substitution and receipt-write-failure fixtures | DISCHARGED |
| 4. Replacement overlapping-run canary with unique attribution | DISCHARGED |

`G-HARNESS-INTEGRITY` stays **FAIL**. Three discharged obligations are not a gate: obligation 2 is
unmet as worded, and the governance limits in §3 are not addressable by more machinery.

**Two owner decisions remain open and are not the writer's to make:**

1. Revise obligation 2 to match the measured behaviour, or keep it as a permanent named failure.
2. `G-HARNESS-INTEGRITY.pass_condition` requires an "exact four-line prompt" while `AGENTS.md`
   mandates five fields including `ADMISSION_TOKEN`. Correcting the text that defines PASS is a spec
   change.

## 5. Cost

Measured by `cost_report.py` over `29de3b3..HEAD`:

```
added_lines: 2126
deleted_lines: 1636
net_lines: +490
files_touched: 118
runtime_behavior_changed: harness-only
physics_behavior_changed: no
known_blocker_reduced: no
blocker_movement_ratio: 0
cost_effectiveness_verdict: DRIFT
```

The verdict is **DRIFT** and is recorded as such. Two things are worth stating alongside it rather
than instead of it:

- **The migration itself was net-deleting where it replaced**: the parser and its fixtures are −677
  lines, and `check_ssot_consistency.py` fell from 2,278 to 1,730. The session is net-positive
  because the evidence packages are new machinery.
- Over the whole chain `b9b00cd..HEAD` the figure is **+45,820 lines across 600 files with no gate
  movement**. That number is the argument for this decision, not against it.

## 6. What was actually achieved, stated without inflation

- The trajectory lane is **sealed on a re-derived measurement** with explicit reopen conditions
  (D-071), and the robustness envelope it was going to run is withdrawn.
- Status is **generated from the registries** and cannot be written by hand. Seven consecutive
  defeats of a prose parser end because the parser is gone, not because it was made cleverer.
- A gate's status can only **move** with an accepted evidence package whose contract is provably
  frozen before its inputs — mechanising an anti-refitting rule that had rested on prose since D-069.
- The mutation battery **lives in the repository and runs in the suite**, so "N of N verified" is a
  claim anyone can re-derive rather than one whose evidence was deleted with a scratch directory.

None of this moved a gate, and the record says so everywhere it appears.

## 7. Then stop

`NEXT_SESSION_PROMPT` is rewritten to reflect the freeze. No further harness work is authorised. The
board is 6 PASS / 2 FAIL, `G-F10-INDEPENDENT-FLRW` is closed on current measurement, and
`G-HARNESS-INTEGRITY` remains FAIL with its residuals stated above.
