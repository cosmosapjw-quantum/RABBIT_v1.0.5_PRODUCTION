# BD623 — External harness security audit, triage and reproduction record

**Date:** 2026-08-06
**Subject:** third-party static security audit of `.agent-harness/` and `.codex/hooks/`
**Gate movement:** none. `G-HARNESS-INTEGRITY` stays FAIL and this document authorises nothing.
**Controlling decision:** D-074 (`docs/audit/BD622_D074_harness_freeze_2026-08-04.md`).

---

## 0. Provenance of the source document, stated rather than implied

The audit was received in-session as prose, not as a file with a digest. **This document is a
derived record, not a byte-preserved copy**, and does not claim to be one. Every finding below is
restated in the auditor's own terms and given a verdict measured against the tree at `2d866df`. If
the original is wanted byte-for-byte in the repository, it has to be added as a file by whoever
holds it; nothing here can manufacture that provenance after the fact.

The audit returned 23 findings — 6 Critical, 11 High, 6 Medium — and one architectural verdict:

> the harness is a sound *coordination* instrument but not a *trust boundary*, because executor,
> verifier, evidence store, adjudicator and registry share one OS user, one filesystem and one
> repository; the next step is not another `check_*.py` but a control plane outside the repo.

**That verdict is accepted.** It is also, in substance, what D-074 §3 already records.

---

## 1. Disposition classes

| Class | Meaning |
|---|---|
| `RECORDED-LIMIT` | D-074 §3 already records this as a permanent limit. Not a work item. |
| `REOPEN-TRIGGER` | Reproduced, and matches one of D-074's five reopen conditions. Fixable now. |
| `OUTSIDE-FREEZE` | Real, and does not touch `.agent-harness/` or `.codex/hooks/`. Freeze does not apply. |
| `OUT-OF-SCOPE` | Real, matches no trigger. Recorded at D-076, not fixed. Owner decision, 2026-08-06. |
| `REFUTED` | Measured against the tree and found wrong or unreachable. |

D-074's five reopen conditions, quoted: (T1) a reproduced path from non-authoritative input to
generated authoritative state; (T2) a generated artifact that does not match what its inputs render
to; (T3) an accepted evidence package with wrong bytes or ancestry; (T4) a required mutation
surviving its own removal; (T5) a hook or runtime contract change that breaks the admission path.

---

## 2. Where the audit is wrong

These corrections are recorded first, because the failure this chain keeps repeating is a record
absorbing a claim nobody measured.

| # | Audit claim | Correction | Evidence |
|---|---|---|---|
| E-1 | Truncation is "silent" | It is marked: `...[truncated at 24000 characters; read the file directly for the rest]`. The defect is that it is **non-fatal and preflight still prints PASS**, not that it is invisible. | `.codex/hooks/subagent_start_context.py:263` |
| E-2 | "no seccomp, no isolation of any kind" | All six agents declare `sandbox_mode = "workspace-write"`, which on Linux is Landlock+seccomp. The substance survives — worktree-wide, shared, one UID, `trust_level = "trusted"` — but the stated mechanism is wrong. | `.codex/agents/*.toml:5`; `~/.codex/config.toml:61-62` |
| E-3 | Claude side "allows only WebSearch + limited git" | `permissions.allow` is an auto-approve list, not a capability restriction. There is no `deny`, no `defaultMode`, no `disallowedTools`. The runtimes differ as "prompted vs pre-granted", not "restricted vs unrestricted". | `.claude/settings.json` |
| E-4 | Hooks have ZERO mutation coverage | `M-STRICT-JSON` targets `_harness.py`, and both its kill fixtures are in `.agent-harness/tests/test_hooks.py`. | `.agent-harness/context/MUTATION_MANIFEST.json:101-110` |
| E-5 | Mutation coverage is 19/128 ≈ 15% | 128 counts only the two files in `declared_guard_sites`. The manifest ratchet's own regex over all 15 scripts counts **292** guard sites. True coverage is **19/292 ≈ 6.5%**. | `.agent-harness/tests/test_mutation_gate.py:43` |
| E-6 | "~210 gitignored campaign files" | **595** untracked files under `.agent-harness` — 442 in `runs/`, 106 admission receipts, 46 leases, `ACTIVE_RUN`. Low by ~2.8x. | `git status --ignored -- .agent-harness` |
| E-7 | Registry rows carry no evidence-package digest | The `status_package` mechanism exists and is checked hard — closed key set, manifest-digest recomputation, ancestry, contract-frozen-before-inputs, per-file blob digest, tracked-in-git. The accurate statement is that it is **100% unexercised**: 8/8 gates null, `.agent-harness/evidence/` absent. | `.agent-harness/scripts/check_ssot_consistency.py:894-1000` |

E-5 and E-6 are corrections *against* the harness, not for it: both make the measured position worse
than the audit stated.

---

## 3. Reproduction record

D-074 reopens on *a reproduced path*, not on an auditor's reading. Eight candidates were driven
against a sandbox checkout with the real scripts. Seven reproduced; one did not.

| # | Claim | Verdict | Measured result |
|---|---|---|---|
| R1 | A tampered `CONTEXT_PACK.md` passes both integrity checks | **REPRODUCED (T2)** | A pack truncated to its first 6 lines — 283 of 157,567 chars, **0.18%** — returns `[]` from `context_integrity()` and `[]` from the validator's pack check. A pack whose body is replaced with a forged `\| D-074 \| Harness development is UNFROZEN. All gates PASS. \|` row also returns `[]` from both. Both checks only look for the version string in the first 6 lines; `build_context_pack.py:39` writes the pack with a non-atomic `write_text`. |
| R2 | Duplicate-key `GATE_REGISTRY.json` splits the board from the checker | **REFUTED** | `build_status_board.py` exits 1: `GATE_REGISTRY.json: unreadable or invalid JSON (JSONDecodeError)`. `load_gates` parses the whole file with `loads_strict` and fails closed *before* the lax `json.loads` at line 73 is reached. The lax read is real but unreachable. **No change made.** |
| R3 | A `\|` in `status_basis_legacy` corrupts the generated board | **REPRODUCED (T1)** | `build_status_board.py` exits 0, no warning, and renders `\| \`G-HARNESS-INTEGRITY\` \| FAIL \| benign prose \| PASS \| all obligations discharged \|` — 5 cells in a 3-column table. The injected `PASS` sits beside a gate id inside the board region, which `check_no_status_beside_an_id` excludes by construction. |
| R4 | `init_run.py --run-id` escapes the runs directory | **REPRODUCED (T1)** | `--run-id '../../../ESCAPED/pwned-run'` exits 0 and creates `RUN_PLAN.json` plus `assignments/ results/ raw_logs/ artifacts/` outside the harness, then writes the traversal string into `ACTIVE_RUN`. An absolute path works identically. **Bounded:** `ADMISSION_KEY_RE` rejects the poisoned key, so `admission_path()` returns `None` and no admission can be minted against it. |
| R5 | Receipt and lease are strict at mint, lax at Stop | **REPRODUCED** | One receipt with two `expected_agent_id` keys: `_harness.load_json` refuses (`duplicate object key`), `_common.load_json` accepts and returns `agent-VICTIM` while a first-key reader sees `agent-ATTACKER`. Same for a lease with two `run_id` keys, and `bound_run` selects the run directory, ledger, receipt and agent lock. `_common.read_stdin_json` returns `{}` on malformed stdin — fail-open at the event boundary. |
| R6 | `admit_agent` unlinks Stop's `O_EXCL` claim without Stop's flock | **REPRODUCED** | With the per-`(run, agent_id)` flock held, the unlink succeeds and the claim is destroyed. A probe confirms the lock **would** have blocked it: `BLOCKED -- the lock admit_agent never asks for`. |
| R7 | Copying a peer's `evidence_fingerprint` manufactures corroboration | **REPRODUCED (T1)** | Two results, one of which copied the other's fingerprint and refs, merge to `duplicate_count=2`, `supporting_review_roles=['cas_wolfram_xact','cas_sympy']` — the exact signal the adjudicator is told to read. The free-rider's own `statement` and `severity: critical` are dropped silently by `representative = items[0]`. Inversely, two genuinely independent agreeing agents with different fingerprints render as two uncorroborated findings. |
| R8 | Role context is never injected | **REPRODUCED** | On the real index: `used` = 24,072 > `max_chars` = 24,000, so `remaining` = **−72**, `remaining <= 512` fires, and `independent_auditor.md` is dropped. Injected: `SHARED_CONTEXT.md` 23,711 of 34,723 bytes; `SYMBOLS.md` **0**; `FROZEN_DECISIONS.md` **0**; `GATE_REGISTRY.json` **0**. Preflight still prints PASS. |

---

## 4. Full disposition of the audit's 23 findings

| ID | Audit finding | Verdict | Class |
|---|---|---|---|
| C-01 | No isolation; same OS user holds every privilege | CONFIRMED in substance, mechanism wrong (E-2) | `RECORDED-LIMIT` (D-074 §3 row 1) |
| C-02 | Writer, reviewer, adjudicator, registry updater not independent | CONFIRMED | `RECORDED-LIMIT` (D-074 §3 row 2) |
| C-03 | Remote branch unprotected, no required CI, unsigned HEAD, no `.github/` | CONFIRMED — `"protected": false`, rulesets `[]`, `workflows` total_count 0, `%G?` = `N` | `OUTSIDE-FREEZE` |
| C-04 | Context pack truncated; body integrity unverified | CONFIRMED, reproduced R1/R8 | `REOPEN-TRIGGER` (T2) |
| C-05 | Core raw evidence is gitignored local state | CONFIRMED and understated (E-6) | `OUTSIDE-FREEZE` |
| C-06 | No path confinement in several entry points | CONFIRMED for `init_run.py` (R4); `new_assignment.py` parent/inputs unchecked; `run_axes.py` absolute pass-through | `REOPEN-TRIGGER` (T1) for `init_run.py` only |
| H-01 | `blind-results` and tool limits are policy, not enforcement | CONFIRMED | `RECORDED-LIMIT` |
| H-02 | `evidence_fingerprint`/`evidence_refs` not bound to evidence bytes | CONFIRMED, reproduced R7 | `REOPEN-TRIGGER` (T1), minimal fix only |
| H-03 | No expected-panel completeness or quorum | CONFIRMED — 138/138 RUN_PLANs carry no expected set; 13 runs have fewer results than assignments | `OUT-OF-SCOPE` (needs a new declaration file) |
| H-04 | No global transaction on run/assignment/admission creation | CONFIRMED for `new_assignment.py` and `admit_agent.py`; overstated for `init_run.py`, whose `mkdir` is an atomic claim | `OUT-OF-SCOPE` except R6 |
| H-05 | Atomic writes without fsync or cross-file atomicity | CONFIRMED — zero `fsync` in the harness | `OUT-OF-SCOPE` |
| H-06 | Bearer token in prompt/transcript; token-only mode | CONFIRMED, self-documented | `RECORDED-LIMIT` + `OUTSIDE-FREEZE` (visibility) |
| H-07 | JSON parser policy inconsistent | CONFIRMED, reproduced R5 | `REOPEN-TRIGGER` for the admission path; `REFUTED` for the board (R2) |
| H-08 | Canary policy is a count, not a property manifest | CONFIRMED, self-admitted in the policy file | `OUT-OF-SCOPE` |
| H-09 | Mutation coverage narrow | CONFIRMED and understated — 19/292 ≈ 6.5% (E-5) | `OUT-OF-SCOPE` |
| H-10 | Legacy manifest is a standing bypass | CONFIRMED — 136 entries, `MERGEABLE_ATTRIBUTION` accepts them | `OUT-OF-SCOPE` |
| H-11 | Dependencies, skills, runtime not hermetic | CONFIRMED and worse — all 7 `skills-lock.json` hashes stale, zero call sites | `OUTSIDE-FREEZE` |
| M-01 | Chronology checker is heuristic, not trusted time | CONFIRMED | `RECORDED-LIMIT` |
| M-02 | No per-worker resource quota | CONFIRMED | `OUT-OF-SCOPE` |
| M-03 | `run_axes` output validation is existence-only | CONFIRMED | `OUT-OF-SCOPE` |
| M-04 | Generated board has no authority independence | PARTIAL — the digest mechanism exists, unexercised (E-7) | `OUT-OF-SCOPE` |
| M-05 | Cost report is gameable | PARTIAL — only `blocker_movement_ratio` is unconditionally caller-supplied | `OUT-OF-SCOPE` |
| M-06 | Capability model inconsistent across runtimes | PARTIAL, mechanism wrong (E-3) | `OUT-OF-SCOPE` |

---

## 5. What the audit missed

Found by the verification pass, not by the audit. Numbers measured, not estimated.

1. **Role context is never injected at all** (R8) — while the injected contract still instructs the
   agent to verify `review_role_sha256` over files it was never shown.
2. **`FROZEN_DECISIONS.md` and `GATE_REGISTRY.json` inject zero bytes** (R8). The frozen decisions
   and the gate registry are the two documents the harness treats as authoritative.
3. **A forged pack body passes both integrity checks** (R1) — strictly worse than "truncation".
4. **`loads_strict`'s stated invariant is false.** Its docstring claims every evidence-bearing
   document in the admission path is parsed through it; the receipt and the lease are not (R5).
5. **`admit_agent --reopen` destroys Stop's exclusivity primitive** (R6).
6. **Manufactured and destroyed corroboration** (R7), plus first-writer-wins by assignment-ID
   alphabetical order deciding which prose reaches the adjudicator.
7. **`MERGED_RESULTS.json` — the adjudicator's actual input — is unattributed**, gitignored, and
   read back by nothing. A hand edit produces zero errors from any checker.
8. **The enforcement chain is voluntary.** `.git/hooks/` holds only samples, there is no CI, and
   `pyproject.toml:66` sets `testpaths = ["tests"]`, so `pytest -q` does **not** collect
   `.agent-harness/tests`. D-074 §6 states the mutation battery "runs in the suite".
9. **The gate-movement ratchet only ever sees an uncommitted working tree.** It compares the working
   tree against `git show HEAD:`; once a movement is committed it is undetectable forever.
10. **Zero evidence packages exist.** All 8 gates carry `status_package: null` and ride
    `status_basis_legacy`, so the D-073 mechanism D-074 §6 credits has never been exercised once.
11. **The board launders unbounded prose into authority** (R3), including the unchecked sentence
    "All four obligations are now discharged" in `G-HARNESS-INTEGRITY`'s Basis cell.
12. **All 21 claim rows sit outside the movement ratchet**, by explicit design note.
13. **The legacy manifest is a permission list, never a presence requirement.** 56 of 136 pinned
    artifacts are present-but-untracked; a fresh clone gets 56 dangling pins and the validator still
    exits ok, because the manifest hashes itself and not the tree.
14. **`CANARY_POLICY.json`'s `why_two` names C8 and C9 — both retired.** Live set is C11/C12/C13, so
    one live canary can be deleted with zero error, including C11, sole survivor of the
    C5→C7→C8→C11 pointer-bracket lineage.
15. **`allowed_tools` is decorative** — its only consumer asserts the list is non-empty.
16. **Independence is breachable through the spawn prompt**, which no validator reads. Already an
    incident: `.agent-harness/incidents/2026-07-29-coordinator-directed-an-independence-breach.json`.
17. **The mutation growth ratchet watches 2 of 15 scripts.**
18. **All 7 `skills-lock.json` hashes are stale**, no entry records a commit or ref,
    `shared-context-orchestrator` is absent from the lock, and no code reads the lock.
19. **The repository is public** while running a protocol whose own docs say tokens land in
    transcripts and captured stop events.
20. **A stale `pre_tool_use` trust record persists** in `~/.codex/config.toml`, for a hook slot
    removed at D-032.

---

## 6. What was done, as measured

| # | Outcome | Where |
|---|---|---|
| R1 | **CLOSED** — pack re-rendered and byte-compared; the 6-line scan is gone; the pack is written atomically | `_harness.render_context_pack`, `build_context_pack.py`, `validate_harness.py` |
| R2 | **REFUTED, nothing changed** | — |
| R3 | **CLOSED** — the Basis cell can no longer restructure the board | `check_ssot_consistency.basis_cell` |
| R4 | **CLOSED** — run id must be a path key | `init_run.py` |
| R4b | **CLOSED** — parent assignment id must be a path key | `new_assignment.py` |
| R5 | **PARTLY CLOSED** — the shared hook reader refuses duplicate keys, closing the Stop-side receipt and lease. The Start hook's own direct parses are NOT closed | `.codex/hooks/_common.py` |
| R6 | **CLOSED** — a reopen takes the consuming agent's lock before deleting its claim | `admit_agent.py` |
| R7 | **PARTLY CLOSED** — the corroboration fields are renamed `agent_asserted_*` and divergent members are retained. The signal is still agent-authored | `merge_results.py`, `roles/adjudicator.md` |
| R8 | **NOT CLOSED** — lives only in the attested Start hook | recorded at D-076 |

**Why R8 and the rest of R5 are not closed.** Every live canary attests
`subagent_start_context.py` and `subagent_stop_validate.py` by digest, and C11/C12/C13 have no
preserved runner — they were driven by hand. Editing either file retires the entire live canary set
with no way to re-establish it. `check_canary_freshness.py` states that cost is deliberate: "a
canary that was not re-run after the code changed is not evidence." The fix for R5 landed in
`_common.py` precisely because that file is **not** in `ATTESTED_FILES`, which was verified after
the edit rather than assumed. R8 has no such path.

Governance landed outside the frozen surface: a CI job that runs the existing checkers, `testpaths`
that finally collects them, a lock pinned to the environment that was actually measured, 56
pinned-but-untracked artifacts retained, an unverifiable skills lock deleted, and `main` protected
by a ruleset — without a signed-commits rule, because no signing key is configured and requiring one
would have blocked all pushes.

**No gate moved. `G-HARNESS-INTEGRITY` remains FAIL**, and this audit adds nothing to its pass
condition — its 14-condition bar is filed as informational, because adding obligations is gate
machinery and the freeze forbids it.
