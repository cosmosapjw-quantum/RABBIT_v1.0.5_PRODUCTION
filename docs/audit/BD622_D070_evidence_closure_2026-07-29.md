# BD622 D-070 Part A — evidence closure and SSOT integrity machinery

Date: 2026-07-29
Lane: owner-granted 2026-07-29 blocker-clearance chain
Status: **IMPLEMENTED — NO GATE MOVEMENT — ADJUDICATION IS PART B**

Part A closes bookkeeping gaps and builds the machinery that makes them
detectable. Part B — adjudication of D-067 and of the D-069 r4 report, and any
gate reconsideration — is a separate row and has not happened. The board is
unchanged at **6 PASS / 2 FAIL**.

## 1. What was missing

**D-068 had no record at all.** It shipped inside `ed7bc49` with no audit report,
no `INDEX.md` anchor, and no row in `FROZEN_DECISIONS.md`, `DECISION_LOG.md`, or
`VALIDATION_LEDGER.md`. It was found by survey two decisions later. That is the
F-D065-05 contradiction class recurring *inside the remedy chain for F-D065-05*.
Written up retrospectively in `BD622_D068_cost_discipline_2026-07-29.md`, which
says plainly that a retrospective record is itself a provenance weakness.

**The hook-fixture count read 39, 35, 35 and `12 -> 35` across four surfaces.**
Counting `def test_` per commit resolves it: **35 at the D-067 seal `ed7bc49`,
39 at `07e3507`**, which added four regression fixtures for the round-3 findings.
So the frozen row and the D-065-era ledger text were *temporally correct* and
have been annotated, not rewritten. The genuine defect was narrower: the D-067
report asserted `39 pass` in prose and `12 -> 35` in its own cost block **at the
same commit**. A surface that contradicts itself cannot be resolved by precedence
rules, only by measurement.

**The controlling overlay named no gate.** It said "the board remains 6 PASS /
2 FAIL" and left three gates and seven claims with no checkable statement
anywhere current. Fixed by an explicit gate board and claim board.

**`## Next action` was 18 decisions stale** — last updated at D-051, under a
heading that reads as live guidance, containing a `G-F10C1-REGRESSION` FAIL
statement that a reader could mistake for current. Marked historical in place;
nothing deleted.

## 2. Four fail-closed tools

| Tool | Closes |
|---|---|
| `check_ssot_consistency.py` + `SSOT_FACTS.json` | F-D065-05 permanently: registry vs prose, per-fact commit pinning, decision inventory |
| `check_chronology.py` | F-D065-04 item 6: the impossible-chronology class |
| `cost_report.py` | F-D065-06: cost accounting produced mechanically, never invented |
| `run_axes.py` | Makes a multi-axis contract one overnight instead of four |

`scripts/audit/d070_cost_probe.py` measures per-evaluation cost for candidate
resolutions. It carries a source guard that refuses to run if the file gains an
integration import, and a content guard that rejects its own report if any key
could be read as a physics number. It has not been run — the frozen r4
integration owns the machine.

## 3. The review — 5/5 FAIL, and why that is the headline

A five-reviewer mixed-model adversarial panel (Opus, Sonnet, Haiku — retiring the
"same-model reviews" limitation D-065 recorded) reviewed the first cut: 21
findings, 16 at critical/high, **7 confirmed by independent refutation**.

The decisive finding: **`check_ssot_consistency.py` printed `ok: true` after
checking ZERO status assertions.** Flip both blocking gates `fail -> pass` in the
authoritative registry, neutralise the two sentences naming them, and it went
green while every handoff surface still said FAIL. That is finding F-D065-05
reproduced with the polarity reversed, in the script written to close F-D065-05.
Measured alongside it: all 21 registry claims had zero controlling-prose
coverage, so the claim half was inert as delivered.

Also confirmed and fixed:

- the controlling region truncated at the first heading after the controlling
  overlay, leaving `NEXT_SESSION_PROMPT.md`'s live standing sections unscanned;
- a `prior` entry missing `as_of_commit` collapsed a guard to `"" in line`,
  always true — one deleted JSON key disabled two checks;
- negation was read as agreement ("no longer FAIL" asserted FAIL);
- artifact detection missed `.agent-harness/scripts/`, exactly where D-067 and
  D-068 shipped their code;
- newest-first overlay ordering was assumed, not enforced, in files the docstring
  itself calls append-only;
- every decision-id pattern was capped at D-099, and the failure direction was
  silence;
- `check_chronology.py`'s round-minute detector could be defeated by a sub-minute
  UTC offset: Python parses `23:05:30+00:00:30` to a non-round wall clock that
  orders as exactly `23:05:00Z`, so the checker printed the fabricated round
  minute in its own "verified chain" output as clean;
- an axis could exit 0 having produced nothing and be recorded a clean success;
- an empty commit range produced a clean, healthy-looking cost report.

**The master fix is a coverage floor.** Requiring every gate and every non-exempt
claim to be asserted somewhere current converts a missed phrasing or a too-narrow
region from a silent pass into a loud failure. That floor legitimately failed the
tree, which is why the gate board exists.

The panel also **over-called one finding** — a claimed live registry-vs-prose
contradiction at `PROJECT_STATE.md:1634`. It is historical narrative, correctly
superseded three paragraphs later. It was checked and rejected, not actioned.

Separately, the round-minute rule was measured before being trusted: 3 of 44
commits sit at `seconds==00` against 0.73 expected, but the modal second-value in
that same set occurs 4 times and `00` was selected *after* being flagged. A
look-elsewhere artifact. Round minutes in git commit times are now a warning;
in hand-typeable envelope stamps they remain a default failure.

## 4. Verification

- `check_ssot_consistency.py`: **55/55** regression cases; the pre-fix file
  scores 15/55. Negative control against a real worktree at `a0d63a8` reproduces
  the D-064 three-gate contradiction.
- `check_chronology.py`: rediscovers F-D065-04 on the real D-063 artifacts
  unaided; 32/32 regression cases; 4 intended exit-code changes vs the pre-fix
  file.
- `cost_report.py`: matches `git diff --shortstat` exactly (9122/131/121); five
  fail-closed paths exit 2; empty range exits 3.
- `run_axes.py`: 5 axes at cap 4 finish in 5 s against ≥42 s serial; over-budget
  axis SIGTERM'd while siblings complete; silent axis caught.
- `test_hooks.py` **39/39** unchanged. Validator wiring confirmed by negative
  control: an injected gate contradiction surfaces through
  `validate_harness.py` and clears on restore.

## 5. What is deliberately not claimed

- **No gate or claim status moved.** Board stays 6 PASS / 2 FAIL.
- These tools detect *bookkeeping* falsification and drift. They do not address
  the D-067 residual: harness and subagents share one OS user, so a deliberately
  misbehaving agent can still forge repo-local artifacts.
- The coverage floor proves every gate is *stated* where it is current and agrees
  with the registry. It does not prove the registry is right — that is what
  adjudication is for.
- The D-069 contract's `4.754 s/eval` at order 60 / `y_max` 30 still has no
  committed backing artifact. The probe can check it; it has not been run.

## 6. Cost

```text
production_source_lines_changed: 0
runtime_behavior_changed: yes -- harness validation only
physics_behavior_changed: no
known_blocker_reduced: no -- Part A moves no gate; it makes the bookkeeping
  falsification-resistant and closes the F-D065-05 mechanism
blocker_movement_ratio: 0.0
reviews: 5 reviewers, mixed-model, 5/5 FAIL, 7 confirmed findings, all fixed
tests: 39 hook fixtures unchanged; 55 new SSOT regression cases
```
