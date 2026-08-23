# Run-local validation ledger

Run: `run-20260823-ode-four-loop-final-implementation`  
Raw logs: `raw_logs/`  
Machine summary: `artifacts/VALIDATION_SUMMARY.json`

| Log | Command class | Exit | Result |
|---|---|---:|---|
| 00 | environment/head/package capture | 0 | captured |
| 01 | pre-edit focused non-slow | 0 | 10 pass, 3 deselect |
| 02 | invalid nonfinal RED | 1 | expected failure: did not raise |
| 03 | invalid nonfinal GREEN | 0 | 1 pass |
| 04 | Hubble RED matrix | 1 | 5 expected failures |
| 05 | Hubble GREEN matrix | 0 | 5 pass |
| 06 | attempted exact-HEAD mutants | 1 | contaminated by worktree conftest; inadmissible |
| 07 | corrected exact-HEAD mutants | 1 | 11 expected failures |
| 08 | C1 adversarial focused | 0 | 18 pass |
| 09 | candidate focused non-slow | 0 | 28 pass, 3 deselect |
| 10 | candidate focused all | 1 | 3 fail, 28 pass |
| 11 | exact-HEAD focused all | 0 | 13 pass |
| 12 | production-not-slow, system env | 2 | 30 collection errors; no verdict |
| 13 | measured-env install | 0 | complete |
| 14 | measured-env import/device | 0 | exact versions and CPU JAX verified |
| 15 | production-not-slow, measured env | 0 | 396 pass, 8 skip |
| 16 | gold, measured env | 1 | 5 fail, 100 pass, 2 skip |
| 17 | exact-HEAD selected gold failures | 1 | identical five failures |
| 18 | full default, measured env | 1 | 54 fail, 2587 pass, 80 skip |
| 19 | candidate focused, measured env | 1 | 3 fail, 28 pass |
| 20 | exact-HEAD focused, measured env | 0 | 13 pass |
| 21 | shared-context harness validation | 0 | harness `ok: true`; current-run failed dedicated receipt remains explicitly open |
| 22 | replacement-review assignment verification | 0 | assignment, sealed role, and result-template hashes PASS |

Independent adjudication:

- `results/A-IF-REVIEW2.json`: completed under a sealed logical adjudicator
  role; `REJECT/STOP_INVALID`; SHA-256
  `c84de9341601ccb91dfd257f7311ac427275b767f033467f0043601406a40932`.
- The earlier dedicated-runtime attempt is a recorded host-boundary error only;
  its unconsumed receipt is not represented as scientific evidence.

Validation summary:

- red-green discrimination: VALIDATED;
- fake accepted-path identity: VALIDATED;
- real candidate compatibility: FAILED;
- production-not-slow: PASSED;
- gold: FAILED, selected failures baseline-reproduced;
- full default: FAILED;
- endpoint, D-071, physical-prefix, public-production, QKE: NOT RUN / FORBIDDEN.
