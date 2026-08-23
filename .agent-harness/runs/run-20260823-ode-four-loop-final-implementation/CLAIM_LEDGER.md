# Run-local claim ledger

Run: `run-20260823-ode-four-loop-final-implementation`  
Head: `78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`

This ledger is run-local because the candidate was rejected and no canonical
project claim changed. It does not amend `docs/harness/CLAIM_LEDGER.md`.

| Claim ID | Status | Evidence | Disposition / ceiling |
|---|---|---|---|
| IF-EVIDENCE-MAP | VALIDATED | `artifacts/FOUR_LOOP_EVIDENCE_MAP.json` | 34 IDs exactly; bounded inventory only |
| IF-C1-CONTRACT | SPECIFIED | `results/A-IF-C1.json` | coherent before execution; strict-boundary contact was an explicit stop |
| IF-C1-CANDIDATE | IMPLEMENTED | `artifacts/C1_REJECTED_CANDIDATE.patch` | isolated and executed; REJECTED, not admitted |
| IF-C1-ADVERSARIAL | VALIDATED | `raw_logs/02`--`08` | fake corruption/probe distinctions discriminate and pass after RED |
| IF-C1-FAKE-IDENTITY | VALIDATED | `artifacts/valid_dummy_*.json` | byte-identical fake valid path only |
| IF-C1-REAL-TRAJECTORY | VALIDATED | `artifacts/raw_trajectory_*collision_on.json`; logs 19/20 | 2,096 tail negatives; C1 rejects, HEAD passes |
| IF-C1-FINAL-ADJ | VALIDATED | `results/A-IF-REVIEW2.json` | independent REJECT/STOP_INVALID; zero blocker/gate movement |
| IF-C1-BLOCKER-MOVEMENT | PROPOSED | logs 10, 18, 19 | FAIL; zero blocker/gate movement |
| IF-POSITIVITY-COORDINATE | PROPOSED | final report section 10 | derived direction; no implementation/certificate |
| IF-C2-RODAS | PROPOSED | `A-CH-DECIDE.json` | REWORK; event units/tolerances/budgets absent |
| IF-C3-SLOW-MANIFOLD | PROPOSED | `A-MAC-ADJ2.json` | REWORK; static discriminator only |
| IF-PRODUCTION-NOT-SLOW | VALIDATED | log 15 | 396 pass, 8 skip; no promotion implication |
| IF-GOLD | VALIDATED | logs 16/17 | executed FAIL; five selected failures reproduce on HEAD |
| IF-FULL-DEFAULT | VALIDATED | log 18 | executed FAIL; 54 failures, three C1-direct |
| G-F10-INDEPENDENT-FLRW | SPECIFIED | gate registry | retained FAIL/CLOSED_ON_CURRENT_MEASUREMENT |
| G-HARNESS-INTEGRITY | SPECIFIED | gate registry | retained FAIL |
| IF-ENDPOINT-PROMOTION | FORBIDDEN | anti-drift scope | not earned |
| IF-QKE-PUBLIC-PRODUCTION | FORBIDDEN | anti-drift scope | out of scope |
