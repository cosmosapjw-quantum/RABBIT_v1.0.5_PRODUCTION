# Generated status board

This file is rendered by `.agent-harness/scripts/build_status_board.py` from
`.agent-harness/context/GATE_REGISTRY.json` and
`.agent-harness/context/CLAIM_REGISTRY.jsonl`. It is the only authority for
current gate and claim status. Do not edit it, and do not restate a status
anywhere else: prose about a status is not checked and has been wrong before.

<!-- BEGIN GENERATED STATUS BOARD -->

| Gate | Status | Basis |
|---|---|---|
| `G-F10-CATALOGUE` | PASS | nine rows execute with rowwise invariants at module bytes 760a7c04 |
| `G-F10-COVARIANCE-METROLOGY` | PASS | D-065 upholds it only on the frozen matched family and module bytes |
| `G-F10-INDEPENDENT-FLRW` | FAIL | D-065 restored FAIL; D-071 closed the lane on current measurement |
| `G-F10-PERFORMANCE` | PASS | measured whole-endpoint reduction, closed with its profile |
| `G-F10-SCOPE` | PASS | the F-10 fence holds and downstream authority is closed |
| `G-F10C1-RADIAL` | PASS | D-004/D-014: frozen direct and five-profile envelope |
| `G-F10C1-REGRESSION` | PASS | retained tree passes its frozen regression set |
| `G-HARNESS-INTEGRITY` | FAIL | D-065 restored FAIL; obligations 1, 3 and 4 adjudicated discharged at round 13, obligation 2 wording falsified |

| Claim | Status |
|---|---|
| `C-F10-B3V2-DESIGN` | PROPOSED |
| `C-F10-CATALOGUE` | VALIDATED |
| `C-F10-COVARIANCE-METROLOGY` | VALIDATED |
| `C-F10-FULL-AUTHORITY` | FORBIDDEN |
| `C-F10-INDEPENDENT` | IMPLEMENTED |
| `C-F10-INDEPENDENT-GALERKIN-M1` | VALIDATED |
| `C-F10-INDEPENDENT-MAXENT3-DESIGN` | VALIDATED |
| `C-F10-INDEPENDENT-POINTWISE-M1` | VALIDATED |
| `C-F10-METROLOGY-R3` | VALIDATED |
| `C-F10-PERF` | VALIDATED |
| `C-F10-ROW9-CLOSURE` | VALIDATED |
| `C-F10-SCOPE` | IMPLEMENTED |
| `C-F10-TRAJECTORY-R2` | IMPLEMENTED |
| `C-F10-W3-PARTIAL` | DERIVED |
| `C-F10-W5-LOCALIZATION` | VALIDATED |
| `C-F10C1-CACHE` | DEPRECATED |
| `C-F10C1-ENDPOINT` | IMPLEMENTED |
| `C-F10C1-N48` | VALIDATED |
| `C-HARNESS-INTEGRITY` | IMPLEMENTED |
| `C-R6-ORBIT-CHART` | VALIDATED |
| `C-R6-ORBIT-CHART-RABBIT-APPLICABILITY` | VALIDATED |

<!-- END GENERATED STATUS BOARD -->
