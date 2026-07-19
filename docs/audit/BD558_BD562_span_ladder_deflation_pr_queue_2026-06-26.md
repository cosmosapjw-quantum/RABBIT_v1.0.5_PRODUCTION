# BD558-BD562 Span-Ladder Deflation PR Queue

Date: 2026-06-26

Scope: continue reducing the largest span-ladder executable surfaces without
changing physics outputs, raw observables, endpoint claims, default optimization
policies, or QKE scope.

| PR | Target | Acceptance | Status |
| --- | --- | --- | --- |
| BD558 | Extract endpoint observable/status extraction from `_row_from_source`. | Missing/failed endpoint rows remain raw and explicit; `_row_from_source` shrinks. | done |
| BD559 | Extract shared child row provenance metadata used by resolution/composition child rows. | Resolution and composition row tests preserve labels, inputs, and raw child status. | done |
| BD560 | Split `_resolution_case_row_from_child` selected summary assembly. | Existing resolution matrix/ablation tests pass; no summary key is dropped. | done |
| BD561 | Split `_attach_h_refinement_metadata` or `_h_refinement_attempt_summary` if still cost-effective. | Nested attempt telemetry remains surfaced; no double-count or clipping introduced. | done |
| BD562 | `/review` checkpoint for BD558-BD561. | Records line cost, largest functions, blocker movement, and whether to pivot to endpoint runs. | done |

Anti-drift note: this queue does not add a runtime readiness gate.  It is a
scope-control ledger for code deflation and maintainability work.

## BD561 Result

`_h_refinement_attempt_linear_system_payload()` now owns the h-refinement
attempt linear-system serialization fields.  This preserves the existing row
field provenance while reducing `_h_refinement_attempt_summary()` from 376 to
305 lines.  The new regression is not a count-lock: it exercises row-field
normalization, fallback behavior, shape serialization, and optional iterative
solver values that would be lost if nested attempt telemetry were replaced by
schema-only placeholders.

Line cost: source +78/-72, test +61/-0, net +67.  Exact token counters are
UNAVAILABLE because the harness does not expose per-PR token accounting.
Blocker movement ratio: 0.10, because this is a span-ladder maintainability
deflation PR and does not claim endpoint wall or physics-readout progress.
Cost verdict: ACCEPT_WITH_LIMITS; the target function shrank by 71 lines, but
the added test/helper surface means the next checkpoint should reassess whether
continued span-ladder deflation is still the best use of PR budget.

## BD562 Result

See `docs/audit/BD562_span_ladder_deflation_review_2026-06-26.md`.
Adversarial checkpoint verdict: REQUEST PIVOT AFTER THIS CHECKPOINT.  BD558-BD561
improved reviewability but added net code and did not directly improve endpoint
wall, endpoint reach, or physics-readout blockers.  The next queue should be
endpoint-facing and should quantify same-recipe before/after wall and observable
deltas.
