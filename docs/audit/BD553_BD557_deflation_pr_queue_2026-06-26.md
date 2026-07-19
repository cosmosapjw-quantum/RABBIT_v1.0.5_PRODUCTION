# BD553-BD557 Deflation PR Queue

Date: 2026-06-26

Scope: continue the post-BD552 AP65 no-QKE deflation cycle while preserving
physics outputs, default policies, raw failure evidence, public-dispatch scope,
and QKE exclusion.  These PRs are not endpoint speed claims; endpoint wall
claims require same-recipe long runs.

| PR | Target | Acceptance | Status |
| --- | --- | --- | --- |
| BD553 | Lift row wall-second summary aggregation out of `build_augmented_continuous_ap65_source_rhs_prototype_artifact`. | Helper test proves live-row aggregation; artifact builder no longer owns this nested summary logic. | done |
| BD554 | Lift row-level phase-2 Jacobian refresh count aggregation out of the artifact builder. | Helper test covers refresh-vs-evaluation fallback without changing artifact summary values. | done |
| BD555 | Reassess helper extraction cost versus large-function reduction. | `/review` note records line cost, remaining largest functions, and whether to continue this cycle. | done |
| BD556 | Switch to `_build_augmented_continuous_ap65_full_bbn_span_ladder_artifact_impl` deflation. | Focused test covers runtime-linked behavior in span-ladder row/progress packaging, not a synthetic count lock. | done |
| BD557 | Continue span-ladder deflation or pivot to endpoint-improving PR planning based on BD556 cost. | Decision document names the next performance/physics blocker and required run evidence. | done |

Anti-drift note: this is a queue ledger only.  It adds no runtime gate and does
not claim solver validation, endpoint progress, or publication readiness.
