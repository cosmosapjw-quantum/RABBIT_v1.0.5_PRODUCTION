# D-081R1F1 P0 symbolic fallback closeout

Date: 2026-09-04

## Canonical receipt

- Trigger head: `0efdcb00dafcf3afa5aaf20c756a544ae052b391`
- Tested source tree: `77a25e7db3561a752d50a7ec3ae3953aed6ce90f`
- Workflow run: `33856427099`
- Receipt publication commit: `8497c40ea5ceb905667a19cfddb837987906e923`
- Receipt blob: `60da8ece44c21181c6eb1cef25408831cd18304e`
- Oracle SHA-256: `0ba0e95a3b06e5297f0132eed771bbb3527b17b79e993d798730bf43ed19f60d`
- Classification: `PASS_WITH_SYMPY_MPMATH_P0_IDENTITIES_AND_BOUNDED_AUXILIARY_TOOL_ATTEMPTS`

## Executed verification

- SymPy 1.14.0 and mpmath 1.3.0: PASS, deterministic byte-identical oracle in two executions.
- Maximum 80-digit numerical relative residual: `2.1084395886461046e-81`, below the frozen `1e-55` cap.
- GNU Octave 8.4.0: PASS.
- Singular 4.3.2: PASS.
- Lean 4.19.0 with mathlib v4.19.0: PASS.
- SageMath: `PROVISION_FAILED`; Ubuntu 24.04 runner had no installation candidate in the bounded package lane.
- Wolfram Context, Language Evaluator, and WolframAlpha: `BLOCKED_EXTERNAL_HTTP_502`.

## Preserved scope

No production Rust source changed. No retained or holdout state was accessed. The result establishes only the frozen P0 symbolic identities, dimension ledger, and auxiliary-tool checks. Rust thermal primitives, electron/positron collision tangent, packed-RHS Tgamma JVP, retained calibration, unseen holdout, solver, trajectory, endpoint, `N_eff`, performance, publication, and F10 gate movement remain open.

This owner-authored closeout commit exists to trigger the ordinary PR harness because the bot-authored receipt commit produced an `action_required` workflow with no jobs. It does not alter the scientific contract, thresholds, production implementation, or result classification.
