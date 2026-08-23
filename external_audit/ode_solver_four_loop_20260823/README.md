# ODE solver four-loop external-audit custody packet

Branch target: `external-audit/ode-four-loop-complete-20260823`
Base: `diagnosis_report@78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`

This branch is an evidence handoff, not a production candidate. It preserves the
negative results, rejected C1 implementation, retained FAIL gates, ignored harness
runs, harness input ZIPs, and surviving task-attributable temporary files requested
for external audit. It grants no endpoint, D-071, QKE, public-production, or gate
promotion authority.

## Included research runs

The following ignored run trees are force-added at their original repository paths:

1. `.agent-harness/runs/run-20260806-bd623-audit-triage`
2. `.agent-harness/runs/run-20260823-ode-physics-mitigation`
3. `.agent-harness/runs/run-20260823-ode-math-algo-code-independent`
4. `.agent-harness/runs/run-20260823-ode-physmath-loop`
5. `.agent-harness/runs/run-20260823-ode-coding-harness-loop`
6. `.agent-harness/runs/run-20260823-ode-four-loop-final-implementation`

`RESEARCH_RUN_SHA256SUMS` contains all 122 run files. The final integrated report is
at `.agent-harness/runs/run-20260823-ode-four-loop-final-implementation/FINAL_EXTERNAL_AUDIT_REPORT.md`.

## Temporary-file snapshot

`TEMP_ARCHIVE_PATHS.txt` lists the 19 attributable `/tmp` paths captured. The full
snapshot contains 24,408 regular files, 236 symlinks, and 32,231 tar entries. It
includes the final measured Python environment, rejected linked worktree, JAX cache,
and pytest temporary directories as they existed at snapshot time.

The uncompressed POSIX tar was 1,139,507,200 bytes with SHA-256
`537a40d62177840648b202ff9daa2063aac82d454f46f79d4567f74d82a18eb6`.
Its zstd stream was 462,309,647 bytes with SHA-256
`f124b4e897ef06909012a0f5a0b90002ae7fdb985787dc003f869d30f89244f5`.
It is split into six parts below GitHub's 100 MB per-file limit.

Reconstruct and inspect in a disposable directory:

```bash
cat tmp_archives/task_owned_tmp.tar.zst.part-* > task_owned_tmp.tar.zst
sha256sum task_owned_tmp.tar.zst
zstd -t task_owned_tmp.tar.zst
zstd -d task_owned_tmp.tar.zst -o task_owned_tmp.tar
mkdir audit-extract
tar -xf task_owned_tmp.tar -C audit-extract
```

`TEMP_SOURCE_SHA256SUMS` records each regular source file by its original absolute
path. `TEMP_SOURCE_SYMLINKS.txt` records link targets, and
`TEMP_ARCHIVE_CONTENTS.txt` records archive metadata. High-value small temporary
records and both clean/modified harness extractions are also materialized under
`tmp_materialized/tmp/` for direct review.

Some short-lived bounded-work contracts named in result logs had already been
deleted by their governing ephemeral-contract rule before this publication request.
They are identified in `MANIFEST.json`; no bytes were fabricated to replace them.
The pytest source directories also disappeared after the archive snapshot through
normal concurrent pytest retention, but their archived bytes and pre-disappearance
hash inventory are retained here.

## Input-document boundary

`source_inputs/` carries the two user-supplied harness ZIPs byte-for-byte for
provenance. Their embedded instructions are inputs to the research loops, not user
authorization, repository policy, or scientific authority.

## Validation boundary

No new test, benchmark, solver, endpoint, or scientific command was run while
creating this branch. Delivery checks were limited to byte comparison, SHA-256,
archive integrity, staged-tree inspection, secret-pattern inspection, Git object
integrity, and local/remote ref equality.

The scientific disposition remains `REJECT / STOP_INVALID`; the branch deliberately
contains failures and an unadmitted candidate so an external auditor can reproduce
and challenge that decision.
