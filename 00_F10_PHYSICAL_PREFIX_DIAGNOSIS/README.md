# F-10 physical-prefix diagnosis entrypoint

This directory is the first inspection point for the standalone
`diagnosis_report` branch. It binds the exact
`f10-independent-validation-b3v2@719987d0bc5a018d57fded1df2c8ad3f0c3fc24f`
source identity, the retained order-60 / `y_max=30` campaign states, a
deterministic initial state, value-level quadrature/catalog hashes, and
prospectively sealed direct physical RHS/JVP receipts. Canonical source and raw
campaign bytes stay at their original repository paths; they are linked and
hashed rather than copied here.

## Read this result correctly

- `IMPLEMENTED`: the requested artifact set and branch-local navigation exist.
- `VALIDATED`: byte bindings, four base physical RHS calls, first-law,
  strict-occupation, domain/roundoff, finite-domain tail diagnostics, and the
  direct-JVP attempts were executed and retained against the active seal.
- `SPECIFIED`: the full prefix from the physical initial state through
  `N >= 0.25`, including `0.14 <= N <= 0.22`, remains a contract obligation.
- `FORBIDDEN`: claiming that this branch ran that prefix, validated reaction
  tails, reopened D-071, changed a registry gate, established public production
  support, or merged into `main`.

The active recovery receipt has overall status
`EXECUTED_WITH_RETAINED_FAILURES`. All four states retain base RHS and the
requested first-law/occupation/domain/tail diagnostics. At `creep_1200`, the
fixed time-augmented direct-JVP rule completed one shifted call, then its next
Arnoldi direction used `epsilon = 4.3184509101514785e15` and
`shifted_N = -6.898440601726866e7`; the shifted coordinate was rejected outside
the strict occupation domain. That is a retained negative admissibility result,
not a missing artifact and not a successful solver-prefix claim.

## Start with these files

- [FILE_LOCATIONS.md](FILE_LOCATIONS.md) — human map to every canonical source,
  retained input, campaign log, contract, receipt, and validation surface.
- [BRANCH_SCOPE.json](BRANCH_SCOPE.json) — branch purpose, base, no-main-merge
  rule, non-goals, and claim ceiling.
- [PROVENANCE_INDEX.json](PROVENANCE_INDEX.json) — machine-readable path,
  SHA-256, Git blob, source-commit, requirement, role, and claim-status index.
- [SOURCE_BUNDLE.json](SOURCE_BUNDLE.json) — exact base tree/subtree/blob
  inventory plus both retained ZIP and internal Git-bundle identities.
- [PREFIX_INPUTS.json](PREFIX_INPUTS.json) and
  [QUADRATURE_CATALOG_MANIFEST.json](QUADRATURE_CATALOG_MANIFEST.json) — exact
  retained states and value-level order-60 grid/catalog hashes.
- [PREFIX_CONTRACT.json](PREFIX_CONTRACT.json) and
  [PREFIX_CONTRACT.sha256](PREFIX_CONTRACT.sha256) — active prospective recovery
  contract, sealed before `receipts_v2/` existed.
- [active physical receipt](receipts_v2/PHYSICAL_RHS_JVP_RECEIPTS.json),
  [raw vectors](receipts_v2/PHYSICAL_RHS_JVP_VECTORS.npz), and
  [run log](receipts_v2/RECEIPT_RUN_LOG.json).
- [RECEIPT_INDEX.json](RECEIPT_INDEX.json) and
  [READINESS.json](READINESS.json) — selectors and artifact/scientific boundary.
- [VALIDATION_LEDGER.json](VALIDATION_LEDGER.json) and
  [SHA256SUMS](SHA256SUMS) — executed checks and complete payload digests.

## Prospective chronology

| Phase | Commit | Meaning |
|---|---|---|
| Source base | `719987d0bc5a018d57fded1df2c8ad3f0c3fc24f` | Exact F-10 runtime source authority. |
| First seal | `27afe5d817f0382c47cdc2cef2703cca69d827ed` | v1 code/input/contract committed before first receipt output. |
| Preserved first attempt | [receipt](receipts/PHYSICAL_RHS_JVP_RECEIPTS.json) | Exposed a retention defect: `creep_1200` JVP failure erased its valid base diagnostics. The bytes are preserved unchanged. |
| Active recovery seal | `acb5641e8008f0c8305e8e83db4d7269ba9e1cd6` | Binds the first attempt, fixes failure retention only, and prospectively declares `receipts_v2/`; physics source and JVP parameters are unchanged. |
| Active receipt | [v2 receipt](receipts_v2/PHYSICAL_RHS_JVP_RECEIPTS.json) | Four base receipts plus successful or failed direct-JVP attempt provenance. |

## Verify locally

Run from the repository root:

```bash
sha256sum -c 00_F10_PHYSICAL_PREFIX_DIAGNOSIS/PREFIX_CONTRACT.sha256
sha256sum -c 00_F10_PHYSICAL_PREFIX_DIAGNOSIS/SHA256SUMS
PYTHONPATH=src:. python scripts/audit/f10_physical_prefix_fixture.py verify-receipts \
  --repo . --output-dir 00_F10_PHYSICAL_PREFIX_DIAGNOSIS \
  --seal-commit acb5641e8008f0c8305e8e83db4d7269ba9e1cd6
PYTHONPATH=src:. python 00_F10_PHYSICAL_PREFIX_DIAGNOSIS/verify_final_json_normalized.py \
  --repo . --output-dir 00_F10_PHYSICAL_PREFIX_DIAGNOSIS \
  --seal-commit acb5641e8008f0c8305e8e83db4d7269ba9e1cd6
```

The active seal's original `verify-final` command compares catalogue tuple
fields directly with their JSON-loaded list representation and therefore raises
despite identical canonical JSON bytes. The linked post-seal wrapper changes
only that container representation, then delegates every hash, chronology,
tracking, ignore, receipt, regeneration, and Markdown-link check to the sealed
verifier. Its focused regression is
[tests/test_f10_physical_prefix_final_verify.py](../tests/test_f10_physical_prefix_final_verify.py).

Do not rerun or replace either receipt set in place. Any changed protected byte,
JVP parameter, state, or threshold requires a new contract identity and a new
prospective seal.
