# F-10 Physical-Prefix Diagnosis Report Branch Design

**Date:** 2026-08-11
**Branch:** `diagnosis_report`
**Base:** `f10-independent-validation-b3v2@719987d0bc5a018d57fded1df2c8ad3f0c3fc24f`
**Purpose:** publish a provenance-preserving, remotely inspectable physical-prefix fixture in which every owner-requested artifact resolves to tracked bytes, while keeping fixture completeness separate from a physical-prefix result or D-071 gate movement.

## Acceptance boundary

The branch is complete only when all seven requirements below resolve on the remote branch. A required artifact may not be left `MISSING`, local-only, ignored, or transcript-only.

| Requirement ID | Required artifact | Completion rule |
|---|---|---|
| `REQ-SOURCE` | exact `f10-independent-validation-b3v2` source bundle | exact base commit/tree and per-file Git/blob/SHA-256 inventory resolve; the retained solver-research ZIP and its internal Git history bundle are tracked and digest-checked |
| `REQ-CHECKPOINTS` | order-60, `y_max=30` retained creep checkpoints | all three original V3a state NPZ files and their raw campaign provenance are tracked at their canonical run paths and load with the sealed field/shape contract |
| `REQ-INPUTS` | initial/input state and quadrature/catalog hash manifest | a deterministic initial-state NPZ plus value-level quadrature and catalog manifests are generated from the frozen source and independently recomputed |
| `REQ-RHS-JVP` | physical collision RHS/JVP provenance | prospectively specified direct physical RHS calls and time-augmented directional JVP/Arnoldi observations run on the initial state and all three retained creep states; raw vectors, counters, hashes, environment, and source/input bindings are retained |
| `REQ-RECEIPTS` | first-law, occupation, domain-rejection and tail receipts | each state has an executed machine receipt; limitations or failed predicates remain explicit and do not make the artifact absent |
| `REQ-CONTRACT` | prospectively sealed prefix contract | the machine contract binds source/input hashes, method/JVP rules, interval, output paths, caps, kill criteria and no-refit rule in a Git commit made before the first new physical receipt byte |
| `REQ-ENTRYPOINT` | external-reader navigation | the root README begins with a branch notice and points first to the diagnosis directory, whose Markdown links resolve to canonical tracked locations |

Artifact completeness is not scientific acceptance. `SPECIFIED`, `IMPLEMENTED`, `VALIDATED`, and negative/limited receipt verdicts remain distinct. `FORBIDDEN`: treating finite-difference observation Jacobians as direct JVP evidence, treating retained creep states or static receipts as a completed prefix, changing `G-F10-INDEPENDENT-FLRW`, or claiming public production/QKE support. The controlling ceiling remains `PHYSICAL PREFIX NOT YET RUN` and `D-071 REOPEN NOT EARNED` unless a separately authorized, contract-complete run and adjudication change it.

This task pushes `diagnosis_report` as a standalone remote branch. It does not merge the branch into `main`.

## Exact source authority

Source identity has three non-substitutable layers:

1. **RABBIT runtime source lock.** `SOURCE_BUNDLE.json` records base commit `719987d0bc5a018d57fded1df2c8ad3f0c3fc24f`, its tree, the `src`, `native`, `tests`, `scripts`, and `docs/audit` subtree OIDs, and a complete sorted manifest of the prefix-relevant files with Git blob OID and SHA-256. The remote branch history is the byte authority; no copied source tree is placed in the diagnosis directory.
2. **Retained solver-algorithm source bundle.** The original root artifact `RABBIT_F10_SolverAlgorithm_Blocker_Research_Loop_2026-08-06.zip` is removed from ignore for this exact path and tracked byte-for-byte. Its ZIP SHA-256, internal `REPRODUCIBILITY_MANIFEST.json`, branch `research/solver-algorithm-loop`, commit `b8f11b03d9d59746c4ceddbb0712dfbd3f5386ab`, internal Git bundle, `exprb.py`, JVP/Arnoldi source, contract and integration plan are indexed without unpacking duplicates into the diagnosis directory.
3. **Tail/physics research provenance.** The original root artifact `RABBIT_F10_MathPhysics_Blocker_Research_Loop_2026-08-06.zip` is likewise tracked and hashed because it contains the retained tail-bound calculations and the explicit warning that an equilibrium energy tail is not a universal reaction-domain certificate.

`SOURCE_BUNDLE.json` must expose a deterministic reconstruction command for the runtime tree and verification commands for both ZIPs. An archive, branch name, or prose report without all relevant digests is insufficient.

## External-reader entry point

The first content in the repository root `README.md` is a branch-specific notice that:

1. explains that `diagnosis_report` is a non-production physical-prefix fixture and diagnosis branch;
2. states that it is not a prefix result, gate pass, or mainline merge;
3. directs the reader first to `00_F10_PHYSICAL_PREFIX_DIAGNOSIS/README.md`.

The uppercase `00_` prefix makes the diagnosis directory visible at the beginning of ordinary repository listings. The notice is branch-local and must not be merged into the production README.

## Root diagnosis directory

```text
00_F10_PHYSICAL_PREFIX_DIAGNOSIS/
  README.md
  FILE_LOCATIONS.md
  BRANCH_SCOPE.json
  SOURCE_BUNDLE.json
  PROVENANCE_INDEX.json
  PREFIX_INPUTS.json
  QUADRATURE_CATALOG_MANIFEST.json
  RECEIPT_INDEX.json
  READINESS.json
  PREFIX_CONTRACT.json
  PREFIX_CONTRACT.sha256
  initial_state_order60_ymax30.npz
  receipts/
    PHYSICAL_RHS_JVP_RECEIPTS.json
    PHYSICAL_RHS_JVP_VECTORS.npz
    RECEIPT_RUN_LOG.json
  VALIDATION_LEDGER.json
  SHA256SUMS
```

### Human-readable files

`README.md` is the short start page. It states the claim ceiling, explains the index files, and gives exact verification commands.

`FILE_LOCATIONS.md` is the complete human-readable map. Every requirement has a stable ID, repository-relative Markdown link to canonical bytes, artifact role, provenance class, SHA-256, and validation/limitation note. Source trees, retained run evidence, and research archives remain at their original paths; only newly derived fixtures and indexes live in the diagnosis directory.

### Machine-readable files

`BRANCH_SCOPE.json` records schema ID, branch/base, intended remote ref, no-main-merge rule, claim ceiling, non-goals, and controlling documents.

`SOURCE_BUNDLE.json` records the exact Git source lock and retained source archives described above.

`PROVENANCE_INDEX.json` contains one object per source or evidence artifact with these fields:

```json
{
  "artifact_id": "stable-id",
  "requirement_ids": ["REQ-..."],
  "repo_path": "relative/path",
  "role": "source|source_bundle|input|checkpoint|rhs_provenance|jvp_provenance|receipt|contract|audit",
  "status": "PRESENT_TRACKED|PRESENT_DERIVED_VALIDATED",
  "sha256": "lowercase digest",
  "git_blob_oid": "blob oid or null for a generated pre-commit index",
  "source_commit": "commit oid",
  "claim_status": "IMPLEMENTED|VALIDATED|DERIVED|SPECIFIED|PROPOSED|DEPRECATED|FORBIDDEN",
  "notes": "bounded factual note"
}
```

No requested artifact may use `MISSING`, `NOT_APPLICABLE`, or a null digest. Negative scientific verdicts belong in receipt/readiness fields, not in artifact-presence fields.

`PREFIX_INPUTS.json` records order `60`, `y_max=30`, state dimension `182`, the retained checkpoint paths and NPZ field contracts, and the deterministic initial-state fixture. The initial state is labelled `DERIVED_FROM_FROZEN_SOURCE`, never retained-run evidence.

`QUADRATURE_CATALOG_MANIFEST.json` records value-level hashes, not only source-file hashes:

- canonical little-endian float64 byte hashes for all 60 nodes and weights;
- scalar grid parameters and array shapes/dtypes;
- a stable canonical-JSON hash for self-reaction, electron-reaction, self-event and electron-event catalogs;
- item counts and stable identity fields sufficient to detect reorder or content drift;
- generator function and frozen source hash.

`RECEIPT_INDEX.json` maps first-law, strict occupation, domain rejection, matrix roundoff, tail, RHS, and JVP requirements to the new direct receipts and to the historical logs. It identifies `obs_jac_*.npz` only as retained finite-difference observation Jacobians and never as direct JVP receipts.

`READINESS.json` separates at least these booleans:

- `requested_artifact_set_complete`;
- `fixture_hashes_validated`;
- `static_physical_receipts_executed`;
- `prospective_contract_sealed_before_receipts`;
- `physical_prefix_executed`;
- `reaction_tail_authority_validated`;
- `d071_reopen_earned`.

The first four must be true before push. The last three remain false unless their distinct evidence actually exists. This file is not a project gate or registry input.

`SHA256SUMS` is a sorted digest list for every diagnosis payload except the checksum list itself, both retained ZIPs, and every newly published retained evidence file. Existing sealed files are hashed byte-for-byte and never reformatted.

## Canonical retained evidence

The exact retained V3a creep evidence remains at:

```text
.agent-harness/runs/run-20260805-f10-v3-campaign/
```

The published provenance slice includes:

- campaign source/adjudication: `run_v3.py`, `analyse_v3.py`, `ANALYSIS_V3.json`, `report_verification_output.json`, `r4_reference.json`, and `render_v3.py`;
- exact instrumentation sources at their earlier canonical location under `.agent-harness/runs/run-20260804-f10-v1-diagnostic/instrument/`;
- V3a pins and top-level logs: `v3a_r2/pins_verified.json`, `selftest_result.json`, `driver.log`, and `nohup.log`;
- complete V3a domain evidence: `state_*.npz`, `obs_jac_*.npz`, `ratcheted_cols_*.npz`, all line-buffered JSONL records, `summary.json`, and domain logs.

The `.gitignore` rules for `.agent-harness/runs`, `*.log`, `*.zip`, and `*.sha256` are narrowed with exact trailing exceptions. Unrelated runs, logs, archives, and hashes remain ignored. No hook, admission record, gate registry, or shared-context file is changed.

Historical tracked files are linked rather than copied, including `_independent_noqke.py`, `_trajectory_core.py`, the D-069 driver, relevant Rust collision/quadrature/catalog sources, D-069/D-071/V0/V1/V2/V3 audit reports, and the retained r4 report/stdout.

## Initial state and direct physical receipt execution

The initial fixture is generated with the frozen `build_setup(order=60, y_max=30)` and `initial_state` functions, stored as an NPZ with `allow_pickle=False`-compatible numeric/scalar fields, and regenerated independently during verification. It binds the same 182-component layout as the three retained V3a states.

After the contract-seal commit, a narrow audit runner evaluates the unmodified physical collision/RHS path on four states: the generated initial state plus `state_1200.npz`, `state_2000.npz`, and `state_3000.npz`. For each base call it retains:

- `N`, state/input hashes, temperatures, environment and source hashes;
- full RHS, collision-action and energy-transfer vector hashes plus raw vectors in NPZ;
- `first_law_residual`, strict occupation min/max and pass flag;
- whole-reaction domain-rejection count, matrix-roundoff count and maximum correction;
- equilibrium `y>30` number/energy tail fractions and resolved last-node distortion metrics;
- an explicit `reaction_tail_authority_validated=false` limitation because these quantities do not replace the missing extended-domain high-precision lost-action oracle.

The direct JVP provenance follows the solver survivor's exact time-augmented rule. With `z=(y,N)`, `G(z)=(F(N,y),1)`, normalized Arnoldi direction `v`, and fixed `r=10^-3`,

```text
epsilon = r * max(1, ||z||_2) / ||v||_2
J_G(z)v = ((F(N + epsilon*v_N, y + epsilon*v_y) - F(N,y))/epsilon, 0).
```

The runner starts Arnoldi from `(F(N,y),1)`, uses `m=10`, double modified Gram-Schmidt and the fixed breakdown tolerance declared in the contract. It retains every direct shifted RHS call, direction/epsilon, signal-to-cancellation metrics, orthogonality residual, breakdown status, invariant/occupation/domain diagnostics, and vector hash. No historical `obs_jac_*.npz` value is used to manufacture the direct JVP receipt.

The receipt generator is a physical-correctness blocker move, not a generic manifest gate: it executes the actual collision path needed to decide whether the frozen directional-difference semantics are even admissible on retained physical states. It does not integrate a trajectory or promote a runtime backend.

## Prospective prefix contract and chronology

`PREFIX_CONTRACT.json` binds:

- RABBIT base commit/tree and every source/archive/input hash;
- order 60, `y_max=30`, state layout and exact initial/checkpoint set;
- physical RHS/collision evaluator identity;
- total-energy-coordinate requirement and algebraic `T_gamma` recovery as an unresolved executor obligation, never silently substituted by the old state coordinate;
- EC-EXPRB-K baseline `m=10`, the time-augmented JVP rule above, double MGS and no persistent finite-difference ratchet;
- prefix coverage from the physical initial state through at least `N=0.25`, including `0.14 <= N <= 0.22`;
- full-RHS-equivalent accounting, call projection cap `<=5500`, frozen wall cap `64800 s`, output paths and required raw ledgers;
- first-law, occupation, domain/tail, roundoff, Arnoldi/fallback and reproducibility predicates;
- fail-closed kill semantics, no fallback concealment, no post-output epsilon/`m`/step refit, and immutable retention of failures.

Commit chronology is part of validation:

1. **Seal commit:** track source archives, canonical retained inputs, generator/verification code, generated initial/grid/catalog manifests, and `PREFIX_CONTRACT.json` plus its digest. No new physical receipt output exists yet.
2. **Receipt execution:** run only against the clean seal commit and record that commit plus contract digest in every output.
3. **Receipt commit:** add raw receipt outputs, indexes/readiness/validation ledger and final digest list without changing any sealed source, input, JVP rule, threshold or runner.

A diff between the seal and receipt commits over protected paths must be empty. Any change requires a new contract identity and invalidates the earlier outputs.

## Verification

The implementation must execute and retain:

1. JSON parsing and schema/field checks for every machine-readable file.
2. Internal ZIP manifest and Git-bundle verification plus external ZIP SHA-256 checks.
3. `git ls-files --error-unmatch` for every published artifact.
4. `git check-ignore` proving every required path is visible while unrelated volatile artifacts remain ignored.
5. `sha256sum -c 00_F10_PHYSICAL_PREFIX_DIAGNOSIS/SHA256SUMS`.
6. NPZ loading with `allow_pickle=False`, required state fields, finite values, and `(182,)` state vectors.
7. Byte-for-byte independent regeneration of the initial state and quadrature/catalog manifests.
8. Direct receipt schema, source/input/contract hash and call-accounting checks.
9. Contract chronology and protected-path diff checks.
10. Markdown/JSON link-target existence checks.
11. Focused regression tests for generator/validator code and the relevant existing physics tests.
12. `git diff --check` and exact added/deleted/net line accounting.

Validation results enter `VALIDATION_LEDGER.json` and the existing validation ledger only for commands actually executed. Claim/handoff ledgers are updated when their factual state changes; no receipt is promoted beyond its evidence.

## Delivery

After implementation and adversarial review, push `diagnosis_report` to `origin/diagnosis_report`, fetch the remote ref, and verify exact SHA equality. Do not merge it to `origin/main` in this task.

## Cost-effectiveness boundary

The allowed new code directly executes and verifies physical RHS/JVP behavior on the exact retained states; the remaining files consolidate provenance around that blocker-moving receipt. The final change must report added/deleted/net lines, `token_use_exact: UNAVAILABLE` if the harness exposes no exact counter, blocker-movement ratio, and a cost-effectiveness verdict. The verdict is `DRIFT` if the branch adds a generic gate/manifest dependency, claims endpoint progress from static calls, or leaves any requested artifact ignored or unresolved.
