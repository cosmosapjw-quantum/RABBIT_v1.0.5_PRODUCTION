# F-10 Physical-Prefix Diagnosis Report Branch Design

**Date:** 2026-08-11
**Branch:** `diagnosis_report`
**Base:** `f10-independent-validation-b3v2@719987d0bc5a018d57fded1df2c8ad3f0c3fc24f`
**Purpose:** publish a provenance-preserving map of the exact F-10 physical-prefix inputs and evidence without copying the source tree or presenting a diagnostic bundle as runtime progress.

## Claim and publication boundary

- `SPECIFIED`: this branch defines how an external reviewer locates and verifies the retained physical-prefix prerequisites.
- `IMPLEMENTED` applies only after every declared path, hash, and machine-readable index has been generated and checked on this branch.
- `VALIDATED` applies only to checks that actually run and are recorded.
- `FORBIDDEN`: treating finite-difference observation Jacobians as physical JVP evidence, treating retained creep states as a completed prefix, changing `G-F10-INDEPENDENT-FLRW`, or claiming public production/QKE support.
- The controlling scientific ceiling remains `PHYSICAL PREFIX NOT YET RUN` and `D-071 REOPEN NOT EARNED`.
- This task pushes `diagnosis_report` as a standalone remote branch. It does not merge the branch into `main`.

## External-reader entry point

The first content in the repository root `README.md` will be a branch-specific notice:

1. explain that `diagnosis_report` is a non-production provenance and gap report;
2. state that it is not a gate pass or physical-prefix result;
3. direct the reader first to `00_F10_PHYSICAL_PREFIX_DIAGNOSIS/README.md`.

The uppercase `00_` prefix makes the diagnosis directory visible at the beginning of ordinary repository listings. The notice is intentionally branch-local and must not be merged into the production README.

## Root diagnosis directory

```text
00_F10_PHYSICAL_PREFIX_DIAGNOSIS/
  README.md
  FILE_LOCATIONS.md
  BRANCH_SCOPE.json
  PROVENANCE_INDEX.json
  PREFIX_INPUTS.json
  RECEIPT_INDEX.json
  READINESS.json
  SHA256SUMS
```

### Human-readable files

`README.md` is the short start page. It states the claim ceiling, explains the index files, and gives the exact verification commands.

`FILE_LOCATIONS.md` is the complete human-readable map. Every requirement has:

- a stable requirement ID;
- an evidence status;
- a repository-relative Markdown link to the canonical file;
- the role of that file;
- whether the bytes are retained, derived, specified, or missing;
- the exact blocker caused by a missing item.

Links target the original tracked location. The diagnosis directory does not contain duplicate Python, Rust, NPZ, JSONL, test, or audit bytes.

### Machine-readable files

`BRANCH_SCOPE.json` records schema ID, branch name, base commit, intended remote ref, non-merge rule, claim ceiling, non-goals, and controlling documents.

`PROVENANCE_INDEX.json` contains one object per source or evidence artifact with these fields:

```json
{
  "artifact_id": "stable-id",
  "requirement_ids": ["REQ-..."],
  "repo_path": "relative/path",
  "role": "source|input|checkpoint|rhs_provenance|jacobian_observation|receipt|contract|audit",
  "status": "PRESENT_TRACKED|PRESENT_DERIVED|MISSING|NOT_APPLICABLE",
  "sha256": "lowercase digest or null",
  "git_blob_oid": "blob oid or null",
  "source_commit": "commit oid or null",
  "claim_status": "IMPLEMENTED|VALIDATED|DERIVED|SPECIFIED|PROPOSED|DEPRECATED|FORBIDDEN",
  "notes": "bounded factual note"
}
```

`PREFIX_INPUTS.json` records order `60`, `y_max=30`, state dimension `182`, the three retained creep checkpoint paths and NPZ field contracts, the analytic initial-state generator location, and canonical quadrature/catalog hash definitions. A derived initial-state digest may be recorded, but it must be labelled `DERIVED_FROM_FROZEN_SOURCE`; it is not retained-run evidence.

`RECEIPT_INDEX.json` maps first-law, strict occupation, domain-rejection, finite-tail, matrix-roundoff, RHS, and JVP requirements to exact retained files and selectors. It must distinguish:

- a retained direct receipt;
- a value recomputable from retained bytes;
- a source-level implementation only;
- a missing physical receipt.

In particular, the retained `obs_jac_*.npz` files are finite-difference observation Jacobians. They must not be relabelled as a physical JVP receipt.

`READINESS.json` is a fail-closed requirement matrix. Any missing physical JVP receipt, missing prospective executable prefix contract, untracked path, hash mismatch, or unavailable tail receipt keeps `physical_prefix_ready=false`. This file is a diagnosis summary, not a new project gate and not a registry input.

`SHA256SUMS` is a sorted digest list for the diagnosis files and every newly published retained evidence file. Existing sealed files are hashed byte-for-byte and are never reformatted.

## Canonical retained evidence

The exact retained V3a creep evidence remains at its original path:

```text
.agent-harness/runs/run-20260805-f10-v3-campaign/
```

Only the provenance-bearing slice needed by this report is published:

- campaign source and adjudication: `run_v3.py`, `analyse_v3.py`, `ANALYSIS_V3.json`, `report_verification_output.json`, and `r4_reference.json`;
- V3a run pins and top-level logs: `v3a_r2/pins_verified.json`, `selftest_result.json`, `driver.log`, and `nohup.log`;
- complete V3a domain evidence: retained `state_*.npz`, `obs_jac_*.npz`, `ratcheted_cols_*.npz`, line-buffered JSONL records, `summary.json`, and the domain logs.

The `.gitignore` rule for `.agent-harness/runs` will be narrowed with exact negated paths for this slice. All other volatile run directories remain ignored. No harness script, hook, admission record, gate registry, or shared context file is changed.

Historical files already tracked in Git remain in place and are linked rather than copied, including:

- `src/rabbit/decoupling/_independent_noqke.py`;
- `scripts/audit/_trajectory_core.py`;
- `scripts/audit/d069_independent_trajectory_r4.py`;
- the relevant Rust collision/quadrature/catalog sources;
- the D-069, D-071, V0, V1, V2, and V3 audit reports;
- the retained r4 report and stdout already tracked under their canonical run path.

## Data flow

```text
root README notice
  -> 00_F10_PHYSICAL_PREFIX_DIAGNOSIS/README.md
    -> FILE_LOCATIONS.md for human navigation
    -> READINESS.json for the fail-closed summary
      -> PROVENANCE_INDEX.json / PREFIX_INPUTS.json / RECEIPT_INDEX.json
        -> canonical source, retained state, raw logs, and audit reports
```

No reverse dependency is introduced: runtime source does not import the diagnosis directory, tests do not promote it as a backend, and the branch does not modify public dispatch.

## Missing-evidence policy

- Missing artifacts are represented with `status="MISSING"`, `sha256=null`, and a concrete blocking reason.
- A local-only path is not advertised as externally accessible; it must either be tracked at its original repository path or remain `MISSING`.
- A source implementation is not a receipt.
- A finite-difference Jacobian is not a JVP.
- A prospective contract is not marked sealed unless it binds an executable entry point, source/input hashes, step/JVP policy, output paths, hard caps, kill criteria, and is committed before the first physical output byte.
- No new physical collision call, prefix integration, or solver benchmark is run as part of assembling this diagnosis branch.

## Verification

The branch implementation must run and record:

1. JSON parsing for every machine-readable file.
2. A schema/field check over every index entry.
3. `git ls-files --error-unmatch` for every `PRESENT_TRACKED` path.
4. `git check-ignore` proving every published required path is not ignored and unrelated run paths remain ignored.
5. `sha256sum -c 00_F10_PHYSICAL_PREFIX_DIAGNOSIS/SHA256SUMS`.
6. NPZ loading with `allow_pickle=False`, required fields `t`, `y`, `raw`, `h`, and `order`, and `y.shape == (182,)` for all three retained creep states.
7. Cross-check of recorded source hashes and checkpoint scalars against the retained V3 report and pins.
8. Link-target existence checks for every Markdown and JSON repository path.
9. `git diff --check`.
10. Existing focused claim/scope tests sufficient to prove that no gate or public-support claim moved.

Validation results are appended to the existing validation ledger only for commands actually executed. The claim ledger and handoff state are updated only if the implementation changes their current factual summary.

## Git chronology and delivery

1. Commit this approved design on `diagnosis_report`.
2. Add the root README notice, diagnosis indexes, narrow ignore exceptions, and original-path evidence bytes.
3. Run the verification list and correct only defects in the diagnosis package.
4. Perform an adversarial change review focused on provenance, missing-evidence honesty, and accidental production drift.
5. Commit the verified package with anti-drift line accounting and `token_use_exact: UNAVAILABLE` unless an exact counter becomes available.
6. Push `diagnosis_report` to `origin/diagnosis_report` and verify the remote SHA.
7. Do not merge to `origin/main` in this task.

## Cost-effectiveness boundary

The branch adds no runtime path and makes no blocker-movement claim. Its value is external auditability of bytes that already exist and an explicit inventory of what still blocks a lawful physical-prefix run. The cost verdict must therefore be `ACCEPT_WITH_LIMITS` only if the package replaces ambiguity with exact tracked evidence and introduces no new gate or wrapper dependency; otherwise it is `DRIFT`.
