# D-081R1F0 preflight blocker audit

Date: 2026-09-04  
Repository: `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
Audited branch: `research/d081r1f0-rust-c-only-jvp-20260904`  
Audited head: `77e659c31ee03882f83191e6474eb60709002634`  
Audited tree: `a911a56950fe1f853dbdad4cecd6e39870515371`  
Exact parent authority: `8cef907e704149340774214f4da1bd28b79608e9`  
Parent tree: `189e100de980fdbbe654e579d83c939cbdb1cef1`  
Classification: **SOURCE_IMPLEMENTATION_PRESENT; EXECUTION_NOT_ADMISSIBLE**

## 1. Scope of this audit

This is a pre-execution source and workflow audit. It does not report a Rust
compile, test, cross-language comparison, retained-state run, unseen holdout,
performance measurement, solver call, trajectory, endpoint, or `N_eff` result.
The execution marker is intentionally not created in this audit.

The branch is a direct 11-commit descendant of the exact D-081R1E feature
head. Source modules are present for the cloglog/chart tangent, self-action
spectral-`c` JVP, electron-action spectral-`c` JVP, combined action JVP, and
packed-RHS push-forward.

Source inspection finds the c-only formulas and decomposition structurally
consistent with the frozen D-079 Python authority:

- `delta f = q delta c`;
- `delta log q = (1-exp(c)) delta c`;
- fixed-kinematics event tangents differentiate only the Pauli factor;
- `delta(C/(H q))` includes collision, Hubble, and chart-chain terms;
- induced photon-temperature and elapsed-output tangents are present;
- the zero direction has an exact-zero fast path.

This is a source-level observation, not implementation verification.

## 2. Blocking findings

### P0-SW-1 — missing oracle generator

The workflow calls

```text
native/rabbit_cpu/tests/fixtures/d081r1/generate_rust_c_only_jvp_oracle.py
```

for `order8`, `retained-calibration`, and `retained-holdout`, but that file is
absent from the audited tree. The workflow cannot reach the advertised
cross-language gates.

### P0-SW-2 — derivative submodules are not registered in their parent modules

The files

```text
native/rabbit_cpu/src/f10_self_action/c_jvp.rs
native/rabbit_cpu/src/f10_electron_action/c_jvp.rs
```

exist, and `.github/scripts/d081r1f0_register_modules.py` is designed to add

```rust
pub(crate) mod c_jvp;
```

to both parent modules. The script is not invoked by the workflow, and the
parent-module edits are absent from the audited diff. The current head is
therefore expected to fail Rust module resolution before scientific tests.
This expectation remains unverified until an actual compiler run exists.

### P1-TEST-1 — only the exact-zero test is implemented

The committed Rust JVP test file contains one exact-zero-direction test. The
contracted nonzero order-8 comparison, retained calibration, linearity,
first-law, conservation/symmetry, centered-difference, mutation, and unseen
holdout tests are not yet represented in the Rust test source.

### P1-HOLDOUT-1 — unseen holdout is consumed before calibration admission

The workflow generates all three oracle fixtures, including the retained
unseen holdout, before Rust compilation and before order-8 or retained
calibration gates run. A compile or calibration failure would therefore expose
or consume the preregistered holdout before the implementation is admitted.
The workflow must be split so that holdout generation and execution occur only
after a separately durable GREEN calibration head.

### P1-PROV-1 — tested-tree identity is underspecified

The workflow runs `cargo fmt` in the Actions worktree and may commit the
formatted source plus a receipt afterward. The planned receipt records the
triggering `GITHUB_SHA`, but not a complete tested-tree identity, exact changed
blob list, numeric gate payload, or the post-format commit/tree. Publication
must bind the receipt to the exact bytes actually compiled and tested.

### P1-RED-1 — RED-first runtime receipt is not established

The contract prescribes an absent-API compile failure before implementation.
No workflow run or durable runtime receipt is visible for the audited head.
Commit ordering alone does not prove that the prescribed RED execution
occurred. This is a provenance gap, not a physics defect.

## 3. External checks

A fresh Wolfram symbolic replay was attempted for the chart, Pauli, Hubble,
and quotient-rule identities, but the external service returned HTTP 502.
Current formula status is therefore `DERIVED_AND_SOURCE_CROSSCHECKED`, not
`FRESH_WOLFRAM_VERIFIED`.

SciSpace triage supports differentiating the already discretized nonlinear
operator when an analytical Jacobian/JVP is required for stiff kinetic systems,
and retaining finite differences as a validation rather than production path.
That literature does not validate this repository implementation, its support
semantics, or its thresholds.

A local clone/compile attempt in the present external audit environment was
blocked by a container `ClientError`. This is an environment blocker and is not
scientific evidence for or against the implementation.

## 4. Required remediation DAG

### R1F0-P0A — make the source compilable

1. Add explicit `pub(crate) mod c_jvp;` declarations to both parent modules.
2. Keep the registration script only as an idempotence checker, or invoke it
   before a `git diff --exit-code` preflight.
3. Add a cheap workflow preflight that checks every referenced source,
   fixture-generator, contract, and authority path before installing tools.

### R1F0-P0B — complete the oracle and test harness

1. Commit `generate_rust_c_only_jvp_oracle.py` from the frozen D-079 sources.
2. Emit binary64 bit strings, raw metrology, branch/domain diagnostics,
   authority identities, and exact direction definitions.
3. Add Rust tests for nonzero order-8 and retained calibration directions,
   linearity, first law, self moments, CP/mu-tau covariance, centered witnesses,
   and contracted mutations.

### R1F0-P0C — restore holdout discipline

1. Run order-8 and retained calibration in a calibration-only workflow.
2. Publish a durable GREEN calibration commit and exact receipt.
3. Only from that exact head, generate and execute the unseen retained holdout
   once in a separate workflow/job with a separate receipt.

### R1F0-P0D — bind tested and published bytes

The final receipt must record the pre-test head, tested tree, all tested source
blob hashes, formatter diff, post-test/evidence commit and tree, workflow and
artifact identities, raw numerical results, and exact claim ceiling. Remote
readback must confirm those identities before any merge recommendation.

## 5. Stop decision

Do not create `docs/audit/D081R1F0_EXECUTE_2026-09-04` on the audited head.
Doing so would produce a known workflow-assembly failure and would consume the
holdout before calibration. The next admissible mutation is bounded harness
repair, not execution.

## 6. Claim ceiling

The branch currently demonstrates only that a source implementation of the
spectral-`c` analytic JVP has been written on the correct R1E lineage. It does
not yet establish compilation, numerical equivalence, conservation,
fixed-support differentiability, retained-state validity, holdout validity,
solver admission, performance, trajectory completion, endpoint agreement,
`N_eff`, `G-F10-INDEPENDENT-FLRW` movement, release authority, or publication
readiness.
