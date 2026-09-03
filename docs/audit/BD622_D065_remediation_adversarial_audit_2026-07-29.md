# BD622 D-065 — remediation adversarial audit

Date: 2026-07-29  
Audited HEAD: `a0d63a8d0662d03722567bb5bd54d7f1b235e512`  
D-057 audit base: `287588389f1058ba550b41c3c8d616a04a558273`  
Binding run: `run-20260729-f10-d065-remediation-adversarial-audit-r2`  
Assignment-freeze context:
`208c9c60f631907b9f85881e94ae8ed785febf703641579d36cb3a9003f37a8b`  
Final rebuilt context:
`dead6319044effdb2ae78a0fb4480f355fa4fc34469f73e0cec22fa73e0e72b3`

Status: **FAIL-CLOSED — PARTIAL REMEDIATION — STOP/PRESERVE**

## 1. Terminal verdict

The D-058--D-064 chain contains genuine remediation, but does not prove the
terminal claim that all eight F-10 gates pass.

| Gate | D-065 verdict | Evidence ceiling |
|---|---:|---|
| `G-F10C1-RADIAL` | PASS | unchanged retained evidence |
| `G-F10C1-REGRESSION` | PASS | unchanged D-049 authority |
| `G-F10-PERFORMANCE` | PASS | unchanged retained endpoint evidence |
| `G-F10-CATALOGUE` | PASS | bounded row-9 orientation repair |
| `G-F10-COVARIANCE-METROLOGY` | PASS | frozen matched family and module bytes only |
| `G-F10-SCOPE` | PASS | no forbidden downstream work opened |
| `G-HARNESS-INTEGRITY` | **FAIL** | no exact agent-to-assignment binding; fail-open lease-write fallback |
| `G-F10-INDEPENDENT-FLRW` | **FAIL** | unchanged full-domain evidence obligations not met |

The effective matrix is **6 PASS / 2 FAIL**. The raw D-061 and D-063
numerical payloads are preserved. D-061 closes its narrow matched-family
contract. D-063 is useful matched-cell endpoint evidence, but is not
gate-admissible independent-FLRW validation.

No downstream authority follows. Unblinding, public/production claims,
W7/B3, T01--T12, GL64/Radau, Rust/JAX forward development, F-11/Bianchi,
and QKE remain closed.

## 2. Audit method and independent results

The audit used bounded blind reviews for harness/provenance, static physics,
metrology, trajectory evidence, and software/cost, followed by exact-key
adjudication. Review prompts required:

`self-discover -> step-back -> metacognitive self-ask -> CoVe ->
adversarial self-ask -> CCoT -> PDR`

The adjudicator normalized findings by
`(claim_id, evidence_fingerprint, verdict)` and resolved them against the
unchanged gates. Vote count was not treated as evidence.

| Result | SHA-256 | Verdict |
|---|---|---|
| `A-D065-HARNESS.json` | `70ced3ec7a1ac084cb8b6e1d719d14fc3b98d24bbed42d2e11f7c9358bedf330` | FAIL |
| `A-D065-PROVENANCE-HARNESS.json` | `07f732bfa3831f456002e437edf9e8e7904e4f54059b7c785758aba1e2ea3e68` | FAIL |
| `A-D065-PHYSICS.json` | `0f70b12f9861d00172243f85e63ce4faf22b8d0865359145b2d9f2e274a614ec` | bounded PASS |
| `A-D065-METROLOGY.json` | `dd47daf7fce30f434a510e4a467a2ca24eb8828c95be33a809824f4dd6659c87` | matched-family PASS |
| `A-D065-TRAJECTORY.json` | `579c6bc4e05adf93cf9b68fe9605b33a836481b3636b092dba227450274dbacf` | global gate FAIL |
| `A-D065-SOFTWARE-COST.json` | `8afb0ce1d86008039ae3fe94b0471e597d7fdcc28b077fa945ce00d4779d2302` | FAIL / DRIFT |
| `A-D065-ADJUDICATION-R2.json` | `a9425033a46825f022d650042d0a8fee236098e3f67b88845cb2a9f82da11862` | terminal FAIL |

The first dedicated adjudicator launch failed before work because the runtime
rejected its requested model. The replacement used a supported `default`
runtime with a sealed adjudicator role and result template; it did not
impersonate a dedicated runtime.

## 3. Deduplicated findings

### F-D065-01 — exact assignment attribution is bypassable

**Verdict: FAIL / critical**

The D-058 Start lease seals every assignment registered in the active run.
Stop then accepts a stopping agent's declared assignment when its digest is
present in that all-assignment lease. It does not possess a
parent-authenticated `agent_id -> exactly one assignment_id` binding.

A controlled negative represented an agent launched for
`A-D065-METROLOGY` but submitted the valid artifact and envelope for the
different, same-runtime `A-D065-HARNESS` assignment. Stop returned success
and consumed the lease. The 12 focused hook tests contain no corresponding
same-run substitution fixture.

Start also catches a lease-write `OSError` and permits Stop to fall back to
mutable `ACTIVE_RUN`, restoring the authority path that D-057 rejected. The
D-058 overlapping-run canary is therefore a bounded success-path result, not
proof of unconditional lifecycle integrity.

Minimum prospective remedy:

1. atomically bind one spawned `agent_id` to one run, assignment, and digest;
2. make receipt/lease creation failure a hard Start failure;
3. add same-run substitution and receipt-write-failure negative fixtures;
4. pass a replacement live overlapping-run canary with unique write
   attribution.

### F-D065-02 — the orientation repair is real but bounded

**Verdict: PASS / high**

Module SHA-256
`760a7c044081e507fae9d5695b301bd44f6466d96322c46f53b77161e32b558a`
anchors the audited static surface. The 27-event two-orientation
construction, exact target-row accounting, reversal/sign equivalence,
asymmetric flavour/CP checks, conservation, entropy, and the order-one
single-chart falsifier support the row-6/row-9 repair.

`C-F10-ROW9-CLOSURE` remains `VALIDATED` only for this frozen orientation
artifact. Absolute V-A/phase-space normalization and physical catalogue
completeness remain `INCONCLUSIVE`.

### F-D065-03 — covariance metrology passes at its declared ceiling

**Verdict: PASS / high**

D-061 report SHA-256 is
`56529e65b59726610e5d484fb37d1460d60153e6af9feba95254c8d21686fad0`;
the preserved D-060 failure is
`e40d80a747bfc907c67537b5f6898ac952976809cfec8d780a09fb1c8ea629e2`.
The r3/r4 physics values are identical; r4 prospectively changes only the
treatment of the analytically degenerate P-fixed N7X observable.

The five asymmetric family members satisfy weak/mass identities at roughly
`1e-15`, native N3 at at most `5.091972401420414e-11`, the explicit
`E_split` operation graph, off-grid/event equivariance, interval
containment, five mutants, and restoration canary. A retained fresh replay
artifact `/tmp/d065_d061_rerun.json` has SHA-256
`bc3566b211e8568c3b482109b2caabec78c5160a6d52915c55fd766c9a18f812`
and records all eight states and 1,522,656 containment checks.

This is not a general-state theorem, continuum result, absolute-normalization
validation, cross-platform replay, or fully collision-independent MPFR
implementation.

### F-D065-04 — D-063 is supportive but does not close independent FLRW

**Verdict: FAIL / critical**

D-063 report SHA-256 is
`daf4fc06b8cc3bb5b07a92e436fd95f78cb208a10becaac22c7013e7f2fba9d4`.
Its endpoint
`(N_eff,N,t)=(3.034054308076679,7.936698865363719,52678.7319 s)`,
completed-catalogue anchors, block values, coupled-energy sign-mutant kill,
and tighter-rtol holdout are credible supportive evidence.

They do not close the unchanged gate:

1. D-062's prospective full-48-node T8 value is
   `2.0800641747316935e-3 > 2e-3`. D-063 excludes two low-y nodes and obtains
   `1.5881626368027338e-3`.
2. Both excluded nodes are inside the implementation's declared affine
   `[0,24]` evaluation domain; excluding them changes the frozen norm after
   output.
3. Checkpoints retain reduced scalars, not the full 146-state
   occupation/logit vector, temperature, and time needed for independent
   recomputation.
4. `tail_last_node_fraction` is a final GL-node contribution, not an evolved
   `y>24` tail enclosure.
5. The holdout varies `rtol` only; it supplies no radial, angular, domain, or
   tail uncertainty.
6. D-062/D-063 result envelopes contain impossible completion chronology
   relative to their containing commits.

The small quadrature weight of the excluded nodes is a useful steelman, not a
replacement for the frozen predicate. A future attempt requires a
prospective full-domain or derived weighted norm, recomputable full
checkpoints, an evolved beyond-domain tail enclosure, at least one
spatial/domain holdout, and valid adjudication chronology.

### F-D065-05 — D-064 SSOT surfaces contradicted one another

**Verdict: FAIL before D-065 correction / critical**

Before correction, the frozen decisions, claim registry, and gate registry
recorded D-064 PASS flips while shared context and project-state controlling
text still declared those gates FAIL and forbade the already-completed
D-060--D-063 work. Structural hash validation cannot resolve semantic
precedence. D-065 reconciles the current surfaces to the effective
6-PASS/2-FAIL state while retaining D-064 as historical evidence.

### F-D065-06 — scope passes; anti-inflation fails

**Verdict: `G-F10-SCOPE=PASS`; cost verdict `DRIFT`**

D-058--D-064 changes no production physics source and opens no forbidden
downstream programme. It nevertheless touches 93 paths and adds
16,297/deletes 75 lines (`net +16,222`), including 10,686 tracked-run lines
and 4,320 audit-script lines, with zero production-source lines. The
D-060/D-061 scripts are approximately 99.5% identical and D-062/D-063
approximately 98.5% identical; report bodies are also duplicated in stdout
logs. Required cost accounting was absent from the D-064 closeout surfaces.

## 4. Claim-state correction

- `C-F10-ROW9-CLOSURE`: `VALIDATED`, bounded to the frozen orientation
  artifact.
- `C-F10-METROLOGY-R3` and `C-F10-COVARIANCE-METROLOGY`: `VALIDATED`,
  matched-family only.
- `C-F10-TRAJECTORY-R2` and `C-F10-INDEPENDENT`: `IMPLEMENTED` with
  supportive output, not gate-validated.
- `C-HARNESS-INTEGRITY`: `IMPLEMENTED`; the successful-lease path is tested,
  but the global claim is not validated.

## 5. Validation executed

PASS:

```text
python3 .agent-harness/scripts/build_context_pack.py
# context_version=dead6319044effdb2ae78a0fb4480f355fa4fc34469f73e0cec22fa73e0e72b3

venv/bin/python -m pytest -q -p no:cacheprovider \
  .agent-harness/tests/test_hooks.py
# 12 passed

PYTHONPATH=src venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_independent_noqke_comparator.py
# 3 passed

PYTHONPATH=src venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_claim_gates.py tests/test_surface_scope_honesty.py \
  tests/test_audit_hardening_regressions.py
# 33 passed

git diff --check
# PASS
```

The first post-correction validation failed closed because the old audit run
was bound to assignment context `208c9c60...`:

```text
python3 .agent-harness/scripts/validate_harness.py
# ok=false
# Active run was initialized against a stale context version.
```

That failure is preserved. A result-free closeout run was then initialized at
the final context; no canary or scientific authority is implied:

```text
python3 .agent-harness/scripts/validate_harness.py
# ok=true
# context_version=dead6319...
# active_run=run-20260729-f10-d065-adversarial-audit-closeout-r2
```

This final structural PASS only makes the updated pack/run internally
consistent. It does not repair exact assignment attribution or close
`G-HARNESS-INTEGRITY`.

Skipped:

- full D-063 trajectory replay — a stopped partial run is not evidence;
- full Rust release/report bundle — D-057--D-064 does not change production
  or Rust physics code.

### Write-attribution incident

During D-065 finalization, the tracked historical result
`.agent-harness/runs/run-20260728-f10-d057-adversarial-adjudication-r5/results/A-D057R5-CROSS-REJECT.json`
changed concurrently outside the D-065 run. It was not authored by the D-065
main writer. It remains dirty and unadded, is not silently reverted, and its
apparent line deletion is not credited as intentional anti-inflation
deflation. This observation establishes cross-run working-tree concurrency,
but the available parent-side evidence does not establish which runtime wrote
it or that its own registered assignment was violated. It is therefore
excluded from D-065 evidence and accounting. The controlled same-run
cross-assignment Stop test in F-D065-01, not this dirty file, establishes the
assignment-attribution bypass.

## 6. Cost line

The following accounting is for the audited D-058--D-064 remediation chain,
not the smaller D-065 adjudication and SSOT-correction surface:

```text
added_lines: 16297
deleted_lines: 75
net_lines: 16222
files_touched: 93
token_use_exact: UNAVAILABLE — no reliable stage-scoped counter
token_use_basis: harness/API exposes no reliable per-stage counter
runtime_behavior_changed: yes — harness only
physics_behavior_changed: no
known_blocker_reduced: yes — bounded metrology and failure localization
blocker_movement_ratio: 0.25
validation_strengthened: yes
cost_effectiveness_verdict: DRIFT
```

## 7. Stop condition

Preserve D-058--D-064 and all raw PASS/FAIL artifacts. Repair the harness and
trajectory-evidence gates in separate, prospectively frozen slices. Until
both pass, reject F-10 terminal closeout and remain `STOP/PRESERVE`.
