# PR #1 Dynamic Adversarial Re-Audit and Compiled Closure Contract — Revision 2

Date: 2026-08-25 (Asia/Seoul)  
Repository: `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
Pull request: `#1`  
Audited base: `a59531ad9c5f041c65ceed5d1508c4ea30c8dbf5`  
Audited head: `50d3bc5b8093bc33e9311f94505c5ee0711ce51b`  
Disposition: **REQUEST_CHANGES / DO NOT MERGE**  
Compiled contract: `/.codex/audit/pr1-r1c-closure/AUDIT_COMPILED_PACKAGE.json`

## 0. Why this revision exists

The first compiled package correctly found a real P0 false `Solved`, but an
independent contract review found that its dispatch contract was internally
inconsistent:

1. the pass condition required total `P1=0` while three P1 items were explicitly
   deferred to `PR-ODE-R1E` or to the claim firewall;
2. the Python binary64 mirror was described too broadly as independent evidence
   and was placed in final-SHA verification even though it never calls Rust;
3. no exact-head Rust precondition proved that the released PR head actually
   returned the frozen known-bad bits;
4. the stagnation fixture, certificate-attainability rule, and path globs were
   ambiguous or contradictory;
5. the unique-physical-root lemma, external-oracle assertion in occupation
   units, non-degeneration gate, tolerance provenance, and review-loop bound
   were missing.

Revision 2 accepts those findings. The old package is retained for provenance,
but must not be dispatched.

## 1. Evidence classes

The sources must not be conflated.

### 1.1 Repository source evidence

The exact `a59531ad..50d3bc5` diff and the source at `50d3bc5` establish what
Rust and CI currently do.

### 1.2 Executed Rust evidence

GitHub Actions run `32787997031` passed its `validate` and `native-r1c` jobs.
That run is useful but incomplete: it did not execute the revised P0 fixture,
the full revised invariant matrix, or the fresh-review mutations.

### 1.3 Audit-authored binary64 control-flow model

`pr1_r1c_numeric_reproducer.py::solve_audit_model` mirrors the old Rust control
flow. It is **not an independent implementation oracle**. It serves only as a
released baseline hypothesis and golden-data generator.

It therefore belongs in preflight, not in final-SHA proof.

### 1.4 Independent exact-real reference

The Decimal solve in the same file independently evaluates the exact-real
affine-state residual defined by the frozen binary64 inputs. An independent
reviewer further cross-checked the primary fixture using SymPy rational algebra
and mpmath at 120 digits. The following quantities matched:

```text
physical root                   -8.60317773351298611141741279699e-16
nearest binary64 bits           0xbcceff07e8a38d5c
old head returned bits          0xbcceff0807bfa264
occupation error                3.5367629503399001e-12
128 epsilon                     2.8421709430404007e-14
error / threshold               124.4387836347543
```

This cross-check resolves circularity for the primary golden fixture. Rust still
must reproduce the known-bad result at the exact base and then show RED/GREEN
against the corrected contract.

## 2. The P0 remains real

The current `certified_flux_evaluation` begins only after
`occupations_at_extent` has rounded

```text
extent / measure
initial +/- quotient
```

and it does not enclose the complement, `h*flux`, or residual subtraction as one
exact-real interval chain. The primary PairSource fixture therefore returns
`Solved` with an actual occupation-root error about `124.44` times the frozen
threshold.

Revision 2 adds two distinct baseline gates:

1. the audit model and exact-real reference must emit the released baseline
   data;
2. a temporary test-only patch must run against Rust at exactly `50d3bc5` and
   observe `0xbcceff0807bfa264`.

If the second gate disagrees, the stop status is
`REPRODUCER_MODEL_DIVERGENCE`; the implementation must not start.

## 3. Correct threat accounting

The old pass condition was impossible because it treated deferred P1 blockers
as defects that PR #1 had to remove. Revision 2 partitions the catalogue.

### 3.1 Must close in PR #1

```text
P0-001 exact-real state-map/residual interval
P0-002 common authority for every Solved path
P0-003 no contract/tolerance/fixture mutation
P1-002 adjacent-representable uncertifiable interval taxonomy
P1-003 repeated-extent stagnation
P1-004 discarded log/exp hot path
P1-005 malformed report evidence
P1-006 incomplete CI
P1-007 stale final-SHA evidence
```

### 3.2 Deferred typed blockers

```text
P1-001 exact detailed-balance evolution semantics    -> PR-ODE-R1E
P1-008 accumulated CertifiedFrozen error budget      -> PR-ODE-R1E
P1-009 local-to-global scientific claim overreach     -> claim firewall
```

These remain P1. They are acceptable at PR-#1 closeout only when their exact
typed blocker and claim ceiling are present. They may not be silently demoted,
implemented inside PR #1, or counted as open must-close findings.

The fresh-review pass condition is therefore:

```text
must_close_p0_open = 0
must_close_p1_open = 0
deferred_without_required_typed_blocker = 0
deferred_claim_ceiling_violations = 0
new_uncatalogued_p0_p1 = 0
```

It is not `total P1 = 0`.

## 4. Mathematical contract

### 4.1 Dimensions

```text
occupation f                          dimensionless
edge measure m                        quadrature-weighted dimensionless measure
gain, loss, flux                      MeV
step h                                MeV^-1
extent xi and residual                weighted dimensionless occupation
```

### 4.2 Unique physical root

The state map is affine, but the residual is generally quadratic. A second
algebraic root may exist outside the physical capacity interval.

For PairSource,

\[
\partial_{f_1}J=-A(1-f_2)-Bf_2\le0,
\qquad
\partial_{f_2}J=-A(1-f_1)-Bf_1\le0,
\]

and both `df_i/dxi` are positive. Thus `dJ/dxi <= 0`.

For ElasticTransfer,

\[
\partial_{f_1}J=-Af_2-B(1-f_2)\le0,
\qquad
\partial_{f_2}J=A(1-f_1)+Bf_1\ge0,
\]

while `df_2/dxi = -1/m_2`. Again `dJ/dxi <= 0`.

Hence, on the Pauli box,

\[
r'(\xi)=1-hJ'(\xi)\ge1>0.
\]

Together with the analytic inward signs at the capacity endpoints, this proves
existence and uniqueness of the **physical** root. Newton derivatives are useful
proposals; only outward-rounded interval evidence may authorize `Solved`.

The derivation is frozen as `INV-R1C-012` and must be pinned by a physical-box
property test.

## 5. Certificate semantics

### 5.1 Full interval path

The certificate-only path must outward-round:

```text
xi/measure
  -> initial +/- quotient
  -> 1-f
  -> nonnegative products
  -> gain-loss flux interval
  -> h*flux interval
  -> residual interval
```

For residual interval `[r_lo,r_hi]`, define

```text
root_error_abs = max(abs(r_lo), abs(r_hi))
occupation_error_abs = root_error_abs / min(m1,m2)
```

Every `Solved` constructor must pass one common helper. Point binary64 states
remain returned candidates; point derivatives are proposal-only.

### 5.2 Soundness assertion in the accepted units

Whenever an external golden root exists, the test must assert

```rust
report.occupation_error_abs
    >= (report.extent - golden).abs() / min_measure;
```

Checking only `root_error_abs >= |extent-golden|` does not pin the actual gate.

### 5.3 Generalized external-oracle coverage

The primary P0 fixture is not the only external oracle. Revision 2 freezes exact
nearest-root bits for the local PairSource ladder

```text
m=(2,5), A=13, B=17, f=(0.23,0.79)
h=2^-8   -> 0xbf6e4ad0bfc2b909
h=2^-14  -> 0xbf0f8e82450eec7e
h=2^-20  -> 0xbeaf93c82712ec39
h=2^-30  -> 0xbe0f93dd9299eefb
```

The local ladder must remain `Solved`, and every report must enclose its external
root in occupation units.

### 5.4 Non-degeneration

Fail-closed behavior is not an unlimited escape hatch. After both closure work
units:

- the local golden ladder remains `Solved`;
- the sweep legacy ladder retains `solved == edge_applications`,
  `unresolved == 0`, `exact_stationary == 0`;
- every aggregate is finite;
- maximum reported occupation error is at most `128 epsilon`.

Failure is `BLOCKED_CERTIFICATE_OVERCONSERVATISM`, not a mergeable patch.

## 6. Certificate-unattainable semantics

The old documents conflicted between an interval-width rule and the heuristic
`h*delta_J/min(m)`. Revision 2 resolves the conflict.

`h*delta_J/min(m)` and `eta_J` are diagnostics only. They are not a universal
attainability theorem.

`CertificateUnattainableAtStep` is allowed only when:

1. the unique physical root remains interval-enclosed;
2. current, midpoint, lower, and upper candidate acceptance have all failed;
3. lower and upper are adjacent binary64 extents, so no representable extent
   lies between them;
4. the occupation-space bracket width still exceeds `128 epsilon`.

A repeated candidate with an unchanged non-adjacent bracket is instead
`StagnatedInterval`.

The exact stagnation fixture is:

```text
ElasticTransfer
m1=m2=2^-30
A=B=2^-20 MeV
h=1 MeV^-1
f=(1/8,1/4)
```

No other fixture may be substituted.

## 7. Path and scope semantics

All package paths are repository-rooted and begin with `/`.

```text
allowed /native/rabbit_cpu/src/pauli_edge_step.rs
forbidden /src/**
```

therefore do not conflict: `/src/**` means only the root Python package tree.

## 8. Baseline and TDD gates

The execution order is mandatory.

```text
PRE-001  exact remote head SHA
PRE-002  baseline audit model + independent exact-real golden output
PRE-002B temporary Rust patch observes the known-bad bits at 50d3bc5
PRE-003  existing focused Rust suite passes
PRE-004  future contract test is observed RED before production edits
```

The temporary patch is
`/.codex/audit/pr1-r1c-closure/BASELINE_KNOWN_BAD.patch`. It must be reversed and
leave a clean source diff before implementation.

The Python model is deliberately absent from final-SHA proof. Final proof is the
Rust regression, the external golden values, mutations, and final-SHA command
logs.

## 9. Review convergence

At most two fresh-context review rounds are allowed. A round is one
classification-only review plus, when needed, one implementation fix and full
evidence regeneration.

After round two, unresolved must-close P0/P1 findings stop as
`BLOCKED_REVIEW_NONCONVERGENCE`. Deferred typed blockers do not trigger a loop
when their exact contract remains intact.

## 10. Tolerance provenance

`128*f64::EPSILON` is an inherited local binary64 occupation threshold. It is
frozen in PR #1 to prevent post-hoc widening. It is not derived from endpoint,
`N_eff`, abundance, or multi-step accumulated-error requirements.

`PR-ODE-R1E` must explicitly justify, replace, or compose this local threshold
with a global accumulated-error budget. Such a change requires a versioned
contract amendment; it cannot be smuggled into PR #1.

## 11. Scientific claim adjudication

The following local results remain supportable after a sound closure:

```text
Pauli-box preservation for successfully certified local edge candidates
focused elastic weighted-number identity
focused pair CP/lepton-number-difference identity
focused edge reconstruction structure
```

They do not validate:

```text
continuum collision physics
matrix elements or quadrature convergence
temporal order
plasma energy exchange
weak rates
N_eff
BBN abundances or endpoint
production/public dispatch
QKE
```

Exact detailed balance remains a typed forward blocker until `PR-ODE-R1E`.

## 12. Revised status

```text
R1_ORIGINAL_ZERO_BYPASS_CHANNEL       PASS_FOCUSED
R1_LOCAL_PAULI_BOX_PRESERVATION       PASS_ON_SUCCESS
R1_CERTIFICATE_COMPLETENESS           PASS_FIXTURE_ONLY
R1_CERTIFICATE_SOUNDNESS              FAIL_P0
R1_FALLBACK_ROBUSTNESS                FAIL_DYNAMIC
R1_EQUILIBRIUM_STEP_SEMANTICS         BLOCKED_DEFERRED
R1_FROZEN_ERROR_BUDGET                BLOCKED_DEFERRED
R2_EDGE_RECONSTRUCTION                VALIDATED_FOCUSED
R2C/R3C/R4+                           FORBIDDEN
PR #1                                 REQUEST_CHANGES
```

## 13. Dispatch decision

The first package was not dispatchable. Revision 2 is dispatchable only from the
exact released contract commit named by the handoff and manifest. Codex must
execute `PR1-CLOSURE-A`, then `PR1-CLOSURE-B`, run at most two fresh reviews, and
stop without starting `PR-ODE-R1E`, `R2C`, or `R3C`.
