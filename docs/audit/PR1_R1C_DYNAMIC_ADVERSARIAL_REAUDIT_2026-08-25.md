# PR #1 Dynamic Adversarial Re-Audit and Compiled Closure Contract

Date: 2026-08-25 (Asia/Seoul)  
Repository: `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
Pull request: `#1`  
Audited base: `a59531ad9c5f041c65ceed5d1508c4ea30c8dbf5`  
Audited head: `50d3bc5b8093bc33e9311f94505c5ee0711ce51b`  
Disposition: **REQUEST_CHANGES / DO_NOT_MERGE**

## 0. Evidence boundary

This review separates three evidence classes:

1. **Repository source evidence**: the exact `base..head` diff, current Rust source, workflow, PR metadata, and claim documents.
2. **Executed Rust evidence**: GitHub Actions run `32787997031` at `50d3bc5b8093bc33e9311f94505c5ee0711ce51b`. Its `validate` and `native-r1c` jobs passed. The native job executed 21 `pauli_edge_step` tests plus two selected sweep tests; it did not run the complete Rust library suite, `cargo fmt --check`, `cargo clippy -D warnings`, or the two sweep-level B1 regressions.
3. **Independent numerical evidence**: `.codex/audit/pr1-r1c-closure/pr1_r1c_numeric_reproducer.py`, using binary64 control-flow emulation and a 110-digit exact-real affine-state reference. It reproduces a current-head false `Solved`, fallback stagnation, and the exact-detailed-balance hard abort.

The supplied external report is valuable but explicitly static. Its findings N1--N9 are treated as hypotheses until supported by source, execution, or independent arithmetic. Its conclusion that no BLOCKER-class false green exists is overturned by the deterministic P0 fixture below.

## 1. Executive verdict

PR #1 correctly closes the original `initial_flux == 0`/default-report/NaN-swallow channel for the tested cases and removes the midpoint-only completeness failure on the legacy fixture. It also preserves local Pauli-box positivity on every returned candidate.

It does **not** yet provide a sound exact-real root certificate. `certified_flux_evaluation` bounds the products formed from the already-rounded occupations, while `occupations_at_extent` performs an unbounded `extent / measure` and `initial +/- quotient` map before those products. Residual multiplication and subtraction rounding are also outside the reported bound. A power-of-two fixture therefore returns `Solved` although the exact affine-state occupation error is about `124.44` times the advertised `128 eps` ceiling.

```text
P0: 1
P1: 9
P2: 3
merge: REQUEST_CHANGES
R2C/R3C/R4+: FORBIDDEN
```

## 2. P0: current-iterate false `Solved`

### 2.1 Fixture

All values below are exactly representable powers of two.

```text
topology              PairSource
first_measure          2^8       = 256
second_measure         2^-36
A (gain)               2^26 MeV
B (loss)               2^-14 MeV
h                       2^-36 MeV^-1
f1_initial              1 - 2^-40
f2_initial              1 - 2^-6
```

Current-head control flow returns:

```text
outcome                 SOLVED_CURRENT
iteration               3
xi_returned              -0x1.eff0807bfa264p-51
reported root error      4.930380657629558e-30
reported occupation err  3.388131789015988e-19
```

The exact-real affine-state equation

\[
f_1(\xi)=f_{1,0}+\xi/m_1,\qquad
f_2(\xi)=f_{2,0}+\xi/m_2,
\]

\[
r(\xi)=\xi-h\left[A(1-f_1)(1-f_2)-Bf_1f_2\right]
\]

has root

```text
xi_true                 -8.6031777335129861114174...e-16
nearest binary64        -0x1.eff07e8a38d5cp-51
```

The actual error is

```text
|xi_returned-xi_true|/min(m1,m2) = 3.5367629503399001e-12
128 eps                               = 2.8421709430404010e-14
ratio                                  = 124.4387836347543
```

The old returned bits are `0xbcceff0807bfa264`; the independent nearest-root bits are `0xbcceff07e8a38d5c`.

### 2.2 Root cause

The bound currently covers

\[
\left|\operatorname{fl}(\widehat G-\widehat L)-(G-L)\right|
\]

for products at the already-rounded occupations. It omits

```text
extent / measure
initial +/- quotient
1 - occupation
h * flux
extent - step_flux
```

from the exact-real residual enclosure. The missing state-map error dominates `h delta_J` in the fixture. The previous correction added `midpoint_certificate.occupation_error_abs` to the midpoint path, but this P0 travels through the current-iterate path and therefore survives.

The implementation must use an outward-rounded interval from `xi` through the affine state map, Pauli factors, products, flux, and residual. Point binary64 values may remain Newton proposals; only interval evidence may authorize `Solved`.

## 3. Critical assessment of the supplied report

### Adopt

- Hard `UnresolvedFlux` at detailed balance is an audit-phase fail-closed state, not a viable eventual collision operator contract.
- The flux-error term creates a step-dependent certificate-attainability boundary.
- The fallback can halve only every second iteration and can repeat an identical midpoint.
- `flux_mev` is evaluated unnecessarily on the resolved direct-product hot path.
- Numeric capacity-endpoint evaluation is avoidable because Pauli inwardness proves the exact conceptual bracket signs.
- Sweep report guards silently skip anomalous evidence.
- Current CI omits two sweep-level B1 regressions, formatting, and clippy.
- Several tests and the `R3 tangent probe` log overstate what they observe.

### Revise

- `R1_CERTIFICATE_SOUNDNESS = PASS_BOUNDED` is rejected. The bound is bounded only after the state has already been rounded; it does not enclose the exact affine-state residual used by the root theorem.
- The report's approximate conversion from `h delta_J/min(m)` to `eta_J <= about 25` is not a universal theorem. The conversion factor depends on topology, state, and measures. Gate the exact irreducible quantity directly; record `eta_J` separately.
- The claimed correctness consequence of the unconditional log path is mainly latent for finite nonnegative coefficients and factors in `[0,1]`; its measured present impact is performance and misleading error taxonomy. Do not over-promote it to a current physics failure without a reproducer.
- A local `CertifiedFrozen` theorem is valid, but its errors accumulate over two sweeps, many edges, and many outer time steps. No endpoint or scientific authority follows unless a controller consumes an aggregate error budget.

## 4. Additional dynamic findings

### P1 — detailed balance aborts the sweep

For `A=B=1`, `f1=f2=0.5`, and `h=0.25 MeV^-1`, the point is an exact physical detailed-balance state, but direct-product resolution is `UnresolvedForCertificate`, and `implicit_step` returns `UnresolvedFlux`. The existing test deliberately asserts this. Audit honesty is improved; R4+ viability is blocked.

### P1 — fallback stagnates

The independent control-flow reproducer uses

```text
ElasticTransfer
m1=m2=2^-30
A=B=2^-20 MeV
h=1 MeV^-1
f=(1/8,1/4)
```

and reaches an identical midpoint repeatedly near the cap. It exits only after 96 iterations as `UncertainPhysicalBracket`. A repeated-extent detector and typed `StagnatedInterval` failure are required.

### P1 — certificate unattainability is misclassified and detected late

The irreducible term is

\[
E_{occ,irr}=\frac{h\,\delta_J}{\min(m_1,m_2)}.
\]

When this exceeds `128 eps`, root iteration cannot repair it. The current loop can spend its full budget and report a bracket failure. Add `CertificateUnattainableAtStep` and stop when a sign-certified candidate has reducible residual below the irreducible term but the total occupation bound cannot pass.

### P1 — CI evidence is incomplete

The green `native-r1c` job runs local edge tests, the legacy root ladder, and the subnormal denominator test. It does not run:

```text
sweep_report_counts_exact_stationary_without_nan_swallow
unresolved_edge_failure_is_transactional_and_observable
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

A green job therefore does not mechanically imply the complete B1 closure contract.

### P1 — PR evidence provenance is stale

The live PR head is `50d3bc5b8093bc33e9311f94505c5ee0711ce51b`, with two commits and three changed files, while the PR body still names the prior head `5262df6...`, two changed files, and old test counts. A merge gate may not rely on stale prose evidence.

## 5. Scientific and numerical claim audit

### Conventions and dimensions

```text
f                    dimensionless, physical box [0,1]
edge measure         quadrature-weighted dimensionless occupation measure
A, B, J              MeV
h                    MeV^-1
extent               weighted dimensionless occupation
r=extent-hJ          weighted dimensionless occupation
```

The signs and dimensions in the local backward-Euler equation are consistent.

### Positivity

The physical capacity interval and transactional candidate arrays keep returned occupations in `[0,1]`. This is a local algebraic property and survives the audit.

### Discrete invariants

The elastic weighted-number invariant and pair CP-difference invariant remain focused-tested. They validate the edge update algebra, not the event-stream physics, continuum collision integral, weak rates, or endpoint.

### Detailed balance

The current hard abort is scientifically honest but operationally incomplete. It cannot be promoted as a working late-time collision operator. `CertifiedFrozen` is admissible only with local and accumulated error certificates.

### Collision physics

This PR does not independently validate matrix elements, quadrature convergence, finite-electron-mass event assembly, continuum detailed balance, or agreement with the SciPy number-of-record. The direct event action and folded edge action share substantial construction, so their parity is a structural cross-check rather than an independent oracle.

### Temporal accuracy

No temporal consistency or order result exists. The legacy `h=2^-8...2^-30` test measures root closure only. Its printed `R3 tangent probe` label is false. R3C remains forbidden until local certificate soundness and equilibrium semantics close.

### Endpoint and observables

No active `OdeSystem::rhs` path is changed. There is no valid new claim about `N_eff`, weak rates, BBN abundances, endpoint temperature, performance, production support, or public dispatch.

## 6. Revised claim ledger

```text
R1_FAIL_CLOSED_SEMANTICS             PASS_FOCUSED
R1_ORIGINAL_ZERO_BYPASS_CHANNEL      PASS_FOCUSED
R1_CERTIFICATE_COMPLETENESS          PASS_FIXTURE_ONLY
R1_CERTIFICATE_SOUNDNESS             FAIL_P0
R1_LOCAL_PAULI_BOX_PRESERVATION      PASS_ON_SUCCESS
R1_DISCRETE_EDGE_INVARIANTS          PASS_FOCUSED
R1_STIFF_STEP_CERTIFIABILITY         BLOCKED
R1_EQUILIBRIUM_STEP_SEMANTICS        BLOCKED
R1_FALLBACK_ROBUSTNESS               FAIL_DYNAMIC
R2_EDGE_RECONSTRUCTION               VALIDATED_FOCUSED (unchanged)
R2_CONDITIONED_DB_SENSITIVITY        NOT_YET_EVALUATED
R3_OPERATOR_CONSISTENCY              NOT_YET_EVALUATED
R3_TEMPORAL_ORDER                    NOT_YET_EVALUATED
R4-R11                               FORBIDDEN
ACTIVE_DRIVER/ENDPOINT/PUBLIC        NOT_EARNED
```

## 7. Required development sequence

### Amend PR #1 — certificate closure only

1. Add the P0 power-of-two regression and observe RED.
2. Introduce certificate-only outward intervals for the full affine-state-to-residual path.
3. Require interval root and occupation bounds on every `Solved` path.
4. Add `CertificateUnattainableAtStep` and `StagnatedInterval`.
5. Replace numeric capacity-endpoint flux evaluations with an analytic conceptual bracket proof.
6. Move the log/`expm1` value calculation into the unresolved-direct-product arm.
7. Replace silent report skips with hard result invariants; aggregate maximum occupation-error bound.
8. Extend CI and refresh the PR evidence body at the final SHA.
9. Run a fresh-context classification-only review. `P0=0` and `P1=0` are mandatory.

### Separate follow-up PR — equilibrium semantics

Implement `CertifiedFrozen` only when an interval proof gives

\[
\frac{2h\delta_J}{\min(m_1,m_2)}\le128\epsilon.
\]

Record count, maximum local error, sum of local errors, and per-sweep accumulated bound. Temporal audit fixtures require zero frozen edges; equilibrium fixtures require bounded frozen edges. No endpoint claim is allowed until an outer controller consumes the accumulated budget.

### Then

```text
PR-ODE-R2C: conditioned detailed-balance sensitivity
PR-ODE-R3C: rate-scaled representable temporal tests
R4+: remain forbidden until both close
```

## 8. Machine package

```text
.codex/audit/pr1-r1c-closure/AUDIT_COMPILED_PACKAGE.json
  keys:
    p0_p1_threat_catalogue
    audit_compiled_exec_plan
    invariant_test_matrix
    final_independent_audit_contract
    evidence_bundle_schema
    dynamic_evidence
.codex/audit/pr1-r1c-closure/pr1_r1c_numeric_reproducer.py
docs/audit/PR1_R1C_CODEX_HANDOFF_2026-08-25.md
```

The package compilation rule is strict: every P0/P1 threat has an executable detector, mechanical invariant, or typed STOP. Prose-only P0/P1 warnings make the plan invalid.
