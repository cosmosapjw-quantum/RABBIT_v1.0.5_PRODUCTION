# R2/R3 temporal-blocker re-audit request

## Request

Please independently assess this branch as a negative-result / blocker-evidence
packet.  It is **not** a claim that the Rust reconstruction is production-ready,
that the active ODE driver has been repaired, or that an endpoint is validated.

The audit question is narrow:

> Does the R3 failure at the required small-step tangent ladder demonstrate a
> genuine incompatibility between the current certified local Pauli root solve
> and the reconstructed edge sweep, and are the recorded R1/R2 claims stated at
> an appropriate ceiling?

No remediation is included in this branch.

## Exact ancestry and scope

| Item | Value |
|---|---|
| Clean base | `78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b` |
| R1 commit | `fc2de6c98d9ae261370f017abcf7b91f2e29f9a4` |
| R2 integration commit | `486df8497a4364168e8d59c0fde11161f1700193` |
| This audit branch | `external-audit/ode-r2-r3-temporal-blocker-20260824` |
| Runtime promotion | None; the active `OdeSystem::rhs` path is not switched by this branch |
| External writes before this request | None from this run |

Committed files beyond the clean base are intentionally limited to:

- R1's five Rust files for a certified local Pauli implicit root;
- R2's `native/rabbit_cpu/src/electron_spectral.rs`;
- this audit document; and
- the R3 diagnostic additions in `electron_spectral.rs`.

No clipping, projection, tolerance widening, equilibrium shortcut, security
control, endpoint run, or R4--R11 implementation is included.

## Claim audit

| Claim | Status | Evidence | Risk | Required audit focus |
|---|---|---|---|---|
| The local Pauli root no longer returns a midpoint after exhausting its iteration budget. | VALIDATED | R1 focused root tests and `fc2de6c`. | Local primitive only; no composed solver authority. | Check certificate semantics and tail scaling. |
| R2 removes bitwise FD anchor forcing from the reconstructed electron-edge action and zero-step is its only bypass. | IMPLEMENTED | `486df849`, source inspection, focused tests. | Not yet active ODE-driver integration. | Confirm all anchor/zero-fill paths are absent. |
| The raw reconstructed FD edge bank satisfies the R2 detailed-balance gate. | VALIDATED | Exact R2 test output below. | 4-node focused quadrature, not endpoint authority. | Check gain/loss normalization and edge accounting. |
| The R3 sweep satisfies the full tangent/order contract. | FORBIDDEN | Required ladder fails at `h=2^-14`. | Continuing would convert a failed root into an unsupported temporal claim. | Verify reproduction and typed stop. |
| R4--R11 or endpoint conclusions follow. | FORBIDDEN | DAG stop rule after R3 failure. | Any claim would be unsupported. | Confirm no downstream work is represented as done. |

## R1 and R2 evidence carried forward

R1 changed five files by `+636/-2` and introduced a root residual plus
occupation-width certificate.  Its focused checks passed:

```text
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib pauli_edge_step::tests -- --nocapture
cargo test --lib electron_event_falsifiers::frozen -- --nocapture
```

R2 changed only `electron_spectral.rs` by `+553/-54`.  Its focused
checks passed:

```text
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib electron_spectral::tests::unforced_fd_equilibrium_is_an_event_and_edge_null -- --exact --nocapture
cargo test --lib electron_spectral::tests::unforced_equilibrium_jacobian_is_continuous -- --exact --nocapture
cargo test --lib electron_spectral::tests::folded_edges_reconstruct_action_at_five_independent_states -- --exact --nocapture
cargo test --lib electron_spectral::tests::folded_edges_are_boundary_inward_at_five_independent_states -- --exact --nocapture
cargo test --lib electron_spectral::tests::invalid_inputs_fail_without_clipping -- --exact --nocapture
```

At `T_gamma=T_cm=1 MeV`, grid order 4, event rule 4/3, unforced FD
occupations gave:

```text
direct action L1                 = 1.53265705691756586e-39
reconstructed action L1          = 9.15371976919204087e-38
edge net L1                      = 5.15450231997948723e-37
edge traffic L1                  = 2.78142564489345787e-22
normalized detailed balance      = 1.85318717019916229e-15
maximum individual edge ratio    = 7.10542735760107917e-15
```

Those values pass the frozen `1e-12` gates.  The five deterministic
reconstruction states were FD; alternating 0.91/1.07; alternating 0.73/1.11;
a (10^{-35}/10^{-40}) last-node tail; and fixed literal pseudo-random
occupations.  They are focused operator checks, not a continuum convergence
or physical endpoint validation.

## R3 raw failure record

The only R3 source addition is certificate aggregation in
`PauliSweepReport` and this exact test:

```text
cargo test --lib electron_spectral::tests::pauli_sweep_tangent_converges_to_unforced_action -- --exact --nocapture
```

State and rules:

```text
grid order 4; electron rule 4/3
T_gamma = 1.15 MeV; T_cm = 1.0 MeV
deterministic alternating 0.91/1.07 FD occupations
forward-half then reverse-half production sweep
```

Observed output:

```text
h=2^-8:
  applications=60, maximum_edge_iterations=91
  maximum_root_residual_ratio=2.83911429515499858e-14
  maximum_occupation_bracket_width=2.17930312030189462e-40

h=2^-10:
  applications=60, maximum_edge_iterations=93
  maximum_root_residual_ratio=2.83911429515499858e-14
  maximum_occupation_bracket_width=5.44825780075473655e-41

h=2^-12:
  applications=60, maximum_edge_iterations=95
  maximum_root_residual_ratio=2.83911429515499858e-14
  maximum_occupation_bracket_width=1.36206445018868414e-41

h=2^-14:
  error: Pauli edge implicit root did not converge
```

The R3 acceptance ladder requires all four (h) values before tangent-error,
step-doubling, observed-order, and edge-order checks can be accepted.
Therefore the branch status is:

```text
BLOCKED_TEMPORAL_CONSISTENCY
```

This is consistent with R1's design: a non-certified root is returned as an
error instead of being converted to a midpoint success.  No attempt was made
to relax the root certificate, clip a tail, or replace the test state.

## Re-audit checklist

1. Reproduce the exact R3 command and verify the three successful prefix
   measurements plus the `h=2^-14` root failure.
2. Inspect whether the R1 certificate's residual and occupation-width
   requirements are appropriate for the extremely small tail measures.
3. Check that R2 truly has no exact-equilibrium force branch and that
   `action_values` evaluates every stored edge.
4. Check whether R2's detailed-balance statistic uses the intended gain/loss
   traffic normalization.
5. Confirm that this branch makes no claim beyond local R1/R2 validation and
   correctly prohibits R4--R11 continuation.

## Deliberately skipped

No additional tests, endpoints, performance runs, package builds, PR creation,
tagging, release, or solver repair were performed while assembling this
handoff.  Exact token-use counters are unavailable in the active harness;
they are reported as `UNAVAILABLE`, not estimated.

