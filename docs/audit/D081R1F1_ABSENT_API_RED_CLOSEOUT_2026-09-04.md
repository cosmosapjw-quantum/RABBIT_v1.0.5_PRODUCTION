# D-081R1F1 absent `T_gamma` JVP API RED closeout

Date: 2026-09-04  
Repository: `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
Branch: `research/d081r1f1-rust-tgamma-jvp-20260904`  
Stacked base: `research/d081r1f0-rust-c-only-jvp-20260904`  
Base commit/tree: `c29b6d26599da0bead66482373d8ec2cdc8f06c4` / `f467ea43e78611cef47342d1416e2b372e90d6ac`

## Classification

```text
PASS_EXPECTED_ABSENT_TGAMMA_JVP_API_RED
```

This result establishes only that the production Rust `T_gamma` packed-RHS JVP
API specified by the frozen contract was absent before implementation.  It is
a TDD RED result, not a failed implementation and not numerical admission.

## Exact authorities

```text
contract:
docs/audit/D081R1F1_RUST_TGAMMA_PACKED_RHS_JVP_CONTRACT_2026-09-04.md

contract blob:
58eeaf38b9f4edd4c60a01d22d2e101a33b71812

RED trigger commit/tree:
7f31ff4f16a8928196b0c0711150aabb6260a36e
314f2d20a07209326a9bc2bd06fa794ed9540191

RED workflow run:
33850384502

receipt publication commit/tree:
179b60f3e1d6a23922ab8edd6a423fd6620bf125
b6316dd2b095c01b4a32ab4d3a6cbee262fb85d5

receipt:
docs/audit/artifacts/d081r1f1/absent_api_red_receipt.json

receipt blob:
5ac1aced2f7417bb317ebb1267dcce8d014172c7

compiler log SHA-256:
460e46f82683e90d0c7e80d1c151e50fc8ae78105b3f07b03ac7af08f225a72a
```

## Executed RED

The runner used Rust/Cargo 1.94.1 and the exact 174-package offline vendor.  A
temporary import of

```rust
crate::f10_packed_rhs_tgamma_jvp::evaluate_f10_packed_rhs_tgamma_jvp
```

produced the preregistered compiler diagnostic

```text
error[E0432]: unresolved import `crate::f10_packed_rhs_tgamma_jvp`
```

The temporary `native/rabbit_cpu/src/lib.rs` mutation was restored byte for
byte and no production Rust source was committed.  The ordinary PR harness on
the human trigger head also passed its aggregate validator, harness regression
suite, generated-artifact readback, Rust formatting/release/Clippy checks, and
Pauli edge/sweep regressions in run `33850388281`.

## Holdout firewall

During RED:

```text
state_2000 opened:             false
holdout oracle generated:     false
production source committed:  false
```

The preregistered `state_2000.npz` remains sealed until a terminal retained
`state_1200` calibration receipt exists for the implemented thermal column.
`state_3000.npz` remains reserved for the later full-Jacobian node.

## Next admissible node: R1F1-P0

P0 may add only independently testable primitive tangents:

1. QED-off electromagnetic EOS derivatives
   `chi_gamma`, `p_em,T`, and `chi_gamma,T`;
2. moving half-line electron quadrature derivatives
   `dp_i/dT_gamma = p_i/T_gamma` and `dw_i/dT_gamma = w_i/T_gamma`;
3. finite-mass electron energy and velocity derivatives;
4. elastic-event center-of-momentum boost, outgoing momentum, Kallen/phase-space,
   Lorentz-invariant matrix-element, and energy-transfer tangents;
5. outgoing-neutrino interpolation-coordinate and mapped modal-basis tangents;
6. explicit support/correction branch signatures, with branch changes classified
   as `NONDIFFERENTIABLE_DISCRETE_EVENT` rather than projected or tolerated.

P0 must not yet expose the final packed-RHS API or open either retained state.
Every primitive requires analytic-vs-centered-difference tests on fixed
branches, dimensional checks, exact-zero directions, and at least one mutation
that the correct derivative kills.

## Claim ceiling

No production thermal JVP, collision-column parity, packed-RHS column,
retained-state result, full Jacobian, solver callback, trajectory, endpoint,
`N_eff`, performance, release, merge, publication, or
`G-F10-INDEPENDENT-FLRW` movement follows from this closeout.  PR #25 remains
Draft.