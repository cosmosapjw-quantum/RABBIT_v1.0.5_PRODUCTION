# BD634 — D-081R0 Rust-first transition, offline closure, and semantic inventory

## Verdict

`RUST_ENVIRONMENT_CLOSED / EXACT_ORACLE_FROZEN / STATIC_CROSS_LANGUAGE_PARITY_NOT_YET_RUN`

D-081R0 closes two prerequisites for the Rust-first continuation of the F10 numerical-equivalence lane:

1. an exact Rust 1.94.1 + `Cargo.lock` offline build/test environment for `native/rabbit_cpu`;
2. an exact Python source–input–output oracle bundle for the D-080F retained order-60 state.

It also establishes that the existing folded Rust `IsotropicBoltzmannFlrwSystem` is not semantically identical to the D-080F Python comparator. The next implementation must therefore add a separate exact `F10ComparatorSystem`, not disguise the legacy two-bank system with an adapter.

No production physics source, solver source, tolerance, collision catalogue, endpoint, gate, or claim registry is changed by D-081R0.

## Frozen lineage

- D-080F source head: `901a62350b19cf43c17dffe45e96e8b94e4c7ca1`
- D-080F tree: `06e7168ae706734c74b592fb08db37d7eda97eb9`
- Rust `Cargo.toml` blob: `30bc4826c7a7676b5fb5c8610273eb7586551ad8`
- Rust `Cargo.lock` blob: `a1b5035da5c20712d1a2a4ab077da255ff94a014`
- Rust toolchain file blob: `b1dd491cdce6f8b98cfe02f6807980c0dfe48784`
- private Python comparator blob: `de44feee0aa484abe26976c7dc34c579643005b5`
- retained state source commit: `78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`
- retained state SHA-256: `c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380`

## Rust installer and signature status

The supplied standalone archive passed its compressed-stream integrity check and installed the exact requested toolchain:

- archive SHA-256: `294b3d81fa72e62581276290c60c81eb8b58498d333d422ca1dfc432877d0c40`
- detached-signature SHA-256: `942bc6926af6a2130d70e77933e3df09d39beeedafad77710259e6df7eadee08`
- `rustc 1.94.1`, commit `e408947bfd200af42db322daf0fadfe7e26d3bd1`
- `cargo 1.94.1`, commit `29ea6fb6a5db279426f4cc4e17aa385f05a0cfbc`
- `rustfmt 1.8.0-stable`
- `clippy 0.1.94`

The detached signature packet identifies key ID `85AB96E6FA1BE5FE`, but cryptographic verification remains `BLOCKED_MISSING_PUBLIC_KEY`. File integrity and toolchain identity pass; publisher-signature authentication is not claimed.

## Existing offline-vendor coverage

The RABBIT lock contains 174 registry packages. Existing generic and BASS vendor assets together covered 141 exact name/version/checksum triples and missed 33, including load-bearing packages such as `diffsol 0.16.0`, `nalgebra-sparse 0.12.0`, `numpy 0.28.0`, `pyo3 0.28.3`, and `serde_json 1.0.150`.

Therefore no existing vendor was silently merged or promoted. An exact RABBIT-specific vendor was generated from the frozen lock.

## Exact offline closure

GitHub Actions workflow `33609470252` completed `SUCCESS` at head `dbd7a0cd497c10a9b7c4c63f2b24e0d9c1f30e07`.

The generated kit contains:

- all 174 registry packages in the exact lock;
- portable `.cargo/config.toml` replacing crates.io with `vendor`;
- repository-relative network JSON and FLRW gold inputs required by `include_str!`/tests;
- compiled test artifacts;
- metadata, test log, receipt, and per-file hashes.

Offline commands that passed:

```text
cargo metadata --locked --offline
cargo check --release --locked --offline
cargo test --locked --lib --offline --no-run
9 focused ODE/Jacobian/collision test invocations
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked --offline -- -D warnings
```

Focused aggregate: `64 passed / 0 failed`. Long endpoint tests were deliberately not executed in this environment-only node.

Artifacts:

- GitHub artifact ID: `9838506413`
- outer artifact ZIP SHA-256: `391b45bbf6446dc9144bc39321723eda18240d7c8751f45887f9492ffc66f78a`
- inner deterministic kit SHA-256: `431416dc2eee37a7361a4d96e98bf4782534b165fcde116dbe8090593d290371`
- verified manifest entries: `11631`

A local clean recompilation attempt was terminated by the execution-time ceiling before completion and is classified `LOCAL_REPLAY_TIMEOUT`, not PASS or scientific failure. Independently, the artifact's compiled test binary was executed locally for the same nine focused selectors: `64 passed / 0 failed`.

## Exact Python source–input–output oracle

Workflow `33609886995` completed `SUCCESS` at head `2c7704d4fb9586e698b29c66c6db594cb75ce7cd`.

The bundle contains:

- private primal comparator source;
- trajectory and D-079/D-080 derivative sources;
- exact retained `state_1200.npz` input;
- D-080F base RHS;
- D-080F full `182 x 182` Jacobian;
- probe directions and result receipt;
- machine-readable oracle manifest.

Artifacts:

- GitHub artifact ID: `9838533536`
- outer artifact ZIP SHA-256: `d40444ca68e2d709248551f4b888cd8bfb4e7a5811070d1dbb48fddf2a3a4c56`
- inner deterministic tar SHA-256: `d8e618d2d0f37163c6fbdbad57ee9655918ad4a22803f88aea9f32df0ba5f0e2`
- internal manifest entries: `53`, all verified
- matrix file SHA-256: `3594e861cc800bbfa2feedc4ec90f7de1f69e283a450484bdb600345caa0d4ae`
- matrix-content SHA-256: `714b83107c9302b55cd09a9d2c7d2dc260a5b7d1e9289265e6a844766ed388f8`
- base-RHS SHA-256: `36437e28968836023406f260b9b125ea29e992ee4357673bba85f9ed8ae6d2ea`

## Semantic gap: why a new Rust system is required

The exact Python contract is:

```text
state size: 182
ordering: c_e[0:60], c_mu[0:60], c_tau[0:60], T_gamma, elapsed_time
chart: strict-open complementary-log-log
radial grid: affine Gauss-Legendre order 60 on [0,30]
flavour banks: 3
T_gamma column: analytic fixed-support derivative
```

The existing Rust `IsotropicBoltzmannFlrwSystem` contract is:

```text
state size on selected grid: 98
ordering: T_gamma, elapsed_seconds, u_e[0:48], u_heavy[0:48]
chart: pointwise logit
radial grid: exponential positive-half-line Gauss-Legendre order 48
flavour banks: 2 (electron + folded heavy)
T_gamma column: centered finite difference
```

The heavy-bank fold is exact only on the `mu=tau` subspace. Wolfram verifies that the `mu-tau` antisymmetric mode maps to zero and cannot be recovered. Consequently, an adapter cannot certify the full 182-state oracle.

The existing `ode.rs`, `diffsol 0.16.0` BDF/Rodas5P substrate, exact point cache, counters, `rhs`, `jacobian`, and `rhs_and_jacobian` interfaces are reusable. The legacy folded physical system remains valuable but is not the exact F10 target.

## SciSpace and Wolfram findings

SciSpace retrieval supports the methodology: direct differential-system Jacobians in primordial-neutrino decoupling, full-collision preservation, and separate numerical-convergence accounting. It does not validate RABBIT's implementation.

Wolfram returned exact zero residuals for the passive elapsed-state block, determinant factorisation of `I-gamma J`, permutation similarity, action linearity, and cloglog-to-logit identity. It also proves the noninvertibility of the two-bank flavour fold outside `mu=tau`.

## Preserved failures

- source export run `33606614611`: archive path error;
- source export run `33606716827`: successful upload, but local audit detected a self-referential `SHA256SUMS` entry;
- source export run `33609635450`: wrong D-080A source filename;
- exact-vendor run `33607763376`: standalone crate layout omitted repository-relative JSON authority;
- broad full-test vendor run `33608220800`: superseded because it mixed environment closure with potentially long endpoint tests;
- two local clean compile attempts: timed out before a replay receipt was produced.

None of these failures was promoted to scientific evidence. Each was repaired by narrowing packaging/orchestration without changing physics or acceptance thresholds.

## Revised DAG

```text
D-080F Python oracle                           DONE
    ├── D-081R0A source/input/output freeze    DONE
    ├── D-081R0B exact offline Rust closure    DONE
    └── D-081R0C semantic-gap inventory        DONE
                    ↓
D-081R1 exact Rust F10ComparatorSystem
    ├── exact 182-state layout/chart/grid
    ├── primal RHS parity
    ├── selected Jacobian-column parity
    ├── mixed/holdout Jv parity
    ├── moment/first-law/support/failure parity
    └── no solver call
                    ↓
D-081R2 integrate admitted system with existing OdeSystem/diffsol
                    ↓
D-081R3 Rust paired stiff-prefix discriminator
    A: same Rust RHS + finite-difference Jacobian adapter
    B: same Rust RHS + analytic Jacobian
                    ↓
D-081R4 original-initial-state stalled-phase run
                    ↓
endpoint/holdouts/independent audit/gate reconsideration
```

If the explicit dense Jacobian is correct but economically poor in D-081R3, only then may the quarantined matrix-free/JFNK assets be considered.

## Claim ceiling

D-081R0 establishes a reproducible Rust environment and an exact cross-language oracle package. It does **not** establish Rust/Python semantic parity, a Rust full-collision F10 implementation, solver acceleration, trajectory completion, endpoint agreement, `N_eff`, or movement of `G-F10-INDEPENDENT-FLRW`.
