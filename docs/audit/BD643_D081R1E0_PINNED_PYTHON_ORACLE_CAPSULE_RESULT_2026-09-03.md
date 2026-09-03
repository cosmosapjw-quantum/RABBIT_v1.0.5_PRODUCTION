# BD643 D-081R1E0 — pinned Python oracle capsule result

**Status:** `PASS_WITH_PINNED_PYTHON_ORACLE_CAPSULE_SCOPE`

**Claim ceiling:** `FROZEN_RETAINED_ORDER60_PYTHON_PACKED_RHS_ORACLE_ONLY`

## Scope

This node freezes one retained order-60 Python packed-RHS oracle and determines
whether its bytes can be reproduced on heterogeneous GitHub-hosted x86-64
runners after the numerical execution capsule is fully pinned. It does not
implement or admit a Rust RHS, JVP, Jacobian, ODE solver, trajectory, endpoint,
performance result, `N_eff`, or movement of `G-F10-INDEPENDENT-FLRW`.

## Frozen authorities

```text
D-081R1D4 final head:
002086662bf2e553c78f4b247868cb1fd9e43f21

D-081R1D4 final tree:
d01ae7c0d3d9fbe8ce9513d054b835d3596f1de2

private Python comparator Git blob:
de44feee0aa484abe26976c7dc34c579643005b5

packed-RHS trajectory-core Git blob:
465a73f0ce40f7149bebdc2d67103f388e2344d9

retained-state source commit:
78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b

retained-state path:
.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_1200.npz

retained-state SHA-256:
c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380
```

The state has order 60, `y_max = 30`, and packed size 182 in the ordering
`(c_e[0:60], c_mu[0:60], c_tau[0:60], T_gamma, t)`.

## Canonical fixture

```text
fixture path:
native/rabbit_cpu/tests/fixtures/d081r1/retained_packed_rhs_case.json

fixture Git blob:
c06e021b3c9c0ceff0f90536eb76dd0e73dfd4c4

fixture SHA-256:
ce2a64114d85c490f4cd1ba055a7a5331e85dc224a956c15a897621a0476330a
```

The generator evaluates the unchanged trajectory-core RHS and independently
reconstructs the same algebra. Those two paths are bitwise equal inside every
admitted run. The validator checks the strict-open cloglog chart, exact state
layout, component addition, particle/antiparticle half-sums, positive
thermodynamics and Hubble rate, first-law closure, and deterministic replay.

## Diagnostic sequence

### 1. Unpinned heterogeneous runners

Identical source, package versions, retained state, grid nodes and grid weights
produced different occupation/chain low bits and different collision and packed
RHS arrays on heterogeneous hosts. The first divergent layer was the chart.
This was classified as cross-host floating-operator divergence, not a change of
fixed physical input.

### 2. NumPy SIMD baseline only

The environment

```text
NPY_DISABLE_CPU_FEATURES=X86_V3,X86_V4
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

made the chart bytes identical, but Intel and AMD runners still produced two
collision/RHS hash families. This isolated the remaining divergence downstream
of the chart and upstream of the packed RHS.

### 3. NumPy SIMD baseline plus pinned OpenBLAS core

The final aggregate probe added

```text
OPENBLAS_CORETYPE=Haswell
```

and ran four independent replicas on both Intel and AMD hosted runners. Every
replica generated the fixture twice, passed the fixture validator, and uploaded
a compact hash receipt. The aggregate comparison found a single hash family for
the entire fixture, every selected array, and every selected RHS scalar.

```text
workflow run:
33765534480

validated probe head:
6511b40be553ce4cd4fe14e42051601d92562898

validated probe tree:
62b76952c683a2b724d1d6cfc3cbf8e921b38673

replicas:
4 / 4 successful

observed CPU families:
GenuineIntel — Intel Xeon Platinum 8573C
AuthenticAMD  — AMD EPYC 9V74

classification:
BYTE_REPRODUCIBLE_IN_PINNED_NUMPY_OPENBLAS_CAPSULE
```

The aggregate artifact is `9897356367`, with archive digest
`sha256:8e602b3ecb3c78401225780b1df4f7657e5d8a043aee15362425006e39765255`.

## Pinned execution capsule

```text
CPython                    3.12.3
NumPy                      2.4.4
SciPy                      1.17.1
NPY_DISABLE_CPU_FEATURES   X86_V3,X86_V4
OPENBLAS_CORETYPE          Haswell
OPENBLAS_NUM_THREADS       1
OMP_NUM_THREADS            1
MKL_NUM_THREADS            1
```

This capsule is the byte-replay authority for the Python oracle. The result
shows empirically that pinning the OpenBLAS target was sufficient to remove the
observed Intel/AMD split in the tested hosted runners. It does not prove
bitwise identity on arbitrary operating systems, compilers, BLAS builds, or
non-x86 architectures.

## Selected canonical hashes

```text
packed_state                 623985fb3393bfa6d7eb4c78e7a5c359d14621146fc3f020956c4685eab54cc8
grid_nodes                  5429623566911d43c71625eb7e14722bd6108d07894800e0a0ca06cc99558c1f
grid_weights                dd8480b3591b9bf84e85936fbed411529e3dd4676bc856ba341f2c07c276bf06
occupation                  87336f6cbd54791bf01d89561e5166c7c0ec7cc9c1fc0d3312cadebb92ae14aa
cloglog_chain               426c19c56f61026761ec35df925f10977d0e97b7784a15542c185e577f5f621f
self_native                 99af5519073ac615ffe03578f995177a93e5424a014fa36b020a44661e69dc2a
electron_native             929e3a1e20e5ed72c911df9ecdde4dff3db8a0a1f6404d113636e3d3d2c3bb5c
total_native                77c801d3e314a1376d8466860d6e804a03e1282c2ee49468bf5c9d943d7d6bf3
self_modal                  d2753761f7223f55574b85589b051359f2a2381c6938c76a0ea633657d4686be
electron_modal              c8e5a0a74364b53ea472eb4516b525c074ca51ade54a109012070057df1ee1de
total_modal                 7bcb567f7c62ff39ca6dad324b7c3acdbdfda7da7a5d8e828e1929255142de43
pair_rate                   34c3df1952759eb6b7cd0daac6c83572db5ece1e81d8abe57b72c13e9eb9d3de
spectral_rhs                e5b8511d65cca6f5029d194abb8e7f817c58e90008b1a9a35e5efe65bccc3708
packed_rhs_trajectory_core  027d9b31471e66de24d735d724dc73e7b06b3b66958b5a9f5d4bb2bfab0029fc
```

Selected binary64 scalars were also identical in every replica:

```text
temperature RHS  c020fe25574d3bf6
elapsed RHS      43faf23f28d97593
Q_nu             3bfc050e1efc2fa0
Q_em             bbfc050e1efc2fa0
```

## Scientific interpretation

The diagnostic supports the following narrow causal statement:

> In the tested Python/NumPy/SciPy wheels on heterogeneous GitHub x86-64
> runners, fixing NumPy ufunc dispatch was insufficient, while additionally
> forcing the OpenBLAS `Haswell` core restored full byte identity of the
> retained collision and packed-RHS oracle.

The test does not establish that OpenBLAS is the only possible source of
cross-host divergence. It establishes that the pinned capsule is sufficient
for the canonical replay required by the next Rust parity node.

This is consistent with numerical-reproducibility literature: non-associative
reductions can change with architecture-specific BLAS kernels, while pinned or
accurately rounded reductions can restore deterministic results. The literature
does not authorize the RABBIT event catalogue, thresholds, fixture bytes, or
physics claims; those remain repository-specific authorities.

## Next admissible node

D-081R1E now proceeds on the dedicated implementation branch with:

1. exact order-60 grid byte identity;
2. strict 182-state and cloglog contract;
3. independent Rust thermodynamic reconstruction;
4. admitted D4 collision action to packed RHS;
5. passive elapsed-time, covariance, mutation, and transactional-failure gates;
6. exact-offline Rust 1.94.1 admission.

## Non-claims

This result does not admit:

- Rust collision or packed-RHS parity;
- an analytic Rust JVP or Jacobian;
- `diffsol` integration;
- trajectory progress, convergence, or speedup;
- endpoint or `N_eff` agreement;
- release or publication readiness;
- movement of `G-F10-INDEPENDENT-FLRW`.
