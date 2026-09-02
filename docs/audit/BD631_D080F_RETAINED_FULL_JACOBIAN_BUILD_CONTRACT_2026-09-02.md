# BD631 — D-080F retained production-order full-Jacobian build contract

**Date:** 2026-09-02  
**Repository:** `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
**Frozen D-080E head:** `5f2b04c13ea0ca012312634820cb09cc652bbbc8`  
**Comparator Git blob:** `de44feee0aa484abe26976c7dc34c579643005b5`  
**D-080E facade/core blobs:** `0913b3be5ad66af27ae7115deb603c88556cd6b4` / `915196691eb166f5624d413a46d314b32faacfe6`

## 1. Purpose

D-080F must replace the D-080E linear 180-column projection by one actually executed, fixed-state, order-60 construction of

```text
J_static = [ J_c | J_Tgamma | 0_elapsed ]
```

in the exact packed ordering

```text
(c_e[0:60], c_mu[0:60], c_tau[0:60], T_gamma, elapsed_time).
```

The measured matrix has shape `182 x 182`.  Its float64 payload is prospectively fixed to

```text
8 * 182^2 = 264,992 bytes.
```

This is a construction/admission experiment only.  It must not call an ODE integrator, modify the collision equations, widen tolerances, change quadrature, execute a trajectory, or move a scientific gate.

## 2. Frozen physical and numerical state

The only qualifying fixture is recovered byte-for-byte from

```text
source commit:
78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b

path:
.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_1200.npz

SHA-256:
c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380
```

The frozen configuration is

```text
order = 60
y_max = 30
incoming polar order = 4
final polar order = 4
final azimuth order = 4
electron radial order = 24
N = 0.16286930247517223
T_cm = 10 exp(-N) MeV
```

The comparator uses natural units `hbar=c=k_B=1`.  No physical coefficient, event family, support rule, matrix-roundoff rule, state definition, output definition, or failure semantic may change.

## 3. Prepared-state seal

Before matrix construction, the D-080E fixed-state caches must be saturated.  Every reachable numerical array in the prepared state, kinematic batches, weak matrices, modal-basis cache, primal collision action, thermodynamics, grid and state must then be:

1. made read-only, including its ultimate NumPy backing allocation;
2. listed with shape, dtype, bytes and SHA-256;
3. bound to a SHA-256 content fingerprint together with the state/configuration contract;
4. assigned an explicit unique-allocation byte count.

Cache hit counters may increase during construction.  The following quantities must remain unchanged:

- kinematic, self-matrix, electron-matrix and modal-basis miss counts;
- entry counts for the same four cache families;
- estimated cache bytes;
- prepared-array content fingerprint;
- read-only status of every recorded array.

Any fingerprint change, cache miss, cache entry growth or writable array is a fail-closed integrity failure.

## 4. Matrix construction

The 180 spectral basis columns are evaluated through the admitted prepared D-079 JVP.  The admitted D-080C thermal column is appended, followed by an exact-zero elapsed-time input column.

The actual construction timer includes the thermal column and all 180 spectral columns.  It excludes post-build serial-oracle and original-RHS audits so construction cost is not inflated by validation work.

The following six spectral columns are independently recomputed by the frozen non-prepared D-079 serial oracle:

```text
0, 59, 60, 119, 120, 179.
```

Two deterministic mixed spectral/thermal directions must additionally satisfy both:

```text
J v = prepared directional JVP
```

and a centered-difference ladder of the unchanged original packed RHS on the same discrete support branch.

## 5. Prospective gates

### Integrity

```text
matrix shape                    = 182 x 182
elapsed input column norm       = 0 exactly
prepared fingerprint unchanged = true
all prepared arrays read-only   = true
cache miss delta                = 0
cache entry delta               = 0
```

### Same-physics equivalence

```text
max prepared-action residual       <= 5e-10
max selected serial-column residual <= 5e-10
```

The original-RHS mixed-direction witness uses a separate truncation-sensitive gate:

```text
all epsilon samples on same branch = true
best blockwise residual per direction < 6e-3
```

Spectral, photon-temperature and elapsed-output rows are normalized separately because their native dimensions differ.

### Resource route

The route is `EXPLICIT_CALLBACK_CANDIDATE` only if all integrity/equivalence gates pass and

```text
measured construction time <= 900 s
unique cache bytes          <= 2 GiB.
```

If correctness and integrity pass but either resource bound fails, the route is `MATRIX_FREE_OR_SPLIT_CANDIDATE`.  A projection alone can never admit the explicit route.

## 6. Required evidence

The qualifying package must contain:

- the full `182 x 182` float64 matrix;
- the base packed RHS;
- the two mixed probe directions;
- the prepared-array manifest and fingerprint;
- matrix content and file digests;
- measured preparation and construction times;
- explicit cache and process-memory accounting;
- selected serial-column residuals;
- original-RHS directional ladders;
- diagnostic timing, memory, residual and integrity plots;
- a machine receipt and `SHA256SUMS`.

## 7. Literature and symbolic role

Froustey–Pitrou–Volpe provide the closest neutrino-decoupling precedent for direct differential-system Jacobian computation.  Matrix-free chaining and Jacobian-split Newton–Krylov literature motivate the fallback route when dense-block construction is too expensive.  These works motivate the route decision only; they do not validate this private discretization or cache.

The accompanying stateless Wolfram calculation checks state size, matrix bytes, D-080E projection formulas, the idealized batch-reuse expression and the passive elapsed-column Newton determinant identity.  It is formula-level corroboration, not executable build evidence.

## 8. Claim ceiling

A successful D-080F may establish only:

> On one exact retained fixed state and one CI host, the same-physics order-60 square static Jacobian was explicitly built under a sealed immutable prepared-state contract, and its construction resources and equivalence residuals admit either an explicit-callback candidate or a matrix-free/split candidate.

It must not claim:

- a portable build time or speedup;
- a production cache implementation;
- BDF, JFNK or Newton convergence improvement;
- stalled-prefix completion;
- trajectory, endpoint or `N_eff` agreement;
- movement of `G-F10-INDEPENDENT-FLRW`;
- release or publication readiness.
