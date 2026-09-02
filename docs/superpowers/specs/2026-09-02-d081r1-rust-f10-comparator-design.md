# D-081R1 exact Rust F10 comparator design

## Status

`APPROVED_ARCHITECTURE / IMPLEMENTATION_NOT_STARTED`

## Goal

Add a separate Rust `F10ComparatorSystem` that reproduces the frozen D-080F Python comparator's 182-state static RHS and Jacobian contract before any trajectory or performance experiment.

## Chosen approach

Create a new exact system and reuse only low-level Rust primitives behind explicit parity gates. Do not adapt `IsotropicBoltzmannFlrwSystem` into the target contract: its state ordering, logit chart, positive-half-line grid, two-bank flavour fold, and finite-difference thermal column are materially different.

Rejected alternatives:

1. **Legacy-system adapter:** cannot recover the `mu-tau` antisymmetric mode and changes the radial discretisation.
2. **Python callback from Rust:** gives no independent production implementation and makes Python the runtime authority.
3. **Immediate solver integration:** confounds semantic porting with BDF behaviour and performance.

## Architecture

### `f10_comparator.rs`

Owns the exact state contract:

```rust
pub(crate) struct F10Grid {
    pub(crate) order: usize,
    pub(crate) y_max: f64,
    pub(crate) nodes: Vec<f64>,
    pub(crate) weights: Vec<f64>,
}

pub(crate) struct F10ComparatorSystem {
    grid: F10Grid,
    t_start_mev: f64,
}

pub(crate) struct F10Diagnostics {
    pub(crate) q_nu_mev5: f64,
    pub(crate) q_em_mev5: f64,
    pub(crate) number_moments: [f64; 3],
    pub(crate) energy_moments: [f64; 3],
    pub(crate) support_signature: Vec<u64>,
}
```

State ordering is fixed:

```text
c_e[0:n], c_mu[0:n], c_tau[0:n], T_gamma, elapsed_time
```

`dimension() = 3*n + 2`. `T_cm = t_start_mev * exp(-N)` is explicit time dependence, not a state coordinate.

### Chart

Use the strict complementary-log-log map

```text
f = -expm1(-exp(c))
df/dc = exp(c-exp(c))
```

No clipping, projection, floor, or repair is allowed. Saturation or nonfinite values return typed failure.

### Grid

Use `quadrature::gauss_legendre_rule(order)` followed by the exact affine map

```text
y = 0.5*y_max*(x+1)
w = 0.5*y_max*w_x
```

The first gate compares Rust nodes and weights with the frozen Python oracle. The existing exponential positive-half-line grid is not used.

### Physics assembly

Preserve three independent flavour banks. Low-level event, kinematic, matrix-element, and quadrature primitives may be reused only if an unfurled six-species test proves catalogue, sign, multiplicity, finite-electron-mass, support, and normalisation parity. The folded two-bank output is never used as the exact oracle surface.

### Derivatives

Implement:

```rust
fn rhs_static(&self, n: f64, y: &[f64], out: &mut [f64]) -> Result<F10Diagnostics, F10Error>;
fn jvp_static(&self, n: f64, y: &[f64], v: &[f64], out: &mut [f64]) -> Result<(), F10Error>;
fn jacobian_static(&self, n: f64, y: &[f64], out_row_major: &mut [f64]) -> Result<(), F10Error>;
fn rhs_and_jacobian_static(...) -> Result<F10Diagnostics, F10Error>;
```

The elapsed input column is exact zero. The analytic `T_gamma` column follows D-080A/B/C and must not use finite differences.

### Solver boundary

D-081R1 must not call `ode::solve`. After static admission, D-081R2 implements `OdeSystem` by delegating to the admitted methods. Existing `ode.rs` and pinned `diffsol 0.16.0` remain unchanged unless telemetry needs a separately reviewed extension.

## Fixture design

Commit a compact JSON fixture generated from the exact oracle artifact. It contains:

- source and artifact hashes;
- `N`, order, `y_max`, nodes, weights, packed state;
- base RHS;
- selected columns `0,59,60,119,120,179,180,181`;
- two original and two holdout directions and `Jv` outputs;
- moment, first-law, support, and failure expectations.

The full 182×182 matrix remains in the external oracle artifact and is not duplicated in Git.

## Admission gates

1. exact state size/order and strict-domain failure semantics;
2. grid nodes/weights within prospectively frozen ULP/relative bounds;
3. RHS blockwise scaled residual;
4. selected-column residual, including analytic `T_gamma` and exact-zero elapsed columns;
5. contribution-scaled action residual on four directions;
6. number/energy moment and differentiated first-law residuals;
7. support signature identity;
8. bitwise repeatability on one host;
9. designated mutations killed: chart sign, flavour swap, heavy fold, grid map, Pauli sign, multiplicity, omitted channel, thermal-column fallback, nonzero elapsed column.

No threshold may be changed after Rust output is inspected. A failure routes to semantic repair, not tolerance widening.

## Error model

```rust
pub(crate) enum F10Error {
    InvalidDimension,
    NonFiniteInput,
    OccupationOutsideStrictOpenInterval,
    GridMismatch,
    SupportBranchChanged,
    CollisionFailure(&'static str),
    NonFiniteOutput,
}
```

Raw input remains observable on failure. No failed state is clipped or replaced.

## Evidence and claims

D-081R1 may claim only static cross-language semantic admission. It may not claim runtime speed, BDF convergence, trajectory completion, endpoint agreement, `N_eff`, or gate movement.
