# Solver Audit: Type-I PSTF no-QKE BBN Path

## Executive summary

The most likely root cause is not the old A-mode transport issue and not a pure max-step/controller failure. The current stall is best explained by a mismatch between a near-boundary nuclear reaction network and the way the direct-abundance `trace_boundary` policy is embedded inside a Rosenbrock/Rodas stage formulation. The accepted state remains finite and nonnegative, but the internal stages and rejected candidates repeatedly leave the trace-abundance domain, especially around Be7/Li-chain trace species. That is a solver-coordinate and invariant-domain problem, exposed by real nuclear stiffness and made worse by scalar RMS error control.

The best next code change is to stop evolving the constrained trace abundances directly in linear `X` coordinates inside Rodas5P. Replace trace species indices `[3, 4, 5, 6, 7, 8]` with a positivity-preserving internal coordinate, preferably `u_i = log(X_i / X_ref_i)` or a carefully scaled softplus/log coordinate, compute the nuclear RHS in decoded physical `X`, and return `du_i/dN = (dX_i/dN) / X_i` with the Jacobian taken in the transformed state. Acceptance should still be judged in physical abundance units, not raw log units.

If that does not remove the stage/candidate domain-rejection loop, the next tier should be a network-specific positive implicit production-destruction block. Do not first spend effort only on `max_steps`, endpoint chaining, or rejection growth heuristics; those can complete a run but will not fix the mechanism causing invalid stages.

## Ranked diagnosis of the seven hypotheses

### 1. Nuclear network too stiff near trace boundaries

Partly true, but incomplete. The nuclear network is stiff, and the reported maximum `|dX/dN|` of about 370 confirms a hard local scale. However, stiffness alone does not explain why stage 5 can produce Be7-like trace excursions of order `-3.30` when the accepted abundance is around `4.6e-12`. That magnitude points to a bad representation of the invariant domain in the numerical stages, not merely to ordinary stiffness.

### 2. Scalar RMS norm is inappropriate near zero trace abundances

True as a contributor. The current norm

```text
sqrt(mean((err_vec / (atol + rtol * abs(y_new)))^2))
```

is weak in two ways. First, a scalar RMS over the full 62-vector can dilute one or a few bad trace components. Second, with `atol = 1e-10`, the effective scale for Be7/Li7/Li6 can be far larger than the species itself, so physical trace errors are not consistently protected. It also does not control internal-stage domain distance. This should be fixed, but it is probably not the first-order cause of the repeated domain rejection loop.

### 3. `trace_boundary` RHS policy is mathematically inconsistent with Rosenbrock stages

This is the strongest explanation. A Rosenbrock method relies on a locally smooth vector field and linearized stage prediction. The current policy clips trace abundances for flux evaluation and conditionally zeros RHS components when a constrained species is nonpositive and still decreasing. That can make the RHS one-sided and nonsmooth. The stage variables themselves are still linear direct-`X` predictors, so they can cross the boundary even if the RHS evaluated at the boundary points inward. Repeated stage/candidate boundary failures are exactly what this mechanism predicts.

### 4. Frozen-source full-JVP Jacobian missing important derivatives

Likely secondary. Missing derivatives from collision-payload reuse or source freezing can degrade Newton/Rosenbrock predictions, but the failure signature is concentrated in `X_phase2`, not in the A-mode or geometry blocks. More importantly, once the RHS contains a hard `max(X,0)` plus conditional zeroing, the Jacobian is not a clean derivative of a smooth physical vector field anyway. Fix the variable/domain formulation first; then re-evaluate whether the frozen-source Jacobian needs enrichment.

### 5. Reverse flux or log-space flux evaluation creates large effective stiffness near Li/Be/Li6

Possible and worth auditing. For tiny species, a reverse or destruction flux with a large coefficient can create very large effective stiffness even when the abundance is tiny. This could be real physics or a flux-evaluation artifact. The key diagnostic is a per-reaction split into nonnegative production and destruction flows for Li7, Be7, and Li6 at the last accepted state and at the failed trial states. If a destruction term is independent of the species it destroys, or if a log-difference implementation creates catastrophic cancellation, that is a network bug or flux-form bug.

### 6. Rodas5P needs positivity-preserving trace variables or a constrained network block

Yes. This is the implementation direction that addresses the observed mechanism. The least invasive next patch is a positivity-preserving trace-coordinate transform inside the existing full Rodas path. The more robust but larger change is a positive implicit production-destruction network substep.

### 7. Primarily a controller issue

No. Controller improvements can reduce wasted attempts, and increasing `max_steps` might let a staging run finish, but the run is dominated by trace-domain rejections. A controller-only fix would treat the symptom while leaving invalid stage dynamics in place.

## Best next implementation change

### Patch A: log-coordinate trace abundances in the full Rodas state

Use this as the immediate next code change.

Keep the user-facing physical state layout unchanged, but change the internal solver state for constrained trace species:

```text
linear solver state:
  [Sigma_plus, Sigma_minus, T_gamma, T_nu_e, T_nu_x, A_modes_flat,
   X_n, X_p, X_D, u_T, u_He3, u_He4, u_Li7, u_Be7, u_Li6]

where
  X_i = X_ref_i * exp(u_i)
```

Choose `X_ref_i` only for numerical centering; it should not act as an output floor. For example, use a species-specific expected scale or simply initialize with `u_i = log(max(X_i, X_init_floor) / X_ref_i)`. If a species is exactly zero at initialization, encode it using a documented tiny initialization sentinel and log the sentinel status.

The RHS wrapper should decode to physical abundances, evaluate the nuclear network on nonnegative physical `X`, and transform derivatives:

```python
def unpack_solver_state(z):
    geom, A, X_linear, u_trace = split(z)
    X_trace = X_ref_trace * jnp.exp(u_trace)
    X = assemble_X(X_linear, X_trace)
    return geom, A, X, X_trace


def rhs_solver_state(N, z, payload):
    geom, A, X, X_trace = unpack_solver_state(z)

    dgeom, dA, dX = rhs_physical(N, geom, A, X, payload,
                                  trace_policy="none_or_assert")

    dX_linear = dX[linear_species]
    du_trace = dX[trace_species] / X_trace

    return pack(dgeom, dA, dX_linear, du_trace)
```

Then compute the Rodas/JVP Jacobian in the transformed state, not by differentiating the old direct-`X` RHS with a boundary clamp:

```python
J_z = jax.jvp_or_jacfwd(rhs_solver_state, z)
W = I / (gamma * h) - J_z
```

This removes the discontinuous `max(X, 0)` branch from normal stage evaluation for the transformed species. For these species, stage-domain rejection should no longer be needed because physical decoded `X` cannot be negative. Keep NaN/Inf checks and keep domain checks for any still-linear abundance species.

### Error norm must be computed in physical variables

Do not use the raw log-coordinate error directly as though it were an abundance error. Convert the embedded error estimate back to a first-order physical abundance error:

```python
err_X_trace = X_trace_new * err_u_trace
```

Then use a block-aware acceptance norm:

```text
err_norm = max(
    rms_scaled(geometry_thermo),
    rms_scaled(A_modes),
    rms_scaled(X_major),
    rms_scaled(X_trace_physical)
)
```

Use species-aware absolute tolerances for trace species. Avoid both extremes: `1e-10` is too coarse for Li/Be-scale diagnostics, but forcing pure relative `1e-8` down to `1e-30` will cause artificial h-collapse. A reasonable policy is:

```text
scale_i = atol_X_i + rtol_X_i * max(abs(X_i_new), X_active_floor_i)
```

where `X_active_floor_i` prevents inactive species from imposing meaningless relative accuracy, while `atol_X_i` is small enough to resolve the diagnostic species of interest.

### Keep the old direct-X path only as a shadow diagnostic

To preserve the evidence, do not delete the raw negative information from the diagnostic stream. Instead:

1. Store the raw solver coordinate candidates and stages, now `u` for transformed trace species.
2. Store decoded physical stage/candidate `X` before acceptance.
3. Store a debug-only `legacy_direct_X_trial` computed by the old direct-X predictor for the same step when `debug_trace_shadow=True`.
4. Store flux-level production/destruction contributions for each rejected or high-stress attempt.
5. Make final output explicit: `physical_X_final = decode(z_final)`, `positivity_enforced_by_coordinate=True`, `no_output_truncation=True`.

This does not hide negative final outputs because the evolution variable itself is positive by construction. It is different from clipping an invalid accepted `X` after the fact.

## Flux-evaluation cleanup to do with Patch A

The trace transform will prevent negative stages, but it will not fix a malformed nuclear flux. Add these checks around Li7, Be7, and Li6:

```text
for each reaction r:
  log F_forward_r
  log F_reverse_r
  signed net flow or separate nonnegative directional flows
  contribution to each dX_i
  contribution to P_i and D_i * X_i
```

The network should be expressible as

```text
dX_i/dN = P_i(X, T) - D_i(X, T) * X_i
```

with `P_i >= 0` and `D_i >= 0` for a production-destruction split. If a destruction contribution for a trace species is not proportional to that species, or if subtracting nearly equal forward/reverse fluxes produces sign noise at tiny `X`, fix the flux representation. Prefer separate directional nonnegative reaction extents over a single unstable `forward - reverse` value in trace regimes.

## If Patch A still stalls: constrained implicit network block

If log-coordinate Rodas still reaches max steps because `du/dN` is too stiff, move the nuclear phase-2 update into a positive implicit production-destruction solve. Hold geometry/temperature/A-mode quantities fixed or extrapolated over the nuclear substep, and solve either in log variables or with a Patankar-like update.

A simple first version is backward-Euler in log variables:

```text
Find u_{n+1} such that
  exp(u_{n+1}) - X_n - h * F(exp(u_{n+1}), T, rates) = 0.
```

Use Newton or damped Newton with a residual check. Because `X = exp(u)`, trial abundances remain positive. For better conservation, implement a flux-wise production-destruction / modified Patankar form where nonnegative reaction extents are limited by available reactants and baryon/charge conservation residuals are explicitly monitored.

## Controller changes: useful, but after the domain fix

After Patch A, change the controller to avoid wasting attempts:

```text
h_new = safety * h * err_norm^(-1 / order_controller)
```

with standard lower and upper factors, a stronger shrink after domain/nonfinite failures, and a less restrictive post-rejection growth cap if the next accepted step is clean. Endpoint chaining and increasing `max_steps` are acceptable staging conveniences, but they should not be advertised as solving the root cause.

## Focused test plan

### Test 1: local failed-step replay

Start from the last accepted state near `T_gamma ≈ 0.0799021 MeV`. Replay only one step with the old direct-X Rodas path and with the transformed trace-coordinate path. Use the same `h` values that caused failures: `0.1`, `0.0458`, `0.0439`, and `0.00879`. The expected result is that the transformed path has no negative decoded trace stages, while the old path reproduces the stage/candidate violations.

### Test 2: network-only frozen-background integration

Freeze `T_gamma`, `H_rate`, weak rates, and collision payload. Integrate only the 9D `X_phase2` network over the same `N` interval. Compare:

1. direct-X Rodas with `trace_boundary`,
2. direct-X Rodas without boundary clipping but with logging,
3. log-coordinate Rodas,
4. positive implicit production-destruction substep.

If only direct-X fails, the problem is solver policy. If log-coordinate and positive implicit methods both reveal enormous physical stiffness or inconsistent flux signs, the problem is in the network/flux physics.

### Test 3: flux decomposition at stress points

At each rejected attempt, log for Li7, Be7, and Li6:

```text
P_i, D_i, P_i / X_i, D_i, dX_i, dlogX_i,
reaction-wise forward/reverse extents,
finite-difference derivative ∂dX_i/∂X_i,
JVP derivative used by Rodas.
```

A large negative direct-X candidate with moderate `D_i` suggests solver-stage mismatch. A huge `D_i` or `P_i / X_i` suggests real stiffness requiring a network-specific implicit block. A mismatch between finite-difference and JVP derivatives suggests Jacobian construction or nonsmooth policy error.

### Test 4: Jacobian audit with and without boundary policy

At the last accepted state, compare JAX JVP columns against finite differences for the trace species. Do this in four modes:

```text
physical direct-X RHS, no boundary
physical direct-X RHS, trace_boundary
transformed log RHS
transformed log RHS with payload frozen/rebuilt
```

The direct-X boundary mode will likely show nondifferentiability near zero. The transformed log RHS should be much cleaner.

### Test 5: convergence and conservation

Run a short segment and a full span with `(rtol, atol)` varied by factors of 10, and with `h_max = 0.1, 0.05, 0.02`. Track:

```text
final X values,
baryon/charge residuals,
number of rejected attempts,
minimum decoded stage X,
maximum |du/dN|,
flux decomposition maxima.
```

A successful policy fix should reduce trace-domain rejections to zero for transformed species and make final abundances converge under step refinement.

## BBN-code precedent

The relevant precedent is not that standard BBN codes use exactly this transform. It is that mature BBN and nucleosynthesis codes treat abundance evolution as a stiff ODE/network problem and use adaptive or implicit numerical machinery rather than allowing invalid negative abundances to be silently truncated. PArthENoPE solves coupled ODEs from nuclear statistical equilibrium to asymptotic abundances and later versions use ODEPACK-based infrastructure. PRIMAT computes the time evolution of light-element abundances through freeze-out. AlterBBN documents adaptive step comparison with a full step and two half-steps. More broadly, nucleosynthesis reaction networks are well known to be extraordinarily stiff, and implicit methods are standard in nuclear astrophysics.

## Bottom line

Implement the trace-log coordinate patch first, with a physical-space block error norm and full transformed-state JVP. Keep the negative direct-X evidence as a shadow diagnostic, not as an accepted output. If the transformed path still h-collapses, escalate to a positive implicit production-destruction network substep and audit Li7/Be7/Li6 reverse-flux decomposition.
