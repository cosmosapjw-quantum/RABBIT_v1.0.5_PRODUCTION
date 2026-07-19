## Verdict

The blocker diagnosis is correct at the category level: the current q14 dynamic collision failure is a dynamic collision-source/linearization problem, not primarily weak-rate history and not primarily the split phase-2 BBN corrector. But I would sharpen the diagnosis: this is not merely “the collision Jacobian is incomplete.” In the attached code, the PSTF radial collision source appears to be produced in occupation-distribution mode space and then injected directly as `dA_modes` in the logit-like `A_modes` variable. That is a concrete sign/variable-space consistency problem unless there is an unstated contract that `C_modes` has already been converted to `dA/dN`. The attached code does not show that conversion; in fact the angular collision path does perform such a conversion explicitly, while the radial path does not.

So the next PR should not be another readiness wrapper and should not start by stabilizing the wrong ODE. First repair or prove the radial `C_modes -> dA_modes` coordinate conversion. Then add a source-response Jacobian for the corrected source.

## Conventions I used for the audit

I am reading `q` as dimensionless momentum/temperature variable, `N` as e-fold time, and the live `A_modes` as the additive distribution-logit modes used to reconstruct a Fermi-Dirac-like occupation. The code itself confirms this in the analytic A-mode Jacobian: it reconstructs `logit = nodal_A + q_nodes` and uses `df/dA = -f(1-f)` in `src/rabbit/validation/augmented_continuous_ap65_rhs.py:11331-11361`. Therefore a collision source in occupation space, (C_f = df/dN), cannot be added directly as (dA/dN); projected locally it should satisfy roughly

[
\frac{dA}{dN} \sim -,\frac{C_f}{f(1-f)}
]

with the actual implementation using the same angular projection/inversion as the existing angular bridge.

## 1. Blocker diagnosis

Yes: the q14/q15 dynamic Laguerre non-LRS failure is primarily in the dynamic collision path, not the weak-rate history and not the split phase-2 network corrector.

The strongest evidence is the contrast between these runs. The strict isotropic equal-temperature full-BBN diagnostic reaches the endpoint with `physical_full_bbn_span_ready=true`, `T_final_MeV=0.00914138906397148`, `Yp=0.16412585886748188`, and `Sigma_H≈4.64e-15` in `isotropic_sigma0_equalT_full_bbn_N4p8_max512.json`. The non-LRS no-collision angular-grid rows also reach the full endpoint; they fail angular-resolution readiness because `Sigma_H` differs by `0.012155211646054737`, not because the host/runtime collapses. By contrast, `bd185_q14_collision_N4p8_cache_bound_probe.json` fails with dynamic collision active after 344 payload build attempts, 343 selected host steps, 342 full-JVP Jacobian evaluations, 549 LU factorizations, and 4392 linear solves.

The phase-2 corrector does not look like the row killer in the current artifact. In the q14 collision probe, the phase-2 conservative corrector remains finite, has Newton residual maxima around `1e-11`, records zero raw Newton-trial negative counts, and the failed `[2.8,4.8]` subspan’s phase-2 reference replay reaches `N≈3.908`. The visible failure is host h-collapse: the accepted step samples decay to `2e-12`, rejected samples also hit `2e-12`, and the terminal source exception is `RuntimeError: continuous AP65 host Rodas5P prototype reached h_min.` That is a stiff/source-linearization failure signature, not a network nonfinite signature.

One correction to the current narrative: the BD185 terminal trace does not show the collision source dominated by the highest q node. Its `collision_dA_abs_by_q_fraction` is about `0.396`, `0.308`, `0.159` in the first three q bins, and the highest-q fraction is about `6.46e-13`. High-q Laguerre behavior may still amplify other rows, but for this current trace the immediate collision `dA` budget is low-q concentrated.

## 2. Dynamic collision algebra, dimensions, and signs

The q raw/energy weight split is conceptually right. `_q_laguerre_weights_from_energy` computes raw Laguerre weights as

```python
q_energy_weights / (exp(q) * q**3)
```

in `src/rabbit/jax/augmented_typeI_replay.py:633-640`, and `_build_live_source_grid_jax` carries both `q_energy_weights` and `q_laguerre_weights` into the live source grid at `src/rabbit/jax/augmented_typeI_replay.py:663-720`. The combined source path repeats the same inversion when external q grids are supplied in `src/rabbit/transport/augmented_typeI_weak_network.py:3630-3633`. The radial collision builder then uses `p1_weights=weights * exp(q)` at `src/rabbit/transport/augmented_collision_bridge.py:2367-2372`, which is the expected plain-quadrature conversion if `weights` are raw Gauss-Laguerre weights. Moment integrals also multiply by `exp(q)` through `_laguerre_plain_weights` in `src/rabbit/transport/augmented_collision_bridge.py:2622-2638`.

The p4 interpolation is mechanically consistent. The scalar interpolation handles the upper endpoint explicitly in `src/rabbit/collisions/pstf_contractions.py:1581-1594`; the vectorized grid computes `p4_energies = e1 + e2 - e3`, marks valid entries inside the p4 grid, linearly interpolates p4 momenta, and zeros invalid entries in `src/rabbit/collisions/pstf_contractions.py:1762-1769`. I do not see an obvious p4 off-by-one or endpoint sign error in the attached implementation.

The `dQ_*_N` bookkeeping is mostly coherent for the `standard_3t_plasma` normalization. The dynamic source computes `H_rate_s` from `hubble_3T * _MEV_TO_S` before building the source in `src/rabbit/jax/augmented_typeI_replay.py:1203-1216`. The standard 3T energy closure converts a per-time thermodynamic target into a per-e-fold target by dividing by `H_MeV` in `src/rabbit/transport/augmented_collision_bridge.py:1837-1844`. Those `dQ_nue_pair_N` and `dQ_nux_bank_N` terms are then passed into the live 3T RHS when collision is applied in `src/rabbit/jax/augmented_typeI_replay.py:1853-1860`. I would not interpret `raw` radial energy-normalization rows as calibrated 3T thermodynamic rows, but the q14 collision artifact appears to auto-resolve the active collision row to `standard_3t_plasma`.

The serious inconsistency is the radial source projection into `dA_modes`.

The radial provider reconstructs the live distribution and projects it to occupation-number modes:

```python
f_nodal = reconstruct_distribution(A, q, basis)
F_by_species_modes_q = project_nodal_to_modes(f_nodal, basis, weights)
```

at `src/rabbit/transport/augmented_collision_bridge.py:1600-1605`. The PSTF contraction contract says the `F*_modes` are scalar occupation-number multipoles and returns the mode-space source for the fermion gain-loss polynomial in `src/rabbit/collisions/pstf_contractions.py:2359-2364`. The process wrapper stores that as `C_modes` directly from the contraction in `src/rabbit/collisions/pstf_process_catalog.py:839-863`.

But `_radial_moments_to_dA_modes` then does this:

```python
C_modes = np.asarray(moment.source.C_modes, dtype=float)
dA[label_to_index[species]] += np.moveaxis(C_modes, 0, 1)
```

at `src/rabbit/transport/augmented_collision_bridge.py:1797-1800`.

That is not the same operation the angular bridge uses. The angular bridge explicitly calls `distribution_rhs_to_augmented_rhs(df_nodal, f_nodal, basis_matrix, angular_weights, ...)` in `src/rabbit/transport/augmented_collision_bridge.py:3185-3192` and again in `src/rabbit/transport/augmented_collision_bridge.py:3622-3628`. The radial bridge skips that conversion.

Given the live RHS adds `collision.dA_modes` directly to the transport `dA` at `src/rabbit/jax/augmented_typeI_replay.py:1912-1918`, this is likely a real sign/coordinate bug. If the radial `C_modes` are (df/dN), then direct injection as (dA/dN) has the wrong sign under the code’s own logit convention. For example, if a collision term wants to reduce an overoccupied mode, (df/dN<0); since (df/dA=-f(1-f)), the corresponding (dA/dN) should be positive, not negative.

So my answer to question 2 is: q weights, p4 interpolation, and the H conversion for standard 3T energy closure look internally consistent; the radial source-to-`dA_modes` mapping does not.

## 3. Host Jacobian

Yes, the host Jacobian is missing a structurally important derivative. But fix the source-coordinate issue first, otherwise you will build a better Jacobian for the wrong source.

The current full-JVP policies differentiate the live JAX RHS at a frozen collision payload. `_frozen_source_jax_jvp_jacobian` passes `collision.dQ_nue_pair_N`, `collision.dQ_nux_bank_N`, and `collision.dA_modes` as constants into the compiled JVP in `src/rabbit/validation/augmented_continuous_ap65_rhs.py:11129-11168`. The policy metadata explicitly says `dynamic_ap65_collision_payload_derivative_wrt_current_state` is ignored for `frozen_source_jax` and `frozen_source_jax_full_jvp` in `src/rabbit/validation/augmented_continuous_ap65_rhs.py:10393-10469`.

The optional collision diagonal Jacobian is only a damping heuristic. It computes `diag = dA/A` only where `A*dA < 0`, clips it into `[-cap,0]`, and puts it on the A diagonal in `src/rabbit/validation/augmented_continuous_ap65_rhs.py:11414-11461`. The policy description itself admits it misses growth-sign and off-diagonal derivatives in `src/rabbit/validation/augmented_continuous_ap65_rhs.py:10470-10499`.

A real source-response Jacobian needs at least these blocks:

[
\frac{\partial, dA_{\rm coll}}{\partial A},\quad
\frac{\partial, dQ_{\rm coll}}{\partial A},\quad
\frac{\partial, dA_{\rm coll}}{\partial T_\gamma,T_{\nu_e},T_{\nu_x}},\quad
\frac{\partial, dQ_{\rm coll}}{\partial T_\gamma,T_{\nu_e},T_{\nu_x}}.
]

The first block is the urgent one because `dA_modes` is the term being added directly to the stiff A transport RHS. The temperature blocks matter too because the current JVP includes the heat-capacity response of the 3T RHS to frozen `dQ`, but not the collision-source response of `dQ` itself.

A local-q/species/mode block Jacobian is a reasonable pilot, but it must be treated as a hypothesis, not as guaranteed structure. The radial collision source is not truly q-local: each p1 row integrates over p2 and p3, and p4 interpolation couples to the p4 grid. So I would build a q4/q5 brute-force finite-difference source-response reference first, then compare a cheap approximation against it. If the local-q approximation misses most of the source-response norm, move to a denser A-block response or a low-rank/global species-mode approximation. A naive full finite-difference A-block at q14 would be too expensive: the A block is roughly `4 * 3 * 14 = 168` columns, and one dynamic payload build is already about `0.09–0.11 s` in the BD185 artifact.

## 4. Fixed three-mode non-LRS representation

The fixed three-mode diagonal non-LRS live representation is adequate for staging the runtime blocker. It is not adequate for final physical interpretation of collision-on anisotropic results.

The attached docs and code are consistent on this point. The live non-LRS path uses `A_modes` with shape `(species, 3, N_q)` and fixed diagonal plus/minus modes, not a generic live ell/m hierarchy. The runtime dataclass enforces `dA_modes` shape `(n_species, 3, N_q)` in `src/rabbit/transport/augmented_nonlrs_transport.py:121-146`. The generic angular decomposition module explicitly says it defines contracts and “does not wire a solver path” in `src/rabbit/transport/angular_decomposition.py:1-8`.

So do not block the source-coordinate/Jacobian fix on a generic ell/m full-BBN implementation. That would be premature. But also do not interpret a successful collision-on fixed-three-mode endpoint as an ell-converged non-LRS result. The right wording is “fixed diagonal three-mode S2-projection staging result.” After it runs, then implement the real non-LRS mode/ell ladder and angular-grid convergence.

## 5. Python-native performance limit

This is not primarily a Python-native limit in the current evidence.

The q14 collision artifact reports about `137.7 s` selected wall time and about `33.8 s` dynamic collision payload build wall time. That means collision payload construction is significant, roughly a quarter of selected wall time, but not the whole failure. The solver also did 342 full-JVP Jacobian evaluations, 549 LU factorizations, 4392 linear solves, 4190 source evaluations, and then collapsed to `h≈2e-12`. That is a numerical stiffness/source-linearization signature with a real Python/NumPy cost center attached, not a generic “Python cannot do this” result.

The cost split I see is:

Runtime metadata and JSON retention: this was a real earlier problem, but the current artifact now writes successfully. Keep compact summaries by default; do not put full source arrays into the hot row payload.

NumPy/Python collision source construction: still significant. The source factory/radial-grid cache work helped, but radial PSTF contraction and source-response evaluation are the likely kernels worth compiling later.

JAX JVP and LU cost: large. Any naive full finite-difference source-response Jacobian would multiply the source-build cost badly. This argues for a carefully bounded source-response approximation or a compiled source-response kernel, not for more evidence wrappers.

True numerical stiffness: yes, currently primary. h-collapse at `2e-12` after a finite trace is the operational blocker.

A targeted compiled kernel is justified after the source-coordinate fix and after profiling the corrected source-response path. The target should be the PSTF radial contraction/source-response path: p4 interpolation, kernel contraction, and perhaps the A-block source-response operator. I do not see evidence supporting a full rewrite away from CPU-JAX/Rodas5P.

## 6. What is overengineered

The evidence discipline is useful, but the code surface has grown beyond what is needed to move the runtime physics blocker.

The two biggest modules are too large and too mixed: `augmented_continuous_ap65_rhs.py` is about 16.8k lines, and `augmented_continuous_ap65_full_bbn_span_ladder.py` is about 9.4k lines. They mix solver mechanics, physics RHS, network correction, artifact schema, policy auto-resolution, and claim-boundary language. That is now slowing the physics fix.

What I would freeze or delete now:

Freeze all nonessential readiness, manifest, and promotion gates for the next collision PR. Keep private diagnostic scope and raw state telemetry, but do not add another standalone readiness/hash/figure gate.

Freeze the policy matrix. For the next PR, vary only source-coordinate conversion and one source-response Jacobian approximation. Do not add more h ladder, metadata, or claim-language axes as the main change.

Split runtime payloads from audit payloads. Runtime collision payloads should carry arrays, compact counters, and fingerprints. Artifact writers can expand summaries afterward. `_dynamic_collision_source_payload_from_restart_state_np` already has `json_safe_output`; the design should lean harder into a runtime-native path.

Rename the live non-LRS path. Call it fixed diagonal three-mode S2 projection. Do not let “ell” language leak into artifacts for that path.

Keep raw negative abundance/Yp telemetry. Do not clip candidate states. That part is not overengineering; it is necessary evidence hygiene.

What I would split out:

A `phase2_network_corrector.py` module for the BE/BDF2/Newton Y-solve.

A `dynamic_collision_runtime.py` or similar module for source construction, compact payloads, and source-response Jacobians.

A `rodas_host_step.py` module for host stepping and Jacobian policies.

An artifact summarizer that consumes traces after the run, instead of shaping runtime objects around audit schemas.

## 7. Exact next PR

I would implement this PR:

**BD187: Repair radial collision source coordinate conversion and add a minimal corrected-source response audit.**

Do not make the first change a pure Jacobian PR. The attached code suggests the source being linearized is already in the wrong coordinate.

### Files to touch

In `src/rabbit/transport/augmented_collision_bridge.py`:

Replace the direct radial mapping in `_radial_moments_to_dA_modes`. Accumulate radial `C_modes` first as an occupation-space source, e.g. `C_F_modes[species, mode, q]`. Apply number/energy corrections to that occupation-space source, not directly to `dA_modes`. Then reconstruct the nodal occupation source and convert it using the same conversion contract as the angular bridge:

```python
f_nodal = reconstruct_distribution(A_modes, q_nodes, basis_matrix)
df_dN_nodal = np.einsum("smq,ma->sqa", C_F_modes, basis_matrix)
dA_modes = distribution_rhs_to_augmented_rhs(
    df_dN_nodal,
    f_nodal,
    basis_matrix,
    angular_weights,
    eps_f=eps_f,
)
```

The exact call signature may need local adjustment, but the key point is: radial PSTF `C_modes` should be treated as `df/dN` until the explicit conversion to `dA/dN`.

In `src/rabbit/transport/augmented_typeI_weak_network.py`:

Keep the combined source using the PSTF radial source as the effective source if that is still the intended no-double-counting policy, but update diagnostics to distinguish `radial_distribution_source_abs_max` from `radial_dA_abs_max`. Right now `combined_effective_dA_abs_max` is only meaningful if the radial source is already in A-space.

In `src/rabbit/validation/augmented_continuous_ap65_rhs.py`:

Add only a small source-response reference hook, not a full new policy matrix. For tiny q, compute finite-difference source response of corrected `dA_modes` with respect to A and compare it to the old damping diagonal. This can be test-only or an opt-in diagnostic. The production solver policy can still be added next, after the corrected source is verified.

In `tests/test_augmented_continuous_ap65_rhs.py` and `tests/test_jax_augmented_typeI_replay.py` or a new focused collision test:

Add tests that prove the sign and coordinate conversion.

### Focused tests that would falsify the PR

First, a unit sign test. Construct a one-species/one-mode or small S2 case with known `f_nodal` and a positive occupation source `df/dN`. Assert that the produced `dA/dN` has the opposite sign with magnitude scaled like `1/[f(1-f)]` in the monopole limit. This test should fail under the current direct `dA=C_modes` mapping.

Second, a radial-vs-angular conversion consistency test. Feed the same synthetic `df_dN_nodal` through the angular bridge conversion and through the repaired radial mode path. The resulting `dA_modes` should agree within projection tolerance.

Third, an energy-closure test. For `standard_3t_plasma`, verify that the occupation-space source moments match the target `dQ_nue_pair_N` and `dQ_nux_bank_N` after correction. Do not compute those moments by integrating `dA_modes` as if it were `df/dN`.

Fourth, a tiny q source-response test. At q4 or q5, finite-difference the corrected dynamic source with respect to all A columns and verify that the response has nonzero off-diagonal/cross-mode structure that the current damping diagonal misses. This test should not require full q14.

### Runtime artifact to run after the PR

Run one short corrected-source collision probe before the full endpoint:

`q14`, dynamic collision on, restart/subspan `[2.8, 3.0]` or `[2.8, 3.2]`, same CPU-JAX/Rodas5P target, same split phase-2 corrector, raw candidates preserved.

The PR is falsified if any of these happen:

The corrected radial source fails the sign/coordinate unit tests.

The source-response finite-difference norm is dominated by off-diagonal q/species/mode blocks while the proposed approximation ignores them.

The q14 short subspan still collapses immediately to `h≈2e-12` with the same rejection pattern.

The corrected source produces nonfinite A/T/X states before the host solver failure, which would move the blocker from missing linearization to source normalization or representation.

If the corrected source passes the unit tests but q14 still h-collapses, the next PR should be the actual solver PR: a corrected-source finite-difference/JVP source-response Jacobian, starting from a q4/q5 full FD reference and then a bounded q14 approximation. But the coordinate repair has to come first; otherwise the solver work can make the wrong dynamics look numerically stable.
