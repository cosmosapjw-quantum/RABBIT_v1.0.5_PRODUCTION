# SCIENTIFIC_CONTRACT.md

## Scientific objective

Produce a source-bound design packet for later ODE-solver correctness and dominant-work mitigation. This run computes no new physical observable and earns no gate movement.

## Governing definitions

- admitted result: solver success plus expected terminal evidence, finite raw state, residual/bracket evidence where relevant, complete diagnostics, and all declared physical invariants;
- false success: any result consumed or published after one of those conditions fails;
- full-prefix evidence: a run from the physical initial state covering both `N=0.14` and `N=0.22`, reaching `N>=0.25`, within 5,500 full-RHS-equivalent calls and 64,800 seconds;
- static discriminator: an analysis at retained fixture states; never trajectory or endpoint evidence;
- QoI: the declared endpoint abundance or radiation observable whose error is bounded by a primal residual and adjoint/dual weight.

## Conventions

- All physical conventions, units, signs, normalization, frame, gauge, grids, and boundary/initial conditions remain exactly those of the current source and frozen context.
- Independent Python order-60 trajectory state has 182 components; folded Rust order-60 state has 122. They are different implementations and are never substituted.
- Raw states and rejected/failure states are preserved; clipping cannot establish validity.

## Valid regime

- exact repository head `78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`;
- augmented Type-I PSTF no-QKE private programme only;
- retained physical-prefix fixtures may support static discrimination only;
- no restart state near `N≈0.1653` is assumed or reconstructed.

## Required invariants

- finite and dimensionally admissible temperature, density, Hubble argument, and raw state;
- strict occupation/positivity domain where the governing model requires it;
- exact declared collision invariants and conservative moment constraints;
- no fabricated event, missing residual, or unavailable failure snapshot represented as success;
- raw-state validity is checked before derived clipping or postprocessing.

## Known limits

| Limit | Expected result | Tolerance | Reference/test |
|---|---|---:|---|
| invalid raw accepted state | typed failure, no derived observable | exact fail-closed | adversarial focused unit test design |
| event refinement exhaustion | no terminal event, typed exhausted outcome | exact fail-closed | deterministic mutation/property test design |
| equilibrium/null manifold | conserved moments exact; full-grid residual bounded | candidate-specific, predeclared | static discriminator then independent holdout |
| D-071 full prefix | physical start, both checkpoints, `N>=0.25` | `<=5500` calls and `<=64800 s` | separately authorized governing run only |

## Reference cases

- trusted numerical reference: current SciPy/BDF path, subject to its retained fail-closed claim ceiling;
- prior implementation evidence: A-MAC-ADJ2 and the original 34-finding register;
- retained fixture states: static-only, not restartable trajectory authority;
- published theory: supporting hypotheses only, never current RABBIT validation.

## Numerical requirements

- error metrics must be mixed absolute/relative and scale-aware;
- candidate 3 requires a full-grid primal residual, tail control, invariant projection, and adjoint QoI enclosure on independent holdouts;
- no magic tolerance, retry, or single-vector Krylov result is sufficient;
- no random method is currently proposed; any later randomized compression must predeclare seeds and confidence bounds;
- runtime caps are the frozen full-prefix caps above, not segment timings.

## Failure semantics

NaN/Inf, non-convergence, empty result, clipped invalid values, fabricated or exhausted events, silently replaced missing data, unavailable failure snapshots, unverified fallback approximation, invariant violation, or cap violation are not success.

## Change control

No convention, baseline, tolerance, approximation order, output semantics, dependency, public API, or physical scope is changed in this research run. A later implementation requires separate authority.
