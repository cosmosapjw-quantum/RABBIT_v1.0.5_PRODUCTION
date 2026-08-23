# NEGATIVE_RESULTS

## N-001

ATTEMPT: A read-only finite-tilt numerical comparison using the default network and `ConservativeFrequencyLiouville.from_network(network, line)`.

WHY_IT_WAS_TRIED: To obtain a cheap executable discriminator for the two incompatible direction-frame interpretations.

RESULT: The construction stopped before evaluating physics with `ValueError("reference line outer faces do not match the network")`.

WHY_IT_FAILED: The scratch `LineBoundaryConfig` was not constructed from the locked network's exact outer-face manifest.

WHAT_IT_RULES_OUT: Nothing about the frame hypothesis; it only rules out treating this failed scratch setup as scientific evidence.

WHETHER_TO_REVISIT: Yes, only in a later implementation/test task that builds the line configuration from the canonical network manifest. It was not retried in this loop because the same-failure retry budget was already exhausted.

## N-002

ATTEMPT: Use Pontzen--Challinor as an exact unrestricted Bianchi radiative-transfer hierarchy and Wainwright--Hsu as support for all tilted/class-B backgrounds.

WHY_IT_WAS_TRIED: These were strong literature premises in the seed record.

RESULT: Claim audit narrowed the former to a near-FRW/small-anisotropy hierarchy and the latter to orthogonal class A. The prior author/scope attribution was overbroad.

WHY_IT_FAILED: Citation scope and approximation regime did not match the seed wording.

WHAT_IT_RULES_OUT: Publication-strength claims of an already sourced exact nonlinear hierarchy or one universal normalized Bianchi chart.

WHETHER_TO_REVISIT: Only with a primary derivation matching the exact matter, tilt and symmetry class; absence is retained as a blocker.

## N-003

ATTEMPT: Treat flux-mortar/Schur literature as direct validation of the proposed native/COM E1C cure.

WHY_IT_WAS_TRIED: It supplies an attractive one-interface-flux algebra on nonmatching grids.

RESULT: The primary proof is for Darcy/Stokes-type saddle-point systems, not the Ly-alpha advection/diffusion/recoil/Bose operator.

WHY_IT_FAILED: The physical weak flux, positivity, detailed balance, source ownership and source-identical elimination hypotheses have not been proved for this operator.

WHAT_IT_RULES_OUT: Promoting a mortar witness or Schur reconstruction as implementation/physics evidence.

WHETHER_TO_REVISIT: Yes, after the E1C weak residual is specified; then prove or test the needed inf-sup/positivity/conservation properties directly.

## N-004

ATTEMPT: Require a universally positive scalar-cell remap that exactly preserves both photon number and energy on any target grid.

WHY_IT_WAS_TRIED: It was a strong form of the moving-grid seed.

RESULT: Convexity gives a countercondition: with nonnegative target masses, `E/N` must lie within the convex hull of target representative energies; local bounds/stencils narrow feasibility further.

WHY_IT_FAILED: One scalar mass per target cell does not supply arbitrary independent local number and energy moments.

WHAT_IT_RULES_OUT: Clipping or a generic limiter being labeled a universal two-moment conservative cure.

WHETHER_TO_REVISIT: Only by expanding support, adding moment DOFs, or accepting a fail-closed `REMAP_INFEASIBLE` outcome.

## N-005

ATTEMPT: Interpret accepted history as sufficient to make the whole recombination/radiation problem a finite-dimensional index-one DAE.

WHY_IT_WAS_TRIED: It would allow one standard DAE solver contract to cover all state.

RESULT: Primary delay-equation analysis and the redshift query show that the state contains a history segment. Only each method-of-steps local solve may be a conditional DAE.

WHY_IT_FAILED: A retarded functional state cannot be replaced by the current point state without a finite exact augmentation, which is not supplied.

WHAT_IT_RULES_OUT: A global finite-index claim based solely on the local mass-matrix rank.

WHETHER_TO_REVISIT: Only if an exact finite closure for every active delay channel is derived.

## N-006

ATTEMPT: Claim that two separately exposed physical-operator and numerical-policy digests are mathematically necessary for fail-closed replay.

WHY_IT_WAS_TRIED: Separate digests improve audit localization and were a strong seed recommendation.

RESULT: The independent gate constructed a complete domain-separated aggregate commitment that detects every mutation visible to the two-digest design.

WHY_IT_FAILED: Completeness and unambiguous domain separation are necessary; the number of exposed digests is a representation/auditability choice.

WHAT_IT_RULES_OUT: Using mutation tests alone to prove dual-digest necessity.

WHETHER_TO_REVISIT: Retain dual digests as an operational recommendation and compare audit localization, not correctness, against a complete tagged aggregate.

## N-007

ATTEMPT: The final reviewer's first bounded Python probe invocation.

WHY_IT_WAS_TRIED: Independently reproduce the finite-tilt mismatch and BII chart algebra.

RESULT: It failed before source import with `ModuleNotFoundError` because `PYTHONPATH=src` was absent; the corrected bounded invocation succeeded.

WHY_IT_FAILED: Environment invocation, not source or mathematics.

WHAT_IT_RULES_OUT: Treating the first failed invocation as a physics result.

WHETHER_TO_REVISIT: No; the corrected invocation supplied the diagnostic receipt and no repository file changed.
