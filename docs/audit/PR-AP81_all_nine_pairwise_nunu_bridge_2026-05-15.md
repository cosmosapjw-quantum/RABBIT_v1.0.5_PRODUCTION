# PR-AP81 All-Nine Pairwise Diagonal Nu-Nu Bridge

Date: 2026-05-15

## Scope

This pass wires the staged NumPy AP19/AP33/AP35/AP41 diagonal no-QKE `nu-nu`
source bridge to the same all-nine ordered `{nue,nuebar,nux}` pair coverage
already present in the supported AP6 process catalog.  The default pairwise
route now includes identical-bank self-scattering channels with the descriptor
Fierz factor `2`; each identical-bank source is projected to be number- and
energy-neutral before it is accumulated into the bank source.  The
`include_identical_nunu=False` switch retains the older six off-diagonal
pairwise path for comparison.

## Claim Boundary

This is still a staged no-QKE collision source.  It does not add oscillation
density matrices, a public forward-solver dispatch, production SMC evidence,
promotion-grade full-span collision-coupled BBN validation, or exact angular
delta-function convergence at publication tolerances.

## Evidence

- The first focused test run failed because the default LRS diagonal `nu-nu`
  source still reported `pairwise_evaluations == 6`.
- The same red run also failed because `include_identical_nunu=False` was not
  accepted by the source factory.
- Focused tests now lock default `pairwise_evaluations == 9`, identical-pair
  diagnostics equal to `3`, off-diagonal diagnostics equal to `6`, and the
  off-diagonal-only legacy switch.
- The same tests lock identical-bank number/energy projection diagnostics with
  post-projection residuals below the smoke-grid numerical tolerance.
- Angular and combined source diagnostics now surface the same all-nine pair
  coverage so AP35/AP41/AP65 wrapper artifacts can record the physical pair
  accounting that was actually used.
