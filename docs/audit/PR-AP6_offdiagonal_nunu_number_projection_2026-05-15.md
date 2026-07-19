# PR-AP6 Off-Diagonal `nu-nu` Radial Number And Pair-Energy Projection

Date: 2026-05-15

## Scope

This increment extends the AP6 descriptor-driven `pstf_radial` source path for
diagonal no-QKE `nu-nu` processes.  The default provider now projects all nine
ordered `{nue,nuebar,nux}` radial diagonal `nu-nu` rows for particle-number
conservation before the source is mapped into AP18/AP40 thermo or hierarchy
feedback.

Same-bank rows keep the existing number/energy-neutral projection required by
elastic self-scattering.  Off-diagonal rows use a number-neutral projection for
each ordered output species, then complete unordered pairs receive an
energy-neutral closure that removes finite-quadrature total pair-energy
leakage while preserving the relative raw species energy-transfer difference.

## Boundary

This is classical no-QKE collision-source plumbing for the staged AP6/AP18/AP40
surfaces.  It does not add off-diagonal flavour coherence, a promoted public
solver route, production SMC evidence, or a promotion-grade full BBN span.

## Runtime Evidence

Focused RED/GREEN coverage was added for off-diagonal radial `nu-nu` rows:

- `conserve_nunu_number_moments=False` leaves measurable raw number residuals.
- The number-only comparison provider returns six off-diagonal projected rows with
  `process_contract="pstf_nunu_number_neutral_radial_source_v1"`.
- The default provider returns six off-diagonal projected rows with
  `process_contract="pstf_nunu_number_pair_energy_neutral_radial_source_v1"`.
- Projected off-diagonal number moments are below `1e-16`.
- Projected unordered-pair energy sums are below `1e-16`.
- The projected ordered-pair energy difference matches the number-only value
  within the focused test tolerance, so the closure removes only the common
  finite-quadrature pair-energy residual.

A real AP55 LRS source-budget smoke reported:

- `passed=true`
- `n_radial_moment_sources=18`
- `n_radial_nunu_sources=9`
- `n_radial_nunu_number_projected_sources=9`
- `n_radial_offdiagonal_nunu_number_projected_sources=6`
- `n_radial_offdiagonal_nunu_pair_energy_projected_sources=6`
- `n_radial_offdiagonal_nunu_pair_energy_projected_pairs=3`
- `radial_nunu_max_abs_number_moment=4.828087799349512e-20`
- `radial_offdiagonal_nunu_max_abs_number_moment=2.498747194400186e-20`
- `radial_offdiagonal_nunu_pair_max_abs_energy_residual=9.317362419797304e-20`
- `radial_identical_nunu_max_abs_energy_moment=1.3552527156068805e-20`
- `collision_dA_abs_max=1.627849662743716e-05`

## Self-Review

The change keeps identical same-bank source behavior backward-compatible and
keeps the number-only off-diagonal projection available through an explicit
provider flag for focused comparisons.  The default now exposes separate
diagnostics for all-`nu-nu` number projection and off-diagonal unordered-pair
energy closure, so AP55/AP56/AP57 surfaces can distinguish particle-number
conservation from pair-total-energy closure.
