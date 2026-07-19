# PR-N1 — Non-LRS Characteristic Primitives

## Scope

PR-N1 adds the pure non-LRS S^2 primitives needed for generic
orthogonal Type I characteristic transport, without wiring them into a
driver.

## Delivered

- [`src/rabbit/jax/characteristic_rays_nonlrs_jax.py`](../src/rabbit/jax/characteristic_rays_nonlrs_jax.py)
  with:
  - tensor-product `setup_ray_grid_S2`
  - exact commuting-stretch forward map `mu_phi_current_jax`
  - `extract_stress_plus_S2`
  - `extract_stress_minus_S2`
  - `extract_monopole_S2`
- [`tests/test_pr_n1_nonlrs_primitives.py`](../tests/test_pr_n1_nonlrs_primitives.py)
  with 9 regression locks
- roadmap / implementation-guide sync for the delivered additive scope

## Verification

- Stage 1: [PR-N1_stage1.md](PR-N1_stage1.md)
- Stage 2: [PR-N1_stage2.md](PR-N1_stage2.md)
- Stage 3: [PR-N1_stage3.md](PR-N1_stage3.md)

## Focused results

- new primitive suite: `9 passed`
- no driver wiring performed
- no change to existing characteristic parity numbers claimed

## Auditor note

The prompt-level symmetry sentence was too loose.  The exact discrete
symmetry exposed by the delivered formulas is x↔y exchange:
`Sigma_- -> -Sigma_-`, `phi -> pi/2 - phi`, with `Pi_-` odd under the
transform.  That correction is now locked in the tests and audit docs.
