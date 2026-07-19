# PR-T3B Preflight Audit

## Scope

Landed:
- `src/rabbit/jax/full_boltzmann_collision_preflight.py`
- `tests/test_pr_t3b_collision_preflight.py`

This patch does **not** widen the runtime JAX driver. It is a bounded,
host-side preflight that asks whether the existing isotropic collision
backbone can populate the future full-Boltzmann moment core and be
lifted back onto the explicit `(species, ray, q)` shell as a factorized
`U C V` update.

A follow-up private runtime hook is now also landed in
`src/rabbit/jax/driver_typeI_full_boltzmann.py` behind
`collision_mode="spectral_relaxation_preflight"`.  That mode reuses the
same closure on the explicit shell during actual CPU solves, but remains
audit-only and non-canonical.  A further private extension now allows
`thermo_tier=2` on the same bounded surface, wiring the lifted collision
moments into the JAX 3T thermo primitives without changing any public
backend or default path.

An additional bounded private mode is now landed behind
`collision_mode="projected_physical_preflight"`.  This mode replaces the
spectral-relaxation bank core with a state-dependent projected-physical
source+damping closure, still on the isotropic species banks and still
strictly audit-only.

## Reused surface

The preflight deliberately reuses the narrowest existing bounded pieces:
- `evaluate_species_monopole_collision(...)` logic from
  `rabbit.transport.species_boltzmann_bridge`
- species bank bookkeeping from `rabbit.collisions.species`
- energy-exchange diagnostic from
  `rabbit.thermo.incomplete_decoupling.compute_energy_exchange_rate`

It does **not** pull the unfinished direct kernels
`NuEScatteringOperator` / `PairProcessOperator` into the JAX runtime yet.
It now does, however, evaluate those same existing operators on the
host-side isotropic species banks through an explicit
`T_bank/T_gamma` remap in the new `closure_mode="direct_kernel"`
preflight.

## Core result

The explicit `(species, ray, q)` transport state can be reduced onto the
three isotropic banks `(ν_e, ν̄_e, ν_x)` cleanly. On that bank state:

- the production-bounded source-only closure gives the expected physics
  hierarchy (`ν_e > ν_x`) and vanishes at equal-temperature FD
  equilibrium
- but its state Jacobian is exactly zero

That is the key design finding. The current source-only backbone is a
valid **collision source**, but not yet a valid **state-dependent
moment-core Jacobian** for the full JAX shell.

To close the numerical preflight loop without changing any public
runtime path, the module also exposes an audit-only state-dependent
`closure_mode="spectral_relaxation"` path. With that closure enabled:

- the bank Jacobian becomes finite and block-diagonal by species
- the same bank core can be lifted back onto the explicit ray state with
  a factorized `U C V` operator
- that lifted operator matches dense finite differences on bounded
  explicit-shell regressions
- the lift now pads cleanly into the actual phase-1 full-Boltzmann state
  layout with zero rows/columns outside the transport slice
- the private full-Boltzmann runtime can now carry `thermo_tier=2`,
  emitting the contract
  `collision_preflight_transport_plus_active_plus_3T_low_rank_v1`
  with phase-state sizes `104` (phase 1) and `111` (phase 2) at the
  bounded `(N_mu, N_q) = (4, 6)` regression point
- on that tier-2 surface, the factorized Jacobian matches dense AD
  exactly on all non-thermo rows; the three thermo rows remain a bounded
  preflight approximation with max absolute mismatch below `3e-3`

The next bounded closure on the same shell is
`closure_mode="projected_physical"` / runtime
`collision_mode="projected_physical_preflight"`.  With that closure
enabled:

- the bank core becomes truly state-dependent without the ad hoc
  spectral-relaxation term
- the lifted explicit-shell runtime remains factorized with the same
  `104 -> 111` tier-2 state sizes at `(N_mu, N_q) = (4, 6)`
- the factorized Jacobian matches dense AD on **all** rows at tight
  tolerance for the bounded tier-2 regression point
- the mode is still only a projected physical preflight, not the final
  Hannestad–Madsen + pair Stage-B operator

The newest bounded preflight mode is `closure_mode="direct_kernel"`.
With that closure enabled:

- the bank core is driven by the existing
  `NuEScatteringOperator` and `PairProcessOperator`
- the bank state is explicitly remapped from the shell thermal variable
  to the operator thermal variable via `y = q T_bank / T_gamma`
- the operator output is converted from its natural `[MeV]` rate to
  `d/dN` with the required `1/H` scaling before any bank/thermo
  diagnostics are formed
- equal-temperature FD inputs satisfy detailed balance to bounded
  numerical tolerance on the shared Laguerre grid
- hotter-plasma cases show the expected `ν_e > ν_x` hierarchy from the
  actual operator-backed bank response
- the combined bank-plus-3T source Jacobian is finite and nonzero on
  bounded regressions
- the host preflight now also materializes the missing active-scalar
  Jacobian directions on the augmented input
  `[f_bank, T_gamma, T_nu_e, T_nu_x, H]`, so the direct-kernel surface
  is no longer limited to bank-state finite differences only
- a bounded private runtime candidate now also exists behind
  `collision_mode="direct_kernel_preflight"`: it keeps the primal RHS on
  a host callback, but bypasses AD by injecting an explicit structured
  Jacobian payload through the Rodas5P low-rank hook
- this runtime candidate is currently locked only at the `rhs/jacobian`
  evaluability level, not by an end-to-end solve smoke; it remains
  private and CPU-only until a broader runtime audit is closed

## Verification

- `PYTHONPATH=src pytest -q tests/test_pr_t3b_collision_preflight.py`
  - `6 passed in 0.75s`
- `PYTHONPATH=src pytest -q tests/test_pr_t3a_collisionless_driver.py tests/test_pr_t3b_collision_preflight.py`
  - `12 passed in 17.26s`
- `PYTHONPATH=src pytest -q tests/test_pr_t3a_collisionless_driver.py -k "factorized_jacobian or collision_preflight_update_embeds or collision_preflight_smoke"`
  - `4 passed, 5 deselected in 37.56s`
- `PYTHONPATH=src pytest -q tests/test_pr_t3a_collisionless_driver.py -k "factorized_jacobian or collision_preflight_smoke"`
  - `5 passed, 6 deselected in 90.85s`
- `PYTHONPATH=src pytest -q tests/test_pr_t3a_collisionless_driver.py::test_full_boltzmann_private_tier2_collision_preflight_smoke -vv`
  - `1 passed in 42.44s`
- `PYTHONPATH=src pytest -q tests/test_pr_t3a_collisionless_driver.py tests/test_pr_t3b_collision_preflight.py`
  - `17 passed in 56.72s`
- `PYTHONPATH=src pytest -q tests/test_pr_t3b_collision_preflight.py -k "projected_physical"`
  - `1 passed, 6 deselected in 0.59s`
- `PYTHONPATH=src pytest -q tests/test_pr_t3b_collision_preflight.py -k "direct_kernel"`
  - `2 passed, 7 deselected in 14.69s`
- `PYTHONPATH=src pytest -q tests/test_pr_t3a_collisionless_driver.py -k "projected_physical"`
  - `2 passed, 11 deselected in 51.46s`
- `PYTHONPATH=src pytest -q tests/test_pr_t3a_collisionless_driver.py tests/test_pr_t3b_collision_preflight.py`
  - `20 passed in 101.29s`
- `PYTHONPATH=src pytest -q tests/test_pr_t3a_collisionless_driver.py tests/test_pr_t3b_collision_preflight.py`
  - `22 passed in 85.09s`
- `venv/bin/python -m pytest -q tests/test_pr_t3b_collision_preflight.py -k "direct_kernel"`
  - `4 passed, 7 deselected in 61.02s`
- `venv/bin/python -m pytest -q tests/test_pr_t3a_collisionless_driver.py tests/test_pr_t3b_collision_preflight.py`
  - `24 passed in 133.97s`
- `venv/bin/python -m pytest -q tests/test_pr_t3a_collisionless_driver.py::test_full_boltzmann_private_tier2_direct_kernel_rhs_and_jacobian_smoke tests/test_pr_t3b_collision_preflight.py -k "direct_kernel"`
  - `5 passed, 7 deselected in 206.18s`

## Verdict

Conditional pass.

The smallest credible Stage-B preflight is now closed:
- bank gather is well defined
- species hierarchy is captured
- source-only closure limitation is explicit
- nontrivial moment-core structure is reachable with bounded
  state-dependent relaxation
- explicit-shell lifting can stay factorized instead of materializing a
  dense transport collision block
- the same relaxation closure now runs end-to-end on the private
  full-Boltzmann shell with a matching low-rank Jacobian payload
- a private tier-2 thermo hook now exists on that same shell, but its
  thermo-row Jacobian should still be treated as bounded-preflight
  rather than exact parity
- a second private closure now exists that is genuinely state-dependent
  without the spectral surrogate and restores exact dense-AD parity on
  the bounded tier-2 runtime surface
- the real ν-e / pair operators are now exercised on the host-side
  species-bank surface with bounded detailed-balance and hierarchy locks
- the host-side direct-kernel audit now also exposes finite derivatives
  with respect to the active thermodynamic scalars
  `(T_gamma, T_nu_e, T_nu_x, H)` instead of only the bank state
- a private direct-kernel runtime candidate now evaluates successfully at
  the `rhs/jacobian` level through the custom low-rank solver hook

What remains for a real PR-T3B runtime patch is still substantial:
- close an end-to-end solve smoke and then bounded physics parity on the
  new direct-kernel runtime path
- decide whether to keep the current callback-plus-explicit-Jacobian
  architecture or replace it with a real JAX port
- then promote from private bounded tier-2 smoke to bounded FLRW
  reheating checks
