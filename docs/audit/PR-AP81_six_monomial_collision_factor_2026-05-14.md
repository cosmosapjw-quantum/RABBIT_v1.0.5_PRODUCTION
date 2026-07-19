# PR-AP81 Six-Monomial Collision Statistical Factor

## Scope

AP81 lands the quartic-cancelled fermionic 2-to-2 Pauli gain-loss polynomial
described in `neutrino_collision_term_PSTF.md` into executable collision paths.
The shared statistical factor is now represented by the six signed monomials
`34`, `12`, `123`, `124`, `134`, and `234`; no `1234` quartic term is present.

This is an executable scalar collision-factor algebra upgrade, not a new
pass/fail wrapper.  It covers the diagonal no-QKE scalar occupation-number
factor used by the deterministic `nu-e`/pair references, the deterministic
pairwise diagonal `nu-nu` reference, the staged NumPy AP19/AP33/AP35/AP41
diagonal `nu-nu` source bridge, and the existing SciPy/JAX `nu-e`/pair and JAX
diagonal `nu-nu` kernels.  The staged source bridge now uses the pairwise
reference by default with explicit per-bank number closure and effective-`nu_x`
weighted-energy closure projection; the older fixed-point redistribution helper remains legacy
comparison plumbing.  The AP6 radial follow-up adds a normalized
unit-direction angular momentum-closure weight for the descriptor-driven
`pstf_radial` source path plus an opt-in p-dependent radial Gaussian closure
for small studies, while exact angular Dirac-delta convergence remains future
work.

## Changed

- `src/rabbit/collisions/deterministic_reference.py`
- `src/rabbit/collisions/__init__.py`
- `src/rabbit/collisions/pstf_contractions.py`
- `src/rabbit/collisions/nu_e_scattering.py`
- `src/rabbit/collisions/pair_processes.py`
- `src/rabbit/jax/collisions_jax.py`
- `src/rabbit/jax/nu_nu_scattering_jax.py`
- `src/rabbit/transport/augmented_collision_bridge.py`
- `src/rabbit/transport/augmented_nonlrs_transport.py`
- `src/rabbit/transport/augmented_pstf_distribution.py`
- `src/rabbit/transport/augmented_typeI_weak_network.py`
- `src/rabbit/transport/augmented_typeI_nonlrs_collisionless.py`
- `src/rabbit/transport/augmented_typeI_observables.py`
- `src/rabbit/transport/__init__.py`
- `src/rabbit/weak/augmented_bridge.py`
- `src/rabbit/weak/augmented_angular_rates.py`
- `src/rabbit/weak/live_rates.py`
- `src/rabbit/weak/sigma_plus_kernel.py`
- `src/rabbit/validation/augmented_collision_source_budget.py`
- `tests/test_deterministic_collision_reference.py`
- `tests/test_augmented_collision_bridge.py`
- `tests/test_augmented_angular_rate_inputs.py`
- `tests/test_augmented_nonlrs_nonlinear_transport.py`
- `tests/test_augmented_pstf_distribution.py`
- `tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py`
- `tests/test_augmented_typeI_observables.py`
- `tests/test_augmented_collision_source_budget.py`
- `tests/test_augmented_pstf_capability_registry.py`
- `tests/test_typeI_anisotropic_weak_kernel.py`
- `tests/test_weak_quadrature_reference.py`
- `tests/test_pr_t3b_jax_operator_parity.py`
- Registry-backed roadmap and capability docs

## Hot-Path Follow-Up

The AP80 smoke profile was rerun before this follow-up to target optimization
against current evidence.  On the local CPU-only path, the direct AP77
`N_q=(3,4)` smoke gate took `13.704308991000289 s` for `7596` nfev, while the
AP80 smoke profile took `13.902745158993639 s` for the same nfev budget.  A
`cProfile` pass showed the wrapper overhead was negligible; the dominant pure
Python costs were deterministic collision-source evaluation, weak angular input
extraction, and live weak-rate kernel-cache key construction.

This follow-up keeps the same equations and solve nfev while reducing per-call
Python overhead:

- deterministic reference kernels now use the six-monomial Pauli polynomial
  through an internal already-validated scalar path inside their quadrature
  loops, while the public `collision_statistical_factor(...)` still validates
  inputs;
- live weak-rate channel-kernel cache keys now use a compact temperature,
  correction-level, and quadrature-order signature instead of rebuilding large
  ndarray byte keys on every RHS call;
- `extract_augmented_weak_inputs(...)` now validates the augmented distribution
  once and builds monopoles, plus/minus moment inputs, and CL3 metadata from the
  same moment pass rather than re-walking the angular distribution.
- nonlinear non-LRS transport coevolution now reuses a single reconstructed
  nodal distribution for nodal transport, stress moments, and the AP64/AP65
  weak-network RHS instead of reconstructing the same `A_modes` repeatedly.
- live weak-rate bounded c/f channel geometry and correction arrays are cached
  across temperature changes, leaving only the electron blocking factor
  temperature-dependent per RHS call;
- non-LRS S2 grids now carry a precomputed nodal-to-mode projection matrix so
  nonlinear transport and source-only projection avoid repeated Gram builds and
  small dense solves on every RHS call;
- `stress_moments_from_distribution(...)` now validates the distribution once
  and computes rho, plus/minus stress, and angular monopoles from the same
  validated arrays.
- LRS and non-LRS CL3 angular weak-rate factors now evaluate the ν_e and
  anti-ν_e explicit quadrupole profiles through one paired K2 kernel build,
  reusing the shared `q`, `T_nu`, and `q**2` weights instead of validating and
  reconstructing the same kernel twice per RHS call.

After the follow-up, the same direct AP77 `N_q=(3,4)` smoke gate took
`7.12728450098075 s` with unchanged `7596` nfev, and the AP80 smoke profile
took `7.560119595989818 s` with unchanged `7596` nfev.  The paired K2 change is
small relative to solve and Python-call noise but removes the last obvious
duplicate CL3 kernel build in the current CPU hot path.  This is a CPU
hot-path optimization only; it does not change the no-QKE scope or promote
public production support.

## Numerical Evidence

The focused deterministic reference tests now lock replay-stable non-equilibrium
values for the same fixed Gauss quadrature used by the AP collision references:

- `evaluate_nue_scattering_reference(0.9*f_FD, species="nux", N_q=4)`:
  `C = [4.344961202594062e-25, 1.693819085195735e-27,
  -4.253858969711776e-28, -8.866588862977984e-30]`,
  `dQ_nux = -1.613195360681361e-25`, `number_residual =
  1.134850761186205e-26`.
- `evaluate_pair_annihilation_reference(0.85*f_FD, f_FD, species="nue", N_q=4)`:
  `C = [4.255525672009493e-23, 9.442883536958380e-25,
  9.231474980185903e-26, 3.656228676039949e-27]`,
  `dQ_nue = 6.244596045189361e-23`, `number_residual =
  1.857348430505656e-23`.
- `evaluate_nunu_diagonal_twoto2_reference(0.5*f_FD, 1.2*f_FD, species="nue",
  N_q=4)`: `C = [3.699338186698699e-23, 8.938058422053670e-25,
  2.104373550769815e-25, 6.283838388772933e-27]`, `dQ_nue =
  1.159234205452598e-22`, `number_residual = 2.810867430372965e-23`.
- `build_augmented_lrs_nunu_collision_thermo_source(...)` on the AP19 LRS
  redistribution fixture now reports `dQ_nue_pair_N =
  3.524845077339936e-22`, `dQ_nux_bank_N = -3.524845077339939e-22`,
  `weighted_energy_residual = 3.291384182302405e-37`,
  `raw_weighted_energy_residual = 1.189811421690089e-21`, and
  `max_number_residual = 6.171345341817009e-38` after number/energy closure.

## Verification

```bash
PYTHONPATH=src pytest -q tests/test_deterministic_collision_reference.py tests/test_pr_t3b_jax_operator_parity.py tests/test_pr_t3c_nu_nu_preflight.py
PYTHONPATH=src pytest -q tests/test_augmented_collision_bridge.py tests/test_augmented_typeI_weak_network_3t_solve.py tests/test_augmented_collision_source_budget.py
PYTHONPATH=src pytest -q tests/test_typeI_anisotropic_weak_kernel.py tests/test_augmented_angular_rate_inputs.py
```

Result: `46 passed` for the deterministic/JAX collision-factor bundle, and
`45 passed` for the augmented bridge/3T/source-budget bundle after the staged
source route was switched to the pairwise reference.  The paired CL3 weak-rate
kernel regression adds `29 passed` for the weak-kernel/angular-input bundle.

The focused suite first failed on missing six-monomial exports, then on
placeholder numerical reference values.  It passed after the shared statistical
factor and replay-stable numerical references were landed.

## Directional Momentum-Delta Follow-Up

The descriptor-driven AP6 `pstf_radial` source geometry now uses normalized
unit-direction momentum-delta weights instead of the previous uniform
four-angle factor.  The weight tensor is a Gaussian in
`|e1 + e2 - e3 - e4|`, normalized against the deterministic angular quadrature,
so smoke-scale radial sources keep their angular integral while favoring
vector-closed angular tuples.  This is still a deterministic no-QKE
smoke-scale angular closure, not an exact Dirac-delta convergence claim.  An
opt-in `radial_gaussian` model is also available for small studies; it weights
the full `p1 e1 + p2 e2 - p3 e3 - p4 e4` residual and disables static grid-cache
reuse by construction because the angular tensor is radial-tuple dependent.

Additional verification:

```bash
PYTHONPATH=src pytest -q tests/test_augmented_collision_bridge.py tests/test_pstf_process_catalog.py tests/test_augmented_typeI_nonlrs_nonlinear_collision_feedback_3t.py
```

Result: `73 passed`, including the normalized unit-direction momentum-delta
helper and a real nonlinear non-LRS `pstf_radial` smoke diagnostic carrying
`pstf_radial_unit_direction_momentum_delta_v1`.  The follow-up targeted tests
also lock the opt-in p-dependent radial Gaussian provider and
`pstf_radial_radial_momentum_delta_v1` diagnostics.

The radial Gaussian control is now exposed above the low-level AP6 builder:
AP65 radial and combined nonlinear 3T wrappers, AP66 publication matrix rows,
AP68 guarded forward predictions, and AP69 SMC schema metadata all forward the
`pstf_radial_momentum_delta_model`/`pstf_radial_momentum_delta_sigma` controls.
A real AP65 combined-source smoke path with
`pstf_radial_momentum_delta_model="radial_gaussian"` passed locally, confirming
that the p-dependent closure can run through the staged nonlinear solve surface.

## Boundary

- QKE remains explicitly out of scope.
- AP81 does not promote public production support.
- AP81 lands the scalar occupation-number Pauli polynomial used inside the
  listed kernels/references and routes the staged NumPy diagonal `nu-nu` source
  bridge through the pairwise reference with number/energy closure projections.
  AP6 radial source geometry now has normalized unit-direction momentum-delta
  weights plus an opt-in p-dependent radial Gaussian closure, and the staged
  AP65/AP66/AP68/AP69 diagnostic surfaces can forward that closure, but exact
  angular Dirac-delta convergence at promotion tolerances remains future work.
- Weak-rate angular corrections remain the AP59/AP60 moment-input path and are
  not converted into full anisotropic weak-rate integration here.
