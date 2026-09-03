# BD622 D-040 legacy JAX, Bianchi, and DSMC mitigation survey

Date: 2026-07-23  
Branch: `f10-independent-validation-b3v2`  
Survey run: `run-20260723-f10-d040-legacy-jax-readonly-survey`  
Survey context: `29bf22b3aef7ce7ff2cbd5a718a9bd9e036065ac5dea0ae3fd73820060baeff2`  
Adjudication: `d7d34e412bd3b8d94f382748f58b53c6102a1f806aa64f443c656a29203d481c`

## Executive conclusion

Conventional DSMC should be excluded from a possible future Bianchi programme
as the primary forward solver, precision validator, covariance or derivative
validator, and endpoint-authority path. This is a design recommendation under
D-040, not a theorem that Monte Carlo cannot converge and not authorization to
implement another method.

The owner's concern identifies a real practical risk, but its premise needs one
correction. In the currently stated Type-I, no-anisotropic-stress limit,

\[
\frac{d\Sigma_i}{dN}=-(1-\Sigma^2)\Sigma_i,\qquad
\frac{d\Sigma^2}{dN}=-2(1-\Sigma^2)\Sigma^2 ,
\]

so the physical branch \(0\leq\Sigma^2<1\) has monotone shear-norm damping.
The future collision-coupled \(\Pi[f]\) response has not been specified or
linearized, so a highly oscillatory shear or expansion-rate mode is currently
`SPECULATIVE`. A fast collision scale is plausible; whether the frozen future
system is stiff, oscillatory, overdamped, or merely multiscale must be derived
before selecting a solver.

The legacy JAX tree contains useful formula and software clues, but no solver or
physics implementation that should be restored. Reuse means independent
semantic rederivation of a small set of mechanisms: the diagonal Type-I
characteristic map, explicit angular moment bookkeeping, selective automatic
differentiation, and exact block or low-rank Jacobian organization. Calibrated
closures, clipped-state physics, modified-Jacobian “preconditioning,” broad
Bianchi dispatch, and retired forward drivers remain `DEPRECATED` or unsafe.

D-039 `STOP/PRESERVE` remains binding. Every remedy below is `PROPOSED` design
advice only.

## Exact evidence set

| Evidence | SHA-256 |
|---|---|
| JAX current-tree mapper | `ef5df977295e98decb78326bd6516ffed3984785c342e69296236c6231053be7` |
| physics and convention audit | `2ba9d3fb1ec7a8ca41350f15d4d2b89a3da26348815af6016c438a92acaecfcf` |
| numerical and solver audit | `3534afc268c74be15f0a9a345816ba2c58415844a13d412f229ab30fbea4f4c7` |
| local Bianchi and DSMC audit | `ac87b8d2806bfc2e4c4c490d2d0a38de48225e11e04ce75c9b8f834b0ff2de06` |
| preserved legacy-history recovery | `8df6b7b589c7bda86de08b0f6e589d4f128106926a8a1a9c5840ea1ed2f7b47f` |
| primary-literature matrix | `2761c6a820794230433e21e9be629d6d742275d66798c4afe7e12b5803da74ab` |
| final adjudication | `d7d34e412bd3b8d94f382748f58b53c6102a1f806aa64f443c656a29203d481c` |

The adjudicator normalized 37 raw findings to 34 exact
`(claim_id, evidence_fingerprint, verdict)` keys. Repeated keys were
complementary subfindings rather than independent votes; no genuine verdict
conflict remained.

## What the legacy history actually contains

The preserved rollback authority is
`.publish-local/20260719T052443Z/legacy.git` at
`0b2339c676dba6f36dbf0850419c4d78e1a5f907`; it contains 1,336 commits from
HEAD and 1,342 across named refs. The 37-ref bundle has SHA-256
`9f22f318f50a994a0a17c11ec831c5beaeeb42b67fbcae8111f9fee41af21920`.
It was queried in place without checkout, restoration, or code execution.

The reachable root `692f394149846033fb39e974221db7d41b33e50f` already imported a
mature JAX surface: 40 files and 15,557 lines. There is no recoverable
pre-root incremental genesis. The post-root history nevertheless exposes the
useful mechanism and failure timeline:

- `52d2712b` added diagonal non-LRS characteristic primitives; `3d13c8f`
  connected a compact non-LRS driver.
- `eb850dfa` added the private full-Boltzmann direct-kernel candidate;
  `e5d671b0` added reduced JAX neutrino-electron and pair collision code.
- `b2ff69fc` added fixed-schedule Rodas5P discrete-adjoint replay.
- `3cc4463b` and `c0e4d770` added Class-A characteristic and vector-tilt
  scaffolds.
- `905e6aa9` recorded a JIT tracer leak caused by caching trace-local JAX
  arrays. The safe pattern was to cache host NumPy data and convert it inside
  the active trace. Its physically motivated temperature remap then exposed a
  stiff equilibrium manifold and Rodas5P microsteps; the remap was reverted.
- The same history recorded an under-resolved \(N_\mathrm{eff}=2.993\) result
  against about 3.044, with coarse quadrature, interpolation, placeholder
  matrix elements, and incomplete physics as known causes.
- A hostile audit on preserved ref `815579b2` found a dimensionally wrong
  \(T^4\) reduced collision prefactor; `e1bd7c19` corrected it to \(T^5\).

Later history deliberately removed rather than promoted the broad surface:

- `2c6f2efd` and `3b892482` deleted the augmented and nonlinear non-LRS
  collision/weak-network layers.
- `ddb9d0e9` removed dead or test-only curved-PSTF, Class-B, generic-Jacobian,
  experimental-linear-solve, vector-tilt, and duplicate collision surfaces.
- `0c1d0ac2` retired the Type-I forward line to opt-in/parity status.
- `f52cdb26` retired JAX runtime authority.

Legacy HEAD has 52 `src/rabbit/jax` files and current HEAD has 48. All 48
common paths have identical Git blob IDs. The four legacy-only paths are
`driver_typeI.py`, `driver_typeI_char.py`, `run_extended_bbn.py`, and
`runtime_device.py`. The current tree is therefore a high-level authority and
dispatch deflation, not a corrected rewrite of the retained low-level code.

## Reuse and rejection map

| Historical mechanism | D-040 disposition | Required before any future use |
|---|---|---|
| diagonal Type-I log-stretch characteristic map | `PROPOSED` formula oracle | rederive frame, signs, Jacobian, energy factors, FLRW/LRS embeddings, and finite non-LRS covariance |
| symmetric \(S^2\) quadrature and explicit \(+\)/\(-\) PSTF kernels | `PROPOSED` bookkeeping clue | freeze the full tensor state, stress normalization, degeneracies, rotational/permutation covariance, and absolute null budgets |
| static f64 shapes, `jit`/`vmap`, host-concrete caches | `PROPOSED` engineering pattern | exact cache keys, no swallowed x64 errors, no trace-local cached arrays, exact backend/runtime metadata |
| selective JVP/VJP and gather-core-scatter factors | `PROPOSED` derivative pattern | smooth strict-open coordinates, dense-versus-factor identity, rank/conditioning bounds, conservation and phase falsifiers |
| fixed-schedule discrete-adjoint replay | `PROPOSED` only for exact accepted-schedule replay | replay the actual primary solver schedule, events, controller decisions, and exact arithmetic contract |
| calibrated RTA/AP collision closures | `DEPRECATED` | do not import; replace with independently derived finite-mass full-catalogue physics |
| raw-occupation clipping or clipped rates | unsafe | use a strict-open occupation coordinate and fail raw on invalid states |
| approximate negative diagonals added to the Rosenbrock Jacobian | unsafe | any preconditioner must be solver-side and solution-neutral |
| general Class-B masks and broad geometry dispatch | unsafe | outside the Type-I target and missing constraint/curvature authority |
| retired JAX forward/runtime drivers | `DEPRECATED` | do not restore; Rust AOT remains the implementation target |

Specific physics hazards in the retained code include massless electron
kinematics in reduced collision kernels, incomplete reactions/species,
unproved discrete conservation and entropy, a calibrated momentum-independent
RTA, an LRS-only full driver with `Pi_minus=0`, fixed stress normalization,
clipped occupations, and no complete neutrino first-law or external
finite-\(\Sigma_-\) validation. Internal JAX/NumPy or JAX/Rust parity does not
repair those gaps.

## Why DSMC is not the primary Bianchi route

DSMC is not categorically non-convergent. A recent neutrino DSMC calculation
reports \(N_\mathrm{eff}=3.0439\pm0.0006\) in homogeneous isotropic FLRW using
30 million simulated neutrinos, 4,000 particles per cell, and a step bounded
by both expansion and interaction timescales. The same paper relies on
isotropy to omit momentum directions and notes that explicitly tracing them
would add Monte Carlo noise
([arXiv:2508.08379](https://arxiv.org/abs/2508.08379)). An earlier neutrino
DSMC study is likewise homogeneous and isotropic
([arXiv:2409.07378](https://arxiv.org/abs/2409.07378)).

Those successes do not transfer to the proposed Bianchi precision lane:

1. The needed \(\Pi_+\) and \(\Pi_-\) are signed, cancellation-dominated
   directional moments that vanish at isotropy. Classical particle analysis
   already identifies shear-stress estimates as finite-sampling
   signal-to-noise problems
   ([arXiv:cond-mat/0207430](https://arxiv.org/abs/cond-mat/0207430)).
2. If a decaying observable has amplitude \(A(N)\), a sample-mean relative
   error target away from a zero scales like
   \(N_\mathrm{eff}\propto\mathrm{Var}(\phi)/A(N)^2\); at a zero crossing,
   relative error is undefined. Absolute tolerances and phase error are
   mandatory.
3. Noise in \(\Pi\) is not only reporting noise. It enters the shear RHS, then
   \(H\), per-e-fold collision and weak rates, and abundances. Nonlinear
   feedback can create pathwise bias even when an isolated estimator is
   unbiased.
4. Standard DSMC must resolve collision time, expansion time, and any derived
   shear phase time simultaneously. Oscillation alone does not defeat
   fixed-time Monte Carlo convergence, but it raises the timestep and phase
   burden while signal decay raises the particle burden.
5. Pauli blocking for arbitrary anisotropic nonthermal distributions requires
   a directional occupation estimator. The precision FLRW work uses an
   effective Fermi-Dirac estimate and identifies generalization as additional
   work.
6. The project also needs deterministic covariance, derivatives/Jacobians,
   exact nulls, event reproducibility, and precision endpoint comparisons.
   Conventional DSMC is not an economical authority mechanism for those
   obligations.

Classical propagation-of-chaos theory supplies useful background but no
project guarantee: the Nanbu result is for a spatially homogeneous classical
hard-potential system and is not uniform in time
([arXiv:1302.5810](https://arxiv.org/abs/1302.5810)). It does not cover a
relativistic, multi-species, Fermi-blocked, anisotropic, dynamically coupled
Einstein-Boltzmann system.

A future stochastic calculation may be reconsidered only as a separately
authorized, bounded qualitative cross-check on a frozen background or a
large-signal regime. It would need a deterministic validated F-11A reference,
an independently derived sampler, unbiased monopole and stress estimators,
pathwise positivity/conservation/covariance tests, \(N\)- and timestep
refinement, multiple independent seeds, preregistered confidence intervals,
and absolute-error handling at zero crossings. It must not validate the same
closure from which its control variate or moment guide was derived.

Static Monte Carlo or quasi-Monte Carlo spot checks of individual collision
integrals are a different method from DSMC trajectory evolution and may remain
a future independent-integration option.

## Ranked deterministic mitigation programme

### R1 — premise discriminator

Before solver construction, freeze the exact Type-I frame, gauge, state,
stress normalization, physical branch, and intended expansion/shear
observables. Linearize the coupled two-shear/PSTF system about FLRW and LRS,
derive the \(\Pi[f]\) response spectrum, and distinguish:

- real monotone damping,
- complex oscillatory modes,
- collision versus geometry stiffness,
- transient non-normal growth,
- and signals below the arithmetic/discretization floor.

Stop the special oscillatory-method branch if no material complex or separated
fast mode exists. This is the cheapest high-value discriminator.

### R2 — exact physics and representation contract

Only after R1, independently derive and freeze:

- the diagonal Type-I characteristic and direction map;
- the full PSTF state and \(\Pi_{ab}\) normalization;
- radial and angular bases with rotational, permutation, FLRW, LRS, and
  finite-\(\Sigma_-\) embeddings;
- strict-open \(0<f<1\) occupation coordinates;
- explicit species, spin, degeneracy, identical-particle, and \(2\pi\) ledger;
- finite-electron-mass shells and full no-QKE collision catalogue;
- Pauli blocking, positivity, weak-form conservation, first law, entropy, and
  tail enclosures;
- absolute signal/error budgets for nulls and zero crossings.

No post-hoc projection or calibrated relaxation closure may substitute for an
equivalent discrete weak form.

### R3 — bounded deterministic forward prototype

Only if R1 and R2 pass, consider a Rust-target deterministic prototype:

- analytically remove the exact Type-I streaming/redshift map where possible;
- keep Rodas5P/Rosenbrock as the number-of-record baseline;
- use analytic transport blocks plus selective AD JVP/VJP for the collision
  residual;
- use exact gather-core-scatter or low-rank factors only after dense identity,
  rank, and conditioning tests;
- put preconditioning only in the linear solver, never in a modified method
  Jacobian;
- impose phase-aware maximum steps only for a derived frequency;
- separate radial, angular, collision, time, phase, and event errors;
- compare dense small systems, factored systems, solver families, tolerances,
  invariants, and external anchors.

Quantum asymptotic-preserving, micro-macro, fast-spectral, or exponential
schemes are contingencies, not default choices
([arXiv:1009.3352](https://arxiv.org/abs/1009.3352),
[arXiv:1810.03090](https://arxiv.org/abs/1810.03090),
[arXiv:1010.1472](https://arxiv.org/abs/1010.1472),
[arXiv:1809.00028](https://arxiv.org/abs/1809.00028)). They may be compared
only if R1 measures a collision-stiffness blocker that the baseline cannot
resolve economically without weakening R2.

Low-variance, moment-guided, or time-relaxed Monte Carlo methods are also
research clues rather than transfer-ready remedies
([arXiv:0905.2218](https://arxiv.org/abs/0905.2218),
[arXiv:1207.1005](https://arxiv.org/abs/1207.1005),
[arXiv:1009.2768](https://arxiv.org/abs/1009.2768)). Their published
classical or relaxation-model evidence does not supply the missing quantum
PSTF and coupled-geometry contract.

## Prospective validation ladder

Any later owner-authorized design should advance in this order:

1. exact FLRW embedding and isotropic collision null;
2. exact collisionless diagonal Type-I characteristic;
3. \(\Pi=0\) monotone shear-norm law;
4. LRS finite-shear characteristic and stress response;
5. external finite-\(\Sigma_-\) non-LRS anchor and covariance;
6. weak-shear linear response with derived damping/frequency;
7. finite-mass collision activation with conservation, first-law, positivity,
   entropy, and tail gates;
8. fully coupled deterministic trajectory;
9. only then a separately authorized stochastic qualitative cross-check.

Each stage must separate physics-model error, radial error, angular error,
collision-integral error, time/phase error, event error, and arithmetic error.
A passing earlier stage cannot self-authorize the next one.

## Authority and stop conditions

- JAX remains a frozen local parity/AD/Jacobian component oracle.
- Rust AOT remains the active implementation target.
- Conventional DSMC is excluded from primary, precision, covariance,
  derivative, and endpoint-authority roles.
- Coupled highly oscillatory Type-I shear remains `SPECULATIVE`.
- The reusable legacy mechanisms and R1–R3 programme remain `PROPOSED`.
- F-11/Bianchi implementation and execution remain owner-paused.
- D-039 `STOP/PRESERVE`, W7/B3/Rust/Radau/trajectory/endpoint/unblinding
  prohibitions, QKE exclusion, and public-runtime prohibitions remain
  unchanged.

No code, test, benchmark, collision, trajectory, endpoint, or scientific
output was executed for this survey.
