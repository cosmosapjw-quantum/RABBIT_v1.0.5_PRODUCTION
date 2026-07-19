# RABBIT Rust-First Bottom-Up BBN and Type-I No-QKE Plan

Date: 2026-07-15

Status: ACTIVE, SPECIFIED. This file is the sole normative future-order plan.
It supersedes the collision-first R-03C1B sequence as an execution order, but
does not erase the implementation or evidence already accepted in R-01 through
R-03C1A-R1. Those slices are parked until their consumers become reachable.

The controlling boundaries remain:

- `docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`;
- `bbn_codex_anti_drift_cost_effective_policy.md`;
- `AGENTS.md` and the exact claim vocabulary defined there;
- Rust AOT is the active implementation and repeated-run design target;
- SciPy/BDF is the temporary number-of-record until the Rust endpoint authority
  gate passes;
- JAX is frozen as a local parity, AD, and Jacobian oracle only;
- QKE and public-production claims are FORBIDDEN.

## 1. Diagnosis: the migration did not start bottom-up

The reachable Git history begins at root commit
`692f394149846033fb39e974221db7d41b33e50f` on 2026-04-20. That commit is a
646-file, 101,585-insertion repository snapshot. It already contains 40
`src/rabbit/jax/` files and 15,557 JAX lines, together with thermo, weak,
network, drivers, and tests. The repository is not shallow and has no parent
or configured remote from which an earlier incremental history can be
recovered. The requested “first JAX development” is therefore observable only
as a mature imported snapshot, not as a sequence of small foundational commits.

From that baseline, development proceeded primarily by extending and freezing
an existing Python/JAX system, then porting selected mature vertical slices and
collision algebra into Rust. R-01 was endpoint-consumed, but R-02/R-03 moved
quickly into electron collision catalogue, matrix element, response, and phase
space work before Rust owned a minimal isotropic BBN endpoint. This is a
translation/migration-first history, not a bottom-up Rust BBN construction.

That ordering is a major development bottleneck because it creates all of the
following at once:

1. complex collision code has no small Rust endpoint consumer;
2. solver, thermodynamics, weak-rate, network, and state-contract defects are
   discovered only inside a large coupled path;
3. parity can reproduce a shared approximation or indexing error and be
   mistaken for scientific validation;
4. every collision change inherits a large Python/JAX/Rust comparison surface;
5. substantial effort is spent on specs, falsifiers, and unconsumed private
   modules while the simplest end-to-end scientific question is still open;
6. performance work cannot distinguish activation cost from actual repeated
   BBN integration cost;
7. claims become difficult to scope because solver mechanics, port fidelity,
   continuum accuracy, and physical validation are mixed.

Diagnosis status: DERIVED from reachable Git and live-tree evidence. It does
not imply that prior collision work is scientifically useless. It changes its
dependency position.

## 2. Evidence survey and limits

The 2026-07-15 live `docs/` census contains 291 files (about 93 MiB): 188
Markdown, 46 PNG, 30 TeX, 12 PDF, and supporting logs/code/data. The survey used
full inventory, Git status/tracking classification, PDF metadata/text
extraction, topic routing, cross-reference search, and focused reading of the
controlling and physics-relevant sources. It is not represented as a manual
line-by-line reading of every page. Untracked or ignored dossiers/PDFs can
inform a decision but cannot be the sole authority in a clean checkout.

`docs/jcap_revised_final.pdf` was text-extracted and visually checked at the
metric/Friedmann, weak-rate, and BBN sections. It specifies an axisymmetric
Bianchi-I model, instantaneous neutrino decoupling, coupled shear and neutrino
anisotropic stress, and a Kawano-based BBN calculation. It also argues that the
leading angle-averaged weak-rate correction from a small neutrino quadrupole
vanishes at first order. This is a model and manuscript target for later
LRS work; it is not evidence that the current Rust code has a validated FLRW,
weak, network, collision, or non-LRS endpoint.

External corrective retrieval was used to separate design references from
authorities:

- [PRIMAT](https://www2.iap.fr/users/pitrou/primat.htm) and its precision-BBN
  paper define the high-accuracy comparison lane and explicitly separate
  thermodynamics/cosmology from the nuclear solve;
- [PArthENoPE](https://parthenope.na.infn.it/) supplies an independent public
  coupled-ODE/network comparison tradition;
- [PRyMordial](https://arxiv.org/abs/2307.07061) and
  [LINX](https://arxiv.org/abs/2408.14538) are readable modern cross-checks,
  not authorities to translate line by line;
- [diffsol](https://docs.rs/diffsol/) documents the Rust BDF substrate and
  root/event facilities, but its numerical mechanics still require local
  conformance tests.

Local code and external sources are consulted in this order for claims:
independent derivation or matched external result, executable local tests and
artifacts, implementation inspection, then prose. No single external code is a
truth oracle.

## 3. Architecture rule: one executable ladder

The Rust programme shall grow through one narrow, continuously executable
ladder:

```text
units/constants
  -> ODE contract
  -> tree-level FLRW thermodynamics
  -> Born n<->p weak rates
  -> 9-species/12-reaction network
  -> minimal coupled isotropic BBN endpoint
  -> matched standard anchors
  -> QED and weak corrections
  -> full standard network
  -> electron collision operator
  -> classical neutrino Boltzmann transport/collisions
  -> LRS Type-I
  -> non-LRS Type-I
```

Every layer must have a direct consumer in the next layer. Crate-private code
is allowed while its contract is stabilizing. New public Python dispatch,
capability registry entries, readiness wrappers, manifests, hashes, or figure
gates are forbidden until the corresponding Rust endpoint is executable and a
pre-existing surface is consolidated.

Shared non-negotiable contracts:

- natural units are explicit at each interface; MeV, seconds, and dimensionless
  derivatives may not be mixed implicitly;
- integration uses a monotone independent variable with declared orientation;
- vector absolute tolerances are state-specific;
- terminal events have direction and refined event state/time;
- every solver returns the last raw finite or nonfinite state on failure;
- negative abundances, conservation drift, NaNs, and rejected steps are never
  clipped, repaired, or converted into successful output;
- observables are derived only from a successful state satisfying declared
  normalization and positivity gates;
- configuration is immutable, serializable, and included in comparison
  artifacts; hidden defaults are forbidden;
- comparison tolerances are set from independent accuracy requirements before
  results are inspected.

## 4. Dependency DAG and stage ownership

```text
F-00 evidence/order reset
  `- F-01 dual Rust solver contract
       `- F-02 minimal tree-level FLRW
            |- F-03 Born weak n<->p
            `- F-04 9-species/12-reaction network
                 `- F-05 minimal coupled Rust BBN
                      `- F-06 matched standard anchors
                           `- F-07 QED thermodynamics
                                `- F-08 weak/radiative/recoil corrections
                                     `- F-08N full standard network
                                          `- F-09 electron collisions
                                               `- F-10 classical neutrino Boltzmann
                                                    `- F-11A LRS Type-I
                                                         `- F-11B non-LRS Type-I
```

R-01 through R-03C1A-R1 remain `CLOSED_WITH_LIMITS`. Their collisionless and
electron-sector algorithms may be consumed only at the corresponding F-stage
after revalidation against the new foundation. R-03C1B is PARKED, not the next
action. Historical AP/FB/BD paths remain provenance and cannot select work.

## 5. F-00 - evidence and execution-order reset

Deliverables:

- replace the collision-first specification with this bottom-up DAG;
- record the reachable Git-history limit and document-corpus survey honestly;
- update only the existing project state, decision, claim, validation, and
  next-session files;
- preserve accepted historical rows and mark their future-order authority
  deprecated rather than rewriting executed evidence;
- name F-01 as the first executable blocker and F-09 as the parking point for
  the current collision programme.

Acceptance:

- normative planning lines decrease substantially;
- no runtime, physics, generated registry, Python, JAX, or public API changes;
- Markdown links resolve where files are tracked;
- the five existing harness surfaces agree on next action and claim ceiling.

Claim ceiling: SPECIFIED execution order and DERIVED history diagnosis only.

## 6. F-01 - crate-private dual stiff-solver contract

Implement one Rust `OdeSystem`/`OdeConfig`/`OdeResult` interface with:

- pinned `diffsol` BDF using a Rust linear solver;
- corrected eight-stage nonautonomous Rodas5P using required `df/dt`;
- analytic Jacobian input, vector `atol`, `rtol`, `h_init`, `h_min`, `h_max`,
  and a total attempt budget;
- direction-filtered terminal event detection and refinement;
- accepted/rejected steps, Jacobian evaluations, and linear setup counts;
- fail-loud input/build/linear/nonfinite/step-floor/budget outcomes;
- last raw state and time on every failure.

Required conformance problems:

1. nonautonomous exact solution for both solvers;
2. fixed-step Rodas5P observed order at least 4.5;
3. stiff scalar exact solution and Robertson kinetics;
4. event time/state refinement for both directions and a wrong-direction
   negative control;
5. invalid tolerance/dimension/state inputs;
6. NaN RHS, singular or failed linear solve, `h_min`, and max-attempt exits;
7. BDF `h_max` and budget enforcement;
8. deterministic repeat results on the same host/toolchain.

F-01 validates solver mechanics only. It must not add a Python binding,
backend key, endpoint claim, or BBN result. A large generated lockfile is
reported separately from authored LOC. The current pinned diffsol release's
nalgebra-only build path is locally compile-failing because faer symbols are
exported unconditionally; enabling the minimum compiling feature set is an
implementation constraint, not authority for a second solver path.

## 7. F-02 - minimal tree-level FLRW skeleton

Scope is deliberately smaller than the existing Tier-1 port:

- photons, finite-mass equilibrium electrons/positrons at `mu_e=0`, and three
  massless neutrino species;
- zero QED plasma corrections, zero nuclear energy backreaction, zero shear,
  zero collision distortion;
- Friedmann equation with one declared Planck-mass convention;
- instantaneous neutrino decoupling with an explicit branch:
  conserve total EM+neutrino entropy above `T_dec`, and EM entropy below it;
- integrate temperature versus `N=ln(a/a_start)` from 10 MeV through the BBN
  range and terminate on a decreasing-temperature event.

The legacy frozen GL64 electron EOS is a port-parity reference, not continuum
authority: a preliminary adaptive audit found a relative difference of about
`2.5e-5` in total photon-plus-electron tree-level energy density near 0.1 MeV
(the electron-only energy-density difference is larger). F-02 must replay this
as an executable test, use or validate a threshold-regularizing transform, and
demonstrate convergence in `rho_e`, `P_e`, `d rho_e/dT`, and entropy. A
transformed Born/EOS integral is preferred to merely increasing a fixed
quadrature order.

Acceptance checks:

- dimensions and signs of `rho`, `P`, `H`, `dT/dN`;
- ultrarelativistic and nonrelativistic electron limits;
- `T_nu=T_gamma` above decoupling and the instantaneous-decoupling entropy
  ratio below electron annihilation;
- comoving entropy conservation separately on both sides of the branch;
- BDF/Rodas endpoint agreement within preregistered numerical error;
- independent high-accuracy quadrature comparison over the full temperature
  interval;
- monotone cooling and refined terminal temperature without clipping.

The existing `dT_gamma_dN_tier1` EM-only entropy evolution above decoupling is
not reusable unchanged: it enforces `T_nu=T_gamma` while omitting neutrino
entropy from the conserved total and changes `dT/dN` by about `1.5e-3` at
2 MeV in the independent audit.

Claim ceiling: VALIDATED only for the declared idealized FLRW model.

## 8. F-03 - Born neutron-proton weak sector

Implement exactly the six tree-level charged-current channels with finite
electron mass, zero chemical potentials, no radiative/recoil/weak-magnetism/
finite-temperature correction, and one neutron-lifetime normalization
convention. Rate integrals must use a variable transform that resolves
thresholds and semi-infinite tails.

Acceptance:

- independent high-precision integral reference at a logarithmic temperature
  grid and around thresholds;
- detailed balance in the equilibrium limit;
- positivity and units for each channel and total `n->p`, `p->n` rates;
- high/low-temperature limits and neutron-decay limit;
- equilibrium neutron fraction and standalone freeze-out evolution;
- BDF/Rodas agreement and quadrature convergence.

The current fixed GL64 Born normalization has an observed relative bias of
about `1.5e-6`; matching it is port parity, not continuum validation.

## 9. F-04 - minimal standard nuclear network

Implement a checked, data-driven 9-species state:
`n, p, D, T, He3, He4, Li6, Li7, Be7`, initially with the first 12 reactions
of the accepted 31-reaction PRIMAT-derived table. Species and reactions are
identified by names/stoichiometry, never positional coincidence.

Critical hazard: the current 9-species implementation matches the first 12
entries of `primat_ac2024_31rxn.json`; the standalone
`primat_ac2024_12rxn.json` has a different order. Indexing the latter as if it
were the former is a silent physics error and is FORBIDDEN.

Acceptance:

- exact baryon-number and charge stoichiometry for every forward/reverse pair;
- reaction permutation invariance and named-table identity checks;
- analytic Jacobian versus finite differences away from boundaries;
- equilibrium/detailed-balance test where applicable;
- closed-network baryon conservation and raw negativity failure;
- manufactured one-reaction and small-chain solutions;
- rate unit and density-power checks.

No final abundance claim is allowed at F-04.

## 10. F-05 - minimal coupled Rust BBN endpoint

Couple F-02, F-03, and F-04 without QED or higher weak corrections. Use one
state layout, one RHS, one Jacobian ownership rule, and one terminal event.
Initial conditions, baryon density, temperature range, normalization, and
observable formulas are immutable configuration.

Required outputs are raw terminal state, conservation residuals, solver stats,
event data, and only then `Yp`, D/H, He3/H, Li7/H. Observable construction must
fail if the state is negative, nonfinite, unnormalized, or unsuccessful.

Acceptance:

- BDF/Rodas endpoint agreement;
- tolerance and quadrature convergence;
- baryon conservation across the full solve;
- correct weak-only handoff and no hidden initial-abundance tuning;
- an independent local formulation of the RHS, conservation residuals, and
  observables on fixed states; external-code comparison remains F-06 work;
- deterministic cold and repeated-run timing, labelled separately.

F-05 is the first executable Rust BBN skeleton. Its simplified abundances are
model outputs, not precision-standard validation.

Implementation outcome (2026-07-15): `CLOSED_WITH_LIMITS`. The crate-private
state is `(T_gamma, ln n_b, X_n, X_p, X_D, X_T, X_He3, X_He4, X_Li6,
X_Li7, X_Be7)` versus `N=ln(a)`. A configured late-time eta anchors `n_b`
after electron-positron annihilation, with `d ln(n_b)/dN=-3` throughout the
solve. The weak-only handoff has exactly zero nuclear abundances; no seed,
floor, clipping, or terminal repair is used. The strict physical RHS rejects
negative states, while implicit internal stages use the finite signed
mass-action polynomial and every accepted/event-refinement state remains in
the physical domain.

Both BDF and Rodas5P reach the raw endpoint and agree under tolerance and
quadrature refinement. An independent SciPy formulation using `T_gamma` as
the independent variable, adaptive continuum EOS/Born integrals, and a direct
canonical 12-reaction RHS agrees with the refined Rust observables at
`7.7e-8`--`3.9e-7` relative. The default `0.08 MeV` activation is nevertheless
not on the `0.10`--`0.12 MeV` plateau: it shifts `Yp` by about `1.68e-4` from
`0.10 MeV`. Therefore the default is a declared simplified F-05 configuration,
not an activation-converged, externally matched, precision-standard, or
number-of-record result. F-06 must revisit the matched start condition.

## 11. F-06 - matched standard anchors and authority decision

Reuse or consolidate an existing serialized configuration surface for matched
runs against PRIMAT plus at least one of PArthENoPE, PRyMordial, or LINX; do not
add a new readiness/configuration manifest. Match constants, neutron lifetime,
baryon density, reaction set/order, weak corrections, neutrino treatment,
temperature interval, solver tolerances, and observable definitions. A
comparison with unmatched defaults is diagnostic only.

Promotion requires all of:

- per-block agreement for FLRW, weak rates, network RHS, and final endpoint;
- stable tolerance/quadrature convergence;
- raw failures retained;
- repeated-run Rust benefit measured on the endpoint, not a segment;
- no unexplained cancellation between mismatched blocks;
- substantive superseded Python/JAX runtime surface actually deleted;
- active source LOC below the recorded PORT-00 baseline, preserving the legacy
  R-06 migration-deflation gate.

Only after F-06 may Rust become the private number-of-record for the minimal
model. Public production remains FORBIDDEN.

Implementation outcome (2026-07-15): `CLOSED_WITH_LIMITS`. The existing
`tests/fixtures/flrw_gold_v861.json` now serializes the exact matched profile,
the 12 reaction identities and table hashes, PRIMAT v0.3.2 C/Python live
results, both Rust solvers, and LINX v0.1.2 tolerance-ladder results. The Rust
path uses a `10 -> 0.86164715286738 -> 0.005 MeV` weak/Saha/network sequence,
the ideal high-temperature instantaneous-decoupling entropy normalization,
Born rates, the exact AC2024 piecewise-loglinear rows, and raw failure
semantics. Positive log-abundance coordinates prevent invalid implicit stages;
they are not clipping or endpoint repair.

Live PRIMAT thermodynamic blocks and weak totals agree with Rust on fixed
temperature grids; the largest weak-total relative discrepancy is below
`2e-7`. The direct Python rate/RHS formulation, Rust analytic Jacobian and
conservation tests, and the independently implemented PRIMAT matched-table
endpoint all pass separately. At `rtol=1e-9`, Rust BDF gives
`Yp=0.2424118898167283` and `D/H=2.429829551078614e-5`; Rodas5P gives
`Yp=0.2424119070631264` and `D/H=2.429829855058169e-5`. Both meet the frozen
PRIMAT endpoint budgets. LINX independently checks the RHS and Kvaerno3
integration, but shares PRIMAT background arrays, the same nuclear rows, and
the Rust Saha initial state; it is not independent background or nuclear-data
validation. No aggregate agreement is used to excuse a failed component.

The optimized full matched BDF endpoint measured `0.746599 s` on its first
in-process execution and `0.723545 s` on the immediate repeat, with bitwise
identical state and solver counters. F-06 also deletes the two superseded
high-level JAX Type-I drivers, their runtime/device wrappers, public dispatch
and inference branches, and their runtime/policy/benchmark tests and scripts.
Public forward dispatch is now `auto`/`scipy` only; retained JAX weak,
network/collision, Rodas5P, full-Boltzmann, and alternate-geometry code is a
non-dispatchable frozen component oracle. Conservative active source is
`55,860` Python plus `9,517` Rust = `65,377` lines, below the reconstructed
PORT-00 implementation baseline `68,268` by `2,891` lines.

Rust is therefore the private number-of-record only for this exact minimal
zero-QED/Born/12-reaction profile. SciPy remains the general corrected research
runtime and comparison authority after F-08N and until the F-08D strict
standalone authority blocker is resolved. Precision-standard, collision,
anisotropy, QKE, and public-production claims remain FORBIDDEN.

## 12. F-07, F-08, and F-08N - standard-physics completion

F-07 adds finite-temperature QED thermodynamics behind a zero-QED switch and
validates sign, dimensions, asymptotic limits, entropy, and independent tables.

F-08 adds weak corrections one independently switchable term at a time:
zero-temperature radiative, finite nucleon mass/recoil, weak magnetism,
finite-temperature radiative, and any declared incomplete term. Each term
requires derivation/provenance, detailed balance, neutron-lifetime
normalization, isolated delta, and comparison with precision references.

The executable F-08 order is now frozen more narrowly:

1. F-08A adds the zero-temperature Coulomb plus resummed radiative block
   (`CCR`) as one lifetime-normalized model. Coulomb-only and radiative-only
   pieces may be tested internally, but they are not endpoint authority modes.
2. F-08B adds the finite-nucleon-mass Fokker-Planck correction with weak
   magnetism disabled.
3. F-08C activates weak magnetism through the same finite-mass coupling
   algebra, not as a scalar multiplier.
4. F-08D adds the complete balance-restoring finite-temperature radiative
   block. A virtual or bremsstrahlung fragment alone cannot be promoted.

The legacy Python correction levels are bounded comparison material, not an
authority to translate: their Coulomb channel ownership, bounded-channel-only
Sirlin use, scalar finite-mass factors, floors, and missing thermal block do
not implement the PRIMAT precision convention.

F-08N expands the named network from 12 to the selected standard reaction set
and validates reaction-table provenance, nested-network convergence, and
abundance sensitivity. It may not tune rates to recover a preferred number.

Precision-standard abundance claims require the combined F-07/F-08/F-08N
matched comparison; isolated parity is insufficient.

F-07 implementation outcome (2026-07-15): `CLOSED_WITH_LIMITS`. Rust now
implements an exact zero switch, PRIMAT's leading `dPa+dPe3` convention, and
the complete scalar `dPa+dPb+dPe3` pressure convention. Energy density,
entropy, and their temperature derivatives are derived from the one pressure
function. The zero branch is bitwise identical to F-06 through EOS, FLRW,
freeze-out, baryon normalization, and the matched endpoint.

Independent SciPy integrals check all three pressure terms; the `dPb`
double integral and its first two temperature derivatives use a formulation
that shares neither Rust's triangular mapping nor analytic derivatives.
A live PRIMAT v0.3.2 custom-table off/on run validates the leading endpoint
delta. PRIMAT's public Python/C endpoint tables omit `dPb`, so the complete
endpoint is only IMPLEMENTED with BDF/Rodas, conservation, convergence, and
raw-failure checks; it is not externally validated. Direct 48-by-48 exchange
quadrature also leaves a material repeated-endpoint cost. F-07 therefore
closes `ACCEPT_WITH_LIMITS`, without precision-standard or performance
promotion, and F-08A becomes active.

F-08A implementation outcome (2026-07-15): `CLOSED_WITH_LIMITS`. Rust now
selects either the bitwise-preserved Born model or one PRIMAT v0.3.2
zero-temperature CCR model in the existing weak consumer. The latter applies
the relativistic attractive electron-proton Coulomb factor on exactly four
physical branches, the resummed radiative factor on all six branches, and the
same combined model to the neutron-lifetime phase-space normalization. The
threshold product is evaluated by its analytic limit rather than a rate floor
or clipped endpoint.

Independent adaptive physical-channel integrations validate the component
factors, both lifetime normalizations, all six rates, totals, detailed balance,
limits, and quadrature stability. Live PRIMAT C/Python custom-table executions
validate the isolated CCR endpoint delta under the exact leading-QED,
12-reaction matched profile. A separate SciPy DOP853 solve uses `T_gamma` as
its independent variable, adaptive electron/weak quadrature, and nested public
PRIMAT QED-pressure splines; its fresh-process repeat is bitwise identical and
both Rust solvers agree at the standalone activation and final endpoints.
LINX was reviewed as a formula cross-check only; no F-08A LINX numerical run is
claimed. F-08A therefore closes the zero-temperature CCR blocker without
precision-standard, full-network, transport, public-runtime, or production
promotion, and F-08B finite nucleon mass with weak magnetism disabled becomes
active.

F-08B implementation outcome (2026-07-15): `CLOSED_WITH_LIMITS`. Rust now
adds PRIMAT's first-order-in-`1/M_N` finite-nucleon-mass Fokker-Planck terms to
the existing CCR channel integrands and lifetime normalization; this is not a
scalar post-hoc multiplier. The anomalous weak-magnetism coefficient is
exactly `delta_kappa=0`. The `n->p` direction owns `m_p/m_e`, the `p->n`
direction owns `m_n/m_e`, and the vacuum normalization owns `m_p/m_e`.
Direction-specific quadrature scales replace the shared hotter-bath scale
after an extreme-temperature-ratio falsifier exposed contamination of a tiny
reverse channel.

Independent compact-versus-expanded integrands, adaptive component/rate
integrals, lifetime normalization, all six channels, GL convergence, limits,
raw failures, and the mass-corrected detailed-balance target pass. The
first-order truncation leaves a nonzero detailed-balance residual
(`3.745706491106527e-5` relative at `1 MeV`); it is exposed rather than tuned
away. A separate `T_gamma`-variable DOP853 standalone solve and a live matched
PRIMAT v0.3.2 Python endpoint pass a 17/17 validator. PRIMAT C was SKIPPED
because it hardcodes physical weak magnetism and cannot represent this
`delta_kappa=0` slice.

For the exact leading-QED/CCR/F-08B/12-reaction profile, external PRIMAT Python
gives `Yp=0.2466842688457449`, `D/H=2.455665403302237e-5`,
`He3/H=1.0433375736211641e-5`, and `Li7/H=5.486433620053931e-10`;
the isolated F-08B-minus-CCR `Yp` shift is
`0.001125972051388563`.
Rust BDF gives handoff `X_n=0.24009432294275196` and
`Yp=0.24668541343980116`; Rust Rodas5P gives
`X_n=0.24009432335028977` and `Yp=0.24668542688179274`. Adversarial review
found no BLOCKER/HIGH/MEDIUM finding. It retained two LOW limits: default GL64
is not precision authority for negligible individual channels at artificial
`T_gamma/T_nu=1000`, and no pre-F-08B saved CCR bit-pattern fixture exists,
although the current eight-value forward drift lock passes.

The final Rust release gate passes 142/142 all-target tests plus 0 doctests.
The broader Python bundle preserves the inherited public-SciPy CL2 AlterBBN
RED (`1 failed, 39 passed, 12 skipped`); it is outside the private F-08B path
and was not hidden, skipped, or used to widen a tolerance.

F-08B therefore validates only this matched no-weak-magnetism profile. It is
not stock PRIMAT, precision-standard, C-backend, thermal-weak, full-network,
transport, QKE, public-runtime, or production validation. F-08C closes
separately below and does not retroactively broaden F-08B.

F-08C implementation outcome (2026-07-15): `CLOSED_WITH_LIMITS`. Rust now
activates PRIMAT's physical anomalous weak-magnetism coefficient
`delta_kappa=3.70589007463` through the same generalized first-order
finite-mass evaluator used by F-08B; the exact `delta_kappa=0` F-08B branch is
preserved, and no scalar weak-magnetism multiplier was introduced. Independent
compact-versus-expanded checks cover 252 points. The physical `s=+1`
couplings are `(f1,f2,f3)=(2.487954860124953,-1.5945873479212611,
0.1066324877963081)`, the PRIMAT lifetime normalization is
`Fn=1.7547510462240024`, and the no-CCR weak-magnetism normalization shift is
exactly zero within the frozen `2e-15` ceiling. Six-channel unequal-temperature
anchors, equal-temperature totals, raw failures, limits, and quadrature checks
pass. The first-order modified-detailed-balance residual remains nonzero at
`3.6755293490786656e-5` at `1 MeV`; it was not tuned away.

The independent weak-only `T_gamma`-variable standalone solve was refined to
GL320. Its maximum F-08C `X_n` change from GL160 is
`9.259088912250135e-9`, below the pre-Rust `2e-8` refinement ceiling; the
pre-Rust Rust-comparison budgets remain `3e-7` in `N` and `6e-8` in `X_n`.
The first unsplit high-temperature GL ladder was honestly RED at
`4.0784416011074853e-8`; splitting the physical momentum panels closed that
numerical defect without widening a ceiling or changing the weak formula.

Fresh PRIMAT v0.3.2 Python and C matched custom-table endpoints each repeat
twice with exact float equality. Python gives
`(Yp,D/H,He3/H,Li7/H)=(0.24683601806557295,2.4565015281772263e-5,
1.0434637936584758e-5,5.488592195819055e-10)` and C gives
`(0.24683586374148064,2.4565071974942807e-5,
1.0434644927261921e-5,5.488576741277125e-10)`. Rust matched endpoints in
`(Xn_handoff,Yp,D/H,He3/H,Li7/H)` order are
`(0.24019629115929314,0.2468369814321261,2.4564568596661556e-5,
1.0434585907028622e-5,5.488720109289649e-10)` for BDF and
`(0.24019629154783906,0.24683701452683648,2.45645716039475e-5,
1.0434586301065364e-5,5.488719893518859e-10)` for Rodas5P. The external
validator passes 23/23 checks and all 5/5 mutation probes are rejected.

Final Rust closeout passes `cargo test --release --all-targets` at 150/150,
strict Clippy with `-D warnings`, formatting, and both focused F-08C consumer
tests. The stored compact comparison replay passes, unsupported live mode exits
`2`, and the hardened replay exact-locks coordinated component, rate,
freeze-out, and endpoint drift; its 5/5 root mutation probes include coordinated
C-endpoint drift. Python `py_compile`, JSON parsing, and diff checks pass. The
targeted public-Python lane preserves only the inherited AlterBBN `Yp` RED
(`1 failed, 5 passed, 1 skipped`, gap
`1.0513455427477447e-3 > 5e-4`), while the baseline/null/gold subset passes
`15 passed, 11 skipped`.

Exact Rust source grows from `12,677` lines at the F-08B boundary to `13,464`
at F-08C (`+787`). Python under `src/rabbit` remains `55,860`, so active source
is `69,324`, or `1,056` above the PORT-00 baseline `68,268`. No Python/JAX
deflation or generated-lockfile change occurred. All-authored added/deleted/net
lines remain unavailable because no non-Rust F-08B stage boundary was frozen;
the F-08C cost verdict is `ACCEPT_WITH_LIMITS` with blocker-movement ratio
`0.75` and measured endpoint progress.

F-08C validates only physical weak magnetism in this matched custom-table,
first-order-in-`1/M_N`, weak-only-standalone, ideal instantaneous-decoupling,
12-reaction profile. F-08D closes separately below and does not retroactively
broaden F-08C.

F-08D implementation outcome (2026-07-15): `CLOSED_WITH_LIMITS`. Rust adds the
complete directional `L_CCRTh` block -- true-photon emission/absorption,
differential bremsstrahlung, `L1`, and `L2+3` -- to F-08C. It is complete only
within the declared first-order-in-alpha, infinite-nucleon-mass thermal
radiative approximation. The four terms are composed before endpoint use; no
virtual or bremsstrahlung fragment is exposed as an endpoint authority model.
The raw directional correction is additive to F-08C and uses the unchanged
F-08C lifetime normalization.

Independent direct integration validates the complete directional block and
the separately resolvable `2e8 K`, `n->p` subterms. The private runtime table
contains 57 log-spaced temperature knots. Direct midpoint comparison over all
56 intervals in both directions passes the frozen ceiling, with maximum
difference-to-ceiling ratio `0.8787904334047045`. Below the declared low-
temperature floor the correction is exactly zero, and temperatures above
`10 MeV`, profile mismatch, nonfinite values, and nonpositive consumer totals
remain raw failures rather than clipped outputs.

The independent external evidence bundle passes 25/25 checks and all 5/5
mutation probes are rejected. Within the exact leading-QED, F-08C,
instantaneous-decoupling, 12-reaction matched profile, the F-08D component and
conditional endpoint are VALIDATED. Rust BDF gives
`(Xn_handoff,Yp,D/H,He3/H,Li7/H)=(0.24020238686934622,
0.24683356615720803,2.4564374518213406e-5,1.0434556632413961e-5,
5.488674122424027e-10)`; Rodas5P gives
`(0.24020238770812863,0.24683358495050578,2.4564376830218522e-5,
1.043455690926778e-5,5.488673896701071e-10)`. The isolated BDF F-08D-minus-
F-08C shifts have the independently frozen signs and envelopes.

Strict standalone/table precision authority remains BLOCKED. The Rust
standalone `X_n` differs from the frozen external authority by order `1e-6`,
above the `8e-8` ceiling, while the independent C global-versus-local endpoint
has a `Y_p` spread of `1.0021545698846168e-6`. Neither discrepancy is hidden by
widening a budget or tuning a physical constant. Conditional matched-endpoint
agreement is not stock-PRIMAT or precision-standard validation.

Final closeout passes `cargo test --release --all-targets` at 160/160, strict
Clippy with `-D warnings`, formatting, doctests, focused component/table/
standalone/endpoint tests, stored compact replay, 25/25 external checks, and
5/5 mutation rejection. Exact Rust source grows from `13,464` lines at F-08C
to `15,020` at F-08D (`+1,556`). Python under `src/rabbit` remains `55,860`, so
active source is `70,880`, or `2,612` above the PORT-00 baseline `68,268`. No
Python/JAX deflation occurred. Exact token use is `UNAVAILABLE` because the
harness exposes no reliable PR-scoped counter. The F-08D verdict is
`ACCEPT_WITH_LIMITS`, with blocker-movement ratio `0.75`: the complete thermal
weak block and conditional endpoint moved, but strict standalone authority did
not.

F-08N implementation outcome (2026-07-15): `CLOSED_WITH_LIMITS`. Rust adds
Rabbit's selected 31-row AC2024 central-rate topology behind
`NetworkExtent::Selected31`, preserves the exact 12-row prefix, binds rows and
species by identity, consumes stored reverse coefficients verbatim, evolves
Li6, and reuses the existing corrected endpoint. The source audit covers all
31 identities and 60-knot tables; independent rate/RHS/Jacobian anchors,
log-coordinate Jacobian refinement, conservation, nested 31-minus-12 effects,
BDF/Rodas plus tighter-BDF convergence, and deterministic repeat pass.

The pre-Rust PRIMAT campaign passes 46/46 checks and rejects 5/5 mutations.
Final Rust release passes 167/167 plus 0 doctests. The consolidated replay
closes every echoed F-08N object schema, rejects unknown-key overclaims 22/22
and semantic mutations 9/9 in adversarial review, and rehashes the repo-local
reaction JSON. Unsupported live mode exits `2`. The two original conservation
REDs remain recorded; the final `5e-9` ceiling was not widened.

The evidence is conditional. This selected set is not a named stock PRIMAT
network, its exact reverse comparison requires private Python injection with no
exact C counterpart, all 31 Rust rows activate at the F-06 handoff instead of
PRIMAT's 17-then-31 staging, shared central rates do not validate measurements,
and `sigma` uncertainty is not propagated. F-08D strict standalone precision
remains BLOCKED; no precision-standard, transport, QKE, or public-production
claim follows.

F-08N adds `+921` Rust lines; the consolidated replay and fixture add `+673`
and `+254`, for exact pre-SSOT component net `+1,848`. Active source is
`15,941` Rust plus `55,860` Python = `71,801`, `3,533` above PORT-00. No
Python/JAX deflation occurred; exact token use is `UNAVAILABLE`. The verdict is
`ACCEPT_WITH_LIMITS`, blocker-movement ratio `0.75`, with measured endpoint
progress.

F-09 electron collisions is now the sole active implementation slice. R-03C1B
may re-enter only as an endpoint-consumed F-09 sub-slice. F-10 classical
neutrino Boltzmann and F-11 LRS/non-LRS Type-I remain parked; QKE and public
production remain FORBIDDEN.

F-09A implementation outcome (2026-07-15): `IMPLEMENTED / VALIDATED` within
the declared massless-MB thermal-moment model.
Rust now evaluates the electron-only massless Maxwell--Boltzmann thermal energy
moment of Escudero (2019) with the repository's existing tree-level HM constants,
couples it equal-and-opposite to a finite-electron-mass leading-QED three-
temperature FLRW system, and consumes that background through F-08C weak rates
and the selected-31 network to `T_gamma=0.005 MeV`. This is a crate-private
executable test endpoint, not a public or non-test runtime path. Exact equilibrium
null, sign, ninth-power scaling, coupling multiplicity, entropy production,
combined first-law closure, an independent QED-off FLRW endpoint, BDF/Rodas5P,
five-point coupled-Jacobian parity, a full-leg tolerance ladder, and bitwise
same-binary release repeat are checked.

The F-09A claim ceiling is strict. Its thermal collision moment is classical and
massless while the bath EOS is finite-mass and Fermi--Dirac. It excludes finite-
`m_e` collision phase space, FD/Pauli collision corrections, spectral evolution,
neutrino-neutrino transfer, independent QED-on full-BBN endpoint validation,
precision `N_eff`/abundances, runtime promotion, QKE, and public production.
Consequently F-09 remains `IN_PROGRESS`; F-09B owns finite-mass FD/Pauli
quadrature, absolute normalization, and replacement of the analytic source in
the same private endpoint before F-10 may open.

F-09B implementation outcome (2026-07-16): `VALIDATED / CLOSED_WITH_LIMITS`
inside the declared isotropic thermal tree-level model.  Rust now evaluates a
finite-vacuum-electron-mass FD/Pauli electron/positron energy moment in
independent plasma-frame/CM coordinates and consumes it in the existing
three-temperature FLRW, corrected weak, and selected-31 private endpoint.  The
preserved HM support/root point density remains a structurally different slow
construction oracle rather than the endpoint hot path.

The independent Python CM formulation reproduces the order-six Rust elastic
and pair components and, on a `(6,6,8)` through `(24,16,24)` radial/polar/
azimuth ladder, bounds the fixed endpoint rule's representative total-source
remainder at about `1.9e-4`.  The direct HM construction reaches the same
finite-mass FD total to about `0.3%` at order 32 and the analytic massless-MB
limit to about `0.38%`.  Exact equilibrium null, temperature-ordering sign,
entropy production, event energy, elastic number and pair lepton-number
topology, multiplicity, low-temperature suppression, and the combined FLRW
first law pass.  A separate adaptive-EOS SciPy solve gives the QED-off cold
endpoint `N=7.93669848646`, `T_nue=0.00358318527801 MeV`,
`T_nux=0.00357679030472 MeV`, and thermal `N_eff=3.03409327085`; DOP853 and
Radau agree before Rust comparison to `5.4e-12` in `N` and below `3.0e-13 MeV`
in the neutrino temperatures.  Rust BDF/Rodas, tighter BDF, five-point coupled
Jacobian, first-law/baryon/density checks, and same-binary repeat pass.

The closeout has a material performance limit.  In the full release run the
default thermal-BBN legs measured about `123 s` for BDF and `544 s` for
Rodas5P, versus the F-09A order-of-tens-of-seconds baseline; no pre-existing
numerical slowdown authority was available, so this is recorded rather than
declared acceptable for repeated production.  F-10 may open at collisionless
isotropic transport, but before an energy-grid endpoint evaluates this source
per node it must directly reduce the measured activation/cold wall or prove a
bounded operator reduction.  A standalone lookup/readiness surface is not an
allowed substitute.

The claim ceiling remains thermal isotropic `L0`, zero chemical potentials,
tree-level HM couplings, and vacuum `m_e`.  Collision thermal masses,
radiative collision corrections, neutrino-neutrino reactions, spectral
distortions, lepton asymmetry, oscillations, QKE, precision observables,
runtime promotion, and public production remain outside authority.  Exact Rust
source is `18,449` lines, `+1,043` from the F-09A boundary; Python remains
`55,860`, so active source is `74,309`, `6,041` above PORT-00.  A committed
F-09A whole-tree diff boundary is absent, so all-file added/deleted counts are
`UNAVAILABLE`; exact token use is also `UNAVAILABLE` because the harness
exposes no reliable stage counter.  The cost verdict is `ACCEPT_WITH_LIMITS`
and blocker-movement ratio is `0.75`: the finite-mass/FD physics, independent
normalization, and endpoint-consumption blockers moved, while repeated-run
cost remains open.

## 13. F-09 - resume electron collisions

Only here may R-02/R-03 collision code re-enter the live ladder. Revalidate
catalogue topology, HM algebra, supplied-event response, finite-mass support,
quadrature, normalization, detailed balance, entropy production, and
conservation against the now-executable standard endpoint.

R-03C1B becomes a candidate sub-slice of F-09, not an autonomous next project.
Its output must be consumed by a physical operator and then an endpoint;
pointwise, segment-only, or table-only progress cannot be called endpoint
progress. Existing private code may be refactored or deleted if the bottom-up
interface exposes a simpler ownership boundary.

F-09A supplies the bounded analytic thermal-moment baseline described above.
F-09B must next revalidate and integrate the preserved finite-mass HM point
density with zero-chemical-potential FD/Pauli occupations. Before replacing the
analytic source it must freeze an independently evaluated continuum anchor,
show radial/angular/support convergence, absolute normalization, equilibrium
null, elastic-neutrino-number and combined-energy conservation, entropy sign,
and cold suppression. Any lookup/interpolation needed for endpoint cost must be
derived from that operator, tested against direct points, and consumed by the
existing endpoint; a standalone table is not progress.

## 14. F-10 - classical neutrino Boltzmann, still no QKE

Add successively:

1. collisionless isotropic redshifting distribution;
2. energy-discretized isotropic Boltzmann evolution;
3. electron/positron collision sources from F-09;
4. neutrino-neutrino classical diagonal-density channels;
5. energy/number conservation, detailed balance, equilibrium null, entropy
   sign, resolution convergence, and matched decoupling observables.

QKE coherence, flavour off-diagonals, oscillation Hamiltonians, and any claim
of full Standard-Model transport remain FORBIDDEN.

F-10A implementation outcome (2026-07-16): `VALIDATED / CLOSED_WITH_LIMITS`
for collisionless isotropic massless transport.  The unique momentum
coordinate is `q=a p` with `a=1` initially.  An arbitrary bounded occupation
vector is an ODE state and obeys the exact collisionless Liouville equation
`df(q)/dN=0`; its discretized energy density is consumed in the Friedmann rate
and an evolved cosmic-time state through the existing photon-temperature
event.  The implementation therefore closes the collisionless redshifting
and energy-discretized substrate, but no collision operator.

Gauss--Laguerre orders 12/24/36 converge to the analytic zero-chemical-
potential FD pair number and energy integrals, while arbitrary occupations
preserve `n_nu a^3` and `rho_nu a^4`.  Every occupation bit is unchanged after
both Rust solvers.  The analytic state Jacobian and explicit `N` derivative
match five-point stencils; invalid grids and occupations fail raw.  An
independent adaptive-EOS continuum-FD SciPy solve gives
`N=7.93804264044181` and elapsed time `52796.0351521035 s` at
`T_gamma=0.005 MeV`; DOP853/Radau agree to `1.5e-13` in `N` and `1.6e-7 s`.
Rust BDF/Rodas give `N=7.93804264042577` and `7.93804264041354`, with elapsed
times `52796.0351545763 s` and `52796.0351520691 s`, on the 24-node grid.

F-10 remains `IN_PROGRESS`.  F-10B0 first moved the measured F-09B endpoint
wall; F-10B1 then added the first endpoint-consumed electron/positron spectral
action.  F-10B2 replaces the fragile direct-occupation ODE coordinate with an
exact logit transform, selects a 16-node/event-six/four direct rule from a
component and endpoint ladder, and bounds the remaining direct-support
finite-difference sensitivity.  F-10C0 implements and validates the first
same-flavour identical-neutrino CM component.  F-10C1 replaces its historical
GL16 resolution RED with the fixed positive exponential N48 representation,
selected only after a direct ladder and a five-profile/two-auxiliary-rule
state-basis envelope.  That is a bounded radial representation result, not a
complete transport or precision result: only row 1 and the same-flavour-
identical subset of row 3 execute; row 3 is incomplete and seven rows are
wholly absent. By owner decision, the current goal stops when F-10 closes the
collision-coupled isotropic FLRW endpoint; LRS/non-LRS Type-I is owner-paused,
while QKE and public production remain forbidden.

F-10A consolidates the shared Gaussian rules while adding the endpoint-
consumed grid: `+553/-78`, net `+475` Rust lines.  Exact Rust source is
`18,924`; Python remains `55,860`, so active source is `74,784`, `6,516` above
PORT-00.  The above-400 net is accepted only because the indivisible slice
includes arbitrary-grid state, physical moments, analytic Jacobian/explicit-
time derivative, independent endpoint consumption, raw failures, and focused
tests; it adds no policy or readiness surface. Exact token use is
`UNAVAILABLE`. The cost verdict is `ACCEPT_WITH_LIMITS`, blocker-movement ratio
`1.0` for the named collisionless/energy-grid substrate; spectral collision
and repeated-run performance blockers remain explicitly open.

F-10B0 performance-prerequisite outcome (2026-07-16): `VALIDATED /
CLOSED_WITH_LIMITS` for the existing thermal endpoint only.  The direct CM
operator now uses radial-six, polar-four, azimuth-eight quadrature.  An
executable six-temperature, two-flavour envelope bounds polar-four versus the
former polar-six rule below `2e-5` relatively.  An independent polar-three
falsification reached `4.69e-5` for electron flavour and `7.63e-5` for heavy
flavour, so order three was rejected.  No interpolation table, fitted factor,
policy knob, or readiness surface was added.

The reduced rule is consumed by the full corrected selected-31 endpoint.  In
the most recent like-for-like release measurements, its default BDF/Rodas5P
legs take `54.21/181.73 s`, versus the preceding F-10A full-suite record of
`123.71/393.10 s`, improvements of `2.28x/2.16x`.  The complete release suite
moves from `516.93 s` for 197 tests to `241.78 s` for 198 tests.  Independent
QED-off DOP853/Radau gives the updated midpoint
`(N,T_nue,T_nux,N_eff)=(7.936698486852856,0.003583185277385943,
0.003576790300999642,3.034093261751178)`; its solver differences remain
`5.37e-12`, `2.96e-13 MeV`, `9.49e-14 MeV`, and `5.50e-10`.  The change from
the former polar-six anchor is about `4e-10` in `N` and `-9.1e-9` in thermal
`N_eff`.

At the F-10B0 boundary this closed only the measured pre-node cost
prerequisite; spectral electron collisions were still absent from the
endpoint.  Exact Rust source was `19,006` lines, net `+82` from the uncommitted F-10A stage
snapshot; gross additions/deletions are `UNAVAILABLE` because that snapshot
has no frozen blob.  Python remains `55,860`, so active source is `74,866`,
`6,598` above PORT-00.  Exact token use is `UNAVAILABLE` because the harness
exposes no reliable stage-scoped counter.  The cost verdict is
`ACCEPT_WITH_LIMITS`, blocker-movement ratio `1.0` for the named endpoint-wall
prerequisite; it does not establish spectral, precision, runtime, QKE, or
public-production authority.

F-10B1 coarse spectral-action outcome (2026-07-16): `VALIDATED /
CLOSED_WITH_LIMITS` for the first conservative electron/positron action and its
private FLRW consumer.  The state now contains separate zero-lepton-asymmetry
electron and degenerate-heavy neutrino-pair occupations on the F-10A `q=ap`
grid.  Finite-vacuum-electron-mass HM support densities, FD electron baths,
Pauli gain/loss factors, and their analytic occupation response are assembled
as conservative discrete event fluxes.  Elastic fluxes are antisymmetrized
between momentum cells; pair fluxes are symmetrized between neutrino and
antineutrino cells.  The resulting quadrature number and lepton-number moments
are identities to the tested `5e-12` relative roundoff ceiling.  The exact
common-temperature FD reference branch enforces the analytic detailed-balance
null without fitting a relaxation rate.

The action is consumed in a QED-off spectral FLRW state with equal-and-opposite
electromagnetic energy debit.  Its analytic Pauli/occupation Jacobian, the
Friedmann and energy-debit rank-one terms, the photon-temperature derivative,
and the explicit-`N` stencil pass focused five-point checks.  The combined
first law closes to `3e-15` relatively.  At the deliberately coarse
radial-six/polar-four grid, BDF and Rodas5P reach
`(N,t,N_eff)=(7.9366049343,52668.2376 s,3.0366891471)` and
`(7.9366080628,52668.3669 s,3.0366525197)` at
`T_gamma=0.005 MeV`; these are regression readouts, not precision predictions.

The resolution evidence is deliberately RED for promotion.  At the
`T_gamma/T_cm=1.2/1.0` thermal anchor, the coarse action energy moments are
only `0.90544/0.90482` of the independently evaluated F-09B CM totals.  A
radial/angular ladder reaches `1.00162/1.00170` at order `32/24`, demonstrating
normalization recovery but not endpoint convergence.  An order-eight/six
endpoint changed the BDF readout to `N=7.9366361078` and
`N_eff=3.0356361978`; its BDF/Rodas state comparison also exceeded the frozen
`3e-5` nodal relative ceiling at a non-underflow occupation
`f~2.25e-5` (`1.11e-9` absolute).  It was rejected as the F-10B1 baseline.
The direct support-boundary explicit-`N` stencil varies at about `1.1e-4`
relatively across the checked step ladder, so smoothing/resolution remains a
named correctness blocker.

The final F-10B1 release gate passes `207/207` Rust tests; strict formatting,
all-target/all-feature Clippy, and doctests pass.  Exact Rust source is
`20,200` lines, net `+1,194` from the uncommitted F-10B0 stage total.  Gross
additions/deletions are `UNAVAILABLE` because that stage has no frozen blob.
Python under `src/rabbit` remains `55,860`, so active source is `76,060`,
`7,792` above PORT-00.  Exact token use is `UNAVAILABLE` because the harness
exposes no reliable stage-scoped counter.  The cost verdict is
`ACCEPT_WITH_LIMITS`, blocker-movement ratio `0.75`: the operator,
conservation/Jacobian contract, and endpoint consumption moved, while
resolution-qualified decoupling, an independent spectral endpoint,
neutrino-neutrino reactions, precision/runtime authority, QKE, and public
production remain open or forbidden.

F-10B2 resolution/coordinate outcome (2026-07-16): `VALIDATED /
CLOSED_WITH_LIMITS` for a resolution-selected private electron spectral
endpoint.  A 12-node direct-occupation state exposed the honest RED: BDF
reached the cold event, but Rodas5P failed `nonfinite_rhs` when a far-tail
occupation approached floating-point zero.  No absolute tolerance, Pauli
factor, or support rule was widened.  The ODE state is now
`u=log(f/(1-f))`, with stable logistic reconstruction, exact chain-rule
Jacobians, and raw failure if the floating-point map cannot represent a strict
`0<f<1` occupation.  The collision action, physical energy moment, and Pauli
algebra are unchanged.  RHS calls omit the unused full action Jacobian; this
was only a roughly five-percent grid-12 segment improvement and is not called
endpoint progress.

The component decomposition separates momentum-grid order from event
quadrature.  Event order four/three is rejected; at 16 nodes the retained
event-six/four energy moments are `0.9907787/0.9908687` of the independent
F-09B thermal anchors, while the independent high-order component check still
reaches `1.0016176/1.0017004` at grid/event `32/32/24`.  On the retained
16-node rule, BDF and Rodas5P reach
`(N,t,N_eff)=(7.9367312298,52681.0360 s,3.0330775769)` and
`(7.9367453070,52681.5412 s,3.0329326538)` at
`T_gamma=0.005 MeV`, satisfy the frozen nodal envelope, and retain physical
open occupations.  The maximum reported relative distortion is restricted to
initial occupations above `1e-8`, where it is about `0.117`; it is a regression
diagnostic, not a precision distortion claim.

A same-code BDF grid ladder gives `N_eff=3.0326021077`, `3.0330775769`, and
`3.0331966346` at 12, 16, and 20 nodes.  The 16-to-20 changes are
`2.12e-6` in `N` and `1.19e-4` in `N_eff`, smaller than the 12-to-16 changes
by factors about 10 and 4.  The grid-20 Rodas leg was deliberately not run;
there is still no structurally independent full spectral endpoint.  A
selected-rule explicit-`N` five-point step ladder is therefore required in
the executable gate and bounds, rather than erases, support-boundary
sensitivity.  These limits prohibit treating the frozen grid-20 BDF constants,
the dual-solver agreement, or the reported distortion/`N_eff` as precision or
external validation.

F-10B2 adds no RTA, fitted collision table, policy selector, readiness surface,
or public caller.  The final release gate passes `209/209` tests in `239.62 s`;
strict formatting, all-target/all-feature Clippy, and doctests pass. Exact Rust
source is `20,458` lines, net `+258` from F-10B1. Python remains `55,860`, so
active source is `76,318`, `8,050` above PORT-00. Gross additions/deletions are
`UNAVAILABLE` because the stage boundary has no frozen blob. Exact token use is
`UNAVAILABLE` because the harness exposes no reliable stage-scoped counter.
The cost verdict is `ACCEPT_WITH_LIMITS`, blocker-movement ratio `0.75`:
momentum selection, tail stability, and bounded support differentiation moved;
independent spectral validation, neutrino--neutrino reactions,
collision-radiative corrections, precision/runtime authority, QKE, and public
production remain open or forbidden.

Historical F-10C0 same-flavour implementation outcome (2026-07-16):
`IN_PROGRESS` at that boundary.
The implemented slice contains only massless diagonal
`nu_a+nu_a <-> nu_a+nu_a` and its identical-antineutrino copy for each
zero-lepton folded flavour shape.  The CM event stream owns all four legs.  Its
global coefficient is 16, equal to the tagged target-row coefficient 64
divided among the four roles; no coefficient from the retained uncalibrated
Python/JAX placeholder is used.  With `s_y=2 y_1 y_2(1-mu)` the retained
same-flavour invariant average is `K_s=s_y^2/4`.  The action uses Pauli
gain-minus-loss, a stable logit affinity, nonnegative linear interpolation for
both outgoing legs, and the same basis for deposition.  Events with either
outgoing energy outside the represented grid are excluded rather than clipped.
The analytic logit Jacobian is consumed directly by the private FLRW system.

The remaining scope is frozen here as a compact `SPECIFIED` zero-lepton,
massless, diagonal nine-row catalogue rather than being inferred from the old
Python/JAX placeholder.  Let
`K_s=(p_1.p_2)(p_3.p_4)` and `K_t=(p_1.p_4)(p_2.p_3)`; `c` is the directed
matrix-element coefficient multiplying `G_F^2 K`, and the global four-leg
weak form uses `eta*c`.  `E` denotes the electron-flavour pair shape and `X`
the degenerate muon/tau pair shape.  Multiplicity `m` counts microscopic
families represented by the folded row.

| Row | Folded microscopic family | `m` | Shape block | Kernel | `c` | `eta` |
|---:|---|---:|---|---|---:|---:|
| 1 | electron same-sign identical `nu nu` and `antinu antinu` | 2 | `EEEE` | `K_s` | 32 | 1/2 |
| 2 | electron same-pair `nu antinu` elastic | 1 | `EEEE` | `K_t` | 128 | 1/2 |
| 3 | heavy same-sign identical and distinct-flavour elastic | 6 | `XXXX` | `K_s` | 32 | 1/2 |
| 4 | heavy same-flavour `nu antinu` elastic | 2 | `XXXX` | `K_t` | 128 | 1/2 |
| 5 | heavy distinct-flavour opposite-sign elastic | 2 | `XXXX` | `K_t` | 32 | 1/2 |
| 6 | muon-pair to/from tau-pair conversion | 1 | `XXXX` | `K_t` | 32 | 1 |
| 7 | electron/heavy same-sign elastic | 4 | `EXEX` | `K_s` | 32 | 1/2 |
| 8 | electron/heavy opposite-sign elastic | 4 | `EXEX` | `K_t` | 32 | 1/2 |
| 9 | electron-pair to/from heavy-pair conversion | 2 | `EEXX` | `K_t` | 32 | 1 |

This catalogue is a design specification derived against the classical
collision conventions in [Hannestad and Madsen
(1995)](https://arxiv.org/abs/astro-ph/9506015), [Dolgov, Hansen, and Semikoz
(1997)](https://arxiv.org/abs/hep-ph/9703315), and [Froustey, Pitrou, and Volpe
(2020)](https://arxiv.org/abs/2008.01074).  It is not validation evidence.
Row 1 is executable in F-10C0, as is the same-flavour-identical subset of row
3 because the same operator acts on each of the two degenerate heavy flavours.
The distinct-heavy part of row 3 and every channel in rows 2 and 4--9 remain
absent.  Thus only one catalogue row is complete; row 3 is partial and seven
rows are wholly absent.

The component-level status is `VALIDATED` within that same-flavour-identical
family ceiling.
Focused release tests pass `8/8`: the tagged Maxwell--Boltzmann target-row
normalization, common and affine FD nulls, eventwise number and energy
identities, nonnegative entropy production, `T^5` scaling, five-point logit
Jacobian, invalid-input failures, and the production-versus-tagged-target
weak-action comparison.  At radial orders 24/32/40/48/64 the production-target
residuals are `4.26067%`, `7.08402%`, `4.82719%`, `3.53719%`, and `2.07348%`.
The 24-to-32 pair is not monotone; only the order-32 onward sequence decreases.
These tests validate the coefficient, measure, sign, conservation, and response
of the same-flavour component, not the selected endpoint resolution or the
complete neutrino--neutrino sector.

The component is also `IMPLEMENTED` in the same private electron-spectral FLRW
consumer.  The executed BDF endpoint is
`(N,t,N_eff)=(7.936729279099723,52680.93842348345 s,3.033105958925452)`;
Rodas5P gives
`(7.936745110163169,52681.52513623983 s,3.032937181617912)`.
Focused combined-first-law, action-activation, coupled-Jacobian, and
explicit-`N` support checks pass.  Neutrino self-scattering has no EM energy
debit: its discrete total-neutrino energy moment closes internally.  These
endpoint values are regression readouts only.  The final full-crate F-10C0
gate passes `217/217` release tests in `260.70 s` plus 0 doctests; strict
release all-target Clippy and formatting also pass.  This regression/static
result does not close F-10C.

At the historical F-10C0 boundary, the selected GL16 nonnegative-linear weak action is about `7.54%` below the
high-resolution tagged-target value, so selected-grid radial/infrared
convergence is RED for promotion.  A global barycentric basis is rejected as
catastrophically ill-conditioned.  A local quadratic basis is not promoted:
although it improves one weak-action probe, it uses negative weights and
produces a much stiffer nodal action.  The retained linear basis preserves
nonnegative, bounded local interpolation and exact eventwise conservation.
The next executable slice was therefore required to reduce the production-target discrepancy by a
defensible grid, basis, or infrared treatment before completing row 3 and
adding the seven wholly absent folded zero-lepton channel rows.  Those
catalogue parts are `SPECIFIED`, not implemented.

At that historical F-10C0 interim snapshot exact Rust source is `21,383` lines, net `+925` from
the F-10B2 total; Python remains `55,860`, so active source is `77,243`,
`8,975` above PORT-00.  All-authored additions/deletions/net are `UNAVAILABLE`
because the uncommitted F-10B2 boundary has no frozen blob.  Exact token use is
`UNAVAILABLE` because the harness exposes no reliable F-10C0-scoped counter.
This is interim working-tree accounting, not a complete PR cost line; a frozen
boundary is required before PR-scoped all-surface accounting can be claimed.
The interim cost verdict is `ACCEPT_WITH_LIMITS`, blocker-movement ratio
`0.50`: a primary-normalized, endpoint-consumed physics component and its
failure boundary now execute, but selected-grid convergence and catalogue
completion remain open.  No runtime lookup table, readiness, policy, manifest,
hash, figure, public-dispatch, or QKE surface is added.  The final release gate passes
`217/217` tests in `260.70 s` plus 0 doctests; strict release all-target Clippy
and formatting pass without changing this RED resolution boundary.

F-10C1 radial-representation outcome (2026-07-16): `VALIDATED /
CLOSED_WITH_LIMITS` for the declared five-profile envelope. The nonnegative
linear interpolation/deposition algebra and all collision coefficients remain
fixed. The shared positive half-line rule maps Gauss--Legendre
`t in (0,1)` through `y=-3 ln(1-t)` and selects N48 as the smallest tested
order that passes both direct and state-basis gates. Direct N32/40/48/64
weak-action residuals are `1.873525%`, `1.227393%`, `0.852441%`, and
`0.489704%`. Five smooth nonthermal profiles evaluated against independent
compact-30 and exponential-4 N128 auxiliary integrations have compact/
exponential residual pairs `0.940/1.105%`, `1.752/1.738%`, `1.717/1.757%`,
`1.924/1.977%`, and `1.709/1.707%`; the worst value is
`1.97715455697% < 2.1%`. The auxiliary exact references agree within `3e-8`,
number/energy close within `1.79e-14`, lost-domain fraction remains below
`7.61e-5`, and entropy has the correct sign. FD number/energy errors are about
`5.4e-9/5.18e-8`, and the actual electron/heavy thermal-anchor residuals are
`0.593592%/0.599410%`.

The composite N32 candidate is rejected despite a favourable direct ladder
because its auxiliary error is about `4.1%` and its electron/heavy anchors
miss by `1.3399%/1.3489%`. Exponential N32 reaches `4.321%` on the declared
profile envelope, and N40 reaches about `2.41--2.48%`; neither ceiling was
widened. The state dimension changes from 34 to 98. This closes only the
declared radial representation. It does not establish arbitrary-distribution
or continuum convergence, complete neutrino scattering, independent full-
spectral authority, precision distortion/`N_eff`, repeated-run promotion,
Type-I, QKE, or public production.

The first matched N48 cold endpoint took `2181.07 s` with maximum RSS
`51,020 KB`. A prepared static self-event geometry candidate completed the
same frozen solver envelope in `1980.52 s`, a `9.195%` (`1.1013x`) wall
reduction below the programme's about-`10%` partial-blocker threshold; maximum
RSS instead increased to `52,692 KB` (`+3.277%`). BDF remained bitwise equal,
while the same-step-count Rodas path moved by `-1.20878e-7` in `N`,
`-0.0106445 s` in elapsed time, and `+3.00450e-6` in `N_eff`, all within its
frozen envelope. The candidate's `112` Rust lines and cache-only delegation
test were removed. The negative result remains in the validation and decision
ledgers; it is not endpoint progress.

The retained no-cache tree then repeated both BDF and Rodas outputs bitwise in
`2191.66 s`, with maximum RSS `50,440 KB`. Against the initial no-cache record
this is `+0.486%` wall and `-1.137%` RSS. The electron values-only allocation
cleanup therefore remains an arithmetic-neutral simplification with zero
claimed endpoint movement. The same cold wall remains the performance blocker.

At the historical F-10C1 working-tree boundary Rust source is `21,769` lines,
net `+386` from the historical F-10C0 snapshot. Python remains `55,860`, so
active source is `77,629`, `9,361` above PORT-00. All-authored additions,
deletions, and net are `UNAVAILABLE` because F-10C0 has no frozen blob. Exact
token use is `UNAVAILABLE` because the harness exposes no reliable F-10C1-
scoped counter. Twelve existing surfaces are in scope: four Rust sources, six
SSOT/handoff documents, and two report sections; the generated lockfile delta
is zero and no superseded Python/JAX source is deleted in this slice. The cost
verdict is `ACCEPT_WITH_LIMITS`, blocker-movement ratio `0.50`: the named
radial-representation blocker closes with executable falsifiers, but the
measured cold wall, partial row 3, seven wholly absent rows, and independent
full-spectral authority remain open. The next performance edit must target a
measured dominant cold-endpoint consumer without adding another telemetry,
readiness, table, or cache surface for a segment-only or sub-threshold result.

F-10C2 catalogue/performance outcome (2026-07-16): `VALIDATED /
CLOSED_WITH_LIMITS`. All nine frozen zero-lepton massless classical diagonal
rows execute through four topology-equivalent production contractions, while
the seven folded channels remain explicit test oracles. Primary-tagged
normalization, the explicit six-species enumerator, separate `K_s/K_t`
topologies, equilibrium/null identities, number/energy conservation, entropy,
`T^5` scaling, analytic response/Jacobian, and the coupled first law pass. The
retained exact solver-local BDF point cache changed the matched dual-solver wall
`2191.66 -> 968.38 s` (`55.815%`, `2.263x`) with bitwise-identical readouts and
`+1.229%` RSS. Four-topology aggregation then changed the completed-catalogue
command wall `1520.92 -> 1184.77 s` (`22.102%`, `1.284x`). The retained full
release record is `230/230` plus zero doctests. These results close
`G-F10-CATALOGUE` and `G-F10-PERFORMANCE` only; the endpoint values remain
same-code private regression readouts.

Independent-validation outcome through D-029 (2026-07-18): whole-program
FortEPiaNO 1:1 authority is `DEPRECATED` because its QKE/oscillation/precision
surface exceeds this classical diagonal no-QKE objective. Its hash-locked
initial-collision slice is contextual only. The direct pointwise candidate and
the full-degree Galerkin-Petrov candidate are retained as `VALIDATED` static
failures under D-027 and D-028. A fresh blind three-node
maximum-relative-entropy design produced a sound fixed-triple exact-arithmetic
derivation and a locally passing standalone binary64 stencil audit, but the
two artifacts do not define one coherent semidiscrete collision method. The
executable adds a quadrature-weight prior, target-dependent nearest-exterior
support selector, and exact-node one-hot branch outside the fixed-triple
continuity/covariance derivation. Support-switch continuity and Jacobian
suitability, the frozen 24-self/15-electron collision-specific binary64
conservation/detailed-balance/entropy contract, and the unchanged native
mu-tau `1e-10` co-gate are absent. Independent adjudication therefore returns
`reject-design`; no implementation, collision call, GL64, Radau, trajectory,
endpoint, or RABBIT-unblinding authority follows.

`G-F10-INDEPENDENT-FLRW` consequently remains `FAIL`, so F-10 is not complete.
This closeout adds no production Python, Rust, JAX, public API, policy knob,
telemetry, or benchmark surface. Exact token use is `UNAVAILABLE` because the
harness exposes no assignment-scoped counter. The design-audit blocker
movement ratio is `0.25` with verdict
`FAILURE_MODE_LOCALIZED_REJECT_IMPLEMENTATION`: the failed method is bounded
before code growth, but the independent endpoint blocker does not move.

## 15. F-11 - LRS then non-LRS Type-I

Owner scope decision (2026-07-16): `OWNER-PAUSED`. The current programme ends
after F-10 completes and validates the collision-coupled isotropic neutrino
Boltzmann FLRW endpoint. Do not implement, validate, or schedule any F-11
Bianchi/LRS/non-LRS work until the owner gives a new explicit instruction. The
material below is retained only as a future specification, not current
execution authority.

F-11A first freezes its conventions in this tracked specification from stable
primary references, including [Barrow 1976](https://doi.org/10.1093/mnras/175.2.359),
[Rothman and Matzner 1984](https://doi.org/10.1103/PhysRevD.30.1649), and
[Campanelli 2011](https://doi.org/10.1103/PhysRevD.84.123521). It then uses the
local revised manuscript as a non-sole LRS/axisymmetric reproduction target on
top of the validated isotropic endpoint. Add geometry/shear without
anisotropic stress, then collisionless neutrino quadrupole/backreaction, then
validated collision terms. Check FLRW and zero-shear limits, sign conventions,
Hamiltonian/Friedmann constraint, stress-energy conservation, small-shear
linearity, and resolution/tolerance convergence.

F-11B removes axial symmetry only after F-11A closes. Use the full trace-free
shear and anisotropic-stress tensors, test rotational/permutation covariance,
recover every LRS embedding and FLRW limit, and compare invariant scalars
independently of component convention. The manuscript's LRS abundance bounds
cannot validate non-LRS physics.

Novel collision-shear physics without an external oracle may be VALIDATED only
inside its declared model through independent derivation, conservation,
symmetry, limiting cases, convergence, and reproducibility. It may not be
described as externally validated.

## 16. PR, cost, and evidence discipline

Each implementation PR names one blocker and normally edits one physics layer
plus its focused tests. Budget is assessed on policy-defined net lines,
including docs and tests: `net_lines <= 80` is preferred, 80-200 requires
direct evidence, 200-400 requires blocker movement or net consolidation, and
above 400 is presumed drift unless a hard blocker moves and the work cannot be
split. Generated Cargo lock changes are also disclosed separately.

Every closeout reports:

```text
added_lines:
deleted_lines:
net_lines:
files_touched:
token_use_exact:
token_use_basis:
runtime_behavior_changed: yes/no
physics_behavior_changed: yes/no
known_blocker_reduced: yes/no
blocker_movement_ratio: 0.00..1.00
validation_strengthened: yes/no
cost_effectiveness_verdict: ACCEPT / ACCEPT_WITH_LIMITS /
  FAILURE_MODE_RELOCATION / NO_PROGRESS / DRIFT
generated_lockfile_lines:
cumulative_rust_source_additions:
superseded_python_jax_deletions:
endpoint_progress: NONE / PARTIAL / MEASURED
validation_commands_and_raw_outcomes:
remaining_risks:
```

No PR may add a standalone diagnostic/readiness/manifest/hash/figure gate
unless it deletes or consolidates more obsolete surface and directly moves a
physics, solver, correctness, or measured endpoint blocker. Already-cheap
segments are not optimization targets while activation/cold endpoint or
physics-correctness blockers remain.

## 17. Immediate execution packet

The active sequence is:

1. F-00 is closed as the plan/ledger reset without runtime change;
2. F-01 is closed with limits at crate-private dual-solver mechanics;
3. F-02 is closed with limits at the declared ideal two-leg FLRW model;
4. F-03 is closed with limits at the independently anchored six-channel Born
   kernel and standalone neutron freeze-out;
5. F-04 is closed with limits at the checked 9-species/12-reaction kernel;
6. F-05 is closed with limits at the first coupled minimal Rust BBN endpoint;
7. F-06 is closed with limits at the matched minimal private Rust
   number-of-record and actual source deflation;
8. F-07 is closed with limits at scalar finite-temperature QED thermodynamics;
9. F-08A is closed with limits at the independently checked zero-temperature
   Coulomb plus resummed radiative weak block;
10. F-08B is closed with limits at the additive finite-nucleon-mass
    Fokker-Planck block with `delta_kappa=0`;
11. F-08C is closed with limits at physical weak magnetism through that shared
    finite-mass algebra;
12. F-08D is closed with limits at the complete true-photon plus differential-
    bremsstrahlung plus `L1` plus `L2+3` directional thermal-radiative block
    and its conditional 12-reaction endpoint; strict standalone `X_n`
    precision authority remains blocked;
13. F-08N is closed with limits at Rabbit's selected 31-row AC2024 central-rate
    network, nested 31-minus-12 effects, and dual-solver corrected endpoint;
    rate uncertainty and F-08D strict precision remain unresolved;
14. F-09 is closed with limits at the finite-mass FD/Pauli thermal moment and
    its private three-temperature endpoint;
15. F-10A is closed with limits at the collisionless `q=ap` distribution and
    FLRW consumer;
16. F-10B0 is closed with limits at the measured direct thermal-endpoint wall
    reduction;
17. F-10B1 is closed with limits at the first coarse conservative electron
    spectral action and private FLRW endpoint;
18. F-10B2 is closed with limits at the logit-coordinate, resolution-selected
    electron endpoint and bounded direct-support derivative, opening only
    classical diagonal neutrino--neutrino collisions in F-10C;
19. F-10C0 validates the same-flavour identical-neutrino CM component and
    executes it in the private dual-solver endpoint;
20. F-10C1 selects the fixed positive exponential N48 representation within a
    declared five-profile/two-auxiliary-rule `2.1%` envelope. A prepared static-
    geometry candidate was removed after a matched `9.195%` cold-wall reduction
    missed the cost-policy threshold and raised RSS. F-10C remains open at the
    measured cold-endpoint wall, incomplete row 3, and seven wholly absent rows.

The current implementation now contains independently checkable solver, ideal
FLRW, Born weak/freeze-out, named-network, and coupled-minimal-endpoint
substrates plus the bounded F-07 scalar-QED, F-08A zero-temperature CCR,
F-08B first-order finite-mass/no-weak-magnetism, F-08C physical weak-magnetism,
F-08D thermal-radiative, F-08N selected-network, F-09 finite-mass FD/Pauli
thermal collision, F-10A/F-10B1/F-10B2 isotropic distribution layers, and the
bounded F-10C0 same-flavour self-collision component and F-10C1 profile-bounded
N48 radial representation. F-10 is the only active implementation slice. Its
current spectral endpoint is test-private: the electron grid-20 refinement is
a same-code BDF anchor, N48 is not arbitrary-distribution or continuum
authority, while row 3 is partial and seven neutrino--neutrino rows are wholly
absent. The accepted corrected
abundance number-of-record profile therefore
remains unchanged: Rabbit's custom selected 31-row central-rate network,
first-order nucleon-mass expansion, and ideal instantaneous decoupling. F-08D
strict standalone authority remains blocked; rate uncertainty, converged
spectral transport, and independent QED-on endpoint validation remain absent,
so the new output is not precision-standard, runtime-promoted, or
public-production authority.
