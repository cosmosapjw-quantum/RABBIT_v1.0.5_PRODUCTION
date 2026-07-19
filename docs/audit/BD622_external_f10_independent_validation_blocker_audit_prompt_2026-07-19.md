# BD622 external F-10 independent-validation blocker audit

Use this document as the complete instruction set for an external auditor who has live read access to the repository. Do not replace it with a short code review. The required product is a reproducible mathematical, physical, numerical, and algorithmic diagnosis of the repeated F-10 independent-validation failures, followed by concrete replacement-method advice.

## 1. Role and required standard

You are an independent hostile-but-constructive audit team for a scientific BBN code. Work as four named review roles and then reconcile their findings:

1. **Early-universe kinetic theorist** — derives the classical flavour-diagonal neutrino collision operator, reaction catalogue, equilibrium laws, conserved moments, entropy production, and FLRW energy exchange.
2. **Discrete kinetic-method specialist** — audits quadrature, interpolation/deposition duality, Galerkin/Petrov structure, conditioning, conservation, covariance, positivity, and candidate replacement discretizations.
3. **Stiff-ODE and floating-point specialist** — audits binary64 error propagation, Jacobians, BDF/Rodas5P/Radau interfaces, event finding, transforms, tolerances, cache exactness, and stiffness separation.
4. **Reproducibility and evidence adjudicator** — reconstructs decisions D-025 through D-029, distinguishes existing evidence from assertions, and prevents post-output gate changes or claim inflation.

Do not provide hidden chain-of-thought. Provide checkable derivations, equations, intermediate identities, source references, residual definitions, condition estimates, and reproducible commands or prospective test specifications sufficient for another expert to verify every conclusion.

The final report should be readable as a third-party scientific review, not an internal project-status summary. Return the executive summary in Korean if you can do so accurately; equations and technical analysis may remain in English.

## 2. Repository and observed audit snapshot

- Repository: `/home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION`
- Observed branch: `bd612-remediation`
- Observed HEAD: `0b2339c676dba6f36dbf0850419c4d78e1a5f907`
- Observed shared-context version: `d9515008ac63edd39547b354b2a80f5f56df244e840c179b81e75fd1655a46a8`
- Active evidence run: `.agent-harness/runs/run-20260718-f10-maxent3-rejected-final/`

The worktree was dirty at prompt creation. Begin by recording the actual branch, HEAD, working-tree status, current context version, platform, compiler/interpreter versions, and UTC timestamp. Do not clean, reset, stage, commit, or overwrite anything. Treat the snapshot above as a drift detector, not as authority if the live checkout differs.

This audit is **read-only by default**. The request to audit does not authorize implementation, a new collision execution, GL48/GL64 comparator output, Jacobian/Radau construction, trajectory, endpoint, or RABBIT physical output. If such evidence is necessary, specify a prospective experiment and the exact owner authorization required; do not execute it.

## 3. Mandatory first reads, in order

Read each file completely before drawing conclusions:

1. `AGENTS.md`; `docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`; `bbn_codex_anti_drift_cost_effective_policy.md`.
2. `.agent-harness/generated/CONTEXT_PACK.md`.
3. `.agent-harness/context/FROZEN_DECISIONS.md` (especially D-025 through D-029), `GATE_REGISTRY.json`, and `CLAIM_REGISTRY.jsonl`.
4. `docs/harness/PROJECT_STATE.md`, `CLAIM_LEDGER.md`, `VALIDATION_LEDGER.md`, and `NEXT_SESSION_PROMPT.md`.
5. `docs/TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md` (especially the F-10 independent-validation boundary) and `docs/IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md`.
6. Under `docs/RABBIT_report/sections/`: `07_the_collision_integral_general_structure.tex`, `08_the_hannestad_madsen_collision_kernel.tex`, `09_pair_annihilation_and_creation.tex`, `11_incomplete_decoupling_and.tex`, `13_weak_n_p_interconversion_rates.tex`, `14_qed_equation_of_state.tex`, `16_solver_architecture.tex`, `17_the_full_coupled_system.tex`, `20_validation.tex`, `A01_physical_constants.tex`, `A03_weak_coupling_constants.tex`, and `A05_full_phase_space_ray_boltzmann_equation.tex`.

`docs/jcap_revised_final.pdf` may be used as background only. Its Bianchi-I/F-11 material is outside this audit and must not be used to expand scope.

## 4. Hard scope and claim discipline

The audited target is the smallest classical, flavour-diagonal, zero-lepton-asymmetry, flat-FLRW, no-QKE full-spectral neutrino-decoupling slice needed to clear `G-F10-INDEPENDENT-FLRW`.

The following are binding:

- QKE, oscillations, density matrices, Bianchi-I/F-11, public production promotion, new inference surfaces, and new wrappers are **FORBIDDEN**.
- Rust AOT is the implementation and repeated-run target.
- SciPy/BDF remains the temporary number-of-record until independent endpoint authority passes.
- JAX is a frozen local parity/AD/Jacobian oracle, not a forward-development target.
- Whole-program FortEPiaNO one-to-one authority is **DEPRECATED** under D-025. Its classical diagonal slice is contextual evidence only.
- Existing invariant and parity tests are necessary but not sufficient for independent physical validation.
- A favourable segment-only benchmark is not endpoint progress.
- Do not weaken, replace, reinterpret, or tune a registered norm or cap after seeing its output. A gate may be challenged only as a separate meta-finding with a prospective alternative.
- Do not treat harness process status as physical evidence.

Use exactly these status words when applicable: **IMPLEMENTED**, **VALIDATED**, **DERIVED**, **SPECIFIED**, **PROPOSED**, **SPECULATIVE**, **DEPRECATED**, and **FORBIDDEN**.

## 5. Current claims to verify, not trust

Reconstruct the evidence chain yourself. The repository currently states:

| Item | Reported state requiring verification |
|---|---|
| Rust catalogue | `G-F10-CATALOGUE = pass`; nine frozen zero-lepton rows, four production topology contractions, seven isolated folded-channel oracles. |
| Rust performance | `G-F10-PERFORMANCE = pass`; exact-point BDF Jacobian cache reports a 55.815% endpoint improvement, and four-topology aggregation reports a 22.102% whole-endpoint improvement. |
| Scope | `G-F10-SCOPE = pass`; flat FLRW no-QKE only. |
| Independent validation | `G-F10-INDEPENDENT-FLRW = fail`. |
| Harness integrity | Reported fail because static validation passes while trusted hook activation and write attribution remain unproved. This is a process defect, not proof of a physics defect. |
| D-027 pointwise comparator | Common-FD total null passes, but self number, self energy, and electron exchange residuals are reported as `1.9782e-10`, `2.3350e-10`, and `1.0684e-7`, against `1e-10`, `1e-10`, and `1e-8`. Rejected as an evolving RHS. |
| D-028 Galerkin-Petrov comparator | All recorded static discriminators except native mu-tau relative-L-infinity covariance pass. Reported failure: `4.666064056497196e-10 > 1e-10`. GL64/Radau/trajectory/endpoint were correctly not run. |
| D-029 three-node relative-entropy route | Rejected before collision implementation. The fixed-triple negative-KL theorem is reported sound, but the local prior, target-dependent support selector, exact-node one-hot branch, continuity/Jacobian/covariance proof, and collision-specific binary64 ledger do not form one coherent semidiscrete method. |
| Regression count | Older evidence says `230/230`; the live source may contain a different test total. Determine whether this is simply historical evidence or a stale current-tree claim. Do not infer a pass without a current authorized run. |
| Audit-visible source risks | Verify that `electron_spectral.rs` overwrites a recognized common-FD action with exact zero, that the stopped Fort exporter occupies roughly `isotropic_boltzmann.rs:1531–3924`, and that this module's header still describes an obsolete incomplete catalogue. These are audit leads, not pre-adjudicated root causes. |

Do not collapse these into “the Rust solver is correct and the validators are bad.” At least three causal classes remain open:

1. a hidden Rust physics/normalization/catalogue error;
2. an independent discretization or binary64-conditioning failure;
3. an ill-posed or inadequately justified validation observable/gate.

Multiple causes may coexist.

## 6. Primary source and evidence map

### Rust production physics

- `native/rabbit_cpu/src/neutrino_self_spectral.rs`: `interpolation_bracket`, `outgoing_energies`, `averaged_s_invariant_dimensionless`, `azimuth_averaged_t_invariant_dimensionless`, `build_event_stream`, `interpolated_logit`, `stable_gain_minus_loss`, `bracket_outputs`, `accumulate_folded_channel`, and `evaluate_impl`.
- `native/rabbit_cpu/src/electron_spectral.rs`: `build_event_stream`, `conservative_explicit_action`, `exact_reference_state`, and `evaluate_isotropic_electron_spectral_action_impl`.
- `native/rabbit_cpu/src/isotropic_boltzmann.rs`: `IsotropicBoltzmannFlrwSystem` and its `OdeSystem` implementation.
- `native/rabbit_cpu/src/ode.rs`: `exact_jacobian_cache_key_matches`, `solve_bdf`, `rodas_attempt`, `solve_rodas5p`, and `weighted_norm`.

### Independent comparator

- `src/rabbit/decoupling/_independent_noqke.py`: grid/basis (53–110), transforms (137–176), EOS/thermodynamics (185–394), reaction catalogues (394–585), Pauli factor (671), quadrature/kinematics (706–897), collision matrices (910–1004), modal/native maps (1019–1029), event measure (1039), self/electron assembly (1055/1131), and evaluator (1236).
- `tests/test_independent_noqke_comparator.py`

### Existing accepted and failed evidence

- Accepted Rust catalogue/performance: `.agent-harness/runs/run-20260716-f10c2-perf/`.
- D-027: `run-20260718-f10-minimal-independent-noqke-r2-implementation/{artifacts/M1_POINTWISE_STATIC_RESULT.json,ADJUDICATION.md}` under `.agent-harness/runs/`.
- D-028: `run-20260718T095658Z/{artifacts/GL48_STATIC_R1.json,ADJUDICATION.md,results/A-CONSERVATIVE-METHOD-ADJUDICATION.json,results/A-GL48-COVARIANCE-ADJUDICATION.json}` under `.agent-harness/runs/`.
- D-029: `run-20260718-f10-static-fail-closeout/results/{A-THREE-NODE-MAXENT-DERIVATION.json,A-THREE-NODE-BINARY64-AUDIT.json,A-THREE-NODE-DESIGN-ADJUDICATION.json,A-F10-METHOD-REOPEN-AUDIT.json}` under `.agent-harness/runs/`.
- Bounded BD622 routing map: `.agent-harness/runs/run-20260718-f10-maxent3-rejected-final/results/A-BD622-EVIDENCE-MAP.json`; use it to locate evidence, not as a scientific verdict.

At prompt creation, the six listed files respectively hashed to `1c6600a8…63a0`, `501ae08a…05a`, `5ac3a3ea…ae5`, `78e141fd…bc1`, `535370c1…ac3`, and `abdc4e01…748`; full values are recorded in the bounded BD622 routing map below.

Recompute these hashes. If they differ, report drift and use the live files while keeping the historical artifacts tied to their recorded hashes.

## 7. Safe initial commands

These commands inspect existing state and evidence. They do not authorize a new physical run.

```bash
cd /home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION
git status --short --branch
git rev-parse HEAD
git log -12 --oneline --decorate
python3 .agent-harness/scripts/validate_harness.py
sha256sum native/rabbit_cpu/src/{neutrino_self_spectral,electron_spectral,isotropic_boltzmann,ode}.rs src/rabbit/decoupling/_independent_noqke.py tests/test_independent_noqke_comparator.py
python3 -m json.tool .agent-harness/context/GATE_REGISTRY.json
python3 -m json.tool .agent-harness/runs/run-20260718-f10-minimal-independent-noqke-r2-implementation/artifacts/M1_POINTWISE_STATIC_RESULT.json
python3 -m json.tool .agent-harness/runs/run-20260718T095658Z/artifacts/GL48_STATIC_R1.json
rg -n 'D-02[5-9]|G-F10|G-HARNESS' .agent-harness/context .agent-harness/runs
rg -n 'fn (build_event_stream|accumulate_folded_channel|conservative_explicit_action|solve_bdf|solve_rodas5p)|def (_assemble_self|_assemble_electron|evaluate_independent_collision_action)' \
  native/rabbit_cpu/src src/rabbit/decoupling/_independent_noqke.py
```

Do not run the full release suite, comparator, collision kernel, GL48/64, Radau, trajectory, or endpoint merely to “see what happens.” First finish the static audit. Put any proposed execution in the prospective experiment contract required below.

## 8. Competing hypotheses

Attempt to falsify each hypothesis rather than selecting one early:

- **H1 — production-correct/comparator-ill-conditioned:** Rust is physically and discretely consistent, while D-027 and D-028 fail because their representations do not preserve the same weak invariants or amplify roundoff during native mass inversion.
- **H2 — shared hidden physics error:** Rust and independent work share a convention, normalization, row-multiplicity, matrix-element, support, or EOS error that equilibrium nulls and aggregate invariants cannot expose.
- **H3 — coordinate mismatch:** The D-028 weak/modal action is accurate, but conversion to native action through a small-`y` factor such as `1/y^2` makes the registered covariance observable ill-conditioned.
- **H4 — floating-point reduction defect:** event ordering, cancellation, summation, interpolation endpoints, or label-dependent operation order produces a binary64 covariance defect without a continuum-physics defect.
- **H5 — gate-metrology defect:** the native relative-L-infinity `1e-10` co-gate does not measure a scientifically meaningful or numerically well-conditioned invariant. This can be a meta-finding only; it cannot retroactively pass D-028.
- **H6 — insufficient structural independence:** the comparator independently rewrites code but inherits enough catalogue, convention, or reduction structure that agreement would not constitute an independent validation.
- **H7 — solver is secondary:** because static collision checks fail before time integration, BDF/Rodas5P/Radau work cannot remove the present blocker, although solver defects may still affect the eventual endpoint.
- **H8 — provenance drift:** accepted historical evidence is sound for its hash-locked tree, but current documentation overstates it as a current-tree regression result.

Add hypotheses if evidence demands it. For every accepted or rejected hypothesis, provide the observation that discriminates it from its nearest competitor.

## 9. Required mathematical and physical audit

### 9.1 Re-derive the continuum operator

Starting from the invariant phase-space measure, derive the convention actually required by the code for each classical fermionic `1+2 <-> 3+4` channel:

\[
C_1[f] = \frac{1}{2E_1}
\int \prod_{i=2}^{4}\frac{d^3p_i}{(2\pi)^3 2E_i}
(2\pi)^4\delta^{(4)}(p_1+p_2-p_3-p_4)
\,\overline{|\mathcal M|^2}\,
\big[f_3f_4(1-f_1)(1-f_2)-f_1f_2(1-f_3)(1-f_4)\big].
\]

Do not assume this prefactor matches the implementation. Trace all spin averages/sums, identical-particle factors, particle/antiparticle degeneracies, flavour multiplicities, crossing conventions, weak couplings, powers of `2*pi`, and conversion from `df/dt` to the independent variable used by FLRW integration.

For every frozen reaction row, produce an equation-to-code ledger containing:

- physical reaction and all explicit species legs;
- row coefficient and the first-principles multiplicity derivation;
- matrix element and Mandelstam invariant convention;
- symmetry/identical-particle factor;
- target-leg versus global-leg interpretation;
- continuum dimensional scaling;
- Rust function/line range;
- independent-comparator function/line range;
- existing test or missing falsifier.

Independently enumerate the six neutrino/antineutrino species. Do not infer the catalogue by reading Rust row factors and restating them.

### 9.2 Conservation, detailed balance, and entropy

For a reaction with signed stoichiometric coefficients `nu_a`, verify continuum and discrete identities separately:

\[
\sum_a \nu_a q_a = 0, \qquad
\sum_a \nu_a E_a = 0,
\]

and derive the exact weighted discrete moments corresponding to neutrino number and energy. State whether each identity is eventwise, quadrature-level, row-level, catalogue-level, or only approximate after reduction.

For the fermionic entropy

\[
S[f] = -\sum_s g_s\int\frac{d^3p}{(2\pi)^3}
\left[f_s\ln f_s+(1-f_s)\ln(1-f_s)\right],
\]

derive the sign convention for collision entropy production. Show how an event contribution reduces to a nonnegative form proportional to `(F-R) log(F/R)` when applicable. Then determine whether each discrete candidate preserves that sign structurally or only passes sampled states.

Verify all of the following without treating one as a substitute for another:

- exact common-temperature FD detailed balance;
- zero-lepton particle/antiparticle symmetry;
- electron/muon/tau family and CP transformations;
- mu-tau label covariance in continuum and discrete coordinates;
- neutrino self-scattering number and energy moments;
- electromagnetic-neutrino exchange with equal and opposite first-law contribution;
- positivity and `0 < f < 1` under transforms and trial steps;
- expected weak-rate temperature scaling, including every power introduced by nondimensionalization;
- support and tail semantics when outgoing momenta lie outside the stored grid.

### 9.3 FLRW coupling and thermodynamics

Audit the coupling among scale-factor/time variables, `T_gamma`, `T_cm`, neutrino spectra, electromagnetic EOS, Hubble rate, and collision energy transfer. Check:

- units and dimensions at every interface;
- signs of EM energy loss and neutrino gain;
- initial and terminal temperature conditions;
- whether a collision equilibrium null implies the correct FLRW thermal limit;
- finite-electron-mass treatment and QED exclusions/inclusions actually in this F-10 slice;
- whether exact-reference shortcuts are mathematical identities or narrow numerical exceptions;
- whether the endpoint event is monotone and bracketed in both solver paths.

Do not use agreement in final `N_eff` alone to validate collision physics.

## 10. Required discrete-algorithm and floating-point audit

### 10.1 Interpolation/deposition duality

Write the semidiscrete weak form before inspecting implementation details. For each event, determine whether evaluation of outgoing occupations and deposition of signed leg contributions are adjoints under the chosen quadrature/mass matrix. Explicitly test on paper whether the interpolation weights reproduce constants and the energy coordinate and whether deposition preserves the corresponding number/energy moments.

Classify each method:

- D-027 direct target-leg point evaluation;
- D-028 Galerkin-Petrov/modal projection plus native mass inversion;
- D-029 three-node relative-entropy proposal;
- Rust folded event stream with bracket deposition.

For each, state what is conserved by construction, what is conserved only in exact arithmetic, what is sampled evidence, and what is unproved.

### 10.2 Conditioning and covariance

Derive a backward-error explanation for the D-028 observation. At minimum:

- write the modal-to-native mass inversion and identify any `1/y^2`, quadrature-weight, or basis-conditioning amplification;
- estimate the smallest-node amplification and a binary64 roundoff floor;
- separate physical action scale, cancellation scale, absolute perturbation, and relative-L-infinity denominator;
- decide whether `4.666064056497196e-10` is compatible with ordinary rounding, a reduction-order asymmetry, or a deeper discrete covariance defect;
- determine whether the favourable modal residual and failing native residual are mathematically compatible;
- explain what high-precision or compensated/pairwise summation experiment would distinguish the cases, without running it;
- assess whether the current `1e-10` native co-gate is scientifically justified and well-conditioned. If not, file a meta-finding and propose a prospectively frozen replacement; do not reclassify D-028.

Audit exact-node branches, support-selector ties, label permutations, stable sorting, event orientation, and reduction order. For D-029, either supply or disprove a single coherent proof of continuity/Lipschitz behaviour, permutation covariance, exact-node consistency, positive weights, moment feasibility, and Jacobian existence across selector changes. A theorem for one fixed support triple is insufficient unless the global selector contract is included.

### 10.3 Solver and Jacobian boundary

Even though the static collision blocker comes first, audit the eventual interface:

- state vector and complementary-log-log/logit chain rules;
- analytic versus numerical Jacobian completeness;
- exact `(t,state)` BDF cache identity and invalidation;
- BDF versus Rodas5P error norms and rejected-step behaviour;
- Radau independence and what would be shared with SciPy/BDF;
- event interpolation/root finding and terminal-state reproducibility;
- stiffness sources, spectral radius, positivity risks, and tolerance scaling by component;
- why an IMEX/AP integrator can address temporal stiffness but cannot repair an inconsistent spatial collision discretization.

Give the smallest prospective Jacobian directional-derivative contract that must pass before a trajectory is authorized.

## 11. Structural-independence audit

Build an explicit dependency graph for the Rust result and each independent comparator. Distinguish:

- universal physical constants and published formulas that may be shared;
- independently derived conventions and row catalogues;
- copied or mechanically translated algorithms;
- shared grids, quadratures, variable transforms, EOS, reaction reductions, test vectors, and acceptance metrics;
- evidence learned only after unblinding.

An implementation in another language is not structurally independent by itself. Conversely, using the same physical constants is not automatically a violation. State exactly which shared components would make an apparent agreement epistemically circular.

Recommend the minimum independent axes needed for endpoint authority: collision derivation, spatial discretization, time integrator, EOS/FLRW mapping, and output metrology. Explain which axes must be independent and which may be cross-checked rather than reimplemented.

## 12. Replacement-method review

Compare at least the following families; add a better family if justified:

1. **Positive conservative event/deposition or discrete-velocity method** with moment-reproducing deposition and explicit fermionic detailed balance.
2. **Conservative spectral or Galerkin/Petrov method** with a well-conditioned mass matrix and prospectively derived invariant enforcement.
3. **Entropy-stable renormalized moment, variational, or discontinuous-Galerkin method** with an explicit realizability/positivity contract.
4. **Matrix-free weak-form collocation/quadrature method** with constrained local solves and a separately justified stiff integrator such as IMEX/AP.
5. **High-precision or interval reference evaluator** used only as a validation oracle, not as the production endpoint.

For every family, provide:

- the actual semidiscrete equations, not a method name;
- what unknowns are evolved and in which coordinates;
- conservation and entropy proof obligations;
- positivity/Pauli-bound mechanism;
- permutation/CP/mu-tau covariance mechanism;
- tail and off-grid event treatment;
- binary64 conditioning and Jacobian regularity;
- expected complexity in grid size and reaction/event count;
- compatibility with independent Radau and the Rust endpoint;
- structural-independence strength;
- smallest source/test surface;
- pre-output falsifiers and clear abandonment criteria;
- estimated added/deleted/net lines and whether obsolete comparator surface can be deleted;
- blocker-movement ratio and cost-effectiveness verdict.

Treat post-hoc conservation projection as disallowed under the current decision unless you derive it prospectively as part of the method and separately obtain owner authorization. If proposing a Lagrange-multiplier conservative correction, explain whether it preserves positivity, entropy, covariance, and structural independence.

Do not use “maximum entropy” ambiguously. If recommending a local entropy closure, specify its primal/dual variables, feasible set, prior/measure, selector, and global regularity. If recommending Bryan-style maximum entropy, first identify the inverse problem, likelihood, prior, regularization, identifiability, and reconstructed object. Reject it as a category error if the forward nonlinear collision discretization has not genuinely been cast as an inverse problem.

Run a current web-CRAG pass for each proposed method: search broadly, retain primary papers/official documentation, record query and rejection criteria, and test every imported assumption against this fermionic FLRW operator. Start with:

- Gamba and Rjasanow, Galerkin-Petrov Boltzmann discretization: <https://arxiv.org/abs/1710.05903>
- Alonso, Gamba, and Tharkabhushanam, conservative spectral approximation: <https://arxiv.org/abs/1611.04171>
- Abdelmalik and van Brummelen, entropy-stable renormalized moment/DG method: <https://arxiv.org/abs/1602.01312>
- Filbet, Hu, and Jin, asymptotic-preserving quantum Boltzmann method: <https://arxiv.org/abs/1009.3352>
- Hannestad and Madsen, classical no-oscillation neutrino decoupling: <https://arxiv.org/abs/astro-ph/9506015>
- Mangano et al., precision neutrino decoupling: <https://arxiv.org/abs/hep-ph/0506164>
- Froustey and Pitrou, no-oscillation BBN consequences: <https://arxiv.org/abs/1912.09378>
- Froustey, Pitrou, and Volpe, full QKE treatment: <https://arxiv.org/abs/2008.01074> — use only its diagonal/zero-mixing limit for context; QKE is out of scope.

A citation or analogy does not grant method authority. State which assumptions transfer to the fermionic cosmological `2->2` operator and which do not.

## 13. Prospective falsifier package

Without executing prohibited physics, specify an owner-reviewable next experiment package. It must freeze inputs, order of operations, norms, caps, and failure policy before output exists. Include at least:

1. a first-principles explicit-species reaction/multiplicity oracle;
2. common-FD equilibrium nulls at multiple temperatures;
3. resolved non-equilibrium states separating self-scattering and electron exchange;
4. exact conserved weak moments and EM-neutrino first-law closure;
5. entropy production with sign and normalization fixed;
6. CP, family, and exact mu-tau permutation transforms;
7. support-edge, exact-node, selector-tie, and tail cases;
8. arbitrary event/reduction order plus naive, pairwise, compensated, and high-precision arithmetic;
9. native, weak/modal, and physically weighted norms with denominators frozen independently of output;
10. directional Jacobian checks on both sides of any branch or selector boundary;
11. only after static pass: one bounded 10-to-3 MeV segment;
12. only after segment pass and fresh authorization: a full independent endpoint comparison.

For each test, state the mathematical identity, input construction, independent oracle, units, absolute and relative norm, cap derivation, expected conditioning, pass/fail rule, artifact schema, and action on failure. Separate diagnostic localization metrics from binding gates.

## 14. Required final deliverables

Return one self-contained report in this exact order:

1. **Snapshot/evidence manifest** — branch, HEAD, dirty state, context/tool versions, hashes, commands run, and skips with reasons; then a one-page **executive verdict** separating physics, discretization, metrology, and process defects.
2. **Decision chronology** — D-025 through D-029 as `attempt -> discriminator -> observation -> decision -> uncertainty`; then a causal **failure tree**.
3. **Equation-to-code ledger** and **continuum convention audit** — reactions/interfaces, assumptions, dimensions, signs, degeneracies, limits, and conditions.
4. **Discrete proof audit** and **floating-point/conditioning analysis** — proved, sampled, failed, and missing obligations, including a quantitative D-028 backward-error estimate.
5. **Solver/Jacobian audit** and **structural-independence graph** — separate the static blocker from later endpoint risk and show shared versus independent axes.
6. **Ranked P0–P3 findings** and a **replacement-method matrix** — deduplicate by root cause; choose one design and one fallback.
7. **Prospective experiment contract** and **bounded patch sequence** — at most five patches; each moves a physics/solver/endpoint blocker or deletes more obsolete surface than it adds.
8. **Owner decision memo** and **red-team objections** — exact authorization boundary, strongest counterarguments, and evidence that would reverse the recommendation.

Every substantive finding must use this schema:

```text
Finding ID / priority / claim status
Symptom / Source / Mathematical or physical mechanism:
Exact evidence path and line or function / Assumptions:
Consequence / Remedy:
Proof or prospective falsifier / Confidence and what would change it:
```

End with exactly one primary verdict:

- `RUST_PHYSICS_SUSPECT`
- `INDEPENDENT_DISCRETIZATION_SUSPECT`
- `GATE_METROLOGY_SUSPECT`
- `MULTIPLE_ROOT_CAUSES`
- `INCONCLUSIVE_NEEDS_NEW_PROSPECTIVE_DESIGN`

Then state separately:

- whether `G-F10-INDEPENDENT-FLRW` must remain failed;
- whether D-029 should remain closed;
- the smallest owner-authorized next design slice, if any;
- whether the recommended work is cost-effective under the repository policy.

## 15. Prohibited audit shortcuts

Do not:

- conclude from source aesthetics, test count, equilibrium nulls, or endpoint agreement alone;
- call parity, self-consistency, or a static harness check independent validation;
- translate the Rust algorithm into another language and call it independent;
- rerun a failed method with a looser cap, new reduction order, symmetrization, or favourable norm;
- propose another wrapper, manifest, readiness probe, hash gate, dashboard, or segment-only benchmark as blocker progress;
- optimize an already-cheap segment while the independent physics gate remains failed;
- confuse BDF/Rodas5P speed with collision correctness;
- use FortEPiaNO or QKE scope to bypass D-025;
- implement a proposal or generate new physics output without explicit owner authority;
- invent token counts. Report `UNAVAILABLE` and why when the harness exposes no exact counter;
- report **VALIDATED** unless the stated command, proof, independent calculation, or artifact actually exists.

The audit succeeds only if it explains why the failures recur, identifies which uncertainty class remains live, and gives a mathematically coherent, prospectively falsifiable route that can either clear the independent FLRW gate or justify abandoning the route before more code is written.
