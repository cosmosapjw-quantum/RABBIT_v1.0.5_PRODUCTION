# BD641 — D-081R1D4 combined six-species collision-action result

Date: 2026-09-03  
Classification: `SIX_SPECIES_COMBINED_COLLISION_ACTION_ADMITTED`  
Verdict: `PASS_WITH_ORDER8_SIX_SPECIES_COMBINED_ACTION_SCOPE`

## 1. Executed identity

The authoritative exact-head run is:

```text
workflow run:                    33747522233
workflow job:                    100623328563
validated workflow head:        6baa466b199087d2d19ee5feab37d0575b089e08
validated workflow tree:        ef2b4b1426381b0b9ed7efd4597d46171faf470a
validated implementation parent: 8aeae0ff29d15aa45eba39c7ff153193ad7ec948
validated implementation tree:   6a4b92ccbea69fb8332c1c93ce7131ac7ef20cbe
predecessor D-081R1D3 head:       a604cc7af832c799687d29ba05765972f7ba0c9d
predecessor D-081R1D3 tree:       415a5ba83ed7b24fe941be45352e2ac037c87158
```

The exact-head run applied both bounded repair scripts as `NOOP`, proving that the committed implementation and tests already contained the admitted definitions. The later result, receipt, cleanup, and pull-request commits are documentary or repository-hygiene changes and are not represented as independently executed scientific heads.

## 2. Frozen authority and environment

```text
private Python comparator Git blob:
de44feee0aa484abe26976c7dc34c579643005b5

full collision fixture Git blob:
c94d2e72a1f8300b7c20c9c793417a5c4a5fa302

Cargo.lock Git blob:
a1b5035da5c20712d1a2a4ab077da255ff94a014

Rust toolchain:
1.94.1

offline vendor artifact:
9838506413

offline vendor outer SHA-256:
391b45bbf6446dc9144bc39321723eda18240d7c8751f45887f9492ffc66f78a

vendor package directories:
174
```

Validated implementation blobs:

```text
native/rabbit_cpu/src/f10_combined_action.rs
80963b73a498be080527c78cd6207e650117f836

native/rabbit_cpu/src/f10_combined_action_tests.rs
6f6b5a0ccec89b08806ee5c11a0a537b5a0594e7

native/rabbit_cpu/src/lib.rs
111a03b5307d88e6d82710dfefa14f5fee64d26a

.github/workflows/d081r1d4_combined_action.yml
1a6e1c14a58625972c5dc99b07a63cdc92fbae1d
```

## 3. Admitted operator

D-081R1D4 composes, without modifying, the separately admitted D-081R1D2 neutrino self action and D-081R1D3 finite-electron-mass action. At one fixed state and one fixed order-8 grid,

\[
C_{\mathrm{tot}}[u;T_{\rm cm},T_\gamma]
 = C_{\mathrm{self}}[u;T_{\rm cm}]
 + C_{e^\pm}[u;T_{\rm cm},T_\gamma].
\]

The addition is implemented independently in modal and native representations,

\[
\widehat C_{\mathrm{tot},sa}
 = \widehat C_{\mathrm{self},sa}
 + \widehat C_{e^\pm,sa},
\qquad
C_{\mathrm{tot},si}
 = C_{\mathrm{self},si}
 + C_{e^\pm,si},
\]

for six explicit neutrino and antineutrino species, eight spectral degrees per species, and the frozen species ordering. The admitted event substrate comprises 27 self-interaction events and 15 electron-sector events, for 42 frozen events in total.

The combined layer also assembles:

- total signed and absolute number moments;
- total signed and absolute energy moments;
- summed whole-reaction-domain rejection counts;
- summed matrix-roundoff correction counts and their largest correction;
- neutrino and electromagnetic energy-transfer ledgers;
- first-law residual;
- event-space and node-space neutrino H-functional rates;
- electromagnetic H-functional rate and entropy production;
- entropy-duality residual;
- self-event energy residual;
- charge-conjugation and muon–tau diagnostics;
- typed, transactional propagation of either component failure.

It performs no new collision quadrature and introduces no new event coefficient.

## 4. Exact offline admission

The authoritative run passed:

```text
D-081R1D3 ancestry
frozen comparator identity
frozen full-action fixture identity
frozen Cargo.lock identity
Rust 1.94.1 installation
174-package offline vendor recovery and SHA-256 check
cargo metadata --locked --offline
cargo fmt --all
cargo check --release --locked --offline
four focused combined-action tests
cargo clippy --all-targets --all-features --locked --offline -- -D warnings
```

Focused result:

```text
4 passed; 0 failed; 0 ignored; 309 filtered out
focused runtime: 16.18 s
```

Passing tests:

```text
frozen_full_action_contract_is_exact
combined_action_matches_every_frozen_python_case
component_addition_and_physical_ledgers_are_load_bearing
component_failures_propagate_without_a_partial_result
```

The tests establish frozen Python parity of the modal and native total action, both admitted components, total moments, counter aggregation, energy and entropy ledgers, and recorded diagnostics for all three frozen states. They also verify exact component addition, equilibrium near-null behaviour, thermal restoring flow, nontrivial muon–tau response, component-omission and wrong-sign discriminators, full modal/native muon–tau swap equivariance on the distinct state, and transactional failure semantics.

## 5. Conditioned symmetry metrology

The frozen Python comparator defines pairwise relative native residuals. For the muon–tau pair averages

\[
\bar C_\mu=\frac{C_{\nu_\mu}+C_{\bar\nu_\mu}}{2},
\qquad
\bar C_\tau=\frac{C_{\nu_\tau}+C_{\bar\nu_\tau}}{2},
\]

it records

\[
r_{\mu\tau}
=\frac{\|\bar C_\mu-\bar C_\tau\|_\infty}
{\max(\|\bar C_\mu\|_\infty,\|\bar C_\tau\|_\infty)}.
\]

At the frozen equilibrium state the reference numerator and scale are

```text
numerator: 1.3329255900167337e-38
scale:     1.0941546646861551e-35
ratio:     1.2182241076484988e-3
```

so direct relative comparison of the derived scalar is cancellation-sensitive even when the underlying arrays satisfy their frozen hybrid parity contract.

Let \(n_A,s_A\) denote the actual numerator and scale, \(n_E,s_E\) the expected quantities, and \(\delta_\mu,\delta_\tau\) the actual-versus-reference sup-norm errors in the two pair-averaged arrays. The exact identity

\[
\frac{n_A}{s_A}-\frac{n_E}{s_E}
=\frac{n_A-n_E}{s_A}
 +\frac{n_E(s_E-s_A)}{s_A s_E}
\]

and the reverse triangle inequality give

\[
|n_A-n_E|\le \delta_\mu+\delta_\tau,
\qquad
|s_A-s_E|\le \max(\delta_\mu,\delta_\tau),
\]

hence

\[
|r_A-r_E|
\le
\frac{\delta_\mu+\delta_\tau}{s_A}
+\frac{n_E\max(\delta_\mu,\delta_\tau)}{s_A s_E}
+B_{\rm roundoff}.
\]

The exact identity was checked with a stateless Wolfram Language evaluation. The admitted test does not widen the frozen array-parity tolerance. It instead checks that each stored scalar agrees with the scalar recomputed from its own array and that the actual/reference difference lies inside the array-propagated condition bound. The forward residual remains recorded.

Charge-conjugation residuals use the corresponding pair-local normalization and conditioned array-error gate. The load-bearing physical flavour-symmetry test remains the distinct-state equivariance relation

\[
C_{\rm tot}[P_{\mu\tau}u]
=P_{\mu\tau}C_{\rm tot}[u]
\]

in both modal and native representations, together with invariance of the scalar muon–tau residual. This avoids treating a near-null ratio as the sole symmetry test.

No collision coefficient, event catalogue, state, grid, quadrature, fixture, array-parity tolerance, or physical acceptance threshold was changed by these metrology repairs.

## 6. Failure-preserving history

The development history is retained rather than rewritten as an uninterrupted pass.

- Run `33739325198`: two focused tests passed and two failed. The remaining failures localized a charge-conjugation normalization mismatch and a non-discriminating component-omission scale.
- Run `33741259592`: three tests passed and one failed after the first bounded repair. The remaining mismatch was the pair-local charge-conjugation scalar.
- Run `33742086879`: three tests passed and one failed after conditioned charge-conjugation metrology. The remaining direct muon–tau scalar comparison was `9.35958036985416048e-3` versus `1.21822410764849880e-3` in a cancellation-sensitive near-null state.
- Run `33746969006`: the bounded muon–tau condition gate and distinct-state swap-equivariance gate passed; all four tests and strict Clippy passed; the workflow committed the two repaired core files.
- Run `33747522233`: exact committed-source replay; both repair scripts reported `NOOP`; all four tests and strict Clippy passed.

The component-omission negative controls were repaired by scaling each omission against the omitted component itself. No physical expression or threshold changed.

## 7. Literature boundary

The literature review supports the validation strategy but does not replace the frozen repository authority.

- Blaschke and Cirigliano, arXiv:1605.09383, derive collision terms with explicit flavour, particle/antiparticle, and spin structure and discuss isotropic reductions.
- Froustey, Pitrou, and Volpe, arXiv:2008.01074, retain full collision terms in early-Universe neutrino decoupling and solve the coupled kinetic system with a direct Jacobian.
- Pareschi and Rey, *SIAM Journal on Numerical Analysis* 60 (2022), DOI 10.1137/21M1423452, formulate moment-preserving Fourier–Galerkin spectral methods for the Boltzmann equation.
- Conservative deterministic spectral Boltzmann work by Gamba and Haack likewise treats collision invariants and conservation as explicit numerical constraints.

These sources support explicit flavour routing, conservation and entropy ledgers, weak or moment validation, and symmetry-equivariance tests. They do not authorize RABBIT's exact 42-event catalogue, weak-coupling encoding, quadrature bytes, support mask, modal normalization, matrix-roundoff policy, species ordering, fixture, or numerical thresholds. Those remain frozen-oracle contracts.

## 8. Claim ceiling

This result admits only the frozen order-8 static six-species combined collision action and its stated diagnostics.

It does **not** admit:

- the retained order-60, 182-component packed right-hand side;
- a production Python/PyO3 public surface;
- analytic Rust JVP or Jacobian assembly;
- `diffsol` `OdeSystem` integration;
- a stiff trajectory;
- solver robustness or performance;
- endpoint time, spectra, or \(N_{\rm eff}\);
- movement of `G-F10-INDEPENDENT-FLRW`;
- public-production or publication authority.

## 9. Next admissible node

The next node is `D-081R1E`: retained order-60 packed-RHS parity.

It must freeze the 182-component state ordering, compose the admitted collision action with the cosmological and plasma rows, and verify the spectral block, \(T_\gamma\) row, elapsed-time row, energy-density/Hubble reconstruction, first law, admissible-domain refusal, and Python parity before any analytic JVP, Jacobian, `diffsol`, or trajectory claim is attempted.
