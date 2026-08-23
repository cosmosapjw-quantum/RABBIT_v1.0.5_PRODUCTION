# A-PM-PHYS research record

## Frozen question

Which physics-specific remedies are defensible for the sealed RABBIT ODE findings PM-H1, PM-H3, and PM-H5 concerning physical domains, abundance positivity/simplex constraints, occupation bounds, thermodynamic monotonicity, terminal-temperature events, raw-state failure semantics, FLRW/LRS limits, and weak/network equilibrium or conservation?

## Frozen hypotheses

- H-POS: A solver-facing invariant-domain design can prevent or reject nonphysical abundance and occupation states without silently changing the physical model. Prediction: authoritative numerical/physics sources support positivity-preserving coordinates, projections, or invariant-region controls with explicit conservation caveats. Falsifier: the source requires clipping that destroys the conserved totals or detailed balance for the assigned state equations.
- H-EVENT: A continuous, direction-aware terminal event with dense-output root isolation is more physically faithful than endpoint polling. Prediction: authoritative solver documentation requires a continuous event function and sign crossing, while cosmological monotonicity supplies a physics-side direction check. Falsifier: the assigned temperature variable is not monotone or is not continuous over accepted steps.
- H-RAW: Raw-state validation must precede any clipping/sanitization used for constitutive evaluation, and failures must be surfaced rather than converted into solver success. Prediction: authoritative solver interfaces expose failure/status semantics and the repository code has observable finite/domain/invariant boundaries. Falsifier: all admissible intermediate solver states are guaranteed physical by construction and no sanitization exists.
- H-LIMIT: FLRW and weak/network equilibrium or conservation limits provide falsifying tests for Type-I/BBN mitigation. Prediction: primary cosmology/BBN sources identify the isotropic limit, detailed balance/NSE, baryon conservation, charge neutrality, and monotone cooling/entropy relations. Falsifier: the assigned code variables do not represent those limits or the source assumptions mismatch the frozen model.

## Frozen methods

- M-LOCAL (confirmatory): Extract only PM-H1/PM-H3/PM-H5 and their cited implementation locations from the sealed audit and named source files.
- M-PRIMARY (confirmatory): Consult primary papers or authoritative technical sources for invariant-domain ODE integration, positivity/simplex preservation, Fermi occupation bounds, event root finding, thermodynamic monotonicity, FLRW/LRS limits, and BBN equilibrium/conservation.
- M-MAP (confirmatory): For each assigned claim, map equation/variable, remedy, touchpoint, failure risk, and a concrete falsifying BBN-shaped test.
- M-SCHEMA (confirmatory): Validate the result against the sealed template and assignment identity; record exact commands and hashes.

## Stopping rules

Stop when each of PM-H1, PM-H3, and PM-H5 has at least one source-bound remedy and one falsifying test, all material claims are evidence-linked, the result envelope validates, and no out-of-scope execution or edit has occurred. Stop earlier with `inconclusive` or `error` if a sealed finding cannot be mapped without forbidden discovery, a primary source cannot be verified, or the result contract cannot be satisfied.

## Evidence log

Pre-registration frozen before local finding extraction or web retrieval.

- E-BOOTSTRAP: `verify_assignment.py` returned `VERIFY: PASS`; assignment SHA-256 `2790d2d79a46f975bfc060319382a9cb1f570bcc15a674bf1205654ae6950426`, role SHA-256 `9675bef90b109fc1ce0214509c8c27b08526bdb0e68dac2dcbb99221b96eb5e1`, template SHA-256 `936c043e8b3d7dd37516b04a470e39c08f9a36409619e8b859f8c1ede65ad1b0`.
- E-MISSING-INPUT: `wc -l` reported `src/rabbit/network/bbn.py: No such file or directory`; the confirmatory command `test -f src/rabbit/network/bbn.py` exited 1. The assignment lists that path as a required input, and the injected bootstrap requires an error stop when any required file is missing.

## Dead ends and pivots

- The sealed local-input pass stopped before implementation mapping and before any web retrieval. Substituting a similarly named network module would violate the targeted discovery contract; no pivot was permitted.

## Final verdict

ERROR / STOP_INVALID. PM-H1, PM-H3, and PM-H5 remain inconclusive because one sealed required implementation input is absent at current HEAD. No primary-source remedy was claimed and no production/shared file was edited.
