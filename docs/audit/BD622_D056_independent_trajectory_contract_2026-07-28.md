# BD622 D-056 — Independent trajectory/endpoint contract (2026-07-28)

Prospectively frozen before any trajectory output, under the D-052 omnibus
grant, stage 3. Object: a structurally independent collision-coupled
isotropic FLRW trajectory to the cold endpoint on the accepted comparator
bytes `760a7c04` (D-053 final catalogue, D-055 metrology-bounded).
Decides `G-F10-INDEPENDENT-FLRW`.

## Structural independence

Different radial rule (affine GL48-Y24 vs the Rust positive exponential
N48 `y = -3 ln(1-t)`), different angular quadratures, different evolved
variable (complementary log-log vs exact logit), different integrator
(SciPy BDF vs the Rust BDF/Rodas5P pair), different language and
libraries (NumPy/SciPy vs Rust), independent EOS (adaptive quad
finite-mass e+/e- vs the Rust tables), independent constants block.
Shared: physical conventions (Hannestad-Madsen with the
Dolgov-Hansen-Semikoz correction) and the frozen catalogue physics —
the things under test.

## Frozen formulation

State `(c, T_gamma, t)` with `c` the `(3, 48)` pair cloglog coordinates;
independent variable `N = ln a`, `T_cm(N) = 10 e^{-N}` MeV.

- `dc/dN = pair_rate / (H * cloglog_chain_factor(c))` with
  `pair_rate = 0.5 (total[2i] + total[2i+1])` from
  `evaluate_independent_collision_action` (native rows, per unit time).
- `dT_gamma/dN = [ -3 (rho_EM + P_EM) + q_EM / H ] / (d rho_EM / dT)`
  with `q_EM = electron_bath_energy_transfer` (the module's first-law
  partner of the neutrino energy rate) and the EOS from
  `electromagnetic_eos_adaptive`.
- `dt/dN = 1 / H`, converted to seconds by `MEV_TO_INVERSE_SECONDS`.
- Initial condition at `N = 0`: exact equilibrium `u = -y`
  (`c = pair_logits_to_cloglog(-nodes)`), `T_gamma = T_cm = 10` MeV,
  `t = 0`. Terminal event: `T_gamma = 0.005` MeV.
- Trajectory quadrature config (frozen; recorded limitation):
  `incoming_polar_order = 4`, `final_polar_order = 4`,
  `final_azimuth_order = 4`, `electron_radial_order = 24`.
- Integrator: `scipy.integrate.solve_ivp` BDF, `rtol = 1e-6`,
  `atol = 1e-9`, FD Jacobian.
- Endpoint observables: `N_end`, `t_end` (s),
  `N_eff = (8/7)(11/4)^{4/3} rho_nu / rho_gamma` with
  `rho_gamma = pi^2 T_gamma^4 / 15`, and the per-flavour endpoint
  energy-moment enhancements over the frozen comoving equilibrium.

## Frozen acceptance checks (anchors: Rust N48 no-cache endpoint)

Rust anchors (report section 11.7 / F10C1): BDF
`(N, t, N_eff) = (7.9367214190, 52680.2048 s, 3.0333103242)`; Rodas5P
`(7.9367373740, 52680.8451 s, 3.0331291370)`.

| ID | Statement | Threshold |
|---|---|---|
| T1 | Solver completes with the terminal event reached; every sampled occupation strictly inside `(0, 1)`; no evaluator fail-closed error | success |
| T2 | `abs(N_eff - 3.0333103242) <= 3e-3` | band |
| T3 | `abs(N_end - 7.9367214190) <= 3e-3` | band |
| T4 | `abs(t_end - 52680.2048) <= 15 s` | band |
| T5 | Blockwise ordering: endpoint e-flavour energy-moment enhancement `>` mu/tau enhancement `> 0`, and the mu and tau enhancements agree to `<= 1e-6` relative (flavour-universal heavy block) | listed |
| T6 | First-law consistency sampled along the trajectory: `first_law_residual <= 1e-8` at every sampled RHS evaluation | per sample |

Band rationale (frozen): the Rust solver-pair internal spread is
`~1.8e-4` in `N_eff` and `0.5 s` in `t`; the reduced trajectory angular
order (4/4 vs the static 12/12) and `rtol 1e-6` are budgeted at one
additional order; `3e-3` and `15 s` bound both without approaching the
`N_eff` physical scale (`0.033` above 3). A PASS within these bands from
a structurally independent stack is discriminating; tighter precision
claims are out of scope. Pre-freeze quadrature discriminator (recorded,
run before this freeze under the already-authorized static surface): at
a mid-annihilation distorted state (`T_cm = 2`, `T_gamma = 2.7`,
e-heated logits), the 4/4/24 config agrees with the 12/12/48 static
config to `4.454e-4` relative on the absolute energy rate and
`3.937e-5` on the net electromagnetic energy transfer — the band budget
therefore carries about two orders of margin.

Decision rule (frozen): all of T1..T6 pass -> `G-F10-INDEPENDENT-FLRW`
flips fail -> pass by single-writer update citing this run, with the
reduced-angular-order limitation adjudicated and recorded. Any failure:
gate stays FAIL, record, no band refitting. Mechanical errors (solver
stall, exception) rerunnable and recorded; scientific misses are final
for this contract.

## Verification harness

`scripts/audit/d056_independent_trajectory.py` (frozen with this
contract; BLAS pins; module sha self-check; progress log; JSON report).
