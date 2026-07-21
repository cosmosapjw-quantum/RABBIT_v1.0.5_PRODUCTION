# Canonical Shared Context Pack

Context version: `e7f249ed6d84f8bc8d74b83376e176a5e4d6adb021936614056600129c9ea199`
Built at: `2026-07-21T15:48:36+00:00`

This pack contains only the shared Tier-0 context. Assignment-specific context and sibling results are intentionally excluded.

---

## Source: `.agent-harness/context/SHARED_CONTEXT.md`

SHA-256: `92e6bb799f5a740d31f35b1910743452e9d9ae343516df56dd3ad02d5955eff5`

# Shared Context — RABBIT Rust-first FLRW/F-10

## Project objective

- Scientific objective: complete and validate the crate-private Rust-first collision-coupled isotropic neutrino Boltzmann FLRW endpoint through F-10, then stop.
- Current milestone: F-10C2 implements the frozen nine-row catalogue and passes the material whole-endpoint performance and retained full Rust regression gates. The independent pointwise and Galerkin static candidates failed before trajectory authority, and the three-node maximum-relative-entropy route was rejected at design review before implementation; no ordinary continuation is authorized.
- Governing specification: `docs/TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md`, section 14 (`F-10`) and the owner-scope paragraph at section 15.
- Controlling policies: `docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`, `bbn_codex_anti_drift_cost_effective_policy.md`, and root `AGENTS.md`.
- Base revision: `0b2339c` on `bd612-remediation`; the active comparison is the explicitly dirty working tree, not an invented clean F-10C0 commit.

## Common assumptions and conventions

- Homogeneous, isotropic, spatially flat FLRW; `N=ln(a)` with `a=1` initially.
- Massless, flavour-diagonal neutrino and antineutrino occupations with zero lepton asymmetry; the heavy block folds degenerate muon and tau shapes.
- Comoving momentum `q=a p`; dimensionless radial coordinate `y=q/T_ref`; `T_cm=T_ref exp(-N)`; selected endpoint uses `T_ref=T_gamma(N=0)=10 MeV`.
- Evolved strict-open occupation coordinate is `u=log(f/(1-f))`, `f in (0,1)`; unrepresentable/invalid states fail raw rather than clip.
- Natural units `c=hbar=k_B=1`; energies and temperatures in MeV; elapsed-time readout in seconds using `1 MeV=1.519267447878626e21 s^-1`.
- Self action uses global four-leg coefficient 16; corrected tagged coefficients are 64 for identical same-sign and 32 for distinct same-sign families. `K_s` and the independently derived azimuth-averaged `K_t` remain separate invariants; no fitted normalization is allowed.
- Endpoint event is decreasing `T_gamma=0.005 MeV`. Frozen exact-test solver controls are `rtol=2e-7`, per-state `atol=2e-10` except `T_gamma=2e-9` and time `=2e-5`, `h_init=1e-5`, `h_min=1e-12`, `h_max=0.04`.
- Production target is Rust AOT/repeated-run execution. SciPy/BDF remains the temporary number-of-record until endpoint authority closes. JAX is a frozen CPU parity/AD/Jacobian oracle only.
- Current toolchain: rustc/cargo 1.94.1 and Python 3.12.3 on the current Linux host. Exact token-use counter is unavailable in this harness and must be reported `UNAVAILABLE`.

## Frozen independent comparison contract and current failure boundary

- V4+V5+V6+V7+V8 remains the governing endpoint design. The first authorized Fort GL30 execution retained an initial self-collision conservation failure before any accepted DLSODA step; the endpoint comparison remains **FAIL** and no RABBIT unblinding is authorized. Exact V8 (`6c451997`) changes only the dyadic witness and tail-only interval label.
- The approved initial-only discriminator is GL30/40/50/60/70/80 plus NC100, exactly one observer RHS and zero DLSODA steps per row. r1 was rejected before output because its copied source admitted only GL30/40/50. r2 was authorized and executed once: GL30/40/50 completed, then GL60 failed before its RHS because the read-only cache lacked `N0325_alpha03.dat`. r2 is immutable and must not be rerun.
- Retained r2 `D_N/(H n_nu)` is `1.443583767322922e-8`, `1.888741471383307e-10`, and `6.147742723040514e-12` at GL30/40/50; `D_E/(H rho_nu)` is `3.791326682671288e-10`, `2.953209820762704e-12`, and `1.3598611433522145e-13`. These are initial-only diagnostic measurements, not an endpoint pass, grid selection, or cap amendment.
- r3 closed the r2 cache defect nonphysically: alpha-2 is contiguous through `N0336`, alpha-3 through `N0814`, all 1,152 cache files are read-only, and generation/replay manifests are byte-identical. The frozen source/build/cache and initial-only physics boundary passed, but the blind filesystem authority review failed before output because authorization/pre-/post-process failures were not always retained and the dynamic loader/shared libraries were not exact-hash closed. r3 is immutable, unauthorized, and has no physical evidence.
- r4 closed the filesystem/runtime-byte authority defect and executed all seven initial-only rows exactly once with one observer RHS and zero DLSODA steps. The bounded initial-only diagnostic is **VALIDATED**, but endpoint authority remains **FAIL**: GL60 has an isolated raw self-action anomaly, so no unchanged Fort endpoint and no RABBIT unblinding are authorized.
- Independent Golub-Welsch metrology validates the GL50/60/70/80 grids and cached nodes. The GL60 anomaly instead localizes to the exact lower-grid endpoint: `interactions.f90` admits `y4 == ndmv(1)%y`, while `get_interpolated_nudens` in `utilities.f90` advances its bracket only for strict `y > ndmv(ind)%y` and otherwise returns its zero-initialized matrix. Binary64 tuple metrology reproduces the affected-index sets on all seven grids; this is a **DERIVED** source mechanism with **VALIDATED** numerical correlation, not yet a validated corrected endpoint.
- Whole-program FortEPiaNO 1:1 authority is **DEPRECATED** for this programme because its QKE/oscillation and precision-physics surface exceeds the frozen no-QKE objective. At most, a provenance-locked classical, flavour-diagonal, flat-FLRW, no-QKE collision slice may provide contextual external evidence; it cannot replace the required structurally independent full-spectral FLRW gate.
- r5 implements the only prospective Fort correction: exact lower/upper native-grid endpoint copies in `get_interpolated_nudens`. Blind source review passed, and strict nonphysical r4/r5 helper binaries **VALIDATED** the r4 lower-node zero, exact r5 endpoint identities, and unchanged one-ULP/interior behaviour. Physical output nevertheless remained forbidden: the blind authority assignment verified all 1,258 implementation entries and runtime bytes but failed because its named inputs omitted the r4 manifest contents/file-level comparison and the r5 physical controller. r5 is immutable and has zero collision RHS, DLSODA, endpoint, or RABBIT executions.
- r6 closed the r5 authority-context defect and executed the seven frozen initial-only rows once. The exact endpoint interpolation correction causally removes every r4 affected-index signature while leaving native grids and electron actions bitwise unchanged; independent recomputation reproduces all retained moment reductions within two binary64 ulps. This is **VALIDATED** only for the bounded patch-causality claim.
- The r6 exact common-FD H-normalized null measures are `5.33e-14--1.59e-13`, below the unchanged `1e-10` cap. The signed-over-absolute ratios remain `0.1285--0.9172`; an independent derivation shows that this ratio is a direction-dependent `0/0` at the analytic null and therefore is not a well-conditioned null-distance gate. No resolved non-null checkpoint, DLSODA step, Fort endpoint, mapper comparison, tolerance ladder, or RABBIT output was produced, so endpoint authority and `G-F10-INDEPENDENT-FLRW` remain **FAIL**.
- The Fort-specific route is stopped. Do not implement the proposed exact-null observer branch merely to unlock a Fort endpoint, and do not add another Fort wrapper/gate/cache/telemetry loop. Preserve r6 and its raw ratios as contextual evidence. The next implementation slice is the smallest structurally independent full-spectral flavour-diagonal flat-FLRW no-QKE comparator; it must not inherit Fort QKE/oscillation architecture or tune RABBIT constants, grids, collision formulae, caps, or tolerances.
- The minimal comparator boundary is now frozen for M0/M1: one private `src/rabbit/decoupling/_independent_noqke.py` module plus one focused test module, no public export/registry/driver change; separate e/mu/tau flavour-pair spectra in complementary-log-log coordinates; affine finite-interval GL48/64 with `y_max=24/28`; SciPy Radau with numerical Jacobian and dense-output/Brent temperature roots. Constants and comparison semantics may be shared, but collision/EOS/grid/solver/Jacobian/mapper execution code from RABBIT, JAX, Rust, Fort, deterministic-reference, or RTA paths may not be called or translated.
- Immediate authority is static M0/M1 only: independently derived microscopic formula/source boundary and executable common-FD null, resolved non-null conservation/exchange/first-law/entropy/scaling/catalogue, EOS, coordinate, grid, and negative-control discriminators. No trajectory, endpoint, or fresh RABBIT output is authorized until a blind review passes those exact source bytes.
- The direct target-leg pointwise M1 candidate is rejected by D-027: its common-FD total null passes, but S1 self number/energy and electron exchange miss the frozen caps.
- The full-degree entropy-variable Galerkin-Petrov replacement is **IMPLEMENTED**, and `GL48_STATIC_R1` is **VALIDATED** as a failure. S0 passes at `6.59e-12/1.56e-11`; S1 self conservation, exchange, first law, CP, entropy, and resources pass, but native mu-tau covariance is `4.666064056497196e-10` against `1e-10`.
- D-028 is binding: GL64, metric/cap changes, post-hoc symmetry, event-chart averaging, reduction retries, Radau, trajectory, endpoint, and RABBIT unblinding are **FORBIDDEN**. Only materially new, prospectively registered owner-authorized numerical evidence may reopen a different method or norm.
- D-029 rejects the current three-node maximum-relative-entropy design before implementation. Its fixed-triple exact-arithmetic theorem is **DERIVED** and its standalone piecewise binary64 stencil is **VALIDATED** only within its declared local limits, but the executable GL-weight prior, nearest-exterior selector, and exact-node one-hot branch are not derived as one continuous/covariant semidiscrete method. Support-switch/Jacobian behavior, the frozen 24-self/15-electron binary64 collision contract, and the native mu-tau `1e-10` co-gate are absent. No implementation or physical-output authority exists.
- Reference: FortEPiaNO tag `1.4.0`, commit `6cac851dcc8224693937dbf8d1cbd3367c26af53`; keep `/tmp/fortepiano_public_probe` pristine and preserve the separate adapter patch, resolved inputs, strict flags, binary and raw hashes/failures.
- Model: flat FLRW; three massless diagonal active flavours; zero mixing/masses/asymmetry; finite vacuum `m_e`; all tree-level electron and neutrino-self collisions; no off-diagonal state/commutator, QED/thermal mass, muons, low reheating, NSI/non-unitarity, fast-math, QKE, or coherence.
- Constants: `m_e=0.5109989500 MeV`, `G_F=1.1663788e-11 MeV^-2`, `sin^2(theta_W)=0.23122`, `G_N=6.708830746231458e-45 MeV^-2`, `M_Pl=1.2208901285838955e22 MeV`, `1 MeV=1.519267447878626e21 s^-1`; never fit them.
- Boundary: `a=1`, `T_gamma=T_cm=10 MeV`, exact zero-chemical-potential FD, Fort `x=0.051099895,z=1`; stop at internally interpolated decreasing `m_e z/x=0.005 MeV`, with `x_fin=200` safety only. Fixed-`x` or post-hoc endpoint matching fails.
- Only after a unanimous fresh V4+V5+V6+V7+V8 blind design pass may the comparison worktree modify exactly six Fort sources: `const.f90`, `utilities.f90`, `input.f90`, `config.f90`, `equations.f90`, and `fortepiano.f90`. `Makefile`, collision/Hubble/EOS sources, `matter.f90`, `output.f90`, ODEPACK, and every other compiled dependency remain immutable and hash-locked. V6 freezes cache/reduction/build/filesystem/BDF-replay semantics; V7 freezes finite-plus-tail EM and same-code uncertainty; V8 changes no algorithm or cap and only repairs the dyadic negative-control bytes plus tail-correction labeling. A second blind exact-hash review must explicitly permit physical output.
- Grids: GL `Ny=30,40,50,y_max=20`; Newton-Cotes `Ny=100,y=[0.01,20]`; neutral centers `y_j=0.025+0.125j` (`j=0..159`) repeated at half spacing, mapper change below one quarter cap, with `y>20` tail separate.
- Tolerances `(rtol,atol_z,atol_d,atol_o)`: `(1e-6,1e-8,1e-9,1e-9)`, `(2e-7,2e-9,2e-10,2e-10)`, `(5e-8,5e-10,5e-11,5e-11)`; only tightening is allowed. Checkpoints MeV: `10,5,3,2,1.5,1,0.7,0.5,0.3,0.2,0.1,0.05,0.01,0.005`.
- Emit `x,N,z,H,t_seconds`, three spectra before X-folding, native number/energy moments, `N_eff`, electron/self action blocks, energy transfer/first-law, mu-tau, layout/commutator, DLSODA status/stats, and event residual.
- Finest-level internal caps: `Delta N_eff<=1e-4`, energy `<=1e-4`, spectral L1 `<=2.5e-4`, collision L1 `<=5e-3`. Cross-code caps: `|Delta N|<=5e-5`, time `<=5e-5`, `|Delta N_eff|<=5e-4`, energy `<=5e-4`, spectral L1 `<=2e-3`, resolved pointwise `<=1e-2` above `1e-8`, collision L1 `<=2e-2`, first-law `<=1e-8`, conservation/null and mu-tau `<=1e-10`.
- Also require `|Delta O|<=3 sqrt(u_R^2+u_I^2)` with `u_code=max(grid,tolerance,mapper)`; either code uncertainty above half a hard cap is `INCONCLUSIVE`. Structural gates are never uncertainty-rescued. Hash the independent bundle before unblinding RABBIT, and never loosen grids/tolerances/norms afterward.

## Frozen scope

- In scope: structurally independent full-spectral FLRW validation, matched-switch/grid/tolerance checks, and F-10 closeout.
- Immediate slice: fail-closed preservation and owner decision. Fort r6, the pointwise failure, `GL48_STATIC_R1`, and the rejected three-node design artifacts are immutable contextual/failure evidence. Do not implement that design, execute Fort/collision/GL64/comparator trajectory, or tune any norm, cap, grid, formula, or tolerance.
- Retained-tree closeout: `221/221` release tests and doctests passed; the corrected 54-page report rebuilt and rendered cleanly. This closes only `G-F10C1-REGRESSION`.
- Explicit non-goals: F-11 Bianchi/LRS/non-LRS Type-I implementation or scheduling, QKE/coherence/oscillations, public dispatch, public production, precision Standard-Model or publication claims.
- F-10C2 catalogue boundary: all nine frozen rows execute. Rowwise tagged normalization, explicit six-species multiplicity, equilibrium/null, number/energy conservation, entropy, scaling, response, and five-point Jacobian tests pass; full production aggregation is checked against the explicit seven folded channel sum.
- The retained exact-point BDF Jacobian cache improves the same F-10C1 endpoint by 55.815% with bitwise-identical solver outputs. On the completed catalogue, four-topology aggregation reduces the measured endpoint from 1520.92 s to 1184.77 s (22.102%, 1.284x) and RSS by 0.195%.
- Final Rust release regression was 230/230 plus 0 doctests (historical, predating later current-tree edits; the current crate has 240 tests with 2 ignored, and no authorized current-tree full-suite rerun exists, so 230/230 must not be cited as a current-tree pass — see VALIDATION_LEDGER 2026-07-18). BDF gives `(N,t,N_eff)=(7.936693339485084,52677.63448707955 s,3.034035983584400)`; Rodas5P gives `(7.936706017467941,52678.09666722039 s,3.033904967773792)`.
- N48 validation is limited to the declared five smooth profiles and two N128 auxiliary rules. Same-code endpoint `N_eff` is only a regression readout.
- Do not silently strengthen `IMPLEMENTED` into `VALIDATED`, internal parity into independent validation, or a segment/profile speedup into endpoint progress.

## Shared evidence pointers

| Evidence | Path / digest prefix | Boundary |
|---|---|---|
| E-F10C1-SPEC | unified plan / `c76cc5d` | scope/catalogue authority |
| E-F10C1-CLAIMS; VALIDATION | `docs/harness/{CLAIM_LEDGER,VALIDATION_LEDGER}.md` / `35baa7ad;c6baff76` | claim ceilings/executed history |
| E-F10-PERF-PROFILE | prior perf raw logs / `9f7032ba;1904297a;49910c68` | whole-endpoint PASS |
| E-F10-CATALOGUE-TESTS | prior focused/full logs / `71ef2405;1e67ae9e` | catalogue PASS, 230 tests |
| E-F10-INDEPENDENT-DESIGN; PROBE | prior result JSON / `f4751f3e;cb76aca4` | proposed gate/adapter feasibility only |
| E-F10-INDEPENDENT-PREOUTPUT | `run-20260717-f10c2-independent-v2` / merged `75922863`; contract `fba76f51`; adapter `39997c35` | blind RED before output; canonical upstream tests-only PASS |
| E-F10-INDEPENDENT-REMEDIATION | `run-20260717-f10c2-contract-remediation` / contract `fb9d0308`; adjudication `7bd6202b`; Fort audit `93a8fee5`; metric derivation `37e25d1c` | V4 design remediated; numerical output still forbidden pending two blind reviews |
| E-F10-INDEPENDENT-V4-REVIEW; V5 | `run-20260717-f10c2-v4-design-review` / blind design `69a4a945`; feasibility `13281493`; V5 amendment `c34fc9ec` | V4 failed before output; V5 closes named formula/source/controller/state omissions and awaits a new blind design verdict |
| E-F10-INDEPENDENT-V5-REVIEW; V6 | `run-20260717-f10c2-v5-design-review` / physics `9daa7805`; implementability `ee520e65`; metrology `512a4c60`; V6 amendment `213d5db5` | V5 failed before implementation/output; V6 freezes reductions, certificates, uncertainty, cache/build closure, and BDF replay semantics; fresh review required |
| E-F10-INDEPENDENT-V6-REVIEW; V7 | `run-20260717-f10c2-v6-design-review` / physics `716abdda`; implementability `5b8fbfb8`; metrology `ee652b20`; merged `da0af23c`; V7 amendment `8c49dd6a` | V6 failed before implementation/output on two deterministic clauses; V7 adds certified tail enclosure and same-code uncertainty; fresh review required |
| E-F10-INDEPENDENT-V7-REVIEW; V8 | `run-20260717-f10c2-v7-design-review` / physics `a68e4165`; implementability `a4258ed6`; metrology `0a3c2fdb`; merged `fc6931ec`; V8 amendment `6c451997` | V7 core passed two reviews; exact decimal negative control failed metrology; V8 substitutes exact dyadics and downclaims tail interval; fresh review required |
| E-F10-INDEPENDENT-V8-REVIEW | `run-20260717-f10c2-v8-design-review` / physics `7a4528ab`; implementability `9145b338`; metrology `3e0c2954`; merged `820382ac` | unanimous stage-1 PASS for bounded implementation; physical output remains forbidden pending exact-hash stage-2 review |
| E-F10-FORT-CONSERVATION-DIAGNOSIS | `run-20260717-f10-fort-conservation-diagnosis/ADJUDICATION.md` | stock GL30 self action exceeds frozen null cap; initial-grid discriminator selected without cap change |
| E-F10-FORT-INITIAL-R1 | `run-20260717-f10-fort-initial-grid-diagnostic/ADJUDICATION.md` | blind pre-output FAIL on incomplete GL allowlists; no physical output |
| E-F10-FORT-INITIAL-R2 | `run-20260717-f10-fort-initial-grid-diagnostic-r2/ADJUDICATION.md` / `d94a6d19`; manifest `5dbf5110`; failure `c17904fe` | authorized once; GL30/40/50 retained; GL60 cache-closure FAIL before RHS |
| E-F10-FORT-INITIAL-R3 | `run-20260718-f10-fort-initial-grid-diagnostic-r3/ADJUDICATION.md`; manifest `3bc8a98d`; authority review `55631b2e` | cache closure PASS; physics boundary PASS; output authority FAIL before execution on failure retention and dynamic-library hash closure |
| E-F10-FORT-INITIAL-R4 | `run-20260718-f10-fort-initial-grid-diagnostic-r4/ADJUDICATION.md` / `945d67bf`; implementation `d33e2b57`; evidence `d7687d6b`; grid metrology `c4b08ccc`; collision metrology `4a8c1d97` | exact-hash initial-only execution VALIDATED; endpoint authority FAIL; GL60 grid/cache valid and exact lower-endpoint interpolation mechanism isolated; whole-program Fort 1:1 authority DEPRECATED |
| E-F10-FORT-ENDPOINT-INTERP-R5 | `run-20260718-f10-fort-endpoint-interpolation-r5/ADJUDICATION.md` / `c09366cc`; implementation `d9fa8410`; pre-output tests `fd01a39`; authority `16be7ac8` | source correction and nonphysical bit metrology pass; physical-output authority FAIL on missing r4-manifest/controller assignment inputs; zero physical output |
| E-F10-FORT-ENDPOINT-INTERP-R6 | `run-20260718-f10-fort-endpoint-interpolation-r6/ADJUDICATION.md` / `6119570c`; implementation `75613e2b`; evidence `0cd7511b`; physics `aa72d47a`; metrology `abde87c9`; null-gate derivation `17594d64` | exact-endpoint patch causality VALIDATED on seven one-RHS/zero-DLSODA rows; endpoint authority FAIL; Fort route stopped and whole-program authority DEPRECATED |
| E-F10-MINIMAL-INDEPENDENT-R1 | `run-20260718-f10-minimal-independent-noqke-r1/ADJUDICATION.md` / `63f29645`; map `09a8a269`; physics `293b77b`; numerics `5828c0ee` | M0/M1 private NumPy/SciPy comparator contract PASS; no runtime or endpoint authority; gate remains FAIL |
| E-F10-INDEPENDENT-POINTWISE-R2 | `run-20260718-f10-minimal-independent-noqke-r2-implementation` / source `b90d757a`; artifact `cab73790`; adjudication `7e9b16d1` | direct target-leg M1 FAIL; no evolving RHS or trajectory authority |
| E-F10-INDEPENDENT-GALERKIN-GL48 | `run-20260718T095658Z` / source `535370c1`; test `abdc4e01`; artifact `4726c40e`; covariance adjudication `3002ad87` | conservative Galerkin source IMPLEMENTED; GL48 static M1 FAIL on native mu-tau covariance only; GL64 and trajectory forbidden |
| E-F10-INDEPENDENT-MAXENT3-DESIGN | `run-20260718-f10-static-fail-closeout` / derivation `c336130b`; binary64 audit `6f774456`; embedded program `ad1144f3`; adjudication `614999cd` | fixed-triple theorem DERIVED and local stencil audit passes with limits; terminal decision `reject-design`; no implementation/collision/endpoint authority |
| E-F10-INDEPENDENT-ENDPOINT | `NOT_PRODUCED` | independent gate RED |
| E-HARNESS-VALIDATE; AUDIT | bootstrap / `de886f92`; prior audit / `da61814d` | static pass; hooks/write attribution unvalidated |

Rebuild the context pack and refresh any changed evidence hash before spawning a subagent.

## Known disputes and open questions

| Question ID | Question | Required discriminating evidence | Owner |
|---|---|---|---|
| Q-IND-01 | Does a structurally independent full-spectral FLRW solve reproduce the accepted envelope? | Independent formulation/integrator with frozen inputs and tolerances, not same-code replay | future F-10 validation |
| Q-HOOK-01 | Are project hooks enabled by a trusted Codex session? | User/session `/hooks` review; file installation alone is not activation evidence | owner |

---

## Source: `.agent-harness/context/SYMBOLS.md`

SHA-256: `6d5fbe058c2750ed680ce835e6e03438200b36b16743bf6c1be99649030e096b`

# Symbol and Interface Table

| Symbol / interface | Definition | Domain / type | Units / dimensions | Sign / branch convention | Source of truth |
|---|---|---|---|---|---|
| `a` | FLRW scale factor | positive scalar | dimensionless | `a(N=0)=1` | `isotropic_boltzmann.rs` |
| `N` | `ln(a)`, independent ODE variable | real scalar | dimensionless | increases during expansion | `ode.rs`, `isotropic_boltzmann.rs` |
| `p` | physical neutrino momentum magnitude | nonnegative | MeV | future-directed energy `E=p` for massless neutrinos | F-10 spec |
| `q` | comoving momentum `a p` | nonnegative | MeV | constant under collisionless redshifting | `ComovingMomentumGrid` |
| `y` | `q/T_ref` | positive node | dimensionless | selected map `y=-3 ln(1-t)`, `t in (0,1)` | `quadrature.rs` |
| `T_ref` | initial/reference comoving temperature | positive scalar | MeV | endpoint test fixes 10 MeV | `IsotropicBoltzmannFlrwSystem` |
| `T_cm` | `T_ref exp(-N)` | positive scalar | MeV | collision scale, decreases with expansion | `isotropic_boltzmann.rs` |
| `T_gamma` | electromagnetic bath temperature | positive ODE state | MeV | terminal crossing direction is negative | `isotropic_boltzmann.rs` |
| `f_E,f_X` | folded electron and heavy neutrino-pair occupations | vector entries in `(0,1)` | dimensionless | zero lepton asymmetry; heavy mu/tau shapes degenerate | `isotropic_boltzmann.rs` |
| `u` | logit `ln[f/(1-f)]` | finite ODE coordinate | dimensionless | logistic inverse; no clipping | `isotropic_boltzmann.rs` |
| `C[f]` | classical diagonal collision action | grid vector | inverse time in natural units | gain minus loss; equilibrium null | `electron_spectral.rs`, `neutrino_self_spectral.rs` |
| `H` | flat-FLRW Hubble rate | positive scalar | MeV, converted to s^-1 | positive expanding branch | `flrw.rs` |
| `G_F` | Fermi constant | positive scalar | MeV^-2 | repository constant; never refit | `electron_hm.rs` |
| `K_s` | `(p_1.p_2)(p_3.p_4)` | nonnegative on physical event | MeV^4 | metric/product convention inherited from HM event algebra | F-10 spec, `neutrino_self_spectral.rs` |
| `K_t` | `(p_1.p_4)(p_2.p_3)` | nonnegative on physical event | MeV^4 | directed crossed kernel | F-10 spec |
| `eta` | time-orientation/symmetry factor in global four-leg form | positive scalar | dimensionless | same-flavour elastic uses 1/2 | F-10 catalogue |
| `Q_E,Q_X` | electron/heavy collision energy moments | signed scalar | MeV^5 | positive heats that neutrino block; EM receives equal opposite electron debit | `electron_spectral.rs`, `isotropic_boltzmann.rs` |
| `N_eff` | conventional energy-density readout | positive scalar | dimensionless | regression diagnostic only in current same-code endpoint | endpoint test |
| `x_F` | FortEPiaNO independent variable `m_e/T_cm` | positive scalar | dimensionless | starts at `0.051099895` and increases | FortEPiaNO 1.4.0 adapter contract |
| `z_F` | FortEPiaNO photon ratio `T_gamma/T_cm` | positive scalar | dimensionless | event uses `m_e z_F/x_F=0.005 MeV` on the decreasing branch | FortEPiaNO 1.4.0 adapter contract |
| `IsotropicBoltzmannFlrwSystem` | crate-private coupled RHS/Jacobian and physical-state consumer | Rust type | n/a | no public dispatch authority | `isotropic_boltzmann.rs` |

Any CAS axis may introduce internal names, but its result must map them back to this table and the shared CAS contract.

---

## Source: `.agent-harness/context/FROZEN_DECISIONS.md`

SHA-256: `3c78c051001c465a5d6b90493a35b1007436f1ef4ff203e2849958d923720ac8`

# Frozen Decisions and Rejected Alternatives

| Decision ID | Decision | Rationale/evidence | Scope | Reopen condition |
|---|---|---|---|---|
| D-001 | Rust AOT is the active implementation and repeated-run target; SciPy/BDF remains temporary number-of-record. | Owner Rust pivot and anti-drift guardrail | all active F-10 work | endpoint authority, parity, performance, and deflation gates explicitly pass |
| D-002 | JAX is frozen as a local CPU parity/AD/Jacobian oracle. | Runtime/backend role decision | F-10 | explicit owner reauthorization after measured bake-off |
| D-003 | Complete F-10 collision-coupled isotropic FLRW and stop; F-11 LRS/non-LRS Type-I is owner-paused. | Owner instruction dated 2026-07-16 | scheduling and implementation | new explicit owner instruction after F-10 handoff |
| D-004 | Use the positive exponential N48 rule with nonnegative linear interpolation/deposition for the current state representation. | Smallest tested order passing direct and five-profile/two-auxiliary gates; lower orders rejected | F-10C radial representation | new evidence falsifies the frozen envelope without weakening tolerances/conservation |
| D-005 | The prepared self-event geometry cache is rejected and removed. | 9.195% wall reduction was below the about-10% partial-blocker threshold and RSS rose 3.277% | F-10C1 performance | materially better whole-endpoint evidence with acceptable memory and preserved solver envelope |
| D-006 | All nine frozen zero-lepton classical diagonal catalogue rows execute through four topology-equivalent production contractions, while seven isolated folded channels remain executable test oracles. | Primary normalization derivations, explicit six-species enumerator, rowwise tagged targets/Jacobians/invariants, and 230/230 release regression | F-10C2 | new primary or independent evidence falsifies a row without tolerance widening |
| D-007 | Same-code `N_eff`, solver agreement, green tests, or profile-bounded N48 convergence are not independent precision validation. | Claim ledger and research harness rules | reporting | structurally independent full-spectral FLRW evidence |
| D-008 | No new standalone readiness/manifest/hash/figure/telemetry/table/cache/policy surface may be retained unless it consolidates older surface and directly moves a measured endpoint or physics-correctness blocker. | anti-drift and BD397 cost policy | every F-10 patch | controlling policy revision |
| D-009 | One main writer owns production code, shared docs, specs, and gates. Subagents write only registered result artifacts. | uploaded shared-context harness | current and future multiagent runs | none within this harness |
| D-010 | QKE, coherence, oscillation Hamiltonians, public dispatch, public production, and full Standard-Model transport claims are forbidden. | F-10 specification and owner boundary | current programme | explicit new specification and owner authority |
| D-011 | Retain only the solver-local exact `(t,state)` BDF full-Jacobian cache; no global, cross-solve, approximate, or policy-controlled reuse is allowed. | Same-physics endpoint improves 55.815%, outputs and step counts are bitwise identical, RSS rises 1.229%, and independent review is PASS_RETAIN | F-10 repeated-run baseline | an exactness, memory, determinism, or independent-review regression appears |
| D-012 | Retain one shared self-event geometry stream and four topology contractions instead of seven row contractions in production; preserve isolated row evaluators in tests. | Attached full-catalogue profile attributed 22.24% to row contraction and 10.62% to duplicate event construction; whole endpoint improves 22.102% with unchanged BDF output and passing solver envelope | F-10C2 production evaluator | explicit-row sum test or physical invariant fails |
| D-013 | Freeze the FortEPiaNO 1.4.0 diagonal-only neutral contract, grids, tolerance ladder, event, observables, uncertainty rule, and hard caps before any independent output; never tune either solver to the other. | Registered independent design plus official-source feasibility probe; prevents endpoint-led calibration and false fixed-x/QKE comparisons | F-10 independent FLRW gate | pre-output blind review may tighten but not loosen; a hard physics mismatch downgrades evidence to contextual |
| D-014 | The first blind FortEPiaNO contract review is a pre-output FAIL. A separate hashed comparison worktree may align only frozen model-input literals and strict compile inputs; it may not alter collision formulae. No checkpoint/endpoint or RABBIT unblinding is allowed until executable layout/mapping/event/norm definitions receive fresh blind approval. | `run-20260717-f10c2-independent-v2`: mismatched constants, fast-math, layout-order hazard, transformed coordinates, and undefined metrics; canonical `make tests`-only source baseline passes | F-10 independent FLRW gate | a new context version resolves every fatal blocker and a registered blind reviewer explicitly permits output |
| D-015 | Use a two-stage blind gate for the remediated Fort comparison: V4 design approval may authorize only implementation; numerical output requires a second review of the exact patched-source, input, strict-binary, observer, mapper, comparator, and test hashes. Use documented one-step `ITASK=5` with `TCRIT=200`, actual common convex-hull mapping without extrapolation, and `u_code=max(grid,tolerance,mapper)`. | `run-20260717-f10c2-contract-remediation`: independent Fort and metric reviews integrated without endpoint output; resolves reviewer conflicts while preserving the frozen safety boundary and uncertainty rule | F-10 independent FLRW gate | fresh primary-source or blind-review evidence identifies a physics/numerical defect before output; only tightening is allowed |
| D-016 | Replace the rejected V4 implementation boundary and incomplete residual clauses with the hash-locked V5 amendment: exactly six modified Fort sources, immutable Makefile/collision/EOS/ODEPACK dependencies, pre-materialized GL caches, byte-isolated observer state, a fully specified left-first GK15/7 controller, and executable equilibrium-null, signed-Q, direct-EM-derivative, exchange, and first-law gates. | `run-20260717-f10c2-v4-design-review`: blind design `69a4a945` and Fort feasibility `13281493` failed before output on precisely these omissions; V5 amendment `c34fc9ec` transcribes their minimal remedies without changing caps or producing results | F-10 independent FLRW pre-output design | a fresh blind V4+V5 review finds a new physics, numerical, source-boundary, or blinding defect; implementation/output stay forbidden until their separate approvals |
| D-017 | Preserve V4 and V5 bytes and apply exact V6 before implementation: use the specified recursive binary64 reduction, multiplication-free signed-Q test, independent dual quadrature certificates, metric-specific uncertainty operators, complete alpha-2/alpha-3 cache and compiler/filesystem closure, deterministic observer-component equality, and BDF-only same-initial-state checkpoint prefix replays. | `run-20260717-f10c2-v5-design-review`: physics `9daa7805` passed, but implementability `ee520e65` and metrology `512a4c60` failed pre-implementation on executable ambiguity and false error-bound semantics; V6 `213d5db5` closes each named finding without output or cap loosening | F-10 independent FLRW pre-output design | unanimous fresh V4+V5+V6 blind review; any dissent retains the pre-output hard stop |
| D-018 | Preserve V4/V5/V6 bytes and apply exact V7: integrate the EM derivative on `[0,m_e+128T]` with two independent finite transforms, charge the DERIVED `B_tail=1e-18 T^3` enclosure to the unchanged certificate, assert nearest/subnormal arithmetic, and define every grid/tolerance/mapper uncertainty as direct same-code central-versus-alternative movement before maxima. | `run-20260717-f10c2-v6-design-review`: all reviews failed pre-implementation (`716abdda`, `5b8fbfb8`, `ee652b20`) because the old half-line samples necessarily underflow and aggregate cross-metric subtraction admits reflection cancellation; V7 `8c49dd6a` remedies only those two findings without output or cap loosening | F-10 independent FLRW pre-output design | unanimous fresh V4+V5+V6+V7 blind review; any dissent retains the hard stop |
| D-019 | Apply exact V8 on top of immutable V7: Case B uses `d=21*2^-20`, so old aggregate shift is exact `+0.0` and direct same-code movement is exact `42*2^-20`; call `[D1,D1+B_tail]` only a tail-correction interval around a numerical estimate, never a rigorous full-derivative enclosure. | `run-20260717-f10c2-v7-design-review`: physics and implementability passed, but metrology `0a3c2fdb` proved the decimal witness leaves exact `2^-53`; V8 `6c451997` changes only the witness bytes and claim label | F-10 independent FLRW pre-output design | unanimous fresh V4+V5+V6+V7+V8 blind review; output remains separately gated |
| D-020 | Authorize only the exact bounded V4+V5+V6+V7+V8 implementation. Keep all physical FortEPiaNO and RABBIT checkpoint/endpoint bytes forbidden until a fresh blind stage-2 review approves the exact patched-source, build, cache, observer, mapper, comparator, and nonphysical-test hash bundle. | `run-20260717-f10c2-v8-design-review`: physics `7a4528ab`, implementability `9145b338`, and metrology `3e0c2954` unanimously return implementation-only PASS; merged `820382ac` has no errors | F-10 independent FLRW stage 1 | exact implementation bundle exists, strict/nonphysical gates pass, and a registered blind stage-2 reviewer explicitly sets `permit_output=true` |
| D-021 | Preserve the failed GL30 endpoint attempt and r1/r2 initial-grid bundles as immutable evidence. Before any endpoint retry, complete the initial-only GL30/40/50/60/70/80+NC100 discriminator. r3 may change only grid-only alpha-3 cache materialization/replay through GL80; runtime cache writes, collision changes, cap changes, projection, and endpoint/RABBIT execution remain forbidden. | r2 executed GL30/40/50 once and measured rapidly decreasing self-action leakage, then failed before the GL60 RHS because the frozen cache lacked `N0325_alpha03.dat` and the staged directory was correctly read-only; adjudication `d94a6d19` | Fort numerical-floor classification before F-10 endpoint authority | a fresh exact-hash r3 completes all seven initial-only rows and the evidence is prospectively adjudicated |
| D-022 | Preserve r3 as an unauthorized pre-output failure. r4 may change only run-local failure retention and dynamic executable-byte closure while preserving the exact r3 Fort physics/build/cache/rows/caps/state boundary. No diagnostic output is allowed until fresh blind physics and filesystem reviews both authorize the exact r4 manifest. | r3 manifest `3bc8a98d` closes the cache defect; blind physics review `7669a9c8` passes, but blind authority review `55631b2e` fails because authorize/pre-/post-process failures are not always retained and the loader plus five shared-library dependencies are not hash-locked or live-rechecked | Fort initial-only diagnostic execution authority | a fresh r4 manifest binds and rechecks every loaded runtime byte, retains every governed failure, and receives unanimous exact-hash output authority |
| D-023 | Preserve r4 as immutable, **VALIDATED** initial-only diagnostic evidence and a **FAIL** for endpoint authority. **DEPRECATE** whole-program FortEPiaNO 1:1 authority; only its classical flavour-diagonal flat-FLRW no-QKE slice may remain contextual. A fresh run may change only the exact lower/upper native-grid endpoint return in `get_interpolated_nudens`, then repeat the same seven zero-step rows before any endpoint decision. No cap, grid, physical formula/coefficient, state, solver, endpoint, or RABBIT change is allowed. If the isolated correction fails its prospective discriminator, downgrade Fort completely instead of tuning it. | r4 adjudication `945d67bf`; all seven rows ran once under exact runtime-byte authority. Independent grid metrology `c4b08ccc` validates GL60 nodes/weights, while collision metrology `4a8c1d97` and source inspection identify the strict-`>` lower-endpoint zero-return mechanism. The owner confirms whole Fort exceeds the no-QKE objective. | F-10 contextual Fort discriminator and independent-validation boundary | only a fresh exact-hash, blindly authorized initial-only run demonstrates that the one-line semantic correction removes the anomaly without any new physics/readout failure; this still cannot promote Fort to whole-program 1:1 authority |
| D-024 | Preserve r5 as immutable **VALIDATED** nonphysical interpolation metrology and a **FAIL** for physical-output authority. A fresh r6 may change only run-local authority closure by naming the immutable r4 implementation manifest, exact r6 physical controller, and a frozen normalized r4-to-r6 scope comparison. Reuse the exact r5 Fort source correction, contract semantics, strict flags, cache, inputs, rows, one-RHS/zero-DLSODA rule, repetition count, caps, and endpoint/RABBIT prohibition. No r5 result may be overwritten. | r5 implementation `d9fa8410` and pre-output tests `fd01a39` pass; blind authority `16be7ac8` validates manifest/runtime integrity but sets `permit_output=false` because the assignment did not expose the r4 manifest contents/file-level comparison or physical controller. Adjudication `c09366cc`. | Fort r6 initial-only physical-output authority closure | a fresh exact-hash reviewer can reproduce the one-source semantic delta and controller counters solely from registered inputs and explicitly sets `permit_output=true` |
| D-025 | Preserve r6 as immutable **VALIDATED** evidence for the bounded exact-endpoint interpolation-causality claim and a **FAIL** for endpoint authority. Stop the Fort-specific route: do not implement an exact-null observer exception merely to unlock a Fort endpoint. Whole-program FortEPiaNO 1:1 authority remains **DEPRECATED**; the hash-locked classical diagonal initial-collision slice is contextual only. Move `G-F10-INDEPENDENT-FLRW` to the smallest structurally independent full-spectral flat-FLRW no-QKE comparator, retaining raw ratios, the unchanged H-normalized analytic-null cap, and the unchanged signed-over-absolute gate at resolved non-null checkpoints. | r6 authority `55770b47` permitted one seven-row execution; all r4 affected sets cleared, independent raw metrology `abde87c9` reproduced moments within two ulps, and derivation `17594d64` proves the exact common-FD signed/absolute ratio is direction-dependent `0/0`. No DLSODA step, Fort endpoint, or RABBIT output ran; adjudication `6119570c`. | F-10 independent full-spectral validation boundary | only a new owner decision backed by materially new physics evidence may reopen Fort; ordinary gate/wrapper/cache/telemetry work does not qualify |
| D-026 | Implement the independent comparator only as private `src/rabbit/decoupling/_independent_noqke.py` plus one focused test module, with no export/registry/driver surface. Evolve separate e/mu/tau flavour-pair spectra in complementary-log-log coordinates on prospective affine GL48/64, `y_max=24/28`; use SciPy Radau, numerical Jacobian, and dense-output/Brent temperature events. Independently derive the six-state collision catalogue, finite-mass electron action, EOS, reductions, and mapper; runtime reuse or line translation from RABBIT/JAX/Rust/Fort/RTA is forbidden. M0/M1 static physics discriminators are authorized; trajectory, endpoint, and RABBIT unblinding are not. | Independent context map `09a8a269`, physics contract `293b77b`, and numerical contract `5828c0ee`; adjudication `63f29645`. The stronger Radau/three-flavour/affine-grid axes were selected over folded E/X plus BDF because they expose multiplicity/folding errors and avoid both target solver families at small state cost. | F-10 minimal structurally independent no-QKE comparator M0/M1 | blind source/formula review plus passing common-FD null, resolved non-null conservation/exchange/entropy/scaling/catalogue/EOS/grid discriminators may authorize one 10-to-3 MeV segment |
| D-027 | Preserve the exact direct target-leg pointwise r2 source and failed M1 artifact as immutable evidence: the reported common-FD total null passes, but resolved S1 self number/energy conservation and independently reduced electron exchange miss their unchanged caps. The pointwise non-deposition action is rejected as an evolving RHS and receives no trajectory authority. The next slice is design-only for one discrete-event conservative M1 replacement inside the same private source/test surface; post-hoc projection, cap/state/order tuning from output, Rust/Fort/JAX executable reuse, and new public/wrapper surfaces remain forbidden. | r2 primary derivation `62dcdfb9`, exact source `b90d757a`, failed artifact `cab73790`, and adjudication `7e9b16d1`; measured residuals are `1.9782e-10`, `2.3350e-10`, and `1.0684e-7` against `1e-10`, `1e-10`, and `1e-8`. | F-10 independent full-spectral M1 recovery | registered blind derivations agree on the invariant event measure, target/global leg multiplicities, conservative off-grid deposition, detailed balance, support/tail semantics, and structural independence; adjudication may then authorize replacement implementation but still not a trajectory |
| D-028 | Preserve the exact private Galerkin-Petrov source/test and `GL48_STATIC_R1` as immutable **IMPLEMENTED** and **VALIDATED failure** evidence. The candidate passes the common-FD null and every recorded S1 invariant, exchange, first-law, CP, entropy, family, wall, and memory discriminator except the prospectively coded native mu-tau relative-L-infinity covariance: `4.666064056497196e-10 > 1e-10`. Static M1 therefore **FAILS**. Do not change the norm/cap, symmetrize outputs, average event orientations, retry reductions, run GL64, construct Radau, or produce a trajectory/endpoint. | source `535370c1`, test `abdc4e01`, artifact `4726c40e`, method adjudication `ab8b2bc7`, covariance adjudication `3002ad87`. The favourable modal residual is localization only; the selected native mass inversion and its `1/y^2` conditioning are binding. Cost is `+406/-326`, net `+80`, token use `UNAVAILABLE`, blocker movement `0.25`, verdict `FAILURE_MODE_RELOCATION`. | F-10 independent full-spectral validation boundary | only materially new, prospectively registered, owner-authorized numerical evidence and a fresh blind design may reopen a different norm or method; ordinary wrapper, metric, cap, chart, reduction, telemetry, Fort, or same-code work does not qualify |
| D-029 | Reject the current three-node maximum-relative-entropy route at blind design review before implementation. Preserve the fixed-triple derivation, standalone binary64 audit, embedded program, exploratory failures, and terminal adjudication as immutable design evidence. The fixed-triple negative-KL theorem is sound, but the executable's local GL-weight prior, nearest-exterior target-dependent support selector, and exact-node one-hot branch are not derived as one continuous/covariant semidiscrete collision method. Do not implement it or execute collision, GL48/64 comparator, Jacobian/Radau, trajectory, endpoint, or RABBIT output. | derivation `c336130b`, binary64 audit `6f774456`, embedded program `ad1144f3`, adjudication `614999cd`. Support-switch continuity/Lipschitz/Jacobian and tie covariance are unproved; the frozen 24-self/15-electron binary64 conservation/detailed-balance/entropy ledger and unchanged native mu-tau `1e-10` co-gate are absent. Production change `+0/-0/net 0`; token use `UNAVAILABLE`; blocker movement `0.25`; verdict `FAILURE_MODE_LOCALIZED_REJECT_IMPLEMENTATION`. | F-10 independent full-spectral validation boundary | only an explicit owner decision may authorize a new prospectively frozen design that supplies one coherent prior/selector/exact-node contract plus continuity, Jacobian, covariance, collision-specific binary64, native/weak co-gate, cost, and structural-independence evidence before implementation |

Agents must not silently reopen a frozen decision. A proposed reversal is a meta-finding with new evidence and an explicit reopen condition.

---

## Source: `.agent-harness/context/GATE_REGISTRY.json`

SHA-256: `6895b6d0b4e0ca7eb8fe86dcc1bf86a75aa2fd0252176c619d0d21e9d3b07248`

{
  "schema_version": 1,
  "gates": [
    {
      "gate_id": "G-F10C1-RADIAL",
      "spec_refs": ["docs/TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md#14-f-10---classical-neutrino-boltzmann-still-no-qke"],
      "statement": "The selected N48 representation must pass the frozen direct ladder, five-profile two-auxiliary envelope, FD moment, anchor, conservation, entropy, and constructor checks without tolerance widening.",
      "required_evidence": ["E-F10C1-VALIDATION", "E-F10C1-AUX", "E-F10C1-GRID"],
      "pass_condition": "Worst declared profile residual < 2.1%, auxiliary references agree < 3e-8, conservation <= 1.79e-14, domain loss <= 7.61e-5, and focused current-tree tests pass.",
      "fail_condition": "Any threshold is missed, the selected rule changes, or a required focused test fails.",
      "owner": "main",
      "status": "pass"
    },
    {
      "gate_id": "G-F10C1-REGRESSION",
      "spec_refs": ["docs/TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md#14-f-10---classical-neutrino-boltzmann-still-no-qke"],
      "statement": "The retained no-cache tree must pass formatting, release check, strict all-target Clippy, the full release Rust suite, and report build/inspection.",
      "required_evidence": ["E-F10C1-VALIDATION", "E-F10C1-REPORT"],
      "pass_condition": "All listed commands execute successfully on the retained tree; no undefined report references/citations or fatal LaTeX errors; edited pages visually clean.",
      "fail_condition": "Any required command fails or is skipped at final F-10C1 closeout.",
      "owner": "main",
      "status": "pass"
    },
    {
      "gate_id": "G-F10-PERFORMANCE",
      "spec_refs": ["docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md", "bbn_codex_anti_drift_cost_effective_policy.md"],
      "statement": "Repeated-run Rust design requires a material whole-endpoint reduction at the measured N48 cold wall without unacceptable RSS, solver-envelope, or correctness regression.",
      "required_evidence": ["E-F10C1-VALIDATION", "E-F10-PERF-PROFILE"],
      "pass_condition": "A retained change meets the controlling whole-endpoint threshold and preserves the frozen physics/solver gates.",
      "fail_condition": "Only segment speedup is shown, whole-endpoint movement is sub-threshold, or correctness/memory regressions make the candidate unacceptable.",
      "owner": "main",
      "status": "pass"
    },
    {
      "gate_id": "G-F10-CATALOGUE",
      "spec_refs": ["docs/TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md#14-f-10---classical-neutrino-boltzmann-still-no-qke"],
      "statement": "All frozen zero-lepton massless diagonal nine-row classical collision families must execute with checked multiplicities, kernels, coefficients, and rowwise physical invariants.",
      "required_evidence": ["E-F10C1-SPEC", "E-F10-CATALOGUE-TESTS"],
      "pass_condition": "Rows 1-9 and every folded subfamily execute and pass normalization, equilibrium/null, conservation, entropy, scaling, response, and coupled first-law tests.",
      "fail_condition": "Any row or folded subfamily is absent or any required rowwise test fails.",
      "owner": "main",
      "status": "pass"
    },
    {
      "gate_id": "G-F10-INDEPENDENT-FLRW",
      "spec_refs": ["docs/TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md#14-f-10---classical-neutrino-boltzmann-still-no-qke"],
      "statement": "The completed full-spectral collision-coupled FLRW endpoint must be checked by a structurally independent formulation/integrator under frozen inputs.",
      "required_evidence": ["E-F10-INDEPENDENT-ENDPOINT"],
      "pass_condition": "Independent full-spectral endpoint and discriminating blockwise observables agree within predeclared tolerances and limitations are adjudicated.",
      "fail_condition": "Only same-code/partial evidence exists, or a structurally independent candidate fails any frozen static, trajectory, endpoint, covariance, conservation, exchange, or uncertainty gate. The pointwise and Galerkin candidates fail before trajectory authority, and the three-node maximum-relative-entropy route is rejected before implementation because its derivation and executable selector do not define one coherent continuous/covariant collision method.",
      "owner": "main",
      "status": "fail"
    },
    {
      "gate_id": "G-F10-SCOPE",
      "spec_refs": ["docs/TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md#15-f-11---lrs-then-non-lrs-type-i"],
      "statement": "Current work must stop at F-10 and must not implement, validate, or schedule F-11 Type-I, QKE, or public production.",
      "required_evidence": ["E-F10C1-SPEC", "E-F10C1-CLAIMS"],
      "pass_condition": "Assignments and retained changes remain within F-10; F-11 stays OWNER-PAUSED and forbidden claims remain explicit.",
      "fail_condition": "Any current assignment or retained change opens F-11/QKE/public-production work without new owner authority.",
      "owner": "main",
      "status": "pass"
    },
    {
      "gate_id": "G-HARNESS-INTEGRITY",
      "spec_refs": ["AGENTS.md#mandatory-shared-context-protocol-for-subagent-workflows"],
      "statement": "The context index/pack, active run, assignments, unique result paths, hashes, trusted-session hook activation, and single-writer enforcement must satisfy the uploaded harness contract.",
      "required_evidence": ["E-HARNESS-VALIDATE", "E-HARNESS-AUDIT"],
      "pass_condition": "The explicit validator exits zero, trusted-session SubagentStart/SubagentStop activation is evidenced, registered result handoffs are automatically enforced, and no unauthorized shared or production write is attributable to a subagent.",
      "fail_condition": "The validator reports a stale/invalid contract, hook activation is unproved, automatic result enforcement is bypassed, or exclusive write ownership cannot be evidenced.",
      "owner": "main",
      "status": "fail"
    }
  ]
}
