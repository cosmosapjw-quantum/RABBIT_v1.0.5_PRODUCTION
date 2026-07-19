# BD414 External Expanded Audit — Executive Verdict

Date: 2026-06-08
Scope: RABBIT augmented Type-I PSTF no-QKE AP65/BBN solver. External audit of
both the `N_eff_3T ~= 3.115` cold-endpoint physics blocker and the code/
algorithm/performance/overengineering surface. QKE out of scope; no public-
production or publication-ready claims; CPU-JAX + in-tree Rodas5P/AP65 is the
target backend.

Environment caveat: JAX is **not** installed in the audit sandbox. The decisive
physics experiment is reproducible without JAX (the 3T closure imports only
numpy), so it was reproduced independently; the full AP65 endpoint A/B and the
JAX `N_eff_from_3T` were verified by code inspection / algebraic identity. See
`missing_information_and_files.md`.

## One-Paragraph Verdict

The cold-endpoint `N_eff_3T ~= 3.115` is **fully explained and independently
reproduced** as an AP65 thermal-start defect: the standard-anchor run initialises
neutrinos at the photon temperature (`T_nu = T_gamma = 0.8 MeV`) because the
phase-1 prerun updates only the n/p abundance and not the neutrino thermal
history. The fix (PR-N1: integrate the existing 3T closure from ~3 MeV to the
run's `T_gamma0` and use the resulting neutrino temperatures) is a principled
physical repair — proven by a composability check that the entire ~0.080 excess
is the equal-temperature reset — not a retuning. It moves the endpoint to ~3.035,
inside the encoded no-QKE band `[3.00, 3.06]`. The closure conventions and the
`_C_RATE=210` calibrated proxy are sound and correctly scoped (the no-oscillation
analogue of the full-SM 3.043-3.045). The collision `dQ` routing is energy-
conserving and matches the table; the cold endpoint is collision-independent. On
the engineering side, every BD411/BD412 performance number reproduces: the phase-2
network corrector dominates the wall (50.6% -> 64.3% after payload reuse), the
dense linear algebra is negligible, memory (~16 GB non-LRS) is the binding
constraint, and the overengineering is concentrated in out-of-scope publication/
readiness wrappers that should be deleted first. A whole-language rewrite is not
justified.

## Verdicts By Topic

| Topic | Verdict |
|---|---|
| 01 N_eff root cause / AP65 thermal start | **SUPPORTED — THERMAL_START_PRIMARY** |
| 02 3T closure / expected no-QKE N_eff | **SUPPORTED** |
| 03 Boltzmann collision source dQ/dA | **PARTIAL** (dQ energy-conserving + table-matched; dA indirect; full-trajectory audit + `N_eff_dist` missing) |
| 04 Long-run ablation matrix / component wall | **SUPPORTED** (+ ablation-label/override caveat) |
| 05 Performance (phase2/payload/solver/memory) | **SUPPORTED** |
| 06 Overengineering deflation TOP-10 | **SUPPORTED** |
| 07 Solver algorithm (Rodas5P / phase-2 corrector) | **SUPPORTED** |
| 08 Validation / parity / raw-state gaps | **PARTIAL** (FLRW geometry/parity/raw-state OK; thermal start + `N_eff_dist` + Yp/D/H re-val gate Bianchi claims) |

## Highest-Value Findings

1. **Root cause reproduced** (probe p01, max diff 1.2e-8) and **fix proven
   principled** (probe p02 composability 8.7e-9): the ~0.080 excess is entirely
   the equal-temperature reset at 0.8 MeV.
2. **Correction to the internal plan**: BD413's PR-N1 expectation of `T_nu/T_gamma
   ~= 0.705-0.706 at 0.8 MeV` is physically wrong (that is the asymptotic ratio;
   the correct 0.8-MeV ratio is ~0.994). Hard-coding 0.705 would over-cool
   neutrinos. **PR-N1 must compute the ratio, not hardcode it** — and must not
   hardcode 0.8 MeV or 0.994 either, since both are FLRW-specific and a Bianchi
   shear-modified Hubble shifts the mapping.
3. **Performance picture confirmed and reframed**: the 391 s phase-2
   "bookkeeping" is Python orchestration overhead (a residual timer), the cheapest
   real optimization target — by deletion, not new abstraction.
4. **Architecture question answered**: the JAX-Rodas5P / host-phase-2 split
   already exists; no phase-2 change can move the `N_eff` blocker (endpoint is
   phase-2-independent), so performance and physics must be kept on separate
   ledgers.

## Bottom Line

Land PR-N1 (thermal start, computed not hardcoded) first; it is the single fix
for the physics blocker and is small and reuses existing code. Defer every
performance and cleanup PR behind it. Do not claim QKE, public-production,
publication-ready, or Bianchi-extension validity. See
`MAIN_PR_LIST_RECOMMENDATION.md` for the ordered PR plan.
