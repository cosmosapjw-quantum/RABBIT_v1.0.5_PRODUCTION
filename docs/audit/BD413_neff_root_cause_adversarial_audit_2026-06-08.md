# BD413 Neff Root-Cause Adversarial Audit

Date: 2026-06-08

Scope: augmented Type-I PSTF no-QKE AP65/3T solver line. QKE remains out of
scope. This is an evidence audit, not a publication/public-production claim and
not a new runtime gate.

## Executive Verdict

The current high cold-endpoint `N_eff_3T ~= 3.105-3.115` is a real no-QKE
diagnostic-proxy blocker, but the dominant cause is now much more specific:

1. `N_eff_3T` formula and one-vs-three neutrino counting are internally
   consistent. They do not explain the excess.
2. `_C_RATE=210` and the standalone 3T closure are not the immediate cause:
   starting a clean 3T tail at `(T_gamma,T_nu_e,T_nu_x)=(3,3,3) MeV` gives
   `N_eff_3T_asymptotic ~= 3.03476`.
3. Starting the same 3T closure at equal temperatures at `0.8 MeV` gives
   `N_eff_3T_asymptotic ~= 3.11486`, matching the BD199 collision-off endpoint
   and the current AP65 cold endpoint. This is the strongest root-cause signal.
4. Therefore the leading failure mode is not weak rates and not final readout.
   It is an AP65 initial-condition/model-stitching problem: phase-1 prerun
   updates the n/p state but leaves `T_nu_e0=T_nu_x0=T_gamma0=0.8 MeV`, so the
   thermo decoupling history above 0.8 MeV is skipped while neutrinos are still
   initialized too hot.
5. Collision source details still matter. q4 source-split artifacts show
   scalar `dQ` dominates the hot-span `N_eff_3T` path; `dA_only` changes q4
   `N_eff_3T` by about `-0.00974` relative to full, while `dQ_only` is within
   `+6.9e-5` of full. But endpoint high `N_eff` appears even with collision
   terms off, so collision normalization is second-tier, not the primary cause.

## Expected External Reference

The current local target is a no-QKE/classical proxy target, not a full Standard
Model QKE result. Still, external precision calculations are useful sanity
anchors: recent full Standard Model neutrino decoupling computations report
`N_eff ~= 3.0440` with finite-temperature QED and oscillation/collision effects
included. The local 3T code intentionally uses a lower no-QKE classical target
around `3.034`.

References used for context:

- Froustey, Pitrou, Volpe, arXiv:2008.01074, reports `N_eff = 3.0440` with
  flavour oscillations and modern plasma thermodynamics:
  https://arxiv.org/abs/2008.01074
- Bennett et al., arXiv:2012.02726, reports
  `N_eff_SM = 3.0440 +/- 0.0002` and discusses numerical/momentum-discretization
  uncertainty:
  https://arxiv.org/abs/2012.02726
- Mangano et al. relic decoupling record, reports the older `3.046` benchmark:
  https://www.iris.unina.it/handle/11588/202143

## Code Path Ledger

| Path | Status | Evidence | Consequence |
|---|---|---|---|
| `N_eff_from_3T` | IMPLEMENTED / narrow tests passed | `src/rabbit/thermo/nudec_coupled.py` computes `(T_nu_e/T_std)^4 + 2*(T_nu_x/T_std)^4`; JAX mirrors it in `src/rabbit/jax/nudec_coupled_jax.py`. | Counting/readout is unlikely to be the cause. |
| 3T closure calibration | VALIDATED by chain-of-code sweep | `diagnostic_outputs/bd413_neff_root_cause_audit/standalone_3t_sweep.tsv`. | Clean 3 MeV equal-temperature start gives `3.03476`, not `3.115`. |
| AP65 default temperature initialization | IMPLEMENTED and suspect | `_default_restart_kwargs` in `src/rabbit/validation/augmented_continuous_ap65_rhs.py` updates `Xn` under `phase1_prerun` but returns supplied `T_gamma0/T_nu_e0/T_nu_x0`. | Neutrino decoupling phase-1 thermal history is not propagated into AP65 start. |
| Standard anchor mode | IMPLEMENTED and suspect | `src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py` sets `T_nu_e0=T_nu_x0=T_gamma0` in standard anchor mode. | A `T_gamma0=0.8` standard anchor starts with neutrinos too hot. |
| Collision scalar `dQ` | PARTIAL | q4 full and `dQ_only` agree within `~6.9e-5`; direct/table RHS comparison shows solver-consumed q4 `dQ` matches table `total_energy_transfer/H` at final state. | Not primary endpoint cause, still important for q4 source physics. |
| Collision `dA` | PARTIAL | `dA_only` lowers q4 `N_eff_3T` by `~0.00974`, but its scalar `dQ` is zero by construction. | Strong indirect effect, not a valid physical default. |
| Weak rates | PRUNED as direct cause | `N_eff_3T` reads temperatures. Weak rates affect n/p and phase-2 network, not the formula directly. | Weak-rate sweeps may move `Yp`, not the primary `N_eff` blocker. |
| Shear/non-LRS leakage | PRUNED as primary cause | BD411/BD412 LRS and non-LRS endpoints are both high and very close; final shear is tiny. | Fix FLRW thermal history before Bianchi-specific changes. |

## Quantified Evidence

### Cold Endpoint Time Evolution

The common trajectory pattern is not a sudden late readout glitch. `N_eff_3T`
declines from hot pre-asymptotic values to a stable high cold endpoint:

| Artifact | Row history |
|---|---|
| BD199 collision off | `N=1:9.578396`, `2:5.232630`, `3:3.161249`, `4:3.114863`, `4.8:3.114863` |
| BD199 radial collision on q3 | `N=1:9.559652`, `2:5.218497`, `3:3.151747`, `4:3.105381`, `4.8:3.105381` |
| BD411 current LRS | `N=1:9.578401`, `2:5.232642`, `2.75:3.319664`, `3:3.161257`, `4:3.114871`, `4.8:3.114871` |
| BD412 step-base LRS/non-LRS | `N=1:9.578541`, `2:5.232768`, `2.75:3.319752`, `3:3.161344`, `4:3.114957`, `4.8:3.114957` |

Raw extracted table:
`diagnostic_outputs/bd413_neff_root_cause_audit/neff_time_series.tsv`.

### Final Temperature Ratios

Cold endpoint ratios are about 0.9-1.0% hotter than the standard instantaneous
ratio `T_std/T_gamma=(4/11)^(1/3)=0.7137658555`.

| Run | `T_nu_e/T_gamma` | `T_nu_x/T_gamma` | `T_nu_e/T_std` | `T_nu_x/T_std` | `N_eff_3T` |
|---|---:|---:|---:|---:|---:|
| BD199 collision off | `0.72081237` | `0.72034663` | `1.00987231` | `1.00921980` | `3.11486258` |
| BD199 radial q3 collision on | `0.71995305` | `0.71995305` | `1.00866839` | `1.00866839` | `3.10538099` |
| BD411 LRS | `0.72081312` | `0.72034697` | `1.00987336` | `1.00922027` | `3.11487082` |
| BD412 step-base LRS/non-LRS | `0.72082092` | `0.72035052` | `1.00988428` | `1.00922525` | `3.11495676` |

Raw extracted table:
`diagnostic_outputs/bd413_neff_root_cause_audit/final_temperature_ratios.tsv`.

### Standalone 3T Start-Temperature Sweep

This is the decisive discriminator:

| 3T equal-temperature start | `_C_RATE` | `N_eff_3T_asymptotic` |
|---:|---:|---:|
| `T_start=3.0 MeV` | `0` | `3.00069530` |
| `T_start=3.0 MeV` | `210` | `3.03476457` |
| `T_start=0.8 MeV` | `0` | `3.10538099` |
| `T_start=0.8 MeV` | `210` | `3.11486260` |
| `T_start=0.8 MeV` | `280` | `3.11796933` |

The endpoint value is reproduced without AP65 collision dynamics by simply
starting the calibrated 3T closure too late and too hot for neutrinos.

Raw extracted table:
`diagnostic_outputs/bd413_neff_root_cause_audit/standalone_3t_start_temperature_sweep.tsv`.

### Collision Source Split

| q4 source split | `N_eff_3T` at `N=2.75` | `N_eff_3T_asymptotic tail` | Interpretation |
|---|---:|---:|---|
| full / boundary trace | `3.31966368` | `3.11487089` | baseline q4 full source |
| `dQ_only` | `3.31973295` | `3.11493832` | scalar energy source nearly controls hot-span proxy |
| `dA_only` | `3.30992639` | `3.10538376` | hierarchy source changes proxy indirectly; scalar `dQ` is zero |
| source zero | failed by final activation row | incomplete | not a clean endpoint comparator |

Direct/table RHS comparison at q4 `N=2.75` shows final-state solver-consumed
`rhs_final_collision_dQ_*` equals `total_energy_transfer/H` from the 3T table
for full and `dQ_only` rows:

`diagnostic_outputs/bd413_neff_root_cause_audit/direct_vs_table_rhs.tsv`.

This weakens the hypothesis that q4 scalar `dQ` is simply over-normalized at
the sampled final state. It does not prove all source moments are correct
through the full trajectory.

## Hypothesis Tree With Pruning

### H1. Readout formula / degeneracy bug

Status: PRUNED.

Reasoning: Python and JAX formulas use the same `1 + 2` pair convention. Tests
cover formula parity and the heavy-bank heat-capacity convention. The observed
excess is exactly reproduced by hotter neutrino temperature ratios, so no
additional degeneracy factor is needed to explain it.

### H2. `_C_RATE` calibration too high

Status: PARTIAL / not primary.

Evidence: Clean 3 MeV standalone tail with `_C_RATE=210` gives `3.03476`, in
the local no-QKE target band. Retuning C alone cannot explain why AP65 collision
off at `T_gamma0=0.8` gives `3.11486`. However, after AP65 start-state repair,
`C_RATE` should still be rechecked against the no-QKE classical target.

### H3. EOS photon/electron entropy transfer bug

Status: PARTIAL.

Evidence: `C_RATE=0` from 3 MeV gives `3.0007`; `C_RATE=0` from 0.8 MeV gives
`3.10538`. This means the EOS entropy transfer path can produce the expected
instantaneous-like limit if started early enough. It does not rule out EOS
corrections at sub-percent level, but the large excess is explained by the
start-temperature mismatch.

### H4. AP65 initial thermal history missing

Status: SUPPORTED / ACTIONABLE_NEXT.

Evidence:

- `phase1_prerun` in `_default_restart_kwargs` integrates weak-only n/p history
  from `phase1_prerun_T_start_MeV` to `T_gamma0`, then only uses `Xn_at_T0`.
- It does not update `T_nu_e0_MeV` or `T_nu_x0_MeV`.
- Standard anchor mode sets `T_nu_e0_MeV=T_nu_x0_MeV=T_gamma0_MeV`.
- Standalone 3T equal-temperature start at `0.8 MeV` gives exactly the observed
  high endpoint.

This is the leading root-cause hypothesis.

### H5. Collision scalar dQ normalization/composition bug

Status: PARTIAL / second-tier.

Evidence: q4 split shows `dQ_only` almost matches full source, and `dA_only`
lowers the q4/cold-tail proxy by about `0.0097`. BD199 radial q3 collision-on
endpoint is `3.10538`, lower than collision-off `3.11486`. However, the high
endpoint exists without collision terms, and direct q4 final-state comparison
shows the scalar moment matches table `total_energy_transfer/H`.

### H6. Weak-rate physics drives Neff

Status: PRUNED as direct cause.

Reasoning: weak rates feed phase-2 abundance/network evolution. `N_eff_3T` is
computed from temperatures. Weak-rate changes may influence `Yp` and Hubble
history indirectly, but they cannot be the primary direct source of the final
temperature-ratio excess.

### H7. Nonzero shear / Bianchi extension leakage

Status: PRUNED as primary.

Evidence: LRS and non-LRS cold endpoints are both high and nearly identical.
Final shear is near zero in the relevant FLRW controls. Fix the FLRW thermal
history before treating this as a Bianchi extension issue.

## Multi-Agent Debate Summary

Four independent subagents were used and then closed:

1. 3T closure auditor: readout/counting is consistent; endpoint excess is a
   hotter neutrino ratio; collision-bank degeneracy remains worth checking but
   is not needed to explain the main endpoint failure.
2. Collision/Boltzmann auditor: `dQ` affects 3T temperatures directly; `dA`
   affects distributions and weak/stress history indirectly; q4 full and
   `dQ_only` agree, while `dA_only` is a harsh diagnostic ablation.
3. Artifact auditor: timing and reuse knobs barely move `N_eff`; meaningful
   movement comes from collision/source physics and the old q3 radial-vs-off
   endpoint delta.
4. Red-team/MCTS auditor: start with pure 3T C/EOS, same-state AP65-vs-table
   dT, and entropy-only checks before another endpoint run. Local chain-of-code
   checks completed these discriminators and elevated the start-temperature
   mismatch to the top.

## What Succeeded / Failed In This Audit

Succeeded:

- Reproduced the high endpoint as a simple 3T start-temperature effect.
- Quantified the time evolution of `N_eff_3T` across BD199/BD350/BD351/BD411/BD412.
- Quantified direct q4 source component effects.
- Verified closure invariants still pass.
- Confirmed subagents were closed after result integration.

Failed or incomplete:

- No new long endpoint was run in this audit. Existing BD199/BD350/BD351/BD411/BD412
  long endpoint artifacts were used.
- No implementation fix was made here.
- No distribution-integral `N_eff_dist` exists yet, so `A_modes` energy density
  is not directly compared to the 3T proxy.
- No AP65 endpoint run from a thermally prerun restart was executed yet.

## Next Implementation Plan

Do not add another audit gate. Move the blocker by changing the physical start
state semantics and validating with actual endpoint runs.

### PR-N1: Thermo Phase-1 Restart State

Add an opt-in AP65 initial policy that evolves the 3T thermo closure from
`phase1_prerun_T_start_MeV` to `T_gamma0_MeV` and uses the resulting
`T_nu_e0_MeV/T_nu_x0_MeV` in restart kwargs. Keep raw state visible and serialize
both requested and effective initial temperatures. Do not default-on until
endpoint evidence passes.

Expected effect: starting at `T_gamma0=0.8 MeV` should initialize
`T_nu/T_gamma ~= 0.705-0.706` instead of `1.0`; subsequent cold endpoint should
move from `3.1149` toward the local no-QKE target band.

### PR-N2: Current-Head Long Endpoint A/B

Run paired cold endpoints:

- old equal-temperature AP65 start;
- thermally prerun AP65 start;
- collision-off;
- collision-on q3/q4 if feasible.

Record `T_nu/T_gamma`, `N_eff_3T`, `N_eff_3T_asymptotic`, `Yp`, D/H, collision
dQ integrals, and LRS/non-LRS parity.

### PR-N3: Distribution Energy Readout

Add a diagnostic-only `N_eff_dist` from reconstructed neutrino distributions
and compare it with `N_eff_3T`. This is necessary before treating `dA` effects
as physically understood. No clipping; no promotion gate.

### PR-N4: Collision Source Normalization Audit

After start-state repair, re-run source component splits and q-order/pair-leg
ablation. Verify whether radial raw dQ, `standard_3t_plasma`, and angular
zero-scalar policies agree on scalar energy conservation.

## Commands Run

Local commands executed during this audit:

```bash
git status --short
git log --oneline -20
rg -n "def asymptotic_N_eff_3T_payload|N_eff_3T|total_energy_transfer|C_RATE|..."
sed -n '380,620p' src/rabbit/thermo/nudec_coupled.py
sed -n '1,260p' src/rabbit/thermo/incomplete_decoupling.py
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_three_temperature_closure_invariants.py
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python - <<'PY' ... artifact Neff summary ... PY
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python - <<'PY' ... standalone 3T C_RATE sweep ... PY
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python - <<'PY' ... start-temperature sweep ... PY
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python - <<'PY' ... final temperature ratios ... PY
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python - <<'PY' ... direct/table RHS comparison ... PY
```

Validation result:

- `tests/test_three_temperature_closure_invariants.py`: `6 passed`.

Generated local evidence:

- `diagnostic_outputs/bd413_neff_root_cause_audit/neff_artifact_summary.json`
- `diagnostic_outputs/bd413_neff_root_cause_audit/neff_artifact_summary.tsv`
- `diagnostic_outputs/bd413_neff_root_cause_audit/neff_time_series.tsv`
- `diagnostic_outputs/bd413_neff_root_cause_audit/standalone_3t_sweep.tsv`
- `diagnostic_outputs/bd413_neff_root_cause_audit/standalone_3t_start_temperature_sweep.tsv`
- `diagnostic_outputs/bd413_neff_root_cause_audit/final_temperature_ratios.tsv`
- `diagnostic_outputs/bd413_neff_root_cause_audit/direct_vs_table_rhs.tsv`

## Anti-Drift Self-Audit

- `real_blocker_moved`: yes, from broad "Neff wrong" to a concrete,
  quantitatively supported AP65 thermo-start mismatch.
- `gate_removed_or_consolidated`: no new gate added.
- `raw_state_preserved`: yes, only read artifacts and wrote audit extracts.
- `verification`: closure invariant tests passed; chain-of-code TSV/JSON
  extracts generated.
- `remaining_blocker`: implement thermally consistent AP65 restart and validate
  with fresh cold endpoint A/B, then add distribution-energy readout.
