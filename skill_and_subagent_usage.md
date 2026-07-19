# Skill And Subagent Usage

Date: 2026-06-02

## Skill Routing Plan

| Need | Skill / Harness | Status | Use |
|---|---|---|---|
| Repo mapping and git archaeology | `codebase-recon` | Available and used | Repo vitals, hotspots, bug-magnet scan, recent history. |
| Claim falsification / debugging discipline | `superpowers:systematic-debugging` | Available and used | No fixes before source/artifact probes; hypotheses were turned into focused tests/probes. |
| Probe/test generation | `superpowers:test-driven-development` | Available and used for optional patch discipline | Added a small invariant test file and verified it. Full red-first was limited because this was an audit pass and the tested behavior already existed. |
| Architecture review | `brooks-audit` | Available and used | Module-size and runtime-spine analysis. |
| Technical-debt prioritization | `brooks-debt` | Available and used | PR priority and deletion/consolidation candidates. |
| Test quality review | `brooks-test` | Available and used | Count-lock/skeleton-test critique and invariant-test plan. |
| Patch/risk review | `superpowers:requesting-code-review` plus Red-Team subagent | Available; adapted | Used role-agent red-team and local review rather than a merge-style PR review because this was an audit pass. |
| Parallel role audit | `superpowers:dispatching-parallel-agents` and `multi_agent_v1` | Available, partially constrained | Explorer was explicitly forbidden. One new worker agent was spawned; additional spawn hit thread limit, so existing non-Explorer agents were reused. |

## Fallbacks Used

- The first attempted path for `using-superpowers` under `.codex/skills/.system`
  was absent. Fallback used the installed Superpowers plugin path:
  `/home/cosmosapjw/.codex/plugins/cache/openai-curated/superpowers/6188456f/skills/using-superpowers/SKILL.md`.
- A new solver/performance subagent spawn hit the thread limit. Fallback:
  reused already-open non-Explorer subagents via `send_input`, and simulated
  Audit Orchestrator / Code Probe Engineer locally.
- System `python` was not used for pytest; repo `venv/bin/python` was used.

## Subagent Assignment

Explorer-style subagent was not used.

| Role | Agent | Mode | Files / Evidence Focus | Outcome |
|---|---|---|---|---|
| Audit Orchestrator | Main agent | Local | All reports, source, tests, artifacts | Maintained claim ledger, conflict resolution, final plan. |
| Physics Invariant Auditor | `019e852d-e179-7831-b88f-3220512120fa` | Worker | `augmented_pstf_distribution.py`, `augmented_collision_bridge.py`, `nudec_coupled.py`, artifacts | Confirmed local logit/closure/3T/heavy-bank invariants; kept endpoint `N_eff_3T` parity unresolved. |
| Numerical Solver Auditor | `019e7836-a5e8-77d2-9655-9ee82811f33a` | Reused worker | AP65 RHS, JAX Rodas5P, linear strategies, BD278 telemetry | Confirmed dense AP65 LU path and unwired low-rank/block solve; recommended A/B probes. |
| Performance and Memory Auditor | `019e7836-a5e8-77d2-9655-9ee82811f33a` | Same worker | RSS fields, JAX compile/runtime, diagnostic JSON | Confirmed missing per-row RSS/VmHWM/tracemalloc; no language rewrite before profiling. |
| Architecture / Technical Debt Surgeon | `019e7836-b71e-7df0-abeb-5e1ff32507c5` | Reused worker | AP65 RHS, span ladder, Teff, validation plumbing | Confirmed god-modules, Teff import reachability, fail-open default risks. |
| Test and Invariant Gate Designer | `019e7836-b71e-7df0-abeb-5e1ff32507c5` | Same worker | Tests and invariant candidates | Recommended physics/property tests replacing count locks. |
| Git History Archaeologist | `019e7846-667d-7231-ac8c-f9758e94e3bd` | Reused worker | Git history, artifacts, diagnostics | Identified BD261-BD280 sentinel/packet emphasis; separated resolved vs active blockers. |
| Evidence and Artifact Auditor | `019e7846-667d-7231-ac8c-f9758e94e3bd` | Same worker | BD199/BD278/BD279 artifacts, schemas | Separated preserve fields from noisy meta-status plumbing; marked packet-gap claims stale for full repo. |
| Code Probe Engineer | Main agent | Local | pytest, JSON probes, source refs | Wrote and ran `tests/test_three_temperature_closure_invariants.py`; ran low-rank/block/collision probes. |
| Red-Team Reviewer | `019e812b-5114-7661-b18e-2d28dced418c` | Reused worker | Reports, projection/default risks | Warned that projection/default artifacts and `N_eff_3T` parity can invalidate a solver-first interpretation. |
| PR Planner / Release Captain | `019e8149-d9eb-7601-9028-5f402a187c1f` | Reused worker | Reports, source, anti-drift | Proposed six PRs; enforced no new gates without consolidation/deletion. |

## Role Conclusions

### Physics Invariant Auditor

- SUPPORTED: logit convention and distribution-to-augmented RHS.
- SUPPORTED: occupation-space 3T energy closure before logit conversion.
- SUPPORTED locally: heavy-bank degeneracy/sign and 3T equation denominators.
- PARTIAL: `N_eff_3T` physical interpretation; code definition is consistent,
  but endpoint parity is unresolved.
- CONTRADICTED if overbroad: collisional `ell_max=2` exactness.

### Numerical Solver / Performance Auditor

- SUPPORTED: AP65 endpoint path still forms dense `W` and uses LU.
- SUPPORTED: block/low-rank/JFNK/GMRES pieces exist, but AP65 endpoint does not
  route through them.
- PARTIAL: compile/runtime and RSS attribution; fields are missing from endpoint
  artifacts.
- Recommended first solver action: a dense-vs-block/low-rank A/B probe with
  endpoint parity and RSS telemetry.

### Architecture / Test Auditor

- SUPPORTED: AP65 RHS and span ladder are god modules.
- SUPPORTED: validation evidence plumbing dominates and should be consolidated.
- PARTIAL: Teff deletion; it is deprecated but import-reachable and needs a
  call-graph/test fence before removal.
- SUPPORTED: count-lock tests should be replaced by invariant/property tests.

### History / Evidence Auditor

- SUPPORTED: BD279 "missing modules" is a packet gap, not a current repo gap.
- SUPPORTED: recent commits heavily expanded endpoint sentinels/shards/audit
  packets.
- SUPPORTED: preserve raw observable/provenance fields; prune noisy nested
  meta-status fields over time.

### Red-Team Reviewer

- Projection may hide an unprojected FLRW collision bug.
- `N_eff_3T` inconsistency can invalidate a solver-first interpretation.
- Silent defaults can manufacture apparent endpoint success.
- Packet artifacts may be stale; at least one fresh q4 row should be run before
  strong endpoint claims.

### PR Planner

- Keep PR queue to six or fewer real blocker moves.
- Start with parity/RSS/solve wiring, not more claim gates.
- Every PR should delete, consolidate, or demote obsolete plumbing when adding
  a new test or runtime path.

## Disagreements And Final Resolution

| Disagreement | Position A | Position B | Resolution |
|---|---|---|---|
| Solver-first versus physics-parity-first | BD280/solver agents prioritize dense solve wiring. | BD279/red-team prioritize `N_eff_3T` parity and projection/default confounds. | PR plan pairs them: first PR wires measurement/prototype solver path only with parity/RSS; second PR resolves controlled LRS/non-LRS parity. No production/default switch before both. |
| Heavy-bank bug versus local algebra healthy | BD279 points to cold heavy-bank endpoint signature. | BD280 and local tests support bridge/3T sign/degeneracy. | Classify as local SUPPORTED, endpoint PARTIAL. Need parity pair and source-policy provenance, not blind sign flips. |
| Delete Teff now versus later | Architecture wants cleanup. | Source scan shows import reachability. | Defer hard deletion until call graph and rejection tests pass; plan deletion/consolidation PR. |
| Count-lock tests useful versus harmful | Count locks preserve matrix shape. | They are weak physics evidence. | Keep minimal structural smoke, but replace main assurance with invariants and endpoint parity tests. |

## Files Touched By Main Agent

- Added `tests/test_three_temperature_closure_invariants.py`.
- Added audit deliverables:
  - `internal_reaudit_report.md`
  - `hypothesis_falsification_matrix.md`
  - `skill_and_subagent_usage.md`
  - `pr_acceleration_plan.md`
