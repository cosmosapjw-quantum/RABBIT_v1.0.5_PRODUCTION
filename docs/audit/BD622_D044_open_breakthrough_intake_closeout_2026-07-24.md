# BD622 D-044 — Open-breakthrough intake closeout (2026-07-24)

## Scope

Read-only intake of the external response `F10_OPEN_BREAKTHROUGH_LAB_ab95ff52.zip`
(ZIP sha256 prefix `81201b99`) to the D-043-corrected D-042R executable
breakthrough request. Registered run
`run-20260724-f10-open-breakthrough-intake`; closeout run
`run-20260724-f10-open-breakthrough-intake-closeout`; context `815eb4b9`.
No scientific execution of any kind.

## Verdict

```text
package_verdict=PROVENANCE_FAIL
blocking_gate=XG-00-PROVENANCE
PR_DAG=UNREACHABLE
terminal_choice=DO_NOT_REOPEN
```

ZIP transport and the 15-file manifest verify; package admission fails. The
ten binding contract gaps (recorded in the evidence map under
`observed_contract_gaps`): no prospectively sealed `EXPERIMENT_CONTRACT.json`;
no single `run_lab.sh` entry point; three methods run instead of one
prospectively selected finalist; the lab imports the live RABBIT comparator and
emits physical collision actions despite the repository-free nonphysical
boundary; no semantic constant/fixture or physics-mutation evidence; no two
fresh-process byte-identical replays; no raw stdout/stderr or digest-pinned
network-off execution record; no accounting block or exact terminal
owner-choice line; the external `RUN_PLAN` left `status=initialized` after six
result envelopes; numpy/scipy named without version or image pinning.

## Row-6 outcome

Row-6 localization is retained as bounded external evidence only. Label-only
OCAR is rejected: with `P` acting on mu/tau labels at a fixed ordered chart
`z=(1,2;3,4)`, the correct identity is `D(Pf;z) = -D(f;Rz)` where `R`
exchanges incoming/outgoing slots — exact witness `D(f;z)=1/12`,
`D(Pf;z)=5/144=-D(f;Rz)`, not `-1/12`. The narrower, explicitly
orientation-closed `C-R6-ORBIT-CHART` remains `PROPOSED` with
`OWNER-A=REQUIRED_NOT_GRANTED` at intake time.

## Evidence

| Artifact | Path | SHA-256 |
|---|---|---|
| Final adjudication | `.agent-harness/runs/run-20260724-f10-open-breakthrough-intake/results/A-OPEN-INTAKE-ADJUDICATOR.json` | `7beccc8844265af1c91fb00c47c0f44516794c2c54e925dceaedabee8edbc5a4` |
| Merged results (49/49) | `.agent-harness/runs/run-20260724-f10-open-breakthrough-intake/MERGED_RESULTS.json` | `f69bbec38e3f5d5d13e62f0445783f1ae160c0edc9c2fc3cb06bc846722455d6` |
| Evidence map | `.agent-harness/runs/run-20260724-f10-open-breakthrough-intake/artifacts/INTAKE_EVIDENCE_MAP.json` | `a560178bb1efa895d9ba1df8c80d4e78c18f7c410fdf18166db9afef4cc03bfb` |
| Closeout run plan | `.agent-harness/runs/run-20260724-f10-open-breakthrough-intake-closeout/RUN_PLAN.json` | `98b0bcf77a85ecc3ab2c4473369065f731ae3db817d6ecae3d87a7f612b66a2f` |

Both run directories are retained in git by force-add (precedent: EXT-01,
commit `204d3fa`). Canonical decision text: `FROZEN_DECISIONS.md` D-044 row;
ledger row: `docs/harness/VALIDATION_LEDGER.md` (2026-07-24 D-044 entry).

## Boundary

Every scientific gate state is unchanged. The intake authorizes no package
repair or admission, no repository replay, no `_independent_noqke.py` edit,
no W7/B3/T01--T12, no Rust/JAX, no Radau/trajectory/endpoint, no unblinding,
no F-11/Bianchi, no QKE, and no public work.
