# FB73 Publication Figure Renderer V2

## Scope

FB73 adds a new current-artifact figure renderer for the full E2E BBN line.
It does not reuse legacy plotting code.  The renderer consumes the current
FB60 full-BBN diagnostic suite, FB66 freedom-ladder sweep, FB70 continuous
AP65 span ladder, and FB72 AP80-FB71 weak-rate bridge artifacts, validates
their no-public/no-production/no-QKE/not-promoted boundaries, and writes a
hashed manifest plus four diagnostic PNG panels.

This is a diagnostic figure surface only.  It does not run a new solver,
promote public dispatch, claim production SMC validation, add QKE, or make the
all-freedom full-BBN path publication-ready.

## CPU Evidence

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/render_augmented_publication_figures_v2.py \
  --full-bbn-suite-artifact diagnostic_outputs/fb60_full_bbn_diagnostic_suite/fb60_full_bbn_diagnostic_suite_manifest.json \
  --freedom-sweep-artifact diagnostic_outputs/fb66_freedom_ladder_full_bbn_sweep/fb66_freedom_ladder_full_bbn_sweep.json \
  --continuous-span-ladder-artifact diagnostic_outputs/fb70_continuous_ap65_full_bbn_span_ladder_fd.json \
  --weak-rate-bridge-artifact diagnostic_outputs/fb72_full_bbn_weak_rate_bridge_smoke/fb72_full_bbn_weak_rate_bridge.json \
  --output-dir figures/augmented_publication_v2_current
```

Result:

- `passed=true`
- `diagnostic_figure_ready=true`
- `publication_figure_ready=false`
- `artifact_payload_sha256=6a62a214b0d13e38e5d76bac652c8ce02caf4ec93880157b49908a75a567ceae`
- manifest file SHA256 `ba5e899615765a524e3f31c3ec1124b8da8fb13685ea818b2b15aad501421aea`
- `plot_count=4`
- `full_bbn_T_final_MeV_min=0.004996944944314105`
- `full_bbn_T_final_MeV_max=0.005000010963688484`
- `freedom_sweep_completed_rows=8`
- `freedom_sweep_all_freedom_ready=true`
- `weak_rate_bridge_ready=true`
- `weak_rate_bridge_passed_pair_count=4`
- `continuous_ap65_physical_span_ready=false`
- `continuous_ap65_rows_reaching_endpoint=0`
- `publication_readiness_blocker=continuous_ap65_full_bbn_span_not_ready`

Plots:

- `fb73_00_full_bbn_endpoint_coverage.png`, 34414 bytes,
  SHA256 `251acb3e28622aefcffb6cbde43ac5132c818db2a15f3a4b74df3125d0b0056f`
- `fb73_01_freedom_ladder_terminal_yields.png`, 146528 bytes,
  SHA256 `83faf689839a1220998fee247b24f32a8635fcc8962bd1e43ea8157424d74133`
- `fb73_02_weak_rate_bridge_deltas.png`, 34632 bytes,
  SHA256 `893c2ad4a405fe722193a4dcbb7eb0791ee09d4653a9accc51386f71d6824e4c`
- `fb73_03_continuous_ap65_span_boundary.png`, 33445 bytes,
  SHA256 `44208363f0d244688f6c66b5971194c033ddef538c40e8504c6cf00064defada`

The blocker is explicit: FB60 and FB66 provide full-BBN endpoint diagnostic
rows, and FB72 links weak-rate evidence to the full-BBN weak/control index,
but the current continuous AP65 RHS span ladder remains hot-endpoint evidence
rather than a physical full-BBN continuous-source run.
