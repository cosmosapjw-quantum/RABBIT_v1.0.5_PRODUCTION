# FB74 Publication Figure Bundle QA

## Scope

FB74 validates and packages the FB73 current-artifact diagnostic publication
figures.  It consumes an existing FB73 manifest, recomputes the embedded FB73
payload hash, verifies the FB73 file hash, validates every referenced source
artifact and PNG hash, checks diagnostic-only claim labels and captions, copies
the four PNGs into a clean QA bundle, and writes a hashed FB74 manifest.

This is a QA/copy gate only.  It does not rerender figures, call legacy plot
scripts, run a solver, register public dispatch, claim production SMC
validation, add QKE, or make the all-freedom full-BBN path publication-ready.

## CPU Evidence

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/package_augmented_publication_figure_bundle_qa.py \
  --fb73-figure-manifest figures/augmented_publication_v2_current/fb73_publication_figures_v2_manifest.json \
  --output-dir figures/augmented_publication_v2_current_qa \
  --command "PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/render_augmented_publication_figures_v2.py --full-bbn-suite-artifact diagnostic_outputs/fb60_full_bbn_diagnostic_suite/fb60_full_bbn_diagnostic_suite_manifest.json --freedom-sweep-artifact diagnostic_outputs/fb66_freedom_ladder_full_bbn_sweep/fb66_freedom_ladder_full_bbn_sweep.json --continuous-span-ladder-artifact diagnostic_outputs/fb70_continuous_ap65_full_bbn_span_ladder_fd.json --weak-rate-bridge-artifact diagnostic_outputs/fb72_full_bbn_weak_rate_bridge_smoke/fb72_full_bbn_weak_rate_bridge.json --output-dir figures/augmented_publication_v2_current"
```

Result:

- `passed=true`
- `bundle_qa_passed=true`
- `publication_figure_ready=false`
- `artifact_payload_sha256=e22a8f6ea68b24e376b1ded12b6bb531199005bead8b4b7ef6d187f76f645e45`
- manifest file SHA256 `d609ba75756bbb9be0c7dd1fa256b6ad167eda1974a005527f8a08354c664cd5`
- source FB73 file SHA256 `ba5e899615765a524e3f31c3ec1124b8da8fb13685ea818b2b15aad501421aea`
- source FB73 payload SHA256 `6a62a214b0d13e38e5d76bac652c8ce02caf4ec93880157b49908a75a567ceae`
- `plot_count=4`
- `copied_plot_count=4`
- `qa_checks=10`
- rerunning the same command produced the same FB74 payload and manifest file
  hashes; transient cleanup paths are omitted from the manifest and the FB73
  source timestamp is copied under an explicit reproducible-hash policy

Copied plots:

- `00_fb73_00_full_bbn_endpoint_coverage.png`, 34414 bytes,
  SHA256 `251acb3e28622aefcffb6cbde43ac5132c818db2a15f3a4b74df3125d0b0056f`
- `01_fb73_01_freedom_ladder_terminal_yields.png`, 146528 bytes,
  SHA256 `83faf689839a1220998fee247b24f32a8635fcc8962bd1e43ea8157424d74133`
- `02_fb73_02_weak_rate_bridge_deltas.png`, 34632 bytes,
  SHA256 `893c2ad4a405fe722193a4dcbb7eb0791ee09d4653a9accc51386f71d6824e4c`
- `03_fb73_03_continuous_ap65_span_boundary.png`, 33445 bytes,
  SHA256 `44208363f0d244688f6c66b5971194c033ddef538c40e8504c6cf00064defada`

The retained blocker is inherited from FB73: the continuous AP65 full-BBN span
ladder is still hot-endpoint evidence, so this bundle is diagnostic and not a
publication-ready physics result.
