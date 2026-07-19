# PR-G Breakeven

Local rig:

- Python: `venv/bin/python`
- GPU: AMD Radeon RX 6950 XT
- workload: `run_char_batch_tier1(...)`
- settings: `correction_level=0`, `N_mu=6`, `N_q=6`, `rtol=1e-6`,
  `atol=1e-8`, `max_steps=2000`
- env: `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `XLA_PYTHON_CLIENT_ALLOCATOR=platform`,
  `RABBIT_JAX_CACHE_DIR=/tmp/rabbit_jax_cache`

Warm throughput:

| N | CPU ms/solve | GPU ms/solve |
|---|---:|---:|
| 1 | 48.23 | 1156.18 |
| 8 | 21.46 | 141.22 |
| 32 | 11.89 | 37.44 |
| 64 | 10.48 | 19.12 |
| 128 | 12.91 | 9.91 |
| 256 | 11.83 | 7.44 |

Conclusion:

- GPU is a clear loss for small batches
- practical breakeven is around `N≈128`
- `N=256` is already worthwhile on this rig
