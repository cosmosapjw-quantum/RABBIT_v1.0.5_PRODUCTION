# PR-A Stage 1 — Internal Repo Verification

## Paper provenance

`pdftotext -f 16 -l 18 docs/RABBIT_report/RABBIT_report.pdf -` was used
to re-read report pages 16-18. The OCR text around eq (51) reads:

> `Jj = dμj,0 / dμj = e^{-6S} μj (1-μ²j,0)^2 / [μj,0 (1-μ²j)^2]`

That OCR string is internally inconsistent with the ODE written two
paragraphs later in eq (55), so the audit treats the PDF text as
ambiguous rather than authoritative on the sign convention.

## J-related call sites audited

`rg -n "dJ_j|J_vals|i_J|characteristic_rhs_jax|jacobian_jax" src/rabbit/jax`
returned the full JAX call surface:

- `src/rabbit/jax/characteristic_rays_jax.py:24-43`
  `jacobian_jax(...)` added. Needs update: yes.
- `src/rabbit/jax/characteristic_rays_jax.py:55-69`
  `characteristic_rhs_jax(...)` now returns `(dI, dS)` only. Needs
  update: yes.
- `src/rabbit/jax/driver_typeI_char.py:148`
  `i_J = -1` sentinel in the post-PR-A layout. Needs update: yes.
- `src/rabbit/jax/driver_typeI_char.py:251,254,285`
  `J_vals` reconstructed analytically and consumed by stress/monopole
  extractors. Needs update: yes.
- `src/rabbit/jax/driver_typeI_char.py:263`
  RHS unpack site changed from `(dI, dJ, dS)` to `(dI, dS)`. Needs
  update: yes.

No stale three-value unpack of `characteristic_rhs_jax` remained after
the refactor.

## SciPy divergence check

The SciPy reference still evolves `J_j` numerically:

- `src/rabbit/transport/characteristic_rays.py:90-109`
  carries `(dI, dJ, dS)`.
- `src/rabbit/drivers/full_coupled_typeI.py:142-170,749-861,1154-1157`
  still allocates `J` state slots, seeds them to unity, and advances
  them through the solve.

Verdict: **expected divergence**. JAX is now lower-DOF, but parity is
defined at the observable level (`Π₊`, `f̃₀(q)`, yields), not by raw
state layout identity.
