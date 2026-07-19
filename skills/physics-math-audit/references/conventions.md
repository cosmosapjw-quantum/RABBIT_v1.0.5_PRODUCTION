# Project Convention Reference

Edit this file per project.

## Metric

Default: `(-,+,+,+)`.

## Units

Default: keep `c`, `hbar`, `k_B`, and `G` explicit unless natural units are declared.

## Common checks

### Dimension check

For every equation:
- left-hand side dimension;
- right-hand side dimension;
- hidden convention where a constant was set to 1;
- scale variable dimension.

### Sign check

For GR/cosmology:
- metric signature;
- Riemann convention;
- Einstein equation sign;
- stress-energy tensor convention;
- expansion/shear/vorticity signs;
- Fourier transform convention.

### Limit check

Prefer at least:
- isotropic/FLRW limit;
- nonrelativistic or ultrarelativistic limit;
- weak-field or small-perturbation limit;
- zero-coupling limit;
- high/low temperature limit where applicable.
