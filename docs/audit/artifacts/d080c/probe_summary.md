# D-080C deterministic static probe

- classification: `FULL_STATIC_TGAMMA_RHS_COLUMN`
- comparator blob: `de44feee0aa484abe26976c7dc34c579643005b5`
- thermal best block residual: `1.7305993361739309e-07`
- weak-tail best block residual: `2.4549561367864960e-08`
- thermal same branch: `True`
- weak-tail same branch: `True`
- equilibrium dQ_nu/dT_gamma: `8.6069198569433735e-20`
- equilibrium dQ_em/dT_gamma: `-8.6069198569433735e-20`
- elapsed-time input column norm: `0.0000000000000000e+00`

The residual metric is blockwise and dimension-aware.  It does not combine the
MeV^-1 spectral rows, dimensionless photon-temperature row, and MeV^-2 elapsed
output row in one dimensional Euclidean norm.

The manufactured weak-tail state is a controlled static probe, not retained
trajectory evidence.  The claim remains limited to the fixed-support static
original-RHS input column.
