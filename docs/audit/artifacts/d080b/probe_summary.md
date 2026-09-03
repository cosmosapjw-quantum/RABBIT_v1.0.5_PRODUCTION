# D-080B full static T-gamma collision column

- classification: `FULL_STATIC_TGAMMA_COLLISION_COLUMN`
- comparator Git blob: `de44feee0aa484abe26976c7dc34c579643005b5`
- best thermal action residual: `7.008319500e-07`
- best thermal energy residual: `1.345142201e-08`
- all ladder samples same branch: `True`
- equilibrium dQ_nu/dT_gamma: `8.606919857e-20`
- equilibrium dQ_em/dT_gamma: `-8.606919857e-20`
- maximum differentiated first-law residual: `5.904140785e-17`
- minimum support margin: `2.669708214e-03`
- minimum Kallen margin: `2.598020335e-01`
- base reconstruction residual: `0.000000000e+00`
- component-sum residual: `3.669741774e-15`

This result differentiates the frozen static electron collision action with
respect to `T_gamma`, including moving quadrature, kinematics, weak matrix
elements, Pauli blocking, moving output interpolation, pair annihilation,
flavour routing, and the neutrino/electromagnetic energy ledgers.  It does not
construct the full RHS column or call an integrator.
