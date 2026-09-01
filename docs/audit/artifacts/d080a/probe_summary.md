# D-080A T-gamma kinematic tangent probe

- classification: `TGAMMA_KINEMATIC_TANGENT_ONLY`
- comparator Git blob: `de44feee0aa484abe26976c7dc34c579643005b5`
- best kinematic residual: `3.682855771e-11`
- best EOS residual: `1.535891263e-09`
- all centered samples same branch: `True`
- support margin: `2.675897764e-03`
- Kallen margin: `8.332592639e-01`
- freeze-p2 mutation: `1.000000000e+00`
- flip-e2-sign mutation: `2.000000000e+00`
- omit-weight-scale mutation: `1.000000000e+00`
- D-080 records/projection synchronization parent: `4e2604cae418b8834fb29e2d4227deb6f8cf5c0b`

This probe certifies only the smooth incoming-electron quadrature, elastic
kinematics, mapped output coordinates, matrix-element dot products, and QED-off
EOS temperature tangents. It does not assemble the collision or full RHS
`T_gamma` column.
