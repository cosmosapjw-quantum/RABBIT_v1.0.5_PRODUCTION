(* D-080D symbolic closure checks.
   This source is evaluated statelessly through the Wolfram Language plugin.
   It verifies only finite-dimensional block identities; it does not execute
   repository code or any ODE/nonlinear solver. *)

ClearAll[aa, bb, cc, dd, pp, qq, xx, yy, zz, gam, lam,
  jacMat, activeMat, vec];

(* A two-dimensional active block plus one passive elapsed-time accumulator is
   sufficient to prove the generic block identities. *)
jacMat = {{aa, bb, 0}, {cc, dd, 0}, {pp, qq, 0}};
activeMat = {{aa, bb}, {cc, dd}};
vec = {xx, yy, zz};

columnAssemblyResidual = Expand[
  jacMat.vec - Sum[vec[[i]] jacMat[[All, i]], {i, 3}]
];

elapsedColumnResidual = Expand[jacMat.{0, 0, 1}];

newtonDeterminantResidual = Expand[
  Det[IdentityMatrix[3] - gam jacMat]
    - Det[IdentityMatrix[2] - gam activeMat]
];

(* CharacteristicPolynomial uses det(M-lambda I), hence the plus sign below
   when factoring the passive zero eigenvalue. *)
characteristicFactorResidual = Expand[
  CharacteristicPolynomial[jacMat, lam]
    + lam CharacteristicPolynomial[activeMat, lam]
];

activeActionResidual = Expand[
  Most[jacMat.vec] - activeMat.Most[vec]
];

<|
  "ColumnAssemblyResidual" -> columnAssemblyResidual,
  "ElapsedColumnResidual" -> elapsedColumnResidual,
  "NewtonDeterminantResidual" -> newtonDeterminantResidual,
  "CharacteristicFactorResidual" -> characteristicFactorResidual,
  "ActiveActionResidual" -> activeActionResidual
|>
