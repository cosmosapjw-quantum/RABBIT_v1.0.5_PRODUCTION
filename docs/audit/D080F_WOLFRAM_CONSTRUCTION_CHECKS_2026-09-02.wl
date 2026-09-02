(* D-080F production-order construction identities.
   Stateless Wolfram Language source; this checks formula structure only. *)

ClearAll[n, k, c0, c1, tPrep, tMarg, tT, tJ, g, a, b, c, d, r, s];

stateSize[n_] := 3 n + 2;
spectralColumns[n_] := 3 n;
matrixBytes[n_] := 8 stateSize[n]^2;
serialCost[n_, tJ_, tT_] := spectralColumns[n] tJ + tT;
preparedCost[n_, tPrep_, tMarg_, tT_] :=
  tPrep + spectralColumns[n] tMarg + tT;
batchSpeedup[k_, c0_, c1_] :=
  FullSimplify[
    k (c0 + c1)/(c0 + k c1),
    Assumptions -> {k >= 1, c0 > 0, c1 > 0}
  ];

passiveNewtonDeterminantResidual =
  FullSimplify[
    Det[IdentityMatrix[3] - g {{a, b, 0}, {c, d, 0}, {r, s, 0}}]
      - Det[IdentityMatrix[2] - g {{a, b}, {c, d}}]
  ];

<|
  "StateSize60" -> stateSize[60],
  "SpectralColumns60" -> spectralColumns[60],
  "MatrixBytes60" -> matrixBytes[60],
  "SerialProjectionSeconds" ->
    N[serialCost[60, 16.702108392, 13.208925162], 18],
  "PreparedProjectionSeconds" ->
    N[preparedCost[60, 1.81303746, 2.277721835, 13.208925162], 18],
  "BatchSpeedupFormula" -> batchSpeedup[k, c0, c1],
  "PassiveNewtonDeterminantResidual" ->
    passiveNewtonDeterminantResidual
|>
