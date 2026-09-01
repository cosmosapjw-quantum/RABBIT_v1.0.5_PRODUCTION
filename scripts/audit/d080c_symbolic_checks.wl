(* D-080C exact symbolic closure checks.
   Stateless Wolfram Language source; the retained JSON receipt records the
   plugin-evaluated output and does not claim a repository-native replay. *)

ClearAll[T, tau, H, rho, p, chi, action, qem, rhoNu, chain, a0];

rhoTotal[T_] := rhoNu + rho[T];
hubbleLogTangent[T_] := rho'[T]/(2 rhoTotal[T]);

spectralRhs[T_] := action[T]/(H[T] chain);
temperatureNumerator[T_] := -3 (rho[T] + p[T]) + qem[T]/H[T];
temperatureRhs[T_] := temperatureNumerator[T]/chi[T];
timeRhs[T_] := 1/H[T];

hubbleRule = H'[T] -> H[T] hubbleLogTangent[T];

spectralExpected =
  action'[T]/(H[T] chain)
    - spectralRhs[T] hubbleLogTangent[T];

temperatureNumeratorExpected =
  -3 (rho'[T] + p'[T])
    + qem'[T]/H[T]
    - qem[T] hubbleLogTangent[T]/H[T];

temperatureExpected =
  temperatureNumeratorExpected/chi[T]
    - temperatureRhs[T] chi'[T]/chi[T];

timeExpected = -timeRhs[T] hubbleLogTangent[T];

rhsQuotientResiduals =
  FullSimplify[
    {
      D[spectralRhs[T], T] - spectralExpected,
      D[temperatureNumerator[T], T] - temperatureNumeratorExpected,
      D[temperatureRhs[T], T] - temperatureExpected,
      D[timeRhs[T], T] - timeExpected
    } /. hubbleRule,
    Assumptions -> {
      T > 0,
      H[T] > 0,
      chi[T] > 0,
      rhoTotal[T] > 0,
      chain > 0
    }
  ];

elapsedTimeInputColumn =
  D[{spectralRhs[T], temperatureRhs[T], timeRhs[T]}, tau];

differentiatedFirstLawResidual =
  FullSimplify[dQnu + dQem /. dQnu -> -dQem];

restoringSignGate =
  FullSimplify[dQnu > 0 /. dQnu -> -dQem, dQem < 0];

masslessEosSecondDerivativeResidual =
  FullSimplify[D[a0 T^4, {T, 2}] - 12 a0 T^2];

<|
  "rhsQuotientResiduals" -> rhsQuotientResiduals,
  "elapsedTimeInputColumn" -> elapsedTimeInputColumn,
  "differentiatedFirstLawResidual" -> differentiatedFirstLawResidual,
  "restoringSignFollowsFromQemTNegative" -> restoringSignGate,
  "masslessEosSecondDerivativeResidual" ->
    masslessEosSecondDerivativeResidual
|>
