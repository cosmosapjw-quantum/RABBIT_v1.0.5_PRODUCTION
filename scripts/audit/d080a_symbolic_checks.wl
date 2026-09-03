(* D-080A symbolic closure checks for the T_gamma tangent. *)

ClearAll[T, x, m, p1, mu, e1, p2, e2, s, m3, m4, lam, k, qWeight];

p2 = T x;
e2 = Sqrt[p2^2 + m^2];
s = m^2 + 2 p1 (e2 - p2 mu);
lam = s^2 + m3^4 + m4^4 - 2 s m3^2 - 2 s m4^2 - 2 m3^2 m4^2;
k = Sqrt[lam]/(2 Sqrt[s]);
qWeight = T;

kinematicChecks = <|
  "dp2Residual" ->
    FullSimplify[D[p2, T] - p2/T,
      Assumptions -> {T > 0, x > 0}],
  "de2Residual" ->
    FullSimplify[D[e2, T] - p2^2/(T e2),
      Assumptions -> {T > 0, x > 0, m >= 0}],
  "dsResidual" ->
    FullSimplify[
      D[s, T] - 2 p1 (p2^2/(T e2) - p2 mu/T),
      Assumptions -> {T > 0, x > 0, m >= 0, p1 > 0, -1 <= mu <= 1}],
  "dlambdaResidual" ->
    FullSimplify[D[lam, T] - 2 (s - m3^2 - m4^2) D[s, T]],
  "dkLogResidual" ->
    FullSimplify[
      D[k, T] - k (D[lam, T]/(2 lam) - D[s, T]/(2 s)),
      Assumptions -> {s > 0, lam > 0}],
  "quadratureWeightResidual" ->
    FullSimplify[D[qWeight, T] - qWeight/T]
|>;

ClearAll[p, ep, f, g, rhoIntegrand, rho2Target];
ep = Sqrt[p^2 + m^2];
f = 1/(Exp[ep/T] + 1);
g = f (1 - f);
rhoIntegrand = p^2 ep f;
rho2Target = p^2 (-2 ep^2 g/T^3 + ep^3 g (1 - 2 f)/T^4);

thermodynamicChecks = <|
  "fdTemperatureResidual" ->
    FullSimplify[D[f, T] - ep g/T^2,
      Assumptions -> {T > 0, p >= 0, m >= 0}],
  "d2RhoIntegrandResidual" ->
    FullSimplify[D[rhoIntegrand, {T, 2}] - rho2Target,
      Assumptions -> {T > 0, p >= 0, m >= 0}]
|>;

ClearAll[rho, pressure, chi, qem, hub];
temperatureRowResidual = FullSimplify[
  D[(-3 (rho[T] + pressure[T]) + qem[T]/hub[T])/chi[T], T] -
    ((-3 (chi[T] + (rho[T] + pressure[T])/T) +
        qem'[T]/hub[T] - (qem[T]/hub[T]) hub'[T]/hub[T])/chi[T] -
      ((-3 (rho[T] + pressure[T]) + qem[T]/hub[T])/chi[T])
       chi'[T]/chi[T]),
  Assumptions -> {
    T > 0, hub[T] > 0, chi[T] > 0,
    rho'[T] == chi[T],
    pressure'[T] == (rho[T] + pressure[T])/T
  }
];

<|
  "kinematics" -> kinematicChecks,
  "thermodynamics" -> thermodynamicChecks,
  "temperatureRowResidual" -> temperatureRowResidual
|>
