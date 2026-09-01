(* D-080B exact symbolic closure checks.
   Stateless Wolfram Language source; the retained JSON receipt records the
   plugin-evaluated output and does not claim a repository-native replay. *)

ClearAll[T, w, m, c, E1, E2, E3, E4, dE2, dE3, dE4, G];

productRuleResidual =
  Expand[
    D[w[T] m[T] c[T], T]
      - (w'[T] m[T] c[T]
        + w[T] m'[T] c[T]
        + w[T] m[T] c'[T])
  ];

electronLogitResidual =
  FullSimplify[
    D[-E2[T]/T, T] - (-E2'[T]/T + E2[T]/T^2),
    Assumptions -> T > 0
  ];

(* Elastic event: legs 1,3 are neutrinos and 2,4 are electrons.
   T_cm and the incoming-neutrino chart are fixed. *)
elasticAffinityDerivative =
  -dE3/T + (-dE4/T + E4/T^2)
    - 0 - (-dE2/T + E2/T^2);

elasticAffinityReduced =
  FullSimplify[
    elasticAffinityDerivative /. dE2 -> dE3 + dE4
  ];

elasticPositiveEnergyResidual =
  FullSimplify[
    elasticAffinityReduced (E1 - E3)
      - (E4 - E2)^2/T^2,
    Assumptions -> {T > 0, E1 + E2 == E3 + E4}
  ];

(* Pair event: legs 1,2 are neutrinos and 3,4 are electrons; its
   kinematics are independent of T_gamma in the frozen comparator. *)
pairAffinityDerivative = E3/T^2 + E4/T^2;
pairPositiveEnergyResidual =
  FullSimplify[
    pairAffinityDerivative (E1 + E2)
      - (E3 + E4)^2/T^2,
    Assumptions -> {T > 0, E1 + E2 == E3 + E4}
  ];

(* Differentiated event first law for the elastic energy weights. *)
differentiatedFirstLawResidual =
  FullSimplify[
    c ((E1 - E3) + (E2 - E4))
      + c ((-dE3) + (dE2 - dE4))
      /. {E4 -> E1 + E2 - E3, dE4 -> dE2 - dE3}
  ];

<|
  "eventProductRuleResidual" -> productRuleResidual,
  "electronLogitResidual" -> electronLogitResidual,
  "elasticAffinityDerivative" -> elasticAffinityReduced,
  "elasticPositiveEnergyResidual" -> elasticPositiveEnergyResidual,
  "pairPositiveEnergyResidual" -> pairPositiveEnergyResidual,
  "differentiatedFirstLawResidual" -> differentiatedFirstLawResidual
|>
