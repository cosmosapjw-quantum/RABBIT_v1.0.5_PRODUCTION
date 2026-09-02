(* D-080F cancellation-aware action metrology.
   Stateless Wolfram Language source; formula structure only. *)

ClearAll[a1, a2, a3, v1, v2, v3, eta, delta];

contribution = Abs[a1 v1] + Abs[a2 v2] + Abs[a3 v3];

<|
  "TriangleDominance" ->
    FullSimplify[
      Abs[a1 v1 + a2 v2 + a3 v3] <= contribution,
      Assumptions -> Element[{a1, a2, a3, v1, v2, v3}, Reals]
    ],
  "BasisScale" ->
    FullSimplify[
      contribution /. {v1 -> 0, v2 -> 1, v3 -> 0},
      Assumptions -> Element[{a1, a2, a3}, Reals]
    ],
  "ForwardCancellationResidual" ->
    FullSimplify[eta/Abs[delta], Assumptions -> {eta > 0, delta > 0}],
  "ContributionScaledResidual" ->
    FullSimplify[eta/(2 - delta), Assumptions -> {eta > 0, 0 < delta < 1}]
|>
