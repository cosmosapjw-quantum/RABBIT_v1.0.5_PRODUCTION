(* D-080E fixed-state reuse identities. Stateless Wolfram Language source. *)

ClearAll[
  u1, u2, u3, u4, v1, v2, v3, v4,
  w1, w2, w3, w4, aa, bb, t, ww, mm
];

f[u_] := 1/(1 + Exp[-u]);

gain = (1 - f[u1]) (1 - f[u2]) f[u3] f[u4];
loss = f[u1] f[u2] (1 - f[u3]) (1 - f[u4]);
collision = gain - loss;
vars = {u1, u2, u3, u4};
vv = {v1, v2, v3, v4};
zz = {w1, w2, w3, w4};

j[dir_] := Sum[D[collision, vars[[i]]] dir[[i]], {i, 4}];
shifted[dir_] := collision /. Thread[vars -> vars + t dir];

fixedFactorCacheResidual = FullSimplify[
  (D[ww mm shifted[vv], t] /. t -> 0) - ww mm j[vv]
];

directionLinearityResidual = FullSimplify[
  j[aa vv + bb zz] - aa j[vv] - bb j[zz]
];

columns = j /@ IdentityMatrix[4];
basisColumnAssemblyResidual = FullSimplify[j[vv] - vv.columns];

affinity = u3 + u4 - u1 - u2;
dlogloss =
  (1 - f[u1]) v1 + (1 - f[u2]) v2 - f[u3] v3 - f[u4] v4;
daffinity = v3 + v4 - v1 - v2;
stablePauliJvpResidual = FullSimplify[
  j[vv] - (collision dlogloss + gain daffinity)
];

<|
  "fixed_factor_cache_residual" -> fixedFactorCacheResidual,
  "direction_linearity_residual" -> directionLinearityResidual,
  "basis_column_assembly_residual" -> basisColumnAssemblyResidual,
  "stable_pauli_jvp_residual" -> stablePauliJvpResidual
|>
