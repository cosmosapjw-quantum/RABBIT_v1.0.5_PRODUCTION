# BD642 D-081R1E0 — retained order-60 packed-RHS Python oracle contract

Status: prospectively frozen before retained packed-RHS output.

## Scope

This node freezes one deterministic Python oracle fixture at the preserved
order-60 stiff-region state. It does not implement or admit a Rust RHS,
Jacobian, JVP, ODE solver, trajectory, endpoint, performance result, `N_eff`,
or scientific-gate movement.

## Frozen authorities

```text
D-081R1D4 final head:
002086662bf2e553c78f4b247868cb1fd9e43f21

private Python comparator Git blob:
de44feee0aa484abe26976c7dc34c579643005b5

packed-RHS trajectory-core Git blob:
465a73f0ce40f7149bebdc2d67103f388e2344d9

retained-state source commit:
78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b

retained-state path:
.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_1200.npz

retained-state SHA-256:
c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380
```

## Frozen state and discretisation

```text
order                         60
y_max                         30
packed state size             182
state ordering                (c_e[0:60], c_mu[0:60], c_tau[0:60], T_gamma, t)
N                             0.16286930247517223
T_start                       10 MeV
T_cm(N)                       10 exp(-N) MeV
incoming polar order          4
final polar order             4
final azimuth order           4
electron radial order         24
matrix-roundoff policy        comparator default
```

The NPZ member `y` is the packed state authority. The fixture generator must
reject any other key shape, nonfinite value, source SHA, or source-object
identity.

## Packed RHS authority

The frozen trajectory core defines

\[
Y=(c_e,c_\mu,c_\tau,T_\gamma,t),
\qquad
T_{\rm cm}=10e^{-N}.
\]

Let `total` be the six-species collision action in the ordering

\[
(\nu_e,\bar\nu_e,\nu_\mu,\bar\nu_\mu,\nu_\tau,\bar\nu_\tau).
\]

The three pair rates are

\[
P_\alpha=\frac12\left(C_{\nu_\alpha}+C_{\bar\nu_\alpha}\right).
\]

For the complementary-log-log coordinate

\[
f=1-e^{-e^c},
\qquad
q(c)=\frac{df}{dc}=e^{c-e^c},
\]

the packed RHS is

\[
\frac{dc_\alpha}{dN}=\frac{P_\alpha}{Hq(c_\alpha)},
\]

\[
\frac{dT_\gamma}{dN}
=
\frac{-3(\rho_{\rm em}+p_{\rm em})+Q_{\rm em}/H}
{\partial\rho_{\rm em}/\partial T_\gamma},
\]

\[
\frac{dt}{dN}=\frac1H.
\]

The generator must evaluate the unchanged `_trajectory_core.make_rhs` path
and independently reconstruct the same expression. Bitwise equality of those
two Python paths is required before the fixture is written.

## Required fixture contents

The deterministic JSON fixture must preserve binary64 bit patterns for:

- the complete packed state and its three spectral blocks;
- grid nodes and weights;
- occupations and cloglog chain factors;
- self, electron, and total collision actions in native and modal form;
- the three pair rates;
- the full 182-component packed RHS;
- electromagnetic EOS and full thermodynamics;
- self, electron, and total action moments;
- energy-transfer, entropy, symmetry, support, and matrix-correction diagnostics.

It must record all frozen source identities, the exact Python/NumPy/SciPy
environment, and a claim ceiling. No timestamp or host-dependent field is
allowed in the canonical fixture.

## Physics and implementation gates

1. all 182 state entries and RHS entries are finite;
2. every decoded occupation lies strictly in `(0,1)`;
3. every cloglog chain factor is finite and positive;
4. `total == self + electron` and the modal analogue hold in the frozen
   Python evaluation graph;
5. pair rates equal the exact particle/antiparticle half-sums;
6. trajectory-core RHS and independently reconstructed RHS are bitwise equal;
7. the differentiated first-law residual is within `5e-13`;
8. thermodynamic densities, pressure, entropy, and Hubble rate are positive;
9. support and matrix-correction counters are nonnegative and finite;
10. two independent fixture generations are byte-identical.

## Claim ceiling

`FROZEN_RETAINED_ORDER60_PYTHON_PACKED_RHS_ORACLE_ONLY`.

No Rust parity, derivative, solver, trajectory, endpoint, performance, release,
publication, or F10 gate claim follows from this node.
