# Rust ODE Collision Reconstruction — P1 Evidence Note

Date: 2026-08-24  
Branch: `codex/ode-p0-rust-reconstruction-20260824`  
Base commit: `78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`  
Claim ceiling: focused direct-comoving electron-collision reconstruction only; no endpoint or solver promotion

## Outcome

The P0 discriminator selected `COLLISION_DISCRETIZATION_OR_ASSEMBLY_REPAIR`,
not a Radau-first repair. The Python thermal-to-comoving map does not preserve
node-local Pauli inwardness at the moving high-q boundary. P1 therefore leaves
the Python production driver unchanged and reconstructs the collision path on
the existing Rust direct-comoving finite-electron-mass event stream.

The following results are **IMPLEMENTED** and were executed:

- division-free event production/destruction coefficients;
- conservative folded elastic and pair edges whose action reconstructs the
  existing Rust conservative collision action;
- a raw-occupation, Pauli-capacity-bracketed backward-Euler edge extent;
- transactional forward/reverse edge sweeps;
- a collision-only `delta_ln_a / H` substep with electromagnetic temperature
  backreaction fixed by total-energy closure;
- deterministic scoped-thread event construction with original event order;
- focused raw-tail, boundary, equilibrium, conservation, rollback, and timing
  tests.

The following remain **PROPOSED**, not validated or promoted:

- composition with expansion and neutrino self-collision operators;
- an adaptive step-doubling controller and accepted/rejected prefix history;
- parity against the retained Python trajectory, whose simplified collision
  physics is not identical to the Rust finite-mass event action;
- collision-on endpoint authority and replacement of SciPy/BDF as the
  number-of-record.

## P0 binding and corrected signed impact

Retained exact-HEAD raw trajectory:

```text
/tmp/rabbit_raw_trajectory_exact_head_collision_on.json
sha256 2da11255f25761e4b7e7ea330eda2c4948013e4d1df00c5e127720a90f9a678e
```

Corrected discriminator:

```text
/tmp/rabbit_ode_p0_discriminator.py
sha256 a5c1a41189bf069a84fa5c8953ed7571200f3c657f35f6766db25e394ad4f1f7

/tmp/rabbit_ode_p0_result_signed_impact.json
sha256 f1c50509264bc986985cad2f13e9b8605b2cb8d79e951582f4743fec3bc6fd41
```

The exact measured environment was Python 3.12, NumPy 2.4.4, and SciPy
1.17.1. The short prefix reproduced the retained first invalid entry exactly:

```text
sample index       3
N                  -2.302557321652884
bank/node          nue[22]
raw f              -2.3008216172341183e-30
previous N         -2.302570546101056
delta N            1.3224448172088898e-05
nfev/njev/nlu      124 / 3 / 32
stored points      9
```

At `nue[22]`, the local frozen gain-loss update was physical, but node-local
reconstruction of the mapped field failed. At `nue[23]` and `nux[23]`, the
inferred loss coefficient was negative and the upper boundary field pointed
outward. This is the signature of the two interpolation operators, not proof
that the continuum collision operator is nonphysical.

The first P0 report subtracted O(1) moments and lost the signed tail below
floating resolution. The corrected run evaluates the signed delta directly:

```text
nue number moment delta       -1.1152393290390595e-25
nue energy moment delta       -7.802464163482016e-24
N_eff diagnostic delta        -5.290447599947546e-24
weak lambda_np delta           -1.1755851215602610e-18 s^-1
weak lambda_pn delta           -1.1521718908759760e-18 s^-1
```

The corresponding total-minus-total readouts were all `0.0`; they are not
authoritative for this tail-scale question.

## Physics and numerical construction

For an elastic edge `j -> i`, the paired directed quadratures give

```text
A_ij = 1/2 (G_forward_coefficient + L_reverse_coefficient) >= 0
B_ij = 1/2 (L_forward_coefficient + G_reverse_coefficient) >= 0
J_ij = A_ij (1-f_i) f_j - B_ij f_i (1-f_j).
```

The weighted update is `m_i df_i/dt = J_ij` and
`m_j df_j/dt = -J_ij`, so `m_i f_i + m_j f_j` is an algebraic identity.
For pair creation/annihilation, the two orientations instead give

```text
J_ij = A_ij (1-f_i)(1-f_j) - B_ij f_i f_j,
```

and both occupations move with the same weighted extent, preserving the
neutrino-antineutrino difference before folding.

For one edge, write both occupations as affine functions of an extent `xi`.
The accepted interval is the exact intersection of donor, receiver, and hole
capacities. The scalar backward-Euler equation

```text
xi = delta_t J(f_i(xi), f_j(xi))
```

has a monotone residual because `dJ/dxi <= 0`. A safeguarded Newton solve is
therefore bracketed by physical capacities for every finite non-negative
step. No occupation is clipped, projected, or floored. Near detailed balance,
the net uses a logarithmic affinity and `expm1` instead of subtracting two
large positive rates.

The collision-only FLRW candidate freezes `delta_t = delta_ln_a/H` for one
substep. After the edge sweep it solves

```text
rho_em(T_gamma,new) = rho_em(T_gamma,old) - (rho_nu,new - rho_nu,old)
```

with the existing electromagnetic EOS. The state is committed only after all
occupations are strict, finite logit-representable values and the EOS inversion
succeeds.

## Executed focused run data

Toolchain and host:

```text
rustc 1.94.1 (e408947bf 2026-03-25), LLVM 21.1.8
cargo 1.94.1 (29ea6fb6a 2026-03-24)
x86_64, AMD Ryzen 9 5900X, 12 cores / 24 threads
```

Focused kernel/action run:

```text
cargo test --lib pauli -- --nocapture
9 passed; 0 failed; 1 ignored; 240 filtered out

P1_TAIL raw_before_e=1.00000000000000001e-35
        raw_after_e=2.81654998151922333e-4
        raw_before_x=9.99999999999999929e-41
        raw_after_x=2.83609365323408123e-4
        edge_apps=60 nonlinear_iters=526 max_edge_iters=32
```

Event-coefficient factorization:

```text
cargo test --lib dynamic_coefficients_reconstruct_positive_event_gain_and_loss -- --nocapture
1 passed; 0 failed; 248 filtered out
```

Executed solver substep:

```text
cargo test --lib reconstructed_electron_substep_is_raw_bounded_transactional_and_energy_closed -- --nocapture
1 passed; 0 failed; 248 filtered out

delta_N                  1.00000000000000008e-5
frozen dt [MeV^-1]       1.94100387937646160e16
electron tail before     4.02716779214063304e-36
electron tail after      1.98938113478435546e-9
heavy tail before        1.10893901931213646e-40
heavy tail after         4.22668848891454724e-10
T_gamma before [MeV]     1.14999999999999991
T_gamma after [MeV]      1.14999974123257909
rho_nu before [MeV^4]    1.58767666882388436
rho_nu after [MeV^4]     1.58767950410753689
total energy residual    0.0 MeV^4
edge applications        60
nonlinear iterations     372
maximum edge iterations  27
```

Release segment benchmark, 24 momentum nodes, radial order 6, angular order
4, 79,804 retained events, minimum of three samples:

```text
cargo test --release --lib pauli_event_parallel_benchmark -- --ignored --nocapture

workers  parallel [s]  speedup  action
2        0.020620398    1.7433   bitwise equal
4        0.012570839    2.8596   bitwise equal
6        0.009897871    3.6319   bitwise equal
8        0.009060894    3.9673   bitwise equal
12       0.008722009    4.1215   bitwise equal
24       0.007573928    4.7462   bitwise equal
serial   0.035947701    1.0000   reference
```

This is a collision-event segment benchmark, not endpoint progress.

Native AOT SIMD probe:

```text
RUSTFLAGS='-C target-cpu=native' ... pauli_event_parallel_benchmark
enabled AVX2, FMA, SSE4, BMI, and related host features
serial best 0.035692782 s
24-worker best 0.008619585 s
```

The native serial difference versus the portable build was below one percent,
and the native parallel sample was slower than the portable parallel sample.
Explicit SIMD/`target-cpu=native` settings were therefore not retained. The
measured deterministic thread decomposition was retained.

## Validation boundary

No Python default/full suite, Rust full suite, gold test, endpoint, package,
JAX, or Diffrax command was run. The final focused rerun and formatting check
are recorded in the session handoff; this note must not be read as endpoint or
scientific promotion authority.

## Cost line

```text
added_lines: 1710
deleted_lines: 49
net_lines: 1661
files_touched: 8
token_use_exact: UNAVAILABLE
token_use_basis: the active harness exposes no exact token counter
runtime_behavior_changed: yes
physics_behavior_changed: yes, in the new reconstruction candidate only
known_blocker_reduced: yes
blocker_movement_ratio: 0.50
validation_strengthened: yes
cost_effectiveness_verdict: ACCEPT_WITH_LIMITS
```

The line count exceeds the preferred PR budget. It moves a hard physics and
solver blocker, but the work should be split before promotion into event
factorization, collision-step reconstruction, and measured parallel assembly.
No superseded Python/JAX implementation was deleted in this bounded research
unit, so cumulative migration duplication remains open.

## Remaining blockers

1. Add a positivity-preserving neutrino self-collision step or a compatible
   split operator without tail freezing.
2. Compose expansion, electron collision, and self-collision with step
   doubling and a physics-scaled local error norm.
3. Record accepted/rejected raw candidates and demonstrate convergence over a
   short collision-on prefix at two step ladders.
4. Bind a same-physics reference before claiming parity with SciPy.
5. Only then run endpoint and cold-wall authority gates.
