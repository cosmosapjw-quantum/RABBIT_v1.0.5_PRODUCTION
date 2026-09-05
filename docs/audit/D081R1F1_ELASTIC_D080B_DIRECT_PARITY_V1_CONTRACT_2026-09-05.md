# Elastic D-080B direct prefactor parity V1: pre-execution contract

Base: `f3be9c62203e9c380c348ad2589ee742d7cc023a`, tree
`cb92cf283db55817919c5aee51466ab1a9fc3bf1`. The verified zero-raw kink
repair and all previous STOP/PASS receipts remain unchanged.

Two separate comparisons use `(p1, T_gamma) = (2.0, 2.05), (0.75, 0.4)` MeV,
the actual electron mass `0.51099895` MeV, outer weight 1 and the frozen
default collision config: radial 48, incoming polar 12, final polar 12,
azimuth 4, matrix roundoff 1024 ULP. No quadrature or physical constants change.

1. Same-input: frozen D-080A binary64 base/tangent arrays enter the current
   Rust scalar helpers directly.
2. End-to-end: current Rust P0B base/tangent arrays enter those helpers.

Both compare W, W_T, M and signed M_T against direct calls to frozen D-080B
`_measure_and_tangent` and `_elastic_matrix_value_and_tangent`, for all 12
elastic routes. `_electron_matrix_raw` supplies reference raw and scale;
no Rust formula is translated into the oracle. The Rust addition is a
test-only array bridge; the Python comparator applies the numerical gates.

The prefactor domain is exactly `base.support`, with no spectral/action
projection cut. Every reduced measure array carries its original C-order
indices. Rust full arrays are gathered using these indices. Shape, support,
tangent support, applied domain, domain indices and correction masks must
agree exactly. Matrix parity uses only the common supported, uncorrected,
non-kink branch. Measure parity uses its own supported domain. Unsupported
and corrected matrix zeros, and off-domain measure zeros, are checked
separately; zero observed samples are reported as absent coverage, with
the unchanged synthetic boundary suite retaining its separate role.

Kink samples are never silently dropped: preserve full input arrays,
indices, masks and typed Rust status. Python's uncorrected raw=0 with
nonzero M_T is classified separately from ordinary parity. Rust refusal
has a null M_T and `NondifferentiableDiscreteEvent` status. Masks must agree.

Before any output, freeze the existing P1 direct-parity cap **1e-7** for
each case/route/quantity/path: max absolute difference divided by the
maximum absolute value of either compared array, floored at binary64
MIN_POSITIVE. W_T and M_T use their own derivative arrays for this scale.
Record raw maximum absolute difference, global scale and residual,
worst local-relative residual and original index, plus worst absolute
index and exact value bits. Local-relative residual is diagnostic only;
there is no new local threshold. Existing 128-epsilon scalar and centered
caps remain in the unchanged candidate tests.

Authority: P1 contract blob `ff568c9193de5e9f2fecb07c1694df3bf4ea5549`,
D-080A `c585d5865fd68a90a04a76ab540b8437fba8cfce`, D-080B
`78489c43f3046db09d8ba2d96070124ed7b0aa91`, frozen comparator
`de44feee0aa484abe26976c7dc34c579643005b5`.

Run the two array-export tests and the 192 numerical comparisons before
the unchanged 4 candidate plus 29 inherited tests, release, fmt and strict
Clippy, in the existing production capsule. Preserve arrays/metrics even
on failure. No cap widening or production repair is part of this task.
The existing 62-identity SymPy replay is not repeated or relabelled.

Claim ceiling: direct measure/matrix prefactor parity only at these two
physical batches and 12 routes. No full collision-action parity, routing
unification, measure component interface, production mutation battery,
full elastic/P1/thermal RHS/retained/solver/F10 PASS.
