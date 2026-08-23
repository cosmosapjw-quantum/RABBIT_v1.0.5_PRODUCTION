# SCIENTIFIC_CONTRACT.md

## Scientific objective

[이 코드가 계산·검증해야 하는 물리적 또는 수학적 양]

## Governing definitions

[핵심 변수, 함수, tensor, operator, observable]

## Conventions

- metric signature:
- Fourier convention:
- index convention:
- unit system:
- normalization:
- stochastic convention:
- coordinate/gauge convention:

## Valid regime

- parameter range:
- resolution range:
- asymptotic assumptions:
- perturbative order:
- excluded singular regimes:

## Required invariants

- dimensions:
- conservation laws:
- symmetry:
- positivity:
- normalization:
- exact identities:

## Known limits

| Limit | Expected result | Tolerance | Reference/test |
|---|---|---:|---|
|  |  |  |  |

## Reference cases

- analytic toy case:
- trusted numerical reference:
- previous implementation:
- published benchmark:

## Numerical requirements

- target precision:
- convergence order:
- stability expectations:
- random seed policy:
- ensemble size:
- acceptable runtime/memory:

## Failure semantics

다음 상태를 성공처럼 처리하지 않는다: NaN/Inf, non-convergence, empty result, clipped invalid values, silently replaced missing data, fallback approximation outside its regime.

## Change control

convention, baseline, tolerance, approximation order, output semantics 변경은 승인 필요.
