# RABBIT F10 solver/algorithm/programming blocker research loop

**Date:** 2026-08-06  
**Workspace branch:** `research/solver-algorithm-loop`  
**Upstream seed:** `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`, branch `f10-independent-validation-b3v2`, seed commit `2d866dfd3b0a09b52f4ef5d4191b7d713e084ed9`  
**Research status:** `REOPEN_VALIDATION`  
**Gate status:** `D-071 REOPEN NOT EARNED`  
**Formal survivor:** `EC-EXPRB-K`  
**Secondary post-formal survivor:** `EC-PICARD-IE`  

## 0. Executive result

이번 루프의 목표는 order-60, `y_max=30` independent no-QKE trajectory가 현재 SciPy BDF instrument에서 사실상 정지하는 blocker를 **solver·algorithm·programming 층에서** 해소할 후보를 찾는 것이었다. 기존 코드를 단순히 빠르게 만드는 경로, 더 큰 하드웨어, Jacobian reset/cap, supplied analytic Jacobian만으로 끝나는 경로는 D-071의 reopening 조건을 만족하지 못하므로 진단 control로만 남겼다.

연구 루프는 17개 알고리즘 계보를 발산적으로 검토하고, 그중 세 계보를 prospectively sealed formal candidate로 구현했다.

1. `EC-D-JFNK`: 보존좌표 + direct Jacobian-free Newton–Krylov
2. `EC-EXPRB-K`: 보존좌표 + nonautonomous exponential Rosenbrock–Krylov
3. `EC-MLPC-JFNK`: 보존좌표 + 48/24→60/30 two-level nonlinear preconditioned JFNK

formal holdout에서 **`EC-EXPRB-K`만 모든 frozen gate를 통과했다.** order-60/30, 182-state synthetic discriminator에서 noiseless endpoint WRMS는 `9.0768543263e-6`, cancellation-oracle WRMS는 `9.1592110837e-6`, energy residual은 정확히 0, full-RHS-equivalent calls는 352, projected wall은 safety factor 2.5를 포함해 3,960 s였다. 동일 환경의 repeat endpoint와 counters는 bitwise identical이었다.

formal 이후 별도 protocol을 먼저 봉인하고 derivative-free `EC-PICARD-IE`를 구현했다. 이 후보도 frozen gates를 통과했지만 endpoint WRMS `1.8170e-5`, 4,001 calls, projected wall 45,011 s로 margin이 얇다. 따라서 **primary physical candidate는 EC-EXPRB-K**, Picard는 **local contraction이 실제 creep-state에서 증명될 경우에만 사용할 fallback**으로 판정한다.

가장 중요한 제한은 다음이다.

> 생존한 것은 EXPRB-K 단독이 아니라 **energy-constrained EC-EXPRB-K**다.

보존 projection을 제거한 raw EXPRB-K는 energy identity를 noiseless에서 `1.17e-6`, cancellation oracle에서 `6.79e-6`만큼 위반하여 frozen energy gate를 통과하지 못했다. 실제 RABBIT에서는 synthetic exact-trace projection을 쓸 수 없으므로, 이전 math/physics loop에서 유도한 **total comoving energy state와 algebraic `T_gamma` recovery**를 solver backend와 함께 구현해야 한다.

이 결과는 physical RABBIT trajectory를 실행한 것이 아니다. 현재 판정은 다음 세 층을 엄격히 구분한다.

| 층 | 판정 |
|---|---|
| standalone manufactured/cancellation benchmark | `EC-EXPRB-K SURVIVE`, `EC-PICARD-IE SURVIVE` |
| 실제 RABBIT creep-prefix 후보 | `PROPOSED; physical implementation and prefix discriminator required` |
| D-071 gate | `FAIL remains; reopen not earned` |

## 1. Seed evidence와 blocker의 정확한 형태

### 1.1 현재 instrument의 실패

D-071은 60/30 domain holdout이 18시간 wall budget에서 `N≈0.1653`까지만 진행했고, retained post-drop creep를 그대로 외삽하면 약 4.58년, 약 `3.27e7` additional RHS evaluations가 필요하다고 판정했다. 따라서 단순 10–100배 hardware gain이나 per-call vectorization만으로는 evaluation-count miss를 닫을 수 없다.

V3는 이 실패의 기계적 연결고리를 다음처럼 닫았다.

- SciPy `num_jac`의 per-component finite-difference factor가 Jacobian refresh마다 대략 10배씩 ratchet한다.
- creep states에서 ratcheted columns는 pinned-step columns 대비 median 약 0.95–1.64% 오염되고, worst column은 234배까지 틀렸다.
- 같은 states의 true spectral operator는 spectral abscissa 약 `-5.75`이고 positive-real eigenvalue가 없다.
- corrupted fresh Jacobian 뒤 Newton failure가 반복되고 BDF order가 1에 고정된다.
- 건강한 48/24 lane도 ratchet 자체는 겪고 완주하므로, ratchet은 필요조건에 가깝지만 단독 discriminator는 아니다.

따라서 연구 문제는 “BDF를 조금 고친다”가 아니라 다음으로 재정의됐다.

\[
\text{find a method that avoids persistent dense-FD lifetime,}
\]

\[
\text{uses a bounded number of black-box full collision evaluations,}
\]

\[
\text{and preserves the physical energy ledger without fitting after output.}
\]

### 1.2 D-071이 허용하지 않는 shortcut

다음은 유용한 control이지만 reopening route로는 탈락시켰다.

- `jac_factor` reset/cap
- supplied full analytic Jacobian
- faster CPU/GPU, more cores, larger wall budget
- dense finite-difference vectorization만 적용
- 기존 60/30 BDF instrument의 tolerance 조정
- 결과를 본 뒤 JVP epsilon, step, Krylov dimension을 맞추는 것

## 2. 연구 하네스와 claim discipline

코딩 하네스의 단계에 맞춰 다음 durable artifacts를 먼저 만들었다.

- `SCIENTIFIC_CONTRACT.md`
- `SCIENTIFIC_CONTRACT_AMENDMENT_01.md`
- `FORMAL_SWEEP_PROTOCOL.md`
- `FORMAL_SWEEP_ADJUDICATION_01.md`
- `POSTFORMAL_PROTOCOL_01.md`
- TDD unit tests
- v1/v2 formal evidence, post-formal evidence, endpoint arrays

formal protocol은 결과 전에 다음을 고정했다.

- primary: `N∈[0,0.8]`, order 60, `y_max=30`, state size 182
- scaling: 48/24와 96/48, `N∈[0,0.4]`
- cancellation scale: `1e10`
- endpoint norm: absolute component WRMS
- call budget: 5,500 full-RHS-equivalent calls
- projected wall: `calls × 4.5 s × 2.5 < 64,800 s`
- noiseless error: `≤2e-5`
- cancellation error: `≤2e-4`
- energy residual: `≤1e-10` / `≤1e-7`
- same-host bitwise determinism
- memory: `<2 GiB`

v1 이후 허용한 v2 수정은 terminal floating-point microstep 제거와 spectral-block diagnostic 수정뿐이었다. candidate status는 하나도 바뀌지 않았다.

## 3. Benchmark architecture

### 3.1 상태와 차원

synthetic state는

\[
y=(u_{e,1\ldots n},u_{\mu,1\ldots n},u_{\tau,1\ldots n},E,\chi),
\qquad \dim y=3n+2
\]

로 정의했다. order 60에서는 `182` components다. `u`는 Fermi-Dirac background에 대한 smooth spectral distortion이고, occupation은

\[
f_{a,i}=\sigma(-q_i+u_{a,i})\in(0,1)
\]

로 복원된다.

시간은 RABBIT의 `N=ln a`를 모사하는 무차원 변수다. synthetic time을 물리적 MeV 시간으로 해석하지 않는다.

### 3.2 Manufactured nonautonomous kinetic equation

spectral block은 stable but nonnormal operator로 만들었다.

\[
\dot u=A(t)u+bE+\eta\tanh u+s(t),
\]

여기서 `A(t)`의 diagonal decay는 약 `-5.75` 이하이고 upper-triangular coupling이 nonnormality를 만든다. forcing `s(t)`를 exact path로부터 역산하여 exact solution을 알고 있다.

energy scalar는

\[
\dot E=S_{\rm tr}(t)-w^T\dot u
\]

를 만족하므로

\[
E(t)+w^Tu(t)=E_0+\int_0^tS_{\rm tr}(s)\,ds
\]

가 exact invariant다.

### 3.3 Cancellation oracle

collision gain/loss cancellation을 모사하기 위해

\[
F_{\rm obs}=(B+\tfrac12F_{\rm true})-(B-\tfrac12F_{\rm true})
\]

를 사용했다. formal baseline은 `B≈1e10`이다. exact mathematical RHS는 변하지 않지만 floating-point directional differences가 손상된다. 이는 실제 RABBIT collision operator와 동일하다고 주장하는 모델이 아니라, V3가 측정한 Jacobian-column corruption을 공격하는 deterministic numerical oracle다.

### 3.4 Full-RHS-equivalent accounting

모든 perturbed RHS, JVP RHS, vectorized finite-difference column은 각각 하나의 full-RHS-equivalent call로 센다. Python에서 한 번 batch 호출했다고 여러 물리 RHS를 한 call로 축소하지 않았다.

## 4. 발산적 후보 계보와 가지치기

| Family | 연구 상태 | 핵심 판정 |
|---|---|---|
| EC-EXPRB-K | `FORMAL_SURVIVOR` | 모든 frozen gate 통과 |
| EC-PICARD-IE | `POSTFORMAL_SURVIVOR` | 통과하지만 accuracy/wall margin 얇음 |
| EC-D-JFNK | `KILL` | cancellation JVP에서 GMRES 첫 step 실패; call/error도 miss |
| EC-MLPC-JFNK | `KILL` | coarse correction이 fine JVP noise를 제거하지 못함 |
| supplied-Jac Radau/BDF | `CONTROL_ONLY` | synthetic에서는 매우 우수하나 D-071 scope 불충족 |
| LSODA | `CONTROL_ONLY` | synthetic control 우수; formal run은 stiff branch 진입조차 안 함 |
| jac-factor reset/cap | `CONTROL_ONLY` | repair이며 harsh cancellation에서 reset 후에도 columns 부정확 |
| full analytic Jacobian | `CONTROL_ONLY` | 물리 derivation 비용이 크고 기존 instrument tuning |
| relaxed/noise-aware JFNK | `KILL` | linear iterations 0으로 predictor를 그냥 받아들이는 가짜 성공 |
| Anderson Picard | `DEFER` | plain Picard가 매 step 2 evaluations; 추가 state의 이득 없음 |
| limited-memory Broyden | `DEFER` | secant history가 cancellation noise를 누적할 위험 |
| RKC/Chebyshev explicit | `PROPOSED` | real-negative spectral block에서 가능성 있으나 아직 미구현 |
| IMEX collision split | `PROPOSED` | physical split과 commutator error contract가 선행돼야 함 |
| recycled Rosenbrock/EXPRB Krylov | `PROPOSED` | m≈4 포화 측정; 별도 sealed contract 필요 |
| FAS/p-multigrid | `REWORK` | iteration은 줄였지만 first-order cost와 fine JVP noise가 지배 |
| SDC/Parareal/MGRIT | `LOW_PRIORITY` | expensive RHS의 sequential stiffness가 주 문제라 우선순위 낮음 |
| GPU/vectorized dense FD | `KILL_AS_REOPEN_ROUTE` | per-call 비용만 줄이고 evaluation miss를 보존 |

## 5. 후보별 알고리즘

### 5.1 EC-D-JFNK

implicit Euler residual

\[
G(z)=z-y_n-hF(t_{n+1},z)=0
\]

을 Newton–GMRES로 푼다. Jacobian-vector product는 매 호출 독립 forward difference

\[
J_G(z)v\approx v-h\frac{F(z+\epsilon v)-F(z)}{\epsilon}
\]

로 계산하며 persistent component factor를 저장하지 않는다.

**장점:** full Jacobian storage가 없고 physics preconditioner를 붙일 수 있다.  
**실패:** cancellation scale `1e10`에서 첫 step의 GMRES가 `info=80`으로 끝났다. Newton tolerance를 완화한 lane은 0 linear iterations로 explicit predictor를 수용했으므로 genuine implicit solution이 아니었다.

### 5.2 EC-EXPRB-K

비자율성을 처리하기 위해 time을 auxiliary component로 붙인다.

\[
Y=\begin{pmatrix}y\\t\end{pmatrix},\qquad
\mathcal F(Y)=\begin{pmatrix}F(t,y)\\1\end{pmatrix}.
\]

각 step에서 start vector `b=(F,1)`에 대해 augmented Jacobian의 Krylov basis

\[
\mathcal K_m(J,b)=\operatorname{span}\{b,Jb,\ldots,J^{m-1}b\}
\]

를 Arnoldi로 만들고

\[
Y_{n+1}=Y_n+h\varphi_1(hJ_n)\mathcal F(Y_n),
\qquad
\varphi_1(z)=\frac{e^z-1}{z}
\]

의 reduced action을 계산한다. full matrix exponential은 `(m+1)×(m+1)` reduced matrix에만 사용한다. 물리 state의 analytic Jacobian이나 dense `num_jac`는 사용하지 않는다.

formal config는 `h=0.025`, `m=10`, fixed augmented directional step `1e-3`이다.

**왜 살아남았는가:**

- nonlinear Newton residual을 고정밀로 0까지 풀 필요가 없다.
- persistent per-component FD lifetime이 없다.
- step당 `1+m` full RHS calls로 비용이 사전 계산 가능하다.
- stable creep operator의 action을 작은 Krylov subspace로 근사한다.
- post-formal sweep에서 `m=4`부터 accuracy가 사실상 포화됐다.

### 5.3 EC-MLPC-JFNK

48/24 coarse trajectory를 60/30 fine state의 predictor와 two-level preconditioner로 사용한다. coarse residual correction과 fine diagonal complement smoother를 합치되 최종적으로는 fine implicit residual을 푼다.

이 구현은 coarse predictor만 답으로 대체하는 surrogate가 아니다. 그러나 formal cancellation lane에서 fine JVP가 먼저 붕괴했고, noiseless에서도 first-order step 때문에 7,701 effective calls와 `3.41e-5` endpoint error가 남았다.

### 5.4 EC-PICARD-IE

post-formal 후보는 derivative-free projected fixed point

\[
z^{(k+1)}=P_E\left[y_n+hF(t_{n+1},z^{(k)})\right]
\]

를 사용한다. accepted iterate는 반드시 그 RHS가 실제로 평가된 마지막 iterate이며, 그 RHS를 다음 predictor에 재사용한다. 따라서 숨은 extra call이 없다.

formal synthetic Jacobian의 spectral radius가 약 7.9이므로 `h=4e-4`에서는 fixed-point map이 강하게 contractive했고, 모든 step이 2 evaluations에 수렴했다.

하지만 physical nonlinear map의 relevant norm과 nonnormal transient를 아직 측정하지 않았다. 따라서 Picard는 실제 creep-state에서

\[
\|hJ_F\|_H<q<1
\]

또는 equivalent contraction certificate가 확보되어야만 실행 후보가 된다.

## 6. Formal 결과

### 6.1 Candidate table

| Candidate | Status | noiseless WRMS | cancellation WRMS | energy residual | calls | projected wall | scaling exponent |
|---|---|---:|---:|---:|---:|---:|---:|
| EC-D-JFNK | KILL | `2.2713e-5` | FAIL | `2.78e-17` | 9,666 | 108,742.5 s | 0.0243 |
| **EC-EXPRB-K** | **SURVIVE** | **`9.0769e-6`** | **`9.1592e-6`** | **0** | **352** | **3,960 s** | **0** |
| EC-MLPC-JFNK | KILL | `3.4061e-5` | FAIL | `2.78e-17` | 7,701.03 | 86,636.6 s | -0.0040 |
| CONTROL-Radau | scope kill | `1.83e-13` | same | `1.77e-14` | 162 | 1,822.5 s | 0 |
| CONTROL-BDF | scope/energy kill | `1.46e-8` | same | `5.91e-10` | 78 | 877.5 s | 0 |
| CONTROL-LSODA | scope kill | `7.89e-11` | same | `1.21e-11` | 67 | 753.75 s | 0 |
| **EC-PICARD-IE** | **post-formal SURVIVE** | **`1.8170e-5`** | **`1.8171e-5`** | **`≤2.78e-17`** | **4,001** | **45,011.25 s** | **-0.1155** |

`projected wall = calls × 4.5 s × 2.5`이며 64,800 s와 비교한다.

### 6.2 Margin

#### EC-EXPRB-K

- noiseless accuracy headroom: `2e-5 / 9.08e-6 ≈ 2.20×`
- cancellation headroom: `2e-4 / 9.16e-6 ≈ 21.8×`
- call headroom: `5500 / 352 ≈ 15.6×`
- wall headroom: `64800 / 3960 ≈ 16.4×`

#### EC-PICARD-IE

- noiseless accuracy headroom: `≈1.10×`
- call headroom: `≈1.37×`
- wall headroom: `≈1.44×`

따라서 Picard는 성공 여부가 실제 physical RHS cost fluctuation이나 contraction degradation에 민감하다.

## 7. Post-formal stress tests

### 7.1 EXPRB step convergence

| h | endpoint WRMS | calls | observed order |
|---:|---:|---:|---:|
| 0.1 | `1.4633e-4` | 88 | — |
| 0.05 | `3.6725e-5` | 176 | 1.994 |
| 0.025 | `9.0769e-6` | 352 | 2.016 |
| 0.0125 | `2.1950e-6` | 704 | 2.048 |
| 0.00625 | `5.0888e-7` | 1,408 | 2.109 |

측정상 global error는 2차에 가깝다.

### 7.2 Krylov saturation

`h=0.025`에서:

- `m=2`: error `9.0873e-6`, 96 calls
- `m=3`: `9.07695e-6`, 128 calls
- `m=4`: `9.076855e-6`, 160 calls
- `m≥6`: formal `m=10`과 사실상 동일

이는 향후 **별도 sealed physical prefix contract**에서 `m=4`를 시험할 강한 이유다. 그러나 formal survivor는 결과 전에 고정된 `m=10`이며, 본 보고서는 `m=4` 결과를 retroactive formal PASS로 사용하지 않는다.

### 7.3 Cancellation ladder

fixed `epsilon=1e-3`에서 EC-EXPRB-K는 cancellation scale `3e12`까지 `2e-4` gate 안에 남고, `1e13`에서 `4.95e-4`로 탈락한다. 이 수치는 physical gain/loss ratio로 직접 변환되지 않는다. physical prefix에서 actual directional-difference signal-to-cancellation ratio를 측정해야 한다.

### 7.4 Resolution scaling

`m=4`, `h=0.025`에서 state dimension 74, 146, 182, 218, 290 모두 정확히 160 RHS calls였다. 이는 reduced Krylov dimension과 number of steps가 고정되어 call count가 state dimension에 독립적이라는 algorithmic property다. 실제 wall per RHS는 physical collision quadrature와 state dimension에 따라 증가할 수 있으므로 `4.5 s` p95 assumption을 별도로 유지했다.

### 7.5 SciPy factor reset은 충분하지 않다

cancellation scale `1e10`에서 persistent factor는 `1e12×sqrt(eps)`까지 상승하고 max column error는 약 92.3이다. reset-each-refresh control은 factor ratchet을 막지만 harsh cancellation에서 median/max relative error가 여전히 약 1이다. 즉 **reset은 특정 SciPy lifetime bug를 제거할 뿐, cancellation-sensitive numerical differentiation 자체를 해결하지 않는다.**

## 8. Scientific validation과 차원·부호·극한 점검

### 8.1 단위

synthetic benchmark time과 state는 dimensionless다. RABBIT mapping에서 독립변수는 `N=ln a`, RHS는 `dY/dN`이며 무차원이다. physical total-energy coordinate `W=e^{4N}(rho_nu+rho_EM)`는 MeV natural units에서 `[W]=MeV^4`다. synthetic scalar projection은 이 physical equation의 theorem이 아니라 software architecture 검증용 analogue다.

### 8.2 부호와 안정성

spectral block diagonal decay는 음수이며 formal block spectral abscissa는 `-6.35016`, radius는 `7.91488`이다. full-state spectral abscissa가 0에 가까운 것은 energy/clock exact zero modes 때문이며 instability가 아니다.

### 8.3 알려진 극한

- autonomous linear nonnormal system은 matrix exponential reference와 일치한다.
- manufactured exact path에서 RHS와 exact derivative가 test tolerance 안에 일치한다.
- identity grid transfer는 `1e-12` 안에서 identity다.
- JVP는 cancellation 없는 analytic Jacobian action과 비교된다.
- EXPRB step halving은 약 2차를 보인다.
- D-JFNK는 cancellation 없는 linear test에서 implicit-Euler reference와 1차 수렴을 보인다.

### 8.4 보존법칙

energy-constrained lanes는 synthetic invariant를 machine precision에 보존한다. raw lanes의 `1e-6` energy defect를 함께 보존하여 projection 없이는 gate가 닫히지 않음을 증명했다.

## 9. Independent diff review

### 9.1 PASS findings

- `exprb.py`와 `picard.py`에는 `model.jacobian`, SciPy `num_jac`, `solve_ivp` 호출이 없다.
- EXPRB derivative information은 매 Arnoldi direction마다 fresh forward directional difference로만 얻는다.
- Picard는 JVP, Jacobian, GMRES를 전혀 사용하지 않는다.
- every JVP/full RHS가 `CallCounter`에 반영된다.
- formal/post-formal protocol hashes가 현재 bytes와 일치한다.
- formal EXPRB primary/cancellation endpoints와 Picard endpoint가 retained artifacts와 bitwise replay된다.
- 실패 lane은 exception/failure reason과 partial state를 보존한다.

### 9.2 Material limitations

1. **Synthetic projector is too informed for direct physical use.** It uses a manufactured exact trace integral. Physical integration must replace it with total-energy coordinate evolution and algebraic temperature recovery, not copy this projector.
2. **No full RABBIT collision RHS was executed.** Actual RHS cost, JVP noise, nonnormality, and occupation-domain behavior remain unmeasured for the candidate.
3. **The benchmark forcing is manufactured.** Endpoint accuracy shows solver consistency on a blocker-shaped system, not cosmological correctness.
4. **Fixed step is not yet physically adaptive.** Formal `h=0.025` is a synthetic holdout choice. A physical prefix must derive an accepted step envelope prospectively.
5. **EXPRB implementation is a research prototype.** It computes a dense exponential only in the small reduced space, but lacks production error control, restart/recycle policy, and fault-tolerant checkpointing.
6. **Picard contraction is not certified on the true operator.** Eigenvalue negativity alone is insufficient under nonnormality.
7. **Controls do not prove an off-the-shelf solver solution.** LSODA did not enter its stiff branch on this manufactured case; Radau/BDF were supplied an analytic synthetic Jacobian.

## 10. Red-team objections

### Objection A: “EXPRB의 352 calls는 너무 좋으니 benchmark가 쉬운 것 아닌가?”

맞다. 이 수치는 physical claim이 아니다. 그래서 4.5 s physical p95 cost를 곱하는 것만으로 full proof라 하지 않고, actual creep prefix에서 observed Krylov dimension, JVP signal/noise, accepted progress, invariant residual을 다시 측정하도록 PR plan을 둔다.

### Objection B: “보존 projection이 정답을 알려준 것 아닌가?”

energy scalar에 대해서는 manufactured invariant를 정확히 사용한다. 따라서 synthetic energy PASS는 solver 자체의 공로가 아니다. 이 때문에 raw negative control을 필수로 남겼고, physical implementation에서는 exact total-energy balance equation만 허용한다. spectral components나 endpoint solution은 projection으로 맞추지 않는다.

### Objection C: “JFNK는 AD-JVP나 complex-step을 쓰면 살아날 수 있지 않나?”

가능성은 있다. 그러나 그것은 현재 black-box collision code의 differentiability, branch structure, interpolation, domain rejection을 모두 통과해야 한다. 또한 direct implicit Euler의 first-order call cost가 noiseless에서도 9,666 calls였으므로 JVP만 고쳐도 formal budget이 닫히지 않는다. higher-order implicit/EPIRK와 결합하는 별도 family가 필요하다.

### Objection D: “m=4면 160 calls인데 왜 m=10을 추천하나?”

m=4 포화는 formal 결과 이후의 post-formal discovery다. frozen formal claim은 m=10이다. physical prefix 전에는 m=10을 conservative baseline으로 쓰고, m=4는 별도 precontract에서 residual/defect gate를 만족할 때만 승격해야 한다.

### Objection E: “Picard가 더 단순한데 primary로 쓰면 되지 않나?”

Picard wall margin은 1.44×뿐이고 실제 contraction이 조금만 나빠져도 4-iteration cap 또는 18-hour bound를 넘는다. EXPRB는 call/wall margin이 15× 이상이고 nonnormal stable operator action을 직접 처리하므로 primary가 더 안전하다.

## 11. 실제 RABBIT 통합 PR DAG

```text
SA00 source/provenance lock
  -> SA01 pure RHS/call-accounting interface
      -> SA02 total-comoving-energy coordinate + T_gamma algebraic recovery
          -> SA03 physical JVP/noise and invariant static discriminator
              -> SA04 EC-EXPRB-K research backend (m=10 baseline)
                  -> SA05 sealed 60/30 creep-prefix run
                      -> [KILL and retain] OR
                         SA06 resolution/noise/Krylov holdouts
                           -> SA07 sealed full 60/30 trajectory
                               -> SA08 independent replay + D-071 adjudication

SA03 -> PB01 true-operator contraction certificate
          -> PB02 EC-PICARD-IE prefix fallback

SA06 -> OPT01 separately sealed m=4/recycled-Krylov optimization
```

상세 spec은 `reports/RABBIT_SOURCE_INTEGRATION_PR_PLAN.md`에 있다.

## 12. Physical prefix의 최소 reopen discriminator

full trajectory 전에 다음을 하나의 immutable contract로 봉인해야 한다.

- exact upstream commit and frozen module hashes
- order 60 / `y_max=30`, original collision catalog and quadrature
- total-energy state equation and algebraic `T_gamma` recovery
- primary EXPRB `m=10`; step schedule derived before output
- raw full-RHS calls, JVP calls, rejected/failed steps, Krylov dimension
- `N∈[0,0.25]` 또는 최소한 retained stall window `0.14≤N≤0.22` 전체 포함
- call projection upper bound `≤5500` with margin
- `p95 full RHS ≤4.5 s` 또는 새로 측정한 더 보수적인 bound
- endpoint/state comparison to completed 48/24 and static 60/30 anchors
- first-law, occupation, domain-rejection, tail and matrix-roundoff ledgers
- no post-output epsilon/m/step refit

prefix가 통과하지 못하면 후보는 preserved KILL이며 full run을 시작하지 않는다.

## 13. 최종 판정

### 확정된 negative results

- 현 SciPy BDF dense finite-difference path는 60/30 blocker에 부적합하다.
- Jacobian factor reset만으로 cancellation-sensitive derivative problem은 닫히지 않는다.
- direct first-order JFNK와 two-level JFNK는 현재 accuracy/call/cancellation gate를 통과하지 못한다.
- pure solver replacement만으로 frozen energy gate를 만족하지 못한다.

### 생존한 연구 아이디어

- **Primary:** total-energy-constrained exponential Rosenbrock–Krylov, formal `m=10`
- **Secondary:** total-energy-constrained derivative-free implicit Picard, true contraction certificate 조건부
- **Next optimization:** separately sealed `m=4` or recycled Krylov

### claim ceiling

\[
\boxed{\texttt{SYNTHETIC ALGORITHM SURVIVOR}}
\]

\[
\boxed{\texttt{PHYSICAL PREFIX NOT YET RUN}}
\]

\[
\boxed{\texttt{D-071 REOPEN NOT EARNED}}
\]

## 14. Reproduction

```bash
python -m pytest -q
python tools/validate_harness.py
python tools/verify_research_evidence.py
python scripts/generate_research_figures.py
```

formal/post-formal outputs are under:

- `results/formal_v1/`
- `results/formal_v2/`
- `results/postformal/`

figures are under `figures/`.

## 15. Primary references

1. D. A. Knoll and D. E. Keyes, “Jacobian-free Newton–Krylov methods: a survey of approaches and applications,” *J. Comput. Phys.* **193** (2004) 357–397, DOI `10.1016/j.jcp.2003.08.010`.
2. P. Tranquilli and A. Sandu, “Rosenbrock–Krylov Methods for Large Systems of Differential Equations,” *SIAM J. Sci. Comput.* **36** (2014) A1313–A1338, DOI `10.1137/130923336`.
3. PETSc SNES manual, matrix-free operators and user preconditioners; PETSc SNESFAS restriction/interpolation/injection APIs.
4. SUNDIALS CVODE documentation, matrix-free Krylov linear solvers and preconditioning; MRIStep documentation for multirate follow-up.
5. SciPy `solve_ivp` BDF documentation and installed `num_jac` implementation, version pinned in the retained environment.
