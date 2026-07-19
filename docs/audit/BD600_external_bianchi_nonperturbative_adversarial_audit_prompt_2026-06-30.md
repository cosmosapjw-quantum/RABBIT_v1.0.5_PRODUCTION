# [INTEGRATED PHYS-MATH-CODE AUDIT — RABBIT Bianchi-I Non-Perturbative Track, Adversarial]

**버전:** 2026-06-30 · **대상 브랜치:** `feature/bianchi-i-full-nonperturbative` · **감사 형태:** 제3자 독립 적대적 감사

---

## 0. 감사자에게 (READ FIRST)

당신은 이 repo에 **직접 접근 가능한 독립 외부 감사자**다. 별도 zip/스냅샷 없음 — `git log`, 소스, 테스트, `docs/audit/*.md`를 직접 읽고, 필요 시 직접 실행하라:

```
env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/pytest -q -p no:cacheprovider <test>
env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -c "..."
```
(ROCm "No AMD GPUs" 경고는 무해한 CPU fallback. CAS toy check은 system `python3` + sympy/Wolfram/Sage.)

**Source of truth 규칙:** 이 repo에서 진실의 우선순위는 **(1) 코드의 실제 실행 출력 > (2) 명세 수식(`*_ko.md`, `docs/audit/*.md`) > (3) 커밋 메시지/문서 claim**. 문서가 "검증됨"이라 해도 코드 path가 그걸 생산하지 않으면 문서가 틀린 것이다. **self-consistency regression lock(자기 출력 재현)을 절대 validation으로 인정하지 마라** — 외부 기준(PRIMAT/Mangano/독립 코드/해석적 한계)만이 validation이다.

**목표:** 새 아이디어 금지. 주어진 수식/증명/알고리즘/코드/수치 파이프라인 **사이의 깨진 고리**를 찾아라. (1) 물리/수학이 맞는가, (2) 코드가 그 수식을 정확히 구현하는가, (3) 수치가 신뢰 가능한가 — 셋을 **절대 동일시하지 마라**.

---

## 1. 의무 적대적 검증 루프 (MANDATORY — 모든 STEP에 적용)

각 감사 항목마다 아래 7단 루프를 **명시적으로** 수행하고 그 흔적을 출력에 남겨라. 한 단계라도 생략하면 그 항목은 미완료로 간주.

1. **Self-Discover** — 이 항목을 검증하는 데 필요한 추론 모듈을 먼저 스스로 구성하라 (어떤 불변량/한계/대조가 결정적인가?). 즉답하지 말고 검증 구조를 설계.
2. **Step-Back** — 구체 코드로 들어가기 전에 지배 원리로 추상화하라 (예: "이건 detailed balance 문제인가, hyperbolicity 문제인가, quadrature 문제인가?"). 잘못된 추상화 층에서 싸우면 헛수고.
3. **Metacognitive Self-Ask** — "내가 지금 무엇을 당연하게 가정하고 있나? 저자의 프레이밍을 그대로 빌려 쓰고 있지 않나? 내가 놓칠 법한 것은?"을 명시적으로 자문.
4. **CoVe (Chain-of-Verification)** — 결론에 대해 **독립 검증 질문 3~5개**를 생성하고, 각각을 원 추론과 분리해 답한 뒤 불일치를 찾아라. (예: "exp(2I)=‖A n₀‖라면, S=0에서 I는 정확히 0인가? 직접 계산.")
5. **Adversarial Self-Ask** — 자신의 잠정 결론을 **반증**하려 시도하라. "이 PASS가 vacuous하게 통과하는 입력은 없나? 이 ~28%가 setup artifact일 시나리오는?" 깨뜨릴 입력/반례를 능동적으로 만들어라.
6. **CCoT (Contrastive CoT)** — "올바른 구현 vs 흔한 오구현"을 대조해 추론하라 (부호 뒤집힘, 측도 누락, normalization 1개 차이, regime 혼동). 대조가 드러내는 신호를 기록.
7. **PDR (Plan–Do–Review, 점진 심화)** — 판정을 내린 뒤 **한 번 더** 리뷰 패스를 돌려라: 증거가 결론을 지탱하는가, 더 싼 판별 테스트가 있는가, 심각도 등급이 과대/과소인가. 필요 시 STEP을 재진입.

> 규칙: 오류 하나 찾았다고 종료하지 마라. 루프를 끝까지 돌려 **모든** 깨진 고리를 열거하라.

---

## 2. 감사 순서 (STEP 0–8) — 이 코드에 특화

### STEP 0. AUDIT TARGET RECONSTRUCTION
아래 3중 트랙을 분리 재구성하고 "식/정의 → 상태변수 → 함수/모듈 → 출력/테스트" 대응표를 만들어라.

- **트랙 A — PSTF exact-collision substrate (from-scratch, Stage 0–2):** `src/rabbit/collisions/pstf_*.py` (trilinear kernel, wigner, body-frame, x-integral, lrs-moment-kernel, matrix-element, reduced-kernel, collision-operator), `scripts/cas/gen_pstf_pi3.py` → `_pstf_pi3_generated.py` → `pstf_pi3_analytic.py`. 핵심 주장: all-orders Fermi blocking(six-monomial), 정확 detailed balance, CAS-exact F^(3), body-frame factorization, 닫힌형 Iₙ. **이 substrate는 production driver에 미연결임을 명시적으로 확인하라.**
- **트랙 B — production transport/BBN driver:** `forward_likelihood.canonical_forward_solver`, `drivers/full_coupled_typeI`(scipy 기준), `jax/driver_typeI_char`(characteristic LRS/non-LRS), `jax/characteristic_rays_jax` / `characteristic_rays_nonlrs_jax`, `transport/typeI_even_ladder_hierarchy`, `transport/augmented_nonlrs_transport`(AP62 logit), `transport/realizability_diagnostics`(신규).
- **트랙 C — 이번 세션의 검증/측정 주장 (PR#13–19, `docs/audit/*_2026-06-30.md`):** 아래 §4의 high-risk 목록.

각 트랙에서 source of truth가 무엇인지 명시.

### STEP 1. CONTRACT / INTERFACE AUDIT
다음 contract를 **먼저** 복원하라. 불명확하면 그 자체로 P0/P1.
- metric signature (−,+,+,+) 및 χ_ij = E_iE_j − p_ip_j(n_i·n_j) 부호 안전성.
- PSTF projection/trace-free 규약, m=0 LRS contraction.
- 6-monomial 통계인자 `collision_statistical_factor` = f3f4 − f1f2 + f1f2f3 + f1f2f4 − f1f3f4 − f2f3f4 (= (1−f1)(1−f2)f3f4 − f1f2(1−f3)(1−f4)의 quartic 상쇄형) — 부호·정규화.
- characteristic 맵 A = diag(e^{−(S₊+√3S₋)}, e^{−(S₊−√3S₋)}, e^{2S₊}), exp(2I)=‖A n₀‖, J=det(A)/‖A n₀‖³=‖A n₀‖⁻³ (det A=1, trace-free). q→q·‖A n₀‖ 에너지 시프트 규약.
- band-limited logit f=1/(1+e^{−(q+A)}) 의 (0,1) 무조건 realizability; w_s2 정규화(합=2)와 0.5·monopole 관례.
- even-ladder RHS = Σ_H[M_stream⊗(q∂_q) + 1.5 M_angle]·F, A/B/C/tilde 계수.
- 관측량 단위: Yp, D/H, N_eff = (T_νe/T_std)⁴ + 2(T_νx/T_std)⁴, T_std = T_γ(4/11)^{1/3}.
- shape/type, units, domain, boundary/IC, 기대 불변량(보존/positivity/detailed-balance null), solver/regime 가정, test oracle.

### STEP 2. PHYS-MATH AUDIT
순서: (1) 정의/표기 충돌 → (2) 인덱스/trace/PSTF/projection → (3) 부호/정규화 → (4) 단위/차원 → (5) known limit/baseline 회복 → (6) 경계/정칙성/positivity → (7) 숨은 가정 → (8) 반례 special case. 각 항목 통과/의심/실패 + 이유 + 코드 영향.
- **특화 표적:** CAS F^(3)가 명세의 generalized HM angular polynomial Π^(3)와 일치하는가(머신 정밀 toy check). detailed balance가 **callable f4@exact p4**에 의존함 — 격자 보간으로 깨지지 않는가. characteristic 맵의 "diagonal Type-I(orthogonal, Wainwright-Hsu)" 가정이 실제 적용 regime과 일치하는가(tilted/non-orthogonal에서 무효).

### STEP 3. EQUATION-TO-CODE MAPPING AUDIT
- 핵심 식이 코드 어디서 구현되나? 이름만 같고 다른 양을 쓰지 않나?
- approximation/regime이 코드에 분명히 반영되나? (예: even-ladder는 "shear 1차까지만 exact" — `rhs_typeI.py` docstring. δρ는 O(S²) — 1차 모델이 2차 관측량을 부분만 포착.)
- 본문 claim ↔ 실제 path 어긋남. dead code/placeholder/unused param/disconnected module.
- **reduced/surrogate/helper를 production처럼 오인하지 마라:** (i) PSTF substrate는 production 미연결(트랙 A). (ii) `nu_nu_scattering_jax`의 production νν 커널은 미와이어 + `matrix_coeff=1.0`(DHS 보정 PR-T3C로 deferred). (iii) JAX characteristic은 collisionless-only(`forward_likelihood.py:2243` ValueError); non-LRS 충돌은 `driver_typeI_char.py:1587` NotImplementedError. (iv) even-ladder moment hierarchy는 shear 충실 ~28%(보조 엔진, 정확도 경로 아님).

### STEP 4. NUMERICAL / PIPELINE AUDIT
- solver 적합성(stiff RODAS5P/AP65, characteristic 닫힌형, even-ladder RK45), tolerance 민감도, under/overflow/cancellation/negative state, conditioning/divergence, interpolation/tabulation artifact, warm-start/cache/state leak(lru_cache), seed/reproducibility/determinism, OOD/misspecification, postprocessing 의존, baseline 재현 경로.
- **특화 표적:** jax x64 강제 여부(미설정 시 f32 오염). 교차백엔드 parity가 N_q/N_mu/N_theta/N_phi 해상도 의존(Σ_plus=0.5에서 dYp~7e-5로 발산 — 해상도 artifact). max|A| 강전단 성장의 **empirical 천장(PR#17, 증명된 a-priori bound 아님)**. realizability margin이 q_max 격자 의존(FD 고-q 꼬리 underflow). "이게 physics/구현/수치/diagnostics 오류 중 무엇인가" 분류.

### STEP 5. FAILURE MODE SYNTHESIS
failure mode 최대 7개: 유형(physics/math/implementation/numerical/interface/testing) · 심각도(P0–P3) · 관측 증상 · 근본 원인 · 가장 싼 판별 테스트 · 오해석 위험.

### STEP 6. VERIFIER FILTER
A. Physics(known-limit/dimension/sign-norm/positivity/alt-explanation) · B. Code(contract/실제 path/regression/reproducibility) · C. Numerical(tolerance/convergence/baseline/uncertainty). 각 passed/partial/failed.

### STEP 7. MINIMAL REPAIR PLAN
대규모 리팩토링 금지. 최소 수정 패치 **최대 3개**: 무엇을·왜 load-bearing·어떤 failure mode 차단·새 테스트·regression 영향.

### STEP 8. MINIMAL TEST SET
즉시 실행 가능: baseline 재현 1 · edge/adversarial 1 · physics sanity 1 · numerical stability/sensitivity 1 · regression 1. 각 pass/fail 기준.

---

## 3. 우선 의심 표적 (이 코드 특화 — 여기를 먼저 공격하라)

> 아래는 이번 세션이 스스로 "검증됨/한계 고정"이라 주장한 항목들이다. **저자의 결론을 빌리지 말고 §1 적대적 루프로 독립 재검증하라.** 특히 "두 내부 reviewer가 통과시켰다"는 사실은 증거가 아니다.

1. **PR#15 ~28% shear capture (`B2_even_ladder_shear_capture`)** — matched setup(S=Σ_H·n_end=Σ_H·0.5, 동일 grid/IC)이 진짜 apples-to-apples인가? δρ~O(S²)·capture가 S→0에서 0.27로 수렴(≠1)한다는 주장이 setup/normalization artifact가 아님을 **독립적으로** 재현하라. 두 관측량(`energy_density_ratio_from_state` vs characteristic trapezoid)이 같은 양인가.
2. **PR#14 CFL mis-targeted (`BHR_wellposedness`)** — "Bianchi I 공간 동질(D_a=0) → spatial flux Jacobian 없음 → CFL 표적 없음"이 옳은가? M_stream(미분 곱) 실스펙트럼·대각화가 진짜 well-posedness 조건인가? M_angle 복소 고유값이 정말 무해(0차 source)인가, 아니면 강전단에서 성장 모드인가?
3. **PR#16/19 characteristic exact 맵** — `intensity_shift_nonlrs_jax`/`jacobian_nonlrs_jax`의 닫힌형이 실제 LRS Liouville 해와 일치하는가(ODE solve_ivp 대조, S≠0). det A=1·J=‖A n₀‖⁻³ 푸시포워드가 옳은가. PR#19 transitive anchor(non-LRS==LRS==scipy)에서 **공유 편향(같은 quadrature/network 버그)**이 세 경로 모두를 동일하게 틀리게 해 검출 불가한 시나리오를 구성하라.
4. **PR#13 "gap이 νν 밖" (`A2_crosstrack_diagnostic`)** — production N_eff=2.993 런이 νν를 정말 배제하는가(`driver_typeI_full_boltzmann.py:748` 직접 확인). "νν 밖"이 **배제에 의한 동어반복**이 아니라 실제 국소화인가. Bianchi νν operator의 N_q≥12 수렴 주장 재현.
5. **PR#17 realizability** — 0≤f≤1이 vacuous임은 맞다. 그런데 대체 지표(max|A| 천장 8/12)는 **증명된 bound가 아닌 empirical lock**임을 코드/문서가 정직하게 표기했는가, 아니면 어딘가에서 "안정성 증명"으로 격상됐는가. maxent_M2_multiplier가 올바른 closure인가(±1 커널에서 atanh(π) toy check).
6. **PSTF substrate detailed balance** — FD 평형에서 Chat_0=0이 **머신정밀**로 성립하는가, 그리고 그것이 F_0의 값과 무관하게 통계인자만으로 보장되는가(callable f4@exact p4 의존성). CAS F^(3) 결정성(동일 MD5 재생성) 및 명세 Π^(3)와의 머신정밀 일치.
7. **Mangano +0.0095 gap** — "수렴된 physics 오차, solver 밖"이라는 분류가 옳은가, 아니면 weak-rate normalization/e± reheating 해상도/spectral distortion에 숨은 구현 오류인가. exact-kernel이 2.993(<3.0, Mangano보다 더 먼)으로 수렴한다는 사실 재현.
8. **검증의 가장 깊은 구멍 (구조적):** 유한전단 비등방 BBN의 **외부 교차코드 앵커 부재**(`jax_bbn_gold.json` 51 Σ_H cell은 self-consistency lock뿐). 공유 편향 오류는 현재 어떤 테스트도 잡지 못한다 — 이를 P0/P1로 평가하고, 외부 앵커 없이 줄일 수 있는 최소 잔여 위험을 제시하라.

---

## 4. 출력 형식

1. Audit target reconstruction (3트랙 대응표)
2. Contract/interface table
3. Phys-math audit ledger (항목별 통과/의심/실패 + §1 루프 흔적)
4. Equation-to-code mapping audit
5. Numerical/pipeline audit (physics/구현/수치/diagnostics 분류)
6. Ranked failure modes (P0~P3, 최대 7)
7. Verifier results (A/B/C passed/partial/failed)
8. Minimal repair plan (최대 3 패치)
9. Minimal test set (5종, pass/fail 기준)
10. 최종 판정: **치명적 오류 있음 / 부분 통과 / 통과** · 지금 당장 수정할 1개 · 지금 손대면 안 되는 1개

---

## 5. 금지 (HARD)

- 프레임워크 전면 교체·대규모 리팩토링을 기본값으로 제안.
- 실제 bug/contract 문제를 architecture aesthetics로 덮기.
- 수식 재서술만 하고 코드 path 미확인.
- **smoke test / self-consistency regression lock을 validation으로 인정.**
- **논문/커밋/문서 claim을 구현 증거로 취급.** (커밋 메시지의 "verified"는 증거가 아니다.)
- 내부 reviewer 통과를 외부 검증으로 간주.
- repo governance("no publication claims")를 **취약점을 가리는 방패로 오용** — 거꾸로, 어디서 그 governance가 깨지고 publication-grade로 over-claim하는지 적발하라.
- 오류 하나 찾고 조기 종료 (§1 루프 끝까지, 모든 깨진 고리 열거).

---

**한 줄 요약 (감사자용):** 이 코드는 1139-commit 성숙 코드베이스의 비등방 비섭동 트랙이다. FLRW는 외부 앵커됨; characteristic-ray 경로는 정확 기준으로 주장됨; moment hierarchy 한계(~28%)는 측정됨; **그러나 충돌×강비등방 결합은 미완(collisionless-only)이고, 유한전단 비등방은 외부 앵커가 없다.** 당신의 임무는 이 주장들의 깨진 고리를 §1 적대적 루프로 끝까지 찾아내는 것이다.
