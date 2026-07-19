# [INTEGRATED PHYS-MATH-CODE AUDIT — RABBIT Audit-Remediation + Collision-Twin Track, Adversarial]

**버전:** 2026-06-30 · **대상 브랜치:** `feature/bianchi-i-full-nonperturbative` · **감사 형태:** 제3자 독립 적대적 감사
**선행 감사:** `docs/audit/BD600_external_bianchi_nonperturbative_adversarial_audit_prompt_2026-06-30.md` 및 그 응답 `integrated_phys_math_code_audit_2026-06-30.md`(판정: **부분 통과**)의 후속. 이번 대상은 그 외부 감사에 대한 **수정 작업(PR#20–23)** 과 **충돌-트윈 작업(B4-PR1/2/3)** 이다.

---

## 0. 감사자에게 (READ FIRST)

당신은 이 repo에 **직접 접근 가능한 독립 외부 감사자**다. 별도 zip/스냅샷 없음 — `git log`, 소스, 테스트, `docs/audit/*.md`를 직접 읽고, 필요 시 직접 실행하라:

```
env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/pytest -q -p no:cacheprovider <test>
env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -c "..."
venv/bin/python scripts/promotion_check.py            # 클레임 게이트 상태(green/red) 확인
```
(ROCm "No AMD GPUs" 경고는 무해한 CPU fallback. 충돌 트윈은 `jax.config.update("jax_enable_x64", True)` 강제 필요 — 미설정 시 f32 오염.)

**Source of truth 규칙:** 진실의 우선순위는 **(1) 코드의 실제 실행 출력 > (2) 명세 수식 / `docs/audit/*.md` > (3) 커밋 메시지/문서 claim**. 커밋이 "parity-locked / fixed / validated"라 해도 코드 path가 그걸 생산하지 않으면 커밋이 틀린 것이다. **self-consistency / parity-to-a-model 을 절대 physical validation 으로 인정하지 마라** — 외부 기준(PRIMAT/Mangano/독립 코드/해석적 한계)만이 validation이다.

**이번 감사의 핵심 경고:** 이번 작업은 대부분 **"수정(remediation)" 과 "엔지니어링 트윈(parity-locked twin)"** 이다. 두 가지 함정을 특히 경계하라 — (a) "over-claim 을 고쳤다"는 수정이 **정말 over-claim 을 막는가, 아니면 표현만 바꿨는가**; (b) numpy 기준에 머신정밀로 일치하는 JAX 트윈은 **그 numpy 기준이 옳을 때만** 의미가 있다 — **기준 자체가 calibrated approximation 이면 parity 는 calibration 을 상속할 뿐 physics 를 검증하지 않는다.**

**목표:** 새 아이디어 금지. 주어진 수정/트윈/테스트/게이트 **사이의 깨진 고리**를 찾아라. (1) 수정이 실제로 결함을 닫는가, (2) 트윈이 기준을 정확히 복제하는가 **그리고 그 기준이 무엇을 의미하는가**, (3) 게이트가 실제로 fail-closed 인가 — 셋을 **절대 동일시하지 마라**.

---

## 1. 의무 적대적 검증 루프 (MANDATORY — 모든 STEP에 적용)

각 감사 항목마다 아래 7단 루프를 **명시적으로** 수행하고 그 흔적을 출력에 남겨라. 한 단계라도 생략하면 그 항목은 미완료로 간주.

1. **Self-Discover** — 이 항목을 검증하는 데 필요한 추론 모듈을 먼저 스스로 구성하라 (어떤 불변량/한계/대조가 결정적인가?). 즉답하지 말고 검증 구조를 설계.
2. **Step-Back** — 구체 코드로 들어가기 전에 지배 원리로 추상화하라 (예: "이건 parity-to-model 문제인가, calibration 문제인가, fail-closed 게이트 우회 문제인가?"). 잘못된 추상화 층에서 싸우면 헛수고.
3. **Metacognitive Self-Ask** — "내가 지금 무엇을 당연하게 가정하고 있나? 저자의 '수정됨/parity-locked' 프레이밍을 그대로 빌려 쓰고 있지 않나? 내가 놓칠 법한 것은?"을 명시적으로 자문.
4. **CoVe (Chain-of-Verification)** — 결론에 대해 **독립 검증 질문 3~5개**를 생성하고, 각각을 원 추론과 분리해 답한 뒤 불일치를 찾아라. (예: "fail-closed 라면, `promotion_check.py` 를 실제로 돌렸을 때 `GATE_SIGMA_H_TO_0P95` 가 red 로 나오는가? 직접 실행.")
5. **Adversarial Self-Ask** — 자신의 잠정 결론을 **반증**하려 시도하라. "이 firewall 이 놓치는 over-claim 사이트는? 이 fail-closed 게이트를 green 으로 만들 입력/설정은? 이 parity 가 calibration 을 validation 으로 위장하는 지점은?" 깨뜨릴 입력/반례를 능동적으로 만들어라.
6. **CCoT (Contrastive CoT)** — "올바른 구현 vs 흔한 오구현"을 대조해 추론하라 (skip-vs-fail, capped-vs-uncapped 가중치, parity-to-model vs validation, 부호/측도 1개 차이). 대조가 드러내는 신호를 기록.
7. **PDR (Plan–Do–Review, 점진 심화)** — 판정을 내린 뒤 **한 번 더** 리뷰 패스를 돌려라: 증거가 결론을 지탱하는가, 더 싼 판별 테스트가 있는가, 심각도 등급이 과대/과소인가. 필요 시 STEP을 재진입.

> 규칙: 오류 하나 찾았다고 종료하지 마라. 루프를 끝까지 돌려 **모든** 깨진 고리를 열거하라.

---

## 2. 감사 순서 (STEP 0–8) — 이 작업에 특화

### STEP 0. AUDIT TARGET RECONSTRUCTION
아래 2중 트랙을 분리 재구성하고 "주장 → 변경 파일 → 게이트/테스트 → 실제 효과" 대응표를 만들어라.

- **트랙 R — 감사-수정(PR#20–23):**
  - PR#20 `c87b16b` — 성숙도 과대표현(`publication-grade`/`PUBLIC TIER`/`tier-3 eliminates residual gap`/`publication-quality`) downscope + 회귀 firewall `tests/test_claim_gates.py::TestSourceClaimLanguageHygiene`. 변경: `config/backend_capabilities.py`, `jax/driver_typeI*.py`, `collisions/pstf_*.py`(×7 docstring), `docs/audit/A2_*`.
  - PR#21 `8086a69` — M_angle "harmless bounded oscillation" → "측정된 양의 실수부 성장 spectrum". 변경: `transport/even_ladder_wellposedness.py`(`angle_source_eigenvalues` 신규), `tests/test_even_ladder_wellposedness.py`, `docs/audit/BHR_wellposedness_2026-06-30.md`.
  - PR#22 `df0c6bb` — fail-closed 외부 앵커 게이트. `src/rabbit/external/finite_shear_anchor.py`(신규), `tests/test_finite_shear_external_anchor.py`(신규), `config/claim_gates.py`(`GATE_SIGMA_H_TO_0P95` 와이어), `pyproject.toml`(`external_anchor` 마커).
  - PR#23 `92d9bc4` — `GATE_SIGMA_H_TO_0P95` 내부 노드 + Friedmann-잔차 동어반복 반증. `tests/test_typeI_extended_range.py`(신규), `config/claim_gates.py`(permitted_text/notes), `docs/audit/extended_range_internal_sweep_2026-06-30.md`.
- **트랙 T — 충돌-트윈 기반(B4-PR1/2/3):**
  - B4-PR1 `8d66fea` — numpy 기준/parity oracle `tests/test_physical_collision_operator_reference.py` (대상: `collisions/projected_operator.PhysicalCollisionOperator`).
  - B4-PR2 `c29274f` — JAX 연산자 트윈 `jax/collision_operator_jax.physical_collision_rhs_jax` + parity `tests/test_jax_collision_operator_parity.py`.
  - B4-PR3 `13db0e6` — JAX gather-scatter 브리지 트윈 `jax/teff_collision_bridge_jax.apply_gather_scatter_collision_jax` + parity `tests/test_jax_teff_collision_bridge_parity.py`.
  - **미완 B4-PR4(live wiring):** `forward_likelihood.py:2243` ValueError 제거 + `_rhs_core` 충돌 와이어. **아직 미구현임을 확인하고**, 계획(`I_coll` 단일 DOF 누산기 접근)의 타당성을 §3.7로 평가하라.

각 트랙에서 source of truth가 무엇인지, 그리고 **변경이 production path에 도달하는지 / 테스트-전용인지** 명시.

### STEP 1. CONTRACT / INTERFACE AUDIT
다음 contract를 **먼저** 복원하라. 불명확하면 그 자체로 P0/P1.
- **충돌 연산자 모델(`projected_operator.py`):** C₀(q)=κ₀(Γ/H)[Ψ₀ᵗᵃʳᵍᵉᵗ−Ψ₀], C₂=−κ₂(Γ/H)Ψ₂, Ψ₀ᵗᵃʳᵍᵉᵗ=(Tγ−Tν)/Tγ·q·(1−f₀), κ₀=1, κ₂=5/3. Γ/H=(C_rate/π⁵)G_F²·a·F(mₑ/Tγ)·Tγ⁴·Tν/(H/_MEV2S), **C_rate=210 은 "N_eff≈3.044 재현용 calibrated"**(docstring 명시), a_e/a_x≈4.68. **이것은 first-principles 충돌적분이 아니라 RTA relaxation 모델임을 명시 확인.**
- **gather-scatter 브리지(`teff_collision_bridge.py`):** Θⱼ=exp(−2Iⱼ), f̃₀(q)=½ΣⱼwⱼJⱼf(q/Θⱼ), Ψ₀=(f̃₀−f_eq)/f_eq, **δIⱼ=−δρ_ν/(8ρ_ref)** 등방 scatter. δρ 가중치는 exp 비제한, ρ_ref 가중치는 exp 를 q_cap=80 에서 제한(+f0 없음). **"/8" 인자(δΘ/Θ=δρ/4ρ, δI=−δΘ/2Θ)의 유도**를 검증하라.
- **fail-closed 게이트 기계(`claim_gates.py` + `promotion_check.py`):** 게이트는 `required_test_node_ids` 가 **모두 pass** 일 때만 green. `promotion_check.py:107` 은 `pytest returncode==0` 을 pass 로 셈 — **skip 도 returncode 0** 이므로 `skipif` 노드는 거짓 green. PR#22 는 이를 피하려 `pytest.fail`(returncode≠0) 사용. **이 메커니즘 전체를 실행으로 재현하라.**
- shape/type, units, 기대 불변량(detailed-balance null, 에너지 보존, monotonicity), test oracle.

### STEP 2. PHYS-MATH AUDIT
순서: (1) 정의/표기 → (2) 인덱스/측도 → (3) 부호/정규화 → (4) 단위/차원 → (5) known limit → (6) 경계/정칙성 → (7) 숨은 가정 → (8) 반례. 각 항목 통과/의심/실패 + 이유 + 코드 영향.
- **특화 표적:**
  - **RTA 모델의 물리적 지위:** Ψ₀ᵗᵃʳᵍᵉᵗ 의 선형화(`f_FD(q·Tν/Tγ)−f_FD(q)=(ΔT/Tγ)q·f₀(1−f₀)`)가 ΔT/T=1% 에서 ~7% 오차라는 docstring 한계를 트윈/브리지가 상속하는가. q-비의존 Γ(RTA)의 정당성.
  - **δI=−δρ/(8ρ) 유도:** Θ=exp(−2I) 와 δρ_ν∝Θ⁴(또는 Tν⁴) 가정에서 "/8" 이 정확한가, 아니면 4-인자 어딘가 부호/2배 오류인가. 등방 scatter 가 **강전단**에서도 에너지 보존을 정확히 하는가.
  - **M_angle 성장률 해석(PR#21):** max Re λ≈+0.27(1.5Σ_H 단위)가 진짜 성장 모드인가. "장기 진폭은 A_modes logit lock(B5)이 가둔다"는 주장이 **순환 안심**(M_angle 은 B5 가 가두고, B5 max|A| 천장은 그 자체로 empirical lock)인지 평가.
  - **Friedmann 잔차 동어반복(PR#23):** `friedmann_residual_typeI` 이 `Ω:=1−Σ²` 로 정의되어 `|1−Ω−Σ²|≡0` 임을 확인. 따라서 mass_conservation(`|ΣXᵢ−1|`)이 **유일한 비자명** 내부 제약임을 검증.

### STEP 3. EQUATION-TO-CODE MAPPING AUDIT
- 핵심 식이 코드 어디서 구현되나? 이름만 같고 다른 양을 쓰지 않나?
- **트윈 ↔ 기준 정합:** `physical_collision_rhs_jax` 가 `PhysicalCollisionOperator.evaluate` 를 **모든 항**(κ 인자, source/damping 분해, eps-band guard, energy-weight quadrature, per-species 브로드캐스트)에서 복제하는가. `apply_gather_scatter_collision_jax` 가 numpy 브리지의 **scatter guard 순서**(rho_ref 0-guard → nan_to_num(δI) → relax 마지막)를 정확히 복제하는가.
- **claim downscope 가 실제 path에 도달하나:** PR#20 의 firewall(`TestSourceClaimLanguageHygiene`)이 **curated 목록 whack-a-mole** 라서 새 over-claim 사이트를 놓치지 않는가. 게이트 `permitted_text`(83/139행의 정당한 "Publication-grade")와 honest hedge 가 **clobber 되지 않았는가**. src/ 를 직접 re-grep 하여 firewall 밖 잔존 성숙도 표현을 찾아라.
- **reduced/surrogate/calibrated 를 production-validated 로 오인하지 마라:** (i) 충돌 트윈은 **calibrated RTA 모델의 parity 복제**이지 first-principles 충돌이 아니다. (ii) gather-scatter 의 tangency D2 는 P2≡−0.75 상수로 인해 **항상 ~0** — Teff-manifold 가정에 대한 실질 guard 가 아님(diagnostic 무의미 여부 평가). (iii) JAX characteristic 충돌은 **여전히 미연결**(B4-PR4 미완) — 트윈은 존재하나 driver 에 와이어되지 않음.

### STEP 4. NUMERICAL / PIPELINE AUDIT
- jax x64 강제 여부, parity tolerance(트윈 rtol 1e-12, 브리지 rtol 1e-10)가 실제 정밀도를 가리지 않는지, jit-vs-eager fma 재배열, nan_to_num 이 **있어야 할 gradient 를 조용히 0으로** 만들지 않는지, lru_cache/state leak, seed/determinism.
- **특화 표적:**
  - δρ 가중치(exp 비제한) vs ρ_ref 가중치(exp q_cap=80 제한)의 **비대칭**이 고-N_q(예 N_q=80, q_max>80)에서 ρ_ref 에 편향을 넣는가 — parity 테스트가 N_q=20(전부 q<80)이라 이를 놓치는가.
  - `_YP_GOLD` 회귀 lock(PR#23)이 Σ_H>0.75 **caution regime** 값을 hard-lock 하지 않고 reliable(≤0.75)로 제한했는지, 그리고 mass_conservation 1e-12 가 측정 4.6e-14 대비 진짜 lock 인지.
  - fail-closed 게이트의 `external_anchor` 마커가 default suite/`-m production`에서 어떻게 처리되는지(red-by-design 의 부작용), 그리고 `addopts -m` 를 추가하면 `promotion_check` collect 가 노드를 놓쳐 거짓 red→green 전이가 생기는지.

### STEP 5. FAILURE MODE SYNTHESIS
failure mode 최대 7개: 유형(physics/math/implementation/numerical/interface/testing) · 심각도(P0–P3) · 관측 증상 · 근본 원인 · 가장 싼 판별 테스트 · 오해석 위험.

### STEP 6. VERIFIER FILTER
A. Physics(known-limit/dimension/sign-norm/RTA-한계 상속) · B. Code(트윈-기준 정합/firewall 완전성/게이트 fail-closed 실증) · C. Numerical(tolerance/convergence/q_cap 편향/baseline). 각 passed/partial/failed.

### STEP 7. MINIMAL REPAIR PLAN
대규모 리팩토링 금지. 최소 수정 패치 **최대 3개**: 무엇을·왜 load-bearing·어떤 failure mode 차단·새 테스트·regression 영향.

### STEP 8. MINIMAL TEST SET
즉시 실행 가능: baseline 재현 1 · edge/adversarial 1 · physics sanity 1 · numerical stability 1 · regression 1. 각 pass/fail 기준. **`promotion_check.py` 실행으로 게이트 red 를 직접 실증하는 테스트를 1개 포함하라.**

---

## 3. 우선 의심 표적 (이 작업 특화 — 여기를 먼저 공격하라)

> 아래는 이번 세션이 스스로 "수정됨/parity-locked/fail-closed"라 주장한 항목들이다. **저자의 결론과 두 내부 reviewer 통과를 빌리지 말고 §1 적대적 루프로 독립 재검증하라.**

1. **PR#20 claim firewall — 표현 변경인가 실효인가.** `TestSourceClaimLanguageHygiene` 는 **curated 사이트 목록**이다. (a) src/ 전체를 직접 re-grep 하여 firewall 밖에 남은 성숙도 over-claim 을 찾아라(특히 비-pstf, 비-driver 모듈, README/docs). (b) 정당하게 gated 된 `claim_gates.py` permitted_text 의 "Publication-grade"(83/139행)와 honest hedge 3곳이 보존됐는지 확인. (c) downscope 된 문구가 **새로운 거짓 주장**(예 "FLRW-anchored" 가 실제 앵커보다 강한가)을 도입하지 않았는지.
2. **PR#21 M_angle 성장 주장 — 순환 안심 여부.** max Re λ≈+0.27 을 **독립 재계산**하라(`angle_source_eigenvalues` vs 직접 `np.linalg.eigvals(build_angle_matrix(...))`). "성장은 B5 의 A_modes lock 이 가둔다"는 주장과 "B5 max|A| 천장은 empirical lock(증명 아님)"이라는 PR#17 의 정직한 단서를 나란히 놓아라 — **두 empirical lock 이 서로를 보증하는 순환**이면 장기적분에서 진짜 발산 가능성은 누구도 배제하지 못한다. 이를 평가하라.
3. **PR#22 fail-closed 게이트 — 실제로 red 인가, 우회 가능한가.** `venv/bin/python scripts/promotion_check.py` 를 **직접 실행**하여 `GATE_SIGMA_H_TO_0P95` 가 red 임을 실증하라. `pytest -m external_anchor tests/test_finite_shear_external_anchor.py` 가 **fail**(skip 아님)함을 확인. (a) `skipif` 였다면 `promotion_check.py:107` 이 거짓 green 을 줬을 시나리오를 재현. (b) `finite_shear_anchor.load_benchmark` 의 self-run 거부(`source=~rabbit/self`)가 **RABBIT 자체 생성 표를 외부 앵커로 위장하는 시도를 실제로 차단**하는지 합성 벤치마크로 시험. (c) 게이트를 green 으로 만드는 **모든** 경로(env var, 위장 fixture, addopts 조작)를 능동적으로 탐색.
4. **PR#23 내부 노드 — 비자명한가.** `friedmann_residual_typeI(Σ,0)≡0` 동어반복을 확인하고, 따라서 mass_conservation(`|ΣXᵢ−1|`)이 유일한 비자명 제약임을 검증. Σ_H→0.95 sweep 을 재현하여 (a) 솔버가 실제로 완주하는지, (b) Yp 가 0.75 직상에서 급등(0.256→0.324)하는 것이 물리인지 수치 artifact 인지, (c) permitted_text 가 이제 "validated" 를 **외부 앵커 없이는 promote 못 하도록** 정확히 표현하는지.
5. **B4-PR1/2 연산자 트윈 — 가장 깊은 함정: parity ≠ physics.** 트윈은 `PhysicalCollisionOperator`(C_rate=210 **calibrated**, RTA, 선형화 ~7%@1%, q-비의존 Γ, νν 없음)에 머신정밀 일치한다. **이 parity 를 physical validation 으로 격상한 곳이 코드/문서/커밋에 있는가**를 적발하라. 트윈이 기준의 **알려진 한계까지** 충실히 복제하는지(즉 한계도 상속하는지) 확인. 어떤 테스트/문서도 "충돌 물리가 검증됨"이라 말해선 안 된다 — 말한다면 그것이 over-claim.
6. **B4-PR3 브리지 scatter — 물리 유도 vs 우연.** δI=−δρ/(8ρ_ref) 의 "/8" 을 **직접 유도**하여 부호/2배 오류를 찾아라. (a) δρ 가중치(exp 비제한)와 ρ_ref 가중치(exp q_cap=80 제한, f0 없음)의 비대칭이 ρ_ref 를 **올바른 기준 에너지**로 만드는지, 아니면 q_cap 이 고-N_q 편향을 넣는지(N_q=80 에서 재현). (b) 등방 scatter(모든 ray 동일 δI)가 강전단에서 에너지 보존 + FD 형상 보존을 정말 하는지. (c) tangency D2 가 P2≡−0.75 로 **항상 ~0** 임을 확인 — Teff-manifold 이탈에 대한 **실질 guard 가 전무**한지(있다면 그것이 거짓 안심).
7. **B4-PR4 계획(미완) — I_coll 단일 DOF 의 타당성.** 계획은 등방 δI 를 단일 스칼라 `I_coll` 누산기로 state 에 추가한다. **충돌이 강전단에서 진짜 등방인가** — 비등방(ray-의존) 보정을 단일 DOF 가 표현 못 하는 시나리오를 구성. 교차-driver 게이트(JAX-char-collisional vs scipy collisional char)는 **둘 다 같은 calibrated 모델**이므로 shared-calibration parity 이지 validation 이 아님을 명시 평가.
8. **구조적 잔여(불변):** 충돌×비등방 결합 전체가 **calibrated RTA 모델 + 등방-scatter Teff 브리지** 위에 있다 — 둘 다 first-principles 아님 — 그리고 유한전단 비등방은 여전히 외부 앵커 없음(B8). B4 트윈은 JAX 경로가 scipy 의 **calibrated 근사를 충실히 재현**하게 만든 **엔지니어링 성취**(미분가능/jit가능)이지 physics validation 이 아니다. 세션의 프레이밍이 이 구분("parity-locked engineering twin" vs "validated physics")을 정직하게 유지하는지 P0/P1 로 평가하라.

---

## 4. 출력 형식

1. Audit target reconstruction (2트랙 대응표: 주장 → 변경 → 게이트/테스트 → 실제 효과)
2. Contract/interface table (RTA 모델·브리지 scatter·게이트 기계)
3. Phys-math audit ledger (항목별 통과/의심/실패 + §1 루프 흔적)
4. Equation-to-code mapping audit (트윈-기준 정합 + firewall 완전성)
5. Numerical/pipeline audit (physics/구현/수치/diagnostics 분류)
6. Ranked failure modes (P0~P3, 최대 7)
7. Verifier results (A/B/C passed/partial/failed)
8. Minimal repair plan (최대 3 패치)
9. Minimal test set (5종 + promotion_check 실증 1, pass/fail 기준)
10. 최종 판정: **치명적 오류 있음 / 부분 통과 / 통과** · 지금 당장 수정할 1개 · 지금 손대면 안 되는 1개

---

## 5. 금지 (HARD)

- 프레임워크 전면 교체·대규모 리팩토링을 기본값으로 제안.
- 실제 bug/contract 문제를 architecture aesthetics 로 덮기.
- 수식 재서술만 하고 코드 path 미확인.
- **parity-to-a-model / self-consistency / smoke test 를 physical validation 으로 인정.** (트윈이 numpy 와 1e-12 로 일치한다 ≠ 충돌 물리가 옳다.)
- **커밋/문서 claim 을 구현 증거로 취급.** ("parity-locked / fail-closed / fixed" 는 직접 실행으로 실증하라.)
- 내부 reviewer(cavecrew/pr-test-analyzer) 통과를 외부 검증으로 간주.
- repo governance("no publication claims")를 취약점 가리는 방패로 오용 — 거꾸로, 어디서 calibrated approximation 이 validated physics 로 over-claim 되는지 적발하라.
- 오류 하나 찾고 조기 종료 (§1 루프 끝까지, 모든 깨진 고리 열거).

---

**한 줄 요약 (감사자용):** 이번 작업은 선행 외부 감사(부분 통과)에 대한 **수정(PR#20–23: claim downscope/firewall, M_angle 성장 재표기, fail-closed 외부앵커 게이트, 내부 extended-range 노드+Friedmann 동어반복 반증)** 과 **충돌-트윈 기반(B4-PR1/2/3: numpy oracle → JAX 연산자 트윈 → JAX 브리지 트윈, 모두 parity-locked + 미분가능)** 이다. 핵심 질문 둘: (1) "수정"이 표현이 아니라 **실효**로 over-claim 을 닫고 게이트가 **실제 fail-closed** 인가; (2) JAX 트윈이 충실히 복제하는 numpy 기준은 **calibrated RTA 모델**이므로 parity 는 calibration 을 상속할 뿐 — 어디서 이 parity 가 **physics validation 으로 위장**되는가. 당신의 임무는 §1 적대적 루프로 이 깨진 고리를 끝까지 찾아내는 것이다.
