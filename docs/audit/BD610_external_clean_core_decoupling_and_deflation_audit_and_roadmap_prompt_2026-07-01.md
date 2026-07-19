# [INTEGRATED PHYS-MATH-CODE AUDIT + ROADMAP — RABBIT Clean-Core Decoupling Driver + AP65 Deflation, Adversarial]

**버전:** 2026-07-01 · **대상 브랜치:** `feature/bianchi-i-full-nonperturbative` · **감사 형태:** 제3자 독립 적대적 감사 + 후속 계획 제안
**선행:** `docs/audit/BD601_*`(충돌-트윈/수정 감사) 및 세션 내부 근본원인 노트 `docs/audit/BD205_qlaguerre_collision_root_cause_2026-07-01.md`. 이번 대상은 그 이후 랜딩된 **(트랙 C) clean dynamic-collision core + FLRW 탈결합 드라이버(PR-C1…C7)** 와 **(트랙 D) AP65 audit 트랙 대량 삭제(PR-D1/D2, −183.7K LOC)** 이다.

---

## 0. 감사자에게 (READ FIRST)

당신은 이 repo에 **직접 접근 가능한 독립 외부 감사자**다. 별도 zip/스냅샷 없음 — `git log`, 소스, 테스트, `docs/audit/*.md`, `.superpowers/sdd/progress.md`(이번 세션 ledger)를 직접 읽고 필요 시 직접 실행하라:

```
env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/pytest -q -p no:cacheprovider <test>
env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -c "..."
venv/bin/python scripts/promotion_check.py            # 클레임 게이트 상태(green/red)
```
(ROCm "No AMD GPUs" 경고는 무해한 CPU fallback.)

**Source of truth 규칙:** 진실 우선순위 **(1) 코드 실제 실행 출력 > (2) 명세 수식 / `docs/audit/*.md` > (3) 커밋 메시지/문서 claim**. 커밋이 "fixed / conserved / physical / verified"라 해도 코드 path가 그걸 생산하지 않으면 커밋이 틀린 것이다. **self-consistency / conserve-by-construction / parity-to-analytic-limit 를 절대 physical validation 으로 인정하지 마라** — 외부 기준(FRW 탈결합 표준 코드 FortEPiaNO/Grohs/Mangano의 N_eff·T_ν/T_γ, 독립 충돌적분, 해석적 한계)만이 validation이다.

**이번 감사의 핵심 경고 3가지:**
- **(a) "에너지 보존을 by construction 으로 했다"는 것은 두 얼굴이다.** 보존은 옳지만 — 보존되는 그 양이 **물리적으로 틀린 충돌률**일 수 있다. 플라즈마가 잃는 에너지를 중성미자가 얻는 에너지와 **정의상 같게** 만들면 보존 테스트는 통과하지만, 두 양 모두 동일한 quadrature 오차를 공유하면 N_eff 는 self-consistent 하게 틀릴 수 있다. **보존(conservation) 과 정확도(accuracy) 를 절대 동일시하지 마라.**
- **(b) N_eff 근접(3.0488 ≈ SM 3.044)을 validation 으로 위장하는 곳을 적발하라.** 이 clean core 는 **외부 앵커가 없다**(삭제된 AP65 트랙과 동일하게 Bianchi-I+충돌항 논문 부재). 붕괴된 N_eff 가 표준값에 가깝다는 사실이 코드/문서/커밋 어디서 "검증됨"의 근거로 쓰이면 그것이 over-claim.
- **(c) 대량 삭제(−183.7K LOC)가 capability 를 조용히 축소했는지 적발하라.** 삭제된 것은 **비-LRS 강전단 + full-BBN 핵합성 네트워크 + weak-network solve** 를 목표하던 (audit-only) 트랙이다. clean core 는 현재 **FLRW 탈결합 only** 다. 프로젝트가 실제로 **덜 할 수 있게** 되었는데, 어떤 문서/claim/config 가 여전히 존재하지 않는 non-LRS/full-BBN 능력을 광고하는지 찾아라.

**목표:** 새 아이디어(감사 파트) 금지. 주어진 수정/드라이버/삭제/게이트 **사이의 깨진 고리**를 찾아라. 그런 다음 §6 에서 **다음 계획을 제안**하라(이 절만 창의성 허용, 단 measurement-driven·no-over-claim 준수).

---

## 1. 의무 적대적 검증 루프 (MANDATORY — 감사 STEP 0–8 모든 항목에 적용)

각 항목마다 아래 7단 루프를 **명시적으로** 수행하고 흔적을 출력에 남겨라. 한 단계라도 생략하면 그 항목은 미완료.

1. **Self-Discover** — 이 항목 검증에 필요한 추론 모듈을 먼저 스스로 구성(어떤 불변량/한계/대조가 결정적인가). 즉답 금지.
2. **Step-Back** — 코드로 들어가기 전 지배 원리로 추상화("이건 frame-bookkeeping 문제인가, conservation-vs-accuracy 문제인가, coverage-loss 문제인가, over-claim 문제인가?").
3. **Metacognitive Self-Ask** — "저자의 'conserved/physical/verified' 프레이밍을 그대로 빌려 쓰고 있지 않나? 내가 놓칠 법한 것은?"
4. **CoVe** — 결론에 대해 독립 검증 질문 3~5개를 생성, 각각 원 추론과 분리해 답한 뒤 불일치를 찾아라. (예: "conserve-by-construction 이면, rhs 를 z=1.4(탈결합 후반)에서 직접 호출해 ν-gain vs plasma-loss 를 재계산했을 때 여전히 1e-10 인가? 그리고 그 gain 이 **정확한** z⁴·(∫q³C dq) 와 몇 % 다른가? 직접 실행.")
5. **Adversarial Self-Ask** — 잠정 결론을 반증하려 시도. "N_eff=3.0488 을 틀리게 만드는 n_q/rtol/프레임 설정은? 삭제가 놓친 runtime-broken 테스트는? '보존됨'이 가리는 틀린 충돌률 시나리오는?"
6. **CCoT** — "올바른 구현 vs 흔한 오구현" 대조(comoving-frame vs thermal-frame advection, conserve-by-construction vs accurate-transfer, 1/(2π²) vs 1/π², collect-only-clean vs lazy-import-broken).
7. **PDR** — 판정 후 한 번 더 리뷰: 증거가 결론을 지탱하는가, 더 싼 판별 테스트가 있는가, 심각도가 과대/과소인가.

> 규칙: 오류 하나 찾았다고 종료하지 마라. 루프를 끝까지 돌려 **모든** 깨진 고리를 열거하라.

---

## 2. 감사 순서 (STEP 0–8) — 이 작업에 특화

### STEP 0. AUDIT TARGET RECONSTRUCTION
아래 2트랙을 분리 재구성하고 "주장 → 변경 파일 → 게이트/테스트 → 실제 효과" 대응표를 만들어라.

- **트랙 C — clean dynamic-collision core + 탈결합 드라이버:**
  - PR-C1 `a53fda5` — 근본원인 pin + 불변량 계약(`tests/test_qlaguerre_collision_conditioning.py`, `docs/audit/BD205_*`). 주장: 붕괴 원인은 `dT_νx/dN = −T_νx + dQ/(c_v∝T_νx³)` 의 1/T³ 증폭기이고, dQ 모멘트 자체는 정확.
  - PR-C2…C6 `87464ee`/`cc4bb5c`/`3fe94f6`/`a8835dd`/`d83d88b` — `src/rabbit/collisions/dynamic_collision_core.py`: 에너지변수 3T 열역학(`coupled_3T_energy_rhs`), logit 분포(`neutrino_distribution_from_modes`), first-principles 충돌 에너지원(`neutrino_collision_energy_transfer` — `deterministic_reference` 재사용), 뱅크 조립(`collision_bank_energy_sources_per_efold`), 결합 FLRW RHS(`flrw_dynamic_collision_rhs`).
  - PR-C7 `3f7704f` + 수정 `f6edb33` — `src/rabbit/collisions/dynamic_collision_driver.py`(신규) + `tests/test_dynamic_collision_driver.py`. 주장: comoving 프레임 탈결합 적분이 **BBN 종점 도달**, collisionless **N_eff=2.9934**(n_q 불변), collision-on **N_eff=3.0488**, T_νx/T_γ=0.715(붕괴 없음), **에너지 by-construction 보존**, solver **fail-closed**. 수정 f6edb33 은 (i) 보존 테스트가 tautological 이었음(양변이 동일 thermal 모멘트), (ii) `sol.success`/종점 미가드로 실패 시 N_eff=11.55 garbage 반환 — 두 CRITICAL 을 닫았다 주장.
- **트랙 D — AP65 deflation(−183.7K LOC, repo ~307K→121,058):**
  - PR-D1 `25b414b`(validation/ap65 audit leaves −145,230, 77파일) + `2c6f2ef`(inference+jax −14,332).
  - PR-D2 `3b89248`(weak + transport weak-network/collision-bridge −23,546; `weak/__init__`+`transport/__init__` re-export 절단; mixed nonlrs_collisionless 테스트 편집) + `f9e1ff2`(마지막 lazy-import mirror fb71 −588).
  - **B5 carve-out(보존):** `transport/augmented_{nonlrs_transport,typeI_nonlrs_collisionless,typeI_observables}.py` + `augmented_pstf_distribution.py`(생존 substrate) + 그 테스트. FLAG-C(`build_non_lrs_s2_grid`) 자동 해소, FLAG-D(`test_standard_bbn_anchor` = ap65 내부 → 삭제) 적용.

각 트랙에서 source of truth가 무엇인지, **변경이 production path에 도달하는지 / 테스트-전용인지**, 그리고 **삭제가 제거한 것이 무엇인지**(코드+테스트 커버리지)를 명시.

### STEP 1. CONTRACT / INTERFACE AUDIT (먼저 복원; 불명확하면 그 자체로 P0/P1)
- **프레임 계약(comoving):** state = comoving f_ν(Y) on fixed Y=q_nodes, plus T_γ; z≡a·T_γ, a_init=1/T_γ,init ⟹ z_init=1, 열적 q=y/z. massless ⟹ df/dN|_y = C[f]/H (redshift-advection 없음). 종점 = T_γ<1e-2 MeV. **이 프레임이 왜 옳은지, thermal-q 프레임이 왜 틀린 N_eff 를 주는지**를 복원.
- **N_eff readout:** `N_eff = (11/4)^{4/3}/z_final⁴·(I_nue/I_FD + 2·I_nux/I_FD)`, I_pair=∫Y³f dY, I_FD=7π⁴/120. collisionless 에서 I=I_FD·(frozen), z_final=(11/4)^{1/3} ⟹ N_eff=3. **이 유도를 독립 재검**.
- **플라즈마 단위환산 PHYS = T_γ⁴/(2π²):** `collision_bank_energy_sources_per_efold` 의 dQ=(∫q³C dq)/H 는 무차원 모멘트(MeV) — 플라즈마 채널 `coupled_3T_rhs_from_collision_moments` 는 물리 MeV⁴ 를 소비. ρ_pair=(1/2π²)∫p³(f+f̄)dp ⟹ 인자 (1/2π²)T_γ⁴. 드라이버는 **df-유도 gain G=∫Y³ df dY** 를 (T_γ⁴/z⁴)/(2π²)·(2G_nue+4G_nux) 로 플라즈마에 먹인다(= by-construction 보존). **1/(2π²) vs 1/π²(2× g-중복셈)의 유도를 직접 재현하고, "by-construction"이 실제로 무엇을 남기는지 규정.**
- **first-principles 충돌(`deterministic_reference.py`):** `evaluate_nue_scattering_reference`/`evaluate_pair_annihilation_reference`: prefactor=`G_F²·T⁴/(4π³)`, C=prefactor·Σ(weights·M²·S)/q². **이 C 가 진짜 df/dt [MeV] 인지 차원 확인**(G_F²T⁴=무차원 ⟹ C 무차원?인데 코드는 C/H 를 df/dN 로 씀 — T⁴ vs T⁵ prefactor 문제를 직접 검증). detailed-balance null, heating 부호, ν_x<ν_e 계약.
- **fail-closed 가드:** `reached_endpoint = sol.success AND t_events[0].size>0`; 미도달 시 RuntimeError. clip/H_safe excursion 관측치(`max_clip_excursion`, `min_hubble_MeV`) 계약.

### STEP 2. PHYS-MATH AUDIT
순서: 정의/표기 → 인덱스/측도 → 부호/정규화 → 단위/차원 → known limit → 경계/정칙성 → 숨은 가정 → 반례. 각 통과/의심/실패 + 이유 + 코드 영향. 특화 표적:
- **conservation ≠ accuracy(최우선):** 드라이버는 ν-gain 을 interp #2(`_map_thermal_field_to_comoving` = `np.interp(Y/z, q_nodes, C_th)` → Laguerre 모멘트 on Y)로 계산한다. 저자 스스로 이 **round-trip 상대오차가 populated interior 에서 ~10–45%**(`test_interp_roundtrip_bounds_error`, band [0.05,2.0])라 명시. 정확한 이송은 해석적으로 z⁴·(∫q³C dq)/H 다. **plasma 에 먹인 G 가 이 정확값과 몇 % 다른지 in-run z∈[1,1.40] 전 구간에서 측정하고, 그 오차가 collision-on N_eff=3.0488 에 얼마의 편향을 넣는지 정량화**하라. "n_q 수렴(3.068/3.073/3.075@24/32/40)"이 **정확한 값으로의 수렴인지, 아니면 self-consistent-but-biased 값으로의 수렴인지** 판별(정확 이송을 쓴 대체 드라이버와 대조).
- **collisionless N_eff=2.9934 의 0.22% undershoot:** 저자는 "finite-μ QED 플라즈마 EOS(z_final 0.05% high)" 탓이라 주장. **독립 재계산** — z_final 을 `_plasma_temperature_base_rhs` 엔트로피 적분으로 직접 구해 (11/4)^{1/3} 대비 오차를 재현하고, 그것이 정말 EOS 인지 아니면 프레임/quadrature 편향인지 판별. (n_q 불변이라 grid 아님은 저자 주장 — 검증.)
- **충돌 정규화 물리성:** `deterministic_reference` 의 Γ/H 가 실제 ν_e 탈결합(~1.5 MeV 부근 Γ/H~1)을 재현하는가. C_rate 류 calibration 상수가 있는가(없다면 진짜 first-principles). 탈결합 온도가 물리적인가 — 아니면 prefactor(T⁴ vs T⁵) 오류로 결합이 과/소평가되어 N_eff 편향?
- **degeneracy 부기:** ν_e pair=2, ν_x bank=4(BANK_DEGENERACY[NUX]). 드라이버의 (2G_nue+4G_nux)와 모듈 dQ 의 ×4 가 이중계상 없이 정확히 일치하는가.

### STEP 3. EQUATION-TO-CODE MAPPING AUDIT
- comoving↔thermal 이중 보간(interp #1: f를 q_nodes·z 로; interp #2: C를 Y/z 로)이 코드에서 프레임 혼선 없이 구현됐나. state flatten `[f_nue, f_nux, T_gamma]` 인덱싱.
- **보존 테스트가 실제 code path 를 시험하나:** `test_energy_conserving_plasma_coupling` 이 `_make_rhs(collisions=True)` 의 rhs 를 z≠1 에서 호출해 반환 df 로부터 gain 을, dT_γ 로부터 loss 를 독립 산출하는가(f6edb33 의 수정 주장). **그러나 by-construction 이면 이 테스트도 다시 tautological 아닌가** — 무엇을 실제로 constrain 하는지(프레임 인자·부호·degeneracy·dT_base 분리) 규정하고, 그 이상은 보증 못 함을 명시.
- **삭제가 production path 를 건드리지 않았나:** 캐논 경로(`forward_likelihood`, `driver_typeI_char`, `full_coupled_typeI`, 6개 jax 모듈, `pstf_*`)가 삭제 모듈을 import 하지 않음을 직접 재확인. `transport/__init__`·`weak/__init__` 편집 후 re-export 된 심볼을 소비하는 live 코드가 정말 없는지 whole-repo re-grep.

### STEP 4. NUMERICAL / PIPELINE AUDIT
- Radau/LSODA 5자리 일치, rtol 1e-6→1e-9 에서 N_eff 불변 주장 재현. stiff C/H~500@10MeV. clip f∈[0,1] 이 **불리한 상태(f>1 Pauli 위반)를 조용히 봉합**해 붕괴 증상을 가리지 않는지(가드 관측치가 정말 ~0 인지).
- **삭제 검증의 사각지대:** 저자는 `--collect-only` 게이트가 **함수-본문(lazy) import** 를 놓친다는 것을 fb71 에서 발견했다 명시. **당신이 직접** 전체 스위트를 대상으로 삭제된 모듈명(validation/ap65/jax/weak/transport-deleted)에 대한 **문자열·importlib·lazy import** 를 재-sweep 하여 **또 다른 runtime-broken 또는 조용히 skip 되는 테스트**를 찾아라. 전체 `pytest -q -p no:cacheprovider tests/`(또는 최소 `-m "not slow"`)를 실제 실행해 red 개수를 세라.
- **보존된 B5 island 가 dangling stub 인가:** `augmented_nonlrs_transport`(연산자-레벨 realizability)는 **드라이버-레벨 통합이 없다**(operator-level frozen-Σ slice only, `test_b5_*` docstring 명시). clean core 는 이 연산자를 쓰지 않는다. 이 island 가 (a) 여전히 의미있는 게이트인지, (b) 아니면 죽은 코드인지, (c) `build_non_lrs_s2_grid` 가 canonical `test_pstf_collision_contractions` 에 정말 필요한 유일 substrate 인지 판별.

### STEP 5. FAILURE MODE SYNTHESIS
failure mode 최대 7개: 유형(physics/math/implementation/numerical/interface/testing/coverage-loss) · 심각도(P0–P3) · 관측 증상 · 근본 원인 · 가장 싼 판별 테스트 · 오해석 위험.

### STEP 6. VERIFIER FILTER
A. Physics(comoving-frame 정합/known-limit N_eff=3/충돌 정규화/conservation≠accuracy) · B. Code(fail-closed 실증/삭제 import 건전성/보존테스트 비순환성) · C. Numerical(interp 편향/n_q 수렴 의미/lazy-import sweep). 각 passed/partial/failed.

### STEP 7. MINIMAL REPAIR PLAN
대규모 리팩토링 금지. 최소 수정 패치 **최대 3개**: 무엇을·왜 load-bearing·어떤 failure mode 차단·새 테스트·regression 영향.

### STEP 8. MINIMAL TEST SET
즉시 실행 가능: baseline 재현 1(collisionless N_eff=3) · edge/adversarial 1(정확 이송 대비 collision-on N_eff 편향) · physics sanity 1(Γ/H 탈결합 온도) · numerical 1(lazy-import 전체 sweep + 전체 스위트 red 카운트) · regression 1(삭제 후 캐논+게이트 green). 각 pass/fail 기준.

---

## 3. 우선 의심 표적 (여기를 먼저 공격하라)

> 아래는 이번 세션이 스스로 "fixed / conserved / physical / green"이라 주장한 항목이다. **저자 결론과 내부 reviewer(cavecrew/pr-test-analyzer/silent-failure-hunter) 통과를 빌리지 말고 §1 루프로 독립 재검증하라.**

1. **"에너지 by-construction 보존" — 정확도를 가리는가.** 최우선. 플라즈마 loss ≡ ν gain 이 정의상 성립하므로 보존 테스트는 항상 통과한다. 진짜 질문: **먹인 G(interp #2)가 물리적으로 옳은 충돌 에너지 이송인가.** 정확 이송 z⁴·(∫q³C dq)/H 과 in-run 전 구간에서 대조하고, collision-on N_eff 를 (i) 현재 드라이버, (ii) 정확 이송 드라이버 두 방식으로 산출해 차이를 보고하라. 차이가 크면 N_eff=3.0488 은 self-consistent-but-wrong.
2. **collisionless N_eff=2.9934 — 3.000 이 아닌 이유.** 0.22% 를 EOS 로 돌린 주장을 z_final 독립 재계산으로 검증. n_q∈{16,24,32,48} 불변 주장 재현. 만약 EOS 가 아니라 프레임/모멘트 편향이면 P1.
3. **first-principles 충돌의 실체 — 진짜 QKE-free first-principles 인가, 숨은 정규화 있는가.** `deterministic_reference` prefactor `G_F²T⁴/(4π³)` 차원(C가 rate[MeV]인가) + M² 정규화 + detailed-balance 를 직접 검증. Γ/H 가 표준 ν 탈결합(≈1.5 MeV)을 재현하는지 — 안 하면 결합 세기 오류가 N_eff 편향을 낳는다.
4. **1/(2π²) vs 1/π² — g-중복셈.** 인자를 첫 원리에서 유도해 드라이버가 옳은 쪽을 쓰는지, 그리고 spike 가 쓴 1/π²(2×)가 정말 틀렸는지 확인.
5. **fail-closed 가드 — 실제로 red 를 던지나.** `integrate_flrw_decoupling(collisions=False, N_span_max=0.5)` 를 직접 실행해 RuntimeError 를 실증(과거 N_eff=11.55 garbage). solver 실패/종점 미도달 두 경로 모두 fail-closed 인가.
6. **삭제 건전성 — 놓친 dangling/coverage-loss.** (a) 삭제 모듈에 대한 lazy/string import 를 전 테스트 재-sweep(fb71 유형). (b) `promotion_check.py`·`claim_gates.py` 의 `required_test_node_ids` 가 **삭제된 테스트 노드를 가리켜 거짓 green** 이 되지 않는지 — 직접 실행. (c) 삭제된 `test_standard_bbn_anchor`/full-BBN span ladder/M_angle·realizability 계열이 **계약 게이트였는지**(있었다면 coverage 손실 P1).
7. **capability over-claim after deletion.** 삭제 후에도 `config/backend_capabilities.py`·`feature_capabilities.py`(및 `test_augmented_pstf_capability_registry` 가 잠근 문자열)가 **존재하지 않는 non-LRS/full-BBN/weak-network AP65 능력**을 여전히 catalog 하는가. `test_augmented_wbs_status_ledger` 가 참조하는 WBS 문서가 삭제된 작업을 "완료/가용"으로 표기해 stale 한가. clean core 가 **FLRW 탈결합 only** 임을 문서/claim 이 정직히 반영하는지 P0/P1.
8. **B5 island — 살아있는 게이트인가 죽은 stub 인가.** 연산자-레벨 realizability 만 남고 드라이버 통합이 없다. 보존이 정당한지(certified gate) 아니면 삭제 대상이었어야 하는지, 그리고 `augmented_pstf_distribution`(clean core 의존)만이 진짜 필수 substrate 인지.
9. **구조적 잔여(불변):** clean core 의 충돌은 `deterministic_reference` 의 monopole(isotropic) 2-to-2 충돌적분이다 — **비등방(강전단) 충돌·비-LRS·full-BBN 네트워크는 삭제됐고 clean core 엔 없다.** 유한전단 비등방은 여전히 외부 앵커 없음(B8). 세션 프레이밍이 "FLRW 탈결합 블로커를 고쳤다(내부 일관성)" vs "Bianchi-I full-BBN 을 검증했다" 를 정직히 구분하는지 P0 로 평가.

---

## 4. 출력 형식
1. Audit target reconstruction (2트랙 대응표: 주장 → 변경 → 게이트/테스트 → 실제 효과 + 삭제가 제거한 것)
2. Contract/interface table (comoving 프레임·PHYS 인자·deterministic_reference 충돌·fail-closed 가드)
3. Phys-math audit ledger (항목별 통과/의심/실패 + §1 루프 흔적; conservation≠accuracy 정량 포함)
4. Equation-to-code mapping audit (프레임 정합 + 보존테스트 비순환성 + 삭제 import 건전성)
5. Numerical/pipeline audit (interp 편향·n_q 의미·lazy-import 전체 sweep·전체 스위트 red 카운트)
6. Ranked failure modes (P0~P3, 최대 7)
7. Verifier results (A/B/C passed/partial/failed)
8. Minimal repair plan (최대 3 패치)
9. Minimal test set (5종, pass/fail 기준; promotion_check 실증 1 포함)
10. 감사 최종 판정: **치명적 오류 있음 / 부분 통과 / 통과** · 지금 당장 고칠 1개 · 지금 손대면 안 되는 1개
11. **§6 로드맵 제안(아래) — 감사와 분리해 별도 절로.**

---

## 5. 금지 (HARD)
- 프레임워크 전면 교체·대규모 리팩토링을 감사 결론으로 제안(로드맵 §6 에서만 계획 제안 허용).
- 실제 bug/contract 문제를 architecture aesthetics 로 덮기.
- 수식 재서술만 하고 코드 path 미확인.
- **conserve-by-construction / self-consistency / n_q-수렴 / parity-to-analytic-limit 를 physical validation 으로 인정.** (에너지가 1e-10 로 보존된다 ≠ 충돌률이 옳다; N_eff 가 3.044 에 가깝다 ≠ 검증됨.)
- **커밋/문서 claim 을 구현 증거로 취급.** ("fixed / conserved / fail-closed / green" 은 직접 실행으로 실증.)
- 내부 reviewer 통과를 외부 검증으로 간주.
- repo governance("no external anchor / internal consistency only")를 취약점 가리는 방패로 오용 — 거꾸로, 어디서 internal-consistency 가 validated-physics 로 over-claim 되는지, 어디서 삭제가 capability 를 조용히 축소했는지 적발하라.
- 오류 하나 찾고 조기 종료(§1 루프 끝까지, 모든 깨진 고리 열거).

---

## 6. 로드맵 제안 요청 (감사 완료 후, 별도 절)

감사(§1–§5, STEP 0–8)를 **먼저** 끝낸 뒤, 그 발견을 근거로 **다음 작업 계획**을 제안하라. 창의성 허용하되 measurement-driven·no-over-claim 을 준수하고, 각 항목을 아래 형식으로:

> [우선순위 P0–P3] · [작업명] · [왜 지금(어떤 감사 발견/gap 이 이걸 요구하나)] · [수용 기준(내부 불변량 + 있으면 외부 앵커 전략)] · [예상 위험/함정] · [대략 규모(파일/LOC/PR 수)]

**반드시 다루어야 할 후보 축(당신의 감사 발견이 우선순위를 정한다):**
1. **충돌률 정확도** — interp #2 편향을 없애는 방식(정확 이송으로 plasma 결합 + 분포 shape 재정규화, 또는 spectral 미분 advection 을 쓰는 thermal 프레임). collision-on N_eff 를 **정확 이송**으로 재확정.
2. **first-principles 충돌 정규화 검증** — `deterministic_reference` 의 Γ/H 를 독립 계산/문헌 탈결합률과 대조해 prefactor(T⁴ vs T⁵)·M² 정규화를 확정. 이것이 확정돼야 N_eff 의 "small positive" 가 신뢰된다.
3. **비-LRS / 강전단으로의 clean-core 확장** — 삭제된 AP65 비-LRS 능력을 **깨끗하게 재구축**하는 최소 경로. 보존된 B5 island(operator-level realizability)를 드라이버에 통합할지, 아니면 clean core 위에 새로 세울지.
4. **full-BBN 핵합성 네트워크 결합** — 탈결합 solve 가 정확해진 뒤 network 를 bolt-on 하는 인터페이스(Yp/DH 종점). 삭제된 span-ladder 가 하던 역할의 **정직한 대체**.
5. **외부 앵커 전략** — Bianchi-I+충돌항 논문은 없다(수용). 그러나 **FLRW 극한**에서 clean core 를 FortEPiaNO/Grohs/Mangano 의 N_eff·T_ν/T_γ·spectral distortion 과 대조하는 **fail-closed 외부 게이트**(BD601 의 `finite_shear_anchor` 패턴)를 세울 수 있는가. 이것이 clean core 를 "internal-only" 에서 "FLRW-externally-anchored" 로 올리는 유일 경로.
6. **삭제 후 정합화** — capability/config/WBS 문서를 삭제된 능력에 맞게 정직화(§3.7). config 이중화(`backend_capabilities` vs `feature_capabilities`, `jax_params` vs `jax_config`) 통합.
7. **governance 강화** — clean core 의 N_eff/T_ν 를 promote 하려면 어떤 fail-closed 게이트가 필요한가(외부 앵커 없이는 "physical" 이상으로 promote 금지).

로드맵은 **순서·의존성**을 명시하라(무엇이 무엇을 gate 하나). 감사에서 P0/P1 을 찾았다면 그 수정이 로드맵의 선행 항목이어야 한다.

---

**한 줄 요약 (감사자용):** 이번 작업은 (C) q-Laguerre 붕괴 블로커를 comoving-frame 탈결합 드라이버로 고쳤다는 주장(collisionless N_eff=2.9934, collision-on 3.0488, 붕괴 없음, 에너지 by-construction 보존, fail-closed)과 (D) 그 붕괴의 본거지였던 AP65 audit 트랙 −183.7K LOC 삭제(repo 307K→121K, B5 island 보존)다. 핵심 질문 셋: (1) **보존(conservation)이 정확도(accuracy)를 가리는가** — 먹인 interp #2 gain 이 정확 이송과 10–45% 다를 수 있으므로 N_eff 가 self-consistent-but-wrong 인가; (2) first-principles 충돌 정규화(prefactor 차원)와 collisionless N_eff 의 0.22% 가 물리인가 버그인가; (3) −183.7K 삭제가 **capability 를 조용히 축소**했는데(비-LRS/full-BBN/weak-network 소멸, clean core 는 FLRW-only) 어떤 문서/config/claim 이 여전히 존재하지 않는 능력을 광고하는가. §1 적대적 루프로 깨진 고리를 끝까지 찾고, 그 뒤 §6 로드맵을 제안하라. **internal-consistency 를 validated-physics 로, conserve-by-construction 을 correct-rate 로 위장하는 지점을 절대 놓치지 마라.**
