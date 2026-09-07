# JDVP POCv3 — V3 Final Benchmark Report

**Date**: 2026-07-16
**Status**: FINAL — frozen (동결). 이 문서의 숫자는 V3 확장 라벨링 자산에 대한 최종 재보고이며, 이후 인용의 기준이다.
**Author**: sangwon0001
**Goal**: 성능 개선이 아니라 **정직한 최종 숫자**. 과거 연구 문서(derived-da-study, level-extraction-study)가 표준 통계 없이 제시했던 수치를 chance-corrected·CI 병기·교차코퍼스로 재보고하고, 방어 불가능한 주장을 명시적으로 철회한다.
**Companion**: [4축 구인 코드북](./2026-07-16-jdvp-4axis-construct-codebook.md)
**Background**: [protocol-completeness-review §4](./2026-07-15-protocol-completeness-review.md), [v1.6 conversation-unit proposal Change 5](./jdvp-protocol-v1.6-conversation-unit-proposal.md)
**Reproduce**: `scripts/research/{task12_agreement_ci,task3_da_crosscorpus,task_f1_detection,task4_prior_ablation}.py` → `docs/research/v3-final/*.json`

---

## 0. 요약 (한 문단)

6개 관측자(gpt-4.1·sonnet-4·deepseek-v3.2·nemotron-120b·gpt-5.4-nano·gemma4:26b)를 ShareGPT 300대화·4,103턴에서 4축으로 재분석했다. **핵심 교정**: (1) 과거의 "추세 65% 일치"는 raw 지표로, chance 보정하면 weighted κ≈0.51·majority-class baseline 51.7% 위 — 즉 **moderate**이지 high가 아니다. (2) 대화 단위 부트스트랩 CI를 붙이면 상위 모델 순위 주장 대부분(48개 인접쌍 중 32개)이 CI 겹침으로 **방어 불가**하며, 특히 "gemma4:26b가 DV 상관 1위(0.53)"는 철회된다 — 상위 3모델은 통계적 동률이다. (3) DA 파생 계수는 순위는 보존하나(관측자 내 r≈0.85) 절대 스케일·계수는 코퍼스·관측자 간 재현되지 않아 **보편 상수가 아니라 재추정 대상**이다. 결론적으로 V3의 방어 가능한 주장은 "대화 추세 불일치는 대부분 stable 경계에 몰린다(부호 반전은 소수)"라는 정성적 사실과 "{sonnet, deepseek, gemma4:26b} 상위 티어 > gpt-5.4-nano > nemotron"의 티어 구조뿐이다 — 추세 일치 자체는 moderate(κ≈0.51)이며 위치보다 압도적으로 낫다고 말할 수 없다.

---

## 1. 데이터와 방법

### 1.1 자산
- **1차 코퍼스 (ShareGPT)**: 의사결정 큐레이션 300대화, 4,103턴. 6관측자 × 3축(JH/CP/ID) 직접 + DA 파생. 프롬프트: `config/prompts/level_observer_3axis_cot.txt`.
  - 커버리지: 300/300 대화(전 모델). 턴 결측: deepseek·nemotron 일부(대화별 존재 턴만 정렬해 사용).
- **DA 교차검증 코퍼스**: WildChat numeric(30대화, 관측자당 160턴, 직접 4축) + level-extreme-test(10대화, 40턴, 직접 4축, 극단 시나리오 = OOD 홀드아웃).
- **prior-anchoring ablation**: ShareGPT 15대화(≥5턴), gemma4:26b 로컬, prior 유/무 두 조건.

### 1.2 표준 지표 3종 (완료 기준)
| 지표 | 대상 | 통계 |
|---|---|---|
| **Spearman ρ** | 턴 단위 절대 레벨(position) | 대화단위 부트스트랩 95% CI |
| **weighted Cohen's κ** | 대화 단위 추세(rising/stable/falling, 서수) | 대화단위 부트스트랩 95% CI |
| **F1** | 고위임 턴 탐지(level ≥ 7, vs gpt-4.1) | 대화단위 부트스트랩 95% CI |

### 1.3 방법 규약 (v1.6 Change 5 준수)
- **추세 정의**: JSV 레벨열의 최소제곱 기울기 + dead zone `|slope| ≤ θ=0.1 → stable`. 최소 3 스냅샷.
- **chance 보정**: raw % 일치는 참고로만, weighted Cohen's κ(선형 가중)·ordinal Krippendorff α와 병기.
- **부트스트랩**: 모든 점추정에 **대화 단위** 리샘플링 95% CI(턴은 대화 내에서 독립이 아님).
- **서수 견고성**: 레벨은 서수 좌표. 등간 산술은 근사이며, 순위 기반 지표(Spearman)를 병기.
- **참조 관측자**: gpt-4.1(고성능 API 프록시). 단일 참조 편향을 피하려 다관측자 α·전 pairwise도 병기.

---

## 2. Task 1 — 추세 일치 재보고 (raw → chance-corrected)

### 2.1 과거 수치의 재현과 교정
과거 "대화 추세 65% 일치"는 **gemma4:26b vs gpt-4.1의 DA 추세 raw pairwise 일치**였다. 재현: raw **67.7%**. 그러나:

- **majority-class baseline (DA 추세) = 51.7%** — 무조건 "stable"로 찍어도 절반 이상 맞는다.
- **weighted Cohen's κ = 0.507** (gemma4:26b), 0.528 (sonnet) — chance 보정하면 **moderate**.
- **6관측자 ordinal Krippendorff α = 0.30** (DA) — fair.

즉 raw 67.7%는 baseline 대비 +16pp에 불과하다. "추세가 신뢰 가능하다"는 주장은 **턴 단위 대비 상대적으로** 참이며(§3), **절대적으로 높은 일치**는 아니다.

### 2.2 추세 일치 — 전 모델 (vs gpt-4.1, DA 차원)
| 관측자 | raw 일치 [95% CI] | weighted κ [95% CI] | 해석 |
|---|---|---|---|
| sonnet-4 | 0.703 [0.653, 0.753] | **0.528** [0.447, 0.612] | moderate |
| gemma4:26b | 0.677 [0.620, 0.727] | **0.507** [0.417, 0.588] | moderate |
| deepseek | 0.627 [0.573, 0.680] | 0.413 [0.324, 0.500] | fair–moderate |
| gpt-5.4-nano | 0.593 [0.537, 0.647] | 0.292 [0.209, 0.378] | fair |
| nemotron | 0.443 [0.387, 0.500] | 0.145 [0.058, 0.235] | slight |

다관측자 ordinal Krippendorff α (6관측자, 차원별): JH 0.32 · CP 0.27 · ID 0.32 · DA 0.30 — 전 차원 **fair**. 4관측자(gpt41·sonnet·deepseek·gemma4-26b) DA 추세 만장일치는 118/300대화(rising 37 / stable 67 / falling 14)로, 과거 보고(117: 37/65/15)와 일치한다.

**추가 교정 — "부호 반전은 near-zero"는 과장이다.** v1.6 제안서는 불일치가 "거의 전적으로" stable 경계 문제이며 직접 모순(rising↔falling)은 "near-zero"라고 적었다. 전 pairwise로 재집계하면: DA 추세 불일치 중 **부호 반전이 all-6에서 18.4%, strong-4에서도 13.7%**다. 즉 불일치의 다수(~82–86%)가 stable 경계에 몰리는 것은 맞지만, 강한 모델 사이에서도 **약 7건 중 1건은 완전한 방향 반전**이다. "경계에 집중"은 유지되나 "모순 거의 없음"은 철회한다.

4축 전체 (불일치 중 부호 반전 비율, all-6 / strong-4):

| 차원 | all-6 | strong-4 | 경계 집중(strong-4) |
|------|-------|----------|---------------------|
| JH | 22.9% (486/2,123) | 17.4% (126/726) | 82.6% |
| CP | 22.8% (471/2,065) | 17.2% (119/690) | 82.8% |
| ID | 18.6% (390/2,095) | 13.5% (96/711) | 86.5% |
| DA | 18.4% (355/1,931) | 13.7% (87/636) | 86.3% |

**DA만 보면 "경계 집중"이 실제보다 강해 보인다.** 파생 축인 DA와 ID는 반전율이 가장 낮은 반면, 직접 관측 축인 JH·CP는 all-6 기준 약 23%로 DA보다 4.5%p 높다. 위 본문의 "~82–86%"는 DA 기준 수치이며, JH·CP에서는 경계 집중이 77% 수준으로 내려간다. 따라서 "불일치는 대부분 경계 문제"라는 정성적 주장은 **차원 의존적**이며, 축을 명시하지 않은 채 인용해서는 안 된다.

---

## 3. Task 2 — 부트스트랩 CI와 순위 주장 철회

### 3.1 전 모델 3지표 통합표 (vs gpt-4.1, 대화단위 95% CI)

**DA (delegation_awareness, 파생) — 위험 신호 축**
| 관측자 | Spearman ρ (position) | weighted κ (trend) | F1 (high≥7 detect) |
|---|---|---|---|
| sonnet-4 | 0.627 [0.593, 0.656] | 0.528 [0.447, 0.612] | 0.450 [0.386, 0.505] |
| deepseek | 0.651 [0.617, 0.682] | 0.413 [0.324, 0.500] | 0.448 [0.385, 0.502] |
| gemma4:26b | 0.609 [0.568, 0.645] | 0.507 [0.417, 0.588] | 0.452 [0.387, 0.514] |
| gpt-5.4-nano | 0.490 [0.451, 0.526] | 0.292 [0.209, 0.378] | 0.269 [0.195, 0.339] |
| nemotron | 0.228 [0.194, 0.265] | 0.145 [0.058, 0.235] | 0.202 [0.163, 0.245] |

(JH/CP/ID 전체 표는 `docs/research/v3-final/task12_results.json` + `task_f1_results.json`. 패턴 동일: 상위 3모델 상호 CI 겹침, gpt54nano·nemotron 하위.)

### 3.2 철회되는 순위 주장
대화단위 부트스트랩 CI 결과, **48개 인접쌍 순위 주장 중 32개(67%)가 CI 겹침 → 방어 불가**. 구체적으로:

- ❌ **철회: "gemma4:26b가 DV 상관 1위($0 로컬로 cloud 능가)"** (derived-da-study §5). DA DV-상관 gemma4:26b 0.553 [0.512, 0.590] vs sonnet 0.528 [0.485, 0.568] vs deepseek 0.512 [0.470, 0.555] — **셋 다 CI 겹침, 통계적 동률**. "1위"는 근거 없음.
- ❌ 추세 weighted κ의 16개 인접쌍 중 15개가 CI 겹침 — 상위 모델 간 추세 능력 순위는 매길 수 없음.
- ✅ **방어 가능(CI 비겹침)**: 상위 티어 {sonnet, deepseek, gemma4:26b} > **gpt-5.4-nano** > **nemotron** 은 대부분 지표에서 유지. nemotron은 전 지표 명확한 최하위.

**정직한 결론**: V3에서 방어 가능한 모델 주장은 순위가 아니라 **티어**다 — 상위 3모델은 서로 구분 불가하며, "$0 로컬 gemma4:26b가 상위 티어에 든다"까지는 참이나 "1위/능가"는 아니다.

---

## 4. Task 3 — DA 파생 계수 교차 코퍼스 검증

### 4.1 전제 교정 (provenance)
과제는 "ShareGPT에서 추정된 계수를 WildChat에서 검증"으로 서술됐으나, 데이터상 **정반대**다: ShareGPT 라벨에는 직접 DA가 없어(파생만 존재) 계수를 추정할 수 없다. 공개 계수 `DA = 0.162·JH + 0.570·CP + 0.268·ID`는 **WildChat numeric 다관측자 라벨에서 회귀**로 얻어진 값이다(derived-da-study Exp 2). 따라서 정직한 검증은 (A) 공개 계수의 전이, (B) WildChat 내 대화단위 k-fold, (C) 별도 코퍼스(extreme-test) 진짜 홀드아웃이다. 각 관측자는 **자기** JH/CP/ID로 **자기** 직접 DA를 재구성한다(관측자 간 잡음 분리).

### 4.2 결과 (관측자별, gemini 계열은 DA 오염 이력으로 제외 표기)
| 관측자 | WildChat 공식 R² | WildChat r | WildChat MAE | extreme-test r | extreme-test MAE | 재적합 계수 (JH,CP,ID) |
|---|---|---|---|---|---|---|
| gpt-4.1 | −0.27 | 0.90 | 2.03 | 0.99 | 0.67 | (0.92, 0.20, −0.30) |
| sonnet-4 | −0.49 | 0.83 | 2.64 | 0.91 | 1.26 | (0.72, 0.03, −0.01) |
| haiku | 0.35 | 0.90 | 1.48 | 0.95 | 0.65 | (0.81, 0.34, −0.26) |
| gpt-4.1-mini | −1.03 | 0.88 | 1.97 | 0.99 | 0.85 | (0.14, 0.77, −0.18) |
| gpt-4.1-nano | 0.67 | 0.85 | 0.98 | 0.54 | 2.18 | (−0.13, 0.19, 0.95) |
| gemma3-12b | −0.48 | 0.12 | 2.81 | 0.71 | 1.61 | (−0.22, −0.05, 0.79) |
| **pooled(clean)** | **−0.06** | **0.58** | **2.03** | — | — | (0.29, 0.06, 0.33) |

> R²는 코퍼스 간 비교 불가(WildChat DA는 저분산, extreme-test는 0–10 전구간). 전이 판단은 **MAE + Pearson r**로 읽는다.

### 4.3 해석 (정직한 결론)
1. **순위(방향)는 전이된다**: 관측자 내 파생–직접 DA 상관 r≈0.83–0.90(약한 관측자 gemma3-12b 제외), 극단 코퍼스에서 0.91–0.99. → 파생 DA를 **상대 신호**로 쓰는 것은 정당.
2. **절대 스케일은 전이되지 않는다**: 공개 계수의 WildChat 재구성 R²는 대부분 음수(pooled −0.06), MAE≈2점. 파생 DA는 "정확한 점수"가 아니라 "대화 내 방향"에만 써야 한다.
3. **CP 지배(0.57)는 재현되지 않는다**: 재적합 계수는 관측자마다 CP 0.03–0.77로 요동하고, pooled 재적합에서 CP는 0.06로 거의 사라진다(대신 JH·ID 지배). → 공개 계수는 특정 consensus의 산물이며 **보편 상수가 아니다**. v1.6 Change 2("informative annex, 버전화, 재추정 가능")를 **강하게 지지**하고, 계수를 규범 상수로 읽는 어떤 해석도 반박한다.

---

## 5. Task 4 — prior 앵커링 ablation

`scripts/label_sharegpt_3axis.py`는 관측자에게 **직전 턴의 JD/CP/ID 점수**를 프롬프트로 전달한다(conditioning). 이 앵커링이 궤적을 평활화해 추세 응집을 부풀리고 변동성을 깎는지(v1.6 Change 5.3의 우려)를 검증했다. 동일 15대화(234턴)를 gemma4:26b 로컬로 **prior 유(prior_scores)**·**무(independent)** 두 조건으로 라벨링하고 대화쌍으로 비교했다. 두 조건은 prior 블록 외 모든 것(맥락 창·프롬프트·온도·상태 추적)이 동일하다 — 순수 앵커링 ablation.

### 5.1 결과 (n=15 대응쌍, 차원별)
| 차원 | 지표 | prior | indep | 차이(prior−indep) | Wilcoxon p |
|---|---|---|---|---|---|
| JH | volatility | 2.415 | 2.588 | −0.173 | 0.303 |
| JH | lag1 autocorr | 0.010 | −0.079 | **+0.088** | **0.030** |
| CP | volatility | 2.592 | 2.592 | 0.000 | 0.890 |
| CP | lag1 autocorr | 0.155 | 0.051 | +0.104 | 0.121 |
| ID | volatility | 2.294 | 2.446 | −0.152 | 0.252 |
| ID | lag1 autocorr | 0.222 | 0.136 | **+0.086** | **0.026** |
| DA | volatility | 2.130 | 2.151 | −0.021 | 0.720 |
| DA | lag1 autocorr | 0.165 | 0.051 | **+0.114** | **0.041** |

추세 라벨 flip rate(조건 간 rising/stable/falling 변경): JH 33% · CP 40% · ID 33% · DA 13%.

### 5.2 해석 (정직한 결론)
- **앵커링 효과는 실재하나 modest하다.** 가장 뚜렷한 증거는 **lag-1 자기상관 상승** — prior 조건에서 전 차원 상승, JH·ID·DA에서 유의(p<0.05). 관측자가 자기 직전 점수를 보면 궤적이 더 매끄럽고 지속적으로 변한다.
- **volatility 하락은 방향은 맞으나(3/4 차원 음수) 작고 비유의**(p 0.25–0.89). n=15에서 volatility 검정은 **underpowered** — "변동성을 깎는다"는 Change 5.3의 표현은 방향은 지지되나 크기는 이 표본에서 확증되지 않는다.
- **추세 라벨은 조건에 민감하다**: 13–40% 대화가 조건에 따라 추세 라벨이 바뀐다. 대화단위 추세 산출물은 conditioning 방식에 실질적으로 의존한다.
- **판단(30대화 확장 여부)**: 효과의 *존재*는 자기상관으로 이미 확증됐고, volatility 효과 크기가 작아 표본을 2배로 늘려도 정성적 결론은 바뀌지 않을 것으로 판단해 **15대화 파일럿으로 동결**한다(검정력 한계 명시).
- **프로토콜 함의**: conditioning은 산출물에 측정 가능한 흔적을 남긴다 → v1.6 Change 5.3(`conditioning: "independent" | "prior_scores"` 기록 의무)을 지지한다. 변동성 의미가 중요한 용도(오실레이션 탐지 등)에는 **independent 라벨링을 권장**한다. 전면 금지(v1.6 Open Q7)까지는 이 표본이 근거를 주지 못한다 — 효과가 modest하기 때문이다.

_데이터: `docs/research/v3-final/task4_results.json`. 라벨: `data/silver/sharegpt-3axis-ablation-{prior_scores,independent}/`._

---

## 6. 한계 (명시)

1. **골드가 인간이 아니다**: 모든 참조는 LLM 관측이다(gpt-4.1). 구인 타당도(수렴/변별)는 미검증(완성도 검토 §2.1). "심리학 자동화" 주장에는 인간 준거 상관이 필요하다.
2. **표본 규모**: DA 교차검증의 WildChat numeric은 30대화, extreme-test는 10대화(관측자당). prior-anchoring ablation은 15대화(volatility 검정 underpowered). 이들 결론은 방향성 증거이지 정밀 추정이 아니다.
3. **관측자 오염**: gemini 계열 DA=10 과대부여(24.4%)는 헤드라인에서 제외했으나 과거 consensus 계수에는 영향을 줬을 수 있다.
4. **파생 DA의 순환성**: DA를 JH/CP/ID로 파생하면 "무의식적 위임" 사분면은 구성상 CP의 함수가 된다 — 메타인지 주장에는 자기보고 검증이 별도로 필요.
5. **서수를 등간처럼 다룸**: 기울기·평균은 근사. Spearman 병기로 완화했으나 레벨의 등간성은 미검증.
6. **turn 천장**: 자기일관성 r≈0.56은 개선되지 않았다 — 대화 단위 집계가 이를 우회하는 것이지 해결하는 것이 아니다.

---

## 7. 동결되는 주장 (freeze)

**참이라 말할 수 있는 것:**
- 상위 관측자 간 대화 추세 일치는 **moderate**(weighted κ≈0.51, 6관측자 α≈0.30)이며, 불일치의 ~82–86%가 stable/rising 경계에 몰린다 — 즉 추세는 "대략의 방향"으로는 쓸 수 있으나 **높은 일치가 아니다**.
- 관측자 **티어**: {sonnet-4, deepseek, gemma4:26b} > gpt-5.4-nano > nemotron. 상위 3은 통계적 동률.
- 파생 DA는 직접 DA의 **순위**를 잘 보존한다(관측자 내 r≈0.85+); 절대 스케일은 아니다.
- prior-score conditioning은 궤적을 **평활화**한다(자기상관 유의 상승, JH·ID·DA). 효과는 modest하나 실재하므로 **기록·공개해야 한다**(v1.6 Change 5.3).

**더 이상 말하지 않는 것 (철회):**
- ~~"추세 65% 일치 = 높은 신뢰도"~~ → baseline 51.7% 위 moderate(κ≈0.51).
- ~~"추세 불일치는 경계뿐, 부호 반전 near-zero"~~ → strong-4에서도 반전 13.7%.
- ~~"대화 추세가 턴 위치보다 훨씬 신뢰 가능"~~ → 지표를 맞춰 비교하면 둘 다 moderate(위치 Spearman ρ 상위모델 0.6+). 방어 가능한 것은 "부호 반전이 소수(경계 집중)"라는 정성적 사실뿐.
- ~~"gemma4:26b가 DV 상관 1위 / cloud 능가"~~ → 상위 3 동률.
- ~~"DA=0.162·JH+0.570·CP+0.268·ID (CP가 57%를 예측)"를 보편 법칙으로~~ → 코퍼스·관측자 특수, 재추정 대상.

---

*재현 스크립트·산출 JSON: `scripts/research/`, `docs/research/v3-final/`. 근거 문서: derived-da-study(2026-04-06), level-extraction-study(2026-03-31), protocol-completeness-review(2026-07-15), v1.6 proposal.*
