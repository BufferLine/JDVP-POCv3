# Prompt Experiment Log

Trial100 데이터셋 (102 items, gemma3:4b generated) 기준 시스템 프롬프트 변형 실험 기록.

비교 모델: gemma3:12b vs gpt-oss:20b (zero-shot, llm_observer track)

## 실험 환경

- 10-sample quick bench (seed=42, 30 turns)로 빠른 반복
- full 102-item run은 최선 후보에만 적용
- `scripts/quick_prompt_bench.py`로 측정

## Baseline: 원본 프롬프트 (rubric 없음)

25줄, 필드명과 허용값만 나열. 판단 기준 없음.

| 필드 | disagreement (full 102) |
|------|------------------------|
| judgment_holder | 88.0% |
| delegation_awareness | 14.0% |
| cognitive_engagement | 1.7% |
| information_seeking | 37.0% |
| **평균** | **35.2%** |

특징: cognitive_engagement 매우 낮지만 judgment_holder 매우 높음.

## v2: 상세 Decision Rubric

각 필드에 "Select when" 테이블, 경계 사례 예시, "Important Principles" 섹션 포함. 약 80줄.

| 필드 | disagreement (10-sample) |
|------|-------------------------|
| judgment_holder | 86.7% |
| delegation_awareness | 93.3% |
| cognitive_engagement | 16.7% |
| information_seeking | 53.3% |
| **평균** | **62.5%** |

결론: **전체 악화**. 상세 설명이 모델별 해석 분기를 넓힘. judgment_holder 변화 없음.

## v3: 간결 가이드라인 (현재 채택)

각 필드에 1줄 설명 + 3-4개 값별 간단 정의. 약 35줄.

| 필드 | disagreement (10-sample) | vs baseline |
|------|-------------------------|-------------|
| judgment_holder | 30.0% | **-58pp** |
| delegation_awareness | 70.0% | +56pp |
| cognitive_engagement | 16.7% | +15pp |
| information_seeking | 46.7% | +10pp |
| **평균** | **40.8%** | **+5.6pp** |

결론: judgment_holder **대폭 개선** (88→30%). delegation_awareness 악화되었으나 이 필드는 프로토콜에서도 observability: low. 전체 평균은 비슷하나 가장 중요한 필드가 개선됨.

## v4: delegation_awareness 조정

v3 기반에서 delegation_awareness에 "this is the expected state for most turns" 추가, Implicit에 "requires concrete textual evidence" 조건 추가.

| 필드 | disagreement (10-sample) | vs v3 |
|------|-------------------------|-------|
| judgment_holder | 50.0% | +20pp 악화 |
| delegation_awareness | 73.3% | +3pp 비슷 |
| cognitive_engagement | 23.3% | +7pp 악화 |
| information_seeking | 36.7% | -10pp 개선 |
| **평균** | **45.8%** | **+5pp 악화** |

결론: delegation은 거의 안 변하고 judgment_holder가 악화. 한 필드 설명이 다른 필드 판단에 간섭.

## Split-axis: 4축 분리 호출

각 필드를 독립된 시스템 프롬프트로 별도 LLM 호출 (4회/턴).

| 필드 | disagreement (10-sample) | vs v3 |
|------|-------------------------|-------|
| judgment_holder | 93.3% | +63pp 악화 |
| delegation_awareness | 96.7% | +27pp 악화 |
| cognitive_engagement | 66.7% | +50pp 악화 |
| information_seeking | 80.0% | +33pp 악화 |
| **평균** | **84.2%** | **+43pp 악화** |

결론: **가장 나쁜 결과**. 필드 간 교차 추론이 사라지면 각 모델이 더 발산. 4축 동시 판단이 일관성에 필수적.

## 핵심 발견

1. **간결할수록 좋다** — 상세한 설명이 오히려 모델별 해석 분기를 넓힘
2. **필드별 1줄 정의가 최적** — 값 목록만(baseline)보다 낫고, 상세 rubric(v2)보다 나음
3. **4축은 함께 판단해야 한다** — 분리 호출 시 교차 추론 상실로 전필드 급악화
4. **judgment_holder가 프롬프트에 가장 민감** — 30~93% 범위로 변동
5. **delegation_awareness는 프롬프트만으로 개선 어려움** — observability: low 필드, 파인튜닝 후보
6. **한 필드 설명 수정이 다른 필드에 간섭** — 프롬프트 전체를 하나의 단위로 취급해야 함

## 향후 실험 방향

- CoT (Chain-of-Thought) 프롬프트: "먼저 전체 대화를 분석한 후 각 필드를 판단하라"
- 파인튜닝: delegation_awareness 같은 low-observability 필드 전용 학습
- Self-consistency: temperature > 0으로 3-5회 샘플링 후 다수결
- 프롬프트 내 인라인 경계 사례 예시 (few-shot과 별개)
