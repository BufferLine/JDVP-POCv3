# Project Analysis Report (2026-03-24)

5개 병렬 분석 에이전트 (Protocol, Track Quality, Pipeline Ops, Test Coverage, Research Methodology)의 종합 결과.

## 배경

trial100 zero-shot 실험 결과를 기반으로 프로젝트 전반의 부족한 부분을 분석.

- 데이터셋: 102개 합성 인터랙션 (gemma3:4b 생성)
- 모델: gemma3:12b, gpt-oss:20b, qwen3.5:35b (zero-shot llm_observer)
- 핵심 수치: 모델 간 평균 35.2% disagreement (judgment_holder 88%, cognitive_engagement 1.7%)

---

## 1. Protocol & Schema

### 핵심 문제

시스템 프롬프트(`llm_observer_system.txt`)가 25줄 형식 규정일 뿐, 각 enum 값의 의미론적 판단 기준을 전혀 제공하지 않음. 이것이 judgment_holder 88% 불일치의 직접 원인.

### 상세 발견

- `judgment_holder`의 Human vs Shared 경계가 정의되지 않음 — 관찰 가능한 행동이 아니라 추론적 상태
- `cognitive_engagement` 1.7% 불일치 — Active/Reactive/Passive가 텍스트 표면 패턴과 높은 상관관계를 가져 암묵적 합의가 내장됨
- `information_seeking`의 `None` 값이 JSON null과 혼동 위험
- 스키마 검증은 구조적으로 건전하나, 필드 간 의미적 일관성 검증 부재 (예: judgment_holder=AI + delegation_awareness=Absent는 논리적 모순)
- `Undefined` 포함 DV는 delta=null이 되어 trajectory 분석에서 정보 손실

### 권장사항

1. **시스템 프롬프트에 필드별 decision rubric 추가** — 노력: 낮, 영향: 높
2. **필드 간 cross-field 의미적 일관성 제약 추가** — 노력: 중, 영향: 중
3. **프롬프트에 경계 사례 인라인 예시 2-3개 포함** — 노력: 낮, 영향: 높
4. `information_seeking`의 `None`을 `Absent`로 변경 검토 — 프로토콜 breaking change

---

## 2. Extraction Track Quality

### 핵심 문제

heuristic baseline이 거의 모든 경우 기본값을 반환하여 유의미한 비교 하한선이 되지 못함. LLM observer 프롬프트가 의미 정의 없이 형식만 규정.

### 상세 발견

- heuristic의 순차적 if 체인(elif 아님)에서 마지막 매칭이 이전 결과를 무조건 덮어쓰는 우선순위 문제
- heuristic이 human_input만 분석하고 ai_response를 무시 — judgment_holder 판단에 AI 응답 성격이 필수적
- 약 20개 phrase만 존재하여 자연어 변형 대부분을 놓침
- LLM 정규화에서 필드 조합 의미적 일관성 미검증
- evidence_spans category가 비표준화되어 트랙 간 비교 불가
- 앙상블 동률 해소가 알파벳순으로 체계적 편향 (AI가 항상 Human에 우선)
- cheap_ml_baseline의 bag-of-words가 화행(speech act) 구분에 본질적으로 부적합

### 권장사항

1. **시스템 프롬프트에 프로토콜 의미 정의 + 경계 사례 가이드 추가** — 노력: 낮, 영향: 높
2. **앙상블에 confidence 가중치 + 서수 거리 불일치 도입** — 노력: 중, 영향: 중
3. **heuristic에 ai_response 분석 + regex 패턴 확장** — 노력: 중, 영향: 중
4. **self-consistency 샘플링 (temperature > 0, 3-5회)** — 노력: 중, 영향: 중
5. **cheap_ml을 TF-IDF + 로지스틱 회귀로 업그레이드** — 노력: 중-높, 영향: 낮-중

---

## 3. Pipeline & Operations

### 핵심 문제

관측성(observability)이 거의 전무. `src/` 전체에서 logging을 import하는 파일이 2개뿐이고, 실행 시간 측정이 0건.

### 상세 발견

- 카탈로그 upsert 실패 시 런 추적 유실 (try/except 없음)
- "running" 상태 좀비 런 감지 불가 (list_failed_runs 기본 필터가 "failed"만)
- rerun 후 dataset_run 카운터 미갱신
- dataset_runs 테이블에 model_id 컬럼 없어 모델별 조회 불가
- UNIQUE 제약 부재로 중복 런 삽입 가능
- write_json이 비원자적 (쓰기 도중 종료 시 불완전 JSON)
- resume 시 context_module/meta 변경 미감지
- 모델 크기에 따른 timeout 조정 불가 (환경변수 프로세스 단위)

### 권장사항

1. **관측성 기반 구축** — per-turn 레이턴시, 진행률 출력, 카탈로그 duration_seconds — 노력: 중, 영향: 높
2. **모델 인식 timeout 전략** — matrix config에 모델별 timeout, 또는 크기 기반 프리셋 — 노력: 낮, 영향: 높
3. **벤치마크 정합성 검증** — disagreement 0%/100% 시 sanity warning — 노력: 낮, 영향: 높
4. **카탈로그 무결성** — UNIQUE 인덱스, model_id 컬럼, dataset_run_id에 model 포함 — 노력: 중, 영향: 중
5. **원자적 파일 쓰기** — tempfile + os.replace 패턴 — 노력: 낮, 영향: 중
6. **좀비 런 감지** — `--status running --older-than` 필터 — 노력: 낮, 영향: 낮

---

## 4. Test Coverage

### 핵심 문제

87개 테스트 대부분이 "파일 존재 + 키 존재" 수준. run_key 충돌 버그(자기 비교 → 0% 불일치)가 전체 테스트를 통과한 근본 원인은 의미적 정확성 assertion이 없기 때문.

### 상세 발견

- 유일한 compare_runs() 테스트가 서로 다른 track_name만 사용 — 동일 track 다중 모델 비교 미테스트
- `disagreement_rate > 0` 같은 값 범위 assertion이 거의 없음
- resume 입력 변경 감지, 빈 데이터셋 슬라이스, zero-shot 폴백 등 치명적 경로 미테스트
- heuristic 기본값 변경 시 5개 파일 수정 필요 — 기대값이 하드코딩

### 미테스트 치명적 경로

| 경로 | 위험도 |
|------|--------|
| 동일 track_name 다중 실행 비교 | Critical |
| resume 시 입력 변경 감지 | Critical |
| 빈 데이터셋 슬라이스 처리 | High |
| fewshot zero-shot 폴백 메타데이터 | High |
| 1턴 인터랙션 (0 DV) | High |
| non-general context DV extensions | High |

### 권장사항

1. **compare_runs() 전용 단위 테스트** — 동일 track 비교, 키 충돌 없음 검증 — 노력: 중, 영향: Critical
2. **의미적 정확성 assertion 패턴** — `disagreement_rate > 0`, resume 불변성 등 — 노력: 중, 영향: 높
3. **경계 조건 테스트** — 1턴, 빈 슬라이스, 비연속 turn_number 등 — 노력: 중, 영향: 높
4. **Property-based 테스트** (hypothesis) — 노력: 높, 영향: 높
5. **기대값 fixture 파일** — 하드코딩 대신 중앙화된 기대값 — 노력: 높, 영향: 중
6. **pytest-cov 도입** — branch coverage 측정 — 노력: 낮, 영향: 중

---

## 5. Research Methodology

### 핵심 문제

Gold annotation이 완전히 부재하여 정확도 측정 불가. 합성 데이터 생성과 평가가 모두 LLM에 의존하여 순환 의존. 시나리오 다양성이 3개로 극히 제한.

### 상세 발견

- 생성 프롬프트가 jsv_hint를 "hard behavioral target"으로 사용 → 추출 모델이 "맞추는" 것은 생성 모델의 hint 표현 충실도를 간접 측정하는 것
- v1 시나리오 3개가 모두 동일 trajectory 패턴 (Human→Shared→AI) → 역방향/steady-state/edge case 없음
- 프롬프트 ablation 실험 없음 (단일 프롬프트)
- 모델 자체 분산(self-agreement) 미측정 (temperature=0 고정)
- confusion matrix 없어 "어떤 경계가 모호한지" 분석 불가
- few-shot selector가 동일 시나리오의 다른 sample을 제외하지 않음 (데이터 오염)
- 통계적 유의성 검정 없음 (백분율만 보고)

### 35% 불일치율 해석

- 주관적 분류 태스크 기준 합리적 범위 (감정 분석 IAA ~20-40% 불일치)
- 그러나 judgment_holder 88%는 태스크 정의 불명확 신호
- cognitive_engagement 1.7%는 패턴 예측 가능성 신호

### 출판 수준으로의 갭

| 리뷰어 예상 지적 | 현재 상태 | 필요 대응 |
|-----------------|----------|----------|
| gold label 없이 정확도 주장 불가 | gold 없음 | 50건 인간 annotation + IAA |
| 3개 시나리오로 일반화 불가 | 3개, 1 trajectory | 10+개, 3+ trajectory |
| 프롬프트 ablation 없음 | 단일 프롬프트 | 3-5 프롬프트 변형 |
| 통계적 유의성 없음 | 백분율만 | kappa, bootstrap CI |
| 재현 가능성 부족 | seed 있으나 RNG 버그 이력 | deterministic 검증 |

### 권장사항

1. **jsv_hint 대비 정확도 메트릭 추가** — hint-vs-extraction 비교 함수 — 노력: 낮, 영향: 높
2. **50건 인간 annotation 파일럿** — 2명 annotator, IAA 측정 — 노력: 중, 영향: 최고
3. **시나리오 다양성 확대** — 10+개, 역방향/steady-state/edge case — 노력: 중, 영향: 높
4. **Confusion matrix + kappa 통계** — 노력: 낮, 영향: 중
5. **프롬프트 ablation 실험** — 3-5 변형 — 노력: 중, 영향: 높

---

## 우선순위 종합 (5개 분석 교차)

| 순위 | 작업 | 노력 | 영향 | 근거 |
|------|------|------|------|------|
| 1 | 시스템 프롬프트에 decision rubric 추가 | 낮 | 최고 | 3/5 분석 공통 1순위 |
| 2 | 50건 인간 annotation + IAA 측정 | 중 | 최고 | research methodology 필수 |
| 3 | compare_runs() 단위 테스트 + 동일 track 비교 검증 | 중 | 높 | 치명적 버그 재발 방지 |
| 4 | 파이프라인 관측성 (timing, progress, sanity check) | 중 | 높 | 운영 가시성 확보 |
| 5 | 시나리오 다양성 확대 (10+개, 3+ trajectory) | 중 | 높 | 실험 일반화 기반 |
| 6 | 앙상블 고도화 (confidence 가중, 서수 거리) | 중 | 중 | 분석 품질 향상 |
| 7 | confusion matrix + kappa 통계 도입 | 낮 | 중 | 결과 해석 엄밀성 |
| 8 | heuristic baseline에 ai_response 분석 추가 | 중 | 중 | 유의미한 비교 하한선 |
| 9 | 원자적 파일 쓰기 (write_json) | 낮 | 중 | 데이터 무결성 |
| 10 | 프롬프트 ablation 실험 설계 | 중 | 높 | 프롬프트 민감도 파악 |
