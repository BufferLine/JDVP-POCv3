# JDVP 3축 추출 + DA 파생 + 임베딩 증류 연구

**날짜**: 2026-04-06 ~ 04-07
**저자**: sangwon0001
**상태**: 완료

---

## 연구 질문

1. delegation_awareness(DA)를 나머지 3축에서 파생할 수 있는가?
2. 저렴한 로컬 모델로 JDVP 추세를 추적할 수 있는가?
3. 임베딩으로 위임 패턴을 실시간 감지할 수 있는가?

## 주요 발견 요약

### 1. DA 파생 공식

```
DA = 0.162 × JH + 0.570 × CP + 0.268 × ID
```

- CP(인지 수동성)가 DA의 57% 예측 — 수동적일수록 위임 인식 부재
- 소형 모델 자기일관성: 직접 r=0.470 → 파생 r=0.645 (+37%)
- 모델간 일치도: gemma3-12b vs gpt-4.1-mini 직접 r=0.003 → 파생 r=0.613

### 2. Gemini DA 오염 발견

- Gemini가 DA=10을 24.4%에 부여 (claude 0%, codex 0.5%)
- Cloud DA>7 대부분이 탈옥 프롬프트 + gemini 과대평가
- 파생 DA가 "범위가 좁다"가 아니라 "더 정확"

### 3. CoT 프롬프트가 소형 모델 품질 핵심 개선

- "THINK FIRST, then score" → reasoning 먼저 생성 후 점수
- CP 변별력: nano std 0.6 → gemma4:e4b CoT std 1.4 (cloud급)

### 4. 6개 모델 300대화 비교

| 모델 | vs gpt4.1 r | 속도 | 비용 |
|------|------------|------|------|
| sonnet-4 | 0.61 | API | $$ |
| deepseek-v3.2 | 0.58 | API | 저렴 |
| **gemma4:26b** | **0.58** | 0.08t/s | **$0** |
| gpt-5.4-nano | 0.41 | API | 최저 |
| nemotron-120b | 0.20 | API | 저렴 |

gemma4:26b가 $0 로컬로 cloud급 달성 (sonnet과 0.03 차이).

### 5. DV(변화 벡터) 상관이 절대값보다 중요

| 비교 | 절대값 r | **DV 상관** | **대화 추세 일치** |
|------|---------|-----------|----------------|
| sonnet | 0.61 | 0.50 | 64% |
| **gemma4:26b** | 0.58 | **0.53** | **65%** |
| deepseek | 0.58 | 0.36 | 60% |

gemma4:26b가 DV 상관에서 1위. 대화 추세(rising/stable/falling) 65% 일치.

### 6. 임베딩 증류 (LLM → embedding fine-tune)

| 지표 | 원본 | **fine-tuned** |
|------|------|---------------|
| DA R² | 0.187 | **0.386** (2x) |
| CP R² | 0.198 | **0.398** (2x) |
| High DA 감지 F1 | 66.1% | **73.7%** |
| High JH 감지 F1 | 34.9% | **43.7%** |
| 속도 | <5ms | <5ms |

4모델 합의 라벨로 임베딩 fine-tune → R² 2배, F1 +8pp.

### 7. 임베딩의 한계

- 대규모 스크리닝(19K 대화/18초) 가능
- 하지만 "continue please" 수동 패턴을 높은 위임으로 오분류
- 의사결정 위임 vs 태스크 위임 구분 못함
- 스크리닝(1차 필터) 용도로는 가능, 정밀 분석에는 LLM 필요

---

## 실험 상세

### 실험 1: Multi-agent 레벨 융합 (WildChat v2)

- 데이터: 375 interaction, 2,159턴, claude/codex/gemini
- 결과: DA 모델간 일치 극히 낮음 (claude vs codex r=-0.06)
- Z-score calibration: DA std 13.9% 감소, 상관 개선 없음
- Quadrant weighting: JH 명확 시 DA 안정 (autonomous DA std=1.26)

### 실험 2: DA = f(JH, CP, ID) 파생

- Consensus model: DA = 0.114*JH + 0.400*CP + 0.188*ID - 0.197
- 안정성: raw DA mean_std 1.908 → consensus derived 0.877 (54% 감소)
- 비선형 모델(full quadratic R²=0.33)은 실전에서 linear(r=0.645)보다 열화 (과적합)
- 정규화 버전: DA = 0.162*JH + 0.570*CP + 0.268*ID (범위 0-10)

### 실험 3: Gemini DA 오염

- Gemini DA=10: 24.4% (claude 0%, codex 0.5%)
- DA>6 148턴 분류: jailbreak 20, roleplay 1, genuine 36, gemini 과대평가 91
- 2-agent(no gemini)에서 DA>7: 0건 (3-agent: 34건)

### 실험 4: 프롬프트 진화 (ShareGPT 20대화)

| 버전 | JH 분포 | 문제 |
|------|--------|------|
| v1 (원본) | 97% Low | 전부 낮게 |
| v2 (장문) | 67% High | 전부 높게 |
| v3 (간결) | 58% Low / 42% Mid | 중간, High 부족 |
| v3-CoT | 27% Low / 66% Mid / 7% High | 가장 균형 |

핵심 변경: "Who decides the DIRECTION, not who does the work" + "THINK FIRST, then score"

### 실험 5: 6모델 × 300대화 × 4,103턴

- 데이터: ShareGPT 의사결정 큐레이션 300개
- 모델: gpt-4.1, sonnet-4, gpt-5.4-nano, nemotron-120b, deepseek-v3.2, gemma4:26b
- 프롬프트: level_observer_3axis_cot.txt
- 전부 실패 0 (nemotron/deepseek 일부 턴 누락)

### 실험 6: 대화 추세 검증

**만장일치 사례:**
- Rising (37개): 두통 상담 — 정보요청→시도→실패→AI조언 수용 (DA 5→8)
- Falling (15개): Java 테스트 — AI 실패→신뢰 상실→자율 회복 (DA 6→0)
- Stable (65개): 반복 패턴, IELTS 에세이 요청

**불일치 사례 (62개):**
- 대부분 "rising vs stable" 경계선 (slope ±0.1 근처)
- "rising vs falling" 완전 반대 케이스는 거의 없음
- 본질적으로 모호한 대화 (변화가 미미)

### 실험 7: 임베딩 fine-tune + 스크리닝

- Contrastive learning: DA 유사도 기반 pair 8,685개
- 3 epoch fine-tune on MiniLM-L6-v2
- R² 2배 상승, F1 +8pp
- 19K 대화 18초 스크리닝 가능
- 한계: 태스크 위임(continue spam)과 의사결정 위임 혼동

---

## 생성된 자산

### 프롬프트
- `config/prompts/level_observer_3axis_cot.txt` — **최종 채택**
- `config/prompts/level_observer_3axis.txt` — CoT 없는 버전
- `config/prompts/level_observer_4axis.txt` — DA 직접 추출 버전

### 데이터
- `data/open-data/sharegpt/` — 300개 큐레이션 대화, 4,103턴
- `data/silver/sharegpt-3axis-{gpt41,sonnet,deepseek,nemotron,gpt54nano,gemma4-26b}-cot/` — 6모델 라벨
- `data/silver/sharegpt-3axis-{nano,nano-v2,nano-v3,nano-v3b,nano-cot,mini}-*` — 프롬프트 ablation

### 모델
- `models/jdvp-embedding-v1/` — fine-tuned MiniLM-L6-v2

### 스크립트
- `scripts/label_sharegpt_3axis.py` — 범용 라벨링 스크립트

---

## 결론

### 작동하는 것
1. **3축 + DA 파생** — DA 직접 측정보다 안정적이고 정확
2. **CoT 프롬프트** — 소형 모델 변별력 핵심 개선
3. **gemma4:26b 로컬** — $0으로 cloud급 품질 (gpt4.1 r=0.51)
4. **대화 추세 감지** — 4모델 65% 일치, 극단 패턴은 명확
5. **임베딩 증류** — LLM 라벨로 fine-tune하면 R²=0.4, F1=74%

### 한계
1. **턴별 정밀 수치** — 모델간 ±2 차이, 절대값 신뢰 불가
2. **임베딩 단독** — 스크리닝 가능하나 태스크/판단 위임 혼동
3. **데이터 규모** — 300대화로는 임베딩 학습 한계
4. **인간 검증** — gold label이 LLM 합의이지 인간 평가 아님

### 권장 아키텍처

```
[실시간 대화]
     ↓
[임베딩 스크리닝] — <5ms, F1=74%
     ↓ (의심 대화만)
[gemma4:e4b 로컬] — ~2초/턴, 경향 추적
     ↓ (정밀 분석 필요 시)
[gpt4.1 + CoT] — 기준 품질
```

### 다음 단계
1. 임베딩 negative 예시 보강 (태스크 위임 vs 판단 위임 구분)
2. 더 큰 데이터셋으로 임베딩 재학습 (LMSYS-Chat-1M 스크리닝)
3. 인간 평가자 검증 (샘플 50개)
4. 실시간 데모 구현
