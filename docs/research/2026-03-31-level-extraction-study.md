# JDVP Level-Based Extraction Study

**Date**: 2026-03-31 ~ 2026-04-03
**Authors**: sangwon0001 + Claude Opus 4.6
**Status**: Complete — findings ready for protocol discussion

---

## 1. Research Question

Can LLMs reliably extract JDVP cognitive state measurements from real human-AI conversations?

Specifically:
- Can models agree on judgment delegation levels across turns?
- Can they detect changes (delegation vectors) over time?
- What is the minimum model size and optimal prompt design?
- Is the measurement reproducible?

## 2. Background

JDVP measures 4 dimensions of human cognitive state during AI interaction:
- **Judgment Holder (JH)** — who holds judgment authority (Agency)
- **Delegation Awareness (DA)** — does the human notice delegation occurring (Metacognition)
- **Cognitive Engagement (CE)** — effort invested in the judgment process (Effort)
- **Information Seeking (IS)** — exploration beyond AI's initial offer (Exploration)

Previous work used categorical labels (Human/Shared/AI, Explicit/Implicit/Absent, etc.) with a structured JDVP observer prompt. Best result: gemini-2.5-flash at 64.9% accuracy against synthetic gold labels.

## 3. Key Findings

### 3.1 JDVP Observer Prompt Causes Systematic Bias

The existing JDVP observer prompts (v1, v2, v3) introduce systematic `judgment_holder=Shared` bias across all model sizes.

| Condition | Model | JH=Shared rate |
|---|---|---|
| With JDVP prompt | gemma3-12b | 81-95% |
| With JDVP prompt | gpt-5.4-nano | 91% (v1) |
| With JDVP prompt | gemini-flash | 21-30% |
| Without JDVP prompt (free-form) | claude-sonnet-4 | 7% |
| Without JDVP prompt (free-form) | codex/gpt-5.4 | 9% |
| Without JDVP prompt (free-form) | gemini-3 | 1% |

**Conclusion**: The prompt structure itself induces bias, not model capability. Larger models resist it somewhat but are still affected.

### 3.2 Level-Based (0-10) Scoring Outperforms Categorical

Switching from categorical to continuous 0-10 scoring dramatically improved inter-model agreement on synthetic extreme-change data.

| Metric | Categorical | Level-based |
|---|---|---|
| Inter-model agreement (extreme data) | ~12% direction match | 75-80% direction match |
| Correlation (cloud models) | N/A | r=0.94-0.97 |

### 3.3 Prompt Engineering Impact Varies by Model Size

Six prompt versions tested (v1 through v5, plus v4b):

| Prompt | Tokens | Strategy | Best for |
|---|---|---|---|
| v1 | ~300 | Original JDVP observer | - |
| v2 | ~400 | Simplified level descriptions | Large models |
| v3 | ~950 | Detailed protocol-aligned, examples | Too long for small models |
| v4 | ~377 | Contrastive examples, protocol field names | DA improvement but CE/IS dropped |
| v4b | ~390 | Contrastive examples, v2 field names | **Best overall — all models** |
| v5 | ~398 | v2 prompt, AI response removed | Similar to v2 |

Key findings:
- **Field names matter**: Changing `cognitive_passivity` → `cognitive_engagement` (reversed polarity) confused models
- **Contrastive examples help small models disproportionately**: nano improved from 0.68 to 0.90
- **Prompt length has diminishing returns**: ~400 tokens is optimal; beyond 500 tokens performance degrades for small models
- **AI response removal has no impact**: Removing current-turn AI response (logically correct — human hasn't seen it yet) does not affect performance

### 3.4 Model Size Requirements

Controlled experiment: same prompt (v4b), same code path, same API (OpenRouter), only model varies.

**Extreme synthetic data (40 turns)**:

| Model | Cost/turn | avg r vs sonnet-4 |
|---|---|---|
| gemma4:e4b (local) | $0 | **0.96** |
| gemini-2.5-flash | $0.02 | 0.95 |
| gpt-4.1-mini | $0.01 | 0.91 |
| claude-3.5-haiku | $0.08 | 0.91 |
| gpt-4.1-nano | $0.01 | 0.90 |
| claude-sonnet-4 | $0.30 | reference |

**Key finding**: gemma4:e4b (local, free) matches or exceeds cloud models on this task.

### 3.5 Delegation Awareness is the Hardest Dimension

Consistent across all experiments:

| Dimension | Observability (protocol) | Typical inter-model r | Self-consistency (WildChat) |
|---|---|---|---|
| Information Dependency | high | 0.63 | 0.63 |
| Judgment Delegation | medium | 0.60 | 0.60 |
| Cognitive Passivity | medium | 0.55 | 0.55 |
| Delegation Awareness | **low** | 0.47 | **0.47** |

DA requires detecting "absence of metacognitive signals" — inherently hard from text alone. This aligns with the protocol's own observability rating of "low" for DA.

### 3.6 The Fundamental Limit: Self-Consistency on Real Data

**The most important finding of this study.**

Same model (gemma4:e4b), same prompt (v4b), same data, temperature=1.0, run 3 times:

| Data type | Self-consistency (avg r) |
|---|---|
| Extreme synthetic | **0.97** |
| WildChat real | **0.56** |

The model cannot agree with **itself** on real data. This is not a model problem, prompt problem, or inter-model agreement problem. **The signal in real conversational data is too ambiguous for reliable turn-level measurement.**

### 3.7 Task vs Judgment Delegation Distinction

Models consistently conflate "task delegation" with "judgment delegation":
- "Write me a function" → scored as high delegation (incorrect — task request)
- "What should I choose?" → scored as high delegation (correct — judgment transfer)
- "Solve this riddle" → scored as high delegation (incorrect — game/test)

Adding contrastive examples (v4b) improved this distinction in extreme data but the improvement did not transfer to real data.

## 4. Data and Scripts

### Datasets
- `data/open-data/wildchat/v1/` — 50 interactions, 246 turns (English, multi-turn, non-toxic)
- `data/open-data/wildchat/v2/` — 500 interactions, 2831 turns
- `data/open-data/level-extreme-test/v1/` — 10 synthetic interactions with extreme cognitive state changes

### Scripts Created
- `scripts/download_wildchat.py` — WildChat-1M download and JDVP format conversion
- `scripts/label_open_data.py` — 3-agent parallel labeling with majority vote
- `scripts/label_single_agent.py` — Single-agent LLM labeling via API
- `scripts/label_level_experiment.py` — Level-based (0-10) labeling experiment runner
- `scripts/label_progress.py` — Real-time labeling progress monitor
- `scripts/merge_agent_labels.py` — Majority vote merge for multi-agent labels

### Experiment Results
- `data/silver/level-extreme-test-level-experiment/` — All model × prompt version results on extreme data
- `data/silver/wildchat-level-experiment/` — All model × prompt version results on WildChat
- `data/silver/wildchat-3agent-v1/` — Categorical 3-agent labels (claude + codex + gemini CLI)
- `data/silver/wildchat-v2-level/` — Large-scale level labels (claude 500, codex 500, gemini 375)

## 5. Cost Summary

| Experiment | Method | Cost |
|---|---|---|
| WildChat 3-agent categorical (v1, 50 items) | claude/codex/gemini CLI | CLI subscription |
| WildChat level labels (v2, 500 items) | claude/codex/gemini CLI | CLI subscription |
| OpenRouter controlled experiments | 7 models × extreme + WildChat | ~$3 total |
| Local model experiments | gemma3:12b, gemma4:e4b | $0 |

## 6. Conclusions

1. **Level-based (0-10) scoring is superior to categorical** for inter-model agreement
2. **Prompt design matters significantly for small models** — contrastive examples + correct field names
3. **gemma4:e4b is a viable free local alternative** to cloud models (r=0.96 vs sonnet-4)
4. **DA (delegation awareness) is inherently difficult** — lowest observability, lowest self-consistency
5. **Turn-level measurement on real conversational data has a fundamental reliability limit** — self-consistency r=0.56
6. **The JDVP observer prompt itself was a source of bias** — free-form labeling produced more natural distributions
7. **Extreme/clear delegation patterns are reliably detectable** (r=0.97 self-consistency)
8. **Subtle delegation drift in natural conversation is not yet reliably measurable** at the turn level

## 7. Implications for JDVP Protocol

### Recommended Changes
1. **Adopt level-based (0-10) scoring** as primary measurement, with optional bucketing (Low/Mid/High) for reporting
2. **Rename fields to avoid polarity confusion**: keep `cognitive_passivity` and `information_dependency` (not `engagement`/`seeking`)
3. **Include contrastive examples in observer prompts**: especially for task-vs-judgment distinction
4. **Add observability confidence** per field as a required output, not optional

### Open Questions
1. Can conversation-level (rather than turn-level) measurement improve reliability?
2. Would hand-crafted text features (question length, I/you ratio, imperative detection) complement LLM scoring?
3. Can a harness (cheap model → expensive model fallback) improve cost-effectiveness?
4. Does filtering for "decision-relevant" conversations improve measurement quality?

## 8. Related Literature

Key papers validating that unconscious cognitive delegation to AI is a real phenomenon:
- Steyvers & Kumar (2024) — Three Challenges for AI-Assisted Decision-Making
- Gomez et al. (2025) — Taxonomy of Human-AI Interaction Patterns
- Ibrahim et al. (2025) — Measuring and Mitigating Overreliance
- Buijsman et al. (2025) — Autonomy by Design: cognitive and metacognitive deskilling
- Gerlich (2025) — AI Tools: cognitive offloading and critical thinking
- Barcaui (2025) — ChatGPT as Cognitive Crutch (RCT, effect size d=0.68)
