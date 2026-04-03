# JDVP Protocol Enhancement Proposal: Level-Based Scoring

**Proposal ID**: JDVP-2026-001
**Date**: 2026-04-03
**Author**: sangwon0001
**Status**: Draft — for review before PR to JDVP-protocol
**Evidence**: `docs/research/2026-03-31-level-extraction-study.md`

---

## Summary

This proposal recommends adopting continuous 0-10 level scoring as the primary measurement format for JDVP's four core dimensions, based on empirical evidence from a multi-model extraction study.

## Motivation

The current JDVP v1 specification uses categorical labels:
- `judgment_holder`: Human | Shared | AI | Undefined
- `delegation_awareness`: Explicit | Implicit | Absent
- `cognitive_engagement`: Active | Reactive | Passive
- `information_seeking`: Active | Passive | None

Our experiments revealed three problems with categorical measurement:

### Problem 1: Systematic Prompt Bias
The JDVP observer prompt induces `judgment_holder=Shared` bias across all model sizes (81-95% for small models, 21-30% for large models). This bias is absent with level-based scoring.

### Problem 2: Low Inter-Model Agreement
With categorical labels, inter-model agreement on turn-level changes was 12% (near random). With level-based scoring on the same data, agreement rose to 75-80%.

### Problem 3: Boundary Ambiguity
Categorical labels force hard boundaries (e.g., "Is this Shared or AI?") that models interpret inconsistently. Level scoring allows gradation and reduces boundary effects.

## Proposed Changes

### Change 1: Add Level-Based Scoring as Primary Format

Each JSV variable gains a continuous 0-10 score alongside (or replacing) the categorical value:

```json
{
  "judgment_holder": 7,
  "delegation_awareness": 8,
  "cognitive_engagement": 3,
  "information_seeking": 5
}
```

Scale anchors (per dimension):

**judgment_holder (0-10)** — Agency
- 0 = Human fully owns judgment authority
- 5 = Shared judgment, co-deciding
- 10 = Human has fully delegated judgment to AI
- Note: Task requests (code, translate) are NOT judgment delegation

**delegation_awareness (0-10)** — Metacognition
- 0 = Human explicitly aware of delegation boundary
- 5 = Ambiguous, no metacognitive signals
- 10 = No awareness that judgment has shifted

**cognitive_engagement (0-10)** — Effort
- 0 = Deep deliberation, questioning, reasoning
- 5 = Moderate engagement
- 10 = Passive acceptance without processing

**information_seeking (0-10)** — Exploration
- 0 = Actively seeking alternatives beyond AI's offer
- 5 = Open but not actively seeking
- 10 = Accepts first answer, no exploration

### Change 2: JH × DA Matrix as Interpretive Guide

The protocol already identifies the JH × DA matrix as the core signal. This should be explicitly included in observer guidance:

```
JH high + DA low  = Conscious delegation (human chose to delegate)
JH high + DA high = Unconscious delegation (JDVP core concern)
JH low  + DA low  = Autonomous decision-making
JH low  + DA high = Unexamined retention
```

### Change 3: Measurement Timing Clarification

JSV should be measured at the moment of human input, BEFORE the AI response to that input:

- Turn N measurement = human's cognitive state when they wrote turn N
- Context available = all previous turns (human + AI pairs)
- NOT available = current turn's AI response (human hasn't seen it yet)

### Change 4: DV Computation from Levels

Delegation Vectors become simple arithmetic:

```
DV = JSV(turn_n) - JSV(turn_n-1)
```

No ordinal mapping needed. Positive = movement toward delegation/passivity. Negative = movement toward autonomy/engagement.

### Change 5: Observability Confidence

The study confirmed the protocol's observability hierarchy empirically:

| Dimension | Protocol Rating | Measured Self-Consistency |
|---|---|---|
| information_seeking | high | r=0.63 |
| judgment_holder | medium | r=0.60 |
| cognitive_engagement | medium | r=0.55 |
| delegation_awareness | low | r=0.47 |

Recommendation: Require implementations to report per-dimension confidence, weighted by these empirical observability limits.

## Backward Compatibility

Level scores can be mapped to existing categorical values:

```
0-3 → first category (Human / Explicit / Active / Active)
4-6 → middle category (Shared / Implicit / Reactive / Passive)
7-10 → last category (AI / Absent / Passive / None)
```

Existing categorical data remains valid. Level scoring is an extension, not a replacement.

## Evidence Summary

| Finding | Extreme Data | Real Data (WildChat) |
|---|---|---|
| Inter-model agreement (categorical) | ~12% direction | ~12% direction |
| Inter-model agreement (level, r) | 0.94-0.97 | 0.31-0.36 |
| Self-consistency (same model, level) | **0.97** | **0.56** |
| Best local model (gemma4:e4b) | r=0.96 vs sonnet | — |
| Prompt optimization effect (small models) | +0.22 (nano) | — |

### Key Limitation

Turn-level measurement on natural conversational data has a self-consistency ceiling of r≈0.56. This is a property of the task, not the model or prompt. The protocol should acknowledge this limitation and consider:
1. Conversation-level aggregation as a more reliable unit
2. Filtering for "decision-relevant" turns
3. Reporting confidence intervals rather than point estimates

## Implementation Notes

### Recommended Observer Prompt (~400 tokens)
See `scripts/label_level_experiment.py` prompt version v4b for the empirically validated prompt template.

### Minimum Model Requirements
- Cloud: gpt-4.1-nano ($0.01/turn) or above
- Local: gemma4:e4b (9.6GB, 0.6 turns/sec, $0)
- Below 4B parameters: unreliable

### Cost for Large-Scale Labeling
- 1000 turns via gpt-4.1-mini: ~$0.25
- 1000 turns via gemma4:e4b local: $0
- 3-model ensemble (recommended): ~$0.75 or $0 (local)

## Open Questions for Protocol Discussion

1. Should level scoring replace or complement categorical? (This proposal suggests complement with eventual transition)
2. Should the protocol define "decision-relevant turns" as a filtering criterion?
3. How should conversation-level aggregation be standardized?
4. Should the self-consistency limit (r≈0.56 on natural data) be documented as a known measurement bound?
