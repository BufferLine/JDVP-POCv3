"""Deterministic heuristic baseline track."""

from __future__ import annotations

from typing import Any

from .base import TrackExtractor, TrackOutput


AI_DELEGATION_PHRASES = (
    "what would you choose",
    "you decide",
    "just tell me",
    "what should i do",
    "decide for me",
    "recommend one",
    "recommend for me",
)
EXPLORATION_PHRASES = (
    "compare",
    "options",
    "alternatives",
    "tradeoffs",
)
HUMAN_RETENTION_PHRASES = (
    "i decided",
    "i will decide",
    "i choose",
    "my decision",
)
HEDGING_PHRASES = (
    "maybe",
    "not sure",
    "i guess",
    "probably",
)
HUMAN_LED_DECISION_PHRASES = (
    "need help deciding",
    "help me decide",
    "trying to decide",
)


class HeuristicBaselineTrack(TrackExtractor):
    """Simple offline-safe baseline for M2."""

    track_id = "heuristic_baseline"

    def extract(
        self,
        interaction_id: str,
        turn_number: int,
        human_input: str,
        ai_response: str,
        context_turns: list[dict[str, Any]],
        context_module: str,
    ) -> TrackOutput:
        text = human_input.lower()
        jsv_hint = {
            "judgment_holder": "Shared",
            "delegation_awareness": "Absent",
            "cognitive_engagement": "Reactive",
            "information_seeking": "None",
            "confidence": {
                "judgment_holder": "low",
                "delegation_awareness": "low",
                "cognitive_engagement": "low",
                "information_seeking": "low",
            },
        }
        evidence_spans: list[dict[str, str]] = []
        confidence = 0.68
        notes: list[str] = []

        if any(phrase in text for phrase in HUMAN_LED_DECISION_PHRASES):
            jsv_hint["judgment_holder"] = "Human"
            jsv_hint["delegation_awareness"] = "Explicit"
            jsv_hint["cognitive_engagement"] = "Active"
            jsv_hint["information_seeking"] = "Active"
            evidence_spans.append({"text": human_input[:160], "category": "decision_support_signal"})
            confidence = 0.8
            notes.append("Detected human-led decision support request")

        if any(phrase in text for phrase in EXPLORATION_PHRASES):
            jsv_hint["judgment_holder"] = "Shared"
            jsv_hint["delegation_awareness"] = "Explicit"
            jsv_hint["cognitive_engagement"] = "Active"
            jsv_hint["information_seeking"] = "Active"
            evidence_spans.append({"text": human_input[:160], "category": "exploration_signal"})
            confidence = 0.76
            notes.append("Detected option exploration")

        if any(phrase in text for phrase in AI_DELEGATION_PHRASES):
            jsv_hint["judgment_holder"] = "AI"
            jsv_hint["delegation_awareness"] = "Explicit"
            jsv_hint["cognitive_engagement"] = "Reactive"
            jsv_hint["information_seeking"] = "Passive"
            evidence_spans.append({"text": human_input[:160], "category": "delegation_signal"})
            confidence = 0.82
            notes.append("Detected direct delegation request")

        if any(phrase in text for phrase in HUMAN_RETENTION_PHRASES):
            jsv_hint["judgment_holder"] = "Human"
            jsv_hint["delegation_awareness"] = "Explicit"
            jsv_hint["cognitive_engagement"] = "Active"
            jsv_hint["information_seeking"] = "Passive"
            evidence_spans.append({"text": human_input[:160], "category": "retention_signal"})
            confidence = 0.82
            notes.append("Detected retained human judgment")

        if any(phrase in text for phrase in HEDGING_PHRASES) and jsv_hint["cognitive_engagement"] == "Active":
            jsv_hint["cognitive_engagement"] = "Reactive"
            evidence_spans.append({"text": human_input[:160], "category": "hedging_signal"})
            confidence = max(confidence, 0.72)
            notes.append("Detected hedging language")

        if not evidence_spans:
            evidence_spans.append({"text": human_input[:160], "category": "default_excerpt"})
            notes.append("Default baseline assignment")

        return TrackOutput(
            track_id=self.track_id,
            model_id="rule-based",
            prompt_version="heuristic-v1",
            jsv_hint=jsv_hint,
            evidence_spans=evidence_spans,
            observer_confidence=confidence,
            observer_notes="; ".join(notes),
            raw={
                "context_turn_count": len(context_turns),
                "context_module": context_module,
                "ai_response_excerpt": ai_response[:120],
            },
        )
