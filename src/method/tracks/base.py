"""Base interfaces for extraction tracks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrackOutput:
    """Normalized output from one extraction track for one turn."""

    track_id: str
    model_id: str
    prompt_version: str
    jsv_hint: dict[str, Any]
    evidence_spans: list[dict[str, str]] = field(default_factory=list)
    observer_confidence: float | None = None
    observer_notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_overlay_record(self, interaction_id: str, turn_number: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "interaction_id": interaction_id,
            "turn_number": turn_number,
            "track": {
                "track_id": self.track_id,
                "model_id": self.model_id,
                "prompt_version": self.prompt_version,
            },
            "jsv": self.jsv_hint,
            "evidence_spans": self.evidence_spans,
        }
        if self.observer_confidence is not None:
            payload["observer_confidence"] = self.observer_confidence
        if self.observer_notes:
            payload["observer_notes"] = self.observer_notes
        return payload


class TrackExtractor:
    """Interface for all method-layer extraction tracks."""

    track_id: str

    def extract(
        self,
        interaction_id: str,
        turn_number: int,
        human_input: str,
        ai_response: str,
        context_turns: list[dict[str, Any]],
        context_module: str,
    ) -> TrackOutput:
        raise NotImplementedError
