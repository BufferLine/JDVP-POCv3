"""Track adapter for fixture-provided JSV hints."""

from __future__ import annotations

from typing import Any

from .base import TrackExtractor, TrackOutput


class FixtureHintTrack(TrackExtractor):
    """Use `meta.jsv_hint` from the input fixture directly."""

    track_id = "fixture_hint"

    def extract(
        self,
        interaction_id: str,
        turn_number: int,
        human_input: str,
        ai_response: str,
        context_turns: list[dict[str, Any]],
        context_module: str,
    ) -> TrackOutput:
        raise RuntimeError("fixture_hint track is driven by per-turn meta.jsv_hint and cannot be called directly")

    def extract_from_turn(self, interaction_id: str, turn: dict[str, Any], context_module: str) -> TrackOutput:
        meta = turn.get("meta", {})
        hint = meta.get("jsv_hint")
        if hint is None:
            raise ValueError(f"turn {turn['turn_number']} missing meta.jsv_hint")
        return TrackOutput(
            track_id=self.track_id,
            model_id="fixture",
            prompt_version="fixture-v1",
            jsv_hint=hint,
            evidence_spans=[
                {
                    "text": turn.get("human_input", "")[:160],
                    "category": "fixture_hint_source",
                }
            ],
            observer_confidence=1.0,
            observer_notes="Canonical fixture hint",
            raw={"context_module": context_module},
        )
