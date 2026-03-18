"""Track adapter for fixture-provided JSV hints."""

from __future__ import annotations

from typing import Any

from .base import TrackExtractor, TrackOutput


class FixtureHintTrack(TrackExtractor):
    """Use `meta.jsv_hint` from the input fixture directly.

    Unlike other tracks, this track needs access to the full turn dict (including
    ``meta.jsv_hint``).  Call ``set_turns()`` before ``extract()`` so the track
    can look up the current turn's metadata.
    """

    track_id = "fixture_hint"

    def __init__(self) -> None:
        self._turns_by_number: dict[int, dict[str, Any]] = {}

    def set_turns(self, turns: list[dict[str, Any]]) -> None:
        """Register the full interaction turns list for hint lookup."""
        self._turns_by_number = {int(t["turn_number"]): t for t in turns}

    def extract(
        self,
        interaction_id: str,
        turn_number: int,
        human_input: str,
        ai_response: str,
        context_turns: list[dict[str, Any]],
        context_module: str,
    ) -> TrackOutput:
        current_turn = self._turns_by_number.get(turn_number)
        if current_turn is None:
            raise ValueError(
                f"turn {turn_number} not found; call set_turns() before extract()"
            )
        return self.extract_from_turn(interaction_id, current_turn, context_module)

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
