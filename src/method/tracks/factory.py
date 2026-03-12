"""Track factory for M1/M2 pipeline use."""

from __future__ import annotations

from .base import TrackExtractor
from .fixture_hint import FixtureHintTrack
from .heuristic_baseline import HeuristicBaselineTrack


def create_track(track_name: str) -> TrackExtractor:
    if track_name == "fixture_hint":
        return FixtureHintTrack()
    if track_name == "heuristic_baseline":
        return HeuristicBaselineTrack()
    raise ValueError(f"unknown track: {track_name}")
