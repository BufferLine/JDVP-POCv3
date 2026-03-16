"""Track factory for M1/M2 pipeline use."""

from __future__ import annotations

from .base import TrackExtractor
from .cheap_ml_baseline import create_env_backed_cheap_ml_track
from .fewshot_prompt import create_env_backed_fewshot_track
from .fixture_hint import FixtureHintTrack
from .heuristic_baseline import HeuristicBaselineTrack
from .llm_observer import create_env_backed_llm_track


def create_track(track_name: str) -> TrackExtractor:
    if track_name == "fixture_hint":
        return FixtureHintTrack()
    if track_name == "heuristic_baseline":
        return HeuristicBaselineTrack()
    if track_name == "llm_observer":
        return create_env_backed_llm_track()
    if track_name == "fewshot_prompt":
        return create_env_backed_fewshot_track()
    if track_name == "cheap_ml_baseline":
        return create_env_backed_cheap_ml_track()
    raise ValueError(f"unknown track: {track_name}")
