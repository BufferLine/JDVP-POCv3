"""Trajectory builders and continuity checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrajectoryRecord:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def validate_continuity(vectors: list[dict[str, Any]], interaction_id: str) -> None:
    if not vectors:
        raise ValueError("trajectory requires at least one DV")
    previous_to_turn: int | None = None
    for index, vector in enumerate(vectors):
        if vector["interaction_id"] != interaction_id:
            raise ValueError(f"trajectory vector {index} has mismatched interaction_id")
        if int(vector["from_turn"]) >= int(vector["to_turn"]):
            raise ValueError(f"trajectory vector {index} has non-increasing turns")
        if previous_to_turn is not None and int(vector["from_turn"]) != previous_to_turn:
            raise ValueError(f"trajectory continuity broken at vector {index}")
        previous_to_turn = int(vector["to_turn"])


def build_trajectory(interaction_id: str, vectors: list[dict[str, Any]]) -> TrajectoryRecord:
    validate_continuity(vectors, interaction_id)
    return TrajectoryRecord(
        payload={
            "interaction_id": interaction_id,
            "vectors": vectors,
        }
    )
