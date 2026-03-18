"""Run storage helpers for resumable extraction jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared_utils import utc_now as _utc_now


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class StoredTurnExtract:
    jsv_hint: dict[str, Any]
    overlay_row: dict[str, Any]
    extract_record: dict[str, Any]


class RunStorage:
    """Filesystem-backed run storage with checkpoint snapshots."""

    def __init__(self, run_dir: Path, track_name: str) -> None:
        self.run_dir = run_dir
        self.track_name = track_name
        self.extract_root = run_dir / "extracts" / track_name
        self.checkpoint_path = run_dir / "checkpoints" / "progress.json"

    def extract_path_for_turn(self, turn_number: int) -> Path:
        return self.extract_root / f"turn-{turn_number}.json"

    def has_extract_for_turn(self, turn_number: int) -> bool:
        return self.extract_path_for_turn(turn_number).is_file()

    def load_extract_for_turn(self, turn_number: int) -> StoredTurnExtract:
        payload = read_json(self.extract_path_for_turn(turn_number))
        return StoredTurnExtract(
            jsv_hint=payload["jsv_hint"],
            overlay_row=payload["overlay"],
            extract_record=payload,
        )

    def store_extract_for_turn(
        self,
        *,
        interaction_id: str,
        turn_number: int,
        human_input: str,
        ai_response: str,
        context_module: str,
        jsv_hint: dict[str, Any],
        overlay_row: dict[str, Any],
        track_metadata: dict[str, Any],
        raw_track_output: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "interaction_id": interaction_id,
            "turn_number": turn_number,
            "track_name": self.track_name,
            "context_module": context_module,
            "human_input": human_input,
            "ai_response": ai_response,
            "jsv_hint": jsv_hint,
            "overlay": overlay_row,
            "track_metadata": track_metadata,
            "raw_track_output": raw_track_output,
            "stored_at": _utc_now(),
        }
        write_json(self.extract_path_for_turn(turn_number), payload)
        return payload

    def update_checkpoint(
        self,
        *,
        interaction_id: str,
        total_turns: int,
        completed_turns: int,
        resumed_turns: list[int],
        written_turns: list[int],
        status: str,
    ) -> dict[str, Any]:
        payload = {
            "interaction_id": interaction_id,
            "track_name": self.track_name,
            "total_turns": total_turns,
            "completed_turns": completed_turns,
            "resumed_turns": resumed_turns,
            "written_turns": written_turns,
            "status": status,
            "updated_at": _utc_now(),
        }
        write_json(self.checkpoint_path, payload)
        return payload
