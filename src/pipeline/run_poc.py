"""Smoke pipeline for M1 core validation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from src.method.tracks.factory import create_track
from src.method.tracks.fixture_hint import FixtureHintTrack
from src.pipeline.run_storage import RunStorage, write_json, write_jsonl
from src.protocol_core.dv_ordinal import build_dv_sequence
from src.protocol_core.jsv_types import build_jsv_from_hint
from src.protocol_core.schema_validate import CanonicalSchemaValidator
from src.protocol_core.trajectory import build_trajectory


def _load_input(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_jsv_sequence(
    raw_interaction: dict[str, Any],
    track_name: str,
    storage: RunStorage,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[int], list[int]]:
    interaction_id = str(raw_interaction["interaction_id"])
    context_module = str(raw_interaction.get("context_module", "general"))
    jsv_sequence: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []
    extract_rows: list[dict[str, Any]] = []
    resumed_turns: list[int] = []
    written_turns: list[int] = []
    track = create_track(track_name)
    fixture_track = track if isinstance(track, FixtureHintTrack) else None
    for turn in raw_interaction["turns"]:
        meta = turn.get("meta", {})
        turn_number = int(turn["turn_number"])
        if resume and storage.has_extract_for_turn(turn_number):
            stored = storage.load_extract_for_turn(turn_number)
            jsv = build_jsv_from_hint(
                interaction_id=interaction_id,
                turn_number=turn_number,
                timestamp=meta.get("timestamp") or turn.get("timestamp"),
                context_module=context_module,
                hint=stored.jsv_hint,
            )
            jsv_sequence.append(jsv.to_dict())
            overlay_rows.append(stored.overlay_row)
            extract_rows.append(stored.extract_record)
            resumed_turns.append(turn_number)
            storage.update_checkpoint(
                interaction_id=interaction_id,
                total_turns=len(raw_interaction["turns"]),
                completed_turns=len(jsv_sequence),
                resumed_turns=resumed_turns,
                written_turns=written_turns,
                status="running",
            )
            continue
        context_turns = raw_interaction["turns"][: int(turn["turn_number"])]
        if fixture_track is not None:
            track_output = fixture_track.extract_from_turn(interaction_id, turn, context_module)
        else:
            track_output = track.extract(
                interaction_id=interaction_id,
                turn_number=turn_number,
                human_input=str(turn.get("human_input", "")),
                ai_response=str(turn.get("ai_response", "")),
                context_turns=context_turns,
                context_module=context_module,
            )
        jsv = build_jsv_from_hint(
            interaction_id=interaction_id,
            turn_number=turn_number,
            timestamp=meta.get("timestamp") or turn.get("timestamp"),
            context_module=context_module,
            hint=track_output.jsv_hint,
        )
        jsv_sequence.append(jsv.to_dict())
        overlay_row = track_output.to_overlay_record(interaction_id, turn_number)
        overlay_rows.append(overlay_row)
        extract_rows.append(
            storage.store_extract_for_turn(
                interaction_id=interaction_id,
                turn_number=turn_number,
                human_input=str(turn.get("human_input", "")),
                ai_response=str(turn.get("ai_response", "")),
                context_module=context_module,
                jsv_hint=track_output.jsv_hint,
                overlay_row=overlay_row,
                track_metadata=track_output.track_metadata(),
                raw_track_output=track_output.raw,
            )
        )
        written_turns.append(turn_number)
        storage.update_checkpoint(
            interaction_id=interaction_id,
            total_turns=len(raw_interaction["turns"]),
            completed_turns=len(jsv_sequence),
            resumed_turns=resumed_turns,
            written_turns=written_turns,
            status="running",
        )
    return jsv_sequence, overlay_rows, extract_rows, resumed_turns, written_turns


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "workspace-local"
    return result.stdout.strip()


def run_poc(
    input_path: Path,
    run_id: str,
    output_root: Path,
    protocol_schema_root: Path | None = None,
    track_name: str = "fixture_hint",
    resume: bool = False,
) -> Path:
    raw_interaction = _load_input(input_path)
    validator = CanonicalSchemaValidator(protocol_schema_root=protocol_schema_root)
    run_dir = output_root / run_id
    storage = RunStorage(run_dir=run_dir, track_name=track_name)

    jsv_sequence, overlay_rows, extract_rows, resumed_turns, written_turns = _build_jsv_sequence(
        raw_interaction,
        track_name=track_name,
        storage=storage,
        resume=resume,
    )
    dv_sequence = [record.to_dict() for record in build_dv_sequence(jsv_sequence)]
    trajectory = build_trajectory(str(raw_interaction["interaction_id"]), dv_sequence).to_dict()

    for jsv in jsv_sequence:
        validator.validate_jsv(jsv)
    for dv in dv_sequence:
        validator.validate_dv(dv)
    validator.validate_trajectory(trajectory)

    write_json(run_dir / "input" / input_path.name, raw_interaction)
    for jsv in jsv_sequence:
        write_json(run_dir / "canonical" / "jsv" / f"turn-{jsv['turn_number']}.json", jsv)
    for dv in dv_sequence:
        write_json(run_dir / "canonical" / "dv" / f"{dv['from_turn']}-{dv['to_turn']}.json", dv)
    write_json(run_dir / "canonical" / "trajectory.json", trajectory)
    write_jsonl(run_dir / "overlays" / "track_outputs.jsonl", overlay_rows)
    write_jsonl(run_dir / "extracts" / track_name / "extracts.jsonl", extract_rows)
    storage.update_checkpoint(
        interaction_id=str(raw_interaction["interaction_id"]),
        total_turns=len(raw_interaction["turns"]),
        completed_turns=len(jsv_sequence),
        resumed_turns=resumed_turns,
        written_turns=written_turns,
        status="completed",
    )
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "input_path": str(input_path),
            "interaction_id": raw_interaction["interaction_id"],
            "track_name": track_name,
            "resume_enabled": resume,
            "jsv_count": len(jsv_sequence),
            "dv_count": len(dv_sequence),
            "resumed_turns": resumed_turns,
            "written_turns": written_turns,
            "code_revision": _git_revision(),
            "paths": {
                "input": str(run_dir / "input" / input_path.name),
                "canonical_root": str(run_dir / "canonical"),
                "overlay_track_outputs": str(run_dir / "overlays" / "track_outputs.jsonl"),
                "extracts_jsonl": str(run_dir / "extracts" / track_name / "extracts.jsonl"),
                "checkpoint": str(run_dir / "checkpoints" / "progress.json"),
            },
            "output_root": str(run_dir),
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the POCv3 M1 smoke pipeline")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/runs"))
    parser.add_argument("--protocol-schema-root", type=Path, default=None)
    parser.add_argument("--track", default="fixture_hint", choices=["fixture_hint", "heuristic_baseline", "llm_observer"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_dir = run_poc(
        input_path=args.input,
        run_id=args.run_id,
        output_root=args.output_root,
        protocol_schema_root=args.protocol_schema_root,
        track_name=args.track,
        resume=bool(args.resume),
    )
    print(f"Run written: {run_dir}")


if __name__ == "__main__":
    main()
