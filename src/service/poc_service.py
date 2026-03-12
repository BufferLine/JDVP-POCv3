"""Service-oriented orchestration for JDVP POC runs."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.contracts.raw_interaction_validate import RawInteractionValidator
from src.method.tracks.factory import create_track
from src.method.tracks.fixture_hint import FixtureHintTrack
from src.pipeline.run_storage import RunStorage, write_json, write_jsonl
from src.protocol_core.dv_ordinal import build_dv_sequence
from src.protocol_core.jsv_types import build_jsv_from_hint
from src.protocol_core.schema_validate import CanonicalSchemaValidator
from src.protocol_core.trajectory import build_trajectory


@dataclass(frozen=True)
class PipelineArtifacts:
    raw_interaction: dict[str, Any]
    jsv_sequence: list[dict[str, Any]]
    dv_sequence: list[dict[str, Any]]
    trajectory: dict[str, Any]
    overlay_rows: list[dict[str, Any]]
    extract_rows: list[dict[str, Any]]
    resumed_turns: list[int]
    written_turns: list[int]


def _load_input(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def build_pipeline_artifacts(
    *,
    raw_interaction: dict[str, Any],
    track_name: str,
    run_dir: Path,
    protocol_schema_root: Path | None = None,
    resume: bool = False,
) -> PipelineArtifacts:
    RawInteractionValidator().validate(raw_interaction)
    validator = CanonicalSchemaValidator(protocol_schema_root=protocol_schema_root)
    storage = RunStorage(run_dir=run_dir, track_name=track_name)

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

        context_turns = raw_interaction["turns"][:turn_number]
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

    dv_sequence = [record.to_dict() for record in build_dv_sequence(jsv_sequence)]
    trajectory = build_trajectory(interaction_id, dv_sequence).to_dict()

    for jsv in jsv_sequence:
        validator.validate_jsv(jsv)
    for dv in dv_sequence:
        validator.validate_dv(dv)
    validator.validate_trajectory(trajectory)

    return PipelineArtifacts(
        raw_interaction=raw_interaction,
        jsv_sequence=jsv_sequence,
        dv_sequence=dv_sequence,
        trajectory=trajectory,
        overlay_rows=overlay_rows,
        extract_rows=extract_rows,
        resumed_turns=resumed_turns,
        written_turns=written_turns,
    )


def write_run_outputs(
    *,
    artifacts: PipelineArtifacts,
    input_path: Path,
    run_id: str,
    run_dir: Path,
    track_name: str,
    resume: bool,
) -> Path:
    storage = RunStorage(run_dir=run_dir, track_name=track_name)

    write_json(run_dir / "input" / input_path.name, artifacts.raw_interaction)
    for jsv in artifacts.jsv_sequence:
        write_json(run_dir / "canonical" / "jsv" / f"turn-{jsv['turn_number']}.json", jsv)
    for dv in artifacts.dv_sequence:
        write_json(run_dir / "canonical" / "dv" / f"{dv['from_turn']}-{dv['to_turn']}.json", dv)
    write_json(run_dir / "canonical" / "trajectory.json", artifacts.trajectory)
    write_jsonl(run_dir / "overlays" / "track_outputs.jsonl", artifacts.overlay_rows)
    write_jsonl(run_dir / "extracts" / track_name / "extracts.jsonl", artifacts.extract_rows)
    storage.update_checkpoint(
        interaction_id=str(artifacts.raw_interaction["interaction_id"]),
        total_turns=len(artifacts.raw_interaction["turns"]),
        completed_turns=len(artifacts.jsv_sequence),
        resumed_turns=artifacts.resumed_turns,
        written_turns=artifacts.written_turns,
        status="completed",
    )
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "input_path": str(input_path),
            "interaction_id": artifacts.raw_interaction["interaction_id"],
            "track_name": track_name,
            "resume_enabled": resume,
            "jsv_count": len(artifacts.jsv_sequence),
            "dv_count": len(artifacts.dv_sequence),
            "resumed_turns": artifacts.resumed_turns,
            "written_turns": artifacts.written_turns,
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


def run_interaction_file(
    *,
    input_path: Path,
    run_id: str,
    output_root: Path,
    protocol_schema_root: Path | None = None,
    track_name: str = "fixture_hint",
    resume: bool = False,
) -> Path:
    raw_interaction = _load_input(input_path)
    run_dir = output_root / run_id
    artifacts = build_pipeline_artifacts(
        raw_interaction=raw_interaction,
        track_name=track_name,
        run_dir=run_dir,
        protocol_schema_root=protocol_schema_root,
        resume=resume,
    )
    return write_run_outputs(
        artifacts=artifacts,
        input_path=input_path,
        run_id=run_id,
        run_dir=run_dir,
        track_name=track_name,
        resume=resume,
    )
