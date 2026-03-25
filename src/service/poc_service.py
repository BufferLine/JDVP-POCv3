"""Service-oriented orchestration for JDVP POC runs."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.catalog.sqlite_store import CatalogRunRecord, CatalogStore
from src.contracts.raw_interaction_validate import RawInteractionValidator
from src.method.tracks.factory import create_track
from src.method.tracks.fixture_hint import FixtureHintTrack
from src.pipeline.run_storage import RunStorage, write_json, write_jsonl
from src.protocol_core.dv_ordinal import build_dv_sequence
from src.protocol_core.jsv_types import build_jsv_from_hint
from src.protocol_core.schema_validate import CanonicalSchemaValidator
from src.protocol_core.trajectory import build_trajectory
from src.service.contracts import ExternalRunResult, SERVICE_RESPONSE_SCHEMA_VERSION
from src.service.errors import ServiceError, serialize_service_error


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


@dataclass(frozen=True)
class RunRequest:
    input_path: Path
    run_id: str
    output_root: Path
    protocol_schema_root: Path | None = None
    track_name: str = "fixture_hint"
    resume: bool = False
    dataset_id: str | None = None
    dataset_run_id: str | None = None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    interaction_id: str
    track_name: str
    model_id: str
    prompt_version: str
    run_dir: Path
    manifest_path: Path
    trajectory_path: Path
    extracts_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "interaction_id": self.interaction_id,
            "track_name": self.track_name,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "run_dir": str(self.run_dir),
            "manifest_path": str(self.manifest_path),
            "trajectory_path": str(self.trajectory_path),
            "extracts_path": str(self.extracts_path),
        }

    def to_external_dict(self) -> dict[str, Any]:
        return ExternalRunResult(
            run_id=self.run_id,
            interaction_id=self.interaction_id,
            track_name=self.track_name,
            output_root=str(self.run_dir),
            canonical_root=str(self.run_dir / "canonical"),
            trajectory_path=str(self.trajectory_path),
            manifest_path=str(self.manifest_path),
        ).to_dict()


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
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "workspace-local"
    return result.stdout.strip()


def _safe_catalog_upsert(catalog: CatalogStore, record: CatalogRunRecord) -> None:
    try:
        catalog.upsert_run(record)
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "catalog upsert failed for run_id=%s, status=%s",
            record.run_id, record.status, exc_info=True,
        )


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
    if isinstance(track, FixtureHintTrack):
        track.set_turns(raw_interaction["turns"])
    processed_turns: list[dict[str, Any]] = []
    for turn in raw_interaction["turns"]:
        meta = turn.get("meta", {})
        turn_number = int(turn["turn_number"])
        if resume and storage.has_extract_for_turn(turn_number):
            stored = storage.load_extract_for_turn(turn_number)
            current_human = str(turn.get("human_input", ""))
            current_ai = str(turn.get("ai_response", ""))
            stored_human = stored.extract_record.get("human_input", "")
            stored_ai = stored.extract_record.get("ai_response", "")
            input_changed = (current_human != stored_human or current_ai != stored_ai)
            if not input_changed:
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
                processed_turns.append(turn)
                continue

        context_turns = list(processed_turns)
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
        processed_turns.append(turn)

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
) -> RunResult:
    storage = RunStorage(run_dir=run_dir, track_name=track_name)
    track_metadata = (
        artifacts.extract_rows[0].get("track_metadata", {})
        if artifacts.extract_rows
        else {}
    )
    model_id = str(track_metadata.get("model_id", "unknown"))
    prompt_version = str(track_metadata.get("prompt_version", "unknown"))

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
    manifest_path = run_dir / "manifest.json"
    trajectory_path = run_dir / "canonical" / "trajectory.json"
    extracts_path = run_dir / "extracts" / track_name / "extracts.jsonl"
    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "input_path": str(input_path),
            "interaction_id": artifacts.raw_interaction["interaction_id"],
            "track_name": track_name,
            "model_id": model_id,
            "prompt_version": prompt_version,
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
                "extracts_jsonl": str(extracts_path),
                "checkpoint": str(run_dir / "checkpoints" / "progress.json"),
            },
            "output_root": str(run_dir),
        },
    )
    return RunResult(
        run_id=run_id,
        interaction_id=str(artifacts.raw_interaction["interaction_id"]),
        track_name=track_name,
        model_id=model_id,
        prompt_version=prompt_version,
        run_dir=run_dir,
        manifest_path=manifest_path,
        trajectory_path=trajectory_path,
        extracts_path=extracts_path,
    )


def run_interaction(request: RunRequest) -> RunResult:
    run_dir = request.output_root / request.run_id
    catalog = CatalogStore()
    raw_interaction: dict[str, Any] | None = None

    def _write_failure_artifact(exc: Exception) -> None:
        payload: dict[str, Any] = {
            "run_id": request.run_id,
            "track_name": request.track_name,
            "input_path": str(request.input_path),
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }
        raw_response = getattr(exc, "raw_response", None)
        attempt_responses = getattr(exc, "attempt_responses", None)
        if isinstance(raw_response, str):
            payload["raw_response"] = raw_response
        if isinstance(attempt_responses, list):
            payload["attempt_responses"] = [str(item) for item in attempt_responses]
        if raw_interaction is not None and "interaction_id" in raw_interaction:
            payload["interaction_id"] = str(raw_interaction["interaction_id"])
        write_json(run_dir / "errors" / "run_error.json", payload)

    try:
        _safe_catalog_upsert(catalog,
            CatalogRunRecord(
                run_id=request.run_id,
                interaction_id=None,
                dataset_id=None,
                dataset_run_id=request.dataset_run_id,
                track_name=request.track_name,
                model_id=None,
                input_path=str(request.input_path),
                run_dir=str(run_dir),
                status="running",
            )
        )
        raw_interaction = _load_input(request.input_path)
        artifacts = build_pipeline_artifacts(
            raw_interaction=raw_interaction,
            track_name=request.track_name,
            run_dir=run_dir,
            protocol_schema_root=request.protocol_schema_root,
            resume=request.resume,
        )
        result = write_run_outputs(
            artifacts=artifacts,
            input_path=request.input_path,
            run_id=request.run_id,
            run_dir=run_dir,
            track_name=request.track_name,
            resume=request.resume,
        )
        _safe_catalog_upsert(catalog,
            CatalogRunRecord(
                run_id=request.run_id,
                interaction_id=result.interaction_id,
                dataset_id=request.dataset_id,
                dataset_run_id=request.dataset_run_id,
                track_name=request.track_name,
                model_id=result.model_id,
                input_path=str(request.input_path),
                run_dir=str(run_dir),
                status="completed",
            )
        )
        return result
    except FileNotFoundError as exc:
        _write_failure_artifact(exc)
        _safe_catalog_upsert(catalog,
            CatalogRunRecord(
                run_id=request.run_id,
                interaction_id=(
                    str(raw_interaction["interaction_id"])
                    if raw_interaction is not None and "interaction_id" in raw_interaction
                    else None
                ),
                dataset_id=request.dataset_id,
                dataset_run_id=request.dataset_run_id,
                track_name=request.track_name,
                model_id=None,
                input_path=str(request.input_path),
                run_dir=str(run_dir),
                status="failed",
                error_message=str(exc),
            )
        )
        missing_path = str(getattr(exc, "filename", None) or exc)
        is_input = raw_interaction is None
        raise ServiceError(
            code="input_not_found" if is_input else "file_not_found",
            message=f"{'input' if is_input else 'required'} file not found: {missing_path}",
            details={"missing_path": missing_path, "input_path": str(request.input_path)},
        ) from exc
    except ServiceError as exc:
        _write_failure_artifact(exc)
        _safe_catalog_upsert(catalog,
            CatalogRunRecord(
                run_id=request.run_id,
                interaction_id=(
                    str(raw_interaction["interaction_id"])
                    if raw_interaction is not None and "interaction_id" in raw_interaction
                    else None
                ),
                dataset_id=request.dataset_id,
                dataset_run_id=request.dataset_run_id,
                track_name=request.track_name,
                model_id=None,
                input_path=str(request.input_path),
                run_dir=str(run_dir),
                status="failed",
                error_message="service_error",
            )
        )
        raise
    except Exception as exc:
        _write_failure_artifact(exc)
        _safe_catalog_upsert(catalog,
            CatalogRunRecord(
                run_id=request.run_id,
                interaction_id=(
                    str(raw_interaction["interaction_id"])
                    if raw_interaction is not None and "interaction_id" in raw_interaction
                    else None
                ),
                dataset_id=request.dataset_id,
                dataset_run_id=request.dataset_run_id,
                track_name=request.track_name,
                model_id=None,
                input_path=str(request.input_path),
                run_dir=str(run_dir),
                status="failed",
                error_message=str(exc),
            )
        )
        raise ServiceError(
            code="run_execution_failed",
            message="failed to execute JDVP run",
            details={"input_path": str(request.input_path), "track_name": request.track_name, "cause": str(exc)},
        ) from exc


def run_interaction_file(
    *,
    input_path: Path,
    run_id: str,
    output_root: Path,
    protocol_schema_root: Path | None = None,
    track_name: str = "fixture_hint",
    resume: bool = False,
) -> Path:
    result = run_interaction(
        RunRequest(
            input_path=input_path,
            run_id=run_id,
            output_root=output_root,
            protocol_schema_root=protocol_schema_root,
            track_name=track_name,
            resume=resume,
        )
    )
    return result.run_dir


def run_interaction_response(request: RunRequest) -> dict[str, Any]:
    try:
        result = run_interaction(request)
    except Exception as exc:
        return {
            "schema_version": SERVICE_RESPONSE_SCHEMA_VERSION,
            "ok": False,
            "error": serialize_service_error(exc),
        }
    return {
        "schema_version": SERVICE_RESPONSE_SCHEMA_VERSION,
        "ok": True,
        "result": result.to_external_dict(),
    }
