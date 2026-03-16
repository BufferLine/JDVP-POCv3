"""Service-facing dataset-scoped JDVP execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.dataset.manifest_loader import load_dataset_manifest
from src.pipeline.run_storage import write_json
from src.service.contracts import (
    DATASET_RUN_RESULT_SCHEMA_VERSION,
    SERVICE_RESPONSE_SCHEMA_VERSION,
    ExternalDatasetRunResult,
)
from src.service.errors import ServiceError, serialize_service_error
from src.service.poc_service import RunRequest, run_interaction


@dataclass(frozen=True)
class DatasetRunRequest:
    dataset_root: Path
    output_root: Path
    track_name: str = "fixture_hint"
    split: str | None = None
    scenario_id: str | None = None
    max_items: int | None = None
    resume: bool = False


@dataclass(frozen=True)
class DatasetRunResult:
    dataset_id: str
    split: str | None
    track_name: str
    output_root: Path
    item_count: int
    completed_count: int
    failed_count: int
    summary_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "track_name": self.track_name,
            "output_root": str(self.output_root),
            "item_count": self.item_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "summary_path": str(self.summary_path),
        }

    def to_external_dict(self) -> dict[str, Any]:
        return ExternalDatasetRunResult(
            dataset_id=self.dataset_id,
            split=self.split,
            track_name=self.track_name,
            output_root=str(self.output_root),
            item_count=self.item_count,
            completed_count=self.completed_count,
            failed_count=self.failed_count,
            summary_path=str(self.summary_path),
        ).to_dict()


def run_dataset(request: DatasetRunRequest) -> DatasetRunResult:
    try:
        manifest = load_dataset_manifest(request.dataset_root)
        items = manifest.items
        if request.split is not None:
            items = [item for item in items if item.split == request.split]
        if request.scenario_id is not None:
            items = [item for item in items if item.scenario_id == request.scenario_id]
        if request.max_items is not None:
            items = items[: request.max_items]

        batch_root = request.output_root
        rows: list[dict[str, Any]] = []
        completed_count = 0
        failed_count = 0
        for item in items:
            run_id = f"{request.track_name}-{item.interaction_id}"
            try:
                result = run_interaction(
                    RunRequest(
                        input_path=item.interaction_path(request.dataset_root),
                        run_id=run_id,
                        output_root=batch_root / "runs",
                        track_name=request.track_name,
                        resume=request.resume,
                        dataset_id=manifest.dataset_id,
                    )
                )
                completed_count += 1
                rows.append(
                    {
                        "interaction_id": item.interaction_id,
                        "scenario_id": item.scenario_id,
                        "split": item.split,
                        "status": "completed",
                        "run_id": result.run_id,
                        "model_id": result.model_id,
                        "prompt_version": result.prompt_version,
                        "run_dir": str(result.run_dir),
                    }
                )
            except ServiceError as exc:
                failed_count += 1
                rows.append(
                    {
                        "interaction_id": item.interaction_id,
                        "scenario_id": item.scenario_id,
                        "split": item.split,
                        "status": "failed",
                        "run_id": run_id,
                        "error_code": exc.code,
                        "error_message": exc.message,
                    }
                )

        summary_path = batch_root / "dataset_run_summary.json"
        write_json(
            summary_path,
            {
                "schema_version": DATASET_RUN_RESULT_SCHEMA_VERSION,
                "dataset_id": manifest.dataset_id,
                "split": request.split,
                "scenario_id": request.scenario_id,
                "track_name": request.track_name,
                "output_root": str(batch_root),
                "item_count": len(items),
                "completed_count": completed_count,
                "failed_count": failed_count,
                "items": rows,
            },
        )
        return DatasetRunResult(
            dataset_id=manifest.dataset_id,
            split=request.split,
            track_name=request.track_name,
            output_root=batch_root,
            item_count=len(items),
            completed_count=completed_count,
            failed_count=failed_count,
            summary_path=summary_path,
        )
    except FileNotFoundError as exc:
        raise ServiceError(
            code="dataset_manifest_not_found",
            message=f"dataset manifest not found under: {request.dataset_root}",
            details={"dataset_root": str(request.dataset_root)},
        ) from exc


def run_dataset_response(request: DatasetRunRequest) -> dict[str, Any]:
    try:
        result = run_dataset(request)
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
