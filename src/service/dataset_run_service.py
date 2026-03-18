"""Service-facing dataset-scoped JDVP execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.catalog.sqlite_store import CatalogDatasetRunRecord, CatalogStore
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
    dataset_run_id: str
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
            "dataset_run_id": self.dataset_run_id,
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
            dataset_run_id=self.dataset_run_id,
            dataset_id=self.dataset_id,
            split=self.split,
            track_name=self.track_name,
            output_root=str(self.output_root),
            item_count=self.item_count,
            completed_count=self.completed_count,
            failed_count=self.failed_count,
            summary_path=str(self.summary_path),
        ).to_dict()


def _build_dataset_run_id(output_root: Path) -> str:
    return str(output_root.resolve(strict=False))


def _dataset_run_status(*, item_count: int, completed_count: int, failed_count: int) -> str:
    if failed_count == 0:
        return "completed"
    if completed_count == 0 and item_count > 0:
        return "failed"
    return "partial"


def run_dataset(request: DatasetRunRequest) -> DatasetRunResult:
    dataset_run_id: str | None = None
    manifest = None
    items: list = []
    completed_count: int = 0
    failed_count: int = 0
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
        dataset_run_id = _build_dataset_run_id(batch_root)
        summary_path = batch_root / "dataset_run_summary.json"
        catalog = CatalogStore()
        catalog.upsert_dataset_run(
            CatalogDatasetRunRecord(
                dataset_run_id=dataset_run_id,
                dataset_id=manifest.dataset_id,
                track_name=request.track_name,
                output_root=str(batch_root),
                summary_path=str(summary_path),
                split=request.split,
                scenario_id=request.scenario_id,
                item_count=len(items),
                completed_count=0,
                failed_count=0,
                status="running",
            )
        )
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
                        dataset_run_id=dataset_run_id,
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

        status = _dataset_run_status(
            item_count=len(items),
            completed_count=completed_count,
            failed_count=failed_count,
        )
        write_json(
            summary_path,
            {
                "schema_version": DATASET_RUN_RESULT_SCHEMA_VERSION,
                "dataset_run_id": dataset_run_id,
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
        catalog.upsert_dataset_run(
            CatalogDatasetRunRecord(
                dataset_run_id=dataset_run_id,
                dataset_id=manifest.dataset_id,
                track_name=request.track_name,
                output_root=str(batch_root),
                summary_path=str(summary_path),
                split=request.split,
                scenario_id=request.scenario_id,
                item_count=len(items),
                completed_count=completed_count,
                failed_count=failed_count,
                status=status,
            )
        )
        return DatasetRunResult(
            dataset_run_id=dataset_run_id,
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
    except Exception as exc:
        if dataset_run_id is not None and manifest is not None:
            catalog = CatalogStore()
            catalog.upsert_dataset_run(
                CatalogDatasetRunRecord(
                    dataset_run_id=dataset_run_id,
                    dataset_id=manifest.dataset_id,
                    track_name=request.track_name,
                    output_root=str(request.output_root),
                    summary_path=str(request.output_root / "dataset_run_summary.json"),
                    split=request.split,
                    scenario_id=request.scenario_id,
                    item_count=len(items),
                    completed_count=completed_count,
                    failed_count=failed_count,
                    status="failed",
                    error_message=str(exc),
                )
            )
        raise


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
