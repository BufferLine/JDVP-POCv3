"""Service-facing evaluation surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.eval.run_fewshot_benchmark import run_fewshot_benchmark_plan
from src.pipeline.run_storage import read_json
from src.service.contracts import ExternalFewshotBenchmarkResult, SERVICE_RESPONSE_SCHEMA_VERSION
from src.service.errors import ServiceError, serialize_service_error


@dataclass(frozen=True)
class FewshotBenchmarkRequest:
    plan_path: Path
    output_root: Path
    comparison_track: str = "heuristic_baseline"


@dataclass(frozen=True)
class FewshotBenchmarkResult:
    split: str
    item_count: int
    comparison_track: str
    results_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "item_count": self.item_count,
            "comparison_track": self.comparison_track,
            "results_path": str(self.results_path),
        }

    def to_external_dict(self) -> dict[str, Any]:
        return ExternalFewshotBenchmarkResult(
            split=self.split,
            item_count=self.item_count,
            comparison_track=self.comparison_track,
            results_path=str(self.results_path),
        ).to_dict()


def run_fewshot_benchmark(request: FewshotBenchmarkRequest) -> FewshotBenchmarkResult:
    try:
        results_path = run_fewshot_benchmark_plan(
            plan_path=request.plan_path,
            output_root=request.output_root,
            comparison_track=request.comparison_track,
        )
        summary = read_json(results_path)
        return FewshotBenchmarkResult(
            split=str(summary["split"]),
            item_count=int(summary["item_count"]),
            comparison_track=str(summary["comparison_track"]),
            results_path=results_path,
        )
    except FileNotFoundError as exc:
        raise ServiceError(
            code="benchmark_plan_not_found",
            message=f"benchmark plan not found: {request.plan_path}",
            details={"plan_path": str(request.plan_path)},
        ) from exc
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError(
            code="benchmark_execution_failed",
            message="failed to execute few-shot benchmark",
            details={"plan_path": str(request.plan_path), "cause": str(exc)},
        ) from exc


def run_fewshot_benchmark_response(request: FewshotBenchmarkRequest) -> dict[str, Any]:
    try:
        result = run_fewshot_benchmark(request)
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
