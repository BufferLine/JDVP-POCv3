"""Versioned service contracts for external consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SERVICE_RESPONSE_SCHEMA_VERSION = "pocv3-service-response-v1"
RUN_RESULT_SCHEMA_VERSION = "pocv3-run-result-v1"
FEWSHOT_BENCHMARK_RESULT_SCHEMA_VERSION = "pocv3-fewshot-benchmark-result-v1"
SERVICE_ERROR_SCHEMA_VERSION = "pocv3-service-error-v1"


@dataclass(frozen=True)
class ExternalRunResult:
    run_id: str
    interaction_id: str
    track_name: str
    output_root: str
    canonical_root: str
    trajectory_path: str
    manifest_path: str
    schema_version: str = RUN_RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "interaction_id": self.interaction_id,
            "track_name": self.track_name,
            "output_root": self.output_root,
            "canonical_root": self.canonical_root,
            "trajectory_path": self.trajectory_path,
            "manifest_path": self.manifest_path,
        }


@dataclass(frozen=True)
class ExternalFewshotBenchmarkResult:
    split: str
    item_count: int
    comparison_track: str
    results_path: str
    average_field_disagreement_rate: float
    max_field_disagreement_rate: float
    schema_version: str = FEWSHOT_BENCHMARK_RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split": self.split,
            "item_count": self.item_count,
            "comparison_track": self.comparison_track,
            "results_path": self.results_path,
            "average_field_disagreement_rate": self.average_field_disagreement_rate,
            "max_field_disagreement_rate": self.max_field_disagreement_rate,
        }
