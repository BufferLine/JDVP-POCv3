"""Service-facing orchestration surface for POCv3."""

from .contracts import (
    DATASET_RUN_RESULT_SCHEMA_VERSION,
    FEWSHOT_BENCHMARK_RESULT_SCHEMA_VERSION,
    RUN_RESULT_SCHEMA_VERSION,
    SERVICE_ERROR_SCHEMA_VERSION,
    SERVICE_RESPONSE_SCHEMA_VERSION,
)
from .errors import ServiceError

__all__ = [
    "FewshotBenchmarkRequest",
    "FewshotBenchmarkResult",
    "DatasetRunRequest",
    "DatasetRunResult",
    "PipelineArtifacts",
    "RunRequest",
    "RunResult",
    "ServiceError",
    "DATASET_RUN_RESULT_SCHEMA_VERSION",
    "FEWSHOT_BENCHMARK_RESULT_SCHEMA_VERSION",
    "RUN_RESULT_SCHEMA_VERSION",
    "SERVICE_ERROR_SCHEMA_VERSION",
    "SERVICE_RESPONSE_SCHEMA_VERSION",
    "run_fewshot_benchmark",
    "run_fewshot_benchmark_response",
    "run_dataset",
    "run_dataset_response",
    "run_interaction",
    "run_interaction_file",
    "run_interaction_response",
]


def __getattr__(name: str):
    if name in {
        "DatasetRunRequest",
        "DatasetRunResult",
        "run_dataset",
        "run_dataset_response",
    }:
        from . import dataset_run_service

        return getattr(dataset_run_service, name)
    if name in {
        "FewshotBenchmarkRequest",
        "FewshotBenchmarkResult",
        "run_fewshot_benchmark",
        "run_fewshot_benchmark_response",
    }:
        from . import eval_service

        return getattr(eval_service, name)
    if name in {
        "PipelineArtifacts",
        "RunRequest",
        "RunResult",
        "run_interaction",
        "run_interaction_file",
        "run_interaction_response",
    }:
        from . import poc_service

        return getattr(poc_service, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
