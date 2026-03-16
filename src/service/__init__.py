"""Service-facing orchestration surface for POCv3."""

from .contracts import (
    FEWSHOT_BENCHMARK_RESULT_SCHEMA_VERSION,
    RUN_RESULT_SCHEMA_VERSION,
    SERVICE_ERROR_SCHEMA_VERSION,
    SERVICE_RESPONSE_SCHEMA_VERSION,
)
from .errors import ServiceError
from .eval_service import (
    FewshotBenchmarkRequest,
    FewshotBenchmarkResult,
    run_fewshot_benchmark,
    run_fewshot_benchmark_response,
)
from .poc_service import (
    PipelineArtifacts,
    RunRequest,
    RunResult,
    run_interaction,
    run_interaction_file,
    run_interaction_response,
)

__all__ = [
    "FewshotBenchmarkRequest",
    "FewshotBenchmarkResult",
    "PipelineArtifacts",
    "RunRequest",
    "RunResult",
    "ServiceError",
    "FEWSHOT_BENCHMARK_RESULT_SCHEMA_VERSION",
    "RUN_RESULT_SCHEMA_VERSION",
    "SERVICE_ERROR_SCHEMA_VERSION",
    "SERVICE_RESPONSE_SCHEMA_VERSION",
    "run_fewshot_benchmark",
    "run_fewshot_benchmark_response",
    "run_interaction",
    "run_interaction_file",
    "run_interaction_response",
]
