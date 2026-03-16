"""Service-facing evaluation surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.eval.run_fewshot_benchmark import run_fewshot_benchmark_plan


@dataclass(frozen=True)
class FewshotBenchmarkRequest:
    plan_path: Path
    output_root: Path
    comparison_track: str = "heuristic_baseline"


@dataclass(frozen=True)
class FewshotBenchmarkResult:
    results_path: Path


def run_fewshot_benchmark(request: FewshotBenchmarkRequest) -> FewshotBenchmarkResult:
    return FewshotBenchmarkResult(
        results_path=run_fewshot_benchmark_plan(
            plan_path=request.plan_path,
            output_root=request.output_root,
            comparison_track=request.comparison_track,
        )
    )
