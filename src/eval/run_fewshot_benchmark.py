"""Execute a few-shot benchmark plan across one dataset split."""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.eval.ensemble_benchmark import compare_runs
from src.pipeline.run_poc import run_poc
from src.pipeline.run_storage import read_json, write_json


@contextmanager
def _temporary_env(updates: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _build_item_pack(base_pack: dict[str, Any], plan_item: dict[str, Any]) -> dict[str, Any]:
    return {
        **base_pack,
        "source_plan_interaction_id": plan_item["interaction_id"],
        "examples": plan_item["fewshot_examples"],
    }


def run_fewshot_benchmark_plan(
    *,
    plan_path: Path,
    output_root: Path,
    comparison_track: str = "heuristic_baseline",
) -> Path:
    plan = read_json(plan_path)
    base_pack = read_json(Path(plan["fewshot_pack_path"]))

    runs_root = output_root / "runs"
    packs_root = output_root / "fewshot_packs"
    reports_root = output_root / "reports"

    item_summaries: list[dict[str, Any]] = []
    disagreement_rates: list[float] = []

    for item in plan["items"]:
        interaction_id = str(item["interaction_id"])
        input_path = Path(item["input_path"])
        item_pack_path = packs_root / f"{interaction_id}.json"
        write_json(item_pack_path, _build_item_pack(base_pack, item))

        with _temporary_env({"JDVP_FEWSHOT_PACK_PATH": str(item_pack_path)}):
            fewshot_run_dir = run_poc(
                input_path=input_path,
                run_id=f"fewshot-{interaction_id}",
                output_root=runs_root,
                track_name="fewshot_prompt",
            )

        comparison_run_dir = run_poc(
            input_path=input_path,
            run_id=f"{comparison_track}-{interaction_id}",
            output_root=runs_root,
            track_name=comparison_track,
        )
        comparison_report_dir = reports_root / interaction_id
        comparison_summary = compare_runs(
            run_dirs=[fewshot_run_dir, comparison_run_dir],
            output_dir=comparison_report_dir,
        )
        disagreement_rate = float(comparison_summary["field_disagreement_rate"])
        disagreement_rates.append(disagreement_rate)
        item_summaries.append(
            {
                "interaction_id": interaction_id,
                "input_path": str(input_path),
                "fewshot_run_dir": str(fewshot_run_dir),
                "comparison_run_dir": str(comparison_run_dir),
                "comparison_track": comparison_track,
                "fewshot_example_count": int(item["fewshot_example_count"]),
                "field_disagreement_rate": disagreement_rate,
                "comparison_report_dir": str(comparison_report_dir),
            }
        )

    summary = {
        "schema_version": "pocv3-fewshot-benchmark-results-v1",
        "plan_path": str(plan_path),
        "split": plan["split"],
        "item_count": len(item_summaries),
        "comparison_track": comparison_track,
        "average_field_disagreement_rate": (
            sum(disagreement_rates) / len(disagreement_rates) if disagreement_rates else 0.0
        ),
        "items": item_summaries,
    }
    write_json(output_root / "benchmark_results.json", summary)
    return output_root / "benchmark_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a few-shot benchmark plan")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--comparison-track", default="heuristic_baseline", choices=["fixture_hint", "heuristic_baseline"])
    args = parser.parse_args()

    result_path = run_fewshot_benchmark_plan(
        plan_path=args.plan,
        output_root=args.output_root,
        comparison_track=args.comparison_track,
    )
    print(f"Few-shot benchmark results written: {result_path}")


if __name__ == "__main__":
    main()
