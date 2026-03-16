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


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate_item_rates(
    item_summaries: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in item_summaries:
        grouped.setdefault(str(item[key]), []).append(item)

    aggregates: dict[str, dict[str, Any]] = {}
    for group_name, items in grouped.items():
        field_names = sorted(items[0]["field_disagreement_rates"].keys()) if items else []
        aggregates[group_name] = {
            "item_count": len(items),
            "average_field_disagreement_rate": _average(
                [float(item["field_disagreement_rate"]) for item in items]
            ),
            "average_fewshot_example_count": _average(
                [float(item["fewshot_example_count"]) for item in items]
            ),
            "field_disagreement_rates": {
                field_name: _average(
                    [float(item["field_disagreement_rates"][field_name]) for item in items]
                )
                for field_name in field_names
            },
        }
    return aggregates


def run_fewshot_benchmark_plan(
    *,
    plan_path: Path,
    output_root: Path,
    comparison_track: str = "heuristic_baseline",
    max_average_field_disagreement_rate: float | None = None,
    max_field_disagreement_rate: float | None = None,
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
                "scenario_id": str(item["scenario_id"]),
                "context_module": str(item["context_module"]),
                "fewshot_run_dir": str(fewshot_run_dir),
                "comparison_run_dir": str(comparison_run_dir),
                "comparison_track": comparison_track,
                "fewshot_example_count": int(item["fewshot_example_count"]),
                "field_disagreement_rate": disagreement_rate,
                "field_disagreement_rates": comparison_summary["field_disagreement_rates"],
                "comparison_report_dir": str(comparison_report_dir),
            }
        )

    summary = {
        "schema_version": "pocv3-fewshot-benchmark-results-v1",
        "plan_path": str(plan_path),
        "split": plan["split"],
        "item_count": len(item_summaries),
        "comparison_track": comparison_track,
        "average_fewshot_example_count": (
            sum(item["fewshot_example_count"] for item in item_summaries) / len(item_summaries) if item_summaries else 0.0
        ),
        "average_field_disagreement_rate": (
            sum(disagreement_rates) / len(disagreement_rates) if disagreement_rates else 0.0
        ),
        "min_field_disagreement_rate": min(disagreement_rates) if disagreement_rates else 0.0,
        "max_field_disagreement_rate": max(disagreement_rates) if disagreement_rates else 0.0,
        "zero_disagreement_items": sum(1 for rate in disagreement_rates if rate == 0.0),
        "field_disagreement_rates": {
            field_name: _average(
                [float(item["field_disagreement_rates"][field_name]) for item in item_summaries]
            )
            for field_name in (
                sorted(item_summaries[0]["field_disagreement_rates"].keys()) if item_summaries else []
            )
        },
        "scenario_breakdown": _aggregate_item_rates(item_summaries, "scenario_id"),
        "context_module_breakdown": _aggregate_item_rates(item_summaries, "context_module"),
        "items": item_summaries,
    }
    write_json(output_root / "benchmark_results.json", summary)

    failures: list[str] = []
    if max_average_field_disagreement_rate is not None:
        if float(summary["average_field_disagreement_rate"]) > max_average_field_disagreement_rate:
            failures.append(
                "average_field_disagreement_rate "
                f"{summary['average_field_disagreement_rate']:.4f} exceeds "
                f"{max_average_field_disagreement_rate:.4f}"
            )
    if max_field_disagreement_rate is not None:
        if float(summary["max_field_disagreement_rate"]) > max_field_disagreement_rate:
            failures.append(
                "max_field_disagreement_rate "
                f"{summary['max_field_disagreement_rate']:.4f} exceeds "
                f"{max_field_disagreement_rate:.4f}"
            )
    if failures:
        raise ValueError("; ".join(failures))

    return output_root / "benchmark_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a few-shot benchmark plan")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--comparison-track", default="heuristic_baseline", choices=["fixture_hint", "heuristic_baseline"])
    parser.add_argument("--max-average-field-disagreement-rate", type=float, default=None)
    parser.add_argument("--max-field-disagreement-rate", type=float, default=None)
    args = parser.parse_args()

    result_path = run_fewshot_benchmark_plan(
        plan_path=args.plan,
        output_root=args.output_root,
        comparison_track=args.comparison_track,
        max_average_field_disagreement_rate=args.max_average_field_disagreement_rate,
        max_field_disagreement_rate=args.max_field_disagreement_rate,
    )
    print(f"Few-shot benchmark results written: {result_path}")


if __name__ == "__main__":
    main()
