#!/usr/bin/env python3
"""Benchmark multiple dataset-run labels against a shared baseline label."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ensemble_benchmark import compare_runs
from src.pipeline.run_storage import read_json, write_json


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_items(summary_path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(summary_path)
    return {
        str(item["interaction_id"]): item
        for item in payload["items"]
        if str(item["status"]) == "completed"
    }


def benchmark_matrix(*, matrix_summary_path: Path, baseline_label: str, output_root: Path) -> Path:
    matrix = read_json(matrix_summary_path)
    runs_by_label = {str(run["label"]): run for run in matrix["runs"]}
    if baseline_label not in runs_by_label:
        raise ValueError(f"baseline label not found: {baseline_label}")

    baseline_items = _load_items(Path(runs_by_label[baseline_label]["dataset_run_result"]["summary_path"]))
    comparison_rows: list[dict[str, Any]] = []

    for label, run in runs_by_label.items():
        item_summary_path = Path(run["dataset_run_result"]["summary_path"])
        items = _load_items(item_summary_path)
        if label == baseline_label:
            continue

        pair_summaries: list[dict[str, Any]] = []
        for interaction_id, baseline_item in baseline_items.items():
            comparison_item = items.get(interaction_id)
            if comparison_item is None:
                continue

            pair_output_dir = output_root / "pairwise" / label / interaction_id
            pair_summary = compare_runs(
                run_dirs=[
                    Path(str(baseline_item["run_dir"])),
                    Path(str(comparison_item["run_dir"])),
                ],
                output_dir=pair_output_dir,
            )
            pair_summaries.append(pair_summary)

            all_run_dirs = [Path(str(baseline_item["run_dir"]))]
            for all_label, all_run in runs_by_label.items():
                item_map = _load_items(Path(all_run["dataset_run_result"]["summary_path"]))
                if interaction_id in item_map and item_map[interaction_id]["status"] == "completed":
                    all_run_dirs.append(Path(str(item_map[interaction_id]["run_dir"])))
            unique_run_dirs: list[Path] = []
            seen: set[str] = set()
            for run_dir in all_run_dirs:
                key = str(run_dir)
                if key not in seen:
                    seen.add(key)
                    unique_run_dirs.append(run_dir)
            if len(unique_run_dirs) >= 2:
                compare_runs(
                    run_dirs=unique_run_dirs,
                    output_dir=output_root / "ensemble" / interaction_id,
                )

        comparison_rows.append(
            {
                "label": label,
                "track_name": str(run["track_name"]),
                "item_count": len(pair_summaries),
                "average_field_disagreement_rate": _average(
                    [float(item["field_disagreement_rate"]) for item in pair_summaries]
                ),
                "max_field_disagreement_rate": max(
                    [float(item["field_disagreement_rate"]) for item in pair_summaries],
                    default=0.0,
                ),
            }
        )

    summary_path = output_root / "benchmark_matrix_summary.json"
    write_json(
        summary_path,
        {
            "schema_version": "pocv3-benchmark-matrix-summary-v1",
            "matrix_summary_path": str(matrix_summary_path),
            "baseline_label": baseline_label,
            "output_root": str(output_root),
            "comparisons": comparison_rows,
        },
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dataset matrix results against one baseline label")
    parser.add_argument("--matrix-summary", required=True, type=Path)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    summary_path = benchmark_matrix(
        matrix_summary_path=args.matrix_summary,
        baseline_label=args.baseline_label,
        output_root=args.output_root,
    )
    print(f"Benchmark matrix summary written: {summary_path}")


if __name__ == "__main__":
    main()
