"""Compare stored extracts across runs and build simple ensemble reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.pipeline.run_storage import read_json, write_json, write_jsonl
from src.protocol_core.enums import CORE_FIELD_NAMES

CORE_FIELDS = CORE_FIELD_NAMES


def _load_run_manifest(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "manifest.json")


def _load_extracts(run_dir: Path, track_name: str) -> list[dict[str, Any]]:
    path = run_dir / "extracts" / track_name / "extracts.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _index_by_turn(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["turn_number"]): row for row in rows}


def _majority_value(values: list[str]) -> str:
    counts = Counter(values)
    max_count = max(counts.values())
    winners = sorted([value for value, count in counts.items() if count == max_count])
    return winners[0]


def _ensemble_hint(extract_rows: list[dict[str, Any]]) -> dict[str, Any]:
    hints = [row["jsv_hint"] for row in extract_rows]
    ensembled: dict[str, Any] = {
        field_name: _majority_value([str(hint[field_name]) for hint in hints])
        for field_name in CORE_FIELDS
    }
    confidence_values: dict[str, str] = {}
    for field_name in CORE_FIELDS:
        confidence_values[field_name] = _majority_value(
            [str(hint.get("confidence", {}).get(field_name, "medium")) for hint in hints]
        )
    ensembled["confidence"] = confidence_values
    return ensembled


def _disagreement_fields(extract_rows: list[dict[str, Any]]) -> list[str]:
    disagreements: list[str] = []
    for field_name in CORE_FIELDS:
        values = {str(row["jsv_hint"][field_name]) for row in extract_rows}
        if len(values) > 1:
            disagreements.append(field_name)
    return disagreements


def _track_field_values(extract_rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    return {
        str(row["track_name"]): {
            field_name: str(row["jsv_hint"][field_name])
            for field_name in CORE_FIELDS
        }
        for row in extract_rows
    }


def compare_runs(
    *,
    run_dirs: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    manifests = [_load_run_manifest(run_dir) for run_dir in run_dirs]
    interaction_ids = {manifest["interaction_id"] for manifest in manifests}
    if len(interaction_ids) != 1:
        raise ValueError("all runs must share one interaction_id")

    run_rows: dict[str, dict[int, dict[str, Any]]] = {}
    track_names: list[str] = []
    run_keys: list[str] = []
    for run_dir, manifest in zip(run_dirs, manifests):
        track_name = str(manifest["track_name"])
        run_id = str(manifest["run_id"])
        track_names.append(track_name)
        run_key = f"{run_id}:{track_name}"
        run_keys.append(run_key)
        run_rows[run_key] = _index_by_turn(_load_extracts(run_dir, track_name))

    turn_numbers = sorted(set().union(*(rows.keys() for rows in run_rows.values())))
    comparisons: list[dict[str, Any]] = []
    total_field_comparisons = 0
    total_field_disagreements = 0
    field_comparisons = {field_name: 0 for field_name in CORE_FIELDS}
    field_disagreements = {field_name: 0 for field_name in CORE_FIELDS}

    for turn_number in turn_numbers:
        present_rows = []
        missing_tracks = []
        for run_key in run_keys:
            row = run_rows[run_key].get(turn_number)
            if row is None:
                missing_tracks.append(run_key)
            else:
                present_rows.append(row)
        if len(present_rows) < 2:
            continue

        disagreements = _disagreement_fields(present_rows)
        total_field_comparisons += len(CORE_FIELDS)
        total_field_disagreements += len(disagreements)
        for field_name in CORE_FIELDS:
            field_comparisons[field_name] += 1
            if field_name in disagreements:
                field_disagreements[field_name] += 1
        comparisons.append(
            {
                "interaction_id": manifests[0]["interaction_id"],
                "turn_number": turn_number,
                "tracks_present": [row["track_name"] for row in present_rows],
                "missing_tracks": missing_tracks,
                "disagreement_fields": disagreements,
                "disagreement_score": len(disagreements) / len(CORE_FIELDS),
                "track_field_values": _track_field_values(present_rows),
                "ensemble_jsv_hint": _ensemble_hint(present_rows),
            }
        )

    summary = {
        "interaction_id": manifests[0]["interaction_id"],
        "run_ids": [manifest["run_id"] for manifest in manifests],
        "track_names": track_names,
        "turns_compared": len(comparisons),
        "field_disagreement_rate": (
            total_field_disagreements / total_field_comparisons if total_field_comparisons else 0.0
        ),
        "field_disagreement_rates": {
            field_name: (
                field_disagreements[field_name] / field_comparisons[field_name]
                if field_comparisons[field_name]
                else 0.0
            )
            for field_name in CORE_FIELDS
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "benchmark_summary.json", summary)
    write_jsonl(output_dir / "turn_comparisons.jsonl", comparisons)
    write_json(
        output_dir / "ensemble_manifest.json",
        {
            "summary": summary,
            "source_run_dirs": [str(run_dir) for run_dir in run_dirs],
            "files": {
                "summary": str(output_dir / "benchmark_summary.json"),
                "comparisons": str(output_dir / "turn_comparisons.jsonl"),
            },
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ensemble benchmark report from stored runs")
    parser.add_argument("--run-dir", action="append", required=True, dest="run_dirs")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    summary = compare_runs(
        run_dirs=[Path(value) for value in args.run_dirs],
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
