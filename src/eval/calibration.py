"""Human-annotation calibration contracts and low-cost observer reporting."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from src.protocol_core.enums import CORE_FIELD_NAMES, normalize_core_level


CALIBRATION_MANIFEST_FILENAME = "manifest.json"
CALIBRATION_ANNOTATIONS_FILENAME = "annotations.jsonl"


def load_calibration_dataset(dataset_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a consented annotation dataset without treating it as a gold set."""
    manifest_path = dataset_root / CALIBRATION_MANIFEST_FILENAME
    annotations_path = dataset_root / CALIBRATION_ANNOTATIONS_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "jdvp-human-calibration-v1":
        raise ValueError("unexpected calibration manifest schema_version")
    if not manifest.get("consent_basis"):
        raise ValueError("calibration manifest requires consent_basis")
    if manifest.get("measurement_profile") != "human_ai":
        raise ValueError("calibration manifest must declare measurement_profile=human_ai")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(annotations_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("interaction_id"), str) or not isinstance(row.get("turn_number"), int):
            raise ValueError(f"annotation row {line_number} requires interaction_id and turn_number")
        annotations = row.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            raise ValueError(f"annotation row {line_number} requires annotations")
        for annotation in annotations:
            if not isinstance(annotation.get("annotator_id"), str):
                raise ValueError(f"annotation row {line_number} has an annotation without annotator_id")
            _validate_labels(annotation.get("labels"), line_number)
        if row.get("adjudicated_labels") is not None:
            _validate_labels(row["adjudicated_labels"], line_number)
        rows.append(row)
    if not rows:
        raise ValueError("calibration dataset contains no annotation rows")
    return manifest, rows


def _validate_labels(labels: object, line_number: int) -> None:
    if not isinstance(labels, dict):
        raise ValueError(f"annotation row {line_number} labels must be an object")
    for field_name in CORE_FIELD_NAMES:
        if field_name not in labels:
            raise ValueError(f"annotation row {line_number} labels missing {field_name}")
        normalize_core_level(field_name, labels[field_name])


def build_calibration_report(
    *,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    predictions: dict[tuple[str, int], dict[str, Any]],
    run_metadata: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Report human agreement and observer performance on adjudicated rows.

    Unresolved annotations contribute to inter-annotator agreement only. They
    never become an implicit gold label through majority voting.
    """
    agreement = _annotator_agreement(rows)
    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    absolute_error: dict[str, int] = defaultdict(int)
    evaluated_rows = 0

    for row in rows:
        target = row.get("adjudicated_labels")
        prediction = predictions.get((row["interaction_id"], row["turn_number"]))
        if target is None or prediction is None:
            continue
        evaluated_rows += 1
        for field_name in CORE_FIELD_NAMES:
            expected = normalize_core_level(field_name, target[field_name])
            observed = normalize_core_level(field_name, prediction.get(field_name))
            if expected is None or observed is None:
                continue
            total[field_name] += 1
            absolute_error[field_name] += abs(expected - observed)
            if expected == observed:
                correct[field_name] += 1

    field_performance = {
        field: {
            "exact_agreement": round(correct[field] / total[field], 4) if total[field] else None,
            "mean_absolute_error": round(absolute_error[field] / total[field], 4) if total[field] else None,
            "correct": correct[field],
            "total": total[field],
        }
        for field in CORE_FIELD_NAMES
    }
    return {
        "schema_version": "jdvp-calibration-report-v1",
        "dataset_id": manifest.get("dataset_id"),
        "measurement_profile": manifest["measurement_profile"],
        "annotation_rows": len(rows),
        "adjudicated_rows_evaluated": evaluated_rows,
        "annotator_agreement": agreement,
        "observer_performance": field_performance,
        "run_cost_and_latency": _summarize_run_metadata(run_metadata or []),
        "interpretation_boundary": "Descriptive calibration evidence only; not a validated critical-thinking score.",
    }


def _annotator_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matches: dict[str, int] = defaultdict(int)
    comparisons: dict[str, int] = defaultdict(int)
    absolute_error: dict[str, int] = defaultdict(int)
    for row in rows:
        annotations = row["annotations"]
        for left, right in combinations(annotations, 2):
            for field_name in CORE_FIELD_NAMES:
                left_value = normalize_core_level(field_name, left["labels"][field_name])
                right_value = normalize_core_level(field_name, right["labels"][field_name])
                if left_value is None or right_value is None:
                    continue
                comparisons[field_name] += 1
                absolute_error[field_name] += abs(left_value - right_value)
                if left_value == right_value:
                    matches[field_name] += 1
    return {
        field: {
            "pairwise_exact_agreement": round(matches[field] / comparisons[field], 4) if comparisons[field] else None,
            "pairwise_mean_absolute_difference": round(absolute_error[field] / comparisons[field], 4) if comparisons[field] else None,
            "pairwise_comparisons": comparisons[field],
        }
        for field in CORE_FIELD_NAMES
    }


def _summarize_run_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    costs = [float(row["estimated_cost_usd"]) for row in rows if row.get("estimated_cost_usd") is not None]
    return {
        "turns_with_latency": len(latencies),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p95_latency_ms": round(sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)], 2) if latencies else None,
        "turns_with_cost_estimate": len(costs),
        "total_estimated_cost_usd": round(sum(costs), 6) if costs else None,
        "mean_estimated_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
    }
