"""JSON transport adapter over the service layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.service.dataset_run_service import DatasetRunRequest, run_dataset_response
from src.service.eval_service import FewshotBenchmarkRequest, run_fewshot_benchmark_response
from src.service.poc_service import RunRequest, run_interaction_response
from src.shared_utils import load_json as _load_json


def _run_interaction_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    request = RunRequest(
        input_path=Path(payload["input_path"]),
        run_id=str(payload["run_id"]),
        output_root=Path(payload["output_root"]),
        protocol_schema_root=Path(payload["protocol_schema_root"]) if payload.get("protocol_schema_root") else None,
        track_name=str(payload.get("track_name", "fixture_hint")),
        resume=bool(payload.get("resume", False)),
    )
    return run_interaction_response(request)


def _run_fewshot_benchmark_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    request = FewshotBenchmarkRequest(
        plan_path=Path(payload["plan_path"]),
        output_root=Path(payload["output_root"]),
        comparison_track=str(payload.get("comparison_track", "heuristic_baseline")),
        max_average_field_disagreement_rate=(
            float(payload["max_average_field_disagreement_rate"])
            if payload.get("max_average_field_disagreement_rate") is not None
            else None
        ),
        max_field_disagreement_rate=(
            float(payload["max_field_disagreement_rate"])
            if payload.get("max_field_disagreement_rate") is not None
            else None
        ),
    )
    return run_fewshot_benchmark_response(request)


def _run_dataset_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    request = DatasetRunRequest(
        dataset_root=Path(payload["dataset_root"]),
        output_root=Path(payload["output_root"]),
        track_name=str(payload.get("track_name", "fixture_hint")),
        split=(str(payload["split"]) if payload.get("split") is not None else None),
        scenario_id=(str(payload["scenario_id"]) if payload.get("scenario_id") is not None else None),
        max_items=(int(payload["max_items"]) if payload.get("max_items") is not None else None),
        resume=bool(payload.get("resume", False)),
    )
    return run_dataset_response(request)


def handle_json_payload(payload: dict[str, Any]) -> dict[str, Any]:
    operation = str(payload["operation"])
    if operation == "run_interaction":
        return _run_interaction_from_payload(payload)
    if operation == "run_fewshot_benchmark":
        return _run_fewshot_benchmark_from_payload(payload)
    if operation == "run_dataset":
        return _run_dataset_from_payload(payload)
    return {
        "schema_version": "pocv3-service-response-v1",
        "ok": False,
        "error": {
            "schema_version": "pocv3-service-error-v1",
            "code": "unsupported_operation",
            "message": f"unsupported operation: {operation}",
            "details": {"operation": operation},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JDVP service operations via JSON payloads")
    parser.add_argument("--input", required=True, type=Path, help="Path to a JSON request payload")
    args = parser.parse_args()

    response = handle_json_payload(_load_json(args.input))
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
