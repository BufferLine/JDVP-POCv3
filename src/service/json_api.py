"""JSON transport adapter over the service layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.service.eval_service import FewshotBenchmarkRequest, run_fewshot_benchmark_response
from src.service.poc_service import RunRequest, run_interaction_response


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    )
    return run_fewshot_benchmark_response(request)


def handle_json_payload(payload: dict[str, Any]) -> dict[str, Any]:
    operation = str(payload["operation"])
    if operation == "run_interaction":
        return _run_interaction_from_payload(payload)
    if operation == "run_fewshot_benchmark":
        return _run_fewshot_benchmark_from_payload(payload)
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
