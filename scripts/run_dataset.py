#!/usr/bin/env python3
"""Run one JDVP track across a dataset slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.service.dataset_run_service import DatasetRunRequest, run_dataset_response


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one JDVP track across a dataset slice")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--track", default="fixture_hint")
    parser.add_argument("--split", default=None)
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    response = run_dataset_response(
        DatasetRunRequest(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            track_name=args.track,
            split=args.split,
            scenario_id=args.scenario_id,
            max_items=args.max_items,
            resume=args.resume,
        )
    )
    print(json.dumps(response, indent=2))
    if not response.get("ok", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
