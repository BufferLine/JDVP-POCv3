#!/usr/bin/env python3
"""Rerun failed JDVP runs from the SQLite catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catalog.sqlite_store import CatalogStore
from src.service.errors import ServiceError
from src.service.poc_service import RunRequest, run_interaction


def _retry_run_id(original_run_id: str, attempt_index: int) -> str:
    return f"{original_run_id}-retry-{attempt_index:02d}"


def rerun_failed_runs(
    *,
    catalog: CatalogStore,
    status: str = "failed",
    dataset_id: str | None = None,
    scenario_id: str | None = None,
    limit: int | None = None,
    reuse_run_id: bool = False,
    resume: bool = False,
) -> list[dict[str, Any]]:
    selected_runs = catalog.list_runs(
        status=status,
        dataset_id=dataset_id,
        scenario_id=scenario_id,
        limit=limit,
    )
    results: list[dict[str, Any]] = []
    for index, row in enumerate(selected_runs, start=1):
        original_run_id = str(row["run_id"])
        rerun_id = original_run_id if reuse_run_id else _retry_run_id(original_run_id, index)
        try:
            result = run_interaction(
                RunRequest(
                    input_path=Path(str(row["input_path"])),
                    run_id=rerun_id,
                    output_root=Path(str(Path(str(row["run_dir"])).parent)),
                    track_name=str(row["track_name"]),
                    resume=resume,
                )
            )
            results.append(
                {
                    "original_run_id": original_run_id,
                    "rerun_run_id": rerun_id,
                    "status": "completed",
                    "interaction_id": result.interaction_id,
                    "run_dir": str(result.run_dir),
                }
            )
        except ServiceError as exc:
            results.append(
                {
                    "original_run_id": original_run_id,
                    "rerun_run_id": rerun_id,
                    "status": "failed",
                    "error_code": exc.code,
                    "error_message": exc.message,
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun failed JDVP runs from the catalog")
    parser.add_argument("--status", default="failed")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reuse-run-id", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    results = rerun_failed_runs(
        catalog=CatalogStore(),
        status=args.status,
        dataset_id=args.dataset_id,
        scenario_id=args.scenario_id,
        limit=args.limit,
        reuse_run_id=args.reuse_run_id,
        resume=args.resume,
    )
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
