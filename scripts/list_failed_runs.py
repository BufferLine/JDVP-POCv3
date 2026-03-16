#!/usr/bin/env python3
"""List failed or filtered JDVP runs from the SQLite catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catalog.sqlite_store import CatalogStore


def main() -> None:
    parser = argparse.ArgumentParser(description="List cataloged JDVP runs")
    parser.add_argument("--status", default="failed")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--dataset-run-id", default=None)
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--track-name", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--summary-by-scenario", action="store_true")
    parser.add_argument("--dataset-runs", action="store_true")
    args = parser.parse_args()

    store = CatalogStore()
    if args.dataset_runs:
        rows = store.list_dataset_runs(
            status=args.status,
            dataset_id=args.dataset_id,
            scenario_id=args.scenario_id,
            track_name=args.track_name,
            limit=args.limit,
        )
    elif args.summary_by_scenario:
        rows = store.summarize_runs_by_scenario(status=args.status)
    else:
        rows = store.list_runs(
            status=args.status,
            dataset_id=args.dataset_id,
            dataset_run_id=args.dataset_run_id,
            scenario_id=args.scenario_id,
            limit=args.limit,
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
