#!/usr/bin/env python3
"""Show rerun guidance for a partial dataset-generation run."""

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
    parser = argparse.ArgumentParser(description="Show how to resume a failed dataset-generation run")
    parser.add_argument("--generation-run-id", required=True)
    args = parser.parse_args()

    store = CatalogStore()
    run = store.fetch_dataset_generation_run(args.generation_run_id)
    if run is None:
        raise SystemExit(f"generation run not found: {args.generation_run_id}")

    failed_items = store.list_failed_dataset_generation_items(
        generation_run_id=args.generation_run_id,
    )
    payload = {
        "generation_run_id": run["generation_run_id"],
        "status": run["status"],
        "accepted_count": run["accepted_count"],
        "failed_count": run["failed_count"],
        "dataset_root": run["dataset_root"],
        "rerun_note": "Re-run the same dataset generation command with the same dataset root inputs. Accepted items are reused and failed items are retried.",
        "failed_items_preview": failed_items[:10],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
