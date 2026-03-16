#!/usr/bin/env python3
"""List failed dataset-generation items from the SQLite catalog."""

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
    parser = argparse.ArgumentParser(description="List failed cataloged dataset-generation items")
    parser.add_argument("--generation-run-id", required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    rows = CatalogStore().list_failed_dataset_generation_items(
        generation_run_id=args.generation_run_id,
        limit=args.limit,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
