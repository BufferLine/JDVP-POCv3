#!/usr/bin/env python3
"""List dataset generation runs from the SQLite catalog."""

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
    parser = argparse.ArgumentParser(description="List cataloged dataset generation runs")
    parser.add_argument("--status", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--generation-mode", default=None)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    rows = CatalogStore().list_dataset_generation_runs(
        status=args.status,
        dataset_id=args.dataset_id,
        generation_mode=args.generation_mode,
        limit=args.limit,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
