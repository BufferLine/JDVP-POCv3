#!/usr/bin/env python3
"""Generate a small dataset preview for quick human review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.generate_dataset import generate_dataset
from src.dataset.manifest_loader import load_dataset_manifest
from src.pipeline.run_storage import read_json, write_json


DEFAULT_SCENARIO_PACK = ROOT / "config" / "datasets" / "general_scenarios_v2.json"


def build_preview(
    *,
    dataset_root: Path,
    max_items: int = 6,
) -> dict[str, Any]:
    manifest = load_dataset_manifest(dataset_root)
    selected_items = manifest.items[:max_items]
    preview_items: list[dict[str, Any]] = []
    for item in selected_items:
        interaction = read_json(item.interaction_path(dataset_root))
        preview_items.append(
            {
                "interaction_id": item.interaction_id,
                "scenario_id": item.scenario_id,
                "blueprint_id": item.blueprint_id,
                "split": item.split,
                "turn_count": item.turn_count,
                "slot_values": item.slot_values,
                "turns": [
                    {
                        "turn_number": int(turn["turn_number"]),
                        "human_input": str(turn["human_input"]),
                        "ai_response": str(turn["ai_response"]),
                        "jsv_hint": turn.get("meta", {}).get("jsv_hint", {}),
                    }
                    for turn in interaction["turns"]
                ],
            }
        )
    return {
        "schema_version": "pocv3-dataset-preview-v1",
        "dataset_root": str(dataset_root),
        "dataset_id": manifest.dataset_id,
        "item_count": len(preview_items),
        "items": preview_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small preview dataset for quick review")
    parser.add_argument("--dataset-name", default="synthetic-general-rich-preview")
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "preview")
    parser.add_argument("--scenario-pack", type=Path, default=DEFAULT_SCENARIO_PACK)
    parser.add_argument("--count-per-scenario", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-items", type=int, default=6)
    args = parser.parse_args()

    dataset_root = generate_dataset(
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        output_root=args.output_root,
        scenario_pack_path=args.scenario_pack,
        count_per_scenario=args.count_per_scenario,
        seed=args.seed,
    )
    preview = build_preview(dataset_root=dataset_root, max_items=args.max_items)
    preview_path = dataset_root / "preview.json"
    write_json(preview_path, preview)
    print(f"Dataset preview written: {preview_path}")


if __name__ == "__main__":
    main()
