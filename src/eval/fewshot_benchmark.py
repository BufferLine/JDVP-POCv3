"""Build a split-aware benchmark plan for few-shot observer runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.dataset.manifest_loader import load_dataset_manifest
from src.method.fewshot.selector import load_fewshot_pack, select_examples
from src.pipeline.run_storage import write_json


def build_fewshot_benchmark_plan(
    *,
    dataset_root: Path,
    fewshot_pack_path: Path,
    split: str,
    output_path: Path,
    max_examples: int = 3,
) -> Path:
    manifest = load_dataset_manifest(dataset_root)
    pack = load_fewshot_pack(fewshot_pack_path)
    items = manifest.items_for_split(split)
    rows: list[dict[str, Any]] = []
    for item in items:
        selected = select_examples(
            pack=pack,
            target_interaction_id=item.interaction_id,
            context_module=item.context_module,
            max_examples=max_examples,
        )
        row_entry: dict[str, Any] = {
            "interaction_id": item.interaction_id,
            "split": item.split,
            "scenario_id": item.scenario_id,
            "context_module": item.context_module,
            "input_path": str(item.interaction_path(dataset_root)),
            "fewshot_example_count": len(selected),
            "fewshot_examples": selected,
        }
        if len(selected) == 0:
            row_entry["zero_shot_fallback"] = True
        rows.append(row_entry)
    payload = {
        "schema_version": "pocv3-fewshot-benchmark-plan-v1",
        "dataset_id": manifest.dataset_id,
        "split": split,
        "fewshot_pack_path": str(fewshot_pack_path),
        "item_count": len(rows),
        "max_examples": max_examples,
        "items": rows,
    }
    write_json(output_path, payload)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a few-shot benchmark plan from one dataset split")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--fewshot-pack", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=3)
    args = parser.parse_args()

    output_path = build_fewshot_benchmark_plan(
        dataset_root=args.dataset_root,
        fewshot_pack_path=args.fewshot_pack,
        split=args.split,
        output_path=args.output,
        max_examples=args.max_examples,
    )
    print(f"Few-shot benchmark plan written: {output_path}")


if __name__ == "__main__":
    main()
