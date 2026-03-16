"""Synthetic dataset generation and manifest materialization for M6."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.contracts.raw_interaction_validate import RawInteractionValidator
from src.pipeline.run_storage import write_json


DEFAULT_SCENARIO_PACK = Path("config/datasets/general_scenarios_v1.json")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _render_template(template: Any, slots: Mapping[str, str]) -> Any:
    if isinstance(template, str):
        return template.format(**slots)
    if isinstance(template, list):
        return [_render_template(value, slots) for value in template]
    if isinstance(template, dict):
        return {key: _render_template(value, slots) for key, value in template.items()}
    return template


def _choose_slots(slot_options: Mapping[str, Sequence[str]], rng: random.Random) -> dict[str, str]:
    return {name: str(rng.choice(list(values))) for name, values in slot_options.items()}


def _build_interaction(
    *,
    dataset_name: str,
    scenario: dict[str, Any],
    sample_index: int,
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    slots = _choose_slots(scenario.get("slot_options", {}), rng)
    interaction_id = f"{dataset_name}-{scenario['scenario_id']}-{sample_index:03d}"
    turns = []
    for turn_template in scenario["turn_templates"]:
        turn_payload = _render_template(turn_template, slots)
        turns.append(turn_payload)
    interaction = {
        "interaction_id": interaction_id,
        "context_module": scenario["context_module"],
        "participants": {
            "human_id": str(scenario.get("participants", {}).get("human_id", "human-1")),
            "ai_id": str(scenario.get("participants", {}).get("ai_id", "ai-1")),
        },
        "turns": turns,
    }
    item_metadata = {
        "interaction_id": interaction_id,
        "scenario_id": scenario["scenario_id"],
        "scenario_title": scenario.get("title", scenario["scenario_id"]),
        "slot_values": slots,
    }
    return interaction, item_metadata


def _assign_splits(
    interaction_ids: list[str],
    *,
    rng: random.Random,
    train_ratio: float,
    validation_ratio: float,
) -> dict[str, list[str]]:
    shuffled = list(interaction_ids)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_cutoff = int(total * train_ratio)
    validation_cutoff = train_cutoff + int(total * validation_ratio)
    return {
        "train": shuffled[:train_cutoff],
        "validation": shuffled[train_cutoff:validation_cutoff],
        "test": shuffled[validation_cutoff:],
    }


def _manifest_items(
    *,
    items: list[dict[str, Any]],
    split_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    split_by_id = {
        interaction_id: split_name
        for split_name, interaction_ids in split_map.items()
        for interaction_id in interaction_ids
    }
    manifest_items = []
    for item in items:
        manifest_items.append(
            {
                **item,
                "split": split_by_id[item["interaction_id"]],
            }
        )
    return manifest_items


def generate_dataset(
    *,
    dataset_name: str,
    dataset_version: str,
    output_root: Path,
    scenario_pack_path: Path = DEFAULT_SCENARIO_PACK,
    dataset_kind: str = "generated",
    count_per_scenario: int = 1,
    seed: int = 7,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
) -> Path:
    if count_per_scenario < 1:
        raise ValueError("count_per_scenario must be at least 1")
    if train_ratio < 0 or validation_ratio < 0 or train_ratio + validation_ratio > 1:
        raise ValueError("split ratios must be non-negative and sum to at most 1")

    scenario_pack = _load_json(scenario_pack_path)
    rng = random.Random(seed)
    validator = RawInteractionValidator()

    dataset_root = output_root / dataset_name / dataset_version
    interactions_root = dataset_root / "interactions"
    manifest_items_seed: list[dict[str, Any]] = []
    interaction_ids: list[str] = []

    for scenario in scenario_pack["scenarios"]:
        for sample_index in range(count_per_scenario):
            interaction, item_metadata = _build_interaction(
                dataset_name=dataset_name,
                scenario=scenario,
                sample_index=sample_index,
                rng=rng,
            )
            validator.validate(interaction)
            relative_path = Path("interactions") / f"{interaction['interaction_id']}.json"
            write_json(dataset_root / relative_path, interaction)
            manifest_items_seed.append(
                {
                    **item_metadata,
                    "relative_path": str(relative_path),
                    "context_module": interaction["context_module"],
                    "turn_count": len(interaction["turns"]),
                }
            )
            interaction_ids.append(str(interaction["interaction_id"]))

    split_rng = random.Random(seed + 1)
    split_map = _assign_splits(
        interaction_ids,
        rng=split_rng,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    manifest = {
        "schema_version": "pocv3-dataset-manifest-v1",
        "dataset_id": f"{dataset_kind}/{dataset_name}/{dataset_version}",
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_kind": dataset_kind,
        "context_module": scenario_pack["context_module"],
        "scenario_pack_id": scenario_pack["scenario_pack_id"],
        "item_count": len(manifest_items_seed),
        "split_counts": {name: len(values) for name, values in split_map.items()},
        "generation": {
            "seed": seed,
            "count_per_scenario": count_per_scenario,
            "scenario_pack_path": str(scenario_pack_path),
            "train_ratio": train_ratio,
            "validation_ratio": validation_ratio,
            "test_ratio": 1 - train_ratio - validation_ratio,
        },
        "items": _manifest_items(items=manifest_items_seed, split_map=split_map),
    }
    write_json(dataset_root / "manifest.json", manifest)
    write_json(dataset_root / "splits.json", split_map)
    return dataset_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic JDVP research dataset")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--scenario-pack", type=Path, default=DEFAULT_SCENARIO_PACK)
    parser.add_argument("--dataset-kind", choices=["generated", "fixtures", "raw"], default="generated")
    parser.add_argument("--count-per-scenario", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    args = parser.parse_args()

    dataset_root = generate_dataset(
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        output_root=args.output_root,
        scenario_pack_path=args.scenario_pack,
        dataset_kind=args.dataset_kind,
        count_per_scenario=args.count_per_scenario,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    print(f"Dataset written: {dataset_root}")


if __name__ == "__main__":
    main()
