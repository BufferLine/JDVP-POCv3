"""Synthetic dataset generation and manifest materialization for M6."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.catalog.sqlite_store import CatalogDatasetRecord, CatalogStore
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


def _select_variant(
    template: dict[str, Any],
    *,
    field_name: str,
    rng: random.Random,
) -> tuple[str | None, int | None]:
    options_key = f"{field_name}_options"
    options = template.get(options_key)
    if not isinstance(options, list) or not options:
        value = template.get(field_name)
        return (str(value) if isinstance(value, str) else None), None

    selected_index = rng.randrange(len(options))
    selected_value = options[selected_index]
    if not isinstance(selected_value, str):
        raise ValueError(f"{options_key} entries must be strings")
    return selected_value, selected_index


def _materialize_turn_template(
    turn_template: Mapping[str, Any],
    *,
    slots: Mapping[str, str],
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    template = dict(turn_template)
    human_input, human_variant_index = _select_variant(template, field_name="human_input", rng=rng)
    ai_response, ai_variant_index = _select_variant(template, field_name="ai_response", rng=rng)
    if human_input is not None:
        template["human_input"] = human_input
    if ai_response is not None:
        template["ai_response"] = ai_response
    template.pop("human_input_options", None)
    template.pop("ai_response_options", None)

    rendered = _render_template(template, slots)
    variant_metadata = {
        "turn_number": int(rendered["turn_number"]),
        "human_input_variant_index": human_variant_index,
        "ai_response_variant_index": ai_variant_index,
    }
    return rendered, variant_metadata


def _select_turn_templates(
    scenario: Mapping[str, Any],
    *,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], str | None]:
    blueprints = scenario.get("blueprints")
    if isinstance(blueprints, list) and blueprints:
        selected_blueprint = blueprints[rng.randrange(len(blueprints))]
        if not isinstance(selected_blueprint, dict):
            raise ValueError("scenario blueprints must be objects")
        turn_templates = selected_blueprint.get("turn_templates")
        if not isinstance(turn_templates, list) or not turn_templates:
            raise ValueError("scenario blueprint turn_templates must be a non-empty list")
        blueprint_id = selected_blueprint.get("blueprint_id")
        return list(turn_templates), (str(blueprint_id) if blueprint_id is not None else None)

    turn_templates = scenario.get("turn_templates")
    if not isinstance(turn_templates, list) or not turn_templates:
        raise ValueError("scenario must define turn_templates or blueprints")
    return list(turn_templates), None


def _build_interaction(
    *,
    dataset_name: str,
    scenario: dict[str, Any],
    sample_index: int,
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    slots = _choose_slots(scenario.get("slot_options", {}), rng)
    turn_templates, blueprint_id = _select_turn_templates(scenario, rng=rng)
    interaction_id = f"{dataset_name}-{scenario['scenario_id']}-{sample_index:03d}"
    turns = []
    turn_variant_choices = []
    for turn_template in turn_templates:
        turn_payload, variant_metadata = _materialize_turn_template(
            turn_template,
            slots=slots,
            rng=rng,
        )
        turns.append(turn_payload)
        turn_variant_choices.append(variant_metadata)
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
        "blueprint_id": blueprint_id,
        "turn_variant_choices": turn_variant_choices,
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
    CatalogStore().upsert_dataset(
        CatalogDatasetRecord(
            dataset_id=str(manifest["dataset_id"]),
            dataset_root=str(dataset_root),
            dataset_kind=str(manifest["dataset_kind"]),
            scenario_pack_id=str(manifest["scenario_pack_id"]),
            generation_seed=int(seed),
            count_per_scenario=int(count_per_scenario),
        ),
        items=manifest["items"],
    )
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
