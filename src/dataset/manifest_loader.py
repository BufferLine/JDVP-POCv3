"""Split-aware dataset manifest loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.pipeline.run_storage import read_json


@dataclass(frozen=True)
class DatasetItem:
    interaction_id: str
    scenario_id: str
    scenario_title: str
    context_module: str
    relative_path: str
    split: str
    turn_count: int
    slot_values: dict[str, str]

    def interaction_path(self, dataset_root: Path) -> Path:
        return dataset_root / self.relative_path


@dataclass(frozen=True)
class DatasetManifest:
    dataset_root: Path
    dataset_id: str
    dataset_name: str
    dataset_version: str
    dataset_kind: str
    context_module: str
    items: list[DatasetItem]

    def items_for_split(self, split: str) -> list[DatasetItem]:
        return [item for item in self.items if item.split == split]


def load_dataset_manifest(dataset_root: Path) -> DatasetManifest:
    payload = read_json(dataset_root / "manifest.json")
    items = [
        DatasetItem(
            interaction_id=str(item["interaction_id"]),
            scenario_id=str(item["scenario_id"]),
            scenario_title=str(item["scenario_title"]),
            context_module=str(item["context_module"]),
            relative_path=str(item["relative_path"]),
            split=str(item["split"]),
            turn_count=int(item["turn_count"]),
            slot_values={str(key): str(value) for key, value in item.get("slot_values", {}).items()},
        )
        for item in payload["items"]
    ]
    return DatasetManifest(
        dataset_root=dataset_root,
        dataset_id=str(payload["dataset_id"]),
        dataset_name=str(payload["dataset_name"]),
        dataset_version=str(payload["dataset_version"]),
        dataset_kind=str(payload["dataset_kind"]),
        context_module=str(payload["context_module"]),
        items=items,
    )
