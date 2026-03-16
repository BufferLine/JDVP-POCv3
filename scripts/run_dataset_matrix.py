#!/usr/bin/env python3
"""Run multiple track/model configurations over the same dataset slice."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.run_storage import read_json, write_json
from src.service.dataset_run_service import DatasetRunRequest, run_dataset


@contextmanager
def temporary_env(updates: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_matrix(*, config_path: Path) -> Path:
    config = load_json(config_path)
    dataset_root = Path(str(config["dataset_root"]))
    output_root = Path(str(config["output_root"]))
    split = str(config["split"]) if config.get("split") is not None else None
    scenario_id = str(config["scenario_id"]) if config.get("scenario_id") is not None else None
    max_items = int(config["max_items"]) if config.get("max_items") is not None else None
    resume = bool(config.get("resume", False))

    rows: list[dict[str, Any]] = []
    for run_cfg in config["runs"]:
        label = str(run_cfg["label"])
        track_name = str(run_cfg["track_name"])
        env_updates = {str(key): str(value) for key, value in run_cfg.get("env", {}).items()}
        run_output_root = output_root / label
        with temporary_env(env_updates):
            result = run_dataset(
                DatasetRunRequest(
                    dataset_root=dataset_root,
                    output_root=run_output_root,
                    track_name=track_name,
                    split=split,
                    scenario_id=scenario_id,
                    max_items=max_items,
                    resume=resume,
                )
            )
        summary = read_json(result.summary_path)
        rows.append(
            {
                "label": label,
                "track_name": track_name,
                "env": env_updates,
                "dataset_run_result": result.to_dict(),
                "completed_count": summary["completed_count"],
                "failed_count": summary["failed_count"],
            }
        )

    summary_path = output_root / "dataset_matrix_summary.json"
    write_json(
        summary_path,
        {
            "schema_version": "pocv3-dataset-matrix-summary-v1",
            "config_path": str(config_path),
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "split": split,
            "scenario_id": scenario_id,
            "max_items": max_items,
            "runs": rows,
        },
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple track/model configs over one dataset slice")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    summary_path = run_matrix(config_path=args.config)
    print(f"Dataset matrix summary written: {summary_path}")


if __name__ == "__main__":
    main()
