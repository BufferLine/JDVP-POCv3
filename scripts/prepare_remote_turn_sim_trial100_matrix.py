#!/usr/bin/env python3
"""Prepare a remote Ollama matrix config for the trial100 dataset."""

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

from src.pipeline.run_storage import write_json
from src.service.dataset_run_service import DatasetRunRequest, run_dataset


DEFAULT_DATASET_ROOT = Path("data/generated-local-trials/local-turn-sim-trial100/v2")
DEFAULT_OUTPUT_ROOT = Path("data/runs/local-turn-sim-trial100-remote-matrix")
DEFAULT_FEWSHOT_PACK = Path("data/baselines/fewshot_regression_general_v1/fewshot-pack.json")


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


def build_config(
    *,
    base_url: str,
    api_key: str,
    dataset_root: Path,
    output_root: Path,
    fewshot_pack_path: Path,
    include_heuristic: bool,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    if include_heuristic:
        runs.append(
            {
                "label": "heuristic",
                "track_name": "heuristic_baseline",
            }
        )
    common_env = {
        "JDVP_LLM_BASE_URL": base_url,
        "JDVP_LLM_API_KEY": api_key,
        "JDVP_FEWSHOT_PACK_PATH": str(fewshot_pack_path),
    }
    runs.extend(
        [
            {
                "label": "fewshot_gemma3_12b_remote",
                "track_name": "fewshot_prompt",
                "env": {
                    **common_env,
                    "JDVP_LLM_MODEL": "gemma3:12b",
                },
            },
            {
                "label": "fewshot_gpt_oss_20b_remote",
                "track_name": "fewshot_prompt",
                "env": {
                    **common_env,
                    "JDVP_LLM_MODEL": "gpt-oss:20b",
                },
            },
        ]
    )
    return {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "resume": True,
        "runs": runs,
    }


def run_prepared_config(config: dict[str, Any]) -> Path:
    dataset_root = Path(str(config["dataset_root"]))
    output_root = Path(str(config["output_root"]))
    rows: list[dict[str, Any]] = []
    for run_cfg in config["runs"]:
        label = str(run_cfg["label"])
        track_name = str(run_cfg["track_name"])
        env_updates = {str(key): str(value) for key, value in run_cfg.get("env", {}).items()}
        with temporary_env(env_updates):
            result = run_dataset(
                DatasetRunRequest(
                    dataset_root=dataset_root,
                    output_root=output_root / label,
                    track_name=track_name,
                    resume=bool(config.get("resume", False)),
                )
            )
        rows.append(
            {
                "label": label,
                "track_name": track_name,
                "env": env_updates,
                "dataset_run_result": result.to_dict(),
                "completed_count": result.completed_count,
                "failed_count": result.failed_count,
            }
        )
    summary_path = output_root / "dataset_matrix_summary.json"
    write_json(
        summary_path,
        {
            "schema_version": "pocv3-dataset-matrix-summary-v1",
            "config_path": str(output_root / "remote_matrix_config.json"),
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "split": None,
            "scenario_id": None,
            "max_items": None,
            "runs": rows,
        },
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a remote Ollama few-shot matrix for trial100")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fewshot-pack-path", type=Path, default=DEFAULT_FEWSHOT_PACK)
    parser.add_argument("--skip-heuristic", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = build_config(
        base_url=args.base_url,
        api_key=args.api_key,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        fewshot_pack_path=args.fewshot_pack_path,
        include_heuristic=not args.skip_heuristic,
    )
    config_path = args.output_root / "remote_matrix_config.json"
    write_json(config_path, config)
    print(f"Remote matrix config written: {config_path}")
    if not args.execute:
        print(f"Prepared runs: {[run['label'] for run in config['runs']]}")
        print(f"Execute later with: python3 scripts/run_dataset_matrix.py --config {config_path}")
        return

    summary_path = run_prepared_config(config)
    print(f"Dataset matrix summary written: {summary_path}")


if __name__ == "__main__":
    main()
